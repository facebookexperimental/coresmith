# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic stage-realization lint -- an AST-level arithmetic census.

The last known CoreSmith capability gap: RTL generation collapses a spec-declared
multi-stage pipeline into a single-cycle *combinational cloud*. Four generations
of the same failure (18 nested loops in one ``always_comb``; a stage-named FSM
with decorative cycle-burn states + one monolithic task; a mock-synth ifdef
branch; and -- the negative fixture used to develop this module -- a single
``always @(posedge clk)`` whose ``S_COMPUTE`` state evaluates the *whole*
9-mode x 19-RDOQ-cut RD search per cycle by inlining tasks that call tasks that
call tasks: 171 invocations of the innermost candidate evaluator, ~10k runtime
multipliers, elaboration timeout on two yosys versions). The spec side WORKS
(``pipeline_scheduler`` / ``arith_characterize`` / ``latency_audit`` price a
per-stage op budget); prose directives ("realize each stage as a registered
boundary") are GAMEABLE. This module closes the gap with a *deterministic*
pre-yosys census that makes the amplification visible in milliseconds.

What it does (token-level, comment/string-aware; NO hard pyverilog dependency):

1. **Runtime-operand op census** per ``always`` block (and a synthetic
   ``<continuous>`` pseudo-block for module-level continuous assigns): count
   ``* / %`` (multiplier-class) and ``+ - <<  >>`` ops where at least one
   operand is NOT a compile-time constant. Pure constant folds (``4 >> 2``) and
   multiply-by-literal-power-of-two (a shift, not a multiplier) are excluded.
2. **Loop weighting**: a constant-bound ``for`` loop multiplies its body's op
   count by the trip count (yosys ``proc`` unrolls it). Nested loops multiply.
3. **FSM-aware combination (MAX-over-arms)**: within one ``always`` block the
   arms of a ``case`` and the branches of an ``if``/``else if`` chain are
   MUTUALLY EXCLUSIVE per cycle -- exactly one runs. They are combined by MAX
   (worst single path), not SUM, so a legitimate state-gated iterative
   controller (the protocol's own recommended shape: one datapath, one state per
   cycle) is not mistaken for an all-in-one-cycle combinational cloud. Ops
   OUTSIDE any case/if (and separate sibling ``if``/``case`` statements, and the
   selector/condition expressions themselves) execute every cycle and still SUM;
   a ``for`` loop inside ONE arm still executes fully in that state's cycle and
   is trip-weighted + summed. Nested case/if recurse (max within max). The metric
   is maxed per-component (multipliers, total ops, per-callee call weight), an
   upper bound on any single arm -- conservative: a real single-cycle cloud (its
   whole search in one arm) is never let through.
4. **Task/function amplification**: a ``task``/``function`` is a combinational
   cone; calling it N times (directly, or N = product of enclosing loop trips)
   instantiates its body N times. Effective ops for a block = its own
   loop-weighted, arm-maxed ops + for every callee, (call weight) x (callee
   effective ops), expanded transitively over the call graph. A task called from
   3 different arms counts once per arm, but arms are maxed -> it counts once.
   This reproduces the 171x cloud pattern while sparing the FSM controller.
5. **Verdict** against ``CORESMITH_STAGE_LINT_MUL_CAP`` (default 64 -- generous
   vs any legitimate single stage) and, when the block's declared stage map is
   available, ``CORESMITH_STAGE_LINT_FACTOR`` (default 8x) over the stage map's
   total op budget.

Policy notes (documented for the edge-case tests):
  * A ``for`` loop whose bounds are not compile-time-resolvable is weighted x1
    (it cannot be statically unrolled; synthesizable loops need constant bounds
    anyway). This errs lenient, never inventing amplification.
  * A ``task``/``function`` with no call sites contributes nothing (dead code is
    optimized away) -- it is censused as a definition but never expanded.
  * ``genvar`` / ``generate`` unrolls are structural replication the synthesizer
    handles the same as a ``for`` unroll; a ``for`` inside a ``generate`` is
    weighted by its trip count identically. (A ``generate`` that instantiates N
    *stage submodules* is the DESIRED shape -- those live in their own module
    scope, not one always block, so they are not amplified into a single block.)

Gated by ``CORESMITH_STAGE_LINT`` (default ON; set 0/false/no/off to bypass) --
same env-gate + acceptance-path convention as the sibling ``rtl_storage_lint``.
Best-effort and fail-open: any parse failure returns a clean report so RTL
generation is never blocked by the lint crashing.
"""
from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Reuse the sibling lint's comment/string blanker (comment-aware scanning is a
# hard requirement -- `always` in a comment or `a*b` in a "..." literal must not
# trip the token census). Kept as an import so the two lints share one blanker.
from .rtl_storage_lint import _blank_comments_strings


# ---------------------------------------------------------------------------
# Env gates + tunables
# ---------------------------------------------------------------------------
def stage_lint_enabled() -> bool:
    """CORESMITH_STAGE_LINT gate (default ON). 0/false/no/off/'' bypasses."""
    return os.environ.get("CORESMITH_STAGE_LINT", "1").strip().lower() not in {
        "0", "false", "no", "off", "",
    }


def stage_modules_enabled() -> bool:
    """CORESMITH_STAGE_MODULES gate (default ON) -- the structural module-per-stage
    check (Deliverable 2). Only ever produces a finding when the module-per-stage
    protocol APPLIES (multi-stage datapath) AND an arithmetic-census violation
    already flags the block, so it never nukes a legitimately single-module
    N-registered-always pipeline on its own."""
    return os.environ.get("CORESMITH_STAGE_MODULES", "1").strip().lower() not in {
        "0", "false", "no", "off", "",
    }


def _mul_cap() -> int:
    try:
        return int(os.environ.get("CORESMITH_STAGE_LINT_MUL_CAP", "64"))
    except ValueError:
        return 64


def _factor() -> float:
    try:
        return float(os.environ.get("CORESMITH_STAGE_LINT_FACTOR", "8"))
    except ValueError:
        return 8.0


# A `for` loop with unresolvable bounds is weighted x1 (cannot unroll).
_DEFAULT_TRIP = 1
# Recursion depth guard for the call-graph expansion (cycles are unsynthesizable
# but a malformed file must never hang the lint).
_MAX_EXPAND_DEPTH = 64
# Absolute floor on the total-ops FACTOR check: a block is only a *cloud* if it
# instantiates a substantial spatial arithmetic count. Without this floor an
# under-declared (tiny) stage map would false-positive a small legit block whose
# handful of ops slightly exceeds `total_op_slots x FACTOR`. A real cloud is
# hundreds-to-thousands of ops, so this never masks the failure it targets.
_FACTOR_MIN_OPS = 256


# ---------------------------------------------------------------------------
# Low-level scanning primitives
# ---------------------------------------------------------------------------
_WORD_RE = re.compile(r"[A-Za-z_]\w*")
_NUM_RE = re.compile(r"\d[\w']*")
_PARAM_DECL_RE = re.compile(
    r"\b(?:localparam|parameter)\b(?:\s+(?:signed|integer|real|\[[^\]]*\]))*\s*"
    r"([A-Za-z_]\w*)\s*=\s*([^,;]+)",
)


def _find_matching(text: str, start: int, open_ch: str, close_ch: str) -> int:
    """``start`` indexes ``open_ch``; return the index just past its match."""
    depth = 0
    n = len(text)
    i = start
    while i < n:
        c = text[i]
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return n


def _match_begin_end(text: str, begin_start: int) -> tuple[int, int, int]:
    """``begin_start`` indexes a ``begin`` keyword. Return (inner_start,
    inner_end, after_end) for the balanced ``begin..end`` (labels tolerated)."""
    depth = 0
    inner_start = begin_start + 5
    for m in re.finditer(r"\bbegin\b|\bend\b", text[begin_start:]):
        if m.group(0) == "begin":
            if depth == 0:
                inner_start = begin_start + m.end()
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                return inner_start, begin_start + m.start(), begin_start + m.end()
    return inner_start, len(text), len(text)


def _stmt_end(text: str, k: int) -> int:
    """Index of the ``;`` that ends the single statement starting at ``k``
    (respecting (), [] and {} nesting)."""
    depth = 0
    n = len(text)
    i = k
    while i < n:
        c = text[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == ";" and depth == 0:
            return i
        i += 1
    return n


def _const_int(expr: str, params: dict[str, int]) -> Optional[int]:
    """Resolve ``expr`` to an int if it is a decimal literal, a sized-decimal
    literal (``8'd12``), or a known parameter. Else None."""
    e = expr.strip()
    m = re.fullmatch(r"-?\d+", e)
    if m:
        return int(e)
    m = re.fullmatch(r"\d+'\s*[sS]?[dD]\s*(\d+)", e)
    if m:
        return int(m.group(1))
    if e in params:
        return params[e]
    return None


def _parse_params(text: str) -> dict[str, int]:
    """Best-effort integer values for localparam/parameter (for loop bounds and
    constant-operand classification). Only resolves int-literal RHS."""
    params: dict[str, int] = {}
    for m in _PARAM_DECL_RE.finditer(text):
        name, rhs = m.group(1), m.group(2)
        v = _const_int(rhs, params)
        if v is not None:
            params[name] = v
    return params


def _pow2(v: int) -> bool:
    return v > 0 and (v & (v - 1)) == 0


# ---------------------------------------------------------------------------
# Control-structure boundaries (for / case / if-chain) -- used by the
# FSM-aware census to give case arms / else-if branches MAX-over-arms (worst
# single path per cycle) semantics rather than SUM.
# ---------------------------------------------------------------------------
def _parse_trip(header: str, params: dict[str, int]) -> int:
    """Trip count of a ``for (init; cond; incr)`` header. x1 if unresolvable."""
    inner = header.strip()
    if inner.startswith("("):
        inner = inner[1:]
    if inner.endswith(")"):
        inner = inner[:-1]
    parts = inner.split(";")
    if len(parts) < 2:
        return _DEFAULT_TRIP
    init = parts[0]
    start = _const_int(init.split("=")[-1], params) if "=" in init else 0
    cm = re.search(r"(<=|>=|<|>)\s*(.+)$", parts[1].strip())
    if not cm or start is None:
        return _DEFAULT_TRIP
    op, bound = cm.group(1), _const_int(cm.group(2), params)
    if bound is None:
        return _DEFAULT_TRIP
    if op == "<":
        t = bound - start
    elif op == "<=":
        t = bound - start + 1
    elif op == ">":
        t = start - bound
    else:  # ">="
        t = start - bound + 1
    return t if t > 0 else 0


def _kw_at(text: str, k: int, kw: str) -> bool:
    """True if keyword ``kw`` starts at index ``k`` on a word boundary."""
    n = len(text)
    if text[k:k + len(kw)] != kw:
        return False
    e = k + len(kw)
    return e >= n or not (text[e].isalnum() or text[e] == "_")


def _end_of_case(text: str, pos: int) -> int:
    """``pos`` is just past a ``case``/``casez``/``casex`` keyword. Return the
    index just past the matching ``endcase`` (nested cases balanced)."""
    n = len(text)
    j = pos
    while j < n and text[j].isspace():
        j += 1
    if j < n and text[j] == "(":
        j = _find_matching(text, j, "(", ")")
    depth = 1
    for m in re.finditer(r"\b(case|casez|casex|endcase)\b", text[j:]):
        if m.group(1) == "endcase":
            depth -= 1
            if depth == 0:
                return j + m.end()
        else:
            depth += 1
    return n


def _stmt_extent(text: str, k: int) -> int:
    """Return the index just past ONE complete statement starting at/after ``k``.

    Understands the compound statements that carry no terminating ``;`` --
    ``begin..end``, ``case..endcase``, and control statements whose body is
    itself a sub-statement (``if``/``for``/``while``/``repeat``/``forever``,
    including ``else``/``else if`` tails). Falls back to the next top-level
    ``;`` for a plain statement. This lets an FSM arm/branch body be extracted
    exactly whether it is a ``begin`` block or a bare nested construct."""
    n = len(text)
    while k < n and text[k].isspace():
        k += 1
    if k >= n:
        return n
    wm = _WORD_RE.match(text, k)
    word = wm.group(0) if wm else ""
    if word == "begin":
        _s, _e, after = _match_begin_end(text, k)
        return after
    if word in ("case", "casez", "casex"):
        return _end_of_case(text, wm.end())
    if word == "if":
        j = wm.end()
        while j < n and text[j].isspace():
            j += 1
        if j < n and text[j] == "(":
            j = _find_matching(text, j, "(", ")")
        j = _stmt_extent(text, j)  # then-branch
        k2 = j
        while k2 < n and text[k2].isspace():
            k2 += 1
        if _kw_at(text, k2, "else"):
            return _stmt_extent(text, k2 + 4)  # else / else-if tail
        return j
    if word in ("for", "while", "repeat"):
        j = wm.end()
        while j < n and text[j].isspace():
            j += 1
        if j < n and text[j] == "(":
            j = _find_matching(text, j, "(", ")")
        return _stmt_extent(text, j)  # loop body
    if word == "forever":
        return _stmt_extent(text, wm.end())
    return min(_stmt_end(text, k) + 1, n)


def _stmt_body(text: str, k: int) -> tuple[str, int]:
    """Extract ONE statement body starting at/after ``k`` for recursion into the
    census, plus the index just past it. The returned text is fed straight back
    to ``_census_region`` (which re-parses any leading ``begin``/nested
    construct), so ``begin``/``end`` wrappers are harmless -- they carry no ops."""
    n = len(text)
    ks = k
    while ks < n and text[ks].isspace():
        ks += 1
    after = _stmt_extent(text, ks)
    return text[ks:after], after


def _split_case_arms(arm_region: str) -> list[str]:
    """Split a ``case`` body (between selector and ``endcase``) into per-arm
    statement bodies. Each arm is ``<label(s)> : <statement>``; the label part is
    scanned to its separating top-level ``:`` (paren/bracket/brace-depth 0,
    skipping ``::`` and ternary ``?:``), then the statement is extracted whole."""
    arms: list[str] = []
    n = len(arm_region)
    i = 0
    while i < n:
        while i < n and (arm_region[i].isspace() or arm_region[i] == ";"):
            i += 1
        if i >= n:
            break
        depth = 0
        tern = 0
        j = i
        colon = -1
        while j < n:
            c = arm_region[j]
            if c in "([{":
                depth += 1
            elif c in ")]}":
                depth -= 1
            elif depth == 0:
                if c == "?":
                    tern += 1
                elif c == ":":
                    if arm_region[j:j + 2] == "::":
                        j += 2
                        continue
                    if tern > 0:
                        tern -= 1
                    else:
                        colon = j
                        break
            j += 1
        if colon < 0:
            break  # malformed / no more arms
        body, after = _stmt_body(arm_region, colon + 1)
        arms.append(body)
        i = after
    return arms


# ---------------------------------------------------------------------------
# Operand classification + op counting on a FLAT (loop-free) region
# ---------------------------------------------------------------------------
def _prev_ends_operand(text: str, i: int) -> bool:
    """True if a value/operand ends just left of index ``i`` (so a ``+``/``-``/
    ``*`` there is binary, not unary)."""
    j = i - 1
    while j >= 0 and text[j].isspace():
        j -= 1
    if j < 0:
        return False
    return text[j] in ")]}" or text[j].isalnum() or text[j] == "_"


def _classify_operand(text: str, op_pos: int, direction: int,
                      params: dict[str, int]) -> Optional[str]:
    """Classify the operand on one side of a binary operator at ``op_pos``.

    Returns ``"nonconst"`` (a variable / indexed reg / function call / paren or
    concat expression), ``"pow2"`` (a literal power-of-two -- a ``*`` by it is a
    shift), ``"const"`` (any other compile-time constant), or None."""
    n = len(text)
    if direction < 0:  # left operand
        j = op_pos - 1
        while j >= 0 and text[j].isspace():
            j -= 1
        if j < 0:
            return None
        if text[j] in ")]":  # indexed reg / func call / paren expr -> runtime
            return "nonconst"
        # walk back over an identifier/number token
        e = j + 1
        while j >= 0 and (text[j].isalnum() or text[j] in "_'."):
            j -= 1
        tok = text[j + 1:e]
        return _classify_token(tok, params)
    # right operand
    j = op_pos + 1
    while j < n and text[j].isspace():
        j += 1
    if j >= n:
        return None
    if text[j] in "({":  # paren expr / concat -> runtime
        return "nonconst"
    m = _WORD_RE.match(text, j) or _NUM_RE.match(text, j)
    if not m:
        return None
    tok = text[j:m.end()]
    k = m.end()
    while k < n and text[k].isspace():
        k += 1
    if k < n and text[k] in "([":  # func call or indexed access -> runtime
        return "nonconst"
    return _classify_token(tok, params)


def _classify_token(tok: str, params: dict[str, int]) -> str:
    """A bare identifier/number token -> const/pow2/nonconst."""
    v = _const_int(tok, params)
    if v is not None:
        return "pow2" if _pow2(v) else "const"
    # a bare identifier that is a known param resolves via _const_int above;
    # anything else is a runtime signal/variable.
    return "nonconst"


def _count_ops(text: str, params: dict[str, int]) -> tuple[int, int]:
    """(multiplier_class_ops, total_runtime_ops) in a FLAT region, unit weight.

    Multiplier-class = ``* / %`` with >=1 non-constant operand, EXCLUDING a
    multiply by a literal power of two (a shift). Total = multiplier-class +
    non-constant ``+ - << >> ** ``."""
    mul = 0
    ops = 0
    n = len(text)
    i = 0
    while i < n:
        two = text[i:i + 2]
        three = text[i:i + 3]
        if three in ("<<<", ">>>"):
            if _binop_nonconst(text, i, i + 3, params):
                ops += 1
            i += 3
            continue
        if two in ("<<", ">>"):
            if _binop_nonconst(text, i, i + 2, params):
                ops += 1
            i += 2
            continue
        if two == "**":
            if _binop_nonconst(text, i, i + 2, params):
                mul += 1
                ops += 1
            i += 2
            continue
        if two in ("<=", ">=", "==", "!=", "&&", "||", "->", "=="):
            i += 2
            continue
        c = text[i]
        if c in "*/%":
            lv = _classify_operand(text, i, -1, params)
            rv = _classify_operand(text, i, +1, params)
            if "nonconst" in (lv, rv):
                ops += 1
                is_shift = c == "*" and ("pow2" in (lv, rv))
                if not is_shift:
                    mul += 1
            i += 1
            continue
        if c in "+-" and _prev_ends_operand(text, i):
            if _binop_nonconst(text, i, i + 1, params):
                ops += 1
            i += 1
            continue
        i += 1
    return mul, ops


def _binop_nonconst(text: str, op_start: int, op_end: int,
                    params: dict[str, int]) -> bool:
    """True if the binary operator spanning [op_start, op_end) has >=1 runtime
    operand (so it does not constant-fold away)."""
    lv = _classify_operand(text, op_start, -1, params)
    rv = _classify_operand(text, op_end - 1, +1, params)
    return "nonconst" in (lv, rv)


def _count_calls(text: str, defnames: set[str]) -> Counter:
    """Count call sites ``name(`` for each name in ``defnames`` (unit weight)."""
    calls: Counter = Counter()
    for m in re.finditer(r"\b([A-Za-z_]\w*)\s*\(", text):
        name = m.group(1)
        if name in defnames:
            calls[name] += 1
    return calls


# ---------------------------------------------------------------------------
# Per-region census (loop-weighted, single invocation)
# ---------------------------------------------------------------------------
@dataclass
class _RegionCensus:
    mul: int = 0                 # loop-weighted multiplier-class ops (own body)
    ops: int = 0                 # loop-weighted total runtime ops (own body)
    calls: Counter = field(default_factory=Counter)  # {callee: loop-weighted count}


def _add_scaled(dst: _RegionCensus, src: _RegionCensus, w: int = 1) -> None:
    dst.mul += w * src.mul
    dst.ops += w * src.ops
    for name, cw in src.calls.items():
        dst.calls[name] += w * cw


def _max_region_arms(arms: list[_RegionCensus]) -> _RegionCensus:
    """Combine mutually-exclusive arms (case arms / else-if branches) by taking
    the MAX of each metric across arms -- the worst SINGLE path per cycle. Each
    metric (multiplier count, total op count, and per-callee call weight) is
    maxed independently; the result is an upper bound on any single arm, so it
    is conservative (never lets a real single-cycle cloud slip through) while
    still collapsing a state-gated FSM to its heaviest state rather than the sum
    of all states."""
    out = _RegionCensus()
    if not arms:
        return out
    out.mul = max(a.mul for a in arms)
    out.ops = max(a.ops for a in arms)
    keys: set[str] = set()
    for a in arms:
        keys.update(a.calls)
    for k in keys:
        out.calls[k] = max(a.calls.get(k, 0) for a in arms)
    return out


def _census_for(text: str, kw_start: int, wend: int, params: dict[str, int],
                defnames: set[str]) -> tuple[_RegionCensus, int]:
    """Census one ``for`` loop: its body executes fully in this cycle, so the
    body census is weighted by the (constant-bound) trip count and SUMMED. The
    header (loop-control arithmetic) is not counted."""
    n = len(text)
    j = wend
    while j < n and text[j].isspace():
        j += 1
    hdr_end = _find_matching(text, j, "(", ")")
    trip = _parse_trip(text[j:hdr_end], params)
    body, after = _stmt_body(text, hdr_end)
    contrib = _RegionCensus()
    _add_scaled(contrib, _census_region(body, params, defnames), trip)
    return contrib, after


def _census_case(text: str, kw_start: int, wend: int, params: dict[str, int],
                 defnames: set[str]) -> tuple[_RegionCensus, int]:
    """Census a ``case`` statement: the selector expression is evaluated every
    cycle (SUMMED); exactly one arm executes per cycle, so the arm censuses are
    combined by MAX-over-arms (worst single state)."""
    n = len(text)
    j = wend
    while j < n and text[j].isspace():
        j += 1
    if j < n and text[j] == "(":
        sel_end = _find_matching(text, j, "(", ")")
        sel_text = text[j:sel_end]
    else:
        sel_end = j
        sel_text = ""
    after = _end_of_case(text, wend)
    # locate the arm region: between the selector and the matching endcase
    em = None
    for mm in re.finditer(r"\bendcase\b", text[:after]):
        em = mm
    arm_region = text[sel_end:em.start()] if em else text[sel_end:after]

    contrib = _RegionCensus()
    if sel_text:
        sm, so = _count_ops(sel_text, params)
        contrib.mul += sm
        contrib.ops += so
        for name, w in _count_calls(sel_text, defnames).items():
            contrib.calls[name] += w
    arm_cs = [_census_region(a, params, defnames)
              for a in _split_case_arms(arm_region)]
    _add_scaled(contrib, _max_region_arms(arm_cs))
    return contrib, after


def _census_if(text: str, kw_start: int, wend: int, params: dict[str, int],
               defnames: set[str]) -> tuple[_RegionCensus, int]:
    """Census an ``if`` / ``else if`` chain: every condition is evaluated
    (SUMMED); exactly one branch is taken per cycle, so the branch censuses are
    combined by MAX-over-branches. A lone ``if`` (no ``else``) maxes against an
    implicit empty branch -> the taken branch, which is correct."""
    n = len(text)
    branches: list[str] = []
    cond_texts: list[str] = []
    pos = kw_start
    while True:
        wm = _WORD_RE.match(text, pos)  # 'if'
        j = wm.end()
        while j < n and text[j].isspace():
            j += 1
        if j < n and text[j] == "(":
            cond_end = _find_matching(text, j, "(", ")")
            cond_texts.append(text[j:cond_end])
        else:
            cond_end = j
        body, after = _stmt_body(text, cond_end)
        branches.append(body)
        k = after
        while k < n and text[k].isspace():
            k += 1
        if _kw_at(text, k, "else"):
            k2 = k + 4
            while k2 < n and text[k2].isspace():
                k2 += 1
            if _kw_at(text, k2, "if"):
                pos = k2  # else-if: continue the chain
                continue
            ebody, eafter = _stmt_body(text, k + 4)
            branches.append(ebody)
            end_pos = eafter
            break
        end_pos = after
        break

    contrib = _RegionCensus()
    for ct in cond_texts:
        cm, co = _count_ops(ct, params)
        contrib.mul += cm
        contrib.ops += co
        for name, w in _count_calls(ct, defnames).items():
            contrib.calls[name] += w
    branch_cs = [_census_region(b, params, defnames) for b in branches]
    _add_scaled(contrib, _max_region_arms(branch_cs))
    return contrib, end_pos


def _census_region(text: str, params: dict[str, int],
                   defnames: set[str]) -> _RegionCensus:
    """FSM-aware op + call census of ONE invocation of a region body.

    Straight-line ops (and separate sibling statements, including distinct
    ``if``/``case`` statements) SUM -- they all execute in the same cycle. But
    within a single ``case`` the arms, and within a single ``if``/``else if``
    chain the branches, are mutually exclusive per cycle and combine by
    MAX-over-arms (see ``_max_region_arms``). ``for`` loop bodies still execute
    fully in their cycle and are trip-weighted + summed. The result is the worst
    SINGLE datapath path a cycle actually realizes -- what timing/area per stage
    sees -- so a legitimate state-gated iterative controller is not mistaken for
    an all-in-one-cycle combinational cloud."""
    res = _RegionCensus()
    chars = list(text)
    n = len(text)
    i = 0
    pd = 0  # paren/bracket/brace depth (constructs only dispatch at depth 0)
    while i < n:
        c = text[i]
        if c in "([{":
            pd += 1
            i += 1
            continue
        if c in ")]}":
            pd -= 1
            i += 1
            continue
        if pd == 0 and (c.isalpha() or c == "_"):
            wm = _WORD_RE.match(text, i)
            word = wm.group(0)
            wend = wm.end()
            handler = None
            if word == "for":
                j = wend
                while j < n and text[j].isspace():
                    j += 1
                if j < n and text[j] == "(":  # a real for-loop, not `foreach`-ish
                    handler = _census_for
            elif word in ("case", "casez", "casex"):
                handler = _census_case
            elif word == "if":
                handler = _census_if
            if handler is not None:
                contrib, end = handler(text, i, wend, params, defnames)
                _add_scaled(res, contrib)
                for b in range(i, min(end, n)):  # blank the consumed construct
                    if chars[b] != "\n":
                        chars[b] = " "
                i = max(end, i + 1)
                continue
            i = wend  # skip the rest of a non-construct identifier
            continue
        i += 1

    flat = "".join(chars)
    mul, ops = _count_ops(flat, params)
    res.mul += mul
    res.ops += ops
    for name, w in _count_calls(flat, defnames).items():
        res.calls[name] += w
    return res


# ---------------------------------------------------------------------------
# Definition (task/function) + always-block extraction
# ---------------------------------------------------------------------------
_DEF_RE = re.compile(
    r"\b(function|task)\b"
    r"(?:\s+automatic\b)?"
    r"(?:\s+(?:signed|integer|real|void|\[[^\]]*\]|[A-Za-z_]\w*))*?"
    r"\s+([A-Za-z_]\w*)\s*[;(]",
    re.DOTALL,
)


def _extract_defs(src: str) -> dict[str, tuple[int, int, str]]:
    """Map ``{name: (start, end, body)}`` for each task/function (Verilog-2005
    tasks/functions do not nest). ``body`` is everything between the header
    keyword and the matching ``endtask``/``endfunction``."""
    defs: dict[str, tuple[int, int, str]] = {}
    for m in re.finditer(r"\b(function|task)\b", src):
        kw = m.group(1)
        end_kw = "endfunction" if kw == "function" else "endtask"
        em = re.search(r"\b" + end_kw + r"\b", src[m.end():])
        if not em:
            continue
        start, end = m.start(), m.end() + em.end()
        header = src[m.end():m.end() + em.start()]
        # name = last identifier before the first ';' or '(' in the header
        hh = re.split(r"[;(]", header, 1)[0]
        ids = [i for i in re.findall(r"[A-Za-z_]\w*", hh)
               if i not in ("automatic", "signed", "integer", "real", "void")]
        if not ids:
            continue
        name = ids[-1]
        defs[name] = (start, end, header)
    return defs


def _blank_span(chars: list[str], start: int, end: int) -> None:
    for i in range(start, min(end, len(chars))):
        if chars[i] != "\n":
            chars[i] = " "


def _iter_always_blocks(text: str):
    """Yield (label, body) for each ``always``/``initial`` block in ``text``
    (which has task/function defs already blanked out)."""
    for m in re.finditer(r"\b(always|initial)\b", text):
        i, n = m.end(), len(text)
        # optional @(...) / @* sensitivity
        while i < n and text[i].isspace():
            i += 1
        if i < n and text[i] == "@":
            i += 1
            while i < n and text[i].isspace():
                i += 1
            if i < n and text[i] == "(":
                i = _find_matching(text, i, "(", ")")
            elif i < n and text[i] == "*":
                i += 1
            while i < n and text[i].isspace():
                i += 1
        if text[i:i + 5] == "begin" and (
            i + 5 >= n or not (text[i + 5].isalnum() or text[i + 5] == "_")
        ):
            inner_start, inner_end, _ = _match_begin_end(text, i)
            label = ""
            lm = re.match(r"\s*:\s*([A-Za-z_]\w*)", text[i + 5:])
            if lm:
                label = lm.group(1)
            yield (label or f"always@{m.start()}"), text[inner_start:inner_end]
        else:
            semi = _stmt_end(text, i)
            yield f"always@{m.start()}", text[i:semi]


_KW_NOT_TYPE = {
    "module", "function", "task", "if", "else", "for", "while", "case",
    "casez", "casex", "begin", "end", "always", "initial", "assign",
    "generate", "endgenerate", "posedge", "negedge", "wire", "reg", "logic",
    "input", "output", "inout", "integer", "genvar", "localparam", "parameter",
    "return", "repeat", "forever",
}


def _count_module_instances(text: str, module_names: set[str]) -> int:
    """Count module instantiations in ``text``.

    Precise (named-port) form only: ``Type [#(params)] inst ( .port(...`` -- a
    type, an instance name, then a port list that opens with a NAMED connection
    (``.port(``). This is what CoreSmith's generated stage submodules use, and it
    avoids the `for (`/`if (`/declaration false-positives a looser ``word word (``
    regex produces (a two-identifier `TypeName inst (` is ambiguous with control
    flow in Verilog without a full parser). Positional-only instances are not
    counted -- acceptable for this secondary structural check."""
    count = 0
    for m in re.finditer(
        r"\b([A-Za-z_]\w*)\s*(?:#\s*\([^;]*\)\s*)?([A-Za-z_]\w*)\s*\(\s*\.",
        text,
    ):
        if m.group(1) in _KW_NOT_TYPE or m.group(2) in _KW_NOT_TYPE:
            continue
        count += 1
    return count


# ---------------------------------------------------------------------------
# Stage-map inputs
# ---------------------------------------------------------------------------
@dataclass
class StageMap:
    """Declared per-stage arithmetic budget for a block (from the Amaranth model's
    machine-readable STAGE_BUDGET, parsed by ``latency_audit``)."""
    stages: list[dict] = field(default_factory=list)
    declared_latency: Optional[int] = None

    @property
    def present(self) -> bool:
        return bool(self.stages)

    @property
    def stage_count(self) -> int:
        return len(self.stages)

    @property
    def total_op_slots(self) -> int:
        """Sum of per-stage op-list lengths -- the physical datapath op budget
        (one reusable datapath per stage). A correctly-pipelined RTL contains
        ~this many arithmetic ops; the cloud instantiates them all x trip-counts."""
        return sum(len(s.get("ops") or []) for s in self.stages)

    @property
    def applies(self) -> bool:
        """The module-per-stage protocol applies to a genuine multi-stage
        datapath: a declared latency deep enough to pipeline, or many stages / a
        candidate search. Single-stage / trivial blocks are unaffected."""
        if not self.present:
            return False
        if self.declared_latency is not None and self.declared_latency > 10:
            return True
        return self.stage_count >= 3


def stage_map_from_budget(stage_budget: Optional[list[dict]],
                          declared_latency: Optional[int] = None) -> StageMap:
    return StageMap(stages=list(stage_budget or []),
                    declared_latency=declared_latency)


def load_stage_map(project_root: str | os.PathLike, block_name: str) -> StageMap:
    """Best-effort: locate the block's Amaranth model and parse its STAGE_BUDGET /
    DECLARED_LATENCY_CYCLES (reusing ``latency_audit``). Empty StageMap on any
    failure -- the census then relies on the multiplier cap alone."""
    try:
        from .latency_audit import find_block_model, parse_stage_budget
        path = find_block_model(project_root, block_name)
        if path is None:
            return StageMap()
        stages, declared = parse_stage_budget(path.read_text(encoding="utf-8"))
        return StageMap(stages=stages, declared_latency=declared)
    except Exception:  # noqa: BLE001 - never block RTL generation
        return StageMap()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
@dataclass
class BlockCensus:
    name: str
    kind: str                    # 'always' | 'continuous'
    self_mul: int
    self_ops: int
    eff_mul: int
    eff_ops: int
    top_calls: list[tuple[str, int, int]] = field(default_factory=list)
    # (callee, call_weight, callee_eff_mul) -- the amplifiers, biggest first


@dataclass
class DefCensus:
    name: str
    self_mul: int
    self_ops: int
    eff_mul: int
    eff_ops: int
    call_sites: int


@dataclass
class StageLintReport:
    blocks: list[BlockCensus] = field(default_factory=list)
    defs: dict[str, DefCensus] = field(default_factory=dict)
    mul_cap: int = 64
    factor: float = 8.0
    stage_map: StageMap = field(default_factory=StageMap)
    # findings
    mul_violations: list[BlockCensus] = field(default_factory=list)
    factor_violations: list[BlockCensus] = field(default_factory=list)
    # structural (Deliverable 2)
    module_instances: int = 0
    stage_module_deficient: bool = False

    @property
    def worst_mul(self) -> int:
        return max((b.eff_mul for b in self.blocks), default=0)

    @property
    def total_eff_ops(self) -> int:
        return max((b.eff_ops for b in self.blocks), default=0)

    @property
    def ok(self) -> bool:
        return not self.mul_violations and not self.factor_violations


def census_rtl(verilog_src: str, *,
               stage_map: Optional[StageMap] = None,
               mul_cap: Optional[int] = None,
               factor: Optional[float] = None,
               enforce_stage_modules: bool = False) -> StageLintReport:
    """Full arithmetic-realization census of a Verilog source.

    ``stage_map`` (optional) enables the total-ops FACTOR check + the structural
    module-per-stage check. Without it, only the per-block multiplier cap applies.
    """
    sm = stage_map or StageMap()
    cap = _mul_cap() if mul_cap is None else mul_cap
    fac = _factor() if factor is None else factor
    report = StageLintReport(mul_cap=cap, factor=fac, stage_map=sm)

    try:
        blanked = _blank_comments_strings(verilog_src)
    except Exception:  # noqa: BLE001
        return report
    params = _parse_params(blanked)

    # module names defined in this file (so instances of THEM aren't miscounted,
    # and stage submodules can be recognized).
    module_names = set(re.findall(r"\bmodule\s+([A-Za-z_]\w*)", blanked))

    # task/function defs
    def_spans = _extract_defs(blanked)
    defnames = set(def_spans)
    def_self: dict[str, _RegionCensus] = {}
    for name, (start, end, _hdr) in def_spans.items():
        def_self[name] = _census_region(blanked[start:end], params, defnames)

    # transitive expansion of a definition (memoized; cycle-guarded)
    _memo: dict[str, tuple[int, int]] = {}

    def _expand(name: str, stack: frozenset, depth: int) -> tuple[int, int]:
        if name in _memo:
            return _memo[name]
        if name not in def_self or depth > _MAX_EXPAND_DEPTH:
            return 0, 0
        base = def_self[name]
        tot_mul, tot_ops = base.mul, base.ops
        for callee, w in base.calls.items():
            if callee in stack or callee == name:
                continue  # cycle guard (unsynthesizable recursion)
            cm, co = _expand(callee, stack | {name}, depth + 1)
            tot_mul += w * cm
            tot_ops += w * co
        # memoize only when acyclic-safe (no self/stack skip fired materially);
        # the stack-independent value is exact for the common acyclic case.
        _memo[name] = (tot_mul, tot_ops)
        return tot_mul, tot_ops

    for name, base in def_self.items():
        em, eo = _expand(name, frozenset(), 0)
        report.defs[name] = DefCensus(
            name=name, self_mul=base.mul, self_ops=base.ops,
            eff_mul=em, eff_ops=eo, call_sites=0,
        )

    # count global call sites per def (informational)
    all_calls = _count_calls(blanked, defnames)
    # add calls that live inside loops (weighted) are not needed for the count;
    # this is the raw textual call-site tally.
    for name in report.defs:
        report.defs[name].call_sites = all_calls.get(name, 0)

    # ---- per-block census ----
    # Blank task/function def spans so always-block extraction/census excludes
    # their bodies (they are amplified via CALLS, not by textual inclusion).
    chars = list(blanked)
    for name, (start, end, _hdr) in def_spans.items():
        _blank_span(chars, start, end)
    no_defs = "".join(chars)

    def _expand_block(base: _RegionCensus) -> tuple[int, int]:
        tot_mul, tot_ops = base.mul, base.ops
        for callee, w in base.calls.items():
            cm, co = _expand(callee, frozenset(), 0)
            tot_mul += w * cm
            tot_ops += w * co
        return tot_mul, tot_ops

    def _top_calls(base: _RegionCensus) -> list[tuple[str, int, int]]:
        """(callee, call_weight, callee_eff_mul), biggest amplifier first."""
        rows: list[tuple[int, tuple[str, int, int]]] = []
        for callee, w in base.calls.items():
            cm, _ = _expand(callee, frozenset(), 0)
            rows.append((w * cm, (callee, w, cm)))
        rows.sort(key=lambda r: -r[0])
        return [t for _key, t in rows][:5]

    always_text = no_defs
    for label, body in _iter_always_blocks(always_text):
        base = _census_region(body, params, defnames)
        em, eo = _expand_block(base)
        bc = BlockCensus(
            name=label, kind="always", self_mul=base.mul, self_ops=base.ops,
            eff_mul=em, eff_ops=eo, top_calls=_top_calls(base),
        )
        report.blocks.append(bc)

    # synthetic <continuous> pseudo-block: everything OUTSIDE always/task/function
    # (continuous assigns + their function-call chains). Catches a cloud hidden in
    # assign expressions rather than an always block.
    cont_chars = list(no_defs)
    for label_m in re.finditer(r"\b(always|initial)\b", no_defs):
        # blank each always block span
        i, n = label_m.end(), len(no_defs)
        while i < n and no_defs[i].isspace():
            i += 1
        if i < n and no_defs[i] == "@":
            i += 1
            while i < n and no_defs[i].isspace():
                i += 1
            if i < n and no_defs[i] == "(":
                i = _find_matching(no_defs, i, "(", ")")
            elif i < n and no_defs[i] == "*":
                i += 1
            while i < n and no_defs[i].isspace():
                i += 1
        if no_defs[i:i + 5] == "begin":
            _s, _e, after = _match_begin_end(no_defs, i)
            _blank_span(cont_chars, label_m.start(), after)
        else:
            _blank_span(cont_chars, label_m.start(), _stmt_end(no_defs, i) + 1)
    cont_text = "".join(cont_chars)
    cont_base = _census_region(cont_text, params, defnames)
    if cont_base.mul or cont_base.ops or cont_base.calls:
        em, eo = _expand_block(cont_base)
        report.blocks.append(BlockCensus(
            name="<continuous>", kind="continuous",
            self_mul=cont_base.mul, self_ops=cont_base.ops,
            eff_mul=em, eff_ops=eo, top_calls=_top_calls(cont_base),
        ))

    # ---- verdicts ----
    for b in report.blocks:
        if b.eff_mul > cap:
            report.mul_violations.append(b)
    if sm.present and sm.total_op_slots > 0:
        budget = max(sm.total_op_slots * fac, _FACTOR_MIN_OPS)
        for b in report.blocks:
            if b.eff_ops > budget:
                report.factor_violations.append(b)

    # ---- structural module-per-stage (Deliverable 2) ----
    report.module_instances = _count_module_instances(no_defs, module_names)
    if enforce_stage_modules and sm.applies and not report.ok:
        # only bites a block already flagged by the arithmetic census -- it turns
        # the "collapse" finding into the actionable "emit N stage modules" remedy.
        if report.module_instances < sm.stage_count:
            report.stage_module_deficient = True

    return report


# ---------------------------------------------------------------------------
# Standalone structural check (Deliverable 2, testable in isolation)
# ---------------------------------------------------------------------------
@dataclass
class StageModuleReport:
    applies: bool
    declared_stage_count: int
    instantiated_submodules: int
    monolithic_always: bool          # datapath sits in one big always block
    ok: bool


def check_stage_modules(verilog_src: str, stage_map: StageMap) -> StageModuleReport:
    """Structural check: when the module-per-stage protocol applies, the top
    block must instantiate at least ``stage_count`` stage submodules rather than
    carry the whole datapath in one always block."""
    if not stage_map.applies:
        return StageModuleReport(False, stage_map.stage_count, 0, False, True)
    try:
        blanked = _blank_comments_strings(verilog_src)
    except Exception:  # noqa: BLE001
        return StageModuleReport(True, stage_map.stage_count, 0, False, True)
    module_names = set(re.findall(r"\bmodule\s+([A-Za-z_]\w*)", blanked))
    n_inst = _count_module_instances(blanked, module_names)
    # monolithic = a single always block holds a big share of the arithmetic
    rep = census_rtl(verilog_src, stage_map=stage_map)
    heavy = [b for b in rep.blocks if b.kind == "always" and b.eff_ops > 0]
    monolithic = len(heavy) <= 1 and rep.total_eff_ops > 0
    ok = n_inst >= stage_map.stage_count
    return StageModuleReport(True, stage_map.stage_count, n_inst, monolithic, ok)


# ---------------------------------------------------------------------------
# Report formatting -- the retry feedback (Deliverable 3)
# ---------------------------------------------------------------------------
def format_stage_lint_report(report: StageLintReport, block: str = "",
                             *, trajectory: str = "",
                             fresh_session: bool = False) -> str:
    """Directive-rich, anti-pattern-guarded regen feedback (or '' if clean).

    Reproduces the mem-price fixes-4 lesson: a census TABLE (always-block ->
    effective ops -> vs cap), the declared stage map verbatim, and a NUMBERED
    mechanical remedy (module-per-stage), so the regen cannot misread the
    rejection as a granularity ask. ``trajectory`` / ``fresh_session`` carry the
    cross-retry escalation state (Deliverable 3)."""
    if report.ok and not report.stage_module_deficient:
        return ""
    cap = report.mul_cap
    lines: list[str] = []
    lines.append(
        "UNSYNTHESIZABLE COMBINATIONAL CLOUD -- deterministic stage-realization "
        "lint (pre-yosys)" + (f" in {block}" if block else "") + "."
    )
    lines.append(
        "A multi-stage datapath was collapsed into single-cycle combinational "
        "logic. yosys `proc` unrolls every constant-bound `for` loop and inlines "
        "every task/function, so the effective arithmetic below is instantiated "
        "in ONE cycle -- it elaborates for minutes and times out (and never "
        "meets timing)."
    )
    lines.append("")
    lines.append("ARITHMETIC CENSUS (effective ops after loop-unroll + task-inline):")
    lines.append("")
    lines.append(
        f"  {'block / always':<26} {'eff. multipliers':>16} {'eff. ops':>10}  "
        f"{'vs cap':>10}"
    )
    lines.append("  " + "-" * 68)
    _ops_over = {b.name for b in report.factor_violations}
    for b in sorted(report.blocks, key=lambda x: (-x.eff_mul, -x.eff_ops)):
        if b.eff_mul > cap:
            flag = "OVER-mul"
        elif b.name in _ops_over:
            flag = "OVER-ops"
        else:
            flag = "ok"
        lines.append(
            f"  {b.name[:26]:<26} {b.eff_mul:>16,} {b.eff_ops:>10,}  "
            f"{f'{flag} ({cap})':>10}"
        )
    lines.append("  " + "-" * 68)

    # name the amplifiers for the worst block
    if report.mul_violations:
        worst = max(report.mul_violations, key=lambda x: x.eff_mul)
        lines.append("")
        lines.append(
            f"Worst block `{worst.name}`: {worst.eff_mul:,} effective multipliers "
            f"(> cap {cap}) -- {worst.self_mul:,} written directly, the rest from "
            "inlined task/function calls amplified by enclosing loops:"
        )
        for callee, weight, callee_mul in worst.top_calls:
            if callee_mul <= 0:
                continue
            lines.append(
                f"  - `{callee}(...)` called x{weight} (loop-amplified) x "
                f"{callee_mul:,} multipliers each = {weight * callee_mul:,}"
            )
        amps = [(dn, dc) for dn, dc in
                sorted(report.defs.items(), key=lambda kv: -kv[1].eff_mul)
                if dc.eff_mul > 0][:5]
        if amps:
            lines.append("  amplifying task/functions (multipliers per invocation):")
            for dn, dc in amps:
                lines.append(
                    f"    - `{dn}`: {dc.eff_mul:,} eff. multipliers/invocation, "
                    f"{dc.call_sites} textual call site(s)"
                )

    if report.factor_violations:
        budget = int(max(report.stage_map.total_op_slots * report.factor,
                         _FACTOR_MIN_OPS))
        lines.append("")
        if not report.mul_violations:
            # dv-hardening-11: the multiplier metric is GREEN -- say so, or
            # the regen keeps optimizing it (observed: 2 regens polished an
            # already-passing mul census while the ops budget stayed 12x over).
            lines.append(
                f"BINDING CRITERION: TOTAL EFFECTIVE OPS. The multiplier census "
                f"is GREEN (max {max((b.eff_mul for b in report.blocks), default=0):,} "
                f"<= cap {cap}) -- do NOT restructure multipliers further."
            )
        lines.append(
            f"Total effective ops exceed the declared stage-map budget "
            f"({report.stage_map.total_op_slots} op-slots x {report.factor:g}x, "
            f"floor {_FACTOR_MIN_OPS} = {budget:,}) -- the RTL is not sharing one "
            "datapath across stages."
        )
        for b in sorted(report.factor_violations, key=lambda x: -x.eff_ops)[:3]:
            lines.append(
                f"  - SERIALIZE `{b.name}`: {b.eff_ops:,} effective ops in ONE "
                f"cycle (budget {budget:,}). A zero-multiplier unrolled walk "
                f"(compares/adds/shifts over a constant-bound loop) is still a "
                f"cloud. Convert it to an FSM that processes ONE element per "
                f"cycle -- the census is MAX-over-arms, so a genuinely "
                f"sequential walk passes automatically."
            )

    # the declared stage map verbatim
    if report.stage_map.present:
        try:
            from .latency_audit import audit_source, format_stage_map  # noqa: F401
        except Exception:  # noqa: BLE001
            pass
        lines.append("")
        lines.append("DECLARED STAGE MAP (from the model's audited latency budget):")
        for s in report.stage_map.stages:
            nm = s.get("name", "?")
            lat = s.get("latency_cycles", 1)
            it = s.get("iters", 1)
            ops = ", ".join(str(o) for o in (s.get("ops") or [])) or "(none)"
            lines.append(f"  - stage {nm}: {lat} cyc x {it} -> ops [{ops}]")
        if report.stage_map.declared_latency is not None:
            lines.append(
                f"  declared latency: {report.stage_map.declared_latency} cycles."
            )

    # the numbered mechanical remedy (module-per-stage protocol)
    lines.append("")
    lines.append("MANDATORY REMEDY -- realize the stages STRUCTURALLY (do this exactly):")
    lines.append(
        "  1. Emit ONE submodule per named stage above, each with REGISTERED "
        "outputs (`always @(posedge clk)`); the module boundary IS the pipeline "
        "register boundary. Put the stage submodules in this same .v file (the "
        "toolflow de-dups multiple modules per file)."
    )
    lines.append(
        "  2. Write ONE small controller FSM that ITERATES the search/candidates "
        "over CYCLES (mode 0..N over N cycles on ONE reusable datapath), driving "
        "the stage submodules with a valid/ready or enable chain. The controller "
        "must be ~arithmetic-free (indexing + handshakes only)."
    )
    lines.append(
        "  3. Each stage submodule's arithmetic must fit its per-stage op budget "
        f"above (chained delay <= one clock period); NO single always block may "
        f"exceed {cap} effective multipliers, and the module's TOTAL effective "
        "ops must fit the stage-map budget -- multiplier-free unrolled walks "
        "(token/coefficient scans) count and must be serialized over cycles."
    )
    lines.append("  FORBIDDEN (this is what was just rejected):")
    lines.append(
        "  - Evaluating the whole N-candidate / N-mode search in one cycle "
        "(one `always` that calls tasks over `for` loops = the cloud you wrote)."
    )
    lines.append(
        "  - Stage bodies written as `task`/`function`s inlined into ONE always "
        "block: a task is a COMBINATIONAL cone, not a register boundary. Calling "
        "one over a loop MULTIPLIES its arithmetic; it does not pipeline it."
    )
    lines.append(
        "  - Decorative FSM `wait`/cycle-burn states around a body that still "
        "does all the arithmetic in one state. Registering the OUTPUT is not "
        "enough; the ARITHMETIC must be split across the registered stages."
    )

    if report.stage_module_deficient:
        lines.append("")
        lines.append(
            f"STRUCTURAL: the top module instantiates {report.module_instances} "
            f"submodule(s) but the datapath declares {report.stage_map.stage_count} "
            "stages. Instantiate one registered submodule per stage (step 1)."
        )

    # cross-retry escalation (Deliverable 3)
    if trajectory == "identical":
        lines.append("")
        lines.append(
            "TRAJECTORY: your re-submission is STRUCTURALLY IDENTICAL to the last "
            "rejection (same per-block multiplier census). Re-registering outputs "
            "or renaming states did NOT reduce the arithmetic. You MUST physically "
            "SPLIT the search across registered stage submodules (step 1-2) -- "
            "there is no way to pass this gate with the arithmetic in one block."
        )
    if fresh_session:
        lines.append("")
        lines.append(
            "FRESH-SESSION ESCALATION: prior incremental edits did not converge. "
            "DISCARD the collapsed structure and regenerate the block from the "
            "stage map above as N registered stage submodules + one iterative "
            "controller FSM. Do not patch the single-always cloud."
        )
    return "\n".join(lines)


def census_signature(report: StageLintReport) -> str:
    """Stable signature of the census verdict (per-block effective multipliers).
    Two rejections with the same signature = the regen did not move the
    arithmetic (drives the identical-resubmission escalation, Deliverable 3)."""
    import hashlib
    payload = "|".join(
        f"{b.name}:{b.eff_mul}:{b.eff_ops}"
        for b in sorted(report.blocks, key=lambda x: x.name)
    )
    return hashlib.sha1(payload.encode("utf-8", "replace")).hexdigest()[:16]


__all__ = [
    "stage_lint_enabled", "stage_modules_enabled",
    "StageMap", "stage_map_from_budget", "load_stage_map",
    "BlockCensus", "DefCensus", "StageLintReport", "census_rtl",
    "StageModuleReport", "check_stage_modules",
    "format_stage_lint_report", "census_signature",
]
