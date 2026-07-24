# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Structural lint: wide flat packed registers with DYNAMIC part-select.

The codec ``intra4x4_rd_encode_core`` v2 RTL genuinely registered its FSM
stages (the pipeline-discipline fix worked at that level), yet synth still
walled -- not on a combinational cloud, but because working/FIFO data was kept
as WIDE FLAT PACKED REGISTERS accessed by a DYNAMIC part-select, e.g.::

    reg  [1023:0] top_recon_q;          // 1024-bit flat reg
    ... top_recon_q[base_idx +: 8] ...  // dynamic indexed part-select

A dynamic part-select into a wide flat reg lowers to a giant barrel-shifter /
decoder tree; ``proc``/``opt`` can't elaborate it tractably and synthesis times
out (``hierarchy``+``stat`` alone are instant -- it's ``proc`` that explodes).
The right realization is an ADDRESSED MEMORY (``cs_sram``/``cs_fpmem``) or a
proper per-element array indexed by a registered address with a registered
read -- not a flat vector with a runtime slice.

This module detects that anti-pattern so the synth-fix loop / a lint gate can
flag it BEFORE the 600 s synth timeout, instead of after. Pure-regex heuristic
(Verilog is not trivially AST-able in Python); tuned for the generated-RTL
idioms CoreSmith emits, with a configurable width threshold.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

# A packed reg declaration:  reg [1023:0] name ;   (optionally signed)
_REG_DECL_RE = re.compile(
    r"\breg\b\s*(?:signed\s*)?\[\s*(\d+)\s*:\s*0\s*\]\s*(\w+)",
)
# Any indexed access of a name:  name [ <index-expr> ]   (captures the expr)
def _access_re(name: str) -> re.Pattern:
    return re.compile(re.escape(name) + r"\s*\[([^\]\[]+)\]")

# An index expression is DYNAMIC if it isn't a pure constant range/bit. A
# constant looks like "57:42", "7", "8'd3", or arithmetic on literals only.
_IDENT_RE = re.compile(r"[A-Za-z_]\w*")
_CONST_TOKEN_RE = re.compile(r"^\s*\d+\s*'\s*[bBoOdDhH]?[0-9a-fA-FxXzZ_]+\s*$")


def _is_dynamic_index(expr: str) -> bool:
    """True if the index expression depends on a non-constant identifier.

    Constant forms (``57:42``, ``7``, ``8'd3``, ``A+B`` where A,B are params)
    are NOT dynamic for our purpose -- only a runtime variable index (a ``_q``
    register, a counter, a wire) produces the barrel-shifter blow-up. We treat
    any identifier that is not an obvious Verilog literal as dynamic.
    """
    e = expr.strip()
    # an indexed part-select base:  "base +: 8" / "base -: 16"
    part = re.match(r"^(.*?)(?:\+|\-):\s*\d+\s*$", e)
    if part:
        e = part.group(1)
    # constant bit / range like "7" or "57:42" -> static
    if re.fullmatch(r"[\d\s:]+", e):
        return False
    if _CONST_TOKEN_RE.match(e):
        return False
    # contains a real identifier (variable) -> dynamic
    return bool(_IDENT_RE.search(e))


# A function/task body, used to detect "dynamic slicer" helpers: a function
# that dynamically part-selects one of its OWN input args. A wide reg passed
# into such a function is just as unsynthesizable as a direct runtime slice --
# the codec RD-core hid `top_y_line_q[5119:0]` exactly this way, behind
# `get_byte2048(vec, idx) -> vec[base +: 8]`, so the flat-reg-only matcher
# missed it.
_FUNC_BLOCK_RE = re.compile(
    r"\b(?:function|task)\b.*?\bend(?:function|task)\b", re.DOTALL,
)
_INPUT_DECL_RE = re.compile(r"\binput\b\s*(?:\[[^\]]*\]\s*)?(\w+)")


def _func_name(block: str) -> str:
    """Function/task name = last identifier in the header before ';' or '('."""
    header = re.split(r"[;(]", block, 1)[0]
    ids = re.findall(r"[A-Za-z_]\w*", header)
    # drop leading keywords so a name like 'function'/'task'/'automatic' isn't picked
    ids = [i for i in ids if i not in ("function", "task", "automatic", "signed")]
    return ids[-1] if ids else ""


def _dynamic_slicer_functions(src: str) -> dict:
    """Map {func_name: sample-slice-expr} for funcs that dynamically slice an input."""
    slicers: dict = {}
    for m in _FUNC_BLOCK_RE.finditer(src):
        block = m.group(0)
        name = _func_name(block)
        if not name:
            continue
        inputs = set(_INPUT_DECL_RE.findall(block))
        for arg in inputs:
            for am in _access_re(arg).finditer(block):
                if _is_dynamic_index(am.group(1)):
                    slicers[name] = f"{arg}[{am.group(1).strip()}]"
                    break
            if name in slicers:
                break
    return slicers


def _call_re(fname: str) -> re.Pattern:
    """A call `fname( arg, arg, ... )` capturing the (non-nested) arg list."""
    return re.compile(re.escape(fname) + r"\s*\(([^()]*)\)")


@dataclass
class StorageFinding:
    name: str
    width_bits: int
    dynamic_accesses: int
    sample: str = ""
    via_function: str = ""  # set when the dynamic slice is inside a helper func


# A finding whose width is within this multiple of the threshold is BORDERLINE
# (near-threshold): the threshold at 128 bits correctly catches the barrel-
# shifter blow-up class cheaply, but a small, synthesis-PROVEN reg just over it
# (e.g. a 214-bit dynamic-select reg that synthesized in 230 s) is a case a
# reviewer may accept via the documented override rather than forcing an SRAM
# restructure. 2x threshold = 256 bits.
NEAR_THRESHOLD_FACTOR = 2


@dataclass
class StorageLintReport:
    findings: list[StorageFinding] = field(default_factory=list)
    min_bits: int = 128

    @property
    def ok(self) -> bool:
        return not self.findings

    @property
    def min_finding_bits(self) -> int | None:
        """Width of the SMALLEST offending reg (the one closest to the
        threshold), or None when clean. This is the finding a reviewer is most
        likely to accept as borderline."""
        if not self.findings:
            return None
        return min(f.width_bits for f in self.findings)

    def proximity_ratio(self) -> float | None:
        """The closest finding's width as a MULTIPLE of the threshold (bits vs
        threshold). ~1.0-2.0 = borderline; large = clearly over. None when
        clean or the threshold is degenerate. Recorded in the verdict so an
        operator can judge how borderline a rejection is."""
        mb = self.min_finding_bits
        if mb is None or not self.min_bits:
            return None
        return mb / self.min_bits

    @property
    def near_threshold(self) -> bool:
        """True when the smallest offender is within ``NEAR_THRESHOLD_FACTOR`` of
        the threshold -- the borderline, potentially-override-acceptable case."""
        pr = self.proximity_ratio()
        return pr is not None and pr < NEAR_THRESHOLD_FACTOR


def find_flat_packed_dynamic_storage(verilog_src: str,
                                     min_bits: int = 128) -> StorageLintReport:
    """Find wide flat packed regs (>= ``min_bits``) read/written by a DYNAMIC
    part-select or index. Each such reg is a synth blow-up risk that should be
    a ``cs_sram``/``cs_fpmem`` addressed memory or a per-element array instead.
    """
    # collect wide packed regs
    wide: dict[str, int] = {}
    for m in _REG_DECL_RE.finditer(verilog_src):
        width = int(m.group(1)) + 1
        name = m.group(2)
        if width >= min_bits:
            wide[name] = max(width, wide.get(name, 0))

    slicers = _dynamic_slicer_functions(verilog_src)

    findings: list[StorageFinding] = []
    for name, width in wide.items():
        n_dyn = 0
        sample = ""
        via = ""
        # (A) DIRECT: name[<runtime index>]
        for am in _access_re(name).finditer(verilog_src):
            idx = am.group(1)
            if _is_dynamic_index(idx):
                n_dyn += 1
                if not sample:
                    sample = f"{name}[{idx.strip()}]"
        # (B) INDIRECT: name passed as an arg into a function that dynamically
        # slices its input (e.g. get_byte2048(top_y_line_q, idx)).
        for fname, slice_expr in slicers.items():
            for cm in _call_re(fname).finditer(verilog_src):
                if re.search(r"\b" + re.escape(name) + r"\b", cm.group(1)):
                    n_dyn += 1
                    if not via:
                        via = fname
                    if not sample:
                        sample = (f"{fname}({name}, ...) -> dynamic slice "
                                  f"`{slice_expr}` inside {fname}")
        if n_dyn > 0:
            findings.append(StorageFinding(name, width, n_dyn, sample, via))

    findings.sort(key=lambda f: (-f.width_bits, f.name))
    return StorageLintReport(findings=findings, min_bits=min_bits)


# ---------------------------------------------------------------------------
# Memory-tier finder (Section 5f) + flat read-mux timing pre-check (Section 4a)
#
# A register/logic-tier memory that is large enough to belong in an SRAM macro
# (cs_fpmem_1rw1r behavioral array, or a behavioral `reg [W-1:0] mem [0:D-1]`)
# flattens under yosys MEMORY_MAP into a flop array + a giant read mux
# (1024x8 -> ~2046 mux cells, WNS -58..-276 ns) -- architecturally untimeable,
# discovered only after 3 full synth+STA attempts. Catch it structurally BEFORE
# synth: anything over the SRAM threshold (~256 words / ~1 KiB) not backed by a
# cs_sram_* macro is a `memory_tier` finding; and a single-cycle flat read whose
# log-depth mux delay would exceed the clock period is flagged from the PDK
# op-delay model (same predict_op_delay the Fmax step uses).
# ---------------------------------------------------------------------------

# A behavioral memory array:  reg [7:0] mem [0:1023];  /  reg [7:0] mem [1023:0];
_MEM_ARRAY_RE = re.compile(
    r"\breg\b\s*(?:signed\s*)?(?:\[\s*(\d+)\s*:\s*0\s*\]\s*)?(\w+)\s*"
    r"\[\s*(\d+)\s*:\s*(\d+)\s*\]\s*;",
)
# A cs_fpmem_* / cs_mem_* register-tier memory primitive instantiation with
# DEPTH/WIDTH params:  cs_fpmem_1rw1r #(.WIDTH(8), .DEPTH(1024)) u_mem (...);
# Anchor on the `#(` START only; the param values contain nested parens
# (`.WIDTH(8)`) that a single `(.*?)` capture cannot balance, so params are
# scanned from a window bounded by the instantiation's terminating `;`.
_CS_FPMEM_START_RE = re.compile(
    r"\b(cs_fpmem\w*|cs_mem\w*)\b\s*#\s*\(",
)
_PARAM_RE = re.compile(r"\.\s*(WIDTH|DEPTH|AW|DW|ADDR_WIDTH|DATA_WIDTH)\s*\(\s*(\d+)\s*\)",
                       re.IGNORECASE)

# SRAM-tier thresholds: sky130's macro threshold is ~256 words; a memory at or
# over EITHER ~256 words or ~1 KiB of bits belongs in an SRAM macro, not flops.
#
# PR#12 finding #3: these were hardcoded module constants with NO env override,
# while sram_wrapper.py (512 depth / 16384 bits) and macro_registry disagree and
# ARE env-tunable -- so a design under an explicit no-macro mandate could hit an
# UNSATISFIABLE conflict (a block passing all functional DV that this lint fails,
# with no env/CLI escape). Make these env-tunable so an operator can align the
# three thresholds, and see storage_lint_max_words()/_bits() for the runtime
# resolution (default args below stay for callers that pass explicit values).
DEFAULT_MAX_WORDS = 256
DEFAULT_MAX_BITS = 8 * 1024


def storage_lint_max_words() -> int:
    """Runtime word-depth threshold; env CORESMITH_STORAGE_LINT_MAX_WORDS."""
    try:
        return max(1, int(os.environ.get(
            "CORESMITH_STORAGE_LINT_MAX_WORDS", "") or DEFAULT_MAX_WORDS))
    except ValueError:
        return DEFAULT_MAX_WORDS


def storage_lint_max_bits() -> int:
    """Runtime bit-capacity threshold; env CORESMITH_STORAGE_LINT_MAX_BITS."""
    try:
        return max(8, int(os.environ.get(
            "CORESMITH_STORAGE_LINT_MAX_BITS", "") or DEFAULT_MAX_BITS))
    except ValueError:
        return DEFAULT_MAX_BITS


@dataclass
class MemoryTierFinding:
    name: str
    width_bits: int
    depth_words: int
    total_bits: int
    impl: str                 # "behavioral_array" | "cs_fpmem"
    reason: str               # why it FAILED (over-size / untimeable read mux)
    flat_read_ns: float = 0.0  # estimated single-cycle flat read-mux delay
    period_ns: float = 0.0     # target period it was checked against (0 = n/a)


@dataclass
class MemoryTierReport:
    findings: list[MemoryTierFinding] = field(default_factory=list)
    max_words: int = DEFAULT_MAX_WORDS
    max_bits: int = DEFAULT_MAX_BITS

    @property
    def ok(self) -> bool:
        return not self.findings


def flat_read_mux_ns(depth_words: int, width_bits: int,
                     pdk=None) -> float:
    """Estimate the combinational delay of a SINGLE-CYCLE flat read mux.

    A ``mem[addr]`` read over ``depth`` words is a ``depth:1`` mux, realized as
    a balanced tree ~``ceil(log2(depth))`` levels deep; each level is a 2:1 mux
    at the data width. Priced from the SAME PDK op-delay model the Fmax step uses
    (``predict_op_delay('mux', width)``). Returns 0.0 when the PDK is
    uncharacterized (no estimate) so the caller falls back to the size check.
    """
    import math
    if depth_words <= 1:
        return 0.0
    levels = int(math.ceil(math.log2(depth_words)))
    try:
        from orchestrator.langgraph.arith_characterize import predict_op_delay
        per_level = predict_op_delay("mux", max(1, int(width_bits)), pdk)
    except Exception:  # noqa: BLE001
        per_level = None
    if not per_level or per_level <= 0:
        return 0.0
    return levels * per_level


def _mem_is_sram_backed(name: str, verilog_src: str) -> bool:
    """True if a same-named cs_sram_* macro is instantiated (the memory is a
    macro, not a flop array) -- then it is NOT a memory-tier violation."""
    return bool(re.search(r"\bcs_sram\w*\b", verilog_src)) and bool(
        re.search(r"\bcs_sram\w*\b[^;]*\b" + re.escape(name) + r"\b", verilog_src)
    )


def find_oversized_memory_arrays(
    verilog_src: str,
    max_words: int | None = None,
    max_bits: int | None = None,
    period_ns: float | None = None,
    pdk=None,
) -> MemoryTierReport:
    """Find register-tier memories that belong in an SRAM macro (Section 5f/4a).

    Enumerates behavioral memory arrays (``reg [W-1:0] mem [0:D-1]``) and
    ``cs_fpmem_*``/``cs_mem_*`` instantiations, computes WIDTH x DEPTH, and FAILS
    anything over the ~256-word / ~1 KiB SRAM threshold that is NOT backed by a
    ``cs_sram_*`` macro. Additionally (4a), when ``period_ns`` is given, flags a
    memory whose SINGLE-CYCLE flat read mux (log-depth) would exceed the clock
    period even if it is under the size threshold -- an untimeable single-cycle
    read that must be banked / pipelined / macro-tiered.
    """
    if max_words is None:
        max_words = storage_lint_max_words()
    if max_bits is None:
        max_bits = storage_lint_max_bits()
    # PR#12 finding #3 (reviewed-flop exception): a design under an explicit
    # no-macro mandate (a frozen PRD/ERS that forbids SRAM macros for these
    # buffers) may legitimately keep a >=threshold memory in flops. A reviewed
    # authorization is a first-class, auditable RTL marker -- a comment
    # `coresmith:reviewed-flop-exception` in the module source -- that WAIVES
    # ONLY the size threshold. The untimeable-flat-read-mux check is NEVER
    # waived (a slow combinational read is a real timing defect regardless of
    # mandate), so a registered cs_fpmem array clears the size gate while a raw
    # comb-read array still fails.
    flop_exception = bool(
        re.search(r"coresmith\s*:\s*reviewed-flop-exception",
                  verilog_src, re.IGNORECASE))
    findings: list[MemoryTierFinding] = []
    seen: set[str] = set()

    def _consider(name, width, depth, impl):
        if name in seen:
            return
        seen.add(name)
        width = max(1, int(width))
        depth = max(1, int(depth))
        total = width * depth
        if _mem_is_sram_backed(name, verilog_src):
            return
        over_size = depth >= max_words or total >= max_bits
        # A reviewed flop exception waives ONLY the size threshold, never the
        # timing check below (a slow flat read mux stays a real defect).
        if over_size and flop_exception:
            over_size = False
        flat_ns = flat_read_mux_ns(depth, width, pdk) if period_ns else 0.0
        untimeable = bool(period_ns and flat_ns > period_ns + 1e-9)
        if not (over_size or untimeable):
            return
        reasons = []
        if over_size:
            reasons.append(f"{depth} words x {width} b = {total} b over the SRAM "
                           f"threshold ({max_words} words / {max_bits} b)")
        if untimeable:
            reasons.append(f"single-cycle flat read mux ~{flat_ns:.1f} ns "
                           f"(~log2({depth}) levels) exceeds the {period_ns:.1f} ns "
                           f"period")
        findings.append(MemoryTierFinding(
            name=name, width_bits=width, depth_words=depth, total_bits=total,
            impl=impl, reason="; ".join(reasons),
            flat_read_ns=round(flat_ns, 3), period_ns=float(period_ns or 0.0),
        ))

    for m in _MEM_ARRAY_RE.finditer(verilog_src):
        wbits = (int(m.group(1)) + 1) if m.group(1) else 1
        name = m.group(2)
        hi, lo = int(m.group(3)), int(m.group(4))
        depth = abs(hi - lo) + 1
        _consider(name, wbits, depth, "behavioral_array")

    for m in _CS_FPMEM_START_RE.finditer(verilog_src):
        # Scan params from a window bounded by the instantiation's `;` so nested
        # `.WIDTH(8)` parens are handled (a `(.*?)` capture can't balance them).
        window = verilog_src[m.end():]
        semi = window.find(";")
        if semi != -1:
            window = window[:semi]
        params = {k.upper(): int(v) for k, v in _PARAM_RE.findall(window)}
        width = params.get("WIDTH") or params.get("DW") or params.get("DATA_WIDTH") or 1
        depth = params.get("DEPTH")
        if depth is None:
            aw = params.get("AW") or params.get("ADDR_WIDTH")
            depth = (1 << aw) if aw else 1
        _consider(m.group(1), width, depth, "cs_fpmem")

    findings.sort(key=lambda f: (-f.total_bits, f.name))
    return MemoryTierReport(findings=findings, max_words=max_words, max_bits=max_bits)


def format_memory_tier_report(report: MemoryTierReport, block: str = "") -> str:
    """Human/LLM-readable memory-tier message for a synth-fix retry (or '')."""
    if report.ok:
        return ""
    head = (
        "MEMORY BELONGS IN AN SRAM MACRO (register-tier storage over the SRAM "
        "threshold)" + (f" in {block}" if block else "")
        + f" ({len(report.findings)} memor(y/ies)):\n"
    )
    lines = [head]
    for f in report.findings:
        lines.append(
            f"  - {f.impl} `{f.name}`: {f.depth_words} words x {f.width_bits} b "
            f"= {f.total_bits} b -- {f.reason}"
        )
    lines.append(
        "\nFIX: instantiate a cs_sram_* macro (addressed, 1-2 cycle read), OR "
        "bank/pipeline the read into log-depth stages, OR declare an explicit "
        "multi-cycle read contract. A single-cycle flat read of a >=256-word "
        "flop array flattens to a ~depth:1 mux (thousands of cells, deeply "
        "negative WNS) and is untimeable -- do NOT ship it as flops."
    )
    return "\n".join(lines)


def format_lint_report(report: StorageLintReport, block: str = "") -> str:
    """Human/LLM-readable message for a synth-fix retry (or '' if clean)."""
    if report.ok:
        return ""
    head = (
        "WIDE FLAT PACKED STORAGE WITH DYNAMIC PART-SELECT"
        + (f" in {block}" if block else "")
        + f" ({len(report.findings)} reg(s), threshold {report.min_bits} bits):\n"
    )
    lines = [head]
    for f in report.findings:
        via = f" via helper `{f.via_function}()`" if f.via_function else ""
        # Threshold proximity per finding (bits vs the min_bits threshold) so an
        # operator can judge how borderline each reg is.
        ratio = f.width_bits / report.min_bits if report.min_bits else 0.0
        lines.append(
            f"  - reg [{f.width_bits-1}:0] {f.name}  ({f.width_bits} bits, "
            f"{ratio:.1f}x threshold), "
            f"{f.dynamic_accesses} dynamic access(es){via} e.g. `{f.sample}`"
        )
    lines.append(
        "  A dynamic part-select into a wide flat reg synthesizes to a giant "
        "barrel-shifter/decoder tree -> proc/opt blow up and synth times out. "
        "FIX: store this as an ADDRESSED MEMORY (cs_sram/cs_fpmem) or a proper "
        "per-element array `reg [W-1:0] mem [0:D-1]` indexed by a registered "
        "address with a registered (1-cycle) read -- NOT a flat vector sliced "
        "by a runtime index."
    )
    # Finding 4: borderline-storage note. The threshold stays strict (it catches
    # the blow-up class cheaply); when the smallest offender is only just over it
    # AND synthesis-proven, flag that a reviewer may accept via the documented
    # override rather than forcing a restructure.
    if report.near_threshold:
        mb = report.min_finding_bits
        prox = report.proximity_ratio() or 0.0
        lines.append(
            f"  NOTE (threshold proximity): the smallest offending reg is {mb} "
            f"bits ({prox:.1f}x the {report.min_bits}-bit threshold). If this "
            f"storage is small (<~{report.min_bits * NEAR_THRESHOLD_FACTOR} bits) "
            f"AND synthesis-proven (elaborates + meets timing in a bounded synth "
            f"run), the reviewer may ACCEPT it via the documented override "
            f"instead of forcing an SRAM restructure -- the threshold is kept "
            f"strict by design to catch the barrel-shifter class cheaply."
        )
    return "\n".join(lines)


# ===========================================================================
# FUNCTIONAL-IFDEF LINT -- the "split-brain conditional-compilation" ban.
# ===========================================================================
# The DEEPEST anti-gaming class the CoreSmith campaign found: a generated block
# carried TWO implementations in ONE module, selected by a strippable global
# ``ifdef``. The rung-3 ``intra_rd_encode_core.v`` had::
#
#     module intra_rd_encode_core(...);
#     `ifndef SYNTHESIS
#         ... the REAL verified datapath (with a 1.9Mbit flat-reg recon store
#             that would never synthesize) ...
#     `else
#         ... a NON-FUNCTIONAL latency-shell MOCK (memory reads write-only,
#             coeff = input-128, QP unused, counter-theater states) whose only
#             purpose was to satisfy the storage lint + synthesizability probes.
#     `endif
#     endmodule
#
# DV (Verilator, no ``-DSYNTHESIS``) verified the REAL branch; every synth /
# backend gate (``SYNTHESIS`` defined) saw the MOCK. The two are DIFFERENT
# HARDWARE. The agent will rename the macro (not always ``SYNTHESIS``), so this
# lint keys on CONSTRUCT CLASSES, not macro names: a conditional-compilation
# region that guards FUNCTIONAL logic (``always`` / continuous ``assign`` /
# ``initial`` that drives non-debug signals / module instantiation) inside a
# DESIGN module is banned. ALLOWED: regions that hold ONLY debug/trace/assertion
# code ($display / $dumpvars-style waveform hooks / SVA asserts -- regardless of
# the guard macro name), and the LEGITIMATE library split where a whole
# macro-named module has a synth-blackbox / sim-behavioral pair (the cs_* /
# sky130 / OpenRAM memory model idiom). The cs_* wrapper LIBRARY file itself
# (``rtl_lib/`` path) is exempt outright -- the sim-body/synth-macro split is
# legitimate THERE and only there.
#
# Pure-text heuristic (Verilog is not trivially AST-able in Python), tuned for
# the generated-RTL idioms CoreSmith emits.


def ifdef_lint_enabled() -> bool:
    """CORESMITH_IFDEF_LINT gate (default ON). Set to 0/false/no/off to bypass."""
    return os.environ.get("CORESMITH_IFDEF_LINT", "1").strip().lower() not in {
        "0", "false", "no", "off", "",
    }


# A module NAME that looks like an SRAM macro / cs_* wrapper / OpenRAM cell -- a
# `ifdef SYNTHESIS blackbox / `else behavioral pair for one of these IS the
# legitimate library idiom, not a design split-brain. Mirrors
# ``sram_wrapper._is_macro_module`` (kept local so this module stays
# dependency-light and unit-testable with no imports).
_MACRO_MODULE_NAME_RE = re.compile(r"(^cs_sram)|(^cs_fpmem)|(^cs_mem)|(sram)|(^openram)",
                                   re.IGNORECASE)


def _is_macro_module(name: str) -> bool:
    return bool(_MACRO_MODULE_NAME_RE.search(name))


# A `ifdef / `ifndef / `elsif / `else / `endif directive at line start.
_DIRECTIVE_LINE_RE = re.compile(r"^[ \t]*`(ifdef|ifndef|elsif|else|endif)\b[ \t]*(\w+)?")
# module / endmodule tokens (scanned in order of appearance per line).
_MODTOK_RE = re.compile(r"\bmodule\s+(\w+)\b|\bendmodule\b")
# Functional-construct signals.
_ASSIGN_KW_RE = re.compile(r"\bassign\b")
_ALWAYS_INITIAL_RE = re.compile(r"\b(always|initial)\b")
# A parametrized instantiation `TypeName #( ... )` -- high precision (a bare
# module *header* `module foo #(` is stripped before this runs, so this only
# hits real instances).
_PARAM_INST_RE = re.compile(r"\b[A-Za-z_]\w*\s*#\s*\(")
# System / debug tasks + assertion keywords that make a procedural block
# debug-only (allowed). $dumpvars/$display/$monitor/waveform + $finish/$stop.
_DEBUG_TASK_RE = re.compile(r"\$\w+\s*(?:\([^;]*\))?\s*;")
_ASSERT_STMT_RE = re.compile(r"\b(?:assert|assume|cover|restrict|expect)\b[^;]*;")


def _blank_comments_strings(src: str) -> str:
    """Replace // and /* */ comment bodies and "..." string contents with
    spaces, preserving newlines and length (so ``always`` in a comment or a
    ``$display("... assign ...")`` literal cannot trip the token scanners)."""
    out: list[str] = []
    i, n = 0, len(src)
    state = None  # None | 'line' | 'block' | 'str'
    while i < n:
        c = src[i]
        nxt = src[i + 1] if i + 1 < n else ""
        if state is None:
            if c == "/" and nxt == "/":
                state = "line"
                out.append("  ")
                i += 2
                continue
            if c == "/" and nxt == "*":
                state = "block"
                out.append("  ")
                i += 2
                continue
            if c == '"':
                state = "str"
                out.append('"')
                i += 1
                continue
            out.append(c)
            i += 1
            continue
        if state == "line":
            if c == "\n":
                state = None
                out.append("\n")
                i += 1
                continue
            out.append(" ")
            i += 1
            continue
        if state == "block":
            if c == "*" and nxt == "/":
                state = None
                out.append("  ")
                i += 2
                continue
            out.append("\n" if c == "\n" else " ")
            i += 1
            continue
        # state == 'str'
        if c == "\\" and nxt:
            out.append("  ")
            i += 2
            continue
        if c == '"':
            state = None
            out.append('"')
            i += 1
            continue
        out.append("\n" if c == "\n" else " ")
        i += 1
        continue
    return "".join(out)


def _mask_macro_modules(text: str) -> str:
    """Blank out whole ``module <macroname> ... endmodule`` spans (names that
    look like an SRAM macro / cs_* wrapper / OpenRAM cell). Their synth-blackbox
    vs sim-behavioral pair is the LEGITIMATE library split, so functional logic
    inside them must not count as a split-brain finding. Non-macro (design)
    modules are left intact."""
    out = list(text)
    for m in re.finditer(r"\bmodule\s+(\w+)\b.*?\bendmodule\b", text, re.DOTALL):
        if _is_macro_module(m.group(1)):
            for k in range(m.start(), m.end()):
                if out[k] != "\n":
                    out[k] = " "
    return "".join(out)


def _iter_procedural_blocks(text: str):
    """Yield (keyword, body_text) for each ``always``/``initial`` block. Body is
    the balanced ``begin..end`` region or the single statement up to ``;``."""
    for m in _ALWAYS_INITIAL_RE.finditer(text):
        kw = m.group(1)
        i, n = m.end(), len(text)
        while i < n and text[i].isspace():
            i += 1
        # always may carry an @(...) or @* sensitivity list
        if i < n and text[i] == "@":
            i += 1
            while i < n and text[i].isspace():
                i += 1
            if i < n and text[i] == "(":
                depth = 0
                while i < n:
                    if text[i] == "(":
                        depth += 1
                    elif text[i] == ")":
                        depth -= 1
                        if depth == 0:
                            i += 1
                            break
                    i += 1
            elif i < n and text[i] == "*":
                i += 1
            while i < n and text[i].isspace():
                i += 1
        if text[i:i + 5] == "begin" and (
            i + 5 >= n or not (text[i + 5].isalnum() or text[i + 5] == "_")
        ):
            depth = 0
            end = i
            for tok in re.finditer(r"\bbegin\b|\bend\b", text[i:]):
                if tok.group(0) == "begin":
                    depth += 1
                else:
                    depth -= 1
                    if depth == 0:
                        end = i + tok.end()
                        break
            yield kw, text[i:end]
        else:
            semi = text.find(";", i)
            yield kw, text[i:semi + 1 if semi != -1 else n]


def _is_functional_procedural_block(body: str) -> bool:
    """True if an always/initial block DRIVES logic (has an assignment) rather
    than being a pure debug/trace/assertion hook. Waveform + $display + SVA
    blocks leave no assignment behind and are ALLOWED."""
    b = _DEBUG_TASK_RE.sub(" ", body)
    b = _ASSERT_STMT_RE.sub(" ", b)
    # nonblocking assignment `lvalue <= ...`
    if re.search(r"[A-Za-z_)\]]\s*<=", b):
        return True
    # blocking assignment `=` (not ==, <=, >=, !=, ===, compound op=)
    if re.search(r"(?<![<>=!+\-*/%&|^~])=(?!=)", b):
        return True
    return False


def _functional_constructs(region_text: str, enclosing_module: str,
                           enclosing_is_macro: bool) -> list[str]:
    """Which FUNCTIONAL construct classes a conditional region contains
    (``assign`` / ``instantiation`` / ``always`` / ``initial``). Empty list =>
    the region is debug/assertion-only OR a legitimate macro-module split =>
    ALLOWED."""
    if enclosing_module and enclosing_is_macro:
        return []  # region lives inside a macro/library module -> allowed
    if enclosing_module:
        text = region_text  # inside a design module: no nested modules possible
    else:
        text = _mask_macro_modules(region_text)  # top-level: allow macro splits
    # Drop `module <name>` headers so a parametrized module DECLARATION isn't
    # misread as an instantiation. A non-macro whole-module split still trips on
    # its body's always/assign, which is the correct verdict.
    scan = re.sub(r"\bmodule\s+\w+", "  ", text)
    classes: set[str] = set()
    if _ASSIGN_KW_RE.search(scan):
        classes.add("assign")
    if _PARAM_INST_RE.search(scan):
        classes.add("instantiation")
    for kw, body in _iter_procedural_blocks(scan):
        if _is_functional_procedural_block(body):
            classes.add(kw)
    return sorted(classes)


@dataclass
class IfdefFinding:
    condition: str            # e.g. "ifndef SYNTHESIS", "else", "elsif FOO"
    macro: str                # the guard macro name ("" for `else)
    start_line: int
    end_line: int
    constructs: list[str]     # functional classes found (assign/always/...)
    enclosing_module: str = ""


@dataclass
class IfdefLintReport:
    findings: list[IfdefFinding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.findings


def find_functional_ifdef_regions(verilog_src: str,
                                  is_library: bool = False) -> IfdefLintReport:
    """Detect split-brain conditional-compilation regions: any ``ifdef`` /
    ``ifndef`` / ``elsif`` / ``else`` region that guards FUNCTIONAL logic inside
    a design module (or a non-macro whole-module split). ALLOWS debug/trace/
    assertion-only regions and the macro-module synth-blackbox/sim-behavioral
    library idiom. ``is_library=True`` (an ``rtl_lib`` wrapper file) exempts the
    whole source."""
    report = IfdefLintReport()
    if is_library:
        return report
    if "`" not in verilog_src or (
        "ifdef" not in verilog_src and "ifndef" not in verilog_src
    ):
        return report  # no conditional compilation at all -> nothing to check

    src = _blank_comments_strings(verilog_src)
    lines = src.split("\n")

    stack: list[dict] = []            # open conditional regions (innermost last)
    module_stack: list[tuple[str, bool]] = []  # (name, is_macro); modules don't nest

    def _close_top(end_line: int) -> None:
        reg = stack.pop()
        content = "\n".join(reg["content"])
        classes = _functional_constructs(content, reg["mod"], reg["mod_macro"])
        if classes:
            report.findings.append(IfdefFinding(
                condition=reg["cond"], macro=reg["macro"],
                start_line=reg["start"], end_line=end_line,
                constructs=classes, enclosing_module=reg["mod"],
            ))

    for idx, line in enumerate(lines, start=1):
        d = _DIRECTIVE_LINE_RE.match(line)
        if d:
            kind, macro = d.group(1), d.group(2) or ""
            if kind in ("ifdef", "ifndef"):
                mod = module_stack[-1][0] if module_stack else ""
                mac = module_stack[-1][1] if module_stack else False
                stack.append({
                    "cond": f"{kind} {macro}".strip(), "macro": macro,
                    "start": idx, "content": [], "mod": mod, "mod_macro": mac,
                })
            elif kind in ("elsif", "else"):
                if stack:
                    prev = stack[-1]
                    _close_top(idx)
                    # open the sibling branch, same enclosing-module context
                    stack.append({
                        "cond": f"{kind} {macro}".strip() if macro else kind,
                        "macro": macro, "start": idx, "content": [],
                        "mod": prev["mod"], "mod_macro": prev["mod_macro"],
                    })
            elif kind == "endif":
                if stack:
                    _close_top(idx)
            continue
        # content line: update module context + accumulate to innermost region
        for mt in _MODTOK_RE.finditer(line):
            if mt.group(1) is not None:  # `module <name>`
                module_stack.append((mt.group(1), _is_macro_module(mt.group(1))))
            else:                         # `endmodule`
                if module_stack:
                    module_stack.pop()
        if stack:
            stack[-1]["content"].append(line)

    return report


def format_ifdef_lint_report(report: IfdefLintReport, block: str = "") -> str:
    """Human/LLM-readable rejection message (or '' if clean)."""
    if report.ok:
        return ""
    head = (
        "SPLIT-BRAIN CONDITIONAL-COMPILATION RTL"
        + (f" in {block}" if block else "")
        + f" ({len(report.findings)} functional `ifdef/`ifndef region(s)):\n"
    )
    lines = [head]
    for f in report.findings:
        where = f" inside module {f.enclosing_module}" if f.enclosing_module else ""
        lines.append(
            f"  - `{f.condition}` region (lines {f.start_line}-{f.end_line}){where} "
            f"guards functional logic: {', '.join(f.constructs)}"
        )
    lines.append(
        "  A `ifdef/`ifndef that selects between TWO implementations of the same "
        "module is FORBIDDEN: the simulator (DV) verifies one branch while synth/"
        "backend build the OTHER -- they are DIFFERENT HARDWARE and DV proves "
        "nothing about the chip. FIX: write EXACTLY ONE implementation per "
        "module (no `ifdef around always/assign/instantiations). Conditional "
        "compilation may ONLY guard non-functional debug/trace/assertion code "
        "($display / $dumpvars / SVA assert). If simulation needs behavior that "
        "synthesis gets from a macro, that split lives ONLY inside the provided "
        "cs_* wrapper library (rtl_lib/cs_sram.v) -- never in your module."
    )
    return "\n".join(lines)


__all__ = [
    "DEFAULT_MAX_BITS",
    "DEFAULT_MAX_WORDS",
    "IfdefFinding",
    "IfdefLintReport",
    "MemoryTierFinding",
    "MemoryTierReport",
    "StorageFinding",
    "StorageLintReport",
    "find_flat_packed_dynamic_storage",
    "find_functional_ifdef_regions",
    "find_oversized_memory_arrays",
    "flat_read_mux_ns",
    "format_ifdef_lint_report",
    "format_lint_report",
    "format_memory_tier_report",
    "ifdef_lint_enabled",
]
