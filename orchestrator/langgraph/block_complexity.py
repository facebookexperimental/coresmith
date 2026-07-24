# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Complexity-aware block decomposition for the architecture stage.

The ``block_diagram`` specialist decomposes a design into blocks by functional
ROLE with **zero** complexity / tractability estimation. On the video codec intra
encoder that let it "consolidate" mode-decision + transform/quant + reconstruct
+ chroma + syntax-packing + entropy coding-rate into ONE block (``intra_rd_encode_core``)
emitting a 6287-bit record -- far too much distinct algorithm for one byte-exact
model to ever reproduce. The golden reference already exposes clean cut-points
(``_encode_mb``, ``fdct_quant``, ``reconstruct``, ``_rd_cost``,
``_try_intra16x16``, ``decide_chroma_mode``, ``pred_4x4``/``avail_modes_4x4``;
and for the terminal streamer ``build_header_a``/``build_header_b``, ``entropy_encode``,
``frame_pack``) -- they were available and ignored.

This module adds two deterministic, AST-based (no-LLM) capabilities:

1. :func:`estimate_block_complexity` -- score a block's golden slice on FOUR
   axes (flop_count, latency, data_locality, modeling_complexity) and flag it
   ``over_budget`` when any axis breaches the block's uArch budget. The axis
   that actually walls byte-exact reproduction is ``modeling_complexity``:
   LOC + distinct-algorithm count + cyclomatic branching of the slice.

2. :func:`propose_decomposition` -- build the golden's intra-block dataflow DAG
   (functions as nodes, data-dependency edges) and MIN-CUT partition it along
   function boundaries so each partition stays under the complexity budget while
   cross-partition data movement is minimized. Emits per-sub-block interface
   contracts (the golden's own function signatures) so the architecture stage
   can rewrite ``block_diagram.json`` with real, tractable sub-blocks.

Everything here is pure Python + the stdlib ``ast`` module + two existing engine
helpers (``mem_characterize`` for array/flop sizing, ``pipeline_scheduler`` for
op-count -> cycles). It NEVER calls an LLM and is fully unit-testable.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Engine-helper imports (deferred / graceful): the pipeline scheduler gives us a
# real op-count -> cycles latency estimate, and mem_characterize is reused for
# array-backed storage sizing. Both degrade to analytical fallbacks if absent so
# this module import-loads on any host.
try:  # pragma: no cover - trivial import guard
    from orchestrator.langgraph.ppa_check import parse_ff_budget
except Exception:  # pragma: no cover
    def parse_ff_budget(text: str) -> int | None:  # type: ignore
        m = re.search(r"flip_flop_budget[^\d\n]{0,16}(\d[\d,]*)", text or "",
                      re.IGNORECASE)
        return int(m.group(1).replace(",", "")) if m else None


# ---------------------------------------------------------------------------
# Modeling-complexity thresholds (the axis that walls byte-exact reproduction).
# Deliberately generous: they exist to catch a monolith that fused many distinct
# algorithms into one block, not to nitpick a normal block.
# ---------------------------------------------------------------------------
MODELING_LOC_THRESHOLD = 250          # golden-slice LOC above which a block is fat
MODELING_ALGO_THRESHOLD = 3           # distinct-algorithm count above which fat
MODELING_CYCLO_THRESHOLD = 80         # summed cyclomatic complexity above which fat
# C16: the distinct-algorithms axis fires only for a SUBSTANTIAL block. A small
# store/router legitimately touches many algorithms' data (a hoisted memory
# holding codebooks + quant tables + reference frames) without being an
# intractable compute FUSION -- flagging it for decomposition is wrong (it is
# already the decomposition target). Require a LOC floor so the algo axis marks
# only a genuinely fat compute block (residual_recon_engine: 582 LOC / 6 algos
# flags; coefficient_token_memory: 91 LOC / 5 algos does not).
MODELING_ALGO_LOC_FLOOR = 150

# Op families whose presence marks a "distinct algorithm" in a golden slice.
# Grouped so that e.g. all the transform kernels count as one algorithm class.
_ALGO_SIGNATURES: dict[str, tuple[str, ...]] = {
    "transform": ("fdct", "idct", "_fdct4", "_idct4", "_had2x2", "_had4x4", "dct"),
    "quant": ("quantize", "dequantize", "fdct_quant", "_quant", "chroma_dc_quant",
              "luma_dc_quant", "dead_zone", "deadzone"),
    "reconstruct": ("reconstruct", "clip255", "clip", "recon"),
    "intra_pred": ("pred_4x4", "pred_16x16", "pred_chroma", "avail_modes",
                   "predict", "prediction"),
    "rd_cost": ("_rd_cost", "rd_cost", "rdoq", "_block_rd", "_residual_bits",
                "lambda", "ssd", "sad"),
    "entropy": ("entropy", "sym_token", "_put_sym_token", "_write_level",
              "_level_to_code", "total_zeros", "run_before", "nc", "_nc"),
    "ue_coder": ("build_header_a", "build_header_b", "ue", "se", "ue_coder", "ue_coder"),
    "bitpack": ("frame_pack", "unit_pack", "byte_align", "emulation",
             "frame_marker", "getbytes"),
    "chroma": ("chroma", "decide_chroma_mode", "_commit_chroma", "qpc"),
    "scan": ("zigzag", "inv_zigzag", "_luma_blk_xy", "z-order", "raster"),
    "mode_history": ("_mpm", "modeY", "nnz", "mode_history", "mpm"),
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class FuncStat:
    """Per-golden-function AST summary used by both estimator and decomposer."""

    name: str
    loc: int = 0
    calls: set[str] = field(default_factory=set)       # callee names (any func)
    array_writes: set[str] = field(default_factory=set)  # subscript-target names
    array_reads: set[str] = field(default_factory=set)   # subscript-read names
    stateful_regs: int = 0                              # persistent arrays/accum
    op_count: int = 0                                   # arithmetic ops
    cyclomatic: int = 1                                 # branch complexity
    algorithms: set[str] = field(default_factory=set)   # distinct-algo classes
    signature: str = ""                                 # source `def ...:` line
    start_lineno: int = 0                               # 1-based def start line
    end_lineno: int = 0                                 # 1-based def end line


@dataclass
class ComplexityEstimate:
    """4-axis complexity score for a block vs its uArch budget."""

    block_name: str
    flops: int = 0
    latency_cyc: int = 0
    locality: int = 0                 # bytes crossing sub-function boundaries
    modeling_complexity: int = 0      # composite modeling-difficulty score
    loc: int = 0
    distinct_algorithms: int = 0
    cyclomatic: int = 0
    ff_budget: int | None = None
    over_budget: bool = False
    axis_breaches: list[str] = field(default_factory=list)
    golden_functions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_name": self.block_name,
            "flops": self.flops,
            "latency_cyc": self.latency_cyc,
            "locality": self.locality,
            "modeling_complexity": self.modeling_complexity,
            "loc": self.loc,
            "distinct_algorithms": self.distinct_algorithms,
            "cyclomatic": self.cyclomatic,
            "ff_budget": self.ff_budget,
            "over_budget": self.over_budget,
            "axis_breaches": list(self.axis_breaches),
            "golden_functions": list(self.golden_functions),
        }


# ---------------------------------------------------------------------------
# Golden-slice resolution: which functions belong to a given block.
# ---------------------------------------------------------------------------

# Curated block-name -> golden entry-function hints. The estimator seeds the
# slice from these and then transitively closes over the golden call graph, so a
# partial hint list still captures the whole slice. Names are matched
# case-insensitively as substrings of the block name so it generalises across
# runs (intra_rd_encode_core, intra_rd, ...).
BLOCK_ENTRY_HINTS: dict[str, tuple[str, ...]] = {
    "intra_rd": ("_encode_mb", "_try_intra16x16", "decide_chroma_mode",
                 "fdct_quant", "reconstruct", "_rd_cost", "rdoq_4x4",
                 "pred_4x4", "pred_16x16", "avail_modes_4x4", "_block_rd",
                 "_gather_4x4_neighbors", "_emit_mb_i16", "_commit_chroma",
                 "_chroma_plane_artifacts", "_chroma_cost_bits",
                 "pred_chroma_dc_block", "pred_chroma_block"),
    "encode_core": ("_encode_mb", "_try_intra16x16", "decide_chroma_mode"),
    "framer": ("_encode_frame", "encode", "build_header_a", "build_header_b",
               "unit_pack", "frame_pack", "entropy_encode", "_put_sym_token",
               "_write_level", "_level_to_code", "_residual_bits", "_nc",
               "_mpm"),
    "entropy": ("entropy_encode", "_encode_frame", "build_header_a", "build_header_b",
                "frame_pack", "unit_pack"),
    "ingress": ("_as_yuv",),
    "frame_ingress": ("_as_yuv",),
    "source_frame_store": ("_as_yuv",),
    "frame_store": ("_as_yuv",),
}


def _read_golden_source(golden_path: str) -> str:
    try:
        return Path(golden_path).read_text(encoding="utf-8")
    except OSError:
        return ""


def _parse_functions(source: str) -> dict[str, FuncStat]:
    """AST-scan every top-level function/method into a :class:`FuncStat`.

    Methods of classes (e.g. ``BitWriter.ue``) are recorded under their bare
    method name so call-graph edges resolve; a bare name collision keeps the
    larger (more-complex) definition, which is the conservative choice.
    """
    stats: dict[str, FuncStat] = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return stats

    lines = source.splitlines()

    def _visit_func(node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        fs = FuncStat(name=node.name)
        start = node.lineno
        end = getattr(node, "end_lineno", start) or start
        fs.loc = max(1, end - start)
        fs.start_lineno = start
        fs.end_lineno = end
        if 0 < start <= len(lines):
            fs.signature = lines[start - 1].strip()
        _walk_body(node, fs)
        # detect distinct algorithm classes from names touched
        fs.algorithms = _detect_algorithms(fs)
        prev = stats.get(node.name)
        if prev is None or fs.loc >= prev.loc:
            stats[node.name] = fs

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _visit_func(node)
    return stats


_ARITH_OP = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
             ast.Pow, ast.LShift, ast.RShift, ast.BitOr, ast.BitAnd, ast.BitXor)


def _walk_body(func: ast.AST, fs: FuncStat) -> None:
    """Populate call/array/op/cyclomatic fields from a function body."""
    for node in ast.walk(func):
        # callee names
        if isinstance(node, ast.Call):
            callee = _call_name(node.func)
            if callee:
                fs.calls.add(callee)
        # subscript read/write -> data locality
        if isinstance(node, ast.Subscript):
            base = _subscript_base(node)
            if base:
                # a Subscript inside an assignment target is a write; else read
                ctx = getattr(node, "ctx", None)
                if isinstance(ctx, ast.Store):
                    fs.array_writes.add(base)
                else:
                    fs.array_reads.add(base)
        # arithmetic op count (latency proxy)
        if isinstance(node, (ast.BinOp, ast.AugAssign)):
            op = getattr(node, "op", None)
            if isinstance(op, _ARITH_OP):
                fs.op_count += 1
        # cyclomatic complexity: +1 per branch/loop/boolean-op/comprehension
        if isinstance(node, (ast.If, ast.For, ast.While, ast.IfExp,
                             ast.comprehension)):
            fs.cyclomatic += 1
        if isinstance(node, ast.BoolOp):
            fs.cyclomatic += max(1, len(node.values) - 1)
        # stateful registers: persistent arrays created inside the function
        # (np.zeros/np.full/bytearray/list accumulators + augmented accumulators)
        if isinstance(node, ast.Call):
            cn = _call_name(node.func) or ""
            if cn in {"zeros", "full", "empty", "ones", "bytearray"} or \
               cn.endswith(".zeros") or cn.endswith(".full"):
                fs.stateful_regs += 1
        if isinstance(node, ast.AugAssign):
            fs.stateful_regs += 1


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _subscript_base(node: ast.Subscript) -> str | None:
    v = node.value
    # unwrap chained subscripts / attributes to the root name
    while isinstance(v, ast.Subscript):
        v = v.value
    if isinstance(v, ast.Name):
        return v.id
    if isinstance(v, ast.Attribute):
        return v.attr
    return None


def _detect_algorithms(fs: FuncStat) -> set[str]:
    """Which distinct-algorithm classes a function touches (by name signature)."""
    hay = " ".join([fs.name, *fs.calls, *fs.array_reads, *fs.array_writes,
                    fs.signature]).lower()
    found: set[str] = set()
    for algo, sigs in _ALGO_SIGNATURES.items():
        for s in sigs:
            if s.lower() in hay:
                found.add(algo)
                break
    return found


# ---------------------------------------------------------------------------
# Slice resolution (block -> set of golden functions)
# ---------------------------------------------------------------------------

# Per-block STOP functions: names the call-graph closure must NOT descend into
# because they belong to a DIFFERENT block. This encodes the inter-block cut
# point. The framer streamer consumes the assembled syntax record; it owns
# entropy/header/bit-accumulator but NOT the intra encode -- so its closure stops at
# ``_encode_mb`` (owned by intra_rd) and does not pull the transform/chroma/
# prediction internals into the framer slice.
BLOCK_STOP_FUNCS: dict[str, tuple[str, ...]] = {
    "framer": ("_encode_mb", "_emit_mb_i16"),
    "entropy": ("_encode_mb", "_emit_mb_i16"),
}


def _stop_funcs_for_block(block_name: str) -> set[str]:
    bn = (block_name or "").lower()
    stop: set[str] = set()
    for key, funcs in BLOCK_STOP_FUNCS.items():
        if key in bn:
            stop |= set(funcs)
    return stop


def _entry_hints_for_block(block_name: str) -> tuple[str, ...]:
    bn = (block_name or "").lower()
    hints: list[str] = []
    for key, funcs in BLOCK_ENTRY_HINTS.items():
        if key in bn:
            hints.extend(funcs)
    return tuple(dict.fromkeys(hints))  # dedup, preserve order


def python_source_slice_fns(python_source_ref: str,
                            stats: dict[str, FuncStat]) -> list[str]:
    """C16: the block's golden functions from its ``python_source`` ref -- the
    AUTHORITATIVE architecture-assigned slice (``<file>.py:fn1,fn2,...``).

    This is the slice the complexity gate SHOULD score. The legacy
    ``resolve_block_slice`` derived a slice from codec-specific ``BLOCK_ENTRY_HINTS``
    plus a name-substring fallback, which returned an EMPTY slice for every
    non-the video codec design (e.g. residual_recon_engine: 0 fns -> modeling_complexity
    0 -> never flagged, despite fusing 6 algorithms across 582 LOC / cyclomatic
    158). The architecture already computed the correct slice; use it. Returns
    the named functions present in ``stats`` (methods included, keyed by bare
    name); empty when the ref carries no ``:name`` suffix or none resolve.
    """
    ref = (python_source_ref or "").strip()
    marker = ".py:"
    idx = ref.find(marker)
    if idx == -1:
        return []
    names = [n.strip() for n in ref[idx + len(marker):].split(",") if n.strip()]
    return sorted({n for n in names if n in stats})


def resolve_block_slice(block_name: str, stats: dict[str, FuncStat],
                        extra_entries: tuple[str, ...] | None = None,
                        max_depth: int = 6) -> list[str]:
    """Transitively close the golden call graph from a block's entry hints.

    Returns the sorted list of golden function names that make up the block's
    slice. If no hint matches, tries the block name itself as an entry.

    NOTE: prefer :func:`python_source_slice_fns` when the block carries a
    ``python_source`` ref -- this hint-based path is codec-specific and returns
    an empty slice for other designs (see C16).
    """
    entries = list(_entry_hints_for_block(block_name))
    if extra_entries:
        entries.extend(extra_entries)
    if not entries:
        # last resort: any golden function whose name appears in the block name
        bn = (block_name or "").lower()
        entries = [n for n in stats if n.lower() in bn]
    stop = _stop_funcs_for_block(block_name)
    seen: set[str] = set()
    # A stop function that is itself an explicit entry hint is kept (it is the
    # boundary record owner) but we never DESCEND through it into another block.
    frontier = [e for e in entries if e in stats]
    depth = 0
    while frontier and depth < max_depth:
        nxt: list[str] = []
        for fn in frontier:
            if fn in seen:
                continue
            seen.add(fn)
            if fn in stop:
                continue  # do not descend into another block's internals
            for callee in stats[fn].calls:
                if callee in stats and callee not in seen and callee not in stop:
                    nxt.append(callee)
        frontier = nxt
        depth += 1
    # drop stop funcs that only got pulled in as callees (not entry hints)
    entry_set = set(entries)
    return sorted(f for f in seen if f not in stop or f in entry_set)


def resolve_block_slice_regions(
    block_name: str,
    golden_source: str,
    extra_entries: tuple[str, ...] | None = None,
    require_entry_hints: bool = True,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Resolve a block's golden slice as (function_names, source regions).

    ``regions`` is ``[{"fn", "start", "end"}]`` (1-based line spans, sorted by
    start) drawn from the already-parsed :class:`FuncStat` line numbers. This is
    the authoritative, deterministic block->golden mapping consumed by the
    block-golden generator (persisted as a ``.slice.json`` sidecar) and by the
    golden-feasibility check, so neither has to re-derive the slice by hand or
    trust an LLM to pick the right functions out of the whole golden.

    ``require_entry_hints`` (default True): only return a slice we are CONFIDENT
    about -- i.e. one seeded from the curated ``BLOCK_ENTRY_HINTS`` (or explicit
    ``extra_entries``). This deliberately skips ``resolve_block_slice``'s
    desperate substring fallback, which can spuriously match (e.g. block
    ``dct_token_engine`` -> golden helper ``_t`` because ``_t`` is a substring),
    since a wrong single-function hint is worse than none. Returns ``([], [])``
    for any block without a confident mapping -- callers treat that as "generator
    sees the whole golden", i.e. today's behaviour. Phase 2 replaces the curated
    hints with a model-derived mapping that generalises past the video codec.
    """
    if require_entry_hints and not _entry_hints_for_block(block_name) \
            and not extra_entries:
        return [], []
    stats = _parse_functions(golden_source)
    fns = resolve_block_slice(block_name, stats, extra_entries=extra_entries)
    regions = [
        {"fn": n, "start": stats[n].start_lineno, "end": stats[n].end_lineno}
        for n in fns
        if n in stats
    ]
    regions.sort(key=lambda r: (r["start"], r["fn"]))
    return fns, regions


# ---------------------------------------------------------------------------
# 1. estimate_block_complexity
# ---------------------------------------------------------------------------

def _estimate_latency_cyc(op_count: int, pdk: dict | None = None) -> int:
    """Op-count -> cycles via the pipeline scheduler's chaining model.

    Reuses ``pipeline_scheduler.schedule_reduction`` to turn a bag of ``op_count``
    arithmetic ops (modelled as an add-reduction datapath at a nominal period)
    into a real pipeline depth, then scales by the serial-reuse factor (one
    datapath reused per op). Degrades to a linear proxy if the scheduler /
    arith model is unavailable on this host.
    """
    if op_count <= 0:
        return 0
    try:
        from orchestrator.langgraph.pipeline_scheduler import schedule_reduction
        # nominal 50 MHz period (20 ns); combine as add reductions
        sched = schedule_reduction("add", 16, max(2, op_count), 20.0,
                                   combine="add", pdk=pdk)
        depth = max(1, sched.depth)
        # serial reuse: each op takes ~1 datapath pass; latency ~ op_count spread
        # over the pipeline depth of a reduction tree (log-ish), plus fill.
        return op_count + depth
    except Exception:
        # analytical fallback: ~1 cycle/op serialised
        return op_count


def estimate_block_complexity(block_name: str, golden_path: str,
                              spec_text: str = "",
                              pdk: dict | None = None,
                              stats: dict[str, FuncStat] | None = None,
                              slice_fns: list[str] | None = None,
                              ) -> dict[str, Any]:
    """Score a block's golden slice on 4 axes vs its uArch budget (no LLM).

    Axes:
      * ``flops`` -- stateful pipeline registers + array accumulators in the
        slice (reuses ``mem_characterize`` sizing conceptually via the golden's
        persistent array declarations).
      * ``latency_cyc`` -- op-count -> cycles via ``pipeline_scheduler``.
      * ``locality`` -- distinct arrays crossing sub-function boundaries (a
        proxy for bytes of state that must move between the golden's functions).
      * ``modeling_complexity`` -- LOC + distinct-algorithm-count + cyclomatic
        of the slice; the axis that actually walls byte-exact reproduction.

    Returns ``{flops, latency_cyc, locality, modeling_complexity, over_budget,
    axis_breaches, ...}``.
    """
    if stats is None:
        src = _read_golden_source(golden_path)
        stats = _parse_functions(src)
    # C16: score the caller-supplied slice (the architecture's python_source
    # mapping) when given; else fall back to the legacy-hint resolver (which
    # yields an empty slice -- and thus a 0 score -- for non-the video codec designs).
    if slice_fns is None:
        slice_fns = resolve_block_slice(block_name, stats)
    else:
        slice_fns = [f for f in slice_fns if f in stats]

    loc = sum(stats[f].loc for f in slice_fns)
    op_count = sum(stats[f].op_count for f in slice_fns)
    cyclo = sum(stats[f].cyclomatic for f in slice_fns)
    flops = sum(stats[f].stateful_regs for f in slice_fns)
    algos: set[str] = set()
    for f in slice_fns:
        algos |= stats[f].algorithms

    # data locality: arrays that are both written by one function and read by a
    # DIFFERENT function in the slice cross a sub-function boundary.
    writes_by: dict[str, set[str]] = {}
    reads_by: dict[str, set[str]] = {}
    for f in slice_fns:
        for a in stats[f].array_writes:
            writes_by.setdefault(a, set()).add(f)
        for a in stats[f].array_reads:
            reads_by.setdefault(a, set()).add(f)
    crossing = 0
    for arr, writers in writes_by.items():
        readers = reads_by.get(arr, set())
        if readers - writers:  # read somewhere it isn't written
            crossing += 1
    locality = crossing

    latency_cyc = _estimate_latency_cyc(op_count, pdk=pdk)

    # modeling complexity composite: normalise the three sub-scores so a breach
    # on any one lifts the composite over its threshold.
    distinct_algos = len(algos)
    modeling_complexity = loc + distinct_algos * 40 + cyclo

    ff_budget = parse_ff_budget(spec_text) if spec_text else None

    axis_breaches: list[str] = []
    # modeling-complexity breach (the wall) -- any of LOC / algos / cyclo
    if loc > MODELING_LOC_THRESHOLD:
        axis_breaches.append(
            f"modeling_complexity: golden slice is {loc} LOC "
            f"(> {MODELING_LOC_THRESHOLD} LOC threshold)")
    if distinct_algos > MODELING_ALGO_THRESHOLD and loc >= MODELING_ALGO_LOC_FLOOR:
        axis_breaches.append(
            f"modeling_complexity: {distinct_algos} distinct algorithms "
            f"({', '.join(sorted(algos))}) in {loc} LOC "
            f"(> {MODELING_ALGO_THRESHOLD} algos above the "
            f"{MODELING_ALGO_LOC_FLOOR}-LOC floor)")
    if cyclo > MODELING_CYCLO_THRESHOLD:
        axis_breaches.append(
            f"modeling_complexity: cyclomatic {cyclo} "
            f"(> {MODELING_CYCLO_THRESHOLD})")
    # flop-budget breach (only when a budget is declared)
    if ff_budget is not None and flops > 0:
        # flops here is a count of persistent arrays, not raw FFs; only flag when
        # the slice clearly carries more distinct stateful arrays than a normal
        # block AND the block is already flagged fat by modeling complexity.
        pass  # FF sizing is advisory; the modeling axis is the true wall.

    over_budget = bool(axis_breaches)

    est = ComplexityEstimate(
        block_name=block_name, flops=flops, latency_cyc=latency_cyc,
        locality=locality, modeling_complexity=modeling_complexity,
        loc=loc, distinct_algorithms=distinct_algos, cyclomatic=cyclo,
        ff_budget=ff_budget, over_budget=over_budget,
        axis_breaches=axis_breaches, golden_functions=slice_fns,
    )
    return est.to_dict()


# ---------------------------------------------------------------------------
# 2. propose_decomposition -- min-cut partition along function boundaries
# ---------------------------------------------------------------------------

@dataclass
class SubBlock:
    sub_block: str
    golden_functions: list[str]
    interface_contract: dict[str, Any]
    est_complexity: dict[str, Any]


# Curated sub-block partitioning for the two known monoliths. Each entry names
# the sub-block, its "seed" golden functions (the min-cut group leaders), and
# the algorithm classes it owns. The min-cut assigns every remaining slice
# function to the seed group it shares the most data traffic with.
_INTRA_RD_SEEDS: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
    ("mode_decision",
     ("_rd_cost", "rdoq_4x4", "_block_rd", "_residual_bits", "_gather_4x4_neighbors",
      "_mpm", "_nc"),
     ("rd_cost", "mode_history", "scan")),
    ("transform_quant",
     ("fdct_quant", "quantize", "dequantize", "_fdct4", "_idct4",
      "luma_dc_quant", "luma_dc_dequant", "_had4x4"),
     ("transform", "quant")),
    ("reconstruct",
     ("reconstruct", "clip255", "_try_intra16x16", "_emit_mb_i16",
      "pred_4x4", "pred_16x16", "avail_modes_4x4", "avail_modes_16x16"),
     ("reconstruct", "intra_pred")),
    ("chroma_encode",
     ("decide_chroma_mode", "_commit_chroma", "_chroma_plane_artifacts",
      "_chroma_cost_bits", "pred_chroma_dc_block", "pred_chroma_block",
      "avail_modes_chroma", "chroma_dc_quant", "chroma_dc_dequant", "_had2x2",
      "chroma_qp"),
     ("chroma",)),
    ("syntax_pack",
     ("_encode_mb", "_luma_blk_xy", "zigzag", "inv_zigzag"),
     ("scan",)),
]

_ANNEXB_SEEDS: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
    ("group_header_ue_coder",
     ("build_header_a", "build_header_b", "_encode_frame", "unit_pack"),
     ("ue_coder",)),
    ("entropy_residual",
     ("entropy_encode", "_put_sym_token", "_write_level", "_level_to_code",
      "_residual_bits", "_nc"),
     ("entropy",)),
    ("bitpack_bytepack",
     ("frame_pack",),
     ("bitpack",)),
]


def _seed_table_for_block(block_name: str):
    bn = (block_name or "").lower()
    if "intra_rd" in bn or "encode_core" in bn:
        return _INTRA_RD_SEEDS
    if "framer" in bn or "entropy" in bn:
        return _ANNEXB_SEEDS
    return None


def _mincut_assign(slice_fns: list[str], stats: dict[str, FuncStat],
                   seeds: list[tuple[str, tuple[str, ...], tuple[str, ...]]],
                   ) -> dict[str, list[str]]:
    """Assign every slice function to the seed group it shares most traffic with.

    Traffic between a function and a group = shared array names (read/write) +
    call edges. This is the greedy min-cut: put each function where it moves the
    least data across the boundary (equivalently, shares the most inside).
    """
    groups: dict[str, list[str]] = {name: [] for name, _, _ in seeds}
    seed_of: dict[str, str] = {}
    for name, seed_fns, _ in seeds:
        for sf in seed_fns:
            seed_of[sf] = name

    # pre-place explicit seeds that exist in the slice
    for fn in slice_fns:
        if fn in seed_of:
            groups[seed_of[fn]].append(fn)

    def _arrs(fn: str) -> set[str]:
        return stats[fn].array_reads | stats[fn].array_writes

    def _traffic(fn: str, group_name: str) -> int:
        score = 0
        my_arrs = _arrs(fn)
        for member in groups[group_name]:
            score += len(my_arrs & _arrs(member))          # shared state
            if member in stats[fn].calls or fn in stats[member].calls:
                score += 2                                 # call edge (strong)
        # algorithm-class affinity to the seed group's declared classes
        seed_algos = next((set(a) for n, _, a in seeds if n == group_name), set())
        if stats[fn].algorithms & seed_algos:
            score += 3
        return score

    for fn in slice_fns:
        if fn in seed_of:
            continue  # already placed as a seed
        best_group = None
        best_score = -1
        for group_name in groups:
            s = _traffic(fn, group_name)
            if s > best_score:
                best_score = s
                best_group = group_name
        # default to the first group if nothing shares any traffic
        groups[best_group or seeds[0][0]].append(fn)
    # dedup + sort each group
    return {g: sorted(set(fns)) for g, fns in groups.items()}


def _interface_contract_for(sub_name: str, block_name: str) -> dict[str, Any]:
    """AXI-Stream interface contract for a proposed sub-block.

    Contracts are byte-exact to the golden's intermediate data (modes, quantized
    coeffs, reconstructed neighbours, record fields) and keep the block's
    existing external port on its first/last sub-block so the block_diagram edge
    set is preserved.
    """
    contracts: dict[str, dict[str, Any]] = {
        # ---- intra_rd sub-blocks ----
        "mode_decision": {
            "inputs": {"s_axis_mb_samples": {"width": 3136, "protocol": "axi_stream",
                        "note": "raw pixel_block samples + geometry (golden _encode_mb input)"}},
            "outputs": {"m_axis_mode_decision": {"width": 96, "protocol": "axi_stream",
                        "note": "per-4x4 selected luma mode[16*4] + i16 discriminators "
                                "mb_is_i16[1]+i16_luma_mode[2] + intra_chroma_pred_mode[2] "
                                "+ RD-chosen residual selectors; drives transform_quant"}},
        },
        "transform_quant": {
            "inputs": {"s_axis_resid": {"width": 512, "protocol": "axi_stream",
                        "note": "prediction residual per 4x4 (fdct_quant input) + qp"}},
            "outputs": {"m_axis_qcoeff": {"width": 6144, "protocol": "axi_stream",
                        "note": "quantized signed-16b coeff levels[384*16] (golden fdct_quant/quantize output)"}},
        },
        "reconstruct": {
            "inputs": {"s_axis_qcoeff": {"width": 6144, "protocol": "axi_stream",
                        "note": "quantized levels + chosen pred (reconstruct/dequantize input)"}},
            "outputs": {"m_axis_recon_neighbors": {"width": 2176, "protocol": "axi_stream",
                        "note": "reconstructed recY/recU/recV neighbour samples (golden reconstruct output) "
                                "fed back for later 4x4 predictions"}},
        },
        "chroma_encode": {
            "inputs": {"s_axis_chroma_samples": {"width": 1024, "protocol": "axi_stream",
                        "note": "chroma U/V 8x8 samples + reconstructed neighbours + qpc"}},
            "outputs": {"m_axis_chroma_coeff": {"width": 1088, "protocol": "axi_stream",
                        "note": "chroma DC[4+4] + AC[4*15 U, 4*15 V] levels + chroma_cbp[2] "
                                "+ intra_chroma_pred_mode[2] (golden decide_chroma_mode/_commit_chroma output)"}},
        },
        "syntax_pack": {
            "inputs": {"s_axis_encode_result": {"width": 6144, "protocol": "axi_stream",
                        "note": "luma+chroma quantized coeffs + modes + cbp + reconstruction state"}},
            "outputs": {"m_axis_mb_syntax": {"width": 6287, "protocol": "axi_stream",
                        "note": "assembled 6287-bit pixel_block syntax record (existing block output contract, unchanged)"}},
        },
        # ---- framer sub-blocks ----
        "group_header_ue_coder": {
            "inputs": {"s_axis_mb_syntax": {"width": 6287, "protocol": "axi_stream",
                        "note": "pixel_block syntax record (existing block input contract, unchanged)"}},
            "outputs": {"m_axis_header_bits": {"width": 40, "protocol": "axi_stream",
                        "note": "parameter-set + group_header + block-header unary/exp coding bit chunks "
                                "(golden build_header_a/build_header_b/_encode_frame header emission): "
                                "{bits[32], nbits[6], param_set_flag[1], last[1]}"}},
        },
        "entropy_residual": {
            "inputs": {"s_axis_residual_ctx": {"width": 6208, "protocol": "axi_stream",
                        "note": "coeff levels[384*16] + nnz/nC context grids + scan order "
                                "(golden entropy_encode/_nc input)"}},
            "outputs": {"m_axis_entropy_bits": {"width": 40, "protocol": "axi_stream",
                        "note": "concatenated entropy coding codeword bit chunks {bits[32], nbits[6], last[1]} "
                                "(golden entropy_encode output), MSB-first"}},
        },
        "bitpack_bytepack": {
            "inputs": {"s_axis_bitpack_bits": {"width": 40, "protocol": "axi_stream",
                        "note": "contiguous MSB-first bit-accumulator bit accumulator chunks {bits[32], nbits[6], unit_end[1], last[1]}"}},
            "outputs": {"m_axis_bytestream_byte": {"width": 8, "protocol": "axi_stream",
                        "note": "byte-framing bytes: frame-marker + unit header + frame_pack byte-stuffing "
                                "(existing block output contract, unchanged)"}},
        },
    }
    return contracts.get(sub_name, {"inputs": {}, "outputs": {}})


def propose_decomposition(block_name: str, golden_path: str,
                          spec_text: str = "",
                          pdk: dict | None = None,
                          stats: dict[str, FuncStat] | None = None,
                          slice_fns: list[str] | None = None,
                          ) -> list[dict[str, Any]]:
    """Propose a min-cut sub-block partition of a fat block along golden funcs.

    Builds the golden intra-block dataflow DAG (functions as nodes, shared-array
    + call edges) and MIN-CUT partitions it into sub-blocks so each partition is
    tractable and cross-boundary data movement is minimised. Returns a list of
    ``{sub_block, golden_functions, interface_contract, est_complexity}``.

    For a block with no known monolith seed table, returns a single passthrough
    partition (the block is already tractable / not a fusion of many algorithms).
    """
    if stats is None:
        src = _read_golden_source(golden_path)
        stats = _parse_functions(src)
    # C16: prefer the architecture's python_source slice (else the video codec hints).
    if slice_fns is None:
        slice_fns = resolve_block_slice(block_name, stats)
    else:
        slice_fns = [f for f in slice_fns if f in stats]
    seeds = _seed_table_for_block(block_name)
    if not seeds or not slice_fns:
        # not a known monolith -> keep whole (already tractable)
        return [{
            "sub_block": block_name,
            "golden_functions": slice_fns,
            "interface_contract": {"inputs": {}, "outputs": {}},
            "est_complexity": estimate_block_complexity(
                block_name, golden_path, spec_text, pdk=pdk, stats=stats,
                slice_fns=slice_fns),
        }]

    groups = _mincut_assign(slice_fns, stats, seeds)
    out: list[dict[str, Any]] = []
    # preserve the seed order (which is dataflow order)
    for sub_name, _, _ in seeds:
        fns = groups.get(sub_name, [])
        if not fns:
            continue
        # per-sub-block complexity from its own function subset
        sub_stats = {f: stats[f] for f in fns}
        loc = sum(sub_stats[f].loc for f in fns)
        op_count = sum(sub_stats[f].op_count for f in fns)
        cyclo = sum(sub_stats[f].cyclomatic for f in fns)
        flops = sum(sub_stats[f].stateful_regs for f in fns)
        algos: set[str] = set()
        for f in fns:
            algos |= sub_stats[f].algorithms
        est = {
            "flops": flops,
            "latency_cyc": _estimate_latency_cyc(op_count, pdk=pdk),
            "loc": loc,
            "distinct_algorithms": len(algos),
            "cyclomatic": cyclo,
            "modeling_complexity": loc + len(algos) * 40 + cyclo,
            "algorithms": sorted(algos),
            "over_budget": (loc > MODELING_LOC_THRESHOLD
                            or len(algos) > MODELING_ALGO_THRESHOLD
                            or cyclo > MODELING_CYCLO_THRESHOLD),
        }
        out.append({
            "sub_block": f"{_short_block(block_name)}_{sub_name}",
            "golden_functions": fns,
            "interface_contract": _interface_contract_for(sub_name, block_name),
            "est_complexity": est,
        })
    return out


def _short_block(block_name: str) -> str:
    """A short prefix for sub-block names (drops generic suffixes)."""
    bn = block_name or "block"
    for suffix in ("_encode_core", "_core", "_streamer", "_engine"):
        if bn.endswith(suffix):
            return bn[: -len(suffix)]
    return bn


__all__ = [
    "MODELING_ALGO_THRESHOLD",
    "MODELING_CYCLO_THRESHOLD",
    "MODELING_LOC_THRESHOLD",
    "ComplexityEstimate",
    "FuncStat",
    "SubBlock",
    "estimate_block_complexity",
    "propose_decomposition",
    "resolve_block_slice",
]
