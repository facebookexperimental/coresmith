# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Performance roofline -- give every block a THROUGHPUT floor, not just Fmax.

The Fmax step already prices each primitive op combinationally (``add``,
``mul``, ``shift`` ... via ``arith_characterize.predict_op_delay``) so it can
tell whether a datapath closes timing at the target clock. That says nothing
about HOW MANY CYCLES the block spends per unit of work -- so a design that
runs a fixed-N loop on one reusable datapath (sequential MAC, word-serial key
expansion, per-pixel serial edge test) sails through Fmax while being 4-6x
slower than the same silicon pipelined. Nothing in the pipeline measured that
gap, so nothing rejected it.

This module supplies the missing axis: from a block's DATAFLOW SPEC -- ops per
iteration by resource class, loop-carried recurrences (ops + latency +
distance), resource instance counts, iteration count, io framing -- plus the
target clock, it computes the classic modulo-scheduling roofline:

    D        = pipeline depth (pack the op chain so each stage's comb delay
               <= T - margin);   Fmax = 1/T
    RecMII   = max_c  ceil( latency(cycle c) / distance(c) )       recurrence bound
    ResMII   = max_r  ceil( uses_per_iter(r) / instances(r) )      resource bound
    II_min   = max(RecMII, ResMII)
    cyc_per_op_peak = iterations * II_min + (D-1) fill + drain + io_framing
    binding constraint = which recurrence or resource sets the floor

The op delays come from the SAME characterizer the Fmax step uses
(``predict_op_delay``), so the throughput floor is PDK-consistent with the
timing floor. The dataflow spec is DECLARED by the uArch-spec author in a
machine-readable ``perf`` block (see ``parse_perf_spec``) and checked
DETERMINISTICALLY here -- no prose parsing, no LLM. The emitted
``perf_model.json`` carries the peak cyc/op, the binding constraint, and -- when
the spec cross-references an FRD ``PERF-NNN`` requirement -- a pass/fail against
it (with a self-imposed budget = peak x derate when the customer imposed no
hard cap).

Task-agnostic: no design is baked in. Gated by ``CORESMITH_PERF_ROOFLINE``
(default off); best-effort and fail-open -- a parse failure or an
uncharacterized PDK never blocks the pipeline.
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .arith_characterize import predict_op_delay as _predict_delay

# Default derate: an FRD throughput requirement may sit up to this factor above
# the roofline peak and still be "close to the roofline". The self-imposed
# budget (when the customer declines a hard cap) is peak * this factor.
_DEFAULT_DERATE = 2.0


def roofline_enabled() -> bool:
    """True iff CORESMITH_PERF_ROOFLINE is set truthy (default off)."""
    from orchestrator.profile import ensure_applied, flag_enabled
    ensure_applied()
    return flag_enabled("CORESMITH_PERF_ROOFLINE", default=False)


# --------------------------------------------------------------------------- #
# Dataflow spec  (task-agnostic; every field is declared, none baked in)
# --------------------------------------------------------------------------- #
@dataclass
class Op:
    """One primitive op on a datapath path. ``op``/``width`` index the PDK model."""
    op: str
    width: int
    label: str = ""


@dataclass
class Resource:
    """A resource class the datapath binds ops to (a multiplier, an S-box, a port).

    ResMII knob: uses_per_iter / instances. More instances -> lower ResMII
    (higher throughput) at the cost of area."""
    name: str
    op: str
    width: int
    instances: int
    uses_per_iter: float


@dataclass
class RecCycle:
    """A loop-carried dependency cycle.

    ``ops`` are the primitive ops traversed once around the cycle; their
    pipeline latency (in cycles) is derived from the SAME PDK delays + stage
    budget, so the recurrence bound is PDK-aware. ``extra_latency_cyc`` adds
    fixed non-op latency inside the loop (e.g. a memory read in an RMW).
    ``distance`` is the dependency distance (iterations spanned)."""
    name: str
    ops: list[Op]
    distance: int = 1
    extra_latency_cyc: int = 0


@dataclass
class DataflowSpec:
    name: str
    T_ns: float                      # target clock period
    margin_ns: float                 # setup + skew margin
    iterations: int                  # kernel iteration count
    pipeline_chain: list[Op]         # datapath critical path through ONE iter (sets D)
    resources: list[Resource] = field(default_factory=list)
    rec_cycles: list[RecCycle] = field(default_factory=list)
    drain_cyc: int = 0               # output-drain beyond the pipeline
    io_framing_cyc: int = 0          # fixed ingest/egress framing
    op_unit: str = "op"              # what one "op" is (block / frame / item)
    # Cross-reference to the FRD throughput requirement + the design's own claim.
    perf_req_id: str = ""            # e.g. "PERF-001"
    perf_req_cyc_per_op: float | None = None   # the FRD cap (None -> self-impose)
    declared_cyc_per_op: float | None = None   # the design's own claimed cyc/op
    derate: float = _DEFAULT_DERATE
    notes: str = ""


# --------------------------------------------------------------------------- #
# PDK-aware pricing (delegates to the real op-delay characterizer)
# --------------------------------------------------------------------------- #
def _op_delay(op: str, width: int, pdk: dict | None) -> float | None:
    """Combinational delay (ns) via the engine's ``predict_op_delay``.

    Returns None ONLY when the PDK is uncharacterized (no model doc). An op
    outside the characterized vocabulary is still priced (conservative
    unknown-op proxy) by ``predict_op_delay`` -- never treated as timing-free.
    """
    return _predict_delay(op, max(1, int(width)), pdk)


def _stage_partition(chain: list[Op], budget_ns: float,
                     pdk: dict | None) -> tuple[list[float], list[Op], int]:
    """Greedily pack a serial op chain into pipeline stages, each <= budget.

    Returns (per-stage comb delay list, ops whose OWN delay exceeds the budget,
    count of ops the PDK could not price). An op with delay > budget cannot be
    closed at this T (a real Fmax violation) -- reported so the caller flags it
    rather than silently pretend it fits. An unpriced op (uncharacterized PDK)
    conservatively takes its own full stage.
    """
    stages: list[float] = []
    cur = 0.0
    n_in_stage = 0
    over: list[Op] = []
    unpriced = 0
    for o in chain:
        d = _op_delay(o.op, o.width, pdk)
        if d is None:
            unpriced += 1
            d = budget_ns          # conservative: op fills its own stage
        if d > budget_ns + 1e-9:
            over.append(o)
        if n_in_stage > 0 and cur + d > budget_ns + 1e-9:
            stages.append(round(cur, 4))
            cur = 0.0
            n_in_stage = 0
        cur += d
        n_in_stage += 1
    if n_in_stage > 0:
        stages.append(round(cur, 4))
    return stages, over, unpriced


def _cone_delay_ns(ops: list[Op], pdk: dict | None) -> tuple[float, int]:
    """Total combinational delay (ns) of a serial op cone using the REAL
    characterizer delays, plus a count of ops the PDK could not price.

    ``(delay_ns, n_unpriced)``. An unpriced op contributes 0 to the sum here
    (the stage-partition prices it conservatively at a full stage separately),
    so ``n_unpriced`` lets the caller know the cone delay is a lower bound.
    """
    total = 0.0
    unpriced = 0
    for o in ops:
        d = _op_delay(o.op, o.width, pdk)
        if d is None:
            unpriced += 1
        else:
            total += d
    return total, unpriced


def _rec_latency_cyc(rc: RecCycle, budget_ns: float,
                     pdk: dict | None) -> tuple[int, float, bool]:
    """Fmax-COUPLED recurrence latency: ``(latency_cyc, cone_delay_ns, priced)``.

    The latency once around a loop-carried recurrence is the number of
    period-sized pipeline stages its REAL-delay op cone spans -- i.e.
    ``ceil(cone_delay_ns / stage_budget)`` -- NOT a prose-declared "1 cycle".
    This couples II to the clock BY CONSTRUCTION: a cone that fits one period is
    1 stage (II floor 1), a cone longer than the period is >= 2 stages, so an
    II=1 plan is only ever declared when the round cone actually closes the
    clock. ``extra_latency_cyc`` adds fixed NON-op latency (e.g. a memory read)
    and can only LENGTHEN this PDK-derived floor, never shorten it.

    ``priced`` is True only when the recurrence declared ops and every one was
    PDK-priceable -- so a recurrence with no priceable cone (a bare prose
    latency) is surfaced as unpriced rather than silently trusted.
    """
    stages, over, unpriced = _stage_partition(rc.ops, budget_ns, pdk)
    cone_ns, _ = _cone_delay_ns(rc.ops, pdk)
    lat = len(stages)
    for o in over:
        d = _op_delay(o.op, o.width, pdk) or budget_ns
        lat += int(math.ceil(d / budget_ns)) - 1
    # Explicit period coupling: floor the stage count at ceil(cone/budget). For
    # a well-packed cone the greedy stage count already equals/exceeds this, so
    # this never changes a result -- it makes the timing-derived floor the
    # authority even if the greedy packer changes.
    coupled = int(math.ceil(cone_ns / budget_ns)) if cone_ns > 0 else 0
    lat = max(lat, coupled)
    priced = bool(rc.ops) and unpriced == 0
    return lat + int(rc.extra_latency_cyc), round(cone_ns, 4), priced


def analyze(spec: DataflowSpec, pdk: dict | None = None) -> dict:
    """Compute the coupled Fmax + throughput roofline for one block."""
    budget = spec.T_ns - spec.margin_ns
    if budget <= 0:
        raise ValueError(f"{spec.name}: margin {spec.margin_ns} >= T {spec.T_ns}")

    # (a) Fmax / pipeline depth: partition the datapath chain.
    stages, over, unpriced = _stage_partition(spec.pipeline_chain, budget, pdk)
    D = max(1, len(stages))
    fmax_mhz = 1000.0 / spec.T_ns
    fmax_hz = 1e9 / spec.T_ns
    fmax_feasible = not over

    # (b) RecMII -- recurrence bound. Latency is Fmax-COUPLED: the number of
    # period-sized stages the recurrence's REAL-delay op cone spans, so an II=1
    # recurrence is only declared when the round cone actually fits the clock
    # (see _rec_latency_cyc). This makes the declared plan timing-feasible by
    # construction instead of trusting a prose 1-cycle latency.
    rec_terms = []
    unpriced_recs = []
    for rc in spec.rec_cycles:
        lat, cone_ns, priced = _rec_latency_cyc(rc, budget, pdk)
        dist = max(1, int(rc.distance))
        rec_terms.append({"name": rc.name, "latency_cyc": lat,
                          "cone_delay_ns": cone_ns, "distance": dist,
                          "priced": priced,
                          "recmii": int(math.ceil(lat / dist))})
        if rc.ops and not priced:
            unpriced_recs.append(rc.name)
    RecMII = max((t["recmii"] for t in rec_terms), default=0)
    # The binding recurrence's cone delay -- the round critical-path length the
    # II is coupled to (surfaced so a caller can see the plan is feasible).
    round_cone_delay_ns = max((t["cone_delay_ns"] for t in rec_terms),
                              default=0.0)

    # (b) ResMII -- resource bound.
    res_terms = []
    for r in spec.resources:
        inst = max(1, int(r.instances))
        res_terms.append({"name": r.name, "op": r.op, "width": r.width,
                          "uses_per_iter": r.uses_per_iter, "instances": inst,
                          "resmii": int(math.ceil(r.uses_per_iter / inst))})
    ResMII = max((t["resmii"] for t in res_terms), default=0)

    II_min = max(RecMII, ResMII, 1)

    # (c) Peak cyc/op and throughput.
    fill = D - 1
    cyc_per_op_peak = spec.iterations * II_min + fill + spec.drain_cyc + spec.io_framing_cyc
    ops_per_s_peak = fmax_hz / cyc_per_op_peak if cyc_per_op_peak > 0 else 0.0

    # (d) Binding constraint -- the lever a re-spec should pull.
    if RecMII >= ResMII and rec_terms:
        b = max(rec_terms, key=lambda t: t["recmii"])
        binding = {"type": "RecMII", "value": RecMII, "name": b["name"],
                   "detail": (f"recurrence '{b['name']}' latency {b['latency_cyc']}cyc "
                              f"/ distance {b['distance']}"),
                   "lever": "algebraically break the loop-carried recurrence"}
    elif res_terms:
        b = max(res_terms, key=lambda t: t["resmii"])
        binding = {"type": "ResMII", "value": ResMII, "name": b["name"],
                   "detail": (f"resource '{b['name']}' {b['uses_per_iter']} uses/iter "
                              f"/ {b['instances']} instances"),
                   "lever": f"instantiate more '{b['name']}' units (costs area)"}
    else:
        binding = {"type": "none", "value": II_min, "name": "(none)",
                   "detail": "no recurrence or resource declared", "lever": ""}

    # (e) Requirement check. The FRD PERF-NNN cap; when the customer declined a
    # hard cap the engine SELF-IMPOSES peak * derate ("no cap" != "no target").
    rec_floor = spec.iterations * RecMII + fill
    self_imposed = math.ceil(cyc_per_op_peak * max(1.0, spec.derate))
    req = spec.perf_req_cyc_per_op
    req_source = "frd" if req is not None else "self_imposed"
    if req is None:
        req = self_imposed
    claim = spec.declared_cyc_per_op
    meets_req: bool | None = None
    if claim is not None:
        meets_req = claim <= req + 1e-9

    sanity_notes = []
    if not fmax_feasible:
        sanity_notes.append(
            "Fmax INFEASIBLE: op(s) exceed stage budget at this T: "
            + ", ".join(f"{o.op}@{o.width}" for o in over))
    if unpriced:
        sanity_notes.append(f"{unpriced} op(s) unpriced (PDK uncharacterized) -- "
                            "priced conservatively at one stage each")
    if cyc_per_op_peak < rec_floor:
        sanity_notes.append(f"peak {cyc_per_op_peak} below recurrence floor {rec_floor}")
    if unpriced_recs:
        sanity_notes.append(
            "recurrence(s) not fully PDK-priced (latency NOT proven Fmax-coupled): "
            + ", ".join(unpriced_recs)
            + " -- declare the loop-carried op cone so the II is timing-derived, "
            "not a prose latency")
    if meets_req is False:
        sanity_notes.append(
            f"declared {claim} cyc/{spec.op_unit} MISSES {spec.perf_req_id or 'PERF'} "
            f"target {req} ({req_source}); lever: {binding.get('lever','')}")

    return {
        "block": spec.name,
        "op_unit": spec.op_unit,
        "fmax_mhz": round(fmax_mhz, 2),
        "T_ns": spec.T_ns,
        "margin_ns": spec.margin_ns,
        "stage_budget_ns": round(budget, 4),
        "pipeline_depth": D,
        "II_min": II_min,
        "RecMII": RecMII,
        "ResMII": ResMII,
        # Fmax-coupling: the binding recurrence's real-delay cone length (ns).
        # II is derived from ceil(this / stage_budget), so an II=1 plan is only
        # emitted when round_cone_delay_ns <= stage_budget_ns (feasible by
        # construction). Cone > budget -> latency >= 2 stages -> II reflects it.
        "round_cone_delay_ns": round(round_cone_delay_ns, 4),
        "binding_constraint": binding,
        "iterations": spec.iterations,
        "fill_cyc": fill,
        "drain_cyc": spec.drain_cyc,
        "io_framing_cyc": spec.io_framing_cyc,
        "cyc_per_op_peak": cyc_per_op_peak,
        "ops_per_s_peak": round(ops_per_s_peak, 1),
        "recurrence_floor_cyc": rec_floor,
        "stage_delays_ns": stages,
        "recmii_terms": rec_terms,
        "resmii_terms": res_terms,
        # ---- throughput requirement ----
        "perf_req_id": spec.perf_req_id,
        "perf_req_cyc_per_op": req,
        "perf_req_source": req_source,       # "frd" or "self_imposed"
        "self_imposed_budget_cyc_per_op": self_imposed,
        "derate": spec.derate,
        "declared_cyc_per_op": claim,
        "meets_throughput_req": meets_req,    # None when the spec declares no claim
        "fmax_feasible": fmax_feasible,
        "unpriced_ops": unpriced,
        "sanity_notes": sanity_notes,
        "notes": spec.notes,
    }


# --------------------------------------------------------------------------- #
# Machine-readable `perf` block parser  (declared by uarch_spec_generator §6)
# --------------------------------------------------------------------------- #
# A fenced block in the uArch spec markdown:
#
#   ```perf
#   { "op_unit": "block", "target_clock_mhz": 50, "iterations": 64,
#     "pipeline_chain": [["mul",16],["add",16]],
#     "resources": [{"name":"mac","op":"mul","width":16,"instances":1,
#                    "uses_per_iter":64}],
#     "rec_cycles": [{"name":"acc","ops":[["add",16]],"distance":1}],
#     "drain_cyc": 0, "io_framing_cyc": 0,
#     "perf_req_id": "PERF-001", "perf_req_cyc_per_op": 128,
#     "declared_cyc_per_op": 70 }
#   ```
_PERF_BLOCK_RE = re.compile(r"```perf\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)


def _as_op(item: Any) -> Op | None:
    """Accept ["mul",16] / ["mul",16,"label"] / {"op":..,"width":..}."""
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        return Op(str(item[0]), int(item[1]),
                  str(item[2]) if len(item) > 2 else "")
    if isinstance(item, dict) and "op" in item:
        return Op(str(item["op"]), int(item.get("width", 1)),
                  str(item.get("label", "")))
    return None


def parse_perf_spec(source: str, block_name: str = "",
                    default_clock_mhz: float = 50.0) -> DataflowSpec | None:
    """Extract a ``perf`` block from a uArch spec and build a DataflowSpec.

    Returns None when no ``perf`` block is present or it cannot be parsed --
    the caller then simply skips the block (fail-open). JSON only; no code runs.
    """
    m = _PERF_BLOCK_RE.search(source or "")
    if not m:
        return None
    try:
        doc = json.loads(m.group(1))
    except (ValueError, TypeError):
        return None
    if not isinstance(doc, dict):
        return None

    # clock / period
    T_ns = doc.get("T_ns")
    if T_ns is None:
        mhz = doc.get("target_clock_mhz") or default_clock_mhz
        T_ns = 1000.0 / max(1e-6, float(mhz))
    T_ns = float(T_ns)
    margin = doc.get("margin_ns")
    margin_ns = float(margin) if margin is not None else round(0.10 * T_ns, 4)

    chain = [op for op in (_as_op(x) for x in (doc.get("pipeline_chain") or [])) if op]
    resources = []
    for r in (doc.get("resources") or []):
        if isinstance(r, dict) and {"name", "op"} <= set(r):
            resources.append(Resource(
                name=str(r["name"]), op=str(r["op"]),
                width=int(r.get("width", 1)),
                instances=int(r.get("instances", 1)),
                uses_per_iter=float(r.get("uses_per_iter", 1)),
            ))
    rec_cycles = []
    for c in (doc.get("rec_cycles") or []):
        if isinstance(c, dict):
            ops = [op for op in (_as_op(x) for x in (c.get("ops") or [])) if op]
            rec_cycles.append(RecCycle(
                name=str(c.get("name", "rec")), ops=ops,
                distance=int(c.get("distance", 1)),
                extra_latency_cyc=int(c.get("extra_latency_cyc", 0)),
            ))

    def _optnum(key):
        v = doc.get(key)
        return float(v) if isinstance(v, (int, float)) else None

    return DataflowSpec(
        name=str(doc.get("block") or block_name or "block"),
        T_ns=T_ns, margin_ns=margin_ns,
        iterations=int(doc.get("iterations", 1) or 1),
        pipeline_chain=chain, resources=resources, rec_cycles=rec_cycles,
        drain_cyc=int(doc.get("drain_cyc", 0) or 0),
        io_framing_cyc=int(doc.get("io_framing_cyc", 0) or 0),
        op_unit=str(doc.get("op_unit", "op")),
        perf_req_id=str(doc.get("perf_req_id", "")),
        perf_req_cyc_per_op=_optnum("perf_req_cyc_per_op"),
        declared_cyc_per_op=_optnum("declared_cyc_per_op"),
        derate=float(doc.get("derate", _DEFAULT_DERATE) or _DEFAULT_DERATE),
        notes=str(doc.get("notes", "")),
    )


# --------------------------------------------------------------------------- #
# Per-block emission  (microarch phase)
# --------------------------------------------------------------------------- #
def _block_dir(project_root: str | os.PathLike, block_name: str) -> Path:
    return Path(project_root) / ".coresmith" / "blocks" / block_name


def perf_model_for_block(project_root: str | os.PathLike, block_name: str,
                         target_clock_mhz: float | None = None,
                         pdk: dict | None = None) -> dict | None:
    """Locate the block's uArch spec, parse its ``perf`` block, and analyze it.

    Returns the perf_model dict, or None when the spec has no ``perf`` block
    (or is missing). Never raises.
    """
    try:
        spec_path = (Path(project_root) / "arch" / "uarch_specs"
                     / f"{block_name}.md")
        if not spec_path.exists():
            return None
        if target_clock_mhz is None:
            from .latency_audit import resolve_target_clock_mhz
            target_clock_mhz = resolve_target_clock_mhz(project_root)
        spec = parse_perf_spec(spec_path.read_text(encoding="utf-8", errors="replace"),
                               block_name=block_name,
                               default_clock_mhz=target_clock_mhz)
        if spec is None:
            return None
        return analyze(spec, pdk=pdk)
    except Exception:  # noqa: BLE001 - best-effort, never block
        return None


def emit_perf_model(project_root: str | os.PathLike, block_name: str,
                    target_clock_mhz: float | None = None,
                    pdk: dict | None = None) -> Path | None:
    """Compute + persist ``perf_model.json`` into the block state dir.

    Returns the written path, or None when there was nothing to emit (no
    ``perf`` block declared). Never raises.
    """
    model = perf_model_for_block(project_root, block_name,
                                 target_clock_mhz=target_clock_mhz, pdk=pdk)
    if model is None:
        return None
    try:
        bd = _block_dir(project_root, block_name)
        bd.mkdir(parents=True, exist_ok=True)
        out = bd / "perf_model.json"
        out.write_text(json.dumps(model, indent=2))
        return out
    except Exception:  # noqa: BLE001
        return None


def read_perf_model(project_root: str | os.PathLike,
                    block_name: str) -> dict | None:
    """Read a previously emitted perf_model.json (or None)."""
    try:
        p = _block_dir(project_root, block_name) / "perf_model.json"
        if p.exists():
            return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        pass
    return None


def format_perf_fragment(model: dict) -> str:
    """Render the perf model as a compact contract line for a prompt / log."""
    if not model:
        return ""
    bc = model.get("binding_constraint", {})
    req = model.get("perf_req_cyc_per_op")
    src = model.get("perf_req_source", "")
    lines = [
        f"# THROUGHPUT ROOFLINE for {model.get('block','?')} "
        f"(@ {model.get('fmax_mhz','?')} MHz): peak "
        f"{model.get('cyc_per_op_peak','?')} cyc/{model.get('op_unit','op')} "
        f"at II={model.get('II_min','?')}, depth {model.get('pipeline_depth','?')}.",
        f"  requirement {model.get('perf_req_id') or 'PERF'} = {req} cyc/op ({src}); "
        f"binding = {bc.get('type','')} {bc.get('name','')} (={bc.get('value','')}); "
        f"lever: {bc.get('lever','')}",
    ]
    claim = model.get("declared_cyc_per_op")
    if claim is not None:
        verdict = ("MEETS" if model.get("meets_throughput_req")
                   else "MISSES") if model.get("meets_throughput_req") is not None else "?"
        lines.append(f"  declared {claim} cyc/op -> {verdict} the requirement")
    for n in model.get("sanity_notes", []):
        lines.append(f"  ! {n}")
    return "\n".join(lines)


__all__ = [
    "DataflowSpec",
    "Op",
    "RecCycle",
    "Resource",
    "analyze",
    "emit_perf_model",
    "format_perf_fragment",
    "parse_perf_spec",
    "perf_model_for_block",
    "read_perf_model",
    "roofline_enabled",
]
