# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Pipeline scheduler -- decides how much arithmetic lands in each clock stage.

Consumes the per-op combinational delay model from ``arith_characterize`` (piece
#1) and a target clock period, and assigns every operation in a datapath's
dataflow graph (DFG) to a pipeline stage so that **no stage's chained
combinational delay exceeds the period**. This is the capability CoreSmith
lacked -- the reason a rate-distortion mode-search compiled as one giant
combinational cloud (functionally correct, unsynthesizable).

Algorithm: a chaining-aware ASAP pass over the DFG in topological order. It is
the SDC *chaining constraint* (Cong & Zhang, DAC'06; XLS's scheduler) for the
acyclic min-latency case -- and because the SDC constraint matrix is totally
unimodular, this greedy longest-path-with-chaining pass is exact (no LP solver
needed). Two operations chain within a stage iff their summed delay fits the
period; otherwise a register boundary (a new stage) is inserted between them.

Output: per-stage op lists + pipeline depth + each stage's critical delay, plus
any single op whose own delay exceeds the period (which must be decomposed /
multicycled, not just registered). The µArch agent turns this into a per-stage
pipeline contract (piece #3) that constrains both the Amaranth block model and RTL.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .arith_characterize import predict_op_delay as _default_delay


@dataclass
class Node:
    """One operation in the datapath DFG."""
    id: str
    op: str                 # add/sub/mul/cmp/mux/shift/sad/... (keys of the delay model)
    width: int              # operand bitwidth (or #terms for a sad reduction)


@dataclass
class Schedule:
    stages: list[list[str]]              # stage index -> node ids in that stage
    node_stage: dict[str, int]           # node id -> stage
    stage_delay_ns: list[float]          # per-stage critical chained delay
    period_ns: float
    depth: int                           # number of pipeline stages (latency in cycles)
    infeasible: list[str] = field(default_factory=list)   # single ops > period
    uncharacterized: list[str] = field(default_factory=list)  # no delay model entry
    fmax_mhz: float = 0.0                 # 1000 / max stage delay

    def ok(self) -> bool:
        return not self.infeasible


def _topo_order(node_ids: list[str], edges: list[tuple[str, str]]) -> list[str]:
    """Kahn topological sort. Raises on a cycle (combinational loop in the DFG)."""
    preds: dict[str, set[str]] = {n: set() for n in node_ids}
    succ: dict[str, set[str]] = {n: set() for n in node_ids}
    for u, v in edges:
        if u in preds and v in preds:
            preds[v].add(u)
            succ[u].add(v)
    ready = [n for n in node_ids if not preds[n]]
    order: list[str] = []
    indeg = {n: len(preds[n]) for n in node_ids}
    while ready:
        n = ready.pop()
        order.append(n)
        for m in succ[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                ready.append(m)
    if len(order) != len(node_ids):
        raise ValueError("DFG has a cycle (combinational loop) -- not schedulable")
    return order


def schedule_dfg(nodes: list[Node], edges: list[tuple[str, str]],
                 period_ns: float,
                 delay_fn: Callable[[str, int], float | None] | None = None,
                 pdk: dict | None = None) -> Schedule:
    """Chaining-ASAP schedule of a DFG to meet ``period_ns``.

    ``edges`` are (producer_id, consumer_id) data dependencies. ``delay_fn(op,
    width) -> ns`` defaults to the characterized model; inject for tests."""
    dfn = delay_fn or (lambda op, w: _default_delay(op, w, pdk))
    by_id = {n.id: n for n in nodes}
    ids = [n.id for n in nodes]
    order = _topo_order(ids, edges)
    preds: dict[str, list[str]] = {n: [] for n in ids}
    for u, v in edges:
        if u in by_id and v in by_id:
            preds[v].append(u)

    stage: dict[str, int] = {}
    arrival: dict[str, float] = {}   # chained delay within a node's stage, up to its output
    infeasible: list[str] = []
    uncharacterized: list[str] = []

    for nid in order:
        n = by_id[nid]
        d = dfn(n.op, n.width)
        if d is None:
            # An op with no delay estimate is NEVER timing-free. Assuming 0 ns
            # here (the old behavior) is what let an un-priced op (a crypto
            # S-box, a GF multiply) chain into an unregistered combinational
            # cloud. Price it at a full period so it consumes its OWN stage and
            # can never chain with a neighbour -- the conservative assumption.
            uncharacterized.append(nid)
            d = period_ns
        if d > period_ns + 1e-9:
            infeasible.append(nid)   # a single op can't fit a cycle -> must decompose
        p = preds[nid]
        if not p:
            stage[nid] = 0
            arrival[nid] = d
            continue
        s = max(stage[u] for u in p)                       # earliest legal stage
        in_stage = [arrival[u] for u in p if stage[u] == s]
        cand = (max(in_stage) if in_stage else 0.0) + d    # chain with same-stage preds
        if cand <= period_ns + 1e-9:
            stage[nid] = s
            arrival[nid] = cand
        else:
            stage[nid] = s + 1                              # register: new stage
            arrival[nid] = d

    depth = (max(stage.values()) + 1) if stage else 0
    stages: list[list[str]] = [[] for _ in range(depth)]
    for nid, s in stage.items():
        stages[s].append(nid)
    stage_delay = [max((arrival[nid] for nid in grp), default=0.0) for grp in stages]
    worst = max(stage_delay) if stage_delay else 0.0
    return Schedule(
        stages=stages, node_stage=stage, stage_delay_ns=stage_delay,
        period_ns=period_ns, depth=depth, infeasible=infeasible,
        uncharacterized=uncharacterized,
        fmax_mhz=(1000.0 / worst if worst > 0 else 0.0),
    )


def schedule_reduction(op: str, width: int, n_terms: int, period_ns: float,
                       combine: str = "add",
                       delay_fn: Callable[[str, int], float | None] | None = None,
                       pdk: dict | None = None) -> Schedule:
    """Schedule an N-term reduction: N leaf ops feeding a balanced ``combine`` tree.

    This is the canonical search/accumulate shape (e.g. a SAD/SSD over N pixels,
    or evaluating N candidate costs then min-reducing). Answers directly: "how
    many pipeline stages does a single-cycle N-way search actually require?"
    """
    nodes: list[Node] = []
    edges: list[tuple[str, str]] = []
    # N leaf evaluations (independent)
    leaves = []
    for i in range(n_terms):
        nid = f"leaf{i}"
        nodes.append(Node(nid, op, width))
        leaves.append(nid)
    # balanced binary combine tree
    acc_w = width + max(1, n_terms.bit_length())
    level = leaves
    k = 0
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 2):
            if i + 1 < len(level):
                nid = f"comb{k}"
                k += 1
                nodes.append(Node(nid, combine, acc_w))
                edges.append((level[i], nid))
                edges.append((level[i + 1], nid))
                nxt.append(nid)
            else:
                nxt.append(level[i])   # odd one carried up
        level = nxt
    return schedule_dfg(nodes, edges, period_ns, delay_fn=delay_fn, pdk=pdk)


def pipeline_contract_text(sched: Schedule, title: str = "datapath") -> str:
    """Render the schedule as a per-stage pipeline contract for the µArch spec."""
    lines = [
        f"PIPELINE CONTRACT for {title}: {sched.depth} stage(s) @ "
        f"{1000.0 / sched.period_ns:.0f} MHz (period {sched.period_ns:.2f} ns), "
        f"predicted Fmax {sched.fmax_mhz:.0f} MHz.",
    ]
    for s, grp in enumerate(sched.stages):
        lines.append(f"  - Stage {s}: {len(grp)} op(s), critical "
                     f"{sched.stage_delay_ns[s]:.2f} ns -> register the outputs.")
    if sched.infeasible:
        lines.append(f"  ! {len(sched.infeasible)} op(s) exceed one period alone "
                     f"({', '.join(sched.infeasible[:6])}): DECOMPOSE/multicycle "
                     "these (wider op split or sub-stage), do not merely register.")
    if sched.uncharacterized:
        lines.append(f"  ? uncharacterized ops (no delay model): "
                     f"{', '.join(sched.uncharacterized[:6])} -- priced at a "
                     f"FULL stage each (conservative: an un-priced op is never "
                     f"treated as timing-free).")
    return "\n".join(lines)


def pdk_budget_section(target_clock_mhz: float = 50.0,
                       pdk: dict | None = None) -> str:
    """Render the characterized arithmetic timing budget for the µArch prompt.

    Returns '' if the PDK arithmetic model isn't characterized yet (caller skips
    the section). Gives the LLM the real per-op delay + how many chain per stage
    at the target clock, plus the worked search/reduce example -- the concrete
    numbers it needs to size pipeline stages instead of emitting a 1-cycle cloud.
    """
    period = 1000.0 / max(1e-6, target_clock_mhz)
    # Include the crypto/codec primitives (lut = S-box/ROM by ADDRESS width,
    # gfmul = GF(2^W) multiply, xortree = wide XOR mixing) so a round datapath
    # is priced, not silently assumed timing-free.
    probes = [("mux", 8), ("cmp", 16), ("shift", 16), ("add", 16), ("sub", 16),
              ("mul", 16), ("mul", 32), ("sad", 8), ("sad", 16),
              ("lut", 8), ("gfmul", 8), ("gfmul", 32), ("xortree", 32)]
    rows = []
    over_period: list[tuple[str, int, float]] = []
    for op, w in probes:
        d = _default_delay(op, w, pdk)
        if d is None:
            continue
        n = max(1, int(period // d)) if d > 0 else 0
        unit = "terms" if op == "sad" else ("addr-b" if op == "lut" else "b")
        rows.append(f"  - {op}@{w}{unit}: {d:.2f} ns  ->  {n} chain per "
                    f"{period:.0f} ns stage")
        if d > period + 1e-9:
            over_period.append((op, w, d))
    if not rows:
        return ""
    # worked example: a 9-candidate search of 16-term SADs
    ex = schedule_reduction("sad", 16, 9, period, combine="cmp", pdk=pdk)
    # Any SINGLE op whose own delay exceeds the period cannot be registered at
    # its boundary alone -- it MUST be pipelined WITHIN the op. This is the AES
    # failure class: one S-box + GF-multiply round is ~27 ns against a 20 ns
    # clock, so it needs an internal register regardless of round boundaries.
    within_op_rule = ""
    if over_period:
        items = "; ".join(
            f"{op}@{w} = {d:.2f} ns (> {period:.2f} ns period, "
            f"needs >={max(2, int(-(-d // period)))} internal stages)"
            for op, w, d in over_period
        )
        within_op_rule = (
            "\n**PIPELINE-WITHIN-OP RULE (hard):** these single ops each take "
            f"LONGER than one {period:.2f} ns period: {items}. Registering only "
            "at the op's boundary is NOT enough -- you MUST insert register(s) "
            "INSIDE the op (split its combinational logic into the stated number "
            "of stages, ceil(delay/period)) and STATE that internal stage count "
            "in section 6 (Timing), citing the op-delay estimate above. This is "
            "the crypto-round / table-lookup / Galois-field failure class the "
            "old model priced at 0 ns.\n"
        )
    return (
        f"# PDK Timing Budget (characterized on this PDK; "
        f"{target_clock_mhz:.0f} MHz, period {period:.2f} ns)\n\n"
        "Real per-operation combinational delay and how many CHAIN within one "
        "clock period (sky130 std cells, your synth/STA flow):\n"
        + "\n".join(rows) + "\n"
        + within_op_rule + "\n"
        "**PIPELINE RULE (hard):** partition the datapath so the CHAINED "
        "combinational delay in every stage is <= the period above, registering "
        "between stages. A single-cycle exhaustive search / RD-mode-decision "
        "over N candidates is FORBIDDEN -- it becomes one combinational cloud "
        "that is unsynthesizable and fails timing. Either (a) PIPELINE it over "
        "stages (one reusable datapath + FSM, ~N cycles) or (b) PARALLELIZE + "
        "reduce (N copies + a compare/add tree, ~1-2 stages, N x area) -- pick "
        "per the block's flip_flop_budget + area_budget. State the chosen "
        "**pipeline depth (cycles)** and the **arithmetic in each stage** in "
        "section 6 (Timing) of this spec.\n\n"
        f"Worked example -- a 9-candidate 16-term-SAD search at "
        f"{target_clock_mhz:.0f} MHz: 9 PARALLEL SADs + a compare-tree reduce = "
        f"{ex.depth} stage(s) but ~9x the SAD area; serialized onto ONE SAD "
        f"datapath via an FSM = ~9 cycles at ~1x area. A single 16-term SAD "
        f"already fills ~{_default_delay('sad', 16, pdk):.0f}/{period:.0f} ns, so "
        "the search CANNOT be one combinational cycle either way.\n"
    )


__all__ = [
    "Node",
    "Schedule",
    "pdk_budget_section",
    "pipeline_contract_text",
    "schedule_dfg",
    "schedule_reduction",
]


if __name__ == "__main__":  # pragma: no cover
    from orchestrator.profile import apply as _apply_profile
    _apply_profile()
    import sys
    # e.g.  python -m ...pipeline_scheduler sad 16 9 20    (op width nterms period)
    op, width, n, period = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
    sc = schedule_reduction(op, width, n, period)
    print(pipeline_contract_text(sc, title=f"{n}x {op}@{width}"))
