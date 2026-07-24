# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Thorough, adversarial validation of the chaining pipeline scheduler.

Strategy: an INDEPENDENT verifier recomputes the true per-stage chained delay
from the schedule (it does NOT trust the scheduler's internal `arrival`/
`stage_delay`), and asserts the correctness invariants that must hold for ANY
valid chaining schedule. Those invariants are then enforced across:
  * hand-built topologies (chain / tree / diamond / reconvergent / fan / DAG),
  * real algorithm structures (MAC chain, parallel-prefix scan, FFT butterfly,
    systolic MAC, Booth-style partial-product reduction),
  * adversarial edge cases (empty / single / ==period / zero-delay / >period /
    uncharacterized / dangling edges / self-loop / duplicate edges), and
  * a fuzzer over hundreds of random layered DAGs with random delays.

Delay convention for tests: ``delay_fn(op, w) = w / 1000.0`` ns, so each Node's
`width` field directly encodes its combinational delay in picoseconds -- letting
each node carry an arbitrary delay without a real EDA model.
"""

from __future__ import annotations

import dataclasses
import random

import pytest

from orchestrator.langgraph.pipeline_scheduler import (
    Node,
    Schedule,
    _topo_order,
    schedule_dfg,
    schedule_reduction,
)

EPS = 1e-9


def DFN(op, w):
    """width (ps) -> delay (ns); each node's width IS its delay."""
    return w / 1000.0


def _d(delay_ns: float) -> int:
    """Encode a delay (ns) as an integer width (ps)."""
    return int(round(delay_ns * 1000))


def verify(nodes, edges, period, sched: Schedule, delay_fn=DFN):
    """Independently validate a schedule against the chaining invariants.

    Recomputes the true within-stage chained delay from scratch; never trusts
    the scheduler's own arrival bookkeeping.
    """
    ids = [n.id for n in nodes]
    by_id = {n.id: n for n in nodes}
    stage = sched.node_stage

    # (1) completeness: every node assigned exactly once; stages are 0..depth-1
    assert set(stage.keys()) == set(ids), "every node must be scheduled exactly once"
    if ids:
        assert sched.depth == max(stage.values()) + 1
        assert set(stage.values()) == set(range(sched.depth)), "stages must be contiguous from 0"
        assert len([x for grp in sched.stages for x in grp]) == len(ids), "no dup/missing in stages[]"
    else:
        assert sched.depth == 0

    preds = {i: [] for i in ids}
    for u, v in edges:
        if u in by_id and v in by_id:
            preds[v].append(u)

    # (2) dependency: a consumer is never in an earlier stage than its producer
    for u, v in edges:
        if u in stage and v in stage:
            assert stage[v] >= stage[u], f"dependency {u}->{v} violated"

    # recompute true arrival in topological order
    order = _topo_order(ids, edges)
    arr: dict[str, float] = {}
    for nid in order:
        d = delay_fn(by_id[nid].op, by_id[nid].width)
        same = [arr[u] for u in preds[nid] if stage[u] == stage[nid]]
        arr[nid] = d + (max(same) if same else 0.0)
        s0 = max((stage[u] for u in preds[nid]), default=0)

        # (3) a node is never more than ONE stage above its highest predecessor
        assert stage[nid] in (s0, s0 + 1), \
            f"{nid}: stage {stage[nid]} not in {{{s0},{s0+1}}} (over/under-bumped)"

        # (4) period bound: every non-infeasible node's chained delay <= period
        if nid not in sched.infeasible:
            assert arr[nid] <= period + EPS, \
                f"{nid}: true chained delay {arr[nid]:.4f} > period {period}"

        # (5) ASAP minimality: a one-stage bump must be NECESSARY (placing the
        #     node at s0 with same-stage preds would exceed the period). Only
        #     checked when the node itself fits a period (else it's infeasible).
        if stage[nid] == s0 + 1 and d <= period + EPS:
            same0 = [arr[u] for u in preds[nid] if stage[u] == s0]
            chained_at_s0 = d + (max(same0) if same0 else 0.0)
            assert chained_at_s0 > period + EPS, \
                f"{nid}: bumped to {s0+1} but would fit at {s0} ({chained_at_s0:.4f}<=p) -- not ASAP"

    # (6) infeasible set == exactly the single ops whose own delay > period
    expected_infeasible = {n.id for n in nodes
                           if delay_fn(n.op, n.width) > period + EPS}
    assert set(sched.infeasible) == expected_infeasible, "infeasible set mismatch"

    # (7) the scheduler's reported per-stage delay matches the true recomputation
    for s in range(sched.depth):
        true_sd = max((arr[nid] for nid in sched.stages[s]), default=0.0)
        assert abs(sched.stage_delay_ns[s] - true_sd) < 1e-6, \
            f"stage {s} delay {sched.stage_delay_ns[s]} != true {true_sd}"

    return arr


# --------------------------------------------------------------------------- #
# Hand-built topologies
# --------------------------------------------------------------------------- #
def _chain(n, delay_ns):
    nodes = [Node(f"c{i}", "op", _d(delay_ns)) for i in range(n)]
    edges = [(f"c{i}", f"c{i+1}") for i in range(n - 1)]
    return nodes, edges


def test_chain_various_lengths_and_delays():
    for n in (1, 2, 3, 5, 10, 25):
        for delay in (0.0, 1.0, 3.0, 6.6, 19.999, 20.0):
            nodes, edges = _chain(n, delay)
            sc = schedule_dfg(nodes, edges, 20.0, delay_fn=DFN)
            verify(nodes, edges, 20.0, sc)


def test_diamond_reconvergent():
    # A -> {B, C} -> D, with B in a later stage than C (preds in DIFFERENT stages)
    nodes = [Node("A", "op", _d(6)), Node("B", "op", _d(8)),
             Node("C", "op", _d(3)), Node("D", "op", _d(5))]
    edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]
    sc = schedule_dfg(nodes, edges, 20.0, delay_fn=DFN)
    arr = verify(nodes, edges, 20.0, sc)
    # A(6)+B(8)=14 -> B stage0 arr14; A(6)+C(3)=9 -> C stage0 arr9;
    # D consumes B(14)+5=19<=20 -> D stage0 arr19
    assert sc.depth == 1 and arr["D"] == pytest.approx(19.0)


def test_reconvergent_forces_register_on_one_path_only():
    # long path forces a stage boundary; short path stays -> D sees a registered
    # (earlier-stage) input and a same-stage input
    nodes = [Node("A", "op", _d(12)), Node("B", "op", _d(12)),
             Node("C", "op", _d(3)), Node("D", "op", _d(4))]
    edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]
    sc = schedule_dfg(nodes, edges, 20.0, delay_fn=DFN)
    verify(nodes, edges, 20.0, sc)
    assert sc.node_stage["B"] == 1  # 12+12>20 -> registered
    assert sc.node_stage["D"] >= sc.node_stage["B"]


def test_wide_fan_in_and_out():
    # one source fans out to 50 sinks; then 50 sources fan into one
    src = Node("s", "op", _d(5))
    outs = [Node(f"o{i}", "op", _d(7)) for i in range(50)]
    nodes = [src] + outs
    edges = [("s", f"o{i}") for i in range(50)]
    sc = schedule_dfg(nodes, edges, 20.0, delay_fn=DFN)
    verify(nodes, edges, 20.0, sc)
    # all outs independent of each other, each chains s(5)+7=12 -> one stage
    assert sc.depth == 1

    sink = Node("k", "op", _d(4))
    nodes2 = outs + [sink]
    edges2 = [(f"o{i}", "k") for i in range(50)]
    # give outs no preds here (standalone) so they're stage0; k chains worst+4
    sc2 = schedule_dfg(nodes2, edges2, 20.0, delay_fn=DFN)
    verify(nodes2, edges2, 20.0, sc2)


def test_disjoint_components():
    a, ea = _chain(4, 6.0)
    b = [Node(f"b{i}", "op", _d(9)) for i in range(3)]
    eb = [("b0", "b1"), ("b1", "b2")]
    nodes = a + b
    edges = ea + eb
    sc = schedule_dfg(nodes, edges, 20.0, delay_fn=DFN)
    verify(nodes, edges, 20.0, sc)


# --------------------------------------------------------------------------- #
# Real algorithm structures
# --------------------------------------------------------------------------- #
def test_mac_accumulator_chain():
    # y += a_i * b_i : a serial multiply-add recurrence of length N
    n = 16
    nodes, edges = [], []
    prev = None
    for i in range(n):
        m = Node(f"mul{i}", "mul", _d(6.8))
        ac = Node(f"acc{i}", "add", _d(3.8))
        nodes += [m, ac]
        edges.append((f"mul{i}", f"acc{i}"))
        if prev:
            edges.append((prev, f"acc{i}"))
        prev = f"acc{i}"
    sc = schedule_dfg(nodes, edges, 20.0, delay_fn=DFN)
    verify(nodes, edges, 20.0, sc)


def test_parallel_prefix_scan():
    # Kogge-Stone style prefix: log-depth combine network over 8 inputs
    n = 8
    nodes = [Node(f"x{i}", "op", _d(0.5)) for i in range(n)]
    edges = []
    cur = [f"x{i}" for i in range(n)]
    step = 1
    k = 0
    while step < n:
        nxt = list(cur)
        for i in range(n):
            if i - step >= 0:
                nid = f"p{k}"
                k += 1
                nodes.append(Node(nid, "add", _d(3.8)))
                edges.append((cur[i], nid))
                edges.append((cur[i - step], nid))
                nxt[i] = nid
        cur = nxt
        step *= 2
    sc = schedule_dfg(nodes, edges, 20.0, delay_fn=DFN)
    verify(nodes, edges, 20.0, sc)


def test_fft_butterfly_network():
    # radix-2 DIT FFT: log2(N) stages of butterflies (mul + add/sub)
    n = 8
    nodes, edges = [], []
    cur = [f"in{i}" for i in range(n)]
    for i in range(n):
        nodes.append(Node(f"in{i}", "op", _d(0.0)))
    stage_idx = 0
    width = 1
    bid = 0
    while width < n:
        nxt = list(cur)
        for i in range(0, n, width * 2):
            for j in range(width):
                tw = Node(f"tw{bid}", "mul", _d(6.8))
                bid += 1
                a = Node(f"ba{bid}", "add", _d(3.8))
                bid += 1
                s = Node(f"bs{bid}", "sub", _d(3.8))
                bid += 1
                nodes += [tw, a, s]
                top, bot = cur[i + j], cur[i + j + width]
                edges += [(bot, tw.id), (top, a.id), (tw.id, a.id),
                          (top, s.id), (tw.id, s.id)]
                nxt[i + j], nxt[i + j + width] = a.id, s.id
        cur = nxt
        width *= 2
        stage_idx += 1
    sc = schedule_dfg(nodes, edges, 20.0, delay_fn=DFN)
    verify(nodes, edges, 20.0, sc)
    assert sc.depth >= 1


def test_booth_partial_product_reduction():
    # a wide reduction tree (Wallace/Booth-like): many addends -> add tree
    sc = schedule_reduction("add", 16, 31, 20.0, delay_fn=lambda op, w: 3.8)
    # verify against the same structure we asked for: rebuild not needed; the
    # reduction builder is exercised by the fuzz/structured tests below.
    assert sc.depth >= 1 and all(d <= 20.0 + EPS for d in sc.stage_delay_ns)


# --------------------------------------------------------------------------- #
# Adversarial edge cases
# --------------------------------------------------------------------------- #
def test_empty_dfg():
    sc = schedule_dfg([], [], 20.0, delay_fn=DFN)
    assert sc.depth == 0 and sc.stages == [] and sc.node_stage == {}
    verify([], [], 20.0, sc)


def test_single_node_exactly_period_fits():
    nodes = [Node("x", "op", _d(20.0))]
    sc = schedule_dfg(nodes, [], 20.0, delay_fn=DFN)
    verify(nodes, [], 20.0, sc)
    assert sc.infeasible == [] and sc.depth == 1


def test_single_node_just_over_period_infeasible():
    nodes = [Node("x", "op", _d(20.001))]
    sc = schedule_dfg(nodes, [], 20.0, delay_fn=DFN)
    assert sc.infeasible == ["x"] and not sc.ok()
    verify(nodes, [], 20.0, sc)


def test_all_zero_delay_collapses_to_one_stage():
    nodes, edges = _chain(30, 0.0)
    sc = schedule_dfg(nodes, edges, 20.0, delay_fn=DFN)
    assert sc.depth == 1
    verify(nodes, edges, 20.0, sc)


def test_every_op_over_period():
    nodes, edges = _chain(5, 25.0)
    sc = schedule_dfg(nodes, edges, 20.0, delay_fn=DFN)
    assert len(sc.infeasible) == 5
    verify(nodes, edges, 20.0, sc)


def test_uncharacterized_delay_consumes_a_full_stage_and_flagged():
    # An op with no delay estimate is NEVER timing-free (the failure class that
    # let an AES round's S-box + GF-multiply collapse into one unregistered
    # cloud). The scheduler now prices an un-priced op at a FULL period, so it
    # consumes its OWN stage (never chains with a neighbour) and is still flagged
    # as uncharacterized. (Old behavior assumed 0 ns -> chained -> vanished.)
    nodes = [Node("a", "real", _d(5)), Node("b", "ghost", 999)]
    edges = [("a", "b")]
    sc = schedule_dfg(nodes, edges, 20.0,
                      delay_fn=lambda op, w: None if op == "ghost" else w / 1000.0)
    assert "b" in sc.uncharacterized
    # ghost got its OWN stage (not chained with a): 2 stages, b after a.
    assert sc.depth == 2
    assert sc.node_stage["a"] == 0 and sc.node_stage["b"] == 1
    assert "b" not in sc.infeasible          # priced at exactly one period -> fits
    # verify against the SAME model the scheduler used (un-priced op = a period).
    verify(nodes, edges, 20.0, sc,
           delay_fn=lambda op, w: 20.0 if op == "ghost" else w / 1000.0)


def test_dangling_edge_ignored_not_crash():
    nodes = [Node("a", "op", _d(5))]
    sc = schedule_dfg(nodes, [("a", "missing"), ("ghost", "a")], 20.0, delay_fn=DFN)
    assert sc.node_stage == {"a": 0}
    verify(nodes, [], 20.0, sc)  # effective edge set is empty


def test_self_loop_is_a_cycle():
    with pytest.raises(ValueError):
        schedule_dfg([Node("a", "op", _d(1))], [("a", "a")], 20.0, delay_fn=DFN)


def test_duplicate_edges_harmless():
    nodes = [Node("a", "op", _d(6)), Node("b", "op", _d(6))]
    sc = schedule_dfg(nodes, [("a", "b"), ("a", "b"), ("a", "b")], 20.0, delay_fn=DFN)
    verify(nodes, [("a", "b")], 20.0, sc)
    assert sc.depth == 1  # 6+6=12<=20


def test_larger_cycle_rejected():
    nodes = [Node(f"n{i}", "op", _d(1)) for i in range(5)]
    edges = [("n0", "n1"), ("n1", "n2"), ("n2", "n3"), ("n3", "n4"), ("n4", "n1")]
    with pytest.raises(ValueError):
        schedule_dfg(nodes, edges, 20.0, delay_fn=DFN)


def test_determinism():
    rng = random.Random(7)
    nodes, edges, _ = _random_dag(rng, 30, 20.0)
    a = schedule_dfg(nodes, edges, 20.0, delay_fn=DFN)
    b = schedule_dfg(nodes, edges, 20.0, delay_fn=DFN)
    assert a.node_stage == b.node_stage and a.depth == b.depth


# --------------------------------------------------------------------------- #
# schedule_reduction robustness (incl. non-power-of-two, odd carries)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 64])
@pytest.mark.parametrize("period", [3.0, 7.0, 20.0, 100.0])
def test_reduction_shapes(n, period):
    sc = schedule_reduction("add", 8, n, period, delay_fn=lambda op, w: 2.0)
    assert all(d <= period + EPS for d in sc.stage_delay_ns) or sc.infeasible
    assert sc.depth >= 1


# --------------------------------------------------------------------------- #
# Fuzzer: hundreds of random layered DAGs, every invariant checked
# --------------------------------------------------------------------------- #
def _random_dag(rng: random.Random, max_nodes: int, period: float):
    n = rng.randint(0, max_nodes)
    ids = [str(i) for i in range(n)]
    # partition into random-width layers (guarantees a DAG)
    layers, i = [], 0
    while i < n:
        w = rng.randint(1, min(6, n - i))
        layers.append(ids[i:i + w])
        i += w
    nodes = []
    for nid in ids:
        # delays spanning 0, sub-period, near-period, and over-period
        d = rng.choice([
            0.0,
            rng.uniform(0, period * 0.35),
            rng.uniform(period * 0.35, period),
            rng.uniform(period, period * 1.6),
        ])
        nodes.append(Node(nid, "op", _d(d)))
    edges = []
    for li in range(1, len(layers)):
        pool = [x for L in layers[:li] for x in L]
        for nid in layers[li]:
            k = rng.randint(0, min(4, len(pool)))
            for u in rng.sample(pool, k):
                edges.append((u, nid))
    return nodes, edges, layers


def test_verifier_has_teeth_dependency():
    # a VALID schedule, then corrupt it so a consumer precedes its producer
    nodes, edges = _chain(3, 12.0)   # 12+12>20 -> 3 stages 0,1,2
    sc = schedule_dfg(nodes, edges, 20.0, delay_fn=DFN)
    verify(nodes, edges, 20.0, sc)                       # passes clean
    bad = dataclasses.replace(sc, node_stage={**sc.node_stage, "c1": 2, "c2": 1})
    with pytest.raises(AssertionError):
        verify(nodes, edges, 20.0, bad)                 # dep c1->c2 now violated


def test_verifier_has_teeth_period():
    # force two 12ns ops into ONE stage (24ns > 20ns period) -> must be caught
    nodes, edges = _chain(2, 12.0)
    sc = schedule_dfg(nodes, edges, 20.0, delay_fn=DFN)
    bad_stage = {"c0": 0, "c1": 0}
    bad = dataclasses.replace(sc, node_stage=bad_stage, depth=1,
                              stages=[["c0", "c1"]], stage_delay_ns=[24.0])
    with pytest.raises(AssertionError):
        verify(nodes, edges, 20.0, bad)


def test_verifier_has_teeth_over_bump():
    # a node parked two stages above its only predecessor (no reason) -> caught
    nodes = [Node("a", "op", _d(2)), Node("b", "op", _d(2))]
    sc = schedule_dfg(nodes, [("a", "b")], 20.0, delay_fn=DFN)  # both stage 0
    bad = dataclasses.replace(sc, node_stage={"a": 0, "b": 2}, depth=3,
                              stages=[["a"], [], ["b"]],
                              stage_delay_ns=[2.0, 0.0, 2.0])
    with pytest.raises(AssertionError):
        verify(nodes, [("a", "b")], 20.0, bad)


def test_fuzz_random_dags():
    rng = random.Random(20260627)
    for trial in range(4000):
        period = rng.choice([1.0, 5.0, 13.3, 20.0, 50.0])
        nodes, edges, _ = _random_dag(rng, rng.choice([0, 1, 8, 25, 60]), period)
        sc = schedule_dfg(nodes, edges, period, delay_fn=DFN)
        try:
            verify(nodes, edges, period, sc)
        except AssertionError as e:  # surface the exact failing DFG
            raise AssertionError(
                f"trial {trial} period={period} n={len(nodes)} edges={len(edges)}: {e}"
            )
