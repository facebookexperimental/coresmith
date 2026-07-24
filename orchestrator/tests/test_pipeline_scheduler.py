# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Hermetic tests for the chaining pipeline scheduler (delay model injected)."""

from __future__ import annotations

import pytest

from orchestrator.langgraph.pipeline_scheduler import (
    Node,
    pipeline_contract_text,
    schedule_dfg,
    schedule_reduction,
)


def _const(ns):
    return lambda op, w: ns


def test_two_chained_ops_fit_one_stage():
    # 3ns + 3ns = 6ns <= 10ns period -> one stage
    nodes = [Node("a", "add", 8), Node("b", "add", 8)]
    sc = schedule_dfg(nodes, [("a", "b")], 10.0, delay_fn=_const(3.0))
    assert sc.depth == 1 and sc.node_stage == {"a": 0, "b": 0}
    assert sc.stage_delay_ns[0] == pytest.approx(6.0)


def test_two_chained_ops_split_when_over_period():
    # 6ns + 6ns = 12ns > 10ns -> register between them -> two stages
    nodes = [Node("a", "add", 8), Node("b", "add", 8)]
    sc = schedule_dfg(nodes, [("a", "b")], 10.0, delay_fn=_const(6.0))
    assert sc.depth == 2 and sc.node_stage == {"a": 0, "b": 1}


def test_independent_ops_share_a_stage():
    nodes = [Node("a", "add", 8), Node("b", "add", 8)]  # no edge
    sc = schedule_dfg(nodes, [], 10.0, delay_fn=_const(4.0))
    assert sc.depth == 1 and set(sc.stages[0]) == {"a", "b"}


def test_single_op_over_period_is_infeasible():
    nodes = [Node("big", "mul", 64)]
    sc = schedule_dfg(nodes, [], 10.0, delay_fn=_const(15.0))
    assert sc.infeasible == ["big"] and not sc.ok()


def test_combinational_loop_rejected():
    nodes = [Node("a", "add", 8), Node("b", "add", 8)]
    with pytest.raises(ValueError):
        schedule_dfg(nodes, [("a", "b"), ("b", "a")], 10.0, delay_fn=_const(1.0))


def test_long_chain_splits_into_expected_depth():
    # 10 ops of 3ns chained, 10ns period -> 3 fit per stage (9ns) -> ceil(10/3)=4 stages
    nodes = [Node(f"n{i}", "add", 8) for i in range(10)]
    edges = [(f"n{i}", f"n{i+1}") for i in range(9)]
    sc = schedule_dfg(nodes, edges, 10.0, delay_fn=_const(3.0))
    assert sc.depth == 4
    assert all(sd <= 10.0 + 1e-9 for sd in sc.stage_delay_ns)


def test_reduction_tree_depth():
    # 8 leaves + balanced add tree (3 levels). Each op 4ns, period 100 -> all chain
    sc = schedule_reduction("add", 8, 8, 100.0, delay_fn=_const(4.0))
    # leaves(1) + 3 tree levels chained within budget -> still 1 stage (huge period)
    assert sc.depth == 1
    # tight period forces the tree across stages
    sc2 = schedule_reduction("add", 8, 8, 5.0, delay_fn=_const(4.0))
    assert sc2.depth >= 2 and all(sd <= 5.0 + 1e-9 for sd in sc2.stage_delay_ns)


def test_uncharacterized_op_flagged():
    sc = schedule_dfg([Node("x", "weirdop", 8)], [], 10.0, delay_fn=lambda op, w: None)
    assert "x" in sc.uncharacterized


def test_contract_text_renders():
    sc = schedule_reduction("add", 8, 4, 5.0, delay_fn=_const(4.0))
    txt = pipeline_contract_text(sc, title="test")
    assert "PIPELINE CONTRACT" in txt and "Stage 0" in txt
