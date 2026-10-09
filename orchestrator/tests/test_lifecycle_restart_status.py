# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Persisted pending nodes remain resumable across process startup."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import TypedDict

import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from orchestrator.graph_lifecycle import (
    GraphLifecycle,
    _checkpoint_recovery_status,
)


def test_interrupt_has_priority_over_pending_next_node():
    state = SimpleNamespace(
        values={"phase": "drc"},
        next=("drc",),
        tasks=(SimpleNamespace(interrupts=(object(),)),),
    )

    assert _checkpoint_recovery_status(state) == "interrupted"


def test_pending_next_node_recovers_paused_not_done():
    state = SimpleNamespace(
        values={"phase": "pnr"}, next=("drc",), tasks=(),
    )

    assert _checkpoint_recovery_status(state) == "paused"


def test_only_terminal_checkpoint_recovers_done():
    terminal = SimpleNamespace(values={"phase": "complete"}, next=(), tasks=())
    empty = SimpleNamespace(values={}, next=(), tasks=())

    assert _checkpoint_recovery_status(terminal) == "done"
    assert _checkpoint_recovery_status(empty) is None


class _State(TypedDict):
    count: int


@pytest.mark.asyncio
async def test_real_langgraph_pending_checkpoint_recovers_paused(
    tmp_path, monkeypatch,
):
    """A process restart after node A must retain scheduled node B."""

    module_name = "_test_pending_restart_graph"
    module = ModuleType(module_name)

    def build_graph(checkpointer=None):
        graph = StateGraph(_State)
        graph.add_node("first", lambda state: {"count": state["count"] + 1})
        graph.add_node("second", lambda state: {"count": state["count"] + 1})
        graph.add_edge(START, "first")
        graph.add_edge("first", "second")
        graph.add_edge("second", END)
        return graph.compile(
            checkpointer=checkpointer, interrupt_after=["first"]
        )

    module.build_graph = build_graph
    monkeypatch.setitem(sys.modules, module_name, module)
    db = str(tmp_path / "checkpoint.db")
    config = {"configurable": {"thread_id": "pending-test"}}

    original = GraphLifecycle(
        "pending-test", db, module_name, "build_graph", str(tmp_path)
    )
    await original.ensure_graph()
    await original.graph.ainvoke({"count": 0}, config)
    before = await original.graph.aget_state(config)
    assert before.next == ("second",)
    await original.cleanup()

    restarted = GraphLifecycle(
        "pending-test", db, module_name, "build_graph", str(tmp_path)
    )
    try:
        await restarted.ensure_graph()
        after = await restarted.graph.aget_state(config)
        assert after.next == ("second",)
        assert restarted.status == "paused"
    finally:
        await restarted.cleanup()


@pytest.mark.asyncio
async def test_historical_restart_forks_latest_checkpoint_before_interrupt(
    tmp_path, monkeypatch,
):
    """A re-run park must become the thread's latest resumable checkpoint."""

    module_name = "_test_historical_restart_interrupt_graph"
    module = ModuleType(module_name)
    should_park = {"value": False}

    def build_graph(checkpointer=None):
        graph = StateGraph(_State)
        graph.add_node("first", lambda state: {"count": state["count"] + 1})

        def second(state):
            if should_park["value"]:
                interrupt({"type": "postcondition", "supported_actions": ["retry"]})
            return {"count": state["count"] + 1}

        graph.add_node("second", second)
        graph.add_edge(START, "first")
        graph.add_edge("first", "second")
        graph.add_edge("second", END)
        return graph.compile(checkpointer=checkpointer)

    module.build_graph = build_graph
    monkeypatch.setitem(sys.modules, module_name, module)
    lifecycle = GraphLifecycle(
        "restart-test", str(tmp_path / "checkpoint.db"), module_name,
        "build_graph", str(tmp_path),
    )
    config = {"configurable": {"thread_id": "restart-test"}}
    try:
        await lifecycle.ensure_graph()
        await lifecycle.graph.ainvoke({"count": 0}, config)
        assert (await lifecycle.graph.aget_state(config)).values["count"] == 2

        should_park["value"] = True
        result = await lifecycle.restart_from_node("second")
        assert result["restarted"] is True
        await lifecycle.task

        parked = await lifecycle.graph.aget_state(config)
        assert lifecycle.status == "interrupted"
        assert len(parked.tasks) == 1
        assert len(parked.tasks[0].interrupts) == 1

        intr = parked.tasks[0].interrupts[0]
        await lifecycle.graph.ainvoke(Command(resume={intr.id: {"action": "retry"}}), config)
        done = await lifecycle.graph.aget_state(config)
        assert done.next == ()
        assert done.tasks == ()
        assert done.values["count"] == 2
    finally:
        await lifecycle.cleanup()
