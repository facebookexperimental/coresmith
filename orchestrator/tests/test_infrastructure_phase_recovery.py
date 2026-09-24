# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Infrastructure retries preserve the failed phase and design budget."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from orchestrator.langgraph import pipeline_graph as pg


@pytest.mark.parametrize("phase,action", [
    ("synth", "retry_synth"), ("lint", "ask_human"),
    ("sim", "retry_sim"), ("rtl", "retry_rtl"), ("tb", "retry_tb"),
])
def test_infrastructure_retries_the_failed_phase(phase, action):
    assert pg._route_decision({"category": "INFRASTRUCTURE_ERROR"}, [], 1, 3, phase) == action


def test_infrastructure_honors_explicit_escalation():
    diagnostic = {"category": "INFRASTRUCTURE_ERROR", "confidence": .9,
                  "escalate": True, "needs_human": True,
                  "suggested_fix": "Bank the memory with registered copies of controls. " * 2}
    assert pg._route_decision(diagnostic, [], 1, 3, "synth") == "ask_human"


def test_outer_agent_retry_after_synthesis_does_not_regenerate_rtl():
    assert pg.route_after_human({"phase": "synth", "human_response": {"action": "retry"}}) == "synthesize"


@pytest.mark.asyncio
async def test_tool_retry_preserves_attempt_and_uses_existing_synthesis_node(tmp_path, monkeypatch):
    monkeypatch.setattr(pg, "_db", lambda p: SimpleNamespace(
        diagnosis=lambda b: {"category": "INFRASTRUCTURE_ERROR"},
        attempt_history=lambda b: [{"category": "INFRASTRUCTURE_ERROR"}]))
    wait = AsyncMock()
    monkeypatch.setattr(pg.asyncio, "sleep", wait)
    state = {"project_root": str(tmp_path), "current_block": {"name": "core"},
             "attempt": 1, "max_attempts": 3, "phase": "synth", "debug_action": "retry_synth"}
    update = await pg.decide_node(state)
    assert {**state, **update}["attempt"] == 1
    assert pg.route_decision({**state, **update}) == "synthesize"
    wait.assert_awaited_once()
