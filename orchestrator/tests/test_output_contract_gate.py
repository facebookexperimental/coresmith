# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Output-contract ownership gate: verdict normalization, node re-decompose
loop-cap, and routing. The LLM critic is mocked so these are hermetic."""

from __future__ import annotations

import asyncio

from orchestrator.langchain.agents.output_contract_review_agent import (
    OutputContractReviewAgent,
)
from orchestrator.langgraph import architecture_graph as ag


def test_normalize_orphan_forces_fail():
    n = OutputContractReviewAgent._normalize
    assert n({"passed": True, "orphaned_properties": [{"property": "container"}]})["passed"] is False
    assert n({"passed": True, "orphaned_properties": []})["passed"] is True
    assert n({"passed": False, "orphaned_properties": []})["passed"] is False


def _state(**kw):
    base = {"project_root": ".", "requirements": "", "round": 1,
            "block_diagram": {"blocks": [{"name": "a"}], "global_output_contract": None},
            "output_contract_retries": 0}
    base.update(kw)
    return base


def _run(coro):
    return asyncio.run(coro)


def _mock_agent(monkeypatch, *, passed):
    async def fake_review(self, **kw):
        return {"passed": passed,
                "orphaned_properties": ([] if passed else [{"property": "container",
                        "suggested_owner": "NEW: serializer", "suggested_interface": "cfg"}]),
                "summary": "x", "feedback_for_redecomposition": "add serializer"}
    monkeypatch.setattr(OutputContractReviewAgent, "review", fake_review)


def test_node_passes_clean(monkeypatch):
    _mock_agent(monkeypatch, passed=True)
    out = _run(ag.output_contract_review_node(_state()))
    assert out["output_contract_verdict"]["passed"] is True
    assert out["output_contract_verdict"]["_redecompose"] is False
    assert ag.route_after_output_contract_review(out) == "Interface Definition"


def test_node_redecomposes_then_caps(monkeypatch):
    _mock_agent(monkeypatch, passed=False)
    # tries 0 -> re-decompose (feedback + increment to 1)
    out0 = _run(ag.output_contract_review_node(_state(output_contract_retries=0)))
    assert out0["output_contract_verdict"]["_redecompose"] is True
    assert out0["output_contract_retries"] == 1
    assert "OUTPUT-CONTRACT OWNERSHIP" in out0["human_feedback"]
    assert ag.route_after_output_contract_review(out0) == "Block Diagram"
    # tries == max (2) -> exhausted: proceed, NO loop
    out2 = _run(ag.output_contract_review_node(_state(output_contract_retries=2)))
    assert out2["output_contract_verdict"]["_redecompose"] is False
    assert "output_contract_retries" not in out2  # not incremented past cap
    assert ag.route_after_output_contract_review(out2) == "Interface Definition"


def test_gate_env_toggle(monkeypatch):
    # Isolate the output-contract gate from the complexity gate (which, when
    # ON, would take precedence on the clean-diagram route).
    monkeypatch.setenv("CORESMITH_COMPLEXITY_GATE", "0")
    monkeypatch.delenv("CORESMITH_OUTPUT_CONTRACT_GATE", raising=False)
    assert ag._output_contract_gate_enabled() is True   # default ON
    monkeypatch.setenv("CORESMITH_OUTPUT_CONTRACT_GATE", "0")
    assert ag._output_contract_gate_enabled() is False
    # when disabled, a clean diagram routes straight to Interface Definition
    clean = {"block_diagram": {"questions": [], "blocks": [{"name": "a"}]}}
    assert ag.review_diagram(clean) == "Interface Definition"
    monkeypatch.setenv("CORESMITH_OUTPUT_CONTRACT_GATE", "1")
    assert ag.review_diagram(clean) == "Output Contract Review"
