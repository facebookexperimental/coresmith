"""WP-20/WP-33: architecture review feedback is bounded; an exhausted budget parks for a human."""
from __future__ import annotations

from orchestrator.langgraph import architecture_graph as ag


def _hist(n, phase="final_review"):
    return [{"phase": phase, "action": "feedback"} for _ in range(n)] + [
        {"phase": "block_diagram", "action": "continue"}]


def test_feedback_rounds_used_counts_only_the_phase():
    assert ag._feedback_rounds_used(_hist(2), "final_review") == 2
    assert ag._feedback_rounds_used(_hist(2), "block_diagram") == 0
    assert ag._feedback_rounds_used(None, "final_review") == 0


def test_cap_parks_for_a_human_never_forges_accept(monkeypatch):
    monkeypatch.setenv("CORESMITH_ARCH_MAX_FINAL_FEEDBACK", "2")
    events, parked = [], []
    monkeypatch.setattr(ag, "_event", lambda st, node, ev, data: events.append((node, ev, data)))
    monkeypatch.setattr(ag, "interrupt", lambda p: parked.append(p) or {"action": "abort", "parked": True})
    st = {"round": 1, "human_response_history": _hist(2)}
    payload = {"type": "final_review", "message": "review me", "supported_actions": ["accept", "feedback", "abort"]}
    r, capped = ag._cap_feedback(st, "final_review", {"action": "feedback", "feedback": "missing pin"},
                                 "Final Review", payload)
    assert capped and r.get("parked") and r["action"] != "accept"
    assert parked and parked[0]["feedback_budget_exhausted"] is True
    assert parked[0]["unresolved_feedback"] == "missing pin"
    assert parked[0]["message"].startswith("Final Review: the feedback budget")
    assert parked[0]["message"].endswith("review me")
    assert events and events[0][1] == "feedback_cap_exhausted"


def test_below_cap_feedback_passes_through(monkeypatch):
    monkeypatch.setenv("CORESMITH_ARCH_MAX_FINAL_FEEDBACK", "2")
    st = {"round": 1, "human_response_history": _hist(1)}
    r, capped = ag._cap_feedback(st, "final_review", {"action": "feedback", "feedback": "x"}, "Final Review")
    assert not capped and r["action"] == "feedback"


def test_block_diagram_cap_parks_too(monkeypatch):
    monkeypatch.setenv("CORESMITH_ARCH_MAX_FINAL_FEEDBACK", "1")
    monkeypatch.setattr(ag, "_event", lambda *a, **k: None)
    parked = []
    monkeypatch.setattr(ag, "interrupt", lambda p: parked.append(p) or {"action": "abort"})
    st = {"round": 1, "human_response_history": _hist(1, "block_diagram")}
    r, capped = ag._cap_feedback(st, "block_diagram", {"action": "feedback", "feedback": "x"},
                                 "Escalate Diagram", {"type": "block_diagram"})
    assert capped and r["action"] != "continue" and parked[0]["feedback_rounds_cap"] == 1


def test_prompts_follow_contract_reset():
    from pathlib import Path
    root = Path(ag.__file__).resolve().parent.parent
    bd = (root / "langchain/prompts/block_diagram.md").read_text()
    ir = (root / "langchain/prompts/integration_review.md").read_text()
    assert "names them, with their polarity" in bd
    assert "declared names and polarity" in ir
