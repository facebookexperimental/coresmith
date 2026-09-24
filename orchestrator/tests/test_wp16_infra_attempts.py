"""WP-16: infrastructure failures do not consume the attempt budget."""
from __future__ import annotations

import inspect

from orchestrator.langgraph import pipeline_graph as pg


def test_infra_route_asks_human_only_after_configured_retries(monkeypatch):
    monkeypatch.delenv("CORESMITH_INFRA_MAX_RETRIES", raising=False)
    src = inspect.getsource(pg)
    assert 'CORESMITH_INFRA_MAX_RETRIES' in src
    # the old rule (ask_human on the 2nd infra failure) is gone
    assert 'category_counts.get("INFRASTRUCTURE_ERROR", 0) >= 2' not in src


def test_route_decision_does_not_consume_budget_on_infra():
    src = inspect.getsource(pg.decide_node)
    assert '_diag_cat == "INFRASTRUCTURE_ERROR"' in src
    assert "budget not" in src


def test_chip_lead_prompt_distinguishes_transient_failure_from_exhaustion():
    from orchestrator.langchain.agents.chip_lead_agent import CHIP_LEAD_PROMPT
    assert "transient transport/rate-limit failure is not an RTL verdict" in CHIP_LEAD_PROMPT
    assert "Generation/turn-limit" in CHIP_LEAD_PROMPT
    assert "regardless of how many such attempts repeated" not in CHIP_LEAD_PROMPT
    assert "- `pipeline_incomplete`: `abort`." not in CHIP_LEAD_PROMPT
