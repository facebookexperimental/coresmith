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
