"""WP-19: an acceptance failure parks for the chip lead instead of failing signoff."""
from __future__ import annotations

import inspect

from orchestrator.langgraph import pipeline_graph as pg


def test_acceptance_failure_returns_pending_decision():
    src = inspect.getsource(pg.validation_dv_node)
    i = src.find("[ACCEPTANCE-DV] FAILED")
    assert i > 0
    tail = src[i:i + 12000]   # the park block grew with WP-38/41/45/47 guidance
    assert '"pending_decision": True' in tail
    assert '"type": "validation_dv_failure"' in tail
    assert '"phase": "acceptance_dv"' in tail
    assert "ACCEPTANCE_DV_FAILURE" in tail


def test_route_parks_on_pending_decision():
    assert pg.route_after_validation_dv({"validation_dv_result": {
        "passed": False, "pending_decision": True}}) == "validation_dv_decision"
