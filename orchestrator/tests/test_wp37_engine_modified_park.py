"""WP-37: a modified engine checkout parks the run; nothing is reverted."""
from __future__ import annotations

import inspect

from orchestrator.langgraph import pipeline_graph as pg


def test_guard_never_reverts():
    """The guard's only subprocess is `git status`; no checkout/clean argv literals remain."""
    src = inspect.getsource(pg._engine_checkout_guard)
    assert '"status"' in src
    assert '"checkout"' not in src and '"clean"' not in src and "-fdq" not in src


def test_engine_modified_payload_discards_decision_and_names_paths():
    p = pg._engine_modified_payload({"type": "block_failure", "message": "orig", "supported_actions": ["retry"]},
                                    ["orchestrator/langgraph/x.py", "orchestrator/tests/t.py"])
    assert p["engine_modified"] == ["orchestrator/langgraph/x.py", "orchestrator/tests/t.py"]
    assert p["message"].startswith("ENGINE CHECKOUT MODIFIED") and p["message"].endswith("orig")
    assert p["type"] == "block_failure" and p["supported_actions"] == ["retry"]


def test_resolve_interrupt_parks_on_dirty_engine():
    src = inspect.getsource(pg._resolve_interrupt)
    assert "_dirty = _engine_checkout_guard()" in src
    assert "return _park(_engine_modified_payload(payload, _dirty))" in src
