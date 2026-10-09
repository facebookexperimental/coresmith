"""WP-37, after the Architect-CLI simplification: the graph never decides an
interrupt (it only parks), and the engine-checkout guard that once tripped an
engine-owned on-call Architect is gone with its only caller. The engine's
boundary is the park: nothing in the graph reverts, cleans or inspects its own
checkout."""
from __future__ import annotations

import inspect

from orchestrator.langgraph import pipeline_graph as pg


def test_resolve_interrupt_never_decides():
    src = inspect.getsource(pg._resolve_interrupt)
    assert "return _park(payload)" in src
    assert "decide" not in src.split('"""')[-1]
    assert not hasattr(pg, "_engine_modified_payload")


def test_engine_checkout_guard_is_gone_with_its_caller():
    """No engine code reads ``CORESMITH_ENGINE_READONLY`` or runs git against
    its own checkout: a checkout the worker cannot write is the real boundary."""
    assert not hasattr(pg, "_engine_checkout_guard")
    src = inspect.getsource(pg)
    assert "CORESMITH_ENGINE_READONLY" not in src
    assert '"checkout"' not in src and "-fdq" not in src
