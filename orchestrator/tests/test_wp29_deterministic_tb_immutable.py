"""WP-29: the deterministic BFM testbench is regenerated, never reused, and offers no fix_tb."""
from __future__ import annotations

import inspect

from orchestrator.langgraph import pipeline_graph as pg


def test_integration_dv_regenerates_deterministic_tb():
    src = inspect.getsource(pg.integration_dv_node)
    assert "deterministic_tb_modified" in src and "deterministic_tb_reused" in src
    assert 'get("deterministic_bfm")' in src


def test_integration_failure_offers_no_fix_tb_for_deterministic_bfm():
    src = inspect.getsource(pg.integration_dv_node)
    i = src.find('["retry", "fix_rtl", "revise", "abort"]')
    assert i > 0 and "deterministic_bfm" in src[i - 400:i + 200]
