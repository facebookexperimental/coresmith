# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the latency audit -- grounding the model's cycle count.

Covers: parsing the declared STAGE_BUDGET, reconciling stage sums against the
declared total, pricing each stage's per-cycle op chain against the PDK delay
model (feasible vs over-period = a cloud), uncharacterized fall-through, block
model location, the RTL stage-map render, and fail-open behaviour.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.langgraph import latency_audit as la


# A small, well-formed model declaration: 10 + 9*(5+16) = 199 cycles.
GOOD_SOURCE = '''
"""a stage-aligned block model"""
STAGE_BUDGET = [
    {"name": "prefetch", "latency_cycles": 10, "iters": 1, "ops": [],
     "rationale": "registered neighbor read schedule"},
    {"name": "quant", "latency_cycles": 5, "iters": 9, "ops": ["mul16", "add16"],
     "rationale": "4-lane quant mul+add"},
    {"name": "entropy", "latency_cycles": 16, "iters": 9, "ops": ["add16"],
     "rationale": "16-cycle bit-count loop"},
]
DECLARED_LATENCY_CYCLES = 199
'''

# A cloud: one stage chains a huge pile of ops in a single cycle.
CLOUD_SOURCE = '''
STAGE_BUDGET = [
    {"name": "rd_search", "latency_cycles": 1, "iters": 1,
     "ops": ["mul32", "mul32", "mul32", "mul32", "add32", "add32", "add32",
             "add32", "add32", "add32", "cmp32", "cmp32"],
     "rationale": "evaluate everything in one cycle (WRONG)"},
]
DECLARED_LATENCY_CYCLES = 1
'''

# Declares 280 but stages only sum to 50 -> must NOT reconcile.
MISMATCH_SOURCE = '''
STAGE_BUDGET = [
    {"name": "s0", "latency_cycles": 50, "iters": 1, "ops": ["add16"]},
]
DECLARED_LATENCY_CYCLES = 280
'''

# Deterministic delay model for tests: ns scales with width/1000; sad cheap.
def _dfn(op, width, pdk=None):
    if op == "sad":
        return 0.5
    return width / 1000.0   # mul32 -> 0.032 ns ... tiny, always fits


@pytest.fixture(autouse=True)
def _fast_delay(monkeypatch):
    # price against a tiny deterministic model so tests don't need a real PDK
    monkeypatch.setattr(la, "_predict", _dfn)


def test_parse_stage_budget():
    stages, declared = la.parse_stage_budget(GOOD_SOURCE)
    assert declared == 199
    assert [s["name"] for s in stages] == ["prefetch", "quant", "entropy"]
    assert stages[1]["ops"] == ["mul16", "add16"]


def test_parse_missing_is_empty():
    stages, declared = la.parse_stage_budget("x = 1\n")
    assert stages == [] and declared is None


def test_parse_does_not_execute_code():
    # arbitrary code in the model must never run during parsing
    src = "import os\nos.environ['PWNED']='1'\nSTAGE_BUDGET=[]\n"
    la.parse_stage_budget(src)
    import os
    assert os.environ.get("PWNED") != "1"


def test_reconcile_good():
    r = la.audit_source(GOOD_SOURCE)
    assert r.parsed
    assert r.summed_latency == 10 + 9 * 5 + 9 * 16  # 199
    assert r.reconciles
    assert not r.infeasible
    assert r.ok


def test_declared_latency_expression_form_parses():
    """DECLARED_LATENCY_CYCLES as an EXPRESSION (the form the prompt teaches)
    must parse -- ast.literal_eval rejected it, silently disabling reconcile."""
    src = (
        'STAGE_BUDGET=[{"name":"s","latency_cycles":5,"iters":2,"ops":["add16"]}]\n'
        'DECLARED_LATENCY_CYCLES = 5*2 + 0   # expression, not a literal\n'
    )
    stages, declared = la.parse_stage_budget(src)
    assert declared == 10, "expression-form declared latency must evaluate"


def test_reconcile_fires_through_expression():
    """The reconcile MISMATCH guard must actually engage when the declared
    total is an expression that disagrees with the stage sum."""
    src = (
        'STAGE_BUDGET=[{"name":"s","latency_cycles":5,"iters":2,"ops":["add16"]}]\n'
        'DECLARED_LATENCY_CYCLES = 3 + 4   # = 7, but stages sum to 10\n'
    )
    r = la.audit_source(src)
    assert r.declared_latency == 7
    assert r.summed_latency == 10
    assert not r.reconciles, "mismatch via expression must be caught, not no-op'd"


def test_const_eval_rejects_code():
    """_eval_const_int must never execute names/calls -- only arithmetic."""
    import ast as _ast
    assert la._eval_const_int(_ast.parse("2 + 3*4", mode="eval").body) == 14
    assert la._eval_const_int(_ast.parse("-5", mode="eval").body) == -5
    assert la._eval_const_int(_ast.parse("len([1,2])", mode="eval").body) is None
    assert la._eval_const_int(_ast.parse("x + 1", mode="eval").body) is None
    assert la._eval_const_int(_ast.parse("1 // 0", mode="eval").body) is None


def test_reconcile_mismatch_flagged():
    r = la.audit_source(MISMATCH_SOURCE)
    assert r.summed_latency == 50
    assert r.declared_latency == 280
    assert not r.reconciles
    assert not r.ok


def test_cloud_stage_flagged_infeasible():
    # with a model where each op is ~0.5 ns and 20 ns period, the tiny test
    # delay never trips; force a fat per-op delay so the chain exceeds period
    import orchestrator.langgraph.latency_audit as mod

    def _fat(op, width, pdk=None):
        return 3.0  # 12 ops * 3 ns = 36 ns > 20 ns period
    mod._predict = _fat
    try:
        r = la.audit_source(CLOUD_SOURCE, target_clock_mhz=50.0)
        assert "rd_search" in r.infeasible
        assert not r.ok
    finally:
        mod._predict = _dfn


def test_uncharacterized_ops_are_noted_not_crashed():
    src = (
        'STAGE_BUDGET=[{"name":"s","latency_cycles":1,"iters":1,'
        '"ops":["frobnicate8","add16"]}]\nDECLARED_LATENCY_CYCLES=1\n'
    )
    r = la.audit_source(src)
    assert r.parsed
    assert "frobnicate8" in r.stages[0].uncharacterized_ops


def test_stage_map_render_contains_stages_and_total():
    r = la.audit_source(GOOD_SOURCE)
    txt = la.format_stage_map(r, title="enc")
    assert "PIPELINE STAGE MAP" in txt
    assert "quant" in txt and "entropy" in txt
    assert "reconciles" in txt
    assert "registered" in txt.lower()


def test_stage_map_empty_when_no_budget():
    r = la.audit_source("x=1\n")
    assert la.format_stage_map(r) == ""


def test_find_block_model_and_fragment(tmp_path):
    bm = tmp_path / "arch" / "block_models"
    bm.mkdir(parents=True)
    (bm / "intra4x4_rd_encode_core.py").write_text(GOOD_SOURCE)
    found = la.find_block_model(tmp_path, "intra4x4_rd_encode_core")
    assert found is not None and found.name == "intra4x4_rd_encode_core.py"
    frag = la.stage_map_fragment(tmp_path, "intra4x4_rd_encode_core")
    assert "PIPELINE STAGE MAP" in frag and "quant" in frag


def test_find_block_model_token_overlap(tmp_path):
    bm = tmp_path / "arch" / "block_models"
    bm.mkdir(parents=True)
    (bm / "id_decode_regfile_stage.py").write_text(GOOD_SOURCE)
    # block name without the "_stage" suffix still resolves by token overlap
    found = la.find_block_model(tmp_path, "id_decode_regfile")
    assert found is not None


def test_fragment_fail_open_on_missing(tmp_path):
    # no arch/block_models at all -> '' , never raises
    assert la.stage_map_fragment(tmp_path, "whatever") == ""


def test_audit_enabled_env(monkeypatch):
    monkeypatch.delenv("CORESMITH_LATENCY_AUDIT", raising=False)
    assert la.audit_enabled() is False
    monkeypatch.setenv("CORESMITH_LATENCY_AUDIT", "1")
    assert la.audit_enabled() is True
