# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""A-Fix 3(b): block-complexity review gate node + router in architecture_graph.

Hermetic + deterministic -- no LLM (the estimator/decomposer are pure AST). A
synthetic fat 'intra_rd' golden trips the modeling-complexity axis; a
non-matching block name and a golden-less run pass through as no-ops.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from orchestrator.langgraph import architecture_graph as ag


_FAT_GOLDEN = '''
import numpy as np

def _fdct4(a, b, qp):
    d = a + b
    for i in range(4):
        d = d * 2 + b
    return d

def fdct_quant(a, b, qp):
    return quantize(_fdct4(a, b, qp), b, qp)

def quantize(a, b, qp):
    r = a
    for i in range(16):
        r = (r * 13 + 8) >> 4
    return r

def dequantize(a, b, qp):
    return a * qp + 1

def reconstruct(a, b, qp):
    recY[0] = clip255(dequantize(a, b, qp))
    return recY[0]

def clip255(a):
    return 0 if a < 0 else (255 if a > 255 else a)

def pred_4x4(a, b, qp):
    m = avail_modes_4x4(a, b, qp)
    nbr = recY[0]
    for i in range(9):
        p = a + nbr + i
    return p

def avail_modes_4x4(a, b, qp):
    return [0, 1, 2] if a else [2]

def _rd_cost(a, b, qp):
    c = 0
    for i in range(8):
        c += (a - b) * (a - b)
    return c

def decide_chroma_mode(a, b, qp):
    return chroma_dc_quant(a, b, qp)

def chroma_dc_quant(a, b, qp):
    return (a + b) >> qp

def _encode_mb(a, b, qp):
    t = fdct_quant(a, b, qp)
    r = reconstruct(t, b, qp)
    p = pred_4x4(r, b, qp)
    c = _rd_cost(p, b, qp)
    ch = decide_chroma_mode(a, b, qp)
    return c + ch
'''


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _project_with_golden(tmp_path: Path) -> str:
    (tmp_path / "inputs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "inputs" / "golden.py").write_text(_FAT_GOLDEN, encoding="utf-8")
    return str(tmp_path)


def _state(project_root, blocks, **kw):
    base = {"project_root": project_root, "round": 1,
            "block_diagram": {"blocks": blocks},
            "block_complexity_retries": 0}
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# Env gate helpers
# ---------------------------------------------------------------------------

def test_gate_default_on(monkeypatch):
    monkeypatch.delenv("CORESMITH_COMPLEXITY_GATE", raising=False)
    assert ag._complexity_gate_enabled() is True


def test_gate_can_be_disabled(monkeypatch):
    monkeypatch.setenv("CORESMITH_COMPLEXITY_GATE", "0")
    assert ag._complexity_gate_enabled() is False


def test_max_redecompose_default_and_override(monkeypatch):
    monkeypatch.delenv("CORESMITH_COMPLEXITY_MAX_REDECOMPOSE", raising=False)
    assert ag._complexity_max_redecompose() == 2
    monkeypatch.setenv("CORESMITH_COMPLEXITY_MAX_REDECOMPOSE", "5")
    assert ag._complexity_max_redecompose() == 5
    monkeypatch.setenv("CORESMITH_COMPLEXITY_MAX_REDECOMPOSE", "junk")
    assert ag._complexity_max_redecompose() == 2


# ---------------------------------------------------------------------------
# review_diagram routing (default: complexity gate first)
# ---------------------------------------------------------------------------

def test_clean_routes_to_complexity_review(monkeypatch):
    monkeypatch.delenv("CORESMITH_COMPLEXITY_GATE", raising=False)
    clean = {"block_diagram": {"questions": [], "blocks": [{"name": "a"}]}}
    assert ag.review_diagram(clean) == "Complexity Review"


# ---------------------------------------------------------------------------
# Node: no-op passthroughs
# ---------------------------------------------------------------------------

def test_noop_when_no_golden(tmp_path):
    out = _run(ag.block_complexity_review_node(
        _state(str(tmp_path), [{"name": "intra_rd_encode_core"}])))
    v = out["block_complexity_verdict"]
    assert v["passed"] is True
    assert v["_redecompose"] is False
    assert "block_complexity_retries" not in out
    assert "human_feedback" not in out


def test_noop_for_non_matching_block(tmp_path):
    pr = _project_with_golden(tmp_path)
    # a block name that does not resolve to any golden slice -> no breach
    out = _run(ag.block_complexity_review_node(
        _state(pr, [{"name": "fft_butterfly"}])))
    v = out["block_complexity_verdict"]
    assert v["passed"] is True
    assert v["over_budget_blocks"] == []


# ---------------------------------------------------------------------------
# Node: over-budget -> re-decompose feedback -> Block Diagram
# ---------------------------------------------------------------------------

def test_over_budget_redecomposes(tmp_path):
    pr = _project_with_golden(tmp_path)
    out = _run(ag.block_complexity_review_node(
        _state(pr, [{"name": "intra_rd_encode_core"}],
               block_complexity_retries=0)))
    v = out["block_complexity_verdict"]
    assert v["passed"] is False
    assert v["_redecompose"] is True
    assert out["block_complexity_retries"] == 1
    assert "BLOCK COMPLEXITY" in out["human_feedback"]
    # advisory sub-block proposal names surface in the feedback
    assert "Suggested sub-blocks" in out["human_feedback"]
    assert ag.route_after_block_complexity_review(out) == "Block Diagram"


def test_over_budget_exhausted_proceeds(tmp_path):
    pr = _project_with_golden(tmp_path)
    out = _run(ag.block_complexity_review_node(
        _state(pr, [{"name": "intra_rd_encode_core"}],
               block_complexity_retries=2)))  # == max
    v = out["block_complexity_verdict"]
    assert v["passed"] is False
    assert v["_redecompose"] is False
    assert "block_complexity_retries" not in out   # not incremented past cap
    assert "human_feedback" not in out
    # exhausted -> proceed to the clean-diagram target (output-contract on)
    assert ag.route_after_block_complexity_review(out) == "Output Contract Review"


def test_gate_disabled_review_diagram_skips_node(monkeypatch, tmp_path):
    # With the gate OFF, review_diagram does not route through Complexity Review.
    monkeypatch.setenv("CORESMITH_COMPLEXITY_GATE", "0")
    clean = {"block_diagram": {"questions": [], "blocks": [{"name": "a"}]}}
    # output-contract still on by default -> next gate in the chain
    monkeypatch.delenv("CORESMITH_OUTPUT_CONTRACT_GATE", raising=False)
    assert ag.review_diagram(clean) == "Output Contract Review"


# ---------------------------------------------------------------------------
# Router: clean target follows the output-contract gate
# ---------------------------------------------------------------------------

def test_router_pass_target_respects_output_contract_gate(monkeypatch):
    passed = {"block_complexity_verdict": {"_redecompose": False, "passed": True}}
    monkeypatch.setenv("CORESMITH_OUTPUT_CONTRACT_GATE", "1")
    assert ag.route_after_block_complexity_review(passed) == "Output Contract Review"
    monkeypatch.setenv("CORESMITH_OUTPUT_CONTRACT_GATE", "0")
    assert ag.route_after_block_complexity_review(passed) == "Interface Definition"
