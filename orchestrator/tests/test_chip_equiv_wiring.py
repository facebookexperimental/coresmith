# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""A-Fix 5(d): chip-top RTL-vs-model equivalence wired into integration_dv_node.

Hermetic -- the integration sim and the equivalence gate are BOTH monkeypatched
(no verilator). The load-bearing assertion: a loosened/operator-edited testbench
that makes the sim PASS is no longer sufficient -- an equivalence FAIL flips the
DV result to failed and fires the existing integration_dv failure interrupt.

Tests that would actually invoke verilator are marked requires_nix (there are
none here by design -- the equivalence function is monkeypatched).
"""

from __future__ import annotations

import pytest

from orchestrator.langgraph import pipeline_graph
from orchestrator.langgraph.integration_helpers import VerilogModule, VerilogPort

# ---------------------------------------------------------------------------
# Env gate + seed helpers
# ---------------------------------------------------------------------------

def test_chip_equiv_gate_default_on(monkeypatch):
    monkeypatch.delenv("CORESMITH_CHIP_EQUIV", raising=False)
    assert pipeline_graph._chip_equiv_enabled() is True
    monkeypatch.setenv("CORESMITH_CHIP_EQUIV", "0")
    assert pipeline_graph._chip_equiv_enabled() is False


def test_resolve_equiv_seed_pinned(monkeypatch):
    monkeypatch.setenv("CORESMITH_DV_SEED_PIN", "4242")
    assert pipeline_graph._resolve_equiv_seed() == 4242
    monkeypatch.delenv("CORESMITH_DV_SEED_PIN", raising=False)
    assert isinstance(pipeline_graph._resolve_equiv_seed(), int)


# ---------------------------------------------------------------------------
# _maybe_run_chip_equiv gating (real function, early-return paths only)
# ---------------------------------------------------------------------------

def test_maybe_run_chip_equiv_none_when_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("CORESMITH_CHIP_EQUIV", "0")
    assert pipeline_graph._maybe_run_chip_equiv(
        str(tmp_path), "d", "/x/top.v", {"a": "/x/a.v"}) is None


def test_maybe_run_chip_equiv_none_when_block_goldens_off(monkeypatch, tmp_path):
    # legacy profile (conftest pin) -> block_goldens off -> gate no-ops
    monkeypatch.setenv("CORESMITH_CHIP_EQUIV", "1")
    monkeypatch.delenv("CORESMITH_BLOCK_GOLDENS", raising=False)
    monkeypatch.setenv("CORESMITH_PROFILE", "legacy")
    assert pipeline_graph._maybe_run_chip_equiv(
        str(tmp_path), "d", "/x/top.v", {"a": "/x/a.v"}) is None


def test_maybe_run_chip_equiv_none_when_no_chip_model(monkeypatch, tmp_path):
    monkeypatch.setenv("CORESMITH_CHIP_EQUIV", "1")
    monkeypatch.setenv("CORESMITH_BLOCK_GOLDENS", "1")
    # no arch/block_models/_chip_model.py exists under tmp_path
    assert pipeline_graph._maybe_run_chip_equiv(
        str(tmp_path), "d", "/x/top.v", {"a": "/x/a.v"}) is None


# ---------------------------------------------------------------------------
# integration_dv_node wiring (sim + equiv both monkeypatched)
# ---------------------------------------------------------------------------

def _setup_node(monkeypatch, tmp_path, *, sim_passed, equiv_result):
    """Wire integration_dv_node's seams for a reuse-TB (no LLM) run."""
    top = tmp_path / "rtl" / "integration" / "chip_top.v"
    top.parent.mkdir(parents=True, exist_ok=True)
    top.write_text("module chip_top(input clk); endmodule\n")
    tb = tmp_path / "tb" / "chip_tb.py"
    tb.parent.mkdir(parents=True, exist_ok=True)
    tb.write_text("# tb\n")
    blk = tmp_path / "rtl" / "a.v"
    blk.write_text("module a(input clk); endmodule\n")

    monkeypatch.setattr(
        pipeline_graph, "load_architecture_connections", lambda pr: ([], "chip"))
    monkeypatch.setattr(
        pipeline_graph, "parse_verilog_ports",
        lambda p: VerilogModule(name="a", ports=[VerilogPort("clk", "input")]))
    monkeypatch.setattr(
        pipeline_graph, "run_integration_simulation",
        lambda *a, **k: {"passed": sim_passed, "log": "sim ran", "log_path": ""})
    monkeypatch.setattr(
        pipeline_graph, "_maybe_run_chip_equiv",
        lambda *a, **k: equiv_result)

    async def _fake_audit(**kw):
        return {"category": "TESTBENCH", "outer_agent_summary": "x", "audit_path": ""}
    monkeypatch.setattr(pipeline_graph, "_run_top_level_contract_audit", _fake_audit)
    monkeypatch.setattr(pipeline_graph, "write_graph_event", lambda *a, **k: None)

    state = {
        "project_root": str(tmp_path),
        "integration_result": {
            "top_rtl_path": str(top),
            "design_name": "chip",
            "block_rtl_paths": {"a": str(blk)},
        },
        # reuse existing TB (action fix_tb) so no LLM testbench generation runs
        "integration_dv_result": {
            "action_taken": "fix_tb",
            "testbench_path": str(tb),
            "test_count": 1,
        },
    }
    return state


@pytest.mark.asyncio
async def test_loosened_tb_pass_but_equiv_fail_flips_dv_to_failed(monkeypatch, tmp_path):
    captured = {}

    def fake_interrupt(payload):
        captured.update(payload)
        return {"action": "abort"}

    monkeypatch.setattr(pipeline_graph, "interrupt", fake_interrupt)
    state = _setup_node(
        monkeypatch, tmp_path,
        sim_passed=True,
        equiv_result={"passed": False, "skipped": False,
                      "reason": "BYTE DIVERGENCE at offset 3: rtl=0x1 model=0x2"},
    )
    result = await pipeline_graph.integration_dv_node(state)

    # The sim passed but the equivalence gate failed -> DV is failed and the
    # existing failure interrupt fired with the equivalence divergence in the log.
    assert captured.get("type") == "integration_dv_failure"
    assert "EQUIVALENCE FAILED" in captured.get("sim_log", "")
    dv = result["integration_dv_result"]
    assert dv["passed"] is False
    assert dv["aborted"] is True


@pytest.mark.asyncio
async def test_equiv_pass_lets_dv_pass(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline_graph, "interrupt",
                        lambda p: (_ for _ in ()).throw(AssertionError("no interrupt")))
    state = _setup_node(
        monkeypatch, tmp_path, sim_passed=True,
        equiv_result={"passed": True, "skipped": False, "reason": "byte-exact"},
    )
    result = await pipeline_graph.integration_dv_node(state)
    dv = result["integration_dv_result"]
    assert dv["passed"] is True
    assert dv["chip_equiv_result"]["passed"] is True


@pytest.mark.asyncio
async def test_equiv_honest_skip_keeps_dv_pass(monkeypatch, tmp_path):
    state = _setup_node(
        monkeypatch, tmp_path, sim_passed=True,
        equiv_result={"passed": False, "skipped": True,
                      "reason": "chip top is not the single s_axis/m_axis shape"},
    )
    result = await pipeline_graph.integration_dv_node(state)
    dv = result["integration_dv_result"]
    assert dv["passed"] is True


@pytest.mark.asyncio
async def test_equiv_not_run_when_gate_off(monkeypatch, tmp_path):
    # equiv_result=None means _maybe_run_chip_equiv would have returned None
    # (gate off / not applicable) -> a passing sim stays a passing DV.
    state = _setup_node(monkeypatch, tmp_path, sim_passed=True, equiv_result=None)
    result = await pipeline_graph.integration_dv_node(state)
    assert result["integration_dv_result"]["passed"] is True


# ---------------------------------------------------------------------------
# check_chip_model_equivalence honest-skip (no verilator build)
# ---------------------------------------------------------------------------

def test_chip_equiv_honest_skip_non_axis_top(tmp_path):
    # A top with no s_axis/m_axis ports -> honest skip BEFORE any build.
    from orchestrator.langgraph import rtl_model_equiv as rme
    top = tmp_path / "top.v"
    top.write_text("module top(input clk, input rst_n, output [7:0] q); endmodule\n")
    res = rme.check_chip_model_equivalence(
        "d", str(top), {}, "", project_root=str(tmp_path), seed=1)
    assert res["skipped"] is True
    assert not res.get("harness_error")


def test_coerce_int_stream():
    from orchestrator.langgraph.rtl_model_equiv import _coerce_int_stream
    assert _coerce_int_stream(([1, 2, 300], 12), 0xFF) == [1, 2, 44]
    assert _coerce_int_stream(bytes([1, 2, 3]), 0xFF) == [1, 2, 3]
    assert _coerce_int_stream([5, 6], 0xFF) == [5, 6]
    assert _coerce_int_stream({"a": 1}, 0xFF) is None
    assert _coerce_int_stream([], 0xFF) is None
