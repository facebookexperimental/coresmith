# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""chip_top gate-sim: replay integration-DV vectors through the FLAT netlist.

The flat chip netlist is the artifact that becomes silicon. Every functional
gate upstream reads RTL and every PPA gate reads a netlist, so nothing
simulated the assembled top. A per-block gate-sim cannot close that gap: a
block netlist is an intermediate, and its stimulus is that block's own
testbench rather than real chip traffic.
"""
from __future__ import annotations

import pytest

import orchestrator.harness.gate_sim as gs
from orchestrator.langgraph.backend_graph import _run_chip_top_gate_sim


@pytest.fixture()
def run_dir(tmp_path):
    (tmp_path / "sim_build" / "integration").mkdir(parents=True)
    (tmp_path / "sim_build" / "integration" / "test_my_chip.py").write_text(
        "# integration testbench\n")
    (tmp_path / "my_chip.v").write_text("module my_chip(input clk);\nendmodule\n")
    return tmp_path


def _state(run_dir, **kw):
    base = {
        "project_root": str(run_dir),
        "design_name": "my_chip",
        "top_rtl_path": str(run_dir / "my_chip.v"),
    }
    base.update(kw)
    return base


class TestDisabledIsRecordedNotSilent:
    def test_gate_off_reports_disabled_with_a_reason(self, run_dir, monkeypatch):
        monkeypatch.setenv(gs.GATE_SIM_ENV, "0")
        ok, status, reason = _run_chip_top_gate_sim(_state(run_dir), "net.v")
        assert ok is None                      # never True when it did not run
        assert status == gs.STATUS_DISABLED
        assert reason                          # absence is always explained


class TestStimulusResolution:
    def test_missing_integration_tb_is_not_run_not_pass(self, tmp_path, monkeypatch):
        """No integration vectors => cannot judge. Must NOT report success."""
        monkeypatch.setenv(gs.GATE_SIM_ENV, "1")
        state = _state(tmp_path)               # no sim_build/integration
        ok, status, reason = _run_chip_top_gate_sim(state, "net.v")
        assert ok is None and status == gs.STATUS_NOT_RUN
        assert "integration" in reason.lower()

    def test_uses_the_integration_testbench_as_reference(self, run_dir, monkeypatch):
        """The stimulus must be integration DV -- real traffic through the
        assembled chip -- not a per-block testbench."""
        monkeypatch.setenv(gs.GATE_SIM_ENV, "1")
        seen = {}

        def fake_check(*, block, netlist_path, rtl_path, tb_path):
            seen.update(block=block, netlist=netlist_path, tb=tb_path)
            return gs.GateSimResult(ran=True, ok=True, status=gs.STATUS_PASS,
                                    reason="ok", cycles_compared=10,
                                    output_bits_compared=100)

        monkeypatch.setattr(gs, "check_gate_sim", fake_check)
        ok, status, _ = _run_chip_top_gate_sim(_state(run_dir), "flat_net.v")
        assert ok is True and status == gs.STATUS_PASS
        assert seen["tb"].endswith("sim_build/integration/test_my_chip.py")
        assert seen["netlist"] == "flat_net.v", "must judge the FLAT netlist"
        # marked chip_top so the scope guard admits it
        assert seen["block"]["is_chip_top"] is True


class TestVerdicts:
    def test_divergence_is_a_hard_failure(self, run_dir, monkeypatch):
        monkeypatch.setenv(gs.GATE_SIM_ENV, "1")
        monkeypatch.setattr(gs, "check_gate_sim", lambda **k: gs.GateSimResult(
            ran=True, ok=False, status=gs.STATUS_FAIL,
            reason="diverged at cycle 42 on port done"))
        ok, status, reason = _run_chip_top_gate_sim(_state(run_dir), "n.v")
        assert ok is False and status == gs.STATUS_FAIL and "cycle 42" in reason

    def test_plumbing_error_never_crashes_the_backend(self, run_dir, monkeypatch):
        """A gate that explodes must not take the backend down -- nor pass."""
        monkeypatch.setenv(gs.GATE_SIM_ENV, "1")

        def boom(**_kw):
            raise RuntimeError("verilator exploded")

        monkeypatch.setattr(gs, "check_gate_sim", boom)
        ok, status, reason = _run_chip_top_gate_sim(_state(run_dir), "n.v")
        assert ok is None and status == gs.STATUS_NOT_RUN
        assert "RuntimeError" in reason

    def test_not_run_from_the_harness_is_propagated_as_none(self, run_dir, monkeypatch):
        monkeypatch.setenv(gs.GATE_SIM_ENV, "1")
        monkeypatch.setattr(gs, "check_gate_sim", lambda **k: gs.GateSimResult(
            ran=False, ok=True, status=gs.STATUS_NOT_RUN,
            reason="no verilator on PATH"))
        ok, status, _ = _run_chip_top_gate_sim(_state(run_dir), "n.v")
        assert ok is None and status == gs.STATUS_NOT_RUN
