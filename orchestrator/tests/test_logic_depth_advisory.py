# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Finding 3 (pipeline-campaign-3): logic-depth proxy becomes ADVISORY with PDK.

The ltp logic-level proxy measured 887 levels on the un-synthesizable cloud and
881 on the fully-converged staged design -- it cannot discriminate the class
(LCU carry bits count as levels). Worse, its failure SHORT-CIRCUITED the STA so
``wns_ns`` landed NULL and the fail-loud path then rejected the block for a
measurement the proxy itself prevented. Fix: when a real PDK + STA are available
the proxy is ADVISORY (recorded, never gating, never short-circuits STA -- real
WNS is the authority); PDK-absent runs keep it gating. Env
``CORESMITH_LOGIC_DEPTH_ADVISORY_WITH_PDK`` (default ON).
"""
from __future__ import annotations

import pytest

from orchestrator.langgraph import ppa_check as pc
from orchestrator.langgraph import pipeline_graph as pg


class TestStaToolingAvailable:
    def _synth(self, tmp_path):
        nl = tmp_path / "n.v"; nl.write_text("x")
        sdc = tmp_path / "n.sdc"; sdc.write_text("x")
        lib = tmp_path / "n.lib"; lib.write_text("x")
        return {"netlist_path": str(nl), "sdc_path": str(sdc),
                "liberty_path": str(lib)}

    def test_true_when_sta_and_inputs_present(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pc.shutil, "which", lambda _b: "/usr/bin/sta")
        assert pc.sta_tooling_available(self._synth(tmp_path)) is True

    def test_false_when_sta_binary_absent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pc.shutil, "which", lambda _b: None)
        assert pc.sta_tooling_available(self._synth(tmp_path)) is False

    def test_false_when_no_synth_result(self, monkeypatch):
        monkeypatch.setattr(pc.shutil, "which", lambda _b: "/usr/bin/sta")
        assert pc.sta_tooling_available(None) is False

    def test_false_when_a_required_input_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pc.shutil, "which", lambda _b: "/usr/bin/sta")
        sr = self._synth(tmp_path)
        sr["liberty_path"] = str(tmp_path / "missing.lib")
        assert pc.sta_tooling_available(sr) is False


class TestAdvisoryEnvGate:
    def test_default_on(self, monkeypatch):
        monkeypatch.delenv("CORESMITH_LOGIC_DEPTH_ADVISORY_WITH_PDK", raising=False)
        assert pc.logic_depth_advisory_with_pdk_enabled() is True

    def test_off(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_LOGIC_DEPTH_ADVISORY_WITH_PDK", "0")
        assert pc.logic_depth_advisory_with_pdk_enabled() is False


class TestDepthGuardRouting:
    """Both branches of the depth guard inside ``_evaluate_ppa_gate``: an
    over-max depth is ADVISORY when PDK+STA are present (recorded, not gating,
    STA still runs) and GATING when they are absent."""

    def _rig(self, tmp_path, monkeypatch, *, sta_available):
        rtl = tmp_path / "blk.v"
        rtl.write_text("module blk(input clk); endmodule\n")
        # Elaborates fine; over-max depth; skip the cell probe; STA returns a
        # clean WNS on the advisory path.
        monkeypatch.setattr(pc, "probe_synth_generic",
                            lambda *a, **k: {"elaborated": True,
                                             "logic_ff": 4, "mem_bits": 0})
        monkeypatch.setattr(pc, "synth_cell_gate_enabled", lambda: False)
        monkeypatch.setattr(pc, "logic_depth_gate_enabled", lambda: True)
        monkeypatch.setattr(pc, "max_logic_depth", lambda: 500)
        monkeypatch.setattr(pc, "probe_logic_depth",
                            lambda *a, **k: {"elaborated": True,
                                             "logic_depth": 881})
        monkeypatch.setattr(pc, "logic_depth_advisory_with_pdk_enabled",
                            lambda: True)
        monkeypatch.setattr(pc, "sta_tooling_available",
                            lambda _sr: sta_available)
        monkeypatch.setattr(pc, "run_pre_layout_sta",
                            lambda *a, **k: {"wns_ns": 0.0})
        return str(rtl)

    def test_advisory_when_pdk_and_sta_present(self, tmp_path, monkeypatch):
        rtl = self._rig(tmp_path, monkeypatch, sta_available=True)
        ppa_ok, reasons, meta = pg._evaluate_ppa_gate(
            str(tmp_path), "blk", rtl,
            {"ff_count": 4, "chip_area_um2": None},
            require_gate_flag=False,
        )
        # depth did NOT reject the block; it was recorded as advisory only.
        assert ppa_ok is not False
        assert meta.get("logic_depth_advisory") is True
        assert meta.get("logic_depth") == 881
        assert not any("combinational depth" in r for r in reasons)

    def test_gating_when_pdk_absent(self, tmp_path, monkeypatch):
        rtl = self._rig(tmp_path, monkeypatch, sta_available=False)
        ppa_ok, reasons, meta = pg._evaluate_ppa_gate(
            str(tmp_path), "blk", rtl, None, require_gate_flag=False,
        )
        assert ppa_ok is False
        assert any("combinational depth" in r for r in reasons)
        assert "logic_depth_advisory" not in meta
