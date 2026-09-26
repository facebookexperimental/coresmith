# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""A timing-only synth failure goes to the timing-closure loop (A1c)."""
from __future__ import annotations

import asyncio

from orchestrator.langgraph import pipeline_graph as pg


def _state(tmp_path, **over):
    rtl = tmp_path / "rtl" / "core.v"
    rtl.parent.mkdir(parents=True, exist_ok=True)
    rtl.write_text("module core(input clk, input a, output reg q);\n"
                   "always @(posedge clk) q <= a;\nendmodule\n")
    tb = tmp_path / "tb" / "test_core.py"
    tb.parent.mkdir(parents=True, exist_ok=True)
    tb.write_text("# tb\n")
    rpt = tmp_path / "core_sta.rpt"
    rpt.write_text("Startpoint: a\nEndpoint: q\nslack (VIOLATED) -27.58\n")
    bdir = tmp_path / ".coresmith" / "blocks" / "core"
    bdir.mkdir(parents=True, exist_ok=True)
    (bdir / "constraints.json").write_text("[]")
    (bdir / "previous_error.txt").write_text("PPA gate: WNS -27.58 ns")
    st = {"project_root": str(tmp_path), "target_clock_mhz": 64.0, "max_attempts": 5,
          "current_block": {"name": "core", "rtl_target": "rtl/core.v"},
          "attempt": 1, "phase": "synth", "rtl_path": str(rtl), "tb_path": str(tb),
          "sim_passed": True, "synth_success": True, "timing_ok": False,
          "timing_required": True, "sta_report_path": str(rpt), "timing_fix_attempts": 0}
    st.update(over)
    return st


class _Agent:
    calls: list = []

    def __init__(self, result):
        self._r = result

    async def fix_timing(self, **kw):
        _Agent.calls.append(kw)
        return self._r


def _patch_agent(monkeypatch, result):
    import orchestrator.langchain.agents.timing_closure as tc
    monkeypatch.setattr(tc, "TimingClosureAgent", lambda *a, **k: _Agent(result))


class TestDiagnoseFastPath:
    def test_timing_only_failure_routes_to_timing_fix(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_TIMING_FIX", "1")
        out = asyncio.run(pg.diagnose_node(_state(tmp_path)))
        assert out["debug_action"] == "retry_rtl_timing"
        assert pg.route_decision({"debug_action": "retry_rtl_timing"}) == "timing_fix"
        diag = pg._db(str(tmp_path)).diagnosis("core")
        assert diag["category"] == "TIMING_VIOLATION"

    def test_cap_reached_falls_back_to_the_debug_agent(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_TIMING_FIX_MAX", "1")

        async def _diag(**kw):
            return {"category": "UNKNOWN", "confidence": 0.5, "needs_human": False,
                    "escalate": False, "constraints": [], "affected_blocks": []}
        monkeypatch.setattr(pg, "diagnose_failure", _diag)
        out = asyncio.run(pg.diagnose_node(_state(tmp_path, timing_fix_attempts=1)))
        assert out["debug_action"] != "retry_rtl_timing"

    def test_flag_off_uses_the_debug_agent(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_TIMING_FIX", "0")

        async def _diag(**kw):
            return {"category": "UNKNOWN", "confidence": 0.5, "needs_human": False,
                    "escalate": False, "constraints": [], "affected_blocks": []}
        monkeypatch.setattr(pg, "diagnose_failure", _diag)
        out = asyncio.run(pg.diagnose_node(_state(tmp_path)))
        assert out["debug_action"] != "retry_rtl_timing"


class TestTimingFixNode:
    def test_repaired_rtl_is_written_relinted_resimulated_and_resynthesized(
            self, tmp_path, monkeypatch):
        fixed = ("module core(input clk, input a, output reg q);\n"
                 "reg s; always @(posedge clk) begin s <= a; q <= s; end\nendmodule")
        _patch_agent(monkeypatch, {"verilog": fixed, "strategy": "ADD_PIPELINE_STAGE",
                                   "interface_changed": False, "escalate": False})
        monkeypatch.setattr(pg, "lint_rtl", lambda *a, **k: {"clean": True})
        sims = []
        monkeypatch.setattr(pg, "run_simulation",
                            lambda *a, **k: sims.append(a) or {"passed": True})
        st = _state(tmp_path)
        out = asyncio.run(pg.timing_fix_node(st))
        assert (tmp_path / "rtl" / "core.v").read_text().strip() == fixed
        assert sims, "DV must re-run on the repaired RTL"
        assert out["timing_fix_attempts"] == 1
        assert pg.route_after_timing_fix(out) == "synthesize"
        assert _Agent.calls[-1]["sta_report"].startswith("Startpoint")
        assert _Agent.calls[-1]["target_clock_mhz"] == 64.0

    def test_interface_change_escalates(self, tmp_path, monkeypatch):
        _patch_agent(monkeypatch, {"verilog": "module core; endmodule", "strategy": "ADD_PIPELINE_STAGE",
                                   "interface_changed": True, "escalate": False})
        st = _state(tmp_path)
        before = (tmp_path / "rtl" / "core.v").read_text()
        out = asyncio.run(pg.timing_fix_node(st))
        assert out["debug_action"] == "escalate"
        assert pg.route_after_timing_fix(out) == "block_done"
        assert (tmp_path / "rtl" / "core.v").read_text() == before  # untouched
        assert "interface" in (tmp_path / ".coresmith/blocks/core/previous_error.txt").read_text()

    def test_dv_regression_routes_to_diagnose(self, tmp_path, monkeypatch):
        _patch_agent(monkeypatch, {"verilog": "module core(input clk, input a, output reg q); endmodule",
                                   "strategy": "X", "interface_changed": False, "escalate": False})
        monkeypatch.setattr(pg, "lint_rtl", lambda *a, **k: {"clean": True})
        monkeypatch.setattr(pg, "run_simulation", lambda *a, **k: {"passed": False, "log": "boom"})
        out = asyncio.run(pg.timing_fix_node(_state(tmp_path)))
        assert out["phase"] == "sim" and out["sim_passed"] is False
        assert pg.route_after_timing_fix(out) == "diagnose"

    def test_subgraph_wires_the_node(self):
        g = pg.build_block_subgraph()
        assert "timing_fix" in g.nodes
