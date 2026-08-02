# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Backend hardening (live aes PnR escalation): SDC clock-port discovery,
pre-CTS repair_design + dont_use in the PnR templates, env-tunable DRC/LVS
step timeouts. Pure; no EDA tools."""
from __future__ import annotations

from pathlib import Path

_ENGINE = Path(__file__).resolve().parent.parent


class TestPnrTemplatePreCtsRepair:
    def test_reference_template_repairs_before_cts(self):
        tcl = (_ENGINE / "pdk_templates" / "sky130"
               / "pnr_reference.tcl").read_text()
        assert "repair_design" in tcl
        assert tcl.index("repair_design") < tcl.index("clock_tree_synthesis")
        assert "set_max_fanout 16" in tcl
        assert "set_dont_use" in tcl
        assert "lpflow_*" in tcl and "probe" in tcl

    def test_inline_template_matches(self):
        # The OpenROAD PnR TCL generator moved into the sky130 deployment (PR5,
        # byo-pdk): the pre-CTS repair_design + fanout cap invariant now lives
        # there, parameterized from PDKConfig.pnr.max_fanout (== 16 for sky130).
        src = (_ENGINE / "pdk" / "deployments" / "sky130.py").read_text()
        assert src.index("repair_design") < src.index("clock_tree_synthesis")
        assert "set_max_fanout {max_fanout}" in src
        from orchestrator.pdk.deployments.sky130 import DEPLOYMENT
        assert DEPLOYMENT.pdk.pnr.max_fanout == 16


class TestSdcClockDiscovery:
    def _state(self, tmp_path, rtl_text):
        (tmp_path / "rtl").mkdir()
        (tmp_path / "rtl" / "top.v").write_text(rtl_text)
        return {
            "flat_netlist_path": "", "flat_sdc_path": "",
            "current_block": {"name": "top", "rtl_target": "rtl/top.v"},
            "project_root": str(tmp_path),
            "target_clock_mhz": 50.0,
        }

    def test_wb_clk_i_discovered(self, tmp_path):
        from orchestrator.langgraph.backend_graph import _resolve_netlist
        st = self._state(
            tmp_path,
            "module top (input wire wb_clk_i, input wire wb_rst_i,\n"
            "  input wire [37:0] io_in);\nendmodule\n")
        _rtl, sdc = _resolve_netlist(st)
        text = Path(sdc).read_text()
        assert "[get_ports wb_clk_i]" in text
        assert "[get_ports clk]" not in text

    def test_plain_clk_still_works(self, tmp_path):
        from orchestrator.langgraph.backend_graph import _resolve_netlist
        st = self._state(tmp_path,
                         "module top (input clk, input rst);\nendmodule\n")
        _rtl, sdc = _resolve_netlist(st)
        assert "[get_ports clk]" in Path(sdc).read_text()


class TestEdaStepTimeouts:
    def test_drc_lvs_call_sites_env_tunable(self):
        src = (_ENGINE / "langgraph" / "backend_graph.py").read_text()
        assert 'CORESMITH_DRC_TIMEOUT' in src
        assert 'CORESMITH_LVS_TIMEOUT' in src

    def test_eda_timeout_env_override(self, monkeypatch):
        from orchestrator.langgraph.backend_graph import _eda_timeout
        monkeypatch.delenv("CORESMITH_DRC_TIMEOUT", raising=False)
        assert _eda_timeout("CORESMITH_DRC_TIMEOUT", 2400) == 2400
        monkeypatch.setenv("CORESMITH_DRC_TIMEOUT", "5400")
        assert _eda_timeout("CORESMITH_DRC_TIMEOUT", 2400) == 5400


class TestIntegrationTopSelection:
    def _mk(self, tmp_path, files):
        d = tmp_path / "rtl" / "integration"
        d.mkdir(parents=True)
        for name, body in files.items():
            (d / name).write_text(body)
        return d

    def test_root_that_instantiates_children_wins(self, tmp_path):
        from orchestrator.langgraph.backend_graph import _select_integration_top
        d = self._mk(tmp_path, {
            # alphabetically first, but a leaf that instantiates nothing
            "openframe_project_wrapper.v":
                "module openframe_project_wrapper(input a); endmodule\n",
            "user_project_wrapper.v":
                "module user_project_wrapper(input clk);\n"
                "  user_project_wrapper_pads u_p(.clk(clk));\n"
                "  aes_core u_c(.clk(clk));\nendmodule\n",
            "user_project_wrapper_pads.v":
                "module user_project_wrapper_pads(input clk); endmodule\n",
        })
        f, m = _select_integration_top(d)
        assert m == "user_project_wrapper"
        assert f.endswith("user_project_wrapper.v")

    def test_single_file_returned(self, tmp_path):
        from orchestrator.langgraph.backend_graph import _select_integration_top
        d = self._mk(tmp_path, {
            "only_top.v": "module only_top(input a); endmodule\n"})
        f, m = _select_integration_top(d)
        assert m == "only_top" and f.endswith("only_top.v")

    def test_empty_dir(self, tmp_path):
        from orchestrator.langgraph.backend_graph import _select_integration_top
        d = tmp_path / "rtl" / "integration"
        d.mkdir(parents=True)
        assert _select_integration_top(d) == ("", "")


class TestExhaustionReopen:
    def test_exhausted_parks_on_ask_human(self):
        from orchestrator.langgraph.backend_graph import route_after_increment
        assert route_after_increment(
            {"attempt": 4, "max_attempts": 3, "debug_result": {}}
        ) == "ask_human"

    def test_within_budget_routes_to_target(self):
        from orchestrator.langgraph.backend_graph import route_after_increment
        assert route_after_increment(
            {"attempt": 2, "max_attempts": 3,
             "debug_result": {"next_action": "retry_drc"}}) == "drc"

    def test_increment_resets_after_exhaustion(self):
        import asyncio

        from orchestrator.langgraph.backend_graph import increment_attempt_node
        # re-entering with attempt already past the budget -> reset to 1
        out = asyncio.run(
            increment_attempt_node(
                {"attempt": 4, "max_attempts": 3,
                 "current_block": {"name": "top"}}))
        assert out["attempt"] == 1

    def test_increment_bumps_within_budget(self):
        import asyncio

        from orchestrator.langgraph.backend_graph import increment_attempt_node
        out = asyncio.run(
            increment_attempt_node(
                {"attempt": 1, "max_attempts": 3,
                 "current_block": {"name": "top"}}))
        assert out["attempt"] == 2
