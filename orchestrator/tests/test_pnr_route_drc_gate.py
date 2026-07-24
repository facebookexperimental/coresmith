# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""PnR route-DRC honest gate: a routed design that OpenROAD left with
unresolved detailed-route DRC violations must NOT be reported as a passing PnR
(the same false-pass class as the synth/DRC honest gates).

Every observable change is env-gated (``CORESMITH_PNR_ROUTE_DRC_GATE``,
default ON) and tested on BOTH branches. FAIRNESS: only generic synthetic
``route_drc.rpt`` fixtures + mocked EDA calls -- no benchmark/exercise/golden
names, no PDK collateral or real OpenROAD required.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchestrator.langgraph import backend_graph as bg
from orchestrator.langgraph import backend_helpers as bh
from orchestrator.langgraph import macro_registry as mr

# --- generic synthetic collateral -----------------------------------------

def _route_drc_rpt(n: int) -> str:
    """A synthetic OpenROAD ``detailed_route -output_drc`` report with exactly
    ``n`` violation ENTRIES, in the real per-entry format (one
    ``violation type:`` marker line + indented ``srcs:`` / ``bbox`` detail)."""
    blocks = []
    for i in range(n):
        blocks.append(
            f"violation type: Metal Spacing\n"
            f"\tsrcs: net_{i} net_{i + 1}\n"
            f"\tbbox = ( {i} {i} ) - ( {i + 1} {i + 1} ) on Layer met1\n"
        )
    return "".join(blocks)


def _write_route_drc(out_dir: Path, n: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "route_drc.rpt"
    p.write_text(_route_drc_rpt(n))
    return p


# ===========================================================================
# Robust parse: count violation ENTRIES, not raw "violation" substrings.
# ===========================================================================

class TestCountRouteDrcViolations:
    def test_counts_entries(self, tmp_path):
        p = _write_route_drc(tmp_path, 3634)
        assert bh.count_route_drc_violations(p) == 3634

    def test_zero_entries_empty_file(self, tmp_path):
        p = tmp_path / "route_drc.rpt"
        p.write_text("")
        assert bh.count_route_drc_violations(p) == 0

    def test_zero_entries_absent_file(self, tmp_path):
        assert bh.count_route_drc_violations(tmp_path / "nope.rpt") == 0

    def test_substring_robustness(self, tmp_path):
        # A raw substring count of "violation" would over-count on a summary
        # header / detail text; the entry count is exactly the marker lines.
        p = tmp_path / "route_drc.rpt"
        p.write_text(
            "Summary: total violation count reported below (violation report)\n"
            "violation type: Short\n"
            "\tsrcs: a b   # this line also mentions a violation in text\n"
            "\tbbox = ( 0 0 ) - ( 1 1 ) on Layer met2\n"
        )
        raw = p.read_text().count("violation")
        assert raw > 1                                   # substring over-counts
        assert bh.count_route_drc_violations(p) == 1     # one true entry


# ===========================================================================
# run_pnr_flow (legacy deterministic flow used by run_step + tests): success
# must reflect route-DRC. Both branches of CORESMITH_PNR_ROUTE_DRC_GATE.
# ===========================================================================

class TestRunPnrFlowGate:
    def _wire_openroad(self, monkeypatch):
        """Mock the EDA calls so no real OpenROAD runs. The route_drc.rpt in
        out_dir is the only signal the gate reads."""
        monkeypatch.setattr(bh, "generate_pnr_tcl",
                            lambda *a, **k: str(a[3]) + "/pnr.tcl")

        def _fake_openroad(*a, **k):
            return {"success": True, "stdout": "", "log_path": "/tmp/or.log"}
        monkeypatch.setattr(bh, "run_openroad", _fake_openroad)
        monkeypatch.setattr(bh, "parse_openroad_reports", lambda *a, **k: {
            "design_area_um2": 1000.0, "wns_ns": 0.1, "total_power_mw": 1.0,
        })
        monkeypatch.setattr(bh, "parse_pnr_stdout", lambda *a, **k: {})
        monkeypatch.setattr(bh, "render_layout_image", lambda *a, **k: True)

    def _run(self, tmp_path, monkeypatch, n):
        self._wire_openroad(monkeypatch)
        out_dir = tmp_path / "pnr"
        _write_route_drc(out_dir, n)
        return bh.run_pnr_flow(
            "blk", str(tmp_path / "net.v"), str(tmp_path / "s.sdc"),
            str(out_dir),
        )

    def test_gate_on_violations_fail(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_PNR_ROUTE_DRC_GATE", "1")   # default ON
        res = self._run(tmp_path, monkeypatch, 3634)
        assert res["success"] is False
        assert res["route_drc_violations"] == 3634
        assert "3634" in res["error"]
        assert "Route-DRC gate" in res["error"]

    def test_gate_on_clean_passes(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_PNR_ROUTE_DRC_GATE", "1")
        res = self._run(tmp_path, monkeypatch, 0)
        assert res["success"] is True
        assert res["route_drc_violations"] == 0
        assert "error" not in res

    def test_gate_on_absent_report_passes(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_PNR_ROUTE_DRC_GATE", "1")
        self._wire_openroad(monkeypatch)
        out_dir = tmp_path / "pnr"
        out_dir.mkdir(parents=True, exist_ok=True)      # no route_drc.rpt
        res = bh.run_pnr_flow(
            "blk", str(tmp_path / "net.v"), str(tmp_path / "s.sdc"),
            str(out_dir),
        )
        assert res["success"] is True
        assert res["route_drc_violations"] == 0

    def test_gate_off_preserves_legacy_success(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_PNR_ROUTE_DRC_GATE", "0")   # pre-fix
        res = self._run(tmp_path, monkeypatch, 3634)
        assert res["success"] is True                    # legacy: unconditional
        assert res["route_drc_violations"] == 3634       # still reported
        assert "error" not in res


# ===========================================================================
# run_pnr_node (live LLM-driven flow): the inner LLM's success:true on top of
# a non-empty route_drc.rpt must be demoted, and pnr_result.json rewritten.
# ===========================================================================

def _node_state(tmp_path):
    net = tmp_path / "flat.v"
    net.write_text("module blk(input clk);\nendmodule\n")
    return {
        "current_block": {"name": "blk"}, "attempt": 1,
        "project_root": str(tmp_path), "flat_netlist_path": str(net),
        "flat_sdc_path": "", "target_clock_mhz": 50.0, "max_attempts": 3,
        "macro_bindings": None, "synth_gate_count": 0,
    }


def _out_dir(tmp_path, block="blk"):
    return tmp_path / "syn" / "output" / block / "pnr"


class TestRunPnrNodeGate:
    def _wire_success_llm(self, monkeypatch, out_dir):
        """Inner LLM reports success:true (writes pnr_result.json), with the
        route_drc.rpt sitting next to it -- the false-pass this gate closes."""
        monkeypatch.setattr(mr, "discover_macros", lambda *a, **k: {})
        monkeypatch.setattr(bh, "render_layout_image", lambda *a, **k: True)
        routed_def = str(out_dir / "blk_routed.def")

        async def _fake_llm(**kwargs):
            # Emulate the inner LLM writing an (over-optimistic) result JSON.
            res = {"success": True, "routed_def_path": routed_def, "wns_ns": 0.0}
            rp = kwargs.get("result_json_path")
            if rp:
                Path(rp).parent.mkdir(parents=True, exist_ok=True)
                Path(rp).write_text(json.dumps(res))
            return res
        monkeypatch.setattr(bg, "_run_llm_eda_step", _fake_llm)

    @pytest.mark.asyncio
    async def test_gate_on_violations_demote_node(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_PNR_ROUTE_DRC_GATE", "1")   # default ON
        out_dir = _out_dir(tmp_path)
        _write_route_drc(out_dir, 3634)
        self._wire_success_llm(monkeypatch, out_dir)

        result = await bg.run_pnr_node(_node_state(tmp_path))

        assert result["timing_result"]["met"] is False
        assert "Route-DRC gate" in result["previous_error"]
        assert "3634" in result["previous_error"]
        # pnr_result.json rewritten to reflect the demotion.
        rj = json.loads((out_dir / "pnr_result.json").read_text())
        assert rj["success"] is False
        assert rj["route_drc_violations"] == 3634

    @pytest.mark.asyncio
    async def test_gate_on_clean_passes_node(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_PNR_ROUTE_DRC_GATE", "1")
        out_dir = _out_dir(tmp_path)
        _write_route_drc(out_dir, 0)                     # clean route
        self._wire_success_llm(monkeypatch, out_dir)

        result = await bg.run_pnr_node(_node_state(tmp_path))

        assert "previous_error" not in result
        assert result["route_result"]["success"] is True

    @pytest.mark.asyncio
    async def test_gate_off_keeps_node_success(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_PNR_ROUTE_DRC_GATE", "0")   # pre-fix
        out_dir = _out_dir(tmp_path)
        _write_route_drc(out_dir, 3634)
        self._wire_success_llm(monkeypatch, out_dir)

        result = await bg.run_pnr_node(_node_state(tmp_path))

        assert "previous_error" not in result            # legacy: still passes
        assert result["route_result"]["success"] is True
