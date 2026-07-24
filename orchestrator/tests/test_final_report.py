# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Final-report (signoff scorecard) deterministic aggregation tests.

Feeds a FAKE run state + FAKE scoreboard (no LangGraph runtime, no tools) and
asserts the emitted report carries the block coverage %, Fmax, testbench names,
PPA and DV verdicts -- both in the JSON dict and the rendered markdown.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchestrator.langgraph.final_report import (
    build_final_report,
    fmax_mhz,
    render_markdown,
)


class _FakeScoreboard:
    """Duck-typed Scoreboard: canned dv/ppa rows keyed by (block, scope)."""

    def __init__(self, dv, ppa, cov=None):
        self._dv = dv
        self._ppa = ppa
        self._cov = cov or {}

    def latest_dv(self, block=None, scope=None):
        row = self._dv.get((block, scope))
        return [row] if row else []

    def latest_ppa(self, block):
        return self._ppa.get(block)

    def coverage_latest(self, block):
        return self._cov.get(block)


def _write_cov(root: Path, block: str, cov: dict) -> None:
    d = root / ".coresmith" / "blocks" / block
    d.mkdir(parents=True, exist_ok=True)
    (d / "coverage.json").write_text(json.dumps(cov))


def _write_tput(root: Path, block: str, rec: dict) -> None:
    d = root / ".coresmith" / "blocks" / block
    d.mkdir(parents=True, exist_ok=True)
    (d / "throughput.json").write_text(json.dumps(rec))


def _write_chip_tput(root: Path, rec: dict) -> None:
    d = root / ".coresmith"
    d.mkdir(parents=True, exist_ok=True)
    (d / "chip_throughput.json").write_text(json.dumps(rec))


def _passing_setup(tmp_path):
    state = {
        "project_root": str(tmp_path),
        "target_clock_mhz": 100.0,  # 10 ns period
        "block_queue": [
            {"name": "adder", "testbench": "tb/cocotb/test_adder.py"},
            {"name": "mul", "testbench": "tb/cocotb/test_mul.py"},
        ],
        "completed_blocks": [
            {"name": "adder", "success": True, "attempts": 1},
            {"name": "mul", "success": True, "attempts": 2},
        ],
        "integration_dv_result": {
            "passed": True, "test_count": 12,
            "testbench_path": "tb/integration/test_chip_top.py",
            "design_name": "chip_top",
        },
        "validation_dv_result": {
            "passed": True, "test_count": 5, "requirement_count": 8,
            "testbench_path": "tb/validation/test_ers.py",
            "design_name": "chip_top",
        },
        "pipeline_done": True,
    }
    sb = _FakeScoreboard(
        dv={
            ("adder", "rtl"): {"passed": 1, "tests_passed": 4,
                               "tests_total": 4, "source": "gate"},
            ("mul", "rtl"): {"passed": 1, "tests_passed": 6,
                             "tests_total": 6, "source": "gate"},
            ("chip_top", "chip"): {"passed": 1, "tests_total": 12},
            ("chip_top", "validation"): {"passed": 1, "tests_total": 5},
        },
        ppa={
            "adder": {"cells": 1200, "ff": 64, "area_um2": 850.5,
                      "wns_ns": -1.5, "ppa_ok": 1, "elaborated": 1,
                      "budget_ff": 128, "budget_area_um2": 2000.0},
            "mul": {"cells": 5000, "ff": 200, "area_um2": 3200.0,
                    "wns_ns": 0.5, "ppa_ok": 1, "elaborated": 1},
            "chip_top": {"cells": 6500, "ff": 300, "area_um2": 4200.0,
                         "wns_ns": -0.8},
        },
    )
    # Coverage: adder measured (85%), mul NOT applicable (visible reason).
    _write_cov(tmp_path, "adder", {
        "applicable": True, "pct": 85.0, "floor": 70.0,
        "points_total": 20, "points_hit": 17, "uncovered_count": 3,
        "passed": True,
    })
    _write_cov(tmp_path, "mul", {
        "applicable": False, "reason": "no coverage.dat produced by the sim",
    })
    return state, sb


class TestFmax:
    def test_fmax_from_wns(self):
        # 10 ns period, WNS -1.5 -> achieved 11.5 ns -> 86.96 MHz
        assert fmax_mhz(-1.5, 10.0) == pytest.approx(86.96, abs=0.05)

    def test_fmax_positive_slack(self):
        # WNS +0.5 -> achieved 9.5 ns -> 105.26 MHz
        assert fmax_mhz(0.5, 10.0) == pytest.approx(105.26, abs=0.05)

    def test_fmax_none_when_unknown(self):
        assert fmax_mhz(None, 10.0) is None
        assert fmax_mhz(-1.0, None) is None

    def test_fmax_none_when_unphysical(self):
        # A (nonsensical) positive slack larger than the whole period yields a
        # non-positive achieved period -> report as n/a rather than a bogus Fmax.
        assert fmax_mhz(12.0, 10.0) is None


class TestBuildReportPassing:
    def test_signoff_pass_and_facts(self, tmp_path):
        state, sb = _passing_setup(tmp_path)
        r = build_final_report(state, str(tmp_path), scoreboard=sb,
                               now=1_700_000_000.0)

        assert r["schema"].startswith("coresmith.final_report/")
        sign = r["signoff"]
        assert sign["status"] == "PASS"
        assert sign["blocks_total"] == 2 and sign["blocks_passed"] == 2
        assert sign["integration_dv"] == "pass"
        assert sign["validation_dv"] == "pass"

        blocks = {b["name"]: b for b in r["blocks"]}

        # DV verdicts persisted per block
        assert blocks["adder"]["dv"]["passed"] is True
        assert blocks["adder"]["dv"]["tests_total"] == 4

        # coverage % surfaced for adder; visible reason for mul
        assert blocks["adder"]["coverage"]["pct"] == 85.0
        assert blocks["adder"]["coverage"]["applicable"] is True
        assert blocks["mul"]["coverage"]["applicable"] is False
        assert "no coverage.dat" in blocks["mul"]["coverage"]["reason"]

        # PPA cells/ff/area + derived Fmax per block
        assert blocks["adder"]["ppa"]["cells"] == 1200
        assert blocks["adder"]["ppa"]["ff"] == 64
        assert blocks["adder"]["ppa"]["area_um2"] == 850.5
        assert blocks["adder"]["ppa"]["fmax_mhz"] == pytest.approx(86.96, abs=0.05)

        # testbench names present
        tb_names = [t["name"] for t in blocks["adder"]["testbenches"]]
        assert "test_adder.py" in tb_names

        # chip-level
        assert r["chip"]["integration_dv"]["testbench"] == "test_chip_top.py"
        assert r["chip"]["validation_dv"]["testbench"] == "test_ers.py"
        assert r["chip"]["validation_dv"]["requirement_count"] == 8

        # totals / aggregates
        assert sign["coverage_min_pct"] == 85.0
        assert sign["top_fmax_mhz"] == pytest.approx(86.96, abs=0.05)
        # aggregate block area = 850.5 + 3200.0
        assert r["chip"]["aggregate_area_um2"] == pytest.approx(4050.5, abs=0.1)

    def test_markdown_contains_all_signoff_facts(self, tmp_path):
        state, sb = _passing_setup(tmp_path)
        r = build_final_report(state, str(tmp_path), scoreboard=sb)
        md = render_markdown(r)

        # PASS banner
        assert "SIGNOFF: PASS" in md
        # coverage %
        assert "85.00%" in md
        # Fmax present (MHz)
        assert "86.96 MHz" in md or "86.96" in md
        # every testbench listed by name (verification traceability)
        for tb in ("test_adder.py", "test_mul.py", "test_chip_top.py",
                   "test_ers.py"):
            assert tb in md
        # PPA numbers
        assert "1,200" in md  # adder cells
        # not-applicable coverage reason is visible, not blank
        assert "no coverage.dat" in md


class TestBuildReportFailing:
    def test_fail_banner_when_block_and_validation_fail(self, tmp_path):
        state = {
            "project_root": str(tmp_path),
            "target_clock_mhz": 50.0,
            "block_queue": [
                {"name": "core", "testbench": "tb/cocotb/test_core.py"},
            ],
            "completed_blocks": [
                {"name": "core", "success": False, "attempts": 3,
                 "escalated": True},
            ],
            "integration_dv_result": {},   # never ran
            "validation_dv_result": {},
        }
        sb = _FakeScoreboard(
            dv={("core", "rtl"): {"passed": 0, "tests_passed": 2,
                                  "tests_total": 5}},
            ppa={"core": {"cells": 900, "ff": 40, "area_um2": 600.0}},
        )
        _write_cov(tmp_path, "core", {
            "applicable": True, "pct": 40.0, "floor": 70.0,
            "points_total": 10, "points_hit": 4, "uncovered_count": 6,
            "passed": False,
        })
        r = build_final_report(state, str(tmp_path), scoreboard=sb)
        assert r["signoff"]["status"] == "FAIL"
        assert r["signoff"]["blocks_passed"] == 0
        assert r["signoff"]["integration_dv"] == "n/a"
        md = render_markdown(r)
        assert "SIGNOFF: FAIL" in md
        assert "test_core.py" in md

class TestThroughputAccountability:
    def test_throughput_rows_in_report_and_markdown(self, tmp_path):
        state, sb = _passing_setup(tmp_path)
        # adder: measured within ceiling (PASS); mul: no declared cyc/op (n/a).
        _write_tput(tmp_path, "adder", {
            "gate": "measured_throughput", "scope": "block",
            "applicable": True, "passed": True,
            "declared_cyc_per_op": 11, "measured_cyc_per_op": 11.5,
            "threshold_cyc_per_op": 12.1, "ratio": 1.045,
            "artifact_missing": False,
        })
        _write_tput(tmp_path, "mul", {
            "gate": "measured_throughput", "scope": "block",
            "applicable": False, "passed": None,
            "reason": "block declares no §6.1 cyc/op",
        })
        _write_chip_tput(tmp_path, {
            "scope": "chip", "applicable": True, "passed": False,
            "measured_cyc_per_op_chip": 37.0, "budget_cyc_per_op": 21,
            "threshold_cyc_per_op": 23.1, "budget_source": "state",
            "grader_window": "op-start-committed -> DONE visible on status pin",
        })
        r = build_final_report(state, str(tmp_path), scoreboard=sb)

        blocks = {b["name"]: b for b in r["blocks"]}
        assert blocks["adder"]["throughput"]["declared_cyc_per_op"] == 11
        assert blocks["adder"]["throughput"]["measured_cyc_per_op"] == 11.5
        assert blocks["mul"]["throughput"]["applicable"] is False

        sign = r["signoff"]
        assert sign["throughput_blocks_gated"] == 1   # only adder applicable
        assert sign["throughput_blocks_failed"] == 0
        assert sign["chip_measured_cyc_per_op"] == 37.0
        assert r["chip"]["throughput"]["budget_cyc_per_op"] == 21

        md = render_markdown(r)
        assert "Throughput accountability" in md
        # per-block row: declared | measured | ratio | verdict
        assert "11" in md and "11.5" in md
        assert "PASS" in md
        # chip op-window line with the grader window note
        assert "37" in md
        assert "status pin" in md

    def test_throughput_missing_artifact_renders_fail(self, tmp_path):
        state, sb = _passing_setup(tmp_path)
        _write_tput(tmp_path, "adder", {
            "applicable": True, "passed": False, "artifact_missing": True,
            "declared_cyc_per_op": 11, "measured_cyc_per_op": None,
            "threshold_cyc_per_op": 12.1,
        })
        r = build_final_report(state, str(tmp_path), scoreboard=sb)
        assert r["signoff"]["throughput_blocks_failed"] == 1
        md = render_markdown(r)
        assert "artifact missing" in md

    def test_no_throughput_artifacts_is_na_not_crash(self, tmp_path):
        state, sb = _passing_setup(tmp_path)
        r = build_final_report(state, str(tmp_path), scoreboard=sb)
        # no throughput.json written -> every block n/a, section still renders
        for b in r["blocks"]:
            assert b["throughput"]["applicable"] is False
        assert r["signoff"]["throughput_blocks_gated"] == 0
        md = render_markdown(r)
        assert "Throughput accountability" in md


class TestBuildReportFailingExtra:
    def test_no_scoreboard_does_not_crash(self, tmp_path):
        # Missing db path -> Scoreboard opened internally returns nothing.
        state = {
            "project_root": str(tmp_path),
            "block_queue": [{"name": "b0", "testbench": "tb/test_b0.py"}],
            "completed_blocks": [{"name": "b0", "success": True}],
        }
        r = build_final_report(state, str(tmp_path))
        assert r["blocks"][0]["name"] == "b0"
        # b0 marked passed via completed_blocks fallback
        assert r["blocks"][0]["dv"]["passed"] is True
        render_markdown(r)  # must not raise


class TestSignoffTerminalGating:
    """Audit F1: PASS must gate on the TERMINAL chip-level result, never only
    the DV stages -- the false-PASS defect printed 'SIGNOFF SCORECARD: PASS'
    right under the red NOT-pipeline_done banner on both the reference codec runs."""

    def test_validation_pass_but_synth_fail_is_fail(self, tmp_path):
        state, sb = _passing_setup(tmp_path)
        state["pipeline_done"] = False
        state["validation_dv_result"].update(
            chip_top_synthesizable=False,
            synth_fail_reason="chip_top yosys techmap failed: ...",
        )
        r = build_final_report(state, str(tmp_path), scoreboard=sb)
        sign = r["signoff"]
        assert sign["status"] == "FAIL"
        assert "NOT synthesizable" in sign["status_reason"]
        assert sign["chip_top_synthesizable"] is False
        md = render_markdown(r)
        assert "SIGNOFF: FAIL" in md
        assert "Why not PASS" in md

    def test_validation_pass_but_die_budget_fail_is_fail(self, tmp_path):
        state, sb = _passing_setup(tmp_path)
        state["pipeline_done"] = False
        state["validation_dv_result"].update(
            die_budget_ok=False,
            die_rollup_reason="die-area rollup 18.402 mm^2 exceeds 2.600 mm^2",
        )
        r = build_final_report(state, str(tmp_path), scoreboard=sb)
        sign = r["signoff"]
        assert sign["status"] == "FAIL"
        assert "die budget" in sign["status_reason"]
        assert sign["die_budget_ok"] is False

    def test_all_dv_pass_but_not_pipeline_done_is_incomplete(self, tmp_path):
        state, sb = _passing_setup(tmp_path)
        state["pipeline_done"] = False
        r = build_final_report(state, str(tmp_path), scoreboard=sb)
        sign = r["signoff"]
        assert sign["status"] == "INCOMPLETE"
        assert "terminal chip-level result" in sign["status_reason"]
        md = render_markdown(r)
        assert "INCOMPLETE" in md

    def test_pass_keeps_tristates_and_empty_reason(self, tmp_path):
        state, sb = _passing_setup(tmp_path)
        state["validation_dv_result"]["chip_top_synthesizable"] = True
        r = build_final_report(state, str(tmp_path), scoreboard=sb)
        sign = r["signoff"]
        assert sign["status"] == "PASS"
        assert sign["status_reason"] == ""
        assert sign["chip_top_synthesizable"] is True
        assert sign["die_budget_ok"] is None  # gate never ran -> tri-state
