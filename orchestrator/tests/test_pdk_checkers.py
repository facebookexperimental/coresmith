# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""PR3: checker classes wrapping the existing tool-output parsers.

Fixture reports are driven THROUGH the Checker API (``check(req, run_dir)``) and
the pure ``parse_text``/``classify`` helpers. Three-state semantics are asserted
explicitly: a missing report is ``not_run`` and, when the checker is blocking,
that fails the verb (``CheckResult.failed``) -- never a false pass. Includes the
Yosys 0.65 box-format ("N cells") stat regression at both the checker and the
``synthesize_block`` source.
"""

from __future__ import annotations

import json
import types

from orchestrator.pdk.base import ToolRequest
from orchestrator.pdk.checkers import (
    GateSimVerdictChecker,
    LintChecker,
    LogicDepthChecker,
    LvsMatchChecker,
    MagicDrcChecker,
    PnrReportsChecker,
    RouteDrcChecker,
    StaChecker,
    SynthStatChecker,
)

# A REAL Yosys 0.65 generic (no-liberty) `stat` block: the module total is the
# two-token "N cells" box-format line, and there is NO "Number of cells:" line.
# synthesize_block's inline loop (len(parts) >= 3) misses this -> gate_count=0;
# the checker + the count_cells_from_stat backstop must recover it.
_YOSYS_065_BOX_STAT = """
=== tiny_matmul ===

        +----------Local Count, excluding submodules.
        |
     3801 wires
     4113 wire bits
       26 public wires
       14 ports
     3911 cells
       66   $_ANDNOT_
      621   $_AND_
       68   $_DFF_PN0_
       12   $_MUX_
     1178   $_XOR_

Chip area for module '\\tiny_matmul': 0.000000
"""


def _req(design="blk", **inputs):
    return ToolRequest(verb="v", design=design,
                       inputs={k: v for k, v in inputs.items()})


# ---------------------------------------------------------------------------
# SynthStatChecker + the 0.65 box-format regression
# ---------------------------------------------------------------------------
class TestSynthStatChecker:
    def test_box_format_cells_and_ff(self):
        m = SynthStatChecker.parse_text(_YOSYS_065_BOX_STAT)
        assert m["cells"] == 3911
        assert m["ff_count"] == 68  # $_DFF_PN0_ x 68

    def test_check_reads_report_from_run_dir(self, tmp_path):
        (tmp_path / "blk_report.txt").write_text(_YOSYS_065_BOX_STAT)
        r = SynthStatChecker().check(_req(), tmp_path)
        assert r.status == "pass"
        assert r.metrics["cells"] == 3911
        assert r.blocking is False  # advisory -> never gates synth by itself

    def test_missing_report_is_not_run_but_advisory(self, tmp_path):
        r = SynthStatChecker().check(_req(), tmp_path)
        assert r.status == "not_run"
        # advisory: not_run does NOT fail the verb
        assert r.failed is False


# ---------------------------------------------------------------------------
# LogicDepthChecker (advisory)
# ---------------------------------------------------------------------------
class TestLogicDepthChecker:
    def test_parses_length(self, tmp_path):
        (tmp_path / "blk_ltp.txt").write_text(
            "Longest topological path in tiny (length=17)\n")
        r = LogicDepthChecker().check(_req(), tmp_path)
        assert r.status == "pass" and r.metrics["logic_depth"] == 17

    def test_missing_is_advisory_not_run(self, tmp_path):
        r = LogicDepthChecker().check(_req(), tmp_path)
        assert r.status == "not_run" and r.failed is False


# ---------------------------------------------------------------------------
# PnrReportsChecker (advisory; timing_met None -> not_run)
# ---------------------------------------------------------------------------
class TestPnrReportsChecker:
    def test_timing_met(self, tmp_path):
        (tmp_path / "timing_wns.rpt").write_text("wns 0.5\n")
        (tmp_path / "area.rpt").write_text("Design area 955.0 1000.0 60.0%\n")
        r = PnrReportsChecker().check(_req(), tmp_path)
        assert r.status == "pass"
        assert r.metrics["wns_ns"] == 0.5
        assert r.metrics["design_area_um2"] == 955.0

    def test_negative_wns_fails(self, tmp_path):
        (tmp_path / "timing_wns.rpt").write_text("wns -3.42\n")
        r = PnrReportsChecker().check(_req(), tmp_path)
        assert r.status == "fail"

    def test_no_timing_report_is_not_run(self, tmp_path):
        r = PnrReportsChecker().check(_req(), tmp_path)
        assert r.status == "not_run"  # timing_met None -> unmeasured, never clean


# ---------------------------------------------------------------------------
# RouteDrcChecker (BLOCKING; missing report -> not_run -> fail)
# ---------------------------------------------------------------------------
_ROUTE_DRC = """violation type: Metal Short
  srcs: net1 net2
  bbox = ( 1 2 ) - ( 3 4 ) on Layer met1
violation type: Spacing
  srcs: net3
  bbox = ( 5 6 ) - ( 7 8 ) on Layer met2
"""


class TestRouteDrcChecker:
    def test_counts_violation_entries(self, tmp_path):
        (tmp_path / "route_drc.rpt").write_text(_ROUTE_DRC)
        r = RouteDrcChecker().check(_req(), tmp_path)
        assert r.status == "fail" and r.metrics["route_drc_violations"] == 2

    def test_empty_report_is_clean_pass(self, tmp_path):
        (tmp_path / "route_drc.rpt").write_text("")
        r = RouteDrcChecker().check(_req(), tmp_path)
        assert r.status == "pass" and r.metrics["route_drc_violations"] == 0

    def test_missing_report_is_blocking_not_run(self, tmp_path):
        r = RouteDrcChecker().check(_req(), tmp_path)
        assert r.status == "not_run"
        assert r.blocking is True and r.failed is True  # fail-closed


# ---------------------------------------------------------------------------
# StaChecker (advisory; OpenSTA 3.x "wns max N" form)
# ---------------------------------------------------------------------------
class TestStaChecker:
    def test_opensta3_form(self, tmp_path):
        (tmp_path / "sta.rpt").write_text("wns max -1.25\ntns max -3.0\n")
        r = StaChecker().check(_req(), tmp_path)
        assert r.status == "fail" and r.metrics["wns_ns"] == -1.25

    def test_positive_wns_passes(self, tmp_path):
        (tmp_path / "sta.rpt").write_text("wns 0.10\n")
        assert StaChecker().check(_req(), tmp_path).status == "pass"

    def test_missing_is_advisory_not_run(self, tmp_path):
        r = StaChecker().check(_req(), tmp_path)
        assert r.status == "not_run" and r.failed is False


# ---------------------------------------------------------------------------
# MagicDrcChecker (BLOCKING; drc_verdict three-state)
# ---------------------------------------------------------------------------
class TestMagicDrcChecker:
    def test_clean_report_passes(self, tmp_path):
        (tmp_path / "magic_drc.rpt").write_text("Design: blk\nDRC count: 0\n")
        r = MagicDrcChecker().check(_req(), tmp_path)
        assert r.status == "pass"

    def test_dirty_report_fails(self, tmp_path):
        (tmp_path / "magic_drc.rpt").write_text(
            "synth_top 2\n"
            "----------------------------------------\n"
            "Metal4 minimum area < 0.24um^2 (met4.4a)\n"
            "----------------------------------------\n"
            " 25000 25000 25076 25076\n"
            " 80000 80000 80076 80076\n")
        r = MagicDrcChecker().check(_req(), tmp_path)
        assert r.status == "fail" and r.metrics["violations"] == 2

    def test_missing_report_is_blocking_not_run(self, tmp_path):
        r = MagicDrcChecker().check(_req(), tmp_path)
        assert r.status == "not_run"
        assert r.failed is True  # absence of evidence is NOT clean


# ---------------------------------------------------------------------------
# LvsMatchChecker (BLOCKING; wraps reconcile_lvs_match)
# ---------------------------------------------------------------------------
class TestLvsMatchChecker:
    def test_match_uniquely_passes(self, tmp_path):
        (tmp_path / "blk_lvs.rpt").write_text(
            "Subcircuit summary:\n"
            "Final result: Circuits match uniquely.\n")
        r = LvsMatchChecker().check(_req(), tmp_path)
        assert r.status == "pass" and r.metrics["lvs_raw_match"] is True

    def test_mismatch_fails(self, tmp_path):
        (tmp_path / "blk_lvs.rpt").write_text(
            "Final result: Netlists do not match.\n")
        r = LvsMatchChecker().check(_req(), tmp_path)
        assert r.status == "fail"

    def test_missing_report_is_blocking_not_run(self, tmp_path):
        r = LvsMatchChecker().check(_req(), tmp_path)
        assert r.status == "not_run" and r.failed is True

    def test_raw_match_helper(self):
        assert LvsMatchChecker.raw_match_from_report(
            "Final result: Circuits match uniquely.") is True
        assert LvsMatchChecker.raw_match_from_report(
            "Final result: Netlists do not match.") is False


# ---------------------------------------------------------------------------
# LintChecker (BLOCKING; %Error rule)
# ---------------------------------------------------------------------------
class TestLintChecker:
    def test_classify_clean(self):
        assert LintChecker.classify("Warning: unused", returncode=0) is True

    def test_classify_error(self):
        assert LintChecker.classify("%Error: syntax", returncode=1) is False

    def test_check_error_log_fails(self, tmp_path):
        (tmp_path / "lint.log").write_text("%Error: bad module\n")
        r = LintChecker().check(_req(), tmp_path)
        assert r.status == "fail"

    def test_missing_log_is_blocking_not_run(self, tmp_path):
        r = LintChecker().check(_req(), tmp_path)
        assert r.status == "not_run" and r.failed is True


# ---------------------------------------------------------------------------
# GateSimVerdictChecker (BLOCKING; blank verdict is never a pass)
# ---------------------------------------------------------------------------
class TestGateSimVerdictChecker:
    def test_clean_verdict_passes(self, tmp_path):
        (tmp_path / "gate_sim_verdict.json").write_text(
            json.dumps({"cycles_compared": 200000, "mismatches": 0}))
        r = GateSimVerdictChecker().check(_req(), tmp_path)
        assert r.status == "pass" and r.metrics["cycles_compared"] == 200000

    def test_mismatch_fails(self, tmp_path):
        (tmp_path / "gate_sim_verdict.json").write_text(
            json.dumps({"cycles_compared": 10, "mismatches": 3}))
        assert GateSimVerdictChecker().check(_req(), tmp_path).status == "fail"

    def test_blank_verdict_is_blocking_not_run(self, tmp_path):
        (tmp_path / "gate_sim_verdict.json").write_text("")
        r = GateSimVerdictChecker().check(_req(), tmp_path)
        assert r.status == "not_run" and r.failed is True

    def test_missing_cycles_compared_is_not_run(self, tmp_path):
        (tmp_path / "gate_sim_verdict.json").write_text(json.dumps({"ok": True}))
        r = GateSimVerdictChecker().check(_req(), tmp_path)
        assert r.status == "not_run" and r.failed is True


# ---------------------------------------------------------------------------
# synthesize_block box-format regression (the fix at source)
# ---------------------------------------------------------------------------
def test_synthesize_block_recovers_box_format_gate_count(tmp_path, monkeypatch):
    """A generic synth whose `stat` prints only the 0.65 box-format total
    ("N cells") must still report a nonzero gate_count -- the inline loop misses
    it, so the count_cells_from_stat backstop has to recover it."""
    import orchestrator.langgraph.pipeline_helpers as ph

    rtl = tmp_path / "blk.v"
    rtl.write_text("module blk(input a, output b); assign b = a; endmodule\n")
    monkeypatch.setattr(ph, "PROJECT_ROOT", tmp_path)
    # Force the generic (no-liberty) path so the box-format stat is produced.
    monkeypatch.setenv("CORESMITH_SYNTH_GENERIC", "1")

    def _fake_run(cmd, *a, **kw):
        return types.SimpleNamespace(
            returncode=0, stdout=_YOSYS_065_BOX_STAT, stderr="")

    monkeypatch.setattr(ph.subprocess, "run", _fake_run)
    res = ph.synthesize_block({"name": "blk"}, str(rtl))
    assert res["success"] is True
    assert res["gate_count"] == 3911, res["gate_count"]
    assert res["ff_count"] == 68
