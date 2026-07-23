# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Item-11 (B4) coverage helpers + verify_rtl coverage recording."""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.harness import coverage as cov
from orchestrator.harness import verify as V
from orchestrator.state_store.store import Scoreboard


class TestCoverageEnabled:
    def test_off_by_default(self, monkeypatch):
        monkeypatch.delenv("CORESMITH_COVERAGE", raising=False)
        assert cov.coverage_enabled() is False

    def test_on(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_COVERAGE", "1")
        assert cov.coverage_enabled() is True


class TestSummarize:
    def test_counts_hit_and_uncovered(self, tmp_path):
        # verilator_coverage --annotate style: %000000 marks an un-hit point.
        annotated = tmp_path / "cov"
        annotated.mkdir()
        (annotated / "adder.v").write_text(
            "        module adder;\n"        # no coverage prefix -> not a point
            " 000100  assign x = a + b;\n"    # hit
            " %000000 assign y = c - d;\n"    # uncovered
            " 000005  endmodule\n"            # hit
        )
        s = cov.summarize(annotated)
        assert s["points_total"] == 3
        assert s["points_hit"] == 2
        assert len(s["uncovered"]) == 1
        assert s["uncovered"][0]["file"] == "adder.v"
        assert s["pct"] == round(200.0 / 3, 2)

    def test_missing_dir(self, tmp_path):
        s = cov.summarize(tmp_path / "nope")
        assert s["points_total"] == 0


class TestFindCoverageDat:
    def test_finds_top_and_nested(self, tmp_path):
        (tmp_path / "coverage.dat").write_text("x")
        assert cov.find_coverage_dat(tmp_path).name == "coverage.dat"

    def test_none_when_absent(self, tmp_path):
        assert cov.find_coverage_dat(tmp_path) is None


def _annotated_tree(tmp_path: Path, hit: int, uncov: int) -> Path:
    """Fabricate a verilator_coverage --annotate output tree."""
    ann = tmp_path / "coverage_annotated"
    ann.mkdir(parents=True, exist_ok=True)
    lines = ["        module blk;"]
    for i in range(hit):
        lines.append(f" 00010{i % 10}  assign h{i} = a + {i};")
    for i in range(uncov):
        lines.append(f" %000000 assign u{i} = b - {i};")
    lines.append(" 000001  endmodule")
    (ann / "blk.v").write_text("\n".join(lines) + "\n")
    return ann


class TestLineCovGate:
    """The line-coverage floor gate (weak-TB rejector, default ON, floor 70)."""

    def _arm(self, tmp_path, monkeypatch, hit, uncov):
        # A coverage.dat must exist for the gate to be applicable; annotate()
        # is monkeypatched to the fabricated tree so no verilator needed.
        (tmp_path / "coverage.dat").write_text("x")
        ann = _annotated_tree(tmp_path, hit, uncov)
        monkeypatch.setattr(cov, "annotate", lambda sim_dir, **k: ann)

    def test_default_on_floor_70(self, monkeypatch):
        monkeypatch.delenv("CORESMITH_LINE_COV_GATE", raising=False)
        monkeypatch.delenv("CORESMITH_LINE_COV_FLOOR", raising=False)
        assert cov.line_cov_gate_enabled() is True
        assert cov.line_cov_floor() == 70.0

    def test_kill_switch(self, tmp_path, monkeypatch):
        self._arm(tmp_path, monkeypatch, hit=1, uncov=9)
        monkeypatch.setenv("CORESMITH_LINE_COV_GATE", "0")
        assert cov.line_cov_gate_verdict(tmp_path) is None

    def test_below_floor_fails_with_uncovered_feedback(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CORESMITH_LINE_COV_GATE", raising=False)
        monkeypatch.delenv("CORESMITH_LINE_COV_FLOOR", raising=False)
        self._arm(tmp_path, monkeypatch, hit=4, uncov=6)  # 5/11 hit = 45%
        v = cov.line_cov_gate_verdict(tmp_path)
        assert v is not None and v["passed"] is False
        assert v["floor"] == 70.0
        assert "LINE-COVERAGE GATE" in v["report"]
        assert "STRENGTHEN THE TESTBENCH" in v["report"]
        # actionable feedback: names the uncovered file:line points
        assert "blk.v" in v["report"]

    def test_above_floor_passes(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CORESMITH_LINE_COV_GATE", raising=False)
        monkeypatch.delenv("CORESMITH_LINE_COV_FLOOR", raising=False)
        self._arm(tmp_path, monkeypatch, hit=9, uncov=1)  # 10/11 hit = 91%
        v = cov.line_cov_gate_verdict(tmp_path)
        assert v is not None and v["passed"] is True

    def test_verdict_exposes_points_hit_and_uncovered_count(self, tmp_path, monkeypatch):
        # Part A: the verdict carries the persistable metrics the report needs.
        monkeypatch.delenv("CORESMITH_LINE_COV_GATE", raising=False)
        monkeypatch.delenv("CORESMITH_LINE_COV_FLOOR", raising=False)
        self._arm(tmp_path, monkeypatch, hit=4, uncov=6)  # 5/11 hit
        v = cov.line_cov_gate_verdict(tmp_path)
        assert v is not None
        assert v["points_total"] == 11
        assert v["points_hit"] == 5
        assert v["uncovered_count"] == 6

    def test_floor_env_override(self, tmp_path, monkeypatch):
        self._arm(tmp_path, monkeypatch, hit=4, uncov=6)  # 45%
        monkeypatch.setenv("CORESMITH_LINE_COV_FLOOR", "40")
        v = cov.line_cov_gate_verdict(tmp_path)
        assert v is not None and v["passed"] is True

    def test_not_applicable_without_coverage_dat(self, tmp_path, monkeypatch):
        # no coverage.dat -> annotate() returns None -> gate skips (None),
        # never failing a block for missing tooling.
        monkeypatch.delenv("CORESMITH_LINE_COV_GATE", raising=False)
        assert cov.line_cov_gate_verdict(tmp_path) is None

    def test_makefile_injection_follows_gate(self, monkeypatch):
        """The Makefile builder's predicate: --coverage is injected when the
        gate is on (default) OR the explicit opt-in is set; omitted only when
        both are off."""
        monkeypatch.delenv("CORESMITH_COVERAGE", raising=False)
        monkeypatch.delenv("CORESMITH_LINE_COV_GATE", raising=False)
        assert cov.coverage_enabled() or cov.line_cov_gate_enabled()
        monkeypatch.setenv("CORESMITH_LINE_COV_GATE", "0")
        assert not (cov.coverage_enabled() or cov.line_cov_gate_enabled())
        monkeypatch.setenv("CORESMITH_COVERAGE", "1")
        assert cov.coverage_enabled() or cov.line_cov_gate_enabled()


class TestCoverageRecord:
    """Part A: the ALWAYS-return persistence record (applicable + reason)."""

    def _arm(self, tmp_path, monkeypatch, hit, uncov):
        (tmp_path / "coverage.dat").write_text("x")
        ann = _annotated_tree(tmp_path, hit, uncov)
        monkeypatch.setattr(cov, "annotate", lambda sim_dir, **k: ann)

    def test_default_on_by_default(self, monkeypatch):
        # LOCK: coverage line-cov gate is ON by default (floor 70).
        monkeypatch.delenv("CORESMITH_LINE_COV_GATE", raising=False)
        monkeypatch.delenv("CORESMITH_LINE_COV_FLOOR", raising=False)
        assert cov.line_cov_gate_enabled() is True
        assert cov.line_cov_floor() == 70.0

    def test_applicable_record(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CORESMITH_LINE_COV_GATE", raising=False)
        monkeypatch.delenv("CORESMITH_LINE_COV_FLOOR", raising=False)
        self._arm(tmp_path, monkeypatch, hit=9, uncov=1)  # 10/11 = 90.9%
        rec = cov.coverage_record(tmp_path)
        assert rec["applicable"] is True
        assert rec["points_total"] == 11 and rec["points_hit"] == 10
        assert rec["uncovered_count"] == 1
        assert rec["passed"] is True
        assert rec["floor"] == 70.0

    def test_not_applicable_no_coverage_dat_has_visible_reason(self, tmp_path, monkeypatch):
        # No coverage.dat -> not applicable, but a VISIBLE reason (never blank).
        monkeypatch.delenv("CORESMITH_LINE_COV_GATE", raising=False)
        rec = cov.coverage_record(tmp_path)
        assert rec["applicable"] is False
        assert "coverage.dat" in rec["reason"]

    def test_not_applicable_when_gate_disabled(self, tmp_path, monkeypatch):
        (tmp_path / "coverage.dat").write_text("x")
        monkeypatch.setenv("CORESMITH_LINE_COV_GATE", "0")
        rec = cov.coverage_record(tmp_path)
        assert rec["applicable"] is False
        assert "disabled" in rec["reason"]

    def test_na_reason_never_raises(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CORESMITH_LINE_COV_GATE", raising=False)
        # arbitrary non-existent dir -> reason string, no exception
        assert isinstance(cov.coverage_na_reason(tmp_path / "nope"), str)


class TestVerifyRtlCoverageRecording:
    def test_records_coverage_row(self, tmp_path, monkeypatch):
        (tmp_path / ".coresmith").mkdir()
        (tmp_path / "rtl").mkdir()
        (tmp_path / "rtl" / "adder.v").write_text("module adder(); endmodule")
        (tmp_path / "tb" / "cocotb").mkdir(parents=True)
        (tmp_path / "tb" / "cocotb" / "test_adder.py").write_text("# tb")
        spec = {"name": "adder", "rtl_target": "rtl/adder.v",
                "testbench": "tb/cocotb/test_adder.py"}

        import orchestrator.langgraph.pipeline_helpers as ph
        monkeypatch.setattr(ph, "lint_rtl", lambda *a, **k: {"clean": True})
        monkeypatch.setattr(ph, "run_simulation", lambda *a, **k: {
            "passed": True, "log": "PASS", "log_path": "", "tests_passed": 1,
            "tests_total": 1, "tests_failed": 0,
        })
        # coverage helpers -> deterministic annotated dir + summary.
        monkeypatch.setattr(cov, "annotate", lambda sim_dir, **k: tmp_path / "ann")
        monkeypatch.setattr(cov, "summarize", lambda ad, **k: {
            "points_total": 10, "points_hit": 8, "pct": 80.0,
            "uncovered": [{"file": "adder.v", "line": 3}],
        })

        sb = Scoreboard(tmp_path)
        V.verify_rtl(tmp_path, spec, no_equiv=True, coverage=True, scoreboard=sb)
        row = sb.coverage_latest("adder")
        assert row is not None
        assert row["points_hit"] == 8 and row["pct"] == 80.0
