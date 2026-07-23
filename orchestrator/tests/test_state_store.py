# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the engine state store (scoreboard + oracle trust)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from orchestrator.state_store.store import Scoreboard
from orchestrator.state_store.trust import (
    write_oracle_manifest,
    check_oracle_manifest,
)


@pytest.fixture
def pr(tmp_path) -> Path:
    (tmp_path / ".coresmith").mkdir()
    return tmp_path


class TestScoreboardSchema:
    def test_ensure_schema_creates_db(self, pr):
        sb = Scoreboard(pr)
        assert not sb.exists()
        assert sb.ensure_schema() is True
        assert sb.exists()

    def test_queries_on_absent_db_return_empty(self, pr):
        sb = Scoreboard(pr)
        assert sb.latest_dv() == []
        assert sb.dv_rows("blk") == []
        assert sb.latest_ppa("blk") is None
        assert sb.ppa_rows("blk") == []
        assert sb.coverage_latest("blk") is None


class TestScoreboardDv:
    def test_record_and_latest_dv(self, pr):
        sb = Scoreboard(pr)
        sb.record_dv(block="adder", scope="rtl", source="agent",
                     attempt=1, passed=False, tests_passed=3, tests_total=5)
        time.sleep(0.01)
        sb.record_dv(block="adder", scope="rtl", source="gate",
                     attempt=2, passed=True, tests_passed=5, tests_total=5,
                     seed=42)
        latest = sb.latest_dv(block="adder", scope="rtl")
        assert len(latest) == 1
        row = latest[0]
        assert row["passed"] == 1
        assert row["source"] == "gate"
        assert row["seed"] == 42
        assert row["attempt"] == 2
        # dv_rows returns the full history
        assert len(sb.dv_rows("adder")) == 2

    def test_latest_dv_per_scope(self, pr):
        sb = Scoreboard(pr)
        sb.record_dv(block="adder", scope="rtl", passed=True)
        sb.record_dv(block="adder", scope="synth", passed=False)
        sb.record_dv(block="mult", scope="rtl", passed=True)
        rows = sb.latest_dv()
        keyed = {(r["block"], r["scope"]): r for r in rows}
        assert keyed[("adder", "rtl")]["passed"] == 1
        assert keyed[("adder", "synth")]["passed"] == 0
        assert keyed[("mult", "rtl")]["passed"] == 1

    def test_first_divergence_roundtrips_json(self, pr):
        sb = Scoreboard(pr)
        div = {"signal": "m_axis_tdata", "cycle": 17, "got": 3, "want": 4}
        sb.record_dv(block="codec", scope="chip", passed=False,
                     first_divergence=div)
        row = sb.latest_dv(block="codec")[0]
        assert json.loads(row["first_divergence"]) == div


class TestScoreboardPpa:
    def test_record_and_latest_ppa(self, pr):
        sb = Scoreboard(pr)
        sb.record_ppa(block="adder", attempt=1, probe="generic",
                      ff=10, mem_bits=0, elaborated=True, ppa_ok=None)
        sb.record_ppa(block="adder", attempt=2, probe="cellcount",
                      cells=250, elaborated=True, ppa_ok=True,
                      reasons=["ff within budget"])
        latest = sb.latest_ppa("adder")
        assert latest["probe"] == "cellcount"
        assert latest["cells"] == 250
        assert latest["ppa_ok"] == 1
        assert json.loads(latest["reasons"]) == ["ff within budget"]
        assert len(sb.ppa_rows("adder")) == 2

    def test_ppa_ok_none_stays_null(self, pr):
        sb = Scoreboard(pr)
        sb.record_ppa(block="adder", probe="generic", ppa_ok=None)
        assert sb.latest_ppa("adder")["ppa_ok"] is None


class TestScoreboardCoverage:
    def test_record_and_latest_coverage(self, pr):
        sb = Scoreboard(pr)
        sb.record_coverage(block="adder", scope="rtl", points_total=100,
                           points_hit=80, pct=80.0,
                           uncovered=[{"file": "adder.v", "line": 12}])
        row = sb.coverage_latest("adder")
        assert row["points_total"] == 100
        assert row["pct"] == 80.0
        assert json.loads(row["uncovered"])[0]["line"] == 12


class TestScoreboardBestEffort:
    def test_write_never_raises_on_bad_root(self, tmp_path):
        # project_root/.coresmith is a FILE -> mkdir/connect fails -> False,
        # never raises (a scoreboard failure must not fail a pipeline node).
        bad = tmp_path / "run"
        bad.mkdir()
        (bad / ".coresmith").write_text("i am a file, not a dir")
        sb = Scoreboard(bad)
        assert sb.record_dv(block="x", scope="rtl", passed=True) is False
        assert sb.ensure_schema() is False
        assert sb.latest_dv() == []


class TestOracleManifest:
    def _seed(self, root: Path):
        (root / "inputs").mkdir()
        (root / "inputs" / "golden.py").write_text("def golden(x):\n    return x\n")
        (root / "inputs" / "requirements.md").write_text("do the thing")
        (root / "arch").mkdir()
        (root / "arch" / "ers_spec.md").write_text("ERS spec")

    def test_write_then_clean_check_ok(self, pr):
        self._seed(pr)
        manifest = write_oracle_manifest(pr)
        assert manifest is not None
        assert (pr / ".coresmith" / "oracle_manifest.json").exists()
        # golden + requirements + ers should all be hashed
        assert any("golden.py" in k for k in manifest["files"])
        res = check_oracle_manifest(pr)
        assert res["ok"] is True
        assert res["checked"] is True

    def test_no_manifest_is_non_blocking(self, pr):
        self._seed(pr)
        res = check_oracle_manifest(pr)
        assert res["ok"] is True
        assert res["checked"] is False
        assert res["violation"] is None

    def test_tampered_golden_flags_oracle_tamper(self, pr):
        self._seed(pr)
        write_oracle_manifest(pr)
        (pr / "inputs" / "golden.py").write_text("def golden(x):\n    return 0\n")
        res = check_oracle_manifest(pr)
        assert res["ok"] is False
        assert any("golden.py" in c for c in res["changed"])
        assert res["violation"]["category"] == "ORACLE_TAMPER"

    def test_missing_oracle_file_flags(self, pr):
        self._seed(pr)
        write_oracle_manifest(pr)
        (pr / "arch" / "ers_spec.md").unlink()
        res = check_oracle_manifest(pr)
        assert res["ok"] is False
        assert any("ers_spec" in m for m in res["missing"])
