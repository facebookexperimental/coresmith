# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the harness CLI (read-only queries + import isolation).

Verify-subcommand tests live alongside these once the verify module lands.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from orchestrator.harness import blocks as harness_blocks
from orchestrator.harness import cli as harness_cli
from orchestrator.harness import env as harness_env
from orchestrator.state_store.store import Scoreboard

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _args(project_root, **kw):
    base = dict(project_root=str(project_root), json=True)
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture(autouse=True)
def _guard_project_root(monkeypatch):
    # bootstrap_project_root sets CORESMITH_PROJECT_ROOT; guard it so a test
    # can't leak a project root into a later test's frozen imports.
    monkeypatch.setenv("CORESMITH_PROJECT_ROOT", "/nonexistent-guard")


# ---------------------------------------------------------------------------
# B5 risk: harness.cli must import WITHOUT importing orchestrator.langgraph.
# ---------------------------------------------------------------------------
class TestImportIsolation:
    def test_cli_imports_without_langgraph(self):
        code = (
            "import sys\n"
            "import orchestrator.harness.cli as cli\n"
            "assert hasattr(cli, 'register_subcommands')\n"
            "leaked = [m for m in sys.modules "
            "          if m.startswith('orchestrator.langgraph')]\n"
            "assert not leaked, leaked\n"
            "print('IMPORT_OK')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, cwd=str(_REPO_ROOT),
        )
        assert result.returncode == 0, result.stderr
        assert "IMPORT_OK" in result.stdout


class TestEnvBootstrap:
    def test_bootstrap_sets_env(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CORESMITH_PROJECT_ROOT", raising=False)
        root = harness_env.bootstrap_project_root(str(tmp_path))
        assert root == tmp_path.resolve()
        import os
        assert os.environ["CORESMITH_PROJECT_ROOT"] == str(tmp_path.resolve())

    def test_bootstrap_reads_env_when_no_arg(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))
        assert harness_env.bootstrap_project_root(None) == tmp_path.resolve()

    def test_bootstrap_raises_without_root(self, monkeypatch):
        monkeypatch.delenv("CORESMITH_PROJECT_ROOT", raising=False)
        with pytest.raises(ValueError):
            harness_env.bootstrap_project_root(None)

    def test_child_env_has_paths_and_pythonhome(self):
        env = harness_env.harness_child_env()
        assert "PATH" in env


class TestBlockResolution:
    def test_block_specs_wins(self, tmp_path):
        cs = tmp_path / ".coresmith"
        cs.mkdir()
        (cs / "block_specs.json").write_text(json.dumps([{"name": "a"}, {"name": "b"}]))
        (cs / "block_queue.json").write_text(json.dumps([{"name": "z"}]))
        q = harness_blocks.load_block_queue(tmp_path)
        assert [b["name"] for b in q] == ["a", "b"]

    def test_block_queue_fallback(self, tmp_path):
        cs = tmp_path / ".coresmith"
        cs.mkdir()
        (cs / "block_queue.json").write_text(json.dumps([{"name": "q"}]))
        assert harness_blocks.load_block_spec(tmp_path, "q") == {"name": "q"}
        assert harness_blocks.load_block_spec(tmp_path, "nope") is None

    def test_persist_block_queue(self, tmp_path):
        (tmp_path / ".coresmith").mkdir()
        assert harness_blocks.persist_block_queue(tmp_path, [{"name": "p"}]) is True
        assert harness_blocks.block_names(tmp_path) == ["p"]


class TestDvStatus:
    def test_scoreboard_rows(self, tmp_path, capsys):
        (tmp_path / ".coresmith").mkdir()
        sb = Scoreboard(tmp_path)
        sb.record_dv(block="adder", scope="rtl", source="gate", passed=True,
                     tests_passed=5, tests_total=5)
        rc = harness_cli.cmd_dv_status(_args(tmp_path, block=None))
        assert rc == harness_cli.EXIT_PASS
        out = json.loads(capsys.readouterr().out)
        assert out["source"] == "scoreboard"
        assert out["rows"][0]["block"] == "adder"
        assert out["rows"][0]["passed"] == 1

    def test_stale_marker(self, tmp_path, capsys):
        (tmp_path / ".coresmith").mkdir()
        sb = Scoreboard(tmp_path)
        sb.record_dv(block="adder", scope="rtl", passed=True)
        # best_result.json written AFTER the row -> newer mtime -> stale.
        bdir = tmp_path / ".coresmith" / "blocks" / "adder"
        bdir.mkdir(parents=True)
        import time as _t
        _t.sleep(0.02)
        (bdir / "best_result.json").write_text(json.dumps({"sim_passed": True}))
        harness_cli.cmd_dv_status(_args(tmp_path, block="adder"))
        out = json.loads(capsys.readouterr().out)
        assert out["rows"][0]["stale"] is True

    def test_disk_fallback_no_scoreboard(self, tmp_path, capsys):
        bdir = tmp_path / ".coresmith" / "blocks" / "adder"
        bdir.mkdir(parents=True)
        (bdir / "best_result.json").write_text(
            json.dumps({"sim_passed": True, "attempt": 3,
                        "tests_passed": 4, "tests_total": 4})
        )
        rc = harness_cli.cmd_dv_status(_args(tmp_path, block=None))
        assert rc == harness_cli.EXIT_PASS
        out = json.loads(capsys.readouterr().out)
        assert out["source"] == "disk"
        assert out["rows"][0]["block"] == "adder"
        assert out["rows"][0]["passed"] is True


class TestPpa:
    def test_scoreboard(self, tmp_path, capsys):
        (tmp_path / ".coresmith").mkdir()
        sb = Scoreboard(tmp_path)
        sb.record_ppa(block="adder", probe="generic", ff=12, elaborated=True)
        rc = harness_cli.cmd_ppa(_args(tmp_path, block="adder", history=False))
        assert rc == harness_cli.EXIT_PASS
        out = json.loads(capsys.readouterr().out)
        assert out["rows"][0]["ff"] == 12

    def test_disk_fallback_synth_report(self, tmp_path, capsys):
        rep = tmp_path / "syn" / "output" / "adder"
        rep.mkdir(parents=True)
        # A minimal yosys stat cell line: "<flop-cell> <count>".
        (rep / "adder_report.txt").write_text(
            "=== adder ===\n\n   $_DFF_P_ 8\n   $_AND_ 3\n"
        )
        rc = harness_cli.cmd_ppa(_args(tmp_path, block="adder", history=False))
        assert rc == harness_cli.EXIT_PASS
        out = json.loads(capsys.readouterr().out)
        assert out["source"] == "disk"
        assert out["ff"] == 8

    def test_no_data_skip(self, tmp_path, capsys):
        (tmp_path / ".coresmith").mkdir()
        rc = harness_cli.cmd_ppa(_args(tmp_path, block="ghost", history=False))
        assert rc == harness_cli.EXIT_SKIP


class TestCoverage:
    def test_no_data_skip(self, tmp_path, capsys):
        (tmp_path / ".coresmith").mkdir()
        rc = harness_cli.cmd_coverage(_args(tmp_path, block="adder", uncovered=False))
        assert rc == harness_cli.EXIT_SKIP

    def test_scoreboard_row(self, tmp_path, capsys):
        (tmp_path / ".coresmith").mkdir()
        sb = Scoreboard(tmp_path)
        sb.record_coverage(block="adder", points_total=10, points_hit=7, pct=70.0,
                           uncovered=[{"file": "adder.v", "line": 4, "text": "x"}])
        rc = harness_cli.cmd_coverage(_args(tmp_path, block="adder", uncovered=True))
        assert rc == harness_cli.EXIT_PASS
        out = json.loads(capsys.readouterr().out)
        assert out["points_hit"] == 7
        assert out["uncovered"][0]["line"] == 4


class TestContracts:
    def test_empty_contracts_ok(self, tmp_path, capsys):
        (tmp_path / ".coresmith").mkdir()
        rc = harness_cli.cmd_contracts(_args(tmp_path, block="adder"))
        assert rc == harness_cli.EXIT_PASS
        out = json.loads(capsys.readouterr().out)
        assert out["block"] == "adder"
        assert out["edges"] == []


class TestVerifyRegistration:
    def test_verify_and_queries_registered(self):
        import argparse
        ap = argparse.ArgumentParser()
        sub = ap.add_subparsers(dest="cmd")
        harness_cli.register_subcommands(sub)
        args = ap.parse_args(["verify", "model", "adder", "--json"])
        assert args.verify_cmd == "model"
        assert callable(args.func)
        # read-only queries too
        q = ap.parse_args(["dv-status", "--json"])
        assert callable(q.func)

    def test_verify_rtl_handler_dispatch(self, tmp_path, monkeypatch, capsys):
        from orchestrator.harness import cli_verify
        from orchestrator.harness import verify as V
        (tmp_path / ".coresmith").mkdir()
        (tmp_path / ".coresmith" / "block_queue.json").write_text(
            json.dumps([{"name": "adder", "rtl_target": "rtl/adder.v"}])
        )
        monkeypatch.setattr(
            V, "verify_rtl",
            lambda *a, **k: V.VerifyResult(True, verdict="ok"),
        )
        args = SimpleNamespace(
            project_root=str(tmp_path), json=True, block="adder", seed=None,
            tb=None, no_equiv=True, lint_only=False, coverage=False,
        )
        rc = cli_verify.cmd_verify_rtl(args)
        assert rc == 0
        assert json.loads(capsys.readouterr().out)["passed"] is True

    def test_verify_model_unknown_needs_block(self, tmp_path, monkeypatch, capsys):
        from orchestrator.harness import cli_verify
        (tmp_path / ".coresmith").mkdir()
        args = SimpleNamespace(project_root=str(tmp_path), json=True,
                               block=None, all=False, skip_size=False)
        assert cli_verify.cmd_verify_model(args) == cli_verify.EXIT_USAGE
