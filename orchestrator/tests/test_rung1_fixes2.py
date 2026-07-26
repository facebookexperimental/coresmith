# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Rung-1 repeat-run robustness/observability fixes (commit [rung1-fixes-2]).

Defect 1: the `coresmith` CLI was not resolvable inside codex agent shells
(/etc/profile resets PATH). Fixes: daemon start exports CORESMITH_CLI, the
verify skill instructs `$CORESMITH_CLI`, and a best-effort /usr/local/bin
symlink helper.

Defect 2: validation_dv's sim clobbered integration_dv's raw sim log (both wrote
step_logs/integration/integration_sim_attempt<N>.log). Fix: `run_integration_
simulation(sim_scope=...)` namespaces the sim dir + step log.

Defect 3: profile seeding was unobservable. Fix: profile.apply() ALWAYS logs a
status line (profile + seeded + already-set), re-emittable after logging is up.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from pathlib import Path

import pytest

from orchestrator import profile
from orchestrator.harness.env import ensure_cli_symlink

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Defect 3 -- profile.apply() always logs a status line
# ---------------------------------------------------------------------------
class TestProfileStatusLogging:
    def test_logs_when_seeding(self, monkeypatch, caplog):
        monkeypatch.setenv("CORESMITH_PROFILE", "strict")
        for key in profile.STRICT_DEFAULTS:
            monkeypatch.delenv(key, raising=False)
        profile.reset()
        with caplog.at_level(logging.INFO):
            seeded = profile.apply()
        assert set(seeded) == set(profile.STRICT_DEFAULTS)
        lines = [m for m in (r.getMessage() for r in caplog.records)
                 if "coresmith profile=" in m]
        assert lines, "apply() must log a profile status line"
        line = lines[-1]
        assert "profile=strict" in line
        assert "seeded=[" in line and "already_set=[" in line
        # A seeded key is named; nothing was pre-set.
        assert "CORESMITH_PPA_GATE" in line
        assert "already_set=[(none)]" in line

    def test_logs_when_nothing_seeded(self, monkeypatch, caplog):
        monkeypatch.setenv("CORESMITH_PROFILE", "strict")
        for key, val in profile.STRICT_DEFAULTS.items():
            monkeypatch.setenv(key, val)  # already set -> apply() seeds nothing
        profile.reset()
        with caplog.at_level(logging.INFO):
            seeded = profile.apply()
        assert seeded == []
        lines = [m for m in (r.getMessage() for r in caplog.records)
                 if "coresmith profile=" in m]
        assert lines, "apply() must log even when it seeds nothing"
        line = lines[-1]
        assert "profile=strict" in line
        assert "seeded=[(none)]" in line
        # The pre-set keys are reported as already_set.
        assert "already_set=[" in line
        assert "CORESMITH_PPA_GATE" in line

    def test_log_status_reemits_without_reseeding(self, monkeypatch, caplog):
        monkeypatch.setenv("CORESMITH_PROFILE", "strict")
        for key in profile.STRICT_DEFAULTS:
            monkeypatch.delenv(key, raising=False)
        profile.reset()
        profile.apply()
        # A second apply() is idempotent (no re-log); log_status() re-emits.
        with caplog.at_level(logging.INFO):
            profile.log_status()
        lines = [m for m in (r.getMessage() for r in caplog.records)
                 if "coresmith profile=" in m]
        assert lines and "profile=strict" in lines[-1]


# ---------------------------------------------------------------------------
# Defect 2 -- integration vs validation sims use distinct log + sim namespaces
# ---------------------------------------------------------------------------
class TestSimScopeNamespacing:
    def test_integration_and_validation_paths_differ(self, tmp_path, monkeypatch):
        from orchestrator.langgraph import integration_helpers as ih
        from orchestrator.langgraph import pipeline_helpers as ph

        monkeypatch.setattr(ih, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(ph, "_LOG_DIR", tmp_path / ".coresmith" / "step_logs")

        top = tmp_path / "chip_top.v"
        top.write_text("module chip_top(input clk);\nendmodule\n")
        tb = tmp_path / "test_chip_top.py"
        tb.write_text("import cocotb\n\n@cocotb.test()\nasync def t(dut):\n    pass\n")

        def _fake_run(cmd, *a, **k):
            return subprocess.CompletedProcess(cmd, 0, "** TESTS=1 PASS=1 FAIL=0 **", "")

        monkeypatch.setattr(ih.subprocess, "run", _fake_run)
        monkeypatch.setattr(ih, "run_wavekit_vcd_audit", lambda *a, **k: {"ok": True})

        res_int = ih.run_integration_simulation("chip", str(top), {}, str(tb))
        res_val = ih.run_integration_simulation(
            "chip", str(top), {}, str(tb), sim_scope="validation")

        # The two raw sim logs must NOT be the same file (evidence-loss bug).
        assert res_int["log_path"] != res_val["log_path"]
        assert res_int["log_path"].endswith("integration_sim_attempt1.log")
        assert res_val["log_path"].endswith("validation_sim_attempt1.log")
        assert Path(res_int["log_path"]).exists()
        assert Path(res_val["log_path"]).exists()
        # Distinct sim build dirs avoid fingerprint churn between the two runs.
        assert (tmp_path / "sim_build" / "integration" / "Makefile").exists()
        assert (tmp_path / "sim_build" / "validation" / "Makefile").exists()

    def test_default_scope_is_integration(self, tmp_path, monkeypatch):
        # Byte-compatible default: no sim_scope -> historical integration paths.
        from orchestrator.langgraph import integration_helpers as ih
        from orchestrator.langgraph import pipeline_helpers as ph

        monkeypatch.setattr(ih, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(ph, "_LOG_DIR", tmp_path / ".coresmith" / "step_logs")
        top = tmp_path / "chip_top.v"
        top.write_text("module chip_top(input clk);\nendmodule\n")
        tb = tmp_path / "test_chip_top.py"
        tb.write_text("import cocotb\n\n@cocotb.test()\nasync def t(dut):\n    pass\n")
        monkeypatch.setattr(
            ih.subprocess, "run",
            lambda cmd, *a, **k: subprocess.CompletedProcess(cmd, 0, "** TESTS=1 PASS=1 FAIL=0 **", ""))
        monkeypatch.setattr(ih, "run_wavekit_vcd_audit", lambda *a, **k: {"ok": True})
        res = ih.run_integration_simulation("chip", str(top), {}, str(tb))
        assert res["log_path"].endswith("integration_sim_attempt1.log")
        assert (tmp_path / "sim_build" / "integration" / "Makefile").exists()


# ---------------------------------------------------------------------------
# Defect 1 -- CLI resolvable in agent shells
# ---------------------------------------------------------------------------
class TestVerifySkillText:
    def test_skill_instructs_cli_var(self):
        skill = (_REPO_ROOT / "orchestrator" / "langchain" / "prompts"
                 / "skills" / "verify_in_context.md").read_text()
        assert "$CORESMITH_CLI" in skill or "CORESMITH_CLI" in skill
        # Fallback-to-plain-coresmith pattern is documented.
        assert "CORESMITH_CLI:-coresmith" in skill


class TestEnsureCliSymlink:
    def _make_cli(self, tmp_path):
        cli = tmp_path / "repo" / "bin" / "coresmith"
        cli.parent.mkdir(parents=True)
        cli.write_text("#!/bin/sh\necho hi\n")
        return cli

    def test_creates_symlink(self, tmp_path):
        cli = self._make_cli(tmp_path)
        bindir = tmp_path / "usrlocalbin"
        bindir.mkdir()
        link = ensure_cli_symlink(str(cli), bindir=str(bindir))
        assert link is not None
        assert link.is_symlink()
        assert link.resolve() == cli.resolve()

    def test_idempotent(self, tmp_path):
        cli = self._make_cli(tmp_path)
        bindir = tmp_path / "b"
        bindir.mkdir()
        first = ensure_cli_symlink(str(cli), bindir=str(bindir))
        second = ensure_cli_symlink(str(cli), bindir=str(bindir))
        assert first is not None and second is not None
        assert second.resolve() == cli.resolve()

    def test_refreshes_stale_symlink(self, tmp_path):
        cli_old = self._make_cli(tmp_path)
        cli_new = tmp_path / "repo2" / "bin" / "coresmith"
        cli_new.parent.mkdir(parents=True)
        cli_new.write_text("#!/bin/sh\necho new\n")
        bindir = tmp_path / "b"
        bindir.mkdir()
        ensure_cli_symlink(str(cli_old), bindir=str(bindir))
        link = ensure_cli_symlink(str(cli_new), bindir=str(bindir))
        assert link is not None
        assert link.resolve() == cli_new.resolve()

    def test_absent_bindir_returns_none(self, tmp_path):
        cli = self._make_cli(tmp_path)
        assert ensure_cli_symlink(str(cli), bindir=str(tmp_path / "nope")) is None

    def test_wont_clobber_real_file(self, tmp_path):
        cli = self._make_cli(tmp_path)
        bindir = tmp_path / "b"
        bindir.mkdir()
        real = bindir / "coresmith"
        real.write_text("i am a real binary, not a symlink")
        assert ensure_cli_symlink(str(cli), bindir=str(bindir)) is None
        assert not real.is_symlink()
        assert real.read_text().startswith("i am a real binary")

    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores write bits")
    def test_unwritable_bindir_returns_none(self, tmp_path):
        cli = self._make_cli(tmp_path)
        bindir = tmp_path / "ro"
        bindir.mkdir(mode=0o500)
        try:
            assert ensure_cli_symlink(str(cli), bindir=str(bindir)) is None
        finally:
            bindir.chmod(0o700)


class TestDaemonStartExportsCli:
    def _load_cli(self):
        # bin/coresmith has no .py extension -> use an explicit source loader.
        import importlib.machinery
        path = _REPO_ROOT / "bin" / "coresmith"
        loader = importlib.machinery.SourceFileLoader(
            "coresmith_cli_under_test", str(path))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        mod = importlib.util.module_from_spec(spec)
        loader.exec_module(mod)
        return mod

    def test_daemon_env_carries_cli_path(self, tmp_path, monkeypatch):
        mod = self._load_cli()
        # Keep the symlink attempt hermetic (no /usr/local/bin side effects).
        monkeypatch.setattr(
            "orchestrator.harness.env.ensure_cli_symlink", lambda *a, **k: None)

        project_root = mod._project_root(str(tmp_path))
        df = mod._daemon_file(project_root)
        captured: dict = {}

        class _FakeProc:
            returncode = 0

            def poll(self):
                return None

        def _fake_popen(cmd, cwd=None, env=None, **kw):
            captured["env"] = env
            df.parent.mkdir(parents=True, exist_ok=True)
            df.write_text(json.dumps({"pid": 999999, "port": 54321}))
            return _FakeProc()

        monkeypatch.setattr(mod.subprocess, "Popen", _fake_popen)
        args = argparse.Namespace(project_root=str(tmp_path), port=0)
        mod._daemon_start(args)

        assert "CORESMITH_CLI" in captured["env"]
        cli = captured["env"]["CORESMITH_CLI"]
        assert Path(cli).name == "coresmith"
        assert cli == str(_REPO_ROOT / "bin" / "coresmith")
