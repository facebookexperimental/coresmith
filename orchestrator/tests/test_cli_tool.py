# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""PR2: ``coresmith tool <verb>`` + ``coresmith pdk info``.

The exit-code matrix and JSON schema are exercised against the ``mock``
deployment (in-process, no subprocess). The two real-tool smokes run through the
CLI via subprocess (so PROJECT_ROOT freezes at the tmp root) and are skipped
when the binary is absent.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

from orchestrator.harness import cli_tool
from orchestrator.pdk.registry import reset_deployment_cache

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CLI = _REPO_ROOT / "bin" / "coresmith"
_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "tiny_matmul.v"


@pytest.fixture(autouse=True)
def _reset_cache(monkeypatch):
    reset_deployment_cache()
    yield
    reset_deployment_cache()


def _ns(**kw):
    base = dict(
        project_root=None, design=None, rtl=None, script=None, netlist=None,
        sdc=None, gds=None, spice=None, out_dir=None, timeout_s=None, json=False,
        verb=None, emit_verb=None, out=None,
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# Exit-code matrix (mock deployment, in-process)
# ---------------------------------------------------------------------------
class TestExitCodeMatrix:
    @pytest.fixture(autouse=True)
    def _use_mock(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_DEPLOYMENT", "mock")
        reset_deployment_cache()

    def test_pass_exit_0(self, tmp_path):
        rc = cli_tool.cmd_tool_run(_ns(
            verb="run_synth", project_root=str(tmp_path),
            design="okblk", rtl=["x.v"]))
        assert rc == 0

    def test_fail_exit_1(self, tmp_path):
        rc = cli_tool.cmd_tool_run(_ns(
            verb="run_synth", project_root=str(tmp_path),
            design="fail_blk", rtl=["x.v"]))
        assert rc == 1

    def test_infra_exit_3(self, tmp_path):
        rc = cli_tool.cmd_tool_run(_ns(
            verb="run_synth", project_root=str(tmp_path),
            design="infra_blk", rtl=["x.v"]))
        assert rc == 3

    def test_capability_skip_exit_4(self, tmp_path):
        # mock deliberately omits run_lvs.
        rc = cli_tool.cmd_tool_run(_ns(
            verb="run_lvs", project_root=str(tmp_path),
            design="okblk", spice="a.sp", netlist="n.v"))
        assert rc == 4

    def test_missing_project_root_exit_2(self, monkeypatch):
        monkeypatch.delenv("CORESMITH_PROJECT_ROOT", raising=False)
        rc = cli_tool.cmd_tool_run(_ns(verb="run_synth", design="ok", rtl=["x.v"]))
        assert rc == 2


# ---------------------------------------------------------------------------
# JSON schemas + telemetry
# ---------------------------------------------------------------------------
class TestJsonAndTelemetry:
    @pytest.fixture(autouse=True)
    def _use_mock(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_DEPLOYMENT", "mock")
        reset_deployment_cache()

    def test_tool_list_schema(self, capsys):
        rc = cli_tool.cmd_tool_list(_ns(json=True))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["deployment"] == "mock"
        assert "run_synth" in payload["capabilities"]
        assert "run_lvs" not in payload["capabilities"]
        assert set(payload["verbs"]["run_synth"]) == {"impl", "checkers",
                                                       "prompt_notes"}

    def test_toolresult_json_schema(self, tmp_path, capsys):
        rc = cli_tool.cmd_tool_run(_ns(
            verb="run_synth", project_root=str(tmp_path),
            design="okblk", rtl=["x.v"], json=True))
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert set(out) == {"verb", "design", "ok", "tool_ok", "checks",
                            "artifacts", "log_path", "metrics"}
        assert out["ok"] is True and out["tool_ok"] is True
        assert out["checks"][0]["status"] == "pass"
        assert set(out["checks"][0]) == {"name", "status", "metrics", "details",
                                         "blocking"}

    def test_jsonl_record_written(self, tmp_path):
        cli_tool.cmd_tool_run(_ns(
            verb="run_synth", project_root=str(tmp_path),
            design="okblk", rtl=["x.v"]))
        jl = tmp_path / ".coresmith" / "tool_runs" / "tool_runs.jsonl"
        assert jl.exists()
        rec = json.loads(jl.read_text().splitlines()[-1])
        assert set(rec) >= {"ts", "verb", "design", "ok", "metrics"}
        assert rec["verb"] == "run_synth" and rec["ok"] is True

    def test_skip_is_recorded(self, tmp_path):
        cli_tool.cmd_tool_run(_ns(
            verb="run_lvs", project_root=str(tmp_path), design="okblk"))
        jl = tmp_path / ".coresmith" / "tool_runs" / "tool_runs.jsonl"
        rec = json.loads(jl.read_text().splitlines()[-1])
        assert rec["verb"] == "run_lvs" and rec["ok"] is False

    def test_pdk_info_schema(self, capsys):
        rc = cli_tool.cmd_pdk_info(_ns(json=True))
        assert rc == 0
        info = json.loads(capsys.readouterr().out)
        assert info["deployment"] == "mock"
        assert "capabilities" in info and "tools" in info and "pdk" in info


# ---------------------------------------------------------------------------
# emit-script
# ---------------------------------------------------------------------------
class TestEmitScript:
    def test_sky130_pnr_reference_emitted(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CORESMITH_DEPLOYMENT", raising=False)
        reset_deployment_cache()
        out = tmp_path / "pnr_work.tcl"
        rc = cli_tool.cmd_tool_emit_script(_ns(emit_verb="run_pnr", out=str(out)))
        assert rc == 0
        assert out.exists() and out.read_text().strip()

    def test_no_reference_script_skips(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_DEPLOYMENT", "mock")
        reset_deployment_cache()
        rc = cli_tool.cmd_tool_emit_script(
            _ns(emit_verb="run_synth", out=str(tmp_path / "x.ys")))
        assert rc == 4


# ---------------------------------------------------------------------------
# argparse wiring (subprocess, e2e)
# ---------------------------------------------------------------------------
class TestCliWiring:
    def test_tool_list_via_cli(self, tmp_path):
        env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "CORESMITH_DEPLOYMENT": "mock"}
        import os
        e = os.environ.copy()
        e.update(env)
        r = subprocess.run(
            [sys.executable, str(_CLI), "tool", "list", "--json"],
            capture_output=True, text=True, env=e, cwd=str(_REPO_ROOT))
        assert r.returncode == 0, r.stderr
        payload = json.loads(r.stdout)
        assert payload["deployment"] == "mock"


# ---------------------------------------------------------------------------
# Real-tool smoke (skipped when the binary is absent)
# ---------------------------------------------------------------------------
def _run_cli(args, project_root, extra_env=None):
    import os
    e = os.environ.copy()
    e.pop("CORESMITH_DEPLOYMENT", None)  # default -> sky130
    if extra_env:
        e.update(extra_env)
    return subprocess.run(
        [sys.executable, str(_CLI), *args,
         "--project-root", str(project_root), "--json"],
        capture_output=True, text=True, env=e, cwd=str(_REPO_ROOT))


@pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not on PATH")
def test_smoke_run_lint(tmp_path):
    r = _run_cli(["tool", "run_lint", "--rtl", str(_FIXTURE)], tmp_path)
    assert r.returncode == 0, r.stderr + r.stdout
    out = json.loads(r.stdout)
    assert out["ok"] is True and out["verb"] == "run_lint"


@pytest.mark.skipif(shutil.which("yosys") is None, reason="yosys not on PATH")
def test_smoke_run_synth_generic(tmp_path):
    r = _run_cli(
        ["tool", "run_synth", "--design", "tiny_matmul", "--rtl", str(_FIXTURE)],
        tmp_path, extra_env={"CORESMITH_SYNTH_GENERIC": "1"})
    assert r.returncode == 0, r.stderr + r.stdout
    out = json.loads(r.stdout)
    assert out["ok"] is True
    assert out["metrics"]["cells"] > 0, out["metrics"]
    assert out["metrics"]["ff_count"] > 0
    # PR3: the Yosys 0.65 box-format ("N cells") fix means gate_count is now the
    # recovered cell count, not a false 0.
    assert out["metrics"]["gate_count"] > 0, out["metrics"]
