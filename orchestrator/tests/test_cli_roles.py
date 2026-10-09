# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Role-scoped CLI: ``CORESMITH_ROLE=architect|worker|watchdog``."""
import argparse
import contextlib
import io
import subprocess
import sys
from pathlib import Path

import pytest

from orchestrator.harness import cli
from orchestrator.harness.cli import role_allows, role_refusal, role_verb
from orchestrator.state_store.project_db import open_project

_REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("argv,verb", [
    (["frd", "add", "x", "--id", "PERF-1"], ("frd", "add")),
    (["--project-root", "/x", "check", "add", "PERF-1"], ("check", "add")),
    (["frd", "--json"], ("frd", None)),
    (["--help"], (None, None)),
])
def test_role_verb(argv, verb):
    assert role_verb(argv) == verb


@pytest.mark.parametrize("role,argv,ok", [
    ("", ["daemon", "start"], True),                     # unset: no restriction
    ("watchdog", ["daemon", "start"], True),
    ("watchdog", ["state"], True),
    ("watchdog", ["interrupts", "--resolve", "int-1"], True),
    ("watchdog", ["frd", "add", "x"], False),
    ("watchdog", ["run", "start"], True),                # ... and the pipeline run
    ("watchdog", ["run", "pause"], True),
    ("watchdog", ["run", "restart-node", "x"], False),
    ("watchdog", ["stage", "status"], True),
    ("watchdog", ["stage", "next"], False),              # no design authority
    ("watchdog", ["question", "list"], True),
    ("watchdog", ["question", "answer", "Q1"], False),
    ("watchdog", ["state", "check"], True),
    ("watchdog", ["pin", "add", "x"], False),
    ("worker", ["block-done", "fft64"], True),
    ("worker", ["verify", "rtl", "fft64"], True),
    ("worker", ["frd", "verifier", "PERF-1", "--kind", "cocotb"], True),
    ("worker", ["frd", "show", "PERF-1"], True),
    ("worker", ["frd"], True),                           # bare frd = list
    ("worker", ["frd", "add", "x"], False),
    ("worker", ["frd", "edit", "PERF-1"], False),
    ("worker", ["check", "add", "PERF-1", "block_dv"], True),
    ("worker", ["check", "list"], False),
    ("worker", ["question", "add", "why?"], True),
    ("worker", ["question", "answer", "Q1"], False),
    ("worker", ["contract", "set", "e", "k", "v"], False),
    ("worker", ["contracts", "fft64"], True),
    ("worker", ["stage", "next"], False),
    ("worker", ["daemon", "stop"], False),
    ("worker", ["actions"], True),
    ("architect", ["frd", "add", "x"], True),
    ("architect", ["contract", "unlock", "--reason", "r"], True),
    ("architect", ["run", "restart-node", "x"], True),
    ("architect", ["run", "start"], True),               # the Architect starts the frontend itself
    ("architect", ["run", "pause"], True),
    ("architect", ["daemon", "status"], True),
    ("architect", ["daemon", "start"], True),
    ("architect", ["model", "author", "--block", "x"], True),
    ("chip_lead", ["status"], False),                    # unknown role: fail closed
    ("worker", ["--help"], True),
])
def test_role_allows(role, argv, ok):
    assert role_allows(role, argv) is ok


def test_role_refusal_text():
    assert role_refusal("worker", ["frd", "add", "x"]) == "ROLE_FORBIDDEN worker may not run frd add"
    assert role_refusal("worker", ["daemon", "start"]) == "ROLE_FORBIDDEN worker may not run daemon"
    assert role_refusal("chip_lead", ["status"]).startswith("ROLE_FORBIDDEN chip_lead may not run status (unknown role")


def test_architect_role_is_identity_only():
    """``CORESMITH_ROLE=architect`` is compatibility metadata for the actions
    log: it never forbids the Architect anything, the frontend start included."""
    from orchestrator.harness.cli import ROLE_DENY
    assert ROLE_DENY["architect"] == {}
    for argv in (["run", "start"], ["run", "pause"], ["daemon", "start"], ["stage", "next"],
                 ["model", "eval"], ["harness", "author"], ["resume", "--action", "retry"]):
        assert role_allows("architect", argv), argv


def _inproc(root, argv):
    ap = argparse.ArgumentParser(prog="coresmith")
    sub = ap.add_subparsers(dest="cmd")
    cli.register_subcommands(sub)
    args = ap.parse_args([*argv, "--project-root", str(root)])
    args._argv = argv
    buf, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err), pytest.raises(SystemExit) as exc:
        args.func(args)
    return exc.value.code, buf.getvalue(), err.getvalue()


def test_run_wrapper_enforces_role_and_records_actor(tmp_path, monkeypatch):
    monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))
    db = open_project(tmp_path)
    monkeypatch.setenv("CORESMITH_ROLE", "worker")
    rc, out, err = _inproc(tmp_path, ["frd", "add", "fast", "--id", "INV-001"])
    assert rc == 2 and "ROLE_FORBIDDEN worker may not run frd add" in err
    assert db.item("INV-001") is None
    rc, out, err = _inproc(tmp_path, ["question", "add", "why?"])
    assert rc == 0, err
    acts = db.actions()
    assert [(a["argv"][:2], a["rc"], a["actor"]) for a in acts] == [
        (["frd", "add"], 2, "worker"), (["question", "add"], 0, "worker")]
    assert acts[0]["summary"].startswith("ROLE_FORBIDDEN")
    assert db.questions()[0]["asked_by"] == "worker"
    monkeypatch.delenv("CORESMITH_ROLE")
    rc, out, err = _inproc(tmp_path, ["frd", "add", "fast", "--id", "INV-001"])
    assert rc == 0 and db.actions()[-1]["actor"] == "cli"


def _bin(root, *a, role=None):
    env = {"CORESMITH_PROJECT_ROOT": str(root), "PATH": "/usr/bin:/bin", "PYTHONPATH": str(_REPO)}
    if role is not None:
        env["CORESMITH_ROLE"] = role
    return subprocess.run([sys.executable, str(_REPO / "bin" / "coresmith"), *a], capture_output=True, text=True,
                          env=env, timeout=120)


def test_bin_coresmith_enforces_roles_before_dispatch(tmp_path):
    open_project(tmp_path)
    # daemon-lifecycle verbs are dispatched before the harness: the hook covers them
    p = _bin(tmp_path, "daemon", "start", "--project-root", str(tmp_path), role="worker")
    assert p.returncode == 2 and "ROLE_FORBIDDEN worker may not run daemon" in p.stderr
    assert not (tmp_path / ".coresmith" / "daemon.json").exists()
    # the Architect may start the frontend: no role refusal (no daemon here, so the
    # client fails later with its own "no daemon" error, never ROLE_FORBIDDEN)
    p = _bin(tmp_path, "run", "start", "--project-root", str(tmp_path), role="architect")
    assert "ROLE_FORBIDDEN" not in p.stderr and "no daemon" in p.stderr
    p = _bin(tmp_path, "frd", "add", "x", "--id", "INV-001", "--project-root", str(tmp_path), role="watchdog")
    assert p.returncode == 2 and "ROLE_FORBIDDEN watchdog may not run frd" in p.stderr
    # the refusals before dispatch land in the actions log (rc 2, actor = the role)
    refused = [a for a in open_project(tmp_path).actions() if str(a.get("summary") or "").startswith("ROLE_FORBIDDEN")]
    assert [(a["rc"], a["actor"]) for a in refused] == [(2, "worker"), (2, "watchdog")]
    assert refused[-1]["argv"][:2] == ["frd", "add"]
    p = _bin(tmp_path, "status", "--project-root", str(tmp_path), role="watchdog")
    assert p.returncode == 0, p.stderr
    assert open_project(tmp_path).actions()[-1]["actor"] == "watchdog"
    p = _bin(tmp_path, "frd", "add", "x", "--id", "INV-001", "--project-root", str(tmp_path), role="architect")
    assert p.returncode == 0, p.stderr
