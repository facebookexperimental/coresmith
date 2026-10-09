# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""``coresmith state check``: the deterministic rows-vs-disk audit."""
import argparse
import contextlib
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from orchestrator.harness import cli
from orchestrator.harness.cli_state import audit
from orchestrator.harness.tools.register import register
from orchestrator.state_store.project_db import open_project

_REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))
    return tmp_path


def _check(root, as_json=True):
    ap = argparse.ArgumentParser(prog="coresmith")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("state")   # bin/coresmith's daemon `state` verb
    s.add_argument("--project-root")
    s.set_defaults(func=lambda a: print("DAEMON_STATE"))
    cli.register_subcommands(sub)
    args = ap.parse_args(["state", "check", "--project-root", str(root), *(["--json"] if as_json else [])])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit) as exc:
        args.func(args)
    out = buf.getvalue()
    return exc.value.code, (json.loads(out) if as_json else out)


def _codes(probs):
    return sorted({(p["code"], p["where"]) for p in probs})


def test_clean_project_passes(root):
    open_project(root)
    rc, out = _check(root)
    assert rc == 0 and out["ok"] and out["problems"] == []


def test_every_problem_code(root):
    db = open_project(root)
    (root / "rtl").mkdir()
    (root / "rtl" / "a.v").write_text("module a; endmodule\n")
    db.import_block_specs([
        {"name": "a", "tier": 1, "rtl_target": "rtl/a.v", "testbench": "tb/cocotb/test_a.py"},
        {"name": "b", "tier": 1, "rtl_target": "rtl/b.v", "testbench": ""},
        {"name": "noc", "tier": 0, "rtl_target": "", "kind": "primitive", "primitive": "cs_fabric"},
    ])
    # a file-sourced artifact that changed, one that vanished, a DB-sourced one without its view
    (root / "arch" / "uarch_specs").mkdir(parents=True)
    sad = root / "arch" / "sad.md"
    sad.write_text("v1")
    db.register_artifact("sad", "arch/sad.md")
    sad.write_text("v2")
    gone = root / "arch" / "abi.md"
    gone.write_text("abi")
    db.register_artifact("abi", "arch/abi.md")
    gone.unlink()
    db.ensure_db_artifact("frd")
    db.upsert_item("frd", {"id": "PERF-001", "text": "fast", "priority": "must_have"})
    db.upsert_item("frd", {"id": "PERF-002", "text": "nice", "priority": "nice_to_have"})
    # a uArch spec on disk that was never registered; one that was
    (root / "arch" / "uarch_specs" / "a.md").write_text("# a")
    (root / "arch" / "uarch_specs" / "b.md").write_text("# b")
    db.register_artifact("uarch:b", "arch/uarch_specs/b.md")
    rc, out = _check(root)
    assert rc == 1 and not out["ok"]
    assert _codes(out["problems"]) == [
        ("ARTIFACT_MISSING", "abi"),
        ("ARTIFACT_STALE", "sad"),
        ("BLOCK_RTL_MISSING", "b"),
        ("BLOCK_TB_MISSING", "a"),
        ("FRD_UNVERIFIED", "PERF-001"),
        ("PRIMITIVE_UNMATERIALIZED", "noc"),
        ("UARCH_UNREGISTERED", "a"),
        ("VIEW_MISSING", "frd"),
    ]
    assert [p["severity"] for p in out["problems"] if p["code"] == "FRD_UNVERIFIED"] == ["warning"]
    stale = [p for p in out["problems"] if p["code"] == "ARTIFACT_STALE"][0]
    assert "coresmith register sad arch/sad.md" in stale["text"]
    rc, text = _check(root, as_json=False)
    assert rc == 1 and text.startswith("state check: 7 problem(s), 1 warning(s)") and "UARCH_UNREGISTERED" in text


def test_warnings_alone_exit_zero_and_ers_view(root):
    db = open_project(root)
    db.ensure_db_artifact("frd")
    db.upsert_item("frd", {"id": "INV-001", "text": "x", "priority": "must_have"})
    from orchestrator.harness.tools import render
    render.write_frd(db, root)
    rc, out = _check(root)
    assert rc == 0 and _codes(out["problems"]) == [("FRD_UNVERIFIED", "INV-001")]
    db.add_verifier("INV-001", "manual")
    # an ERS registered from arch/ writes the view; deleting it is VIEW_MISSING
    (root / "arch" / "ers.json").write_text(json.dumps({"ers": {"system_invariants": []}}))
    assert register(db, root, "ers", "arch/ers.json")["ok"]
    assert (root / ".coresmith" / "ers_spec.json").exists()
    assert _check(root)[0] == 0
    (root / ".coresmith" / "ers_spec.json").unlink()
    rc, out = _check(root)
    assert rc == 1 and _codes(out["problems"]) == [("VIEW_MISSING", "ers")]
    assert audit(db, root) == out["problems"]


def test_bin_coresmith_state_check(root):
    open_project(root)
    env = {"CORESMITH_PROJECT_ROOT": str(root), "PATH": "/usr/bin:/bin", "PYTHONPATH": str(_REPO)}
    p = subprocess.run([sys.executable, str(_REPO / "bin" / "coresmith"), "state", "check", "--project-root",
                        str(root)], capture_output=True, text=True, env=env, timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "state check: OK" in p.stdout


def _blocks_without_files(db):
    db.import_block_specs([
        {"name": "fft", "tier": 1, "rtl_target": "rtl/fft.v", "testbench": "tb/cocotb/test_fft.py"},
        {"name": "noc", "tier": 0, "rtl_target": "", "kind": "primitive", "primitive": "cs_fabric"},
    ])


def test_missing_rtl_is_info_before_the_blocks_stage(root):
    """Run 2 at `interfaces`: 11 errors, all expected before any RTL exists."""
    from orchestrator.state_store import stages as st
    db = open_project(root)
    _blocks_without_files(db)
    for i, s in enumerate(st.STAGES[:st.STAGES.index("interfaces")]):
        db.stage_set(s, i, "done")
    db.stage_set("interfaces", st.STAGES.index("interfaces"), "active")
    rc, out = _check(root)
    assert rc == 0 and out["ok"] and out["errors"] == 0 and out["info"] == 3
    assert {p["severity"] for p in out["problems"]} == {"info"}
    assert all("expected before the blocks stage; stage is interfaces" in p["text"] for p in out["problems"])
    rc, text = _check(root, as_json=False)
    assert rc == 0 and "3 expected at this stage (info)" in text and "[info] BLOCK_RTL_MISSING" in text
    # from `blocks` on they are errors again
    for i, s in enumerate(st.STAGES[:st.STAGES.index("blocks")]):
        db.stage_set(s, i, "done")
    db.stage_set("blocks", st.STAGES.index("blocks"), "active")
    rc, out = _check(root)
    assert rc == 1 and out["errors"] == 3 and out["info"] == 0


def test_no_stage_machine_keeps_errors(root):
    db = open_project(root)
    _blocks_without_files(db)      # a run that never used the stage machine: not judged by stage
    rc, out = _check(root)
    assert rc == 1 and out["errors"] == 3


def test_pins_view_and_shell_mismatch_warning(root):
    db = open_project(root)
    db.import_block_diagram({"blocks": [{"name": "uart", "interfaces": ["uart_tx"]}]})
    db.pin_add({"name": "uart_tx", "dir": "out", "block": "uart", "port": "uart_tx"})
    assert (root / ".coresmith" / "pins.json").exists()
    db.add_integration_snapshot({"top": "chip_top", "elaborated": True, "boundary_ports": 1, "boundary": ["uart_tx"]})
    rc, out = _check(root)
    assert "PINS_SHELL_MISMATCH" not in {p["code"] for p in out["problems"]}
    db.pin_add({"name": "clk", "dir": "in", "kind": "clock"})
    rc, out = _check(root)
    w = [p for p in out["problems"] if p["code"] == "PINS_SHELL_MISMATCH"]
    assert w and w[0]["severity"] == "warning" and "missing ['clk']" in w[0]["text"]
    (root / ".coresmith" / "pins.json").unlink()
    rc, out = _check(root)
    assert ("VIEW_MISSING", "pins") in _codes(out["problems"])
