# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Requirement-level CLI verbs: ``coresmith prd|frd ...``, ``state write``,
``check add --value`` and per-item ``block_dv`` stamping in ``block-done``."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

from orchestrator.harness import cli, cli_frd
from orchestrator.harness.tools import block as bt
from orchestrator.harness.tools import extract, render
from orchestrator.harness.tools.register import register
from orchestrator.state_store import stages as st
from orchestrator.state_store.project_db import open_project
from orchestrator.systemc_model import frd_eval as fe

_REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _root_env(tmp_path, monkeypatch):
    # the in-process verbs export CORESMITH_PROJECT_ROOT; keep it test-scoped
    monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))


def _parser():
    ap = argparse.ArgumentParser(prog="coresmith")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("state")  # bin/coresmith's daemon `state` verb
    s.add_argument("--project-root")
    s.set_defaults(func=lambda a: print("DAEMON_STATE"))
    cli.register_subcommands(sub)
    return ap


def _run(root, *argv):
    """Run one verb in-process; returns (rc, parsed --json payload)."""
    args = _parser().parse_args([*argv, "--project-root", str(root), "--json"])
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit) as exc:
        args.func(args)
    out = buf.getvalue()
    return exc.value.code, (json.loads(out) if out.strip().startswith("{") else out)


def _perf(root, *extra):
    return _run(root, "frd", "add", "Sustain 60 fps at 1080p", "--id", "PERF-001", "--metric", "fps",
                "--min", "55", "--unit", "fps", "--owner", "fft64", "--derives-from", "KPI-FPS-1",
                "--acceptance", "throughput.py mean >= 55 fps", "--model-check", "arch model frame cycle count", *extra)


# -- frd add ------------------------------------------------------------------
def test_perf_without_bounds_is_refused(tmp_path):
    rc, out = _run(tmp_path, "frd", "add", "fast", "--id", "PERF-001")
    assert rc == 2 and [p["code"] for p in out["problems"]] == ["FRD_NO_BOUNDS"]
    rc, out = _run(tmp_path, "frd", "add", "fast", "--id", "TIME-001", "--max", "10")
    assert rc == 2 and [p["code"] for p in out["problems"]] == ["FRD_NO_METRIC"]
    assert open_project(tmp_path).item("PERF-001") is None
    # a non-PERF item needs no bounds
    rc, out = _run(tmp_path, "frd", "add", "SWMR holds", "--id", "INV-001")
    assert rc == 0 and {a["code"] for a in out["advisories"]} >= {"FRD_NO_ACCEPTANCE", "REQ_UNOWNED"}
    assert "FRD_NO_MODEL_CHECK" not in {a["code"] for a in out["advisories"]}   # model-check prose is optional


def test_frd_add_writes_row_links_artifact_and_render(tmp_path):
    rc, _ = _run(tmp_path, "prd", "add", "60 fps", "--id", "KPI-FPS-1", "--acceptance", ">=55 fps -- throughput.py")
    assert rc == 0
    rc, out = _perf(tmp_path)
    assert rc == 0, out
    db = open_project(tmp_path)
    it = db.item("PERF-001")
    assert (it["metric"], it["bound_min"], it["bound_max"], it["unit"]) == ("fps", 55.0, None, "fps")
    assert it["priority"] == "must_have" and it["section"] == "Performance Requirements"
    assert {(lk["to_id"], lk["rel"]) for lk in db.links(from_id="PERF-001")} == {
        ("KPI-FPS-1", "derives_from"), ("block:fft64", "owned_by")}
    assert db.artifact("frd")["path"] == "db:frd" and db.artifact("prd")["path"] == "db:prd"
    md = (tmp_path / "arch" / "frd_spec.md").read_text()
    assert "**ID**: PERF-001" in md and "**Metric**: fps; min 55; max -; unit fps" in md
    assert [a["code"] for a in out["advisories"]] == ["FRD_UNVERIFIED"]
    # the PRD view is what extract_prd_items reads
    doc = json.loads((tmp_path / ".coresmith" / "prd_spec.json").read_text())
    items, probs = extract.extract_prd_items(doc)
    assert not probs and items[0]["id"] == "KPI-FPS-1" and items[0]["acceptance"] == ">=55 fps -- throughput.py"
    # ids cited in the text link too; duplicate add is refused, edit works
    rc, out = _run(tmp_path, "frd", "add", "x", "--id", "PERF-001", "--metric", "fps", "--min", "1")
    assert rc == 2 and out["problems"][0]["code"] == "FRD_EXISTS"
    rc, out = _run(tmp_path, "frd", "edit", "PERF-001", "--max", "90", "--text", "fps per FR-CPU-2")
    assert rc == 0 and out["item"]["bound_max"] == 90.0 and "FR-CPU-2" in out["derives_from"]
    rc, out = _run(tmp_path, "frd", "edit", "PERF-001", "--min", "none", "--max", "none")
    assert rc == 2 and out["problems"][0]["code"] == "FRD_NO_BOUNDS"


def test_rendered_frd_round_trips_through_register(tmp_path):
    _run(tmp_path, "prd", "add", "60 fps", "--id", "KPI-FPS-1")
    _run(tmp_path, "prd", "add", "Two harts", "--id", "FR-CPU-2", "--priority", "must_have")
    _perf(tmp_path)
    _run(tmp_path, "frd", "add", "latency", "--id", "TIME-001", "--metric", "latency", "--max", "12.5",
         "--unit", "cycles", "--priority", "should_have", "--acceptance", "sim <= 12.5", "--derives-from", "FR-CPU-2")
    _run(tmp_path, "frd", "add", "coherence per FR-CPU-2", "--id", "INV-001", "--acceptance", "no two M",
         "--model-check", "monitor")
    db = open_project(tmp_path)
    before = {i["id"]: i for i in db.items(artifact="frd")}
    md = (tmp_path / "arch" / "frd_spec.md").read_text()
    reqs = {r["id"]: r for r in fe.extract_requirements(md)}
    assert reqs["TIME-001"]["bound_max"] == 12.5 and reqs["TIME-001"]["unit"] == "cycles"
    assert "metric" not in reqs["INV-001"]  # no Metric line -> no bound keys, model_check untouched
    assert reqs["INV-001"]["model_check"] == "monitor"
    res = register(db, tmp_path, "frd", "arch/frd_spec.md")
    assert res["ok"], res
    after = {i["id"]: i for i in db.items(artifact="frd")}
    assert set(after) == set(before) == {"PERF-001", "TIME-001", "INV-001"}
    for iid in before:
        for k in ("text", "priority", "acceptance", "model_check", "metric", "bound_min", "bound_max", "unit"):
            assert after[iid][k] == before[iid][k], (iid, k)
    assert {lk["to_id"] for lk in db.links(from_id="TIME-001", rel="derives_from")} == {"FR-CPU-2"}
    # the stage machine: model_check given -> no FRD_NO_MODEL_CHECK for PERF-001
    codes = {b["code"]: b["ids"] for b in st._entry_requirements(db, tmp_path)}
    assert "PERF-001" not in codes.get("FRD_NO_MODEL_CHECK", [])
    assert "KPI_UNCOVERED" not in codes


def test_check_add_value_derives_status_from_bounds(tmp_path):
    _perf(tmp_path)
    rc, out = _run(tmp_path, "check", "add", "PERF-001", "model_eval", "--value", "52")
    assert rc == 0 and out["status"] == "fail" and out["bounds"]["min"] == 55.0
    rc, out = _run(tmp_path, "check", "add", "PERF-001", "model_eval", "--value", "58")
    assert rc == 0 and out["status"] == "pass"
    rc, out = _run(tmp_path, "check", "add", "PERF-001", "model_eval")
    assert rc == 2
    _run(tmp_path, "frd", "add", "inv", "--id", "INV-001")
    rc, out = _run(tmp_path, "check", "add", "INV-001", "model_eval", "--value", "3")
    assert rc == 2 and "no bounds" in out["error"]


def test_verifier_requires_path_and_entry_and_unverified_list(tmp_path):
    _perf(tmp_path)
    _run(tmp_path, "frd", "add", "should", "--id", "IFACE-001", "--priority", "should_have")
    rc, out = _run(tmp_path, "frd", "verifier", "PERF-001", "--kind", "cocotb", "--path", "tb/cocotb/test_fft64.py")
    assert rc == 2 and out["problems"][0]["code"] == "FRD_VERIFIER_INCOMPLETE"
    rc, out = _run(tmp_path, "frd", "list", "--unverified")
    assert [r["id"] for r in out["items"]] == ["PERF-001"]
    rc, out = _run(tmp_path, "frd", "verifier", "PERF-001", "--kind", "cocotb", "--path", "tb/cocotb/test_fft64.py",
                   "--entry", "test_fps")
    assert rc == 0 and out["verifier"]["block"] == "fft64"  # defaulted from the single owner
    rc, out = _run(tmp_path, "frd", "list", "--unverified")
    assert out["items"] == []
    rc, out = _run(tmp_path, "frd", "verifiers", "--block", "fft64")
    assert [v["entry"] for v in out["verifiers"]] == ["test_fps"]
    rc, out = _run(tmp_path, "frd", "show", "PERF-001")
    assert rc == 0 and out["verifiers"][0]["kind"] == "cocotb" and out["advisories"] == []
    vid = out["verifiers"][0]["id"]
    assert _run(tmp_path, "frd", "verifier-rm", str(vid))[0] == 0
    assert _run(tmp_path, "frd", "verifier-rm", str(vid))[0] == 2
    assert _run(tmp_path, "frd", "retire", "IFACE-001")[0] == 0
    assert "IFACE-001" not in (tmp_path / "arch" / "frd_spec.md").read_text()


def test_state_write_never_overwrites_a_file_sourced_frd(tmp_path):
    (tmp_path / "arch").mkdir()
    src = ("# FRD\n\n## Semantic Invariants\n\n1. **ID**: INV-001\n   - **Requirement**: hand written\n"
           "   - **Acceptance criteria**: x\n   - **Priority**: must_have\n")
    (tmp_path / "arch" / "frd_spec.md").write_text(src)
    db = open_project(tmp_path)
    assert register(db, tmp_path, "frd", "arch/frd_spec.md")["ok"]
    rc, out = _run(tmp_path, "state", "write")
    assert rc == 0 and out["written"] == [] and any(s["kind"] == "frd" and "file-sourced" in s["reason"]
                                                    for s in out["skipped"])
    assert (tmp_path / "arch" / "frd_spec.md").read_text() == src
    rc, out = _run(tmp_path, "frd", "render")
    assert rc == 2 and out["problems"][0]["code"] == "FRD_FILE_SOURCED"
    rc, out = _run(tmp_path, "state", "write", "--force")
    assert rc == 0 and str(tmp_path / "arch" / "frd_spec.md") in out["written"]
    assert render.RENDER_MARKER in (tmp_path / "arch" / "frd_spec.md").read_text()
    # the daemon `state` verb still dispatches to its own handler
    args = _parser().parse_args(["state", "--project-root", str(tmp_path)])
    assert args.verb is None


# -- block-done per-item stamping ---------------------------------------------
class _R:
    def __init__(self, passed=True):
        self.passed, self.verdict, self.infra_error, self.skipped, self.details, self.log_path = passed, "ok", False, False, {}, ""


_XML = """<testsuites><testsuite name="all">
<testcase classname="test_fft64" name="test_fps" time="1"/>
<testcase classname="test_fft64" name="test_sqnr" time="1"><failure message="x"/></testcase>
<testcase classname="test_fft64" name="test_skip" time="0"><skipped/></testcase>
</testsuite></testsuites>"""


@pytest.fixture
def gated_block(tmp_path, monkeypatch):
    import orchestrator.harness.sim_evidence as se
    import orchestrator.harness.verify as V
    import orchestrator.langgraph.contract_conformance as cc
    import orchestrator.langgraph.pipeline_graph as pg
    import orchestrator.langgraph.pipeline_helpers as ph
    monkeypatch.setattr(cc, "run_conformance_stage", lambda pr, n, rtl, **k: {"ran": True, "ok": True})
    def fake_verify(pr, spec, **k):
        (tmp_path / "sim_build" / "fft64" / "results.xml").write_text(_XML)
        return _R()
    monkeypatch.setattr(V, "verify_rtl", fake_verify)
    monkeypatch.setattr(ph, "synthesize_block", lambda spec, rtl, clk, att: {"success": True, "gate_count": 10})
    monkeypatch.setattr(pg, "_evaluate_ppa_gate", lambda pr, n, rtl, res, require_gate_flag=True: (True, [], {"wns_ns": 1.0}))
    monkeypatch.setattr(se, "capture", lambda path, inputs: {"passed": True, "inputs": inputs})
    db = open_project(tmp_path)
    db.import_block_diagram({"blocks": [{"name": "fft64", "tier": 1}], "connections": []})
    (tmp_path / "rtl").mkdir()
    (tmp_path / "rtl" / "fft64.v").write_text("module fft64(input clk); endmodule\n")
    (tmp_path / "tb" / "cocotb").mkdir(parents=True)
    (tmp_path / "tb" / "cocotb" / "test_fft64.py").write_text("# fixture\n")
    (tmp_path / "sim_build" / "fft64").mkdir(parents=True)
    (tmp_path / "sim_build" / "fft64" / "results.xml").write_text(_XML)
    db.ensure_db_artifact("frd")
    for iid in ("PERF-001", "PERF-002", "PERF-003", "INV-001", "PERF-004"):
        db.upsert_item("frd", {"id": iid, "text": iid, "priority": "must_have"})
        db.link_items(iid, "block:fft64", "owned_by")
    db.add_verifier("PERF-001", "cocotb", path="tb/cocotb/test_fft64.py", entry="test_fps")  # bound by tb path
    db.add_verifier("PERF-002", "cocotb", path="tb/other.py", entry="test_missing", block="fft64")
    db.add_verifier("PERF-003", "cocotb", path="tb/other.py", entry="test_fft64.test_sqnr", block="fft64")
    db.add_verifier("PERF-004", "manual")  # not run by block-done
    return db, tmp_path


def test_block_done_stamps_per_item_checks(gated_block, monkeypatch):
    monkeypatch.delenv("CORESMITH_BLOCK_DV_BLANKET", raising=False)
    db, root = gated_block
    res = bt.block_done(db, root, "fft64")
    assert not res["ok"] and db.result("fft64", "best") is None
    latest = {c["item_id"]: c for c in db.checks(kind="block_dv", latest=True)}
    assert latest["PERF-001"]["status"] == "pass" and "test_fps: pass" in latest["PERF-001"]["evidence"]
    assert latest["PERF-002"]["status"] == "tool_error" and "not found in results.xml" in latest["PERF-002"]["evidence"]
    assert latest["PERF-003"]["status"] == "fail"
    assert "INV-001" not in latest and "PERF-004" not in latest
    assert res["unverified_items"] == ["INV-001"] and res["deferred_items"] == ["PERF-004"]
    assert res["verified_items"] == ["PERF-001"] and res["failed_items"] == ["PERF-002", "PERF-003"]
    assert bt._cocotb_results(root / "sim_build" / "fft64" / "results.xml") == {
        "test_fps": "pass", "test_fft64.test_fps": "pass", "test_sqnr": "fail", "test_fft64.test_sqnr": "fail",
        "test_skip": "skipped", "test_fft64.test_skip": "skipped"}
    assert bt._cocotb_results(root / "nope.xml") == {}


def test_block_done_blanket_env_cannot_fabricate_verifier_passes(gated_block, monkeypatch):
    monkeypatch.setenv("CORESMITH_BLOCK_DV_BLANKET", "1")
    db, root = gated_block
    res = bt.block_done(db, root, "fft64")
    assert not res["ok"] and db.result("fft64", "best") is None
    latest = {c["item_id"]: c["status"] for c in db.checks(kind="block_dv", latest=True)}
    assert latest == {"PERF-001": "pass", "PERF-002": "tool_error", "PERF-003": "fail"}


# -- bin/coresmith subprocess ---------------------------------------------------
def _cli(root, *a):
    env = {"CORESMITH_PROJECT_ROOT": str(root), "PATH": "/usr/bin:/bin", "PYTHONPATH": str(_REPO)}
    return subprocess.run([sys.executable, str(_REPO / "bin" / "coresmith"), *a], capture_output=True, text=True,
                          env=env, timeout=120)


def test_bin_coresmith_frd_verbs(tmp_path):
    p = _cli(tmp_path, "frd", "add", "fast", "--id", "PERF-001", "--project-root", str(tmp_path))
    assert p.returncode == 2 and "FRD_NO_BOUNDS" in p.stdout
    p = _cli(tmp_path, "frd", "add", "fast", "--id", "PERF-001", "--metric", "fps", "--min", "55",
             "--project-root", str(tmp_path))
    assert p.returncode == 0, p.stderr
    assert "FRD_UNVERIFIED" in p.stdout and "wrote arch/frd_spec.md" in p.stdout
    assert "FRD_NO_MODEL_CHECK" not in p.stdout   # model-check prose is optional
    p = _cli(tmp_path, "check", "add", "PERF-001", "model_eval", "--value", "52", "--project-root", str(tmp_path))
    assert p.returncode == 0 and "fail" in p.stdout and "min 55" in p.stdout
    p = _cli(tmp_path, "state", "write", "--project-root", str(tmp_path), "--json")
    assert p.returncode == 0, p.stderr
    assert str(tmp_path / "arch" / "frd_spec.md") in json.loads(p.stdout)["written"]
    rows = open_project(tmp_path).actions()
    assert [r["argv"][:2] for r in rows] == [["frd", "add"], ["frd", "add"], ["check", "add"], ["state", "write"]]
    assert rows[0]["rc"] == 2


def test_module_imports_without_langgraph():
    code = ("import sys; import orchestrator.harness.cli_frd, orchestrator.harness.tools.render; "
            "print(any(m.startswith('orchestrator.langgraph') for m in sys.modules))")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(_REPO), timeout=60)
    assert p.stdout.strip() == "False", p.stderr
    assert cli_frd.BOUNDED_KINDS == ("PERF", "TIME")


def test_frd_edit_owner_replaces_the_previous_owner(tmp_path):
    rc, _ = _perf(tmp_path)  # owned by fft64
    assert rc == 0
    db = open_project(tmp_path)
    db.link_items("PERF-001", "KPI-FPS-1", "derives_from")
    rc, out = _run(tmp_path, "frd", "edit", "PERF-001", "--owner", "ifft64")
    assert rc == 0, out
    assert [lk["to_id"] for lk in db.links(from_id="PERF-001", rel="owned_by")] == ["block:ifft64"]
    assert db.links(from_id="PERF-001", rel="derives_from")  # other links untouched
    # the API: exact target, prefix target, rel filter
    db.link_items("INV-009", "block:a", "owned_by")
    db.link_items("INV-009", "block:b", "owned_by")
    db.link_items("INV-009", "PERF-001", "derives_from")
    assert db.unlink_items("INV-009", "block:a") == 1
    assert db.unlink_items("INV-009", "block:*", "owned_by") == 1
    assert [lk["to_id"] for lk in db.links(from_id="INV-009")] == ["PERF-001"]


def test_actions_script_is_a_replayable_shell_script(tmp_path, capsys):
    import shlex
    db = open_project(tmp_path)
    db.record_action(["frd", "add", "Sustain 60 fps", "--id", "PERF-001", "--min", "55"], 0)
    db.record_action(["frd", "add", "x", "--id", "PERF-002"], 2)
    db.record_action(["actions", "--json"], 0)
    rc, out = _run(tmp_path, "actions", "--script")
    assert rc == 0
    lines = out.strip().splitlines()
    assert lines[0] == "#!/bin/sh"
    body = [ln for ln in lines if not ln.startswith("#")]
    assert body[0] == "coresmith frd add 'Sustain 60 fps' --id PERF-001 --min 55"
    assert shlex.split(body[0])[1:] == ["frd", "add", "Sustain 60 fps", "--id", "PERF-001", "--min", "55"]
    assert body[1].endswith("# rc=2") and len(body) == 2  # the `actions` row is skipped


# -- fix 1: measurements decide bounded items in block-done ---------------------
_RUN = Path("/home/ubuntu/coresmith-runs/mcufft-cli-tactics-20260929")


def _copy_run_db(dst: Path) -> None:
    """A consistent copy of the evaluation run's project DB (read-only source)."""
    import sqlite3
    (dst / ".coresmith").mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(f"file:{_RUN / '.coresmith' / 'project.sqlite'}?mode=ro", uri=True)
    out = sqlite3.connect(str(dst / ".coresmith" / "project.sqlite"))
    try:
        src.backup(out)
    finally:
        src.close()
        out.close()


def test_measure_record_writes_env_file_or_cwd(tmp_path, monkeypatch):
    from orchestrator.harness import measure
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CORESMITH_MEASUREMENTS", raising=False)
    measure.record("PERF-003", 107759, unit="fps", test="test_throughput_b2b")
    rows = measure.read(tmp_path / "measurements.jsonl")
    assert [(r["item"], r["value"], r["unit"], r["test"]) for r in rows] == [
        ("PERF-003", 107759.0, "fps", "test_throughput_b2b")]
    target = tmp_path / "sim_build" / "b" / "measurements.jsonl"
    monkeypatch.setenv("CORESMITH_MEASUREMENTS", str(target))
    measure.record("PERF-001", 206)
    (target.parent / "junk").write_text("")
    with open(target, "a") as fh:
        fh.write("not json\n{\"item\": \"PERF-9\"}\n")
    assert [(r["item"], r["value"]) for r in measure.read(target)] == [("PERF-001", 206.0)]
    assert measure.read(tmp_path / "nope.jsonl") == []
    # stdlib only: importable without cocotb
    code = "import sys, orchestrator.harness.measure; print('cocotb' in sys.modules)"
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(_REPO), timeout=60)
    assert p.stdout.strip() == "False", p.stderr


@pytest.mark.skipif(not (_RUN / "sim_build" / "fft64" / "results.xml").exists(), reason="evaluation run not present")
def test_block_dv_measurements_decide_on_the_evaluation_run(tmp_path):
    """The run's fft64 results.xml (every test passed) and verifiers: a passing
    test no longer passes a bounded item -- its measurement and bounds do."""
    import shutil
    _copy_run_db(tmp_path)
    (tmp_path / "sim_build" / "fft64").mkdir(parents=True)
    shutil.copy2(_RUN / "sim_build" / "fft64" / "results.xml", tmp_path / "sim_build" / "fft64" / "results.xml")
    m = tmp_path / "sim_build" / "fft64" / "measurements.jsonl"
    m.write_text("\n".join(json.dumps(r) for r in (
        {"item": "PERF-003", "value": 84175, "unit": "fps", "test": "test_throughput_measure"},  # another test
        {"item": "PERF-003", "value": 107759, "unit": "fps", "test": "test_throughput_b2b"},
        {"item": "PERF-002", "value": 59.2, "unit": "dB", "test": "test_random_1000"},
    )) + "\n")
    db = open_project(tmp_path)
    n0 = len(db.checks())
    res = bt.stamp_item_checks(db, tmp_path, "fft64", "tb/cocotb/test_fft64.py", rtl_sha="abc", actor="t")
    new = {c["item_id"]: c for c in db.checks()[n0:]}
    # measured and within bounds: pass with the value; the verifier's own test's measurement wins
    assert new["PERF-003"]["status"] == "pass" and new["PERF-003"]["value"] == 107759.0
    assert "test_throughput_b2b" in new["PERF-003"]["evidence"] and "fps" in new["PERF-003"]["evidence"]
    # measured and out of bounds (sqnr min 60): the passing cocotb test does not save it
    assert new["PERF-002"]["status"] == "fail" and new["PERF-002"]["value"] == 59.2
    # bounded, test passed, nothing recorded: tool_error, listed as unmeasured
    assert new["PERF-001"]["status"] == "tool_error" and new["PERF-001"]["value"] is None
    assert "no measurement recorded for a bounded item" in new["PERF-001"]["evidence"]
    assert res["unmeasured_items"] == ["PERF-001"]
    # unbounded items keep the verdict path
    assert new["INV-001"]["status"] == "pass" and new["INV-001"]["value"] is None
    assert "PERF-002" in res["failed_items"] and "PERF-003" in res["verified_items"]
    # item status by precedence: PERF-002's block_dv fail outranks its model_eval passes; PERF-001 keeps the
    # run's integration_dv pass (rank 3), which a block_dv tool_error (rank 2) cannot undo
    assert db.item("PERF-002")["status"] == "failed" and db.item("PERF-001")["status"] == "verified"


def test_measurement_must_come_from_verifier_test_with_matching_unit(tmp_path):
    db = open_project(tmp_path)
    db.ensure_db_artifact("frd")
    db.upsert_item("frd", {"id": "PERF-901", "text": "rate", "priority": "must_have",
                           "metric": "fps", "bound_min": 10, "unit": "fps"})
    db.add_verifier("PERF-901", "cocotb", entry="test_required")
    xml = tmp_path / "results.xml"
    xml.write_text('<testsuite><testcase name="test_required"/></testsuite>')
    measurements = tmp_path / "measurements.jsonl"
    measurements.write_text(json.dumps({"item": "PERF-901", "value": 99,
                                         "unit": "fps", "test": "test_other"}) + "\n")
    result = bt.stamp_checks(db, tmp_path, kind="block_dv", items=["PERF-901"],
                             select=lambda _: True, results_xml=xml, measurements=measurements)
    assert result["unmeasured_items"] == ["PERF-901"]
    assert db.latest_check("PERF-901", "block_dv")["status"] == "tool_error"

    measurements.write_text(json.dumps({"item": "PERF-901", "value": 99,
                                         "unit": "MHz", "test": "test_required"}) + "\n")
    result = bt.stamp_checks(db, tmp_path, kind="block_dv", items=["PERF-901"],
                             select=lambda _: True, results_xml=xml, measurements=measurements)
    check = db.latest_check("PERF-901", "block_dv")
    assert check["status"] == "tool_error" and "does not match" in check["evidence"]


def test_multiple_verifiers_produce_one_fail_closed_check(tmp_path):
    db = open_project(tmp_path)
    db.ensure_db_artifact("frd")
    db.upsert_item("frd", {"id": "INV-901", "text": "both properties", "priority": "must_have"})
    db.add_verifier("INV-901", "cocotb", entry="test_a")
    db.add_verifier("INV-901", "cocotb", entry="test_b")
    xml = tmp_path / "results.xml"
    xml.write_text('<testsuite><testcase name="test_a"/><testcase name="test_b"><failure/></testcase></testsuite>')
    result = bt.stamp_checks(db, tmp_path, kind="block_dv", items=["INV-901"],
                             select=lambda _: True, results_xml=xml,
                             measurements=tmp_path / "missing.jsonl")
    rows = db.checks("INV-901", kind="block_dv")
    assert len(rows) == 1 and rows[0]["status"] == "fail"
    assert result["failed_items"] == ["INV-901"]


def test_multiple_bounded_verifiers_keep_conservative_value(tmp_path):
    db = open_project(tmp_path)
    db.ensure_db_artifact("frd")
    db.upsert_item("frd", {"id": "PERF-902", "text": "latency", "priority": "must_have",
                           "metric": "latency", "bound_max": 10, "unit": "cycles"})
    db.add_verifier("PERF-902", "cocotb", entry="test_a")
    db.add_verifier("PERF-902", "cocotb", entry="test_b")
    xml = tmp_path / "results.xml"
    xml.write_text('<testsuite><testcase name="test_a"/><testcase name="test_b"/></testsuite>')
    measurements = tmp_path / "measurements.jsonl"
    measurements.write_text("\n".join(json.dumps(r) for r in [
        {"item": "PERF-902", "value": 6, "unit": "cycles", "test": "test_a"},
        {"item": "PERF-902", "value": 9, "unit": "cycles", "test": "test_b"},
    ]) + "\n")
    bt.stamp_checks(db, tmp_path, kind="block_dv", items=["PERF-902"],
                    select=lambda _: True, results_xml=xml, measurements=measurements)
    check = db.latest_check("PERF-902", "block_dv")
    assert check["status"] == "pass" and check["value"] == 9
    assert "measured 6" in check["evidence"] and "measured 9" in check["evidence"]

    measurements.write_text("\n".join(json.dumps(r) for r in [
        {"item": "PERF-902", "value": 6, "unit": "cycles", "test": "test_a"},
        {"item": "PERF-902", "value": 12, "unit": "cycles", "test": "test_b"},
    ]) + "\n")
    bt.stamp_checks(db, tmp_path, kind="block_dv", items=["PERF-902"],
                    select=lambda _: True, results_xml=xml, measurements=measurements)
    check = db.latest_check("PERF-902", "block_dv")
    assert check["status"] == "fail" and check["value"] == 12


def test_chip_metric_owner_uses_registered_nonstandard_top(tmp_path):
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs" / "task.yaml").write_text("top: accelerator_wrapper\n")
    db = open_project(tmp_path)
    db.ensure_db_artifact("frd")
    for iid, owner in (("PERF-910", "accelerator_wrapper"), ("PERF-911", "mcu")):
        db.upsert_item("frd", {"id": iid, "text": "cells", "priority": "must_have",
                               "metric": "cell_count", "bound_max": 100})
        db.link_items(iid, f"block:{owner}", "owned_by")
    db.upsert_item("frd", {"id": "PERF-912", "text": "ownerless cells",
                           "priority": "must_have", "metric": "cell_count", "bound_max": 100})
    assert [i["id"] for i in bt.chip_top_items(db, ["cell_count"])] == ["PERF-910"]


def test_block_done_sets_measurement_env_and_drops_stale_file(gated_block, monkeypatch):
    import orchestrator.harness.verify as V
    from orchestrator.harness import measure
    monkeypatch.delenv("CORESMITH_BLOCK_DV_BLANKET", raising=False)
    monkeypatch.delenv("CORESMITH_MEASUREMENTS", raising=False)
    db, root = gated_block
    db.edit_item("PERF-001", metric="fps", bound_min=55, unit="fps")
    mfile = root / "sim_build" / "fft64" / "measurements.jsonl"
    mfile.write_text(json.dumps({"item": "PERF-001", "value": 99, "test": "test_fps"}) + "\n")  # stale: must not count
    seen = {}

    def fake_verify(pr, spec, **k):
        seen["env"] = measure.measurements_path()
        seen["stale_gone"] = not mfile.exists()
        (root / "sim_build" / "fft64" / "results.xml").write_text(_XML)
        measure.record("PERF-001", 41, unit="fps", test="test_fps")   # what the TB would do
        return _R()
    monkeypatch.setattr(V, "verify_rtl", fake_verify)
    res = bt.block_done(db, root, "fft64")
    assert not res["ok"] and seen == {"env": mfile, "stale_gone": True}
    assert db.result("fft64", "best") is None
    assert "CORESMITH_MEASUREMENTS" not in __import__("os").environ   # restored
    c = [c for c in res["item_checks"] if c["item"] == "PERF-001"][0]
    assert (c["status"], c["value"]) == ("fail", 41.0)   # 41 fps < min 55 although test_fps passed
    assert db.latest_check("PERF-001", "block_dv")["value"] == 41.0


# -- fix 2: bounds win in check add ----------------------------------------------
def test_check_add_explicit_status_contradicting_bounds_is_refused(tmp_path):
    assert _perf(tmp_path)[0] == 0
    db = open_project(tmp_path)
    rc, out = _run(tmp_path, "check", "add", "PERF-001", "model_eval", "pass", "--value", "35")
    assert rc == 2 and [p["code"] for p in out["problems"]] == ["CHECK_STATUS_CONFLICT"]
    assert db.checks("PERF-001") == []   # nothing written
    rc, out = _run(tmp_path, "check", "add", "PERF-001", "model_eval", "fail", "--value", "35")
    assert rc == 0 and out["status"] == "fail"   # agrees with the bounds
    rc, out = _run(tmp_path, "check", "add", "PERF-001", "model_eval", "--value", "58")
    assert rc == 0 and out["status"] == "pass"


# -- fix 3: status precedence + failed-must-have blocker --------------------------
def test_item_status_precedence_by_check_kind(tmp_path):
    from orchestrator.state_store.ontology import CHECK_KIND_RANK, check_rank
    assert CHECK_KIND_RANK == {"model_eval": 1, "block_dv": 2, "integration_dv": 3, "validation_dv": 4,
                               "acceptance": 4, "signoff": 5}
    assert check_rank("sta") == 2 and check_rank("chip_perf") == 2
    db = open_project(tmp_path)
    db.ensure_db_artifact("frd")
    db.upsert_item("frd", {"id": "INV-001", "text": "x", "priority": "must_have"})
    db.add_check("INV-001", "model_eval", "pass")
    assert db.item("INV-001")["status"] == "verified"
    db.add_check("INV-001", "block_dv", "fail")
    assert db.item("INV-001")["status"] == "failed"
    db.add_check("INV-001", "model_eval", "pass")          # a model pass never overrides an RTL fail
    assert db.item("INV-001")["status"] == "failed"
    db.add_check("INV-001", "block_dv", "pass")            # an RTL pass after an RTL fail is a pass
    assert db.item("INV-001")["status"] == "verified"
    db.add_check("INV-001", "sta", "fail")                 # unknown kinds rank with block_dv: latest decides
    assert db.item("INV-001")["status"] == "failed"
    db.add_check("INV-001", "integration_dv", "pass")
    assert db.item("INV-001")["status"] == "verified"
    db.add_check("INV-001", "block_dv", "fail")            # a lower rank cannot undo it
    assert db.item("INV-001")["status"] == "verified"
    db.add_check("INV-001", "signoff", "tool_error")       # not judged at the top rank: open
    assert db.item("INV-001")["status"] == "open"
    db.set_item_status("INV-001", "waived")
    db.add_check("INV-001", "signoff", "fail")
    assert db.item("INV-001")["status"] == "waived"


def test_frd_show_and_list_print_per_kind_checks(tmp_path):
    assert _perf(tmp_path)[0] == 0
    db = open_project(tmp_path)
    db.add_check("PERF-001", "model_eval", None, value=60)
    db.add_check("PERF-001", "block_dv", None, value=41)
    rc, out = _run(tmp_path, "frd", "show", "PERF-001")
    assert rc == 0 and [c["kind"] for c in out["checks"]] == ["block_dv", "model_eval"]
    assert out["status_from"] == ["block_dv"] and out["item"]["status"] == "failed"
    args = _parser().parse_args(["frd", "show", "PERF-001", "--project-root", str(tmp_path)])
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit):
        args.func(args)
    assert "check block_dv (rank 2): fail value=41 <- decides the status" in buf.getvalue()
    rc, out = _run(tmp_path, "frd", "list")
    assert {(c["kind"], c["status"]) for c in out["items"][0]["checks"]} == {("block_dv", "fail"), ("model_eval", "pass")}
    args = _parser().parse_args(["frd", "list", "--project-root", str(tmp_path)])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit):
        args.func(args)
    assert "[block_dv=fail(41) model_eval=pass(60)]" in buf.getvalue()


def test_failed_must_have_blocks_every_stage(tmp_path, monkeypatch):
    monkeypatch.delenv("CORESMITH_STAGE_FAIL_BLOCKS", raising=False)
    assert _perf(tmp_path)[0] == 0
    db = open_project(tmp_path)
    db.add_check("PERF-001", "block_dv", None, value=41)
    for stage in st.STAGES:
        codes = [b for b in st.entry(db, tmp_path, stage) if b["code"] == "MUST_HAVE_FAILED"]
        assert codes and codes[0]["ids"] == ["PERF-001"], stage
    db.add_check("PERF-001", "block_dv", None, value=58)
    assert not any(b["code"] == "MUST_HAVE_FAILED" for b in st.entry(db, tmp_path, "acceptance"))


def test_failed_must_have_blocker_env_off(tmp_path, monkeypatch):
    monkeypatch.setenv("CORESMITH_STAGE_FAIL_BLOCKS", "0")
    assert _perf(tmp_path)[0] == 0
    db = open_project(tmp_path)
    db.add_check("PERF-001", "block_dv", None, value=41)
    assert db.item("PERF-001")["status"] == "failed"
    assert not any(b["code"] == "MUST_HAVE_FAILED" for s in st.STAGES for b in st.entry(db, tmp_path, s))


# -- fix 5: prd add refuses an existing live id -----------------------------------
def test_prd_add_refuses_existing_id(tmp_path):
    rc, _ = _run(tmp_path, "prd", "add", "MCU runs firmware", "--id", "FR-MCU-1", "--priority", "must_have")
    assert rc == 0
    rc, out = _run(tmp_path, "prd", "add", "duplicate test", "--id", "FR-MCU-1")
    assert rc == 2 and out["problems"][0]["code"] == "PRD_EXISTS" and "prd edit FR-MCU-1" in out["problems"][0]["text"]
    it = open_project(tmp_path).item("FR-MCU-1")
    assert it["text"] == "MCU runs firmware" and it["priority"] == "must_have"   # not clobbered
    assert _run(tmp_path, "prd", "retire", "FR-MCU-1")[0] == 0
    assert _run(tmp_path, "prd", "add", "again", "--id", "FR-MCU-1")[0] == 0     # a retired id may be re-added


# -- fix 8: verifier rebind ----------------------------------------------------------
def test_frd_verifier_rebind_needs_replace(tmp_path):
    assert _perf(tmp_path)[0] == 0   # owned by fft64
    base = ["frd", "verifier", "PERF-001", "--kind", "cocotb", "--path", "tb/cocotb/test_fft64.py",
            "--entry", "test_latency"]
    rc, out = _run(tmp_path, *base)
    assert rc == 0 and out["verifier"]["block"] == "fft64"
    rc, out = _run(tmp_path, *base, "--block", "uart")
    assert rc == 2 and out["problems"][0]["code"] == "VERIFIER_REBOUND"
    db = open_project(tmp_path)
    assert [v["block"] for v in db.verifiers(item_id="PERF-001")] == ["fft64"]
    assert _run(tmp_path, *base, "--block", "fft64")[0] == 0     # same block: an idempotent re-bind
    rc, out = _run(tmp_path, *base, "--block", "uart", "--replace")
    assert rc == 0 and out["verifier"]["block"] == "uart"
    assert [a["code"] for a in out["advisories"]] == ["VERIFIER_BLOCK_NOT_OWNER"]
    assert len(db.verifiers(item_id="PERF-001")) == 1


# -- fix 9: register ers must not re-home FRD items ------------------------------------
@pytest.mark.skipif(not (_RUN / "arch" / "ers_spec.json").exists(), reason="evaluation run not present")
def test_register_ers_keeps_frd_items_on_the_evaluation_run(tmp_path):
    import shutil
    (tmp_path / "arch").mkdir()
    for f in ("frd_spec.md", "ers_spec.json"):
        shutil.copy2(_RUN / "arch" / f, tmp_path / "arch" / f)
    db = open_project(tmp_path)
    assert register(db, tmp_path, "frd", "arch/frd_spec.md")["ok"]
    before = {i: db.item(i) for i in ("INV-001", "INV-002", "INV-003", "INV-004")}
    assert all(b["artifact"] == "frd" and b["model_check"] for b in before.values())
    res = register(db, tmp_path, "ers", "arch/ers_spec.json")
    assert res["ok"], res["problems"]
    warn = [p for p in res["problems"] if p["code"] == "ITEM_OWNED_ELSEWHERE"]
    assert warn and warn[0]["severity"] == "warning"
    assert {"INV-001", "INV-002", "INV-003", "INV-004"} <= set(warn[0]["where"].split(", "))
    for iid, b in before.items():
        after = db.item(iid)
        for k in ("artifact", "text", "acceptance", "model_check", "section", "priority"):
            assert after[k] == b[k], (iid, k)
    assert {i["id"] for i in db.items(artifact="frd")} >= set(before)
    # the ERS affected_blocks links are still added
    owners = {lk["to_id"] for lk in db.links(from_id="INV-002", rel="owned_by")}
    assert {"block:fft64", "block:uart"} <= owners
    assert db.links(from_id="INV-003", to_id="block:mcu", rel="owned_by")
    # fix 7: the ERS view validation_dv reads is written
    view = tmp_path / ".coresmith" / "ers_spec.json"
    assert json.loads(view.read_text()) == json.loads((tmp_path / "arch" / "ers_spec.json").read_text())


def test_register_ers_view_and_owned_elsewhere_synthetic(tmp_path):
    db = open_project(tmp_path)
    db.ensure_db_artifact("frd")
    db.upsert_item("frd", {"id": "INV-001", "text": "frd text", "model_check": "mc", "priority": "must_have"})
    ers = {"ers": {"system_invariants": [{"id": "INV-001", "description": "ers one-liner", "affected_blocks": ["a"]},
                                         {"id": "INV-009", "description": "new", "affected_blocks": ["b"]}]}}
    (tmp_path / ".coresmith").mkdir(exist_ok=True)
    (tmp_path / ".coresmith" / "ers_spec.json").write_text(json.dumps(ers))
    res = register(db, tmp_path, "ers", ".coresmith/ers_spec.json")   # registered from the view path itself
    assert res["ok"] and res["items"]["owned_elsewhere"] == [{"id": "INV-001", "artifact": "frd"}]
    assert db.item("INV-001")["text"] == "frd text" and db.item("INV-009")["artifact"] == "ers"
    assert db.links(from_id="INV-001", to_id="block:a", rel="owned_by")
    assert not any(p["code"] == "ERS_VIEW" for p in res["problems"])   # not rewritten onto itself
