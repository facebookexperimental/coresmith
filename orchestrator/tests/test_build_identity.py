# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""A build's evidence is bound to the architecture inputs it started from
(spec, contracts, target configuration, model + headers, harness, owned
items, worker, tooling) and to the implementation it produced (the complete
final target revision: sources, include-dir headers, discovered assets, the
testbench). Changing any of them makes the published pass stale while the
build row and its results stay as history. A pass published without a build
(block-done, a manual claim) is never a deliverable; the lineage report
tells the two apart. The evidence is judged by its LATEST outcome, and the
storage boundary is immutable and atomic."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from orchestrator.state_store import builds as B
from orchestrator.state_store import stages as st
from orchestrator.state_store.store import Scoreboard
from orchestrator.tests.build_fixtures import complete_build, ready_project, write_env

_complete_build = complete_build   # imported by the stage-gate tests


def _codes(blockers):
    return {b["code"]: b["ids"] for b in blockers}


def test_completed_build_is_current_until_an_input_changes(tmp_path, monkeypatch):
    db = ready_project(tmp_path, monkeypatch, stage="blocks", with_rtl=True)
    bid = _complete_build(db, tmp_path)
    cur = B.current_build_status(db, tmp_path, "tiny")
    assert cur["ok"] and cur["build"]["id"] == bid and cur["stale"] == []
    (tmp_path / "arch" / "uarch_specs" / "tiny.md").write_text("# revised spec\n")
    cur = B.current_build_status(db, tmp_path, "tiny")
    assert not cur["ok"] and any(r.startswith("spec:") for r in cur["reasons"])
    # history survives: the row and the published pass are still there, just not current
    assert B.get_build(db, bid)["status"] == "completed" and db.result("tiny", "best")["build_id"] == bid
    assert "tiny" in " ".join(_codes(st.entry(db, tmp_path, "blocks"))["BLOCK_NOT_BUILT"])


def test_each_identity_axis_invalidates(tmp_path, monkeypatch):
    db = ready_project(tmp_path, monkeypatch, stage="blocks", with_rtl=True)
    bid = _complete_build(db, tmp_path)
    stored = B.get_build(db, bid)["inputs"]

    def axis_after(change):
        change()
        live = B.module_inputs(db, tmp_path, "tiny", target_clock_mhz=50.0)
        return [r.split(":")[0] for r in B.stale_reasons(stored, live)]
    assert axis_after(lambda: None) == []
    assert "model" in axis_after(lambda: (tmp_path / "model" / "tiny_model.cpp").write_text("// v2\n"))
    (tmp_path / "model" / "tiny_model.cpp").write_text("// reference model of tiny\n#include \"tiny_model.h\"\nint tiny_model_marker = 1;\n")
    # a header the model includes is part of the model identity
    assert "model" in axis_after(lambda: (tmp_path / "model" / "tiny_model.h").write_text("// header v2\n"))
    (tmp_path / "model" / "tiny_model.h").write_text("// header of tiny\n")
    assert "harness" in axis_after(lambda: (tmp_path / "model" / "frd_eval" / "frd_eval.cpp").write_text("// h2\n"))
    (tmp_path / "model" / "frd_eval" / "frd_eval.cpp").write_text("// harness\n")
    assert "contracts" in axis_after(lambda: db.import_contracts({"contracts": [dict(db.contracts()["contracts"][0], data_width_bits=32)]}))
    assert "items" in axis_after(lambda: db.edit_item("PERF-001", bound_max=99.0))
    assert "worker" in axis_after(lambda: write_env(tmp_path, {"CORESMITH_LLM_PROVIDER": "codex", "CORESMITH_CODEX_MODEL": "x"}))
    # the clock and the synthesis mode are tooling context
    live = B.module_inputs(db, tmp_path, "tiny", target_clock_mhz=100.0)
    assert any(r.startswith("tooling") for r in B.stale_reasons(stored, live))
    monkeypatch.setenv("CORESMITH_SYNTH_GENERIC", "1")
    assert "tooling" in axis_after(lambda: None)


def test_target_bytes_are_the_implementation_not_an_input(tmp_path, monkeypatch):
    """The target is an input by CONFIGURATION (top, sources, includes,
    defines, parameters, assets): the build writes the bytes, so a seeded or
    repaired implementation is never stale by its own edit; after completion
    the produced identity judges the bytes."""
    db = ready_project(tmp_path, monkeypatch, stage="blocks", with_rtl=False)
    fresh = B.module_inputs(db, tmp_path, "tiny", target_clock_mhz=50.0)
    assert fresh["target"]["sources_present"] is False and fresh["target"]["revision"] is None
    (tmp_path / "rtl" / "tiny.v").write_text("module tiny(); endmodule\n")
    assert B.stale_reasons(fresh, B.module_inputs(db, tmp_path, "tiny", target_clock_mhz=50.0)) == []
    seeded = B.module_inputs(db, tmp_path, "tiny", seed_path=str(tmp_path / "rtl" / "tiny.v"), target_clock_mhz=50.0)
    assert seeded["seed"]["sha256"] == B.file_sha256(tmp_path / "rtl" / "tiny.v")
    (tmp_path / "rtl" / "tiny.v").write_text("module tiny(input clk); endmodule\n")   # the graph repairs the seed
    assert B.stale_reasons(seeded, B.module_inputs(db, tmp_path, "tiny", seed_path=str(tmp_path / "rtl" / "tiny.v"),
                                                   target_clock_mhz=50.0)) == []
    # a configuration change (another source list) IS an input change
    from orchestrator.harness import targets as T
    (tmp_path / "rtl" / "tiny_pkg.v").write_text("package tiny_pkg; endpackage\n")
    T.bind(tmp_path, "tiny", {"top": "tiny", "sources": ["rtl/tiny_pkg.v", "rtl/tiny.v"], "cwd": "."})
    assert ["target"] == [r.split(":")[0] for r in B.stale_reasons(seeded, B.module_inputs(db, tmp_path, "tiny", target_clock_mhz=50.0))]


def test_produced_identity_covers_headers_and_assets_and_refuses_missing_files(tmp_path, monkeypatch):
    """The produced identity is the COMPLETE final target revision: an
    include-dir header or a discovered $readmem asset edited after the build
    makes the pass stale without any rebind; a produced file that does not
    exist is ``missing``, never a valid empty hash."""
    from orchestrator.harness import targets as T
    db = ready_project(tmp_path, monkeypatch, stage="blocks", with_rtl=True)
    inc = tmp_path / "rtl" / "inc"
    inc.mkdir()
    (inc / "defs.vh").write_text("`define W 8\n")
    (tmp_path / "rtl" / "init.hex").write_text("00\n01\n")
    (tmp_path / "rtl" / "tiny.v").write_text(
        '`include "defs.vh"\nmodule tiny(input clk, input rst_n, output reg [`W-1:0] m_out);\n'
        '  reg [7:0] mem [0:1];\n  initial $readmemh("rtl/init.hex", mem);\n'
        "  always @(posedge clk) if (!rst_n) m_out <= 0; else m_out <= mem[0];\nendmodule\n")
    T.bind(tmp_path, "tiny", {"top": "tiny", "sources": ["rtl/tiny.v"], "include_dirs": ["rtl/inc"], "cwd": "."})
    bid = _complete_build(db, tmp_path)
    produced = B.get_build(db, bid)["result"]["produced"]
    assert produced["revision"] and str(inc / "defs.vh") in produced["files_sha256"]
    assert B.current_build_status(db, tmp_path, "tiny")["ok"]
    (inc / "defs.vh").write_text("`define W 16\n")
    cur = B.current_build_status(db, tmp_path, "tiny")
    assert not cur["ok"] and any("defs.vh" in p for p in cur["implementation_changed"])
    (inc / "defs.vh").write_text("`define W 8\n")
    assert B.current_build_status(db, tmp_path, "tiny")["ok"]
    # a header ADDED to the include dir changes the revision without changing any recorded file
    (inc / "extra.vh").write_text("// new\n")
    cur = B.current_build_status(db, tmp_path, "tiny")
    assert not cur["ok"] and any("dependencies changed" in p for p in cur["implementation_changed"])
    (inc / "extra.vh").unlink()
    # a missing produced file is reported, not hashed as None
    out = B.produced_outputs(tmp_path, "tiny", B.target_identity(tmp_path, "tiny"), tb_path=str(tmp_path / "tb" / "nope.py"))
    assert str(tmp_path / "tb" / "nope.py") in out["missing"]
    (tmp_path / "rtl" / "tiny.v").unlink()
    out = B.produced_outputs(tmp_path, "tiny", B.target_identity(tmp_path, "tiny"))
    assert out["missing"] and out["revision"] is None
    # ... and a recorded pass whose source vanished is not current
    cur = B.current_build_status(db, tmp_path, "tiny")
    assert not cur["ok"] and cur["implementation_changed"]


def test_evidence_is_judged_by_its_latest_outcome(tmp_path, monkeypatch):
    """A later failing DV row of the same attempt outranks an earlier pass; a
    coverage row without a verdict or without its attempt, and a synthesis
    row without a positive verdict or finite numbers, are not evidence."""
    db = ready_project(tmp_path, monkeypatch, stage="blocks", with_rtl=True)
    sb = Scoreboard(tmp_path)
    bid = "b-tiny-probe"
    B.record_dispatch(db, build_id=bid, module="tiny", entry="build_module", graph="build", thread_id="t",
                      inputs=B.module_inputs(db, tmp_path, "tiny"))
    sb.record_dv(block="tiny", scope="rtl", source="gate", attempt=1, passed=True, build_id=bid)
    sb.record_coverage(block="tiny", scope="rtl", pct=None, uncovered={}, build_id=bid)
    sb.record_ppa(block="tiny", attempt=1, source="gate", probe="synth", cells=10, build_id=bid, stage="synth")
    problems, _ = B.verify_build_evidence(tmp_path, bid, attempt=1, timing_required=True)
    assert any("coverage_results row is not attempt 1" in p for p in problems)
    assert any("no PPA verdict" in p and "CORESMITH_PPA_GATE" in p for p in problems)
    assert any("no finite measured WNS" in p for p in problems)
    sb.record_ppa(block="tiny", attempt=1, source="gate", probe="synth", cells=10, wns_ns=-0.2, ppa_ok=False, build_id=bid, stage="synth")
    assert any("no positive PPA verdict" in p for p in B.verify_build_evidence(tmp_path, bid, attempt=1, timing_required=True)[0])
    sb.record_coverage(block="tiny", scope="rtl", pct=None, attempt=1, uncovered={}, build_id=bid)
    problems, _ = B.verify_build_evidence(tmp_path, bid, attempt=1, timing_required=False)
    assert any("was not measured" in p and "no percentage recorded" in p for p in problems)
    sb.record_coverage(block="tiny", scope="rtl", pct=80.0, attempt=1, uncovered={"applicable": True}, build_id=bid)
    problems, _ = B.verify_build_evidence(tmp_path, bid, attempt=1, timing_required=False)
    assert any("no declared floor" in p for p in problems) and any("not a recorded pass" in p for p in problems)
    # coverage that was NOT measured is never evidence, whatever the diagnostic reason
    sb.record_ppa(block="tiny", attempt=1, source="gate", probe="synth", cells=10, wns_ns=0.2, ppa_ok=True, build_id=bid, stage="synth")
    from orchestrator.harness import coverage as cov
    unavailable = [
        {"applicable": False, "reason": "line-coverage gate disabled (CORESMITH_LINE_COV_GATE=0)"},
        {"applicable": False, "reason": "no coverage.dat produced by the sim"},
        {"applicable": False, "reason": "verilator_coverage not on PATH"},
        {"applicable": False, "reason": "no coverage points instrumented / annotate produced no output"},
        {"applicable": False, "reason": "coverage not evaluated"},
        {"applicable": False, "reason": "coverage unavailable (evaluation error)"},
    ]
    for rec in unavailable:
        sb.record_coverage(block="tiny", scope="rtl", attempt=1, uncovered=rec, build_id=bid)
        problems, _ = B.verify_build_evidence(tmp_path, bid, attempt=1, timing_required=True)
        assert len(problems) == 1 and "was not measured" in problems[0] and rec["reason"] in problems[0], rec
    # the real shape the engine records with the gate disabled
    monkeypatch_env = dict(os.environ)
    os.environ["CORESMITH_LINE_COV_GATE"] = "0"
    try:
        rec = cov.coverage_record(tmp_path / "no-sim-dir")
    finally:
        os.environ.clear()
        os.environ.update(monkeypatch_env)
    assert rec["applicable"] is False
    sb.record_coverage(block="tiny", scope="rtl", attempt=1, uncovered=rec, build_id=bid)
    assert any("CORESMITH_LINE_COV_GATE must be on" in p for p in B.verify_build_evidence(tmp_path, bid, attempt=1, timing_required=True)[0])
    # a measured closure is
    sb.record_coverage(block="tiny", scope="rtl", attempt=1, pct=91.0, uncovered={"applicable": True, "floor": 70.0, "passed": True}, build_id=bid)
    assert B.verify_build_evidence(tmp_path, bid, attempt=1, timing_required=True)[0] == []
    # the LATEST DV outcome decides: a later failure of the same attempt
    sb.record_dv(block="tiny", scope="rtl", source="gate", attempt=1, passed=False, build_id=bid)
    problems, _ = B.verify_build_evidence(tmp_path, bid, attempt=1, timing_required=True)
    assert problems == ["the latest dv_results row of attempt 1 is not a non-skipped pass"]
    # a pass of a later attempt restores it; a pass recorded by an agent does not
    sb.record_dv(block="tiny", scope="rtl", source="agent", attempt=2, passed=True, build_id=bid)
    assert any("not the gate's own verdict" in p for p in B.verify_build_evidence(tmp_path, bid, attempt=2, timing_required=False)[0])
    sb.record_dv(block="tiny", scope="rtl", source="gate", attempt=2, passed=True, build_id=bid)
    problems, _ = B.verify_build_evidence(tmp_path, bid, attempt=2, timing_required=False)
    assert problems == ["the latest coverage_results row is not attempt 2's (attempt 1)",
                        "the latest synthesis ppa_history row is attempt 1, not 2"]


def test_publication_is_atomic_immutable_and_survives_a_view_export_failure(tmp_path, monkeypatch):
    from orchestrator.state_store.project_db import ProjectDB
    db = ready_project(tmp_path, monkeypatch, stage="blocks", with_rtl=True)
    bid = _complete_build(db, tmp_path)
    first = B.get_build(db, bid)
    # a replay of the completion changes nothing (the record is immutable)
    out = db.publish_build_result("tiny", {"done": True, "attempt": 9}, build_id=bid, build_result={"attempt": 9})
    assert out["replayed"] is True and out["committed"] is False
    again = B.get_build(db, bid)
    assert again["result"] == first["result"] and again["finished_at"] == first["finished_at"]
    assert db.result("tiny", "best")["attempt"] == 1
    # a completion for another module, or of a build that is not active, is refused
    with pytest.raises(ValueError):
        db.publish_build_result("sink", {"done": True}, build_id=bid, build_result={})
    B.record_dispatch(db, build_id="b-x", module="tiny", entry="build_module", graph="build", thread_id="t",
                      inputs=B.module_inputs(db, tmp_path, "tiny"))
    B.mark_status(db, "b-x", "aborted")
    with pytest.raises(ValueError):
        db.publish_build_result("tiny", {"done": True}, build_id="b-x", build_result={})
    assert db.result("tiny", "best")["build_id"] == bid
    # the read-only views are a derived export: their failure is reported, the commit stands
    B.record_dispatch(db, build_id="b-y", module="tiny", entry="build_module", graph="build", thread_id="t",
                      inputs=B.module_inputs(db, tmp_path, "tiny"))

    def boom(self, block):
        raise OSError("views unwritable")
    monkeypatch.setattr(ProjectDB, "export_block_views", boom)
    out = db.publish_build_result("tiny", {"done": True, "attempt": 1}, build_id="b-y", build_result={"attempt": 1})
    assert out["committed"] and out["views_exported"] is False and "views unwritable" in out["view_error"]
    assert B.get_build(db, "b-y")["status"] == "completed" and db.result("tiny", "best")["build_id"] == "b-y"
    assert B.get_build(db, bid)["status"] == "completed"       # the earlier completion is history
    assert db.result("tiny", "best_superseded")["build_id"] == bid


def test_implementation_edited_after_the_build_is_not_current(tmp_path, monkeypatch):
    db = ready_project(tmp_path, monkeypatch, stage="blocks", with_rtl=True)
    bid = _complete_build(db, tmp_path)
    assert B.current_build_status(db, tmp_path, "tiny")["ok"]
    (tmp_path / "rtl" / "tiny.v").write_text("module tiny(); // hand edit\nendmodule\n")
    cur = B.current_build_status(db, tmp_path, "tiny")
    assert not cur["ok"] and cur["implementation_changed"] == [str(tmp_path / "rtl" / "tiny.v")]
    assert B.get_build(db, bid)["status"] == "completed"


def test_manual_or_diagnostic_publication_is_not_a_deliverable(tmp_path, monkeypatch):
    db = ready_project(tmp_path, monkeypatch, stage="blocks", with_rtl=True)
    db.set_result("tiny", "best", {"done": True, "sim_passed": True, "published_by": "cluster"})
    cur = B.current_build_status(db, tmp_path, "tiny")
    assert not cur["ok"] and "names no recorded build" in cur["reasons"][0]
    assert "BLOCK_NOT_BUILT" in _codes(st.entry(db, tmp_path, "blocks"))
    lin = B.lineage(db, tmp_path, "tiny")
    assert lin["modules"]["tiny"]["intended_workflow"] is False and lin["intended_workflow"] is False
    # a pass that names a build the ledger does not know is equally refused
    db.set_result("tiny", "best", {"done": True, "build_id": "b-tiny-forged"})
    assert "is not recorded" in B.current_build_status(db, tmp_path, "tiny")["reasons"][0]
    # ... or a build of another module
    bid = _complete_build(db, tmp_path, "sink")
    db.set_result("tiny", "best", {"done": True, "build_id": bid})
    assert "belongs to module sink" in B.current_build_status(db, tmp_path, "tiny")["reasons"][0]


def test_block_done_cli_publication_carries_no_build_id(tmp_path, monkeypatch):
    """The ``coresmith block-done`` gate still runs and reports; a tool failure
    publishes nothing, and what a pass publishes is a diagnostic publication
    (``published_by``, no build id) that never satisfies the blocks exit."""
    from orchestrator.harness import verify as V
    from orchestrator.harness.tools import block as BT
    db = ready_project(tmp_path, monkeypatch, stage="blocks", with_rtl=True)
    (tmp_path / "tb" / "cocotb" / "test_tiny.py").write_text("import cocotb\n@cocotb.test()\nasync def t(dut):\n    pass\n")
    monkeypatch.setattr(V, "verify_rtl", lambda *a, **k: V.VerifyResult(False, infra_error=True, verdict="no simulator"))
    res = BT.block_done(db, tmp_path, "tiny")
    assert res["ok"] is False and res["tool_error"] is True
    assert db.result("tiny", "best") is None
    # the shape block-done publishes on a pass names an actor, never a build
    db.set_result("tiny", "best", {"done": True, "sim_passed": True, "published_by": "cluster", "attempt": 1})
    assert "names no recorded build" in B.current_build_status(db, tmp_path, "tiny")["reasons"][0]


def test_lineage_requires_the_graphs_own_completion(tmp_path, monkeypatch):
    """``intended_workflow`` needs more than a current pass: the completion
    must name the thread it ran on, that thread must have checkpoints in
    the graph's database, the graph events must carry the build id through
    init and completion, and the evidence rows must verify."""
    db = ready_project(tmp_path, monkeypatch, stage="blocks", with_rtl=True)
    bid = _complete_build(db, tmp_path)
    lin = B.lineage(db, tmp_path, "tiny")
    m = lin["modules"]["tiny"]
    assert m["published"]["ok"] and m["published"]["build_id"] == bid
    assert m["builds"][0]["checkpoints"] == 0 and m["intended_workflow"] is False
    reasons = m["published"]["graph_completion"]["reasons"]
    assert any("no checkpoints" in r for r in reasons) and any("graph events" in r for r in reasons)
    assert _codes(st.entry(db, tmp_path, "blocks")).get("BLOCK_NOT_BUILT") is None   # stage gate: evidence-complete
    # a synthetic checkpoint for the thread is not enough either: no events, and a
    # pipeline-dispatched record needs its subgraph namespace in the pipeline checkpoint
    import sqlite3
    con = sqlite3.connect(str(tmp_path / ".coresmith" / "build_checkpoint.db"))
    con.execute("CREATE TABLE checkpoints (thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT)")
    con.execute("INSERT INTO checkpoints VALUES (?, '', 'c1')", (f"build-{bid}",))
    con.commit()
    con.close()
    m = B.lineage(db, tmp_path, "tiny")["modules"]["tiny"]
    assert m["builds"][0]["checkpoints"] == 1 and m["intended_workflow"] is False
    assert all("checkpoints" not in r for r in m["published"]["graph_completion"]["reasons"])


def test_compare_reports_scope_and_input_differences(tmp_path, monkeypatch):
    db = ready_project(tmp_path, monkeypatch, stage="blocks", with_rtl=True)
    a = _complete_build(db, tmp_path)
    (tmp_path / "arch" / "uarch_specs" / "tiny.md").write_text("# spec v2\n")
    db.register_artifact("uarch:tiny", "arch/uarch_specs/tiny.md")
    b = _complete_build(db, tmp_path)
    res = B.compare(db, tmp_path, a, b)
    assert res["same_module"] and res["a"]["ppa"]["stage"] == "synth" and res["a"]["ppa"]["power_basis"] == "unavailable"
    assert any(r.startswith("spec") for r in res["input_differences"])
    assert [x["id"] for x in B.builds_for(db, "tiny")] == [b, a]              # history, newest first
    assert db.result("tiny", "best_superseded")["build_id"] == a               # the earlier pass archived


def test_record_dispatch_refuses_unknown_entries_and_keeps_rows_immutable(tmp_path, monkeypatch):
    db = ready_project(tmp_path, monkeypatch)
    inputs = B.module_inputs(db, tmp_path, "tiny")
    with pytest.raises(ValueError):
        B.record_dispatch(db, build_id="b-x", module="tiny", entry="manual", graph="build", thread_id="t", inputs=inputs)
    B.record_dispatch(db, build_id="b-y", module="tiny", entry="build_module", graph="build", thread_id="t", inputs=inputs)
    B.mark_status(db, "b-y", "failed", error="boom")
    B.mark_status(db, "b-y", "running", terminal=False)      # a terminal row may be reopened by a resume ...
    assert B.get_build(db, "b-y")["status"] == "running"
    B.mark_status(db, "b-y", "completed")
    B.mark_status(db, "b-y", "failed", error="late")          # ... but a completed one is never moved back
    assert B.get_build(db, "b-y")["status"] == "completed"
    assert B.get_build(db, "b-y")["inputs"] == inputs


def test_spec_revision_leaves_the_recorded_build_stale(tmp_path, monkeypatch):
    db = ready_project(tmp_path, monkeypatch, stage="blocks", with_rtl=True)
    inputs = B.module_inputs(db, tmp_path, "tiny")
    B.record_dispatch(db, build_id="b-r", module="tiny", entry="build_module", graph="build", thread_id="t", inputs=inputs)
    (tmp_path / "arch/uarch_specs/tiny.md").write_text("# revised specification\n")
    assert B.stale_reasons(inputs, B.module_inputs(db, tmp_path, "tiny"))
    assert B.get_build(db, "b-r")["inputs"] == inputs


def test_worker_binding_is_resolved_by_the_adapter(tmp_path, monkeypatch):
    """The recorded worker is what the engine's adapter would run: the
    provider from ``_detect_provider`` and the model id from
    ``_resolve_model`` over the persisted env applied to the ambient one
    (aliases resolved through the adapter's catalogue, the provider-specific
    selector outranking CORESMITH_MODEL, kimi's KIMI_MODEL_NAME outranking
    both), read as a mapping -- the process environment is not mutated.
    Only the provider and the deciding selector are recorded."""
    from orchestrator.langchain.agents import coresmith_llm as L
    monkeypatch.setenv("CORESMITH_MODEL", "generic")
    monkeypatch.setenv("CORESMITH_CODEX_MODEL", "codex-specific")
    monkeypatch.setenv("SECRET_TOKEN", "do-not-record")
    write_env(tmp_path, {"CORESMITH_LLM_PROVIDER": "codex"})
    w = B.worker_binding(tmp_path)
    assert w["provider"] == "codex" and w["provider_key"] == "codex_cli" and w["model_var"] == "CORESMITH_CODEX_MODEL"
    assert w["model_declared"] == "codex-specific" and w["model"] == L._resolve_model("", "codex_cli")
    assert w["explicit"] is False and w["model_source"] == "environment"          # the selector is not persisted
    # an alias is recorded as the id the adapter resolves it to
    write_env(tmp_path, {"CORESMITH_LLM_PROVIDER": "claude", "CORESMITH_MODEL": "sonnet-5"})
    w = B.worker_binding(tmp_path)
    assert w["model_declared"] == "sonnet-5" and w["model"] == L._CLI_MODEL_MAP["sonnet-5"] and w["explicit"] is True
    assert os.environ["CORESMITH_MODEL"] == "generic"                             # nothing mutated
    write_env(tmp_path, {"CORESMITH_LLM_PROVIDER": "kimi", "CORESMITH_KIMI_MODEL": "k-persisted"})
    monkeypatch.setenv("KIMI_MODEL_NAME", "k-env")
    w = B.worker_binding(tmp_path)
    assert w["model_var"] == "KIMI_MODEL_NAME" and w["model_declared"] == "k-env" and w["explicit"] is False
    assert w["model"] == L.KIMI_ENV_MODEL_SENTINEL == L._resolve_model("", "kimi_cli")
    monkeypatch.delenv("KIMI_MODEL_NAME")
    w = B.worker_binding(tmp_path)
    assert w["model_declared"] == "k-persisted" and w["model"] == L._KIMI_MODEL_MAP.get("k-persisted", "k-persisted")
    assert w["explicit"] is True and "do-not-record" not in json.dumps(w)
    # the persisted file overrides the ambient value of the same selector
    monkeypatch.setenv("CORESMITH_KIMI_MODEL", "k-ambient")
    assert B.worker_binding(tmp_path)["model_declared"] == "k-persisted"
    # an unsupported provider is an error the binding reports, never a silent default
    write_env(tmp_path, {"CORESMITH_LLM_PROVIDER": "nope", "CORESMITH_MODEL": "x"})
    w = B.worker_binding(tmp_path)
    assert w["explicit"] is False and w["error"] and "Unsupported" in w["error"]


def test_produced_identity_covers_the_testbenchs_local_imports_and_keeps_immutable_copies(tmp_path, monkeypatch):
    db = ready_project(tmp_path, monkeypatch, stage="blocks", with_rtl=True)
    tbdir = tmp_path / "tb" / "cocotb"
    (tbdir / "vip").mkdir()
    (tbdir / "vip" / "__init__.py").write_text("")
    (tbdir / "vip" / "stream.py").write_text("import helpers\nclass Monitor: pass\n")
    (tbdir / "helpers.py").write_text("def check(x): return x\n")
    (tbdir / "test_tiny.py").write_text("import cocotb\nfrom vip.stream import Monitor\nimport helpers\n")
    deps = B.tb_dependencies(tbdir / "test_tiny.py")
    assert set(map(Path, deps)) == {tbdir / "vip" / "stream.py", tbdir / "helpers.py"}   # transitive, local only
    bid = _complete_build(db, tmp_path)
    produced = B.get_build(db, bid)["result"]["produced"]["files_sha256"]
    assert str(tbdir / "helpers.py") in produced and str(tbdir / "vip" / "stream.py") in produced
    assert B.current_build_status(db, tmp_path, "tiny")["ok"]
    (tbdir / "helpers.py").write_text("def check(x): return not x\n")
    cur = B.current_build_status(db, tmp_path, "tiny")
    assert not cur["ok"] and any("helpers.py" in p for p in cur["implementation_changed"])
    (tbdir / "helpers.py").write_text("def check(x): return x\n")
    # immutable copies under .coresmith/builds/<id>/, keyed by the original path
    snap = B.snapshot_artifacts(tmp_path, bid, produced)
    assert snap["errors"] == [] and Path(snap["dir"]) == tmp_path / ".coresmith" / "builds" / bid
    copy = Path(snap["files"][str(tmp_path / "rtl" / "tiny.v")])
    assert copy.read_bytes() == (tmp_path / "rtl" / "tiny.v").read_bytes() and str(copy).startswith(snap["dir"])
    (tmp_path / "rtl" / "tiny.v").write_text("module tiny(); // overwritten by the next build\nendmodule\n")
    assert B.file_sha256(copy) == produced[str(tmp_path / "rtl" / "tiny.v")]
    # a replay keeps the snapshot: an existing copy with the recorded bytes is kept, other bytes are an error
    again = B.snapshot_artifacts(tmp_path, bid, produced)
    assert again["errors"] == [] and again["files"] == snap["files"]
    bad = B.snapshot_artifacts(tmp_path, bid, {str(tmp_path / "rtl" / "tiny.v"): B.file_sha256(tmp_path / "rtl" / "tiny.v")})
    assert bad["files"] == {} and "immutable" in bad["errors"][0]
    assert B.file_sha256(copy) == produced[str(tmp_path / "rtl" / "tiny.v")]
    # two external files of the same name never collide, and every copy is verified by hash
    ext = tmp_path.parent / f"{tmp_path.name}-vendors"
    (ext / "vendor_a").mkdir(parents=True)
    (ext / "vendor_b").mkdir(parents=True)
    (ext / "vendor_a" / "defs.vh").write_text("`define A 1\n")
    (ext / "vendor_b" / "defs.vh").write_text("`define B 2\n")
    files = {str(ext / "vendor_a" / "defs.vh"): B.file_sha256(ext / "vendor_a" / "defs.vh"),
             str(ext / "vendor_b" / "defs.vh"): B.file_sha256(ext / "vendor_b" / "defs.vh")}
    snap2 = B.snapshot_artifacts(tmp_path, "b-ext", files)
    assert snap2["errors"] == [] and len(set(snap2["files"].values())) == 2
    for orig, dest in snap2["files"].items():
        assert B.file_sha256(dest) == files[orig] and "/external/" in dest
    # a recorded hash that does not match the source (it changed under the archive) is an error, not a copy
    stale = {str(ext / "vendor_a" / "defs.vh"): "0" * 64}
    assert "do not match the recorded hash" in B.snapshot_artifacts(tmp_path, "b-ext2", stale)["errors"][0]
    assert "no recorded hash" in B.snapshot_artifacts(tmp_path, "b-ext3", {str(ext / "nope.vh"): None})["errors"][0]


def test_composition_manifest_covers_every_declared_input_and_the_tooling(tmp_path, monkeypatch):
    from orchestrator.langgraph import pipeline_helpers as ph
    db = ready_project(tmp_path, monkeypatch, stage="blocks", with_rtl=True)
    for m in ("tiny", "sink"):
        _complete_build(db, tmp_path, m)
    (tmp_path / "rtl" / "soc.v").write_text("module soc(); endmodule\n")
    tb = tmp_path / "tb" / "cocotb" / "test_soc.py"
    tb.write_text("import cocotb\n")
    lib = tmp_path / "x.lib"
    lib.write_text("library(x) {}\n")
    monkeypatch.setattr(ph, "LIBERTY_FILE", lib)
    m = B.composition_identity(db, tmp_path, top_rtl_path="rtl/soc.v", files=[str(tb)],
                               context={"liberty_sha256": B.file_sha256(lib), "clock_mhz": 50.0})
    assert set(m["modules"]) == {"tiny", "sink"} and str(tb) in m["files"]
    assert (tmp_path / ".coresmith" / "compositions" / f"{m['sha256']}.json").is_file()
    assert B.composition_status(db, tmp_path, m["sha256"])["ok"]
    tb.write_text("import cocotb  # edited\n")                          # a test input changed
    s = B.composition_status(db, tmp_path, m["sha256"])
    assert not s["ok"] and any("test_soc.py" in r for r in s["reasons"])
    tb.write_text("import cocotb\n")
    lib.write_text("library(y) {}\n")                                    # a backend constraint changed
    s = B.composition_status(db, tmp_path, m["sha256"])
    assert not s["ok"] and any("liberty" in r for r in s["reasons"])
    lib.write_text("library(x) {}\n")
    _complete_build(db, tmp_path, "tiny")                                # a module rebuilt
    s = B.composition_status(db, tmp_path, m["sha256"])
    assert not s["ok"] and any(r.startswith("tiny:") for r in s["reasons"])
    assert not B.composition_status(db, tmp_path, "unknown")["ok"] and not B.composition_status(db, tmp_path, None)["ok"]
    # the verdict measured the top the project selected THEN; selecting another
    # elaborated top (the old file untouched) is another composition
    db.add_integration_snapshot({"tier": "1", "top": "soc", "rtl_path": "rtl/soc.v", "real_blocks": ["tiny", "sink"],
                                 "stub_blocks": [], "elaborated": True})
    m2 = B.composition_identity(db, tmp_path, files=[str(tb)])
    assert m2["top_rtl_path"] == str(tmp_path / "rtl" / "soc.v") and B.composition_status(db, tmp_path, m2["sha256"])["ok"]
    (tmp_path / "rtl" / "soc_v2.v").write_text("module soc(); // v2\nendmodule\n")
    db.add_integration_snapshot({"tier": "1", "top": "soc", "rtl_path": "rtl/soc_v2.v", "real_blocks": ["tiny", "sink"],
                                 "stub_blocks": [], "elaborated": True})
    s = B.composition_status(db, tmp_path, m2["sha256"])
    assert not s["ok"] and any("selected top is now soc_v2.v" in r for r in s["reasons"])
    # a declared input that did not exist when measured is never positive evidence
    m3 = B.composition_identity(db, tmp_path, top_rtl_path="rtl/soc_v2.v", files=[str(tmp_path / "tb" / "missing_tb.py")])
    assert m3["missing"] == [str(tmp_path / "tb" / "missing_tb.py")]
    s = B.composition_status(db, tmp_path, m3["sha256"])
    assert not s["ok"] and any("did not exist when the verdict was measured" in r for r in s["reasons"])


def test_the_adopted_candidate_is_the_selected_top(tmp_path, monkeypatch):
    """The canonical adoption is the hierarchy-validated candidate receipt:
    when one exists it selects the top over an older shell snapshot; an
    edited candidate and a replaced candidate both re-open the verdict; a
    receipt that no longer validates is a problem in itself."""
    from orchestrator.harness.top_module import validated_candidate, write_candidate_receipt
    from orchestrator.state_store.project_db import open_project
    monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))
    (tmp_path / "rtl").mkdir()
    db = open_project(tmp_path)                                   # no task declaration: any top may be adopted
    for name in ("old_shell", "adopted_top", "next_top"):
        (tmp_path / "rtl" / f"{name}.v").write_text(f"module {name}(input a, output z); assign z = a; endmodule\n")
    db.add_integration_snapshot({"tier": "final", "top": "old_shell", "rtl_path": "rtl/old_shell.v", "real_blocks": [],
                                 "stub_blocks": [], "elaborated": True})
    assert B.selected_top(db, tmp_path) == (str(tmp_path / "rtl" / "old_shell.v"), None)   # the shell path
    actual = tmp_path / "rtl" / "adopted_top.v"
    receipt = write_candidate_receipt(tmp_path, "adopted_top", str(actual), [], expected_blocks=[])
    assert validated_candidate(tmp_path) == receipt
    assert B.selected_top(db, tmp_path) == (str(actual), None)                            # the receipt wins
    comp = B.composition_identity(db, tmp_path, top_rtl_path=str(actual))
    assert B.composition_status(db, tmp_path, comp["sha256"])["ok"]
    assert B.composition_identity(db, tmp_path)["top_rtl_path"] == str(actual)             # the default top
    original = actual.read_text()
    actual.write_text(original.replace("assign z = a;", "assign z = ~a;"))
    s = B.composition_status(db, tmp_path, comp["sha256"])
    assert not s["ok"] and any("no longer validates" in r for r in s["reasons"]) and any("bytes changed" in r for r in s["reasons"])
    actual.write_text(original)
    assert B.composition_status(db, tmp_path, comp["sha256"])["ok"]
    write_candidate_receipt(tmp_path, "next_top", str(tmp_path / "rtl" / "next_top.v"), [], expected_blocks=[])
    s = B.composition_status(db, tmp_path, comp["sha256"])
    assert not s["ok"] and any("selected top is now next_top.v" in r for r in s["reasons"])


def test_tooling_snapshot_is_the_engines_own_resolution(tmp_path, monkeypatch):
    from orchestrator.langgraph import pipeline_helpers as ph
    lib = tmp_path / "pdk" / "sky130A" / "libs.ref" / "sky130_fd_sc_hd" / "lib" / "x.lib"
    lib.parent.mkdir(parents=True)
    lib.write_text("library(x) {}\n")
    monkeypatch.setattr(ph, "LIBERTY_FILE", lib)
    t = B.tooling_snapshot(tmp_path, target_clock_mhz=50.0)
    assert t["liberty"] == str(lib) and t["pdk"] == "sky130A" and t["liberty_sha256"] == B.file_sha256(lib)
    assert t["synth_generic"] is False and isinstance(t["yosys"], dict)
    monkeypatch.setattr(ph, "LIBERTY_FILE", tmp_path / "missing.lib")
    t = B.tooling_snapshot(tmp_path, target_clock_mhz=50.0)
    assert t["liberty"] is None and t["pdk"] == "generic (no liberty)"


def test_legacy_run_scoped_binding_survives_a_run_rotation(tmp_path, monkeypatch):
    """A target bound under a run id (a database written before bindings were
    project-scoped) is preserved as the project's binding when the run id
    rotates, and never overwrites a canonical project binding."""
    from orchestrator.harness import targets as T
    db = ready_project(tmp_path, monkeypatch)
    with db._tx() as con:
        con.execute("DELETE FROM run_flags WHERE name='target:tiny'")
    db.set_setting("run_id", "run-legacy")
    db.set_flag("target:tiny", {"top": "tiny", "sources": ["rtl/tiny.v"], "cwd": "."})     # the old, run-scoped row
    assert T.load(tmp_path, "tiny", require_files=False)["top"] == "tiny"
    db.begin_run()
    assert db.run_id() != "run-legacy"
    assert T.load(tmp_path, "tiny", require_files=False)["sources"] == [str(tmp_path / "rtl" / "tiny.v")]
    assert db.get_flag("target:tiny", run_id="")["top"] == "tiny"
    # a canonical binding is never overwritten by a historical one
    T.bind(tmp_path, "tiny", {"top": "tiny", "sources": ["rtl/tiny.v"], "cwd": ".", "defines": {"NEW": 1}})
    db.set_flag("target:tiny", {"top": "stale", "sources": ["rtl/tiny.v"], "cwd": "."})
    db.begin_run()
    assert T.load(tmp_path, "tiny", require_files=False)["defines"] == {"NEW": 1}


def test_model_identity_follows_the_registered_path_and_reports_dependency_errors(tmp_path, monkeypatch):
    db = ready_project(tmp_path, monkeypatch)
    custom = tmp_path / "model" / "impl" / "tiny_impl.cpp"
    custom.parent.mkdir()
    custom.write_text('#include "../tiny_model.h"\n#include "missing_local.h"\nint x;\n')
    db.upsert_model("tiny", path="model/impl/tiny_impl.cpp", sha="x", build_ok=True, smoke_ok=True)
    ident = B.model_identity(tmp_path, "tiny", db)
    assert ident["path"] == str(custom) and ident["path_source"] == "registered"
    assert str(tmp_path / "model" / "tiny_model.h") in ident["deps"] and ident["unresolved_includes"] == ["missing_local.h"]
    assert ident["error"] is None and ident["sha16"] == B.file_sha256(custom)[:16]
    # an unreadable dependency is an error, never an empty valid identity
    hp = tmp_path / "model" / "tiny_model.h"
    os.chmod(hp, 0)
    try:
        ident = B.model_identity(tmp_path, "tiny", db)
        assert ident["error"] and "PermissionError" in ident["error"]
        assert "MODEL_INVALID" in _codes(st.module_ready(db, tmp_path, "tiny"))
    finally:
        os.chmod(hp, 0o644)
