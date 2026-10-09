# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""``best`` means sim AND synth AND timing (A1: the timing false pass), and
it is published only as the committed result of a recorded build: by the
graph's ``block_done`` node, under the LangGraph thread the build was
recorded on, after the build's recorded inputs are re-checked against the
live project and the evidence rows are read back from the database.

In the SoC benchmark rv64_core sat at WNS -27.6 ns with a green
``best_result.json`` on disk, because the DV pass was recorded as ``best``
before synth and timing ran. DV writes ``dv_best`` and only ``block_done``
publishes ``best``; the former ``CORESMITH_DONE_RESULT_GATE=0`` escape no
longer opens anything."""
from __future__ import annotations

from orchestrator.langgraph import pipeline_graph as pg
from orchestrator.state_store import builds as B
from orchestrator.state_store.project_db import open_project
from orchestrator.state_store.store import Scoreboard
from orchestrator.tests.build_fixtures import run_block_done

_BID = "b-core-1"
_THREAD = f"build-{_BID}"


def _state(tmp_path, build_id=_BID, **over):
    st = {"project_root": str(tmp_path), "current_block": {"name": "core"},
          "attempt": 2, "sim_passed": True, "synth_success": True,
          "synth_gate_count": 1234, "timing_required": True, "timing_ok": True, "build_id": build_id,
          "rtl_path": str(tmp_path / "rtl" / "core.v"), "tb_path": str(tmp_path / "tb" / "test_core.py")}
    st.update(over)
    return st


def _prep(tmp_path, *, evidence: bool = True, build_id=_BID, thread_id=_THREAD):
    bdir = tmp_path / ".coresmith" / "blocks" / "core"
    bdir.mkdir(parents=True, exist_ok=True)
    (bdir / "constraints.json").write_text("[]")
    (bdir / "previous_error.txt").write_text("PPA gate: WNS -27.58 ns")
    (tmp_path / "rtl").mkdir(exist_ok=True)
    (tmp_path / "rtl" / "core.v").write_text("module core(); endmodule\n")
    (tmp_path / "tb").mkdir(exist_ok=True)
    (tmp_path / "tb" / "test_core.py").write_text("import cocotb\n")
    db = open_project(tmp_path)
    db.import_block_diagram({"blocks": [{"name": "core", "tier": 1}], "connections": []})
    db.set_result("core", "dv_best", {"sim_passed": True, "tests_passed": 9, "tests_total": 9, "build_id": build_id})
    inputs = B.module_inputs(db, tmp_path, "core")
    B.record_dispatch(db, build_id=build_id, module="core", entry="build_module", graph="build",
                      thread_id=thread_id, inputs=inputs)
    B.mark_started(db, build_id)
    if evidence:
        sb = Scoreboard(tmp_path)
        assert sb.record_dv(block="core", scope="rtl", source="gate", attempt=2, passed=True, build_id=build_id)
        assert sb.record_coverage(block="core", scope="rtl", pct=88.0, attempt=2,
                                  uncovered={"applicable": True, "floor": 80.0, "passed": True}, build_id=build_id)
        assert sb.record_ppa(block="core", attempt=2, source="gate", probe="synth", cells=1234, wns_ns=0.4, ppa_ok=True,
                             build_id=build_id, stage="synth", power_mw=None, power_basis="unavailable")
    return db


def _done(tmp_path, state, thread_id=_THREAD):
    return run_block_done(state, thread_id=thread_id)["completed_blocks"][-1]


class TestDoneResultGate:
    def test_the_gate_knob_no_longer_opens_anything(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_DONE_RESULT_GATE", "0")
        assert pg.done_result_gate_enabled() is True and pg._dv_result_kind() == "dv_best"
        monkeypatch.setenv("CORESMITH_DONE_RESULT_GATE", "1")
        assert pg._dv_result_kind() == "dv_best"

    def test_failed_timing_publishes_no_best(self, tmp_path):
        db = _prep(tmp_path)
        db.set_result("core", "best", {"sim_passed": True})  # a stale pre-gate record
        res = _done(tmp_path, _state(tmp_path, timing_ok=False))
        assert res["success"] is False
        assert db.result("core", "best") is None
        assert db.result("core", "best_superseded") == {"sim_passed": True}   # archived, not erased
        assert db.result("core", "dv_best")["sim_passed"] is True
        assert not (tmp_path / ".coresmith/blocks/core/best_result.json").exists()
        assert (tmp_path / ".coresmith/blocks/core/dv_best_result.json").exists()
        assert B.get_build(db, _BID)["status"] == "failed"

    def test_passed_timing_publishes_best_with_the_facts_and_completes_the_build(self, tmp_path):
        db = _prep(tmp_path)
        res = _done(tmp_path, _state(tmp_path))
        assert res["success"] is True and res["build_id"] == _BID
        best = db.result("core", "best")
        assert best["done"] is True and best["timing_ok"] is True
        assert best["synth_success"] is True and best["sim_passed"] is True
        assert best["tests_passed"] == 9  # the DV facts travel with the pass
        assert best["gate_count"] == 1234
        assert best["build_id"] == _BID and best["published_by"] == "graph"
        build = B.get_build(db, _BID)
        assert build["status"] == "completed" and build["result"]["rows"] == {"dv": 1, "ppa": 1, "coverage": 1}
        assert build["result"]["thread_id"] == _THREAD and build["result"]["produced"]["missing"] == []
        assert str(tmp_path / "rtl" / "core.v") in build["result"]["produced"]["files_sha256"]

    def test_unmeasured_required_timing_publishes_no_best(self, tmp_path):
        db = _prep(tmp_path)
        res = _done(tmp_path, _state(tmp_path, timing_ok=None))
        assert res["success"] is False
        assert db.result("core", "best") is None

    def test_missing_evidence_rows_are_a_persistence_failure(self, tmp_path):
        db = _prep(tmp_path, evidence=False)
        res = _done(tmp_path, _state(tmp_path))
        assert res["success"] is False and res["persistence_failed"] is True
        assert "dv_results" in res["error"] and "ppa_history" in res["error"]
        assert db.result("core", "best") is None
        assert B.get_build(db, _BID)["status"] == "failed_persistence"

    def test_a_failed_best_write_cannot_publish(self, tmp_path, monkeypatch):
        db = _prep(tmp_path)
        from orchestrator.state_store.project_db import ProjectDB

        def boom(self, *a, **k):
            raise RuntimeError("disk full")
        monkeypatch.setattr(ProjectDB, "publish_build_result", boom)
        res = _done(tmp_path, _state(tmp_path))
        assert res["success"] is False and "publication failed" in res["error"]
        assert db.result("core", "best") is None
        assert B.get_build(db, _BID)["status"] == "failed_persistence"

    def test_a_view_export_failure_does_not_fail_the_committed_pass(self, tmp_path, monkeypatch):
        db = _prep(tmp_path)
        from orchestrator.state_store.project_db import ProjectDB

        def boom(self, block):
            raise OSError("views unwritable")
        monkeypatch.setattr(ProjectDB, "export_block_views", boom)
        res = _done(tmp_path, _state(tmp_path))
        assert res["success"] is True and res["views_exported"] is False
        assert db.result("core", "best")["build_id"] == _BID and B.get_build(db, _BID)["status"] == "completed"

    def test_replay_after_completion_is_idempotent(self, tmp_path):
        db = _prep(tmp_path)
        _done(tmp_path, _state(tmp_path))
        first = B.get_build(db, _BID)
        res = _done(tmp_path, _state(tmp_path))          # the node re-executes (checkpoint replay)
        again = B.get_build(db, _BID)
        assert res["success"] is True
        assert again["status"] == "completed" and again["finished_at"] == first["finished_at"]
        assert again["result"] == first["result"]
        assert [b["id"] for b in B.builds_for(db, "core")] == [_BID]   # no second success manufactured

    def test_outside_a_graph_thread_nothing_is_published(self, tmp_path):
        import asyncio
        db = _prep(tmp_path)
        out = asyncio.run(pg.block_done_node(_state(tmp_path)))       # a diagnostic invocation, no LangGraph task
        res = out["completed_blocks"][0]
        assert res["success"] is False and "NO_GRAPH_THREAD" in res["error"]
        assert db.result("core", "best") is None and B.get_build(db, _BID)["status"] == "failed_persistence"

    def test_another_thread_than_the_recorded_one_cannot_complete_the_build(self, tmp_path):
        db = _prep(tmp_path)
        res = _done(tmp_path, _state(tmp_path), thread_id="build-someone-else")
        assert res["success"] is False and "THREAD_MISMATCH" in res["error"]
        assert db.result("core", "best") is None

    def test_inputs_changed_while_the_build_ran_is_not_a_pass(self, tmp_path, monkeypatch):
        """A worker or spec change between dispatch and completion: the
        evidence was earned under other inputs. INPUTS_CHANGED, the build
        fails (not a persistence failure), nothing is published."""
        from orchestrator.tests.build_fixtures import write_env
        db = _prep(tmp_path)
        write_env(tmp_path, {"CORESMITH_LLM_PROVIDER": "codex", "CORESMITH_CODEX_MODEL": "other"})
        res = _done(tmp_path, _state(tmp_path))
        assert res["success"] is False and res["inputs_changed"] is True and "worker" in res["error"]
        assert res["persistence_failed"] is False
        assert db.result("core", "best") is None and B.get_build(db, _BID)["status"] == "failed"

    def test_a_missing_produced_file_cannot_be_published(self, tmp_path):
        db = _prep(tmp_path)
        (tmp_path / "rtl" / "core.v").unlink()
        res = _done(tmp_path, _state(tmp_path))
        assert res["success"] is False and "produced output missing" in res["error"]
        assert db.result("core", "best") is None


class TestConsumers:
    def test_pass_record_falls_back_across_kinds(self, tmp_path):
        db = open_project(tmp_path)
        assert pg._block_pass_record(db, "x") is None
        db.set_result("x", "dv_best", {"sim_passed": True, "rtl_sha1": "abc"})
        assert pg._block_pass_record(db, "x")["rtl_sha1"] == "abc"
        assert pg._block_done_record(db, "x") is None
        db.set_result("x", "best", {"sim_passed": True, "done": True})
        assert pg._block_done_record(db, "x")["done"] is True

    def test_spec_change_invalidates_both_kinds(self, tmp_path):
        db = open_project(tmp_path)
        db.set_result("x", "dv_best", {"sim_passed": True, "spec_sha256": "old"})
        db.set_result("x", "best", {"sim_passed": True, "done": True, "spec_sha256": "old"})
        assert db.invalidate_results_for_specs({"x": "new"}) == ["x"]
        assert db.result("x", "best") is None and db.result("x", "dv_best") is None
        assert db.result("x", "spec_invalidated")["previous_kind"] in ("best", "dv_best")

    def test_legacy_pre_gate_best_is_imported_as_dv_best(self, tmp_path):
        import json
        bdir = tmp_path / ".coresmith" / "blocks" / "y"
        bdir.mkdir(parents=True)
        (bdir / "best_result.json").write_text(json.dumps({"sim_passed": True, "attempt": 1}))
        db = open_project(tmp_path)
        assert db.result("y", "best") is None
        assert db.result("y", "dv_best")["sim_passed"] is True

    def test_legacy_done_best_is_kept(self, tmp_path):
        import json
        bdir = tmp_path / ".coresmith" / "blocks" / "y"
        bdir.mkdir(parents=True)
        (bdir / "best_result.json").write_text(json.dumps({"sim_passed": True, "done": True}))
        db = open_project(tmp_path)
        assert db.result("y", "best")["done"] is True
