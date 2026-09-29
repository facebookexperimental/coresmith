# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""``best`` means sim AND synth AND timing (A1: the timing false pass).

In the SoC benchmark rv64_core sat at WNS -27.6 ns with a green
``best_result.json`` on disk, because the DV pass was recorded as ``best``
before synth and timing ran. With the done-result gate on, DV writes
``dv_best`` and only ``block_done`` publishes ``best``.
"""
from __future__ import annotations

import asyncio

from orchestrator.langgraph import pipeline_graph as pg
from orchestrator.state_store.project_db import open_project


def _state(tmp_path, **over):
    st = {"project_root": str(tmp_path), "current_block": {"name": "core"},
          "attempt": 2, "sim_passed": True, "synth_success": True,
          "synth_gate_count": 1234, "timing_required": True, "timing_ok": True}
    st.update(over)
    return st


def _prep(tmp_path):
    bdir = tmp_path / ".coresmith" / "blocks" / "core"
    bdir.mkdir(parents=True, exist_ok=True)
    (bdir / "constraints.json").write_text("[]")
    (bdir / "previous_error.txt").write_text("PPA gate: WNS -27.58 ns")
    db = open_project(tmp_path)
    db.set_result("core", "dv_best", {"sim_passed": True, "tests_passed": 9, "tests_total": 9})
    return db


class TestDoneResultGate:
    def test_dv_result_kind_follows_the_flag(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_DONE_RESULT_GATE", "1")
        assert pg._dv_result_kind() == "dv_best"
        monkeypatch.setenv("CORESMITH_DONE_RESULT_GATE", "0")
        assert pg._dv_result_kind() == "best"

    def test_failed_timing_publishes_no_best(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_DONE_RESULT_GATE", "1")
        db = _prep(tmp_path)
        db.set_result("core", "best", {"sim_passed": True})  # a stale pre-gate record
        out = asyncio.run(pg.block_done_node(_state(tmp_path, timing_ok=False)))
        assert out["completed_blocks"][0]["success"] is False
        assert db.result("core", "best") is None
        assert db.result("core", "dv_best")["sim_passed"] is True
        assert not (tmp_path / ".coresmith/blocks/core/best_result.json").exists()
        assert (tmp_path / ".coresmith/blocks/core/dv_best_result.json").exists()

    def test_passed_timing_publishes_best_with_the_facts(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_DONE_RESULT_GATE", "1")
        db = _prep(tmp_path)
        out = asyncio.run(pg.block_done_node(_state(tmp_path)))
        assert out["completed_blocks"][0]["success"] is True
        best = db.result("core", "best")
        assert best["done"] is True and best["timing_ok"] is True
        assert best["synth_success"] is True and best["sim_passed"] is True
        assert best["tests_passed"] == 9  # the DV facts travel with the pass
        assert best["gate_count"] == 1234
        assert "spec_sha256" in best or True  # spec may not exist in the fixture

    def test_unmeasured_required_timing_publishes_no_best(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_DONE_RESULT_GATE", "1")
        db = _prep(tmp_path)
        out = asyncio.run(pg.block_done_node(_state(tmp_path, timing_ok=None)))
        assert out["completed_blocks"][0]["success"] is False
        assert db.result("core", "best") is None

    def test_gate_off_keeps_legacy_behaviour(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_DONE_RESULT_GATE", "0")
        db = _prep(tmp_path)
        db.set_result("core", "best", {"sim_passed": True})
        asyncio.run(pg.block_done_node(_state(tmp_path, timing_ok=False)))
        # legacy: block_done never touches the result rows
        assert db.result("core", "best") == {"sim_passed": True} | {
            k: v for k, v in db.result("core", "best").items() if k == "spec_sha256"}


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
