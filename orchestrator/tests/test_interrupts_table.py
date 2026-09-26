# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""C1-3: parked interrupts are rows; one branch can be answered while its
Send() siblings are still running."""
from __future__ import annotations

import asyncio
import threading
import time

import pytest

from orchestrator.state_store.interrupts import interrupt_id_for
from orchestrator.state_store.project_db import open_project


class TestRows:
    def test_park_is_idempotent_and_deterministic(self, tmp_path):
        db = open_project(tmp_path)
        db.begin_run("r1")
        p = {"type": "dv_failure", "block_name": "core", "attempt": 2, "message": "x"}
        iid, out = db.park_interrupt(p, graph="pipeline", node="ask_human")
        iid2, _ = db.park_interrupt({**p, "message": "different prose"},
                                    graph="pipeline", node="ask_human")
        assert iid == iid2 == out["interrupt_id"]
        assert iid == interrupt_id_for(p, graph="pipeline", node="ask_human", run_id="r1")
        assert len(db.interrupts(status="pending")) == 1
        # a different attempt is a different park
        iid3, _ = db.park_interrupt({**p, "attempt": 3}, graph="pipeline", node="ask_human")
        assert iid3 != iid

    def test_resolve_consume_lifecycle(self, tmp_path):
        db = open_project(tmp_path)
        iid, _ = db.park_interrupt({"type": "x"}, graph="pipeline", node="n")
        assert db.consume_interrupt(iid) is None            # nothing queued
        assert db.resolve_interrupt(iid, {"action": "retry"}, resolved_by="cli")
        assert not db.resolve_interrupt(iid, {"action": "skip"})   # only pending rows
        assert db.interrupt(iid)["status"] == "resolved"
        assert db.consume_interrupt(iid) == {"action": "retry"}
        assert db.interrupt(iid)["status"] == "consumed"
        # the same park raised again re-opens the row
        db.park_interrupt({"type": "x"}, graph="pipeline", node="n")
        assert db.interrupt(iid)["status"] == "pending"

    def test_lg_binding_and_resolved_for(self, tmp_path):
        db = open_project(tmp_path)
        a, _ = db.park_interrupt({"type": "a"}, graph="pipeline", node="n")
        b, _ = db.park_interrupt({"type": "b"}, graph="pipeline", node="n")
        db.bind_lg_id(a, "lg-a")
        db.bind_lg_id(b, "lg-b")
        db.resolve_interrupt(a, {"action": "retry"})
        assert db.resolved_for(["lg-a", "lg-b"]) == {"lg-a": {"action": "retry"}}
        assert db.consume_lg_interrupts(["lg-b"]) == [b]
        assert db.interrupt(b)["status"] == "consumed"

    def test_wait_for_resolution_returns_when_answered(self, tmp_path):
        db = open_project(tmp_path)
        iid, _ = db.park_interrupt({"type": "x"}, graph="pipeline", node="n")
        threading.Timer(0.2, lambda: db.resolve_interrupt(iid, {"action": "skip"})).start()
        t0 = time.time()
        assert db.wait_for_resolution(iid, wait_s=3, poll_s=0.05) == {"action": "skip"}
        assert time.time() - t0 < 2
        assert db.wait_for_resolution(iid, wait_s=0.1, poll_s=0.05) is None


class TestParkHelper:
    def test_queued_answer_skips_interrupt(self, tmp_path, monkeypatch):
        from orchestrator.langgraph import pipeline_graph as pg
        monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))
        monkeypatch.setenv("CORESMITH_INTERRUPT_WAIT_S", "3")
        db = open_project(tmp_path)
        raised = []
        monkeypatch.setattr(pg, "interrupt", lambda payload: raised.append(payload) or {"action": "parked"})
        payload = {"type": "dv_failure", "block_name": "core", "supported_actions": ["retry"]}
        iid = interrupt_id_for(payload, graph="pipeline", node="dv_failure", run_id=db.run_id())
        threading.Timer(0.3, lambda: db.resolve_interrupt(iid, {"action": "retry"})).start()
        out = pg._park(payload)
        assert out == {"action": "retry"} and raised == []
        assert db.interrupt(iid)["status"] == "consumed"

    def test_no_answer_parks_with_the_id_in_the_payload(self, tmp_path, monkeypatch):
        from orchestrator.langgraph import pipeline_graph as pg
        monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))
        monkeypatch.setenv("CORESMITH_INTERRUPT_WAIT_S", "0")
        db = open_project(tmp_path)
        raised = []
        monkeypatch.setattr(pg, "interrupt", lambda payload: raised.append(payload) or {"action": "x"})
        pg._park({"type": "prd_questions"}, graph="architecture", node="Escalate PRD")
        assert raised[0]["interrupt_id"].startswith("int-")
        row = db.interrupt(raised[0]["interrupt_id"])
        assert row["status"] == "pending" and row["graph"] == "architecture"

    def test_resolve_interrupt_parks_through_the_table(self, tmp_path, monkeypatch):
        from orchestrator.langgraph import pipeline_graph as pg
        monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))
        monkeypatch.delenv("CORESMITH_ENABLE_CHIP_LEAD", raising=False)
        db = open_project(tmp_path)
        monkeypatch.setattr(pg, "interrupt", lambda payload: {"action": "x"})
        asyncio.run(pg._resolve_interrupt({"type": "uarch_review", "block_name": "b"}))
        assert db.interrupts(status="pending")[0]["block"] == "b"


class _Intr:
    def __init__(self, iid, value):
        self.id, self.value = iid, value


class _Task:
    def __init__(self, interrupts):
        self.interrupts = interrupts


class _Snap:
    def __init__(self, tasks, values=None):
        self.tasks, self.values, self.next = tasks, values or {}, ["x"]


class _RunningTask:
    def done(self):
        return False


class TestDaemonTargetedResume:
    @pytest.mark.asyncio
    async def test_in_flight_resume_with_interrupt_id_is_queued_202(self, tmp_path, monkeypatch):
        from orchestrator.daemon import server as ds
        monkeypatch.setattr(ds, "_PROJECT_ROOT", str(tmp_path))
        db = open_project(tmp_path)
        iid, _ = db.park_interrupt({"type": "dv_failure", "block_name": "b"},
                                   graph="pipeline", node="n")

        async def _ensure():
            return None
        monkeypatch.setattr(ds._pipeline, "ensure_graph", _ensure)
        monkeypatch.setattr(ds._pipeline, "task", _RunningTask())
        resp = await ds.run_resume(ds.ResumeRequest(action="retry", interrupt_id=iid))
        assert resp.status_code == 202
        assert db.interrupt(iid)["status"] == "resolved"
        assert db.interrupt(iid)["resolution"]["action"] == "retry"
        # unknown id -> 404; no id while in flight -> 409 as before
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as e:
            await ds.run_resume(ds.ResumeRequest(action="retry", interrupt_id="nope"))
        assert e.value.status_code == 404
        with pytest.raises(HTTPException) as e:
            await ds.run_resume(ds.ResumeRequest(action="retry"))
        assert e.value.status_code == 409

    @pytest.mark.asyncio
    async def test_boundary_applier_resumes_only_the_answered_branch(self, tmp_path):
        from orchestrator.graph_lifecycle import GraphLifecycle
        db = open_project(tmp_path)
        a, pa = db.park_interrupt({"type": "a", "block_name": "b1"}, graph="pipeline", node="n")
        b, pb = db.park_interrupt({"type": "b", "block_name": "b2"}, graph="pipeline", node="n")
        db.resolve_interrupt(a, {"action": "retry"})
        calls = []
        snaps = iter([
            _Snap([_Task([_Intr("lg-a", pa), _Intr("lg-b", pb)])]),   # after first invoke
            _Snap([_Task([_Intr("lg-b", pb)])]),                       # after targeted resume
            _Snap([_Task([_Intr("lg-b", pb)])]),                       # final status probe
        ])

        class _G:
            async def ainvoke(self, inp, config):
                calls.append(inp)

            async def aget_state(self, config):
                return next(snaps)
        gl = GraphLifecycle.__new__(GraphLifecycle)
        gl.project_root = str(tmp_path)
        gl.graph = _G()
        gl.name = "pipeline"
        gl.status = "idle"
        gl.error_message = ""
        await gl.run_task({"start": True}, {})
        assert gl.status == "interrupted"          # b is still parked
        assert len(calls) == 2
        assert calls[1].resume == {"lg-a": {"action": "retry"}}
        assert db.interrupt(a)["status"] == "consumed"
        assert db.interrupt(b)["status"] == "pending"
        assert db.interrupt(b)["lg_interrupt_id"] == "lg-b"


class TestSurfaces:
    def test_cli_lists_and_resolves(self, tmp_path):
        import argparse

        from orchestrator.harness import cli
        db = open_project(tmp_path)
        iid, _ = db.park_interrupt({"type": "x", "supported_actions": ["retry"]},
                                   graph="pipeline", node="n")
        parser = argparse.ArgumentParser()
        cli.register_subcommands(parser.add_subparsers())
        args = parser.parse_args(["interrupts", "--project-root", str(tmp_path), "--pending"])
        assert cli.cmd_interrupts(args) == 0
        args = parser.parse_args(["interrupts", "--project-root", str(tmp_path),
                                  "--resolve", iid, "--action", "retry", "--feedback", "go"])
        assert cli.cmd_interrupts(args) == 0
        assert db.interrupt(iid)["resolution"] == {"action": "retry", "feedback": "go",
                                                   "rationale": "", "block_actions": {}}
        args = parser.parse_args(["interrupts", "--project-root", str(tmp_path),
                                  "--resolve", iid])
        assert cli.cmd_interrupts(args) != 0   # --action required

    @pytest.mark.asyncio
    async def test_mcp_list_and_queue(self, tmp_path, monkeypatch):
        import json

        from orchestrator import mcp_server as m
        monkeypatch.setattr(m, "_project_root", lambda: str(tmp_path))
        db = open_project(tmp_path)
        iid, _ = db.park_interrupt({"type": "x"}, graph="pipeline", node="n")
        out = json.loads(await m.list_interrupts())
        assert out["count"] == 1 and out["interrupts"][0]["id"] == iid
        monkeypatch.setattr(m._pipeline, "task", _RunningTask())
        out = json.loads(await m.resume_pipeline(action="retry", interrupt_id=iid))
        assert out.get("queued") is True
        assert db.interrupt(iid)["status"] == "resolved"
