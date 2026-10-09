# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""``coresmith build module``: the existing compiled block subgraph runs under
a persistent SQLite checkpoint through the daemon's build entry; success is
the durable record (the published pass and the ``completed`` row committed
together by ``block_done``), a park survives lifecycle recreation and resumes
by build id, a failed persistence write cannot publish, a second start never
overwrites an in-flight build, and the pipeline's own fan-out allocates the
same build identity at its init boundary."""
from __future__ import annotations

import asyncio
import json
import sqlite3
import types

import pytest

from orchestrator.state_store import builds as B
from orchestrator.state_store import stages as st
from orchestrator.state_store.store import Scoreboard
from orchestrator.tests.build_fixtures import (
    events,
    fake_graph_helpers,
    make_build_lifecycle,
    ready_project,
)


@pytest.fixture
def daemon(tmp_path, monkeypatch):
    from orchestrator.daemon import server
    db = ready_project(tmp_path, monkeypatch)
    lc = make_build_lifecycle(tmp_path)
    monkeypatch.setattr(server, "_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(server, "_build", lc)
    monkeypatch.setattr(server, "_pipeline", types.SimpleNamespace(task=None, thread_id="pipeline", status="idle"))
    monkeypatch.setattr(server, "_apply_run_env", lambda where: [])
    monkeypatch.setattr(server, "_preflight_or_400", lambda: None)
    yield server, db, lc
    asyncio.run(lc.cleanup())


async def _settle(db, build_id: str, *, until=("parked",) + B.BUILD_TERMINAL, timeout_s: float = 60.0) -> dict:
    """Wait for the build task and its watcher to classify the row."""
    import time
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        row = B.get_build(db, build_id)
        if row and row["status"] in until:
            await asyncio.sleep(0.05)
            return B.get_build(db, build_id)
        await asyncio.sleep(0.05)
    raise AssertionError(f"build {build_id} did not settle: {B.get_build(db, build_id)}")


async def _start(server, module="tiny", **kw):
    resp = await server._start_module_build(server.BuildModuleRequest(module=module, **kw), entry="build_module")
    assert isinstance(resp, dict), getattr(resp, "body", resp)
    return resp


class TestCompletion:
    async def test_real_subgraph_completes_and_commits_the_recorded_build(self, daemon, tmp_path, monkeypatch):
        server, db, lc = daemon
        calls = fake_graph_helpers(monkeypatch, tmp_path, calls={})
        out = await _start(server)
        bid = out["build_id"]
        assert out["started"] and out["thread_id"] == f"build-{bid}" and out["entry"] == "build_module"
        row = await _settle(db, bid)
        assert row["status"] == "completed", row
        # the graph really ran: worker and tool helpers were called in order
        assert calls["generate_rtl"] == 1 and calls["lint_rtl"] >= 1 and calls["run_simulation"] >= 1
        assert calls["synthesize_block"] >= 1 and calls["evaluate_ppa_gate"] >= 1
        # the published pass names the build, and both were committed together
        best = db.result("tiny", "best")
        assert best["done"] and best["build_id"] == bid and best["published_by"] == "graph"
        res = row["result"]
        assert res["rows"] == {"dv": 1, "ppa": 1, "coverage": 1}
        assert res["produced"]["files_sha256"][str(tmp_path / "rtl" / "tiny.v")]
        # measurement rows carry the build id and the synthesis context
        rows = Scoreboard(tmp_path).rows_for_build(bid)
        dv, ppa, cov = rows["dv"][0], rows["ppa"][0], rows["coverage"][0]
        assert dv["passed"] == 1 and dv["skipped"] == 0 and dv["attempt"] == 1 and dv["run_id"] == db.run_id()
        assert ppa["stage"] == "synth" and ppa["tool"] == "yosys+opensta" and ppa["wns_ns"] == pytest.approx(1.25)
        assert ppa["power_mw"] is None and ppa["power_basis"] == "unavailable" and ppa["clock_mhz"] == 50.0
        assert json.loads(cov["uncovered"])["passed"] is True
        # the persistent checkpoint holds the build's thread; graph events carry the id
        con = sqlite3.connect(str(tmp_path / ".coresmith" / "build_checkpoint.db"))
        assert con.execute("SELECT COUNT(*) FROM checkpoints WHERE thread_id=?", (f"build-{bid}",)).fetchone()[0] > 0
        con.close()
        assert any(e.get("build_id") == bid and e.get("node") == "Block Done" for e in events(tmp_path))
        # the stage machine accepts it as a deliverable; the lineage report calls it the intended workflow
        cur = B.current_build_status(db, tmp_path, "tiny")
        assert cur["ok"], cur["reasons"]
        lin = B.lineage(db, tmp_path, "tiny")
        assert lin["modules"]["tiny"]["intended_workflow"] is True
        assert lin["modules"]["tiny"]["builds"][0]["checkpoints"] > 0 and lin["modules"]["tiny"]["builds"][0]["events"] > 0
        # build state reports from the durable record
        state = await server.build_state(bid)
        assert state["status"] == "completed" and state["current"]["ok"] and state["evidence"]["ppa"]

    async def test_persistence_failure_cannot_publish_a_passed_build(self, daemon, tmp_path, monkeypatch):
        from orchestrator.langgraph import pipeline_graph as pg
        server, db, lc = daemon
        fake_graph_helpers(monkeypatch, tmp_path)
        monkeypatch.setattr(pg, "_record_ppa_row", lambda *a, **k: False)   # the required synthesis row never commits
        out = await _start(server)
        row = await _settle(db, out["build_id"])
        assert row["status"] == "failed_persistence"
        assert "ppa_history" in row["error"]
        assert db.result("tiny", "best") is None
        assert db.result("tiny", "dv_best")["sim_passed"] is True       # the DV fact stays; the pass is not published
        assert st.entry(db, tmp_path, "blocks") and B.current_build_status(db, tmp_path, "tiny")["ok"] is False

    async def test_lifecycle_without_build_identity_publishes_nothing(self, tmp_path, monkeypatch):
        from orchestrator.langgraph import pipeline_graph as pg
        db = ready_project(tmp_path, monkeypatch)
        (tmp_path / ".coresmith" / "blocks" / "tiny").mkdir(parents=True, exist_ok=True)
        db.set_result("tiny", "dv_best", {"sim_passed": True})
        out = await pg.block_done_node({"project_root": str(tmp_path), "current_block": {"name": "tiny"}, "attempt": 1,
                                        "sim_passed": True, "synth_success": True, "synth_gate_count": 3,
                                        "timing_required": False, "timing_ok": True, "build_id": ""})
        res = out["completed_blocks"][0]
        assert res["success"] is False and res["persistence_failed"] and "BUILD_IDENTITY_MISSING" in res["error"]
        assert db.result("tiny", "best") is None


class TestParkResumeRestart:
    async def test_park_survives_lifecycle_recreation_and_resumes_by_build_id(self, daemon, tmp_path, monkeypatch):
        from orchestrator.langgraph import pipeline_graph as pg
        server, db, lc = daemon
        fake_graph_helpers(monkeypatch, tmp_path, lint_clean=False)

        async def decide_ask_human(state):
            return {"debug_action": "ask_human"}
        monkeypatch.setattr(pg, "decide_node", decide_ask_human)        # captured at compile: before ensure_graph
        out = await _start(server)
        bid = out["build_id"]
        row = await _settle(db, bid)
        assert row["status"] == "parked", row
        state = await server.build_state(bid)
        assert state["interrupts"] and state["interrupts"][0]["payload"]["type"] == "human_intervention_needed"
        assert "fix_rtl" in state["interrupts"][0]["payload"]["supported_actions"]
        # the park is in the interrupts table under the build graph
        assert any(r["graph"] == "build" and r["status"] == "pending" for r in db.interrupts(status="pending"))
        # a second start of the same module never overwrites the in-flight build
        resp = await server._start_module_build(server.BuildModuleRequest(module="tiny"), entry="build_module")
        assert resp.status_code == 409 and json.loads(resp.body)["error"] == "BUILD_IN_FLIGHT"
        # the process restarts: a fresh lifecycle recovers the thread from the recorded identity
        await lc.cleanup()
        lc2 = make_build_lifecycle(tmp_path)
        monkeypatch.setattr(server, "_build", lc2)
        recovered = await server.MB.recover_builds(db, tmp_path, lc2)
        assert recovered and recovered[0]["build_id"] == bid and recovered[0]["checkpoint"] == "interrupted"
        # an unsupported action is refused
        with pytest.raises(server.HTTPException) as exc:
            await server.build_resume(server.BuildResumeRequest(build_id=bid, action="approve"))
        assert exc.value.status_code == 400
        # the Architect repairs the RTL and resumes; the same thread continues to completion
        monkeypatch.setattr(pg, "lint_rtl", lambda *a, **k: {"clean": True, "warnings": ""})
        res = await server.build_resume(server.BuildResumeRequest(build_id=bid, action="fix_rtl",
                                                                  rtl_fix_description="fixed the syntax"))
        assert res["resumed"] and res["build_id"] == bid
        row = await _settle(db, bid)
        assert row["status"] == "completed", row
        assert db.result("tiny", "best")["build_id"] == bid
        assert lc2.thread_id == f"build-{bid}"
        # one build row, not two: the resume continued the recorded attempt
        assert [b["id"] for b in B.builds_for(db, "tiny")] == [bid]
        await lc2.cleanup()

    async def test_resume_of_a_terminal_build_is_refused(self, daemon, tmp_path, monkeypatch):
        server, db, lc = daemon
        fake_graph_helpers(monkeypatch, tmp_path)
        out = await _start(server)
        await _settle(db, out["build_id"])
        with pytest.raises(server.HTTPException) as exc:
            await server.build_resume(server.BuildResumeRequest(build_id=out["build_id"], action="retry"))
        assert exc.value.status_code == 409

    async def test_failed_block_records_failed_and_archives_the_previous_pass(self, daemon, tmp_path, monkeypatch):
        from orchestrator.langgraph import pipeline_graph as pg
        server, db, lc = daemon
        fake_graph_helpers(monkeypatch, tmp_path)
        first = await _start(server)
        await _settle(db, first["build_id"])
        assert db.result("tiny", "best")["build_id"] == first["build_id"]
        # a second build whose simulation fails and whose diagnosis escalates
        fake_graph_helpers(monkeypatch, tmp_path, sim_pass=False)

        async def decide_escalate(state):
            return {"debug_action": "escalate"}
        await lc.cleanup()
        lc2 = make_build_lifecycle(tmp_path)
        monkeypatch.setattr(server, "_build", lc2)
        monkeypatch.setattr(pg, "decide_node", decide_escalate)
        second = await _start(server)
        row = await _settle(db, second["build_id"])
        assert row["status"] == "failed" and second["build_id"] != first["build_id"]
        assert db.result("tiny", "best") is None
        assert db.result("tiny", "best_superseded")["build_id"] == first["build_id"]   # history, not erasure
        assert B.get_build(db, first["build_id"])["status"] == "completed"
        await lc2.cleanup()


class TestLifecycleSafety:
    async def test_a_row_dispatched_but_never_checkpointed_is_a_start_failure_not_a_block(self, daemon, tmp_path):
        """The process died between the ledger insert and the first
        checkpoint: recovery records a truthful start failure (history kept)
        and the module is buildable again instead of BUILD_IN_FLIGHT forever."""
        server, db, lc = daemon
        plan = server.MB.plan_module_build(db, tmp_path, "tiny", entry="build_module")
        server.MB.dispatch(db, tmp_path, plan)
        bid = plan["build_id"]
        assert B.get_build(db, bid)["status"] == "dispatched"
        resp = await server._start_module_build(server.BuildModuleRequest(module="tiny"), entry="build_module")
        assert resp.status_code == 409 and json.loads(resp.body)["error"] == "BUILD_IN_FLIGHT"
        recovered = await server.MB.recover_builds(db, tmp_path, lc)
        assert recovered == [{"build_id": bid, "thread_id": plan["thread_id"], "checkpoint": "idle", "status": "error"}]
        row = B.get_build(db, bid)
        assert row["status"] == "error" and "never checkpointed" in row["error"]
        assert server.MB.in_flight_refusal(db, "tiny") is None

    async def test_abort_clears_a_parked_build_and_a_running_one_must_be_paused_first(self, daemon, tmp_path, monkeypatch):
        from orchestrator.langgraph import pipeline_graph as pg
        server, db, lc = daemon
        fake_graph_helpers(monkeypatch, tmp_path, lint_clean=False)

        async def decide_ask_human(state):
            return {"debug_action": "ask_human"}
        monkeypatch.setattr(pg, "decide_node", decide_ask_human)
        out = await _start(server)
        bid = out["build_id"]
        assert (await _settle(db, bid))["status"] == "parked"
        resp = await server._start_module_build(server.BuildModuleRequest(module="tiny"), entry="build_module")
        assert json.loads(resp.body)["error"] == "BUILD_IN_FLIGHT"
        res = await server.build_abort(server.BuildAbortRequest(build_id=bid, reason="wrong spec"))
        assert res["aborted"] and B.get_build(db, bid)["status"] == "aborted"
        assert "wrong spec" in B.get_build(db, bid)["error"] and db.result("tiny", "best") is None
        resp = await server.build_abort(server.BuildAbortRequest(build_id=bid))
        assert resp.status_code == 409 and json.loads(resp.body)["error"] == "BUILD_TERMINAL"
        # the module is buildable again; the aborted row is history
        fake_graph_helpers(monkeypatch, tmp_path)
        out2 = await _start(server)
        assert (await _settle(db, out2["build_id"]))["status"] == "completed"
        assert [b["status"] for b in B.builds_for(db, "tiny")] == ["completed", "aborted"]

    async def test_resume_re_validates_the_recorded_inputs(self, daemon, tmp_path, monkeypatch):
        """Editing the SystemC model (or the worker binding) while the build
        is parked: the resume is refused with BUILD_STALE and the reasons;
        old provenance and new work never mix. The state query says so too."""
        from orchestrator.langgraph import pipeline_graph as pg
        server, db, lc = daemon
        fake_graph_helpers(monkeypatch, tmp_path, lint_clean=False)

        async def decide_ask_human(state):
            return {"debug_action": "ask_human"}
        monkeypatch.setattr(pg, "decide_node", decide_ask_human)
        out = await _start(server)
        bid = out["build_id"]
        assert (await _settle(db, bid))["status"] == "parked"
        (tmp_path / "model" / "tiny_model.cpp").write_text("// edited while parked\n")
        state = await server.build_state(bid)
        assert state["resumable"] is False and any(r.startswith("model:") for r in state["stale"])
        resp = await server.build_resume(server.BuildResumeRequest(build_id=bid, action="fix_rtl"))
        assert resp.status_code == 409
        body = json.loads(resp.body)
        assert body["error"] == "BUILD_STALE" and any("model" in b["text"] for b in body["blocked_by"])
        assert B.get_build(db, bid)["status"] == "parked"
        # restoring the inputs makes the park resumable again
        (tmp_path / "model" / "tiny_model.cpp").write_text("// reference model of tiny\n#include \"tiny_model.h\"\nint tiny_model_marker = 1;\n")
        assert (await server.build_state(bid))["resumable"] is True

    async def test_the_pipeline_and_the_backend_never_run_over_a_build(self, daemon, tmp_path, monkeypatch):
        server, db, lc = daemon
        fake_graph_helpers(monkeypatch, tmp_path)

        async def slow_generate_rtl(block, attempt, callbacks=None):
            await asyncio.sleep(0.4)
            p = tmp_path / "rtl" / "tiny.v"
            p.write_text("module tiny(); endmodule\n")
            return {"rtl_path": str(p), "verilog": p.read_text()}
        from orchestrator.langgraph import pipeline_graph as pg
        monkeypatch.setattr(pg, "generate_rtl", slow_generate_rtl)
        out = await _start(server)
        assert lc.task is not None and not lc.task.done()
        for call in (lambda: server.run_start(server.StartRequest(force=True)),
                     lambda: server.run_resume(server.ResumeRequest(action="retry")),
                     lambda: server.run_continue(),
                     lambda: server.backend_start(server.BackendStartRequest())):
            with pytest.raises(server.HTTPException) as exc:
                await call()
            assert exc.value.status_code == 409 and "module build is running" in exc.value.detail
        await _settle(db, out["build_id"])

    async def test_every_launch_runs_in_one_section_and_a_build_elsewhere_refuses_them(self, daemon, tmp_path, monkeypatch):
        """Two launches scheduled concurrently never interleave: the second
        enters the launch section only after the first left it (an await
        barrier inside the first). Across processes the section is the
        ``graph_launch`` lease and a running build is the ``graph:build``
        lease: held by another live process, every launch -- a build, the
        pipeline, the backend -- is refused."""
        import os
        server, db, lc = daemon
        MB = server.MB
        order: list[str] = []
        gate = asyncio.Event()

        async def first():
            async with MB.launch_section(db, what="first"):
                order.append("first-in")
                await gate.wait()                      # an await inside the critical section
                order.append("first-out")

        async def second():
            await asyncio.sleep(0)                     # scheduled after `first` entered
            async with MB.launch_section(db, what="second"):
                order.append("second-in")
        t1, t2 = asyncio.create_task(first()), asyncio.create_task(second())
        await asyncio.sleep(0.05)
        assert order == ["first-in"] and db.lease(MB.LAUNCH_LEASE)["meta"]["what"] == "first"
        gate.set()
        await asyncio.gather(t1, t2)
        assert order == ["first-in", "first-out", "second-in"] and db.lease(MB.LAUNCH_LEASE) is None
        # a build running in ANOTHER live process (its lease names a live pid that is not ours)
        other = os.getppid()
        assert db.acquire_lease(MB.BUILD_LEASE, 60, meta={"build_id": "b-elsewhere"}, pid=other)
        resp = await server._start_module_build(server.BuildModuleRequest(module="tiny"), entry="build_module")
        assert resp.status_code == 409 and json.loads(resp.body)["error"] == "BUILD_RUNNING"
        resp = await server.run_start(server.StartRequest(force=True))
        assert resp.status_code == 409 and "b-elsewhere" in json.loads(resp.body)["message"]
        resp = await server.backend_start(server.BackendStartRequest())   # the daemon's backend route, the same way
        assert resp.status_code == 409 and json.loads(resp.body)["error"] == "BUILD_RUNNING"
        db.steal_lease(MB.BUILD_LEASE, "test")
        # another process in its launch section: the launch waits for nobody, it refuses
        assert db.acquire_lease(MB.LAUNCH_LEASE, 60, meta={"what": "run start"}, pid=other)
        resp = await server._start_module_build(server.BuildModuleRequest(module="tiny"), entry="build_module")
        assert json.loads(resp.body)["error"] == "WORKSPACE_BUSY"
        db.steal_lease(MB.LAUNCH_LEASE, "test")
        # a real build holds the build lease for its duration and the watcher releases it
        fake_graph_helpers(monkeypatch, tmp_path)
        out = await _start(server)
        held = db.lease(MB.BUILD_LEASE)
        assert held and held["meta"]["build_id"] == out["build_id"] and held["holder_pid"] == os.getpid()
        await _settle(db, out["build_id"])
        await asyncio.sleep(0.2)
        assert db.lease(MB.BUILD_LEASE) is None

    async def test_a_route_delegating_to_the_mcp_launch_is_one_operation(self, daemon, tmp_path, monkeypatch):
        """The daemon's backend routes enter the launch section and delegate
        to the MCP implementation, which enters it too: the same task re-enters
        its own section (no deadlock, the inner launch runs once), while a
        task spawned from inside is not the owner and waits."""
        import types as _types
        from unittest.mock import AsyncMock

        import orchestrator.mcp_server as mcp
        server, db, lc = daemon
        MB = server.MB
        monkeypatch.setattr(mcp, "_project_root", lambda: str(tmp_path))
        monkeypatch.setattr(server, "_backend_preflight", lambda: {"ok": True})
        monkeypatch.setattr(server, "_backend_park_meta", AsyncMock(return_value=[]))
        called: list[str] = []

        async def inner_start(**kwargs):
            called.append("start")
            return {"ok": True}

        async def inner_resume(**kwargs):
            called.append("resume")
            return json.dumps({"ok": True})
        handle = _types.SimpleNamespace(
            launch_backend=mcp._serialized_tool("launch_backend", as_dict=True)(inner_start),
            resume_backend=mcp._serialized_tool("resume_backend")(inner_resume),
            _backend=_types.SimpleNamespace(task=None, thread_id="backend", name="backend"))
        monkeypatch.setattr(server, "_backend_handle", lambda: handle)
        res = await asyncio.wait_for(server.backend_start(server.BackendStartRequest()), timeout=5)
        assert res == {"ok": True} and called == ["start"]
        res = await asyncio.wait_for(server.backend_resume(server.BackendResumeRequest(action="retry")), timeout=5)
        assert res == {"ok": True} and called == ["start", "resume"]
        assert db.lease(MB.LAUNCH_LEASE) is None          # the section was left once, by its owner
        # a task spawned from INSIDE the section is not the owner: it waits for the exit
        order: list[str] = []
        inner_done = asyncio.Event()

        async def spawned():
            async with MB.launch_section(db, what="spawned"):
                order.append("spawned-in")
            inner_done.set()

        async def outer():
            async with MB.launch_section(db, what="outer"):
                order.append("outer-in")
                t = asyncio.create_task(spawned())
                await asyncio.sleep(0.05)
                order.append("outer-out")
            await t
        await asyncio.wait_for(outer(), timeout=5)
        assert order == ["outer-in", "outer-out", "spawned-in"] and inner_done.is_set()

    async def test_a_pipeline_or_backend_running_elsewhere_refuses_a_build_and_vice_versa(self, daemon, tmp_path, monkeypatch):
        import os
        server, db, lc = daemon
        MB = server.MB
        other = os.getppid()
        for graph, code in (("pipeline", "PIPELINE_RUNNING"), ("backend", "BACKEND_RUNNING")):
            assert db.acquire_lease(MB.GRAPH_LEASES[graph], 60, meta={"thread_id": "t"}, pid=other)
            resp = await server._start_module_build(server.BuildModuleRequest(module="tiny"), entry="build_module")
            assert resp.status_code == 409 and json.loads(resp.body)["error"] == code
            db.steal_lease(MB.GRAPH_LEASES[graph], "test")
        # a started graph holds its lease in THIS process until its task ends
        done = asyncio.Event()

        async def run_a_while():
            await done.wait()
        fake = type("L", (), {})()
        fake.task, fake.thread_id, fake.name = asyncio.create_task(run_a_while()), "pipeline", "pipeline"
        assert MB.hold_graph_lease(db, fake, "pipeline")
        held = db.lease(MB.GRAPH_LEASES["pipeline"])
        assert held["holder_pid"] == os.getpid() and held["meta"]["thread_id"] == "pipeline"
        assert MB.foreign_graph_refusal(db) is None                  # our own process: not foreign
        done.set()
        await asyncio.sleep(0.1)
        assert db.lease(MB.GRAPH_LEASES["pipeline"]) is None         # released when the task ended

    async def test_a_retired_task_never_releases_its_replacements_lease(self, daemon):
        """The lease is bound to the exact task and token it was taken for:
        when the old task's completion installs a replacement task (and its
        own lease) before the old cleanup runs, the old cleanup releases only
        its own token and leaves the lifecycle's record alone; the
        replacement's lease goes when the replacement ends."""
        server, db, lc = daemon
        MB = server.MB
        name = MB.GRAPH_LEASES["pipeline"]
        first, second = asyncio.Event(), asyncio.Event()
        fake = types.SimpleNamespace(thread_id="old", task=asyncio.create_task(first.wait()))
        replacement: dict = {}

        def replace(_):
            fake.thread_id, fake.task = "new", asyncio.create_task(second.wait())
            replacement["token"] = MB.hold_graph_lease(db, fake, "pipeline")
        fake.task.add_done_callback(replace)
        old_token = MB.hold_graph_lease(db, fake, "pipeline")
        assert old_token and db.lease(name)["token"] == old_token
        # the same task asking again reuses its ownership: no second token
        assert MB.hold_graph_lease(db, fake, "pipeline") == old_token
        await asyncio.sleep(0)
        first.set()
        for _ in range(8):
            await asyncio.sleep(0)
        row = db.lease(name)
        assert replacement["token"] and replacement["token"] != old_token
        assert not fake.task.done() and row and row["token"] == replacement["token"] and row["meta"]["thread_id"] == "new"
        assert fake._graph_lease["token"] == replacement["token"] and fake._graph_lease["task"] is fake.task
        second.set()
        await fake.task
        for _ in range(4):
            await asyncio.sleep(0)
        assert db.lease(name) is None and fake._graph_lease is None

    async def test_nested_launch_wrappers_own_one_lease_and_a_lease_failure_refuses_the_launch(self, daemon, tmp_path, monkeypatch):
        """The daemon's backend route and the MCP launch it delegates to own
        ONE ``graph:backend`` lease (taken before the launch, bound to the
        started task, released when it ends); a lease that cannot be taken
        refuses the launch before anything starts, and a launch that started
        nothing leaves no lease behind."""
        import types as _types
        from unittest.mock import AsyncMock

        import orchestrator.mcp_server as mcp
        server, db, lc = daemon
        MB = server.MB
        name = MB.GRAPH_LEASES["backend"]
        monkeypatch.setattr(mcp, "_project_root", lambda: str(tmp_path))
        monkeypatch.setattr(server, "_backend_preflight", lambda: {"ok": True})
        monkeypatch.setattr(server, "_backend_park_meta", AsyncMock(return_value=[]))
        backend = _types.SimpleNamespace(task=None, thread_id="backend", name="backend", status="idle")
        monkeypatch.setattr(mcp, "_backend", backend)
        ended = asyncio.Event()
        acquired: list[str] = []
        real_acquire = db.acquire_lease

        def counting_acquire(lease_name, ttl, **kw):
            acquired.append(lease_name)
            return real_acquire(lease_name, ttl, **kw)
        monkeypatch.setattr(db, "acquire_lease", counting_acquire)
        monkeypatch.setattr(server, "_project_db", lambda: db)

        async def inner_start(**kwargs):
            assert db.lease(name)["meta"]["what"] == "backend start"     # owned BEFORE the launch, by the route
            backend.task = asyncio.create_task(ended.wait())
            return {"ok": True}
        handle = _types.SimpleNamespace(launch_backend=mcp._serialized_tool("launch_backend", as_dict=True, graph="backend")(inner_start),
                                        _backend=backend)
        monkeypatch.setattr(server, "_backend_handle", lambda: handle)
        res = await asyncio.wait_for(server.backend_start(server.BackendStartRequest()), timeout=5)
        assert res == {"ok": True}
        assert acquired.count(name) == 1                                  # the nested wrapper reused it
        held = db.lease(name)
        assert held and held["token"] == backend._graph_lease["token"] and backend._graph_lease["task"] is backend.task
        assert db.lease(MB.LAUNCH_LEASE) is None
        ended.set()
        await asyncio.sleep(0.1)
        assert db.lease(name) is None and backend._graph_lease is None     # released with the task
        # a launch that started nothing holds nothing afterwards
        backend.task = None
        handle.launch_backend = mcp._serialized_tool("launch_backend", as_dict=True, graph="backend")(AsyncMock(return_value={"ok": False}))
        res = await asyncio.wait_for(server.backend_start(server.BackendStartRequest()), timeout=5)
        assert res == {"ok": False} and db.lease(name) is None
        # the lease cannot be taken: refused, nothing launched, nothing left behind
        launched = AsyncMock(return_value={"ok": True})
        handle.launch_backend = mcp._serialized_tool("launch_backend", as_dict=True, graph="backend")(launched)

        def refusing_acquire(lease_name, ttl, **kw):
            return None if lease_name == name else real_acquire(lease_name, ttl, **kw)
        monkeypatch.setattr(db, "acquire_lease", refusing_acquire)
        resp = await asyncio.wait_for(server.backend_start(server.BackendStartRequest()), timeout=5)
        assert resp.status_code == 500 and json.loads(resp.body)["error"] == "GRAPH_LEASE_UNAVAILABLE"
        assert launched.await_count == 0 and db.lease(MB.LAUNCH_LEASE) is None and db.lease(name) is None

    async def test_a_build_of_another_module_waits_for_the_running_one(self, daemon, tmp_path, monkeypatch):
        server, db, lc = daemon
        fake_graph_helpers(monkeypatch, tmp_path)
        from orchestrator.langgraph import pipeline_graph as pg

        async def slow_generate_rtl(block, attempt, callbacks=None):
            await asyncio.sleep(0.4)
            p = tmp_path / "rtl" / f"{block['name']}.v"
            p.write_text(f"module {block['name']}(); endmodule\n")
            return {"rtl_path": str(p), "verilog": p.read_text()}
        monkeypatch.setattr(pg, "generate_rtl", slow_generate_rtl)
        out = await _start(server)
        resp = await server._start_module_build(server.BuildModuleRequest(module="sink"), entry="build_module")
        assert resp.status_code == 409 and json.loads(resp.body)["error"] == "BUILD_RUNNING"
        await _settle(db, out["build_id"])
        assert B.builds_for(db, "sink") == []


class TestPipelineFanout:
    async def test_init_block_allocates_the_same_identity_for_the_pipeline(self, tmp_path, monkeypatch):
        from orchestrator.langgraph import pipeline_graph as pg
        db = ready_project(tmp_path, monkeypatch, stage="blocks")
        fake_graph_helpers(monkeypatch, tmp_path)
        out = await pg.init_block_node({"project_root": str(tmp_path), "current_block": {"name": "tiny", "tier": 1},
                                        "target_clock_mhz": 50.0, "max_attempts": 2})
        bid = out["build_id"]
        row = B.get_build(db, bid)
        assert row["entry"] == "run_start" and row["graph"] == "pipeline" and row["status"] == "running"
        assert row["inputs"]["module"] == "tiny" and row["inputs"]["worker"]["explicit"]
        # a dispatched identity is kept, not replaced
        out2 = await pg.init_block_node({"project_root": str(tmp_path), "current_block": {"name": "tiny", "tier": 1},
                                         "target_clock_mhz": 50.0, "max_attempts": 2, "build_id": bid})
        assert out2["build_id"] == bid and len(B.builds_for(db, "tiny")) == 1
        # a replay of the init node for the same task (crash before its checkpoint) reuses the row
        out3 = await pg.init_block_node({"project_root": str(tmp_path), "current_block": {"name": "tiny", "tier": 1},
                                         "target_clock_mhz": 50.0, "max_attempts": 2})
        assert out3["build_id"] == bid and len(B.builds_for(db, "tiny")) == 1

    async def test_pipeline_start_requires_blocks_stage_and_everything_ready(self, daemon, tmp_path):
        server, db, lc = daemon
        resp = await server.run_start(server.StartRequest(force=True))
        assert resp.status_code == 409 and json.loads(resp.body)["error"] == "STAGE_BEFORE_BLOCKS"
        assert st.advance(db, tmp_path)["advanced"]
        assert server.MB.pipeline_start_refusal(db, tmp_path) is None

    async def test_run_start_reuses_current_builds_instead_of_regenerating(self, daemon, tmp_path, monkeypatch):
        """A module already built through ``build module`` keeps its current
        recorded build when the pipeline fans out: its receipt is appended
        and it is not sent through the subgraph again."""
        from orchestrator.langgraph import pipeline_graph as pg
        from orchestrator.tests.build_fixtures import complete_build
        server, db, lc = daemon
        assert st.advance(db, tmp_path)["advanced"]
        (tmp_path / "rtl" / "tiny.v").write_text("module tiny(); endmodule\n")
        bid = complete_build(db, tmp_path, "tiny")
        queue = [{"name": "tiny", "tier": 1}, {"name": "sink", "tier": 1}]
        state = {"project_root": str(tmp_path), "block_queue": queue, "tier_list": [], "current_tier_index": 0,
                 "completed_blocks": [], "target_clock_mhz": 50.0, "max_attempts": 2}
        out = await pg.init_tier_node(dict(state))
        assert out["reuse_blocks"] == ["tiny"]
        assert out["completed_blocks"][0]["build_id"] == bid and out["completed_blocks"][0]["reused_build"] is True
        sends = pg.fan_out_tier({**state, **out, "tier_list": [1]})
        assert [s.arg["current_block"]["name"] for s in sends] == ["sink"]
        # a tier whose every block is reused advances on its receipts
        sends = pg.fan_out_tier({**state, **out, "tier_list": [1], "block_queue": queue[:1]})
        assert sends == ["shell_update"]
        # a block the targeted plan names re-enters regardless of its current build
        out2 = await pg.init_tier_node({**state, "revise_blocks": {"tiny": True}})
        assert not out2.get("reuse_blocks")
        # a stale build is not reused
        (tmp_path / "rtl" / "tiny.v").write_text("module tiny(); // edited\nendmodule\n")
        out3 = await pg.init_tier_node(dict(state))
        assert not out3.get("reuse_blocks") and not out3.get("completed_blocks")

    async def test_pipeline_complete_counts_only_entries_with_a_current_build_receipt(self, tmp_path, monkeypatch):
        from orchestrator.langgraph import pipeline_graph as pg
        from orchestrator.tests.build_fixtures import complete_build
        db = ready_project(tmp_path, monkeypatch, stage="blocks", with_rtl=True)
        monkeypatch.setattr(pg, "write_graph_event", lambda *a, **k: None)
        monkeypatch.setattr(pg, "interrupt", lambda payload: {"action": "abort"})
        monkeypatch.setenv("CORESMITH_REVALIDATE_INCOMPLETE", "0")
        queue = [{"name": "tiny", "tier": 1}, {"name": "sink", "tier": 1}]
        bid = complete_build(db, tmp_path, "tiny")
        state = {"project_root": str(tmp_path), "block_queue": queue, "completed_blocks": [
            {"name": "tiny", "success": True, "build_id": bid},
            {"name": "sink", "success": True, "marked_manually": True},        # the old manual override shape
        ]}
        out = await pg.pipeline_complete_node(dict(state))
        assert out["pipeline_aborted"] is True and out.get("frontend_complete") is not True
        # a receipt for every block: the frontend completes
        complete_build(db, tmp_path, "sink")
        state["completed_blocks"][1] = {"name": "sink", "success": True, "build_id": B.latest_build(db, "sink")["id"]}
        out = await pg.pipeline_complete_node(dict(state))
        assert out["frontend_complete"] is True
        # an entry naming a build that is no longer the published pass does not count
        other = complete_build(db, tmp_path, "tiny")
        out = await pg.pipeline_complete_node(dict(state))                      # tiny's entry still names the old build
        assert out["pipeline_aborted"] is True
        state["completed_blocks"][0]["build_id"] = other
        assert (await pg.pipeline_complete_node(dict(state)))["frontend_complete"] is True

    async def test_run_start_force_aborts_the_discarded_runs_in_flight_builds(self, daemon, tmp_path, monkeypatch):
        server, db, lc = daemon
        inputs = B.module_inputs(db, tmp_path, "tiny")
        B.record_dispatch(db, build_id="b-p", module="tiny", entry="run_start", graph="pipeline", thread_id="pipeline",
                          inputs=inputs, checkpoint_ns="process_block:1")
        B.mark_started(db, "b-p")
        assert server.MB.in_flight_refusal(db, "tiny").code == "BUILD_IN_FLIGHT"
        assert server.MB.abort_pipeline_builds(db, reason="replaced") == ["b-p"]
        assert B.get_build(db, "b-p")["status"] == "aborted" and server.MB.in_flight_refusal(db, "tiny") is None
