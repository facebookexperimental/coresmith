# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""C1-2: the chip-lead ledger, the trip flag, the daemon record and the sim
lock live in the project database; the files are views."""
from __future__ import annotations

import asyncio
import json
import os
import stat

import pytest

from orchestrator.state_store.project_db import open_project


class _Lead:
    def __init__(self, action="retry"):
        self._a = action

    async def decide(self, payload, prior_decisions):
        await asyncio.sleep(0.01)
        return {"action": self._a, "reasoning": "because"}


def _arm(monkeypatch, tmp_path, action="retry"):
    from orchestrator.langgraph import pipeline_graph as pg
    import orchestrator.langchain.agents.chip_lead_agent as cla
    monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("CORESMITH_ENABLE_CHIP_LEAD", "1")
    monkeypatch.setattr(pg, "_CHIP_LEAD_TRIPPED", False)
    monkeypatch.setattr(pg, "_engine_checkout_guard", lambda: [])
    monkeypatch.setattr(cla, "ChipLeadAgent", lambda: _Lead(action))
    db = open_project(tmp_path)
    db.begin_run("r1")
    return pg, db


class TestDecisionsLedger:
    def test_concurrent_branches_get_unique_indices_and_a_view(self, tmp_path, monkeypatch):
        pg, db = _arm(monkeypatch, tmp_path)
        payloads = [{"type": "dv_failure", "block_name": f"b{i}",
                     "supported_actions": ["retry", "skip"]} for i in range(4)]

        async def go():
            return await asyncio.gather(*(pg._resolve_interrupt(p) for p in payloads))
        out = asyncio.run(go())
        assert all(d["action"] == "retry" for d in out)
        rows = db.decisions()
        assert [d["decision_index"] for d in rows] == [1, 2, 3, 4]
        assert sorted(d["block"] for d in rows) == ["b0", "b1", "b2", "b3"]
        view = tmp_path / ".coresmith" / "chip_lead" / "decisions.jsonl"
        lines = [json.loads(ln) for ln in view.read_text().splitlines()]
        assert [ln["decision_index"] for ln in lines] == [1, 2, 3, 4]
        assert not (view.stat().st_mode & stat.S_IWUSR)  # read-only view
        assert db.leases() == []  # the ledger lease was released

    def test_budget_trips_into_a_run_flag(self, tmp_path, monkeypatch):
        pg, db = _arm(monkeypatch, tmp_path)
        monkeypatch.setenv("CORESMITH_CHIP_LEAD_MAX_DECISIONS", "1")
        parked = []
        monkeypatch.setattr(pg, "interrupt", lambda payload: parked.append(payload) or {"action": "parked"})
        p = {"type": "dv_failure", "block_name": "b", "supported_actions": ["retry"]}
        assert asyncio.run(pg._resolve_interrupt(p))["action"] == "retry"
        assert asyncio.run(pg._resolve_interrupt(p))["action"] == "parked"
        assert db.get_flag("chip_lead_tripped")["tripped"] is True
        assert pg._CHIP_LEAD_TRIPPED is True

    def test_trip_survives_a_process_restart(self, tmp_path, monkeypatch):
        pg, db = _arm(monkeypatch, tmp_path)
        db.set_flag("chip_lead_tripped", {"tripped": True})
        assert pg._CHIP_LEAD_TRIPPED is False   # the process cache is cold...
        assert pg._chip_lead_tripped() is True  # ...but the run flag remembers
        parked = []
        monkeypatch.setattr(pg, "interrupt", lambda payload: parked.append(payload) or {})
        asyncio.run(pg._resolve_interrupt({"type": "x", "supported_actions": ["retry"]}))
        assert parked and db.decision_count() == 0
        pg._untrip_chip_lead()
        assert pg._chip_lead_tripped() is False and db.get_flag("chip_lead_tripped") is None

    def test_new_run_resets_the_ledger_index(self, tmp_path, monkeypatch):
        pg, db = _arm(monkeypatch, tmp_path)
        asyncio.run(pg._resolve_interrupt({"type": "x", "supported_actions": ["retry"]}))
        db.begin_run("r2")
        asyncio.run(pg._resolve_interrupt({"type": "x", "supported_actions": ["retry"]}))
        assert db.decisions(run_id="r1")[0]["decision_index"] == 1
        assert db.decisions(run_id="r2")[0]["decision_index"] == 1


class TestDaemonLease:
    def test_daemon_file_is_a_view_of_the_lease(self, tmp_path, monkeypatch):
        from orchestrator.daemon import server as ds
        monkeypatch.setattr(ds, "_PROJECT_ROOT", str(tmp_path))
        monkeypatch.setattr(ds, "_daemon_lease_token", None)
        ds._write_daemon_file(4242)
        db = open_project(tmp_path)
        lease = db.lease("daemon")
        assert lease["holder_pid"] == os.getpid() and lease["meta"]["port"] == 4242
        info = json.loads((tmp_path / ".coresmith" / "daemon.json").read_text())
        assert info["port"] == 4242 and info["pid"] == os.getpid()
        ds._remove_daemon_file()
        assert db.lease("daemon") is None
        assert not (tmp_path / ".coresmith" / "daemon.json").exists()

    def test_a_live_foreign_daemon_refuses_the_start(self, tmp_path, monkeypatch):
        import subprocess, sys
        from orchestrator.daemon import server as ds
        monkeypatch.setattr(ds, "_PROJECT_ROOT", str(tmp_path))
        monkeypatch.setattr(ds, "_daemon_lease_token", None)
        db = open_project(tmp_path)
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            db.acquire_lease("daemon", 300, meta={"port": 1}, pid=p.pid)
            with pytest.raises(SystemExit):
                ds._write_daemon_file(4243)
        finally:
            p.kill(); p.wait()
        # ...but a dead holder is displaced
        ds._write_daemon_file(4243)
        assert db.lease("daemon")["holder_pid"] == os.getpid()
        ds._remove_daemon_file()

    def test_lifecycle_sees_a_foreign_daemon_through_the_lease(self, tmp_path):
        import subprocess, sys
        from orchestrator.graph_lifecycle import GraphLifecycle
        gl = GraphLifecycle.__new__(GraphLifecycle)
        gl.project_root = str(tmp_path)
        db = open_project(tmp_path)
        assert gl._foreign_live_daemon_owns_project() is False
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            db.acquire_lease("daemon", 300, pid=p.pid)
            assert gl._foreign_live_daemon_owns_project() is True
        finally:
            p.kill(); p.wait()
        assert gl._foreign_live_daemon_owns_project() is False  # holder dead
        tok = db.acquire_lease("daemon", 300)  # ourselves
        assert gl._foreign_live_daemon_owns_project() is False
        db.release_lease("daemon", tok)


class TestSimLease:
    def test_verify_chip_holds_the_scope_lease_and_releases_it(self, tmp_path, monkeypatch):
        from orchestrator.harness import verify as V
        db = open_project(tmp_path)
        (tmp_path / ".coresmith").mkdir(exist_ok=True)
        (tmp_path / ".coresmith" / "integration_result.json").write_text(json.dumps({
            "design_name": "chip", "top_rtl_path": "rtl/top.v",
            "block_rtl_paths": [], "tb_path": "tb/test_chip.py"}))
        (tmp_path / "rtl").mkdir(); (tmp_path / "rtl" / "top.v").write_text("module chip; endmodule\n")
        (tmp_path / "tb").mkdir(); (tmp_path / "tb" / "test_chip.py").write_text("# tb\n")
        seen = {}

        def fake_sim(design, top, blocks, tb, attempt, sim_scope=None, project_root=None):
            seen["lease"] = db.lease(f"sim:{sim_scope}")
            return {"passed": True}
        import orchestrator.langgraph.integration_helpers as ih
        monkeypatch.setattr(ih, "run_integration_simulation", fake_sim)
        res = V.verify_chip(tmp_path)
        assert res.passed, res.verdict
        assert seen["lease"] is not None and seen["lease"]["holder_pid"] == os.getpid()
        assert db.leases() == []
