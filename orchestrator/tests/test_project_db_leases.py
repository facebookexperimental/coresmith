# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Leases, run flags and decisions live in the project database (C1)."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time

import pytest

from orchestrator.state_store.leases import LeaseUnavailable, db_lease
from orchestrator.state_store.project_db import open_project


class TestLeasePrimitives:
    def test_acquire_renew_release(self, tmp_path):
        db = open_project(tmp_path)
        tok = db.acquire_lease("sim", 30, meta={"block": "core"})
        assert tok
        assert db.acquire_lease("sim", 30) is None          # held, live
        assert db.renew_lease("sim", tok, 30) is True
        assert db.renew_lease("sim", "wrong-token", 30) is False
        assert db.release_lease("sim", "wrong-token") is False
        assert db.release_lease("sim", tok) is True
        assert db.leases() == []

    def test_live_lease_is_not_reentrant_for_the_same_pid(self, tmp_path):
        db = open_project(tmp_path)
        tok = db.acquire_lease("x", 30)
        assert db.acquire_lease("x", 30) is None
        assert db.renew_lease("x", tok, 30)  # the first token is still the holder

    def test_expired_lease_is_stolen_and_recorded(self, tmp_path):
        db = open_project(tmp_path)
        tok = db.acquire_lease("x", 0.05)
        time.sleep(0.1)
        tok2 = db.acquire_lease("x", 30)
        assert tok2 and tok2 != tok
        row = db.lease("x")
        assert row["stolen_from"]["reason"] == "expired"
        assert row["stolen_from"]["token"] == tok
        assert db.release_lease("x", tok) is False  # the old holder lost it

    def test_dead_holder_on_this_host_is_stolen(self, tmp_path):
        db = open_project(tmp_path)
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        tok = db.acquire_lease("x", 300, pid=p.pid)
        assert db.acquire_lease("x", 300) is None
        p.kill(); p.wait()
        tok2 = db.acquire_lease("x", 300)
        assert tok2 and tok2 != tok
        assert db.lease("x")["stolen_from"]["reason"] == "holder_dead"

    def test_steal_if_expired_false_waits(self, tmp_path):
        db = open_project(tmp_path)
        db.acquire_lease("x", 0.05)
        time.sleep(0.1)
        assert db.acquire_lease("x", 30, steal_if_expired=False) is None

    def test_operator_steal(self, tmp_path):
        db = open_project(tmp_path)
        db.acquire_lease("x", 300, meta={"block": "b"})
        old = db.steal_lease("x", "stuck daemon")
        assert old["meta"]["block"] == "b" and old["stolen_reason"] == "stuck daemon"
        assert db.steal_lease("x", "again") is None


class TestDbLease:
    def test_context_manager_holds_heartbeats_and_releases(self, tmp_path):
        db = open_project(tmp_path)
        with db_lease(db, "sim", ttl_s=0.6, meta={"scope": "integration"}) as tok:
            assert db.lease("sim")["token"] == tok
            time.sleep(0.9)  # longer than the ttl: the heartbeat kept it alive
            assert db.lease("sim") is not None and not db.lease("sim")["expired"]
            assert db.acquire_lease("sim", 1) is None
        assert db.lease("sim") is None

    def test_contention_serializes_two_threads(self, tmp_path):
        db = open_project(tmp_path)
        order: list[str] = []

        def worker(tag):
            with db_lease(db, "sim", ttl_s=5, wait_s=5, poll_s=0.02):
                order.append(f"{tag}-in")
                time.sleep(0.2)
                order.append(f"{tag}-out")

        t1 = threading.Thread(target=worker, args=("a",))
        t2 = threading.Thread(target=worker, args=("b",))
        t1.start(); time.sleep(0.05); t2.start(); t1.join(); t2.join()
        assert order == ["a-in", "a-out", "b-in", "b-out"]

    def test_unavailable_raises_after_wait(self, tmp_path):
        db = open_project(tmp_path)
        db.acquire_lease("sim", 300)
        t0 = time.time()
        with pytest.raises(LeaseUnavailable):
            with db_lease(db, "sim", ttl_s=5, wait_s=0.3, poll_s=0.05):
                pass
        assert time.time() - t0 >= 0.3

    def test_cross_host_holder_is_only_displaced_by_expiry(self, tmp_path, monkeypatch):
        db = open_project(tmp_path)
        import orchestrator.state_store.leases as L
        monkeypatch.setattr(L, "hostname", lambda: "other-host")
        db.acquire_lease("x", 300, pid=999999)   # a pid that is dead HERE
        monkeypatch.setattr(L, "hostname", lambda: "this-host")
        assert db.acquire_lease("x", 300) is None  # not ours to pid-check


class TestRunFlagsAndDecisions:
    def test_flags_are_scoped_by_run_id(self, tmp_path):
        db = open_project(tmp_path)
        db.set_setting("run_id", "run-1")
        db.set_flag("chip_lead_tripped", True)
        assert db.get_flag("chip_lead_tripped") is True
        db.set_setting("run_id", "run-2")
        assert db.get_flag("chip_lead_tripped", False) is False
        assert db.get_flag("chip_lead_tripped", run_id="run-1") is True
        db.clear_flag("chip_lead_tripped", run_id="run-1")
        assert db.get_flag("chip_lead_tripped", run_id="run-1") is None

    def test_decisions_index_per_run(self, tmp_path):
        db = open_project(tmp_path)
        db.set_setting("run_id", "r1")
        assert db.add_decision(action="retry", interrupt_type="x", block="b") == 1
        assert db.add_decision(action="skip") == 2
        db.set_setting("run_id", "r2")
        assert db.decision_count() == 0
        assert db.add_decision(action="retry") == 1
        assert [d["action"] for d in db.decisions(run_id="r1")] == ["retry", "skip"]
        assert db.decisions(run_id="r1", last=1)[0]["action"] == "skip"

    def test_concurrent_decisions_get_unique_indices(self, tmp_path):
        db = open_project(tmp_path)
        seen: list[int] = []

        def w():
            seen.append(db.add_decision(action="retry"))
        ts = [threading.Thread(target=w) for _ in range(8)]
        [t.start() for t in ts]; [t.join() for t in ts]
        assert sorted(seen) == list(range(1, 9))


class TestCli:
    def test_leases_command_lists_and_steals(self, tmp_path):
        import argparse
        from orchestrator.harness import cli
        db = open_project(tmp_path)
        db.acquire_lease("daemon", 300, meta={"port": 1})
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers()
        cli.register_subcommands(sub)
        args = parser.parse_args(["leases", "--project-root", str(tmp_path), "--json"])
        assert cli.cmd_leases(args) == 0
        args = parser.parse_args(["leases", "--project-root", str(tmp_path),
                                  "--steal", "daemon", "--reason", "test"])
        assert cli.cmd_leases(args) == 0
        assert db.leases() == []
        args = parser.parse_args(["leases", "--project-root", str(tmp_path), "--steal", "x"])
        assert cli.cmd_leases(args) != 0  # --steal needs --reason
