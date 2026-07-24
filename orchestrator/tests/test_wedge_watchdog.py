# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Wedge watchdog (dv-hardening-17): in-daemon self-heal for the runner-task
wedge observed 3x live on armD -- graph.ainvoke parks forever AFTER the node's
work completed and checkpointed (py-spy: loop idle, no threads working, no
exception), leaving in-memory status 'running' until a manual daemon restart.

The watchdog detects the signature (running + stale events file + no live
children) and performs the restart-equivalent in-process: cancel the zombie
runner, resume from the checkpoint.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

from orchestrator.graph_lifecycle import GraphLifecycle


def _handle(tmp_path: Path) -> GraphLifecycle:
    h = GraphLifecycle(
        name="testgraph",
        checkpoint_db=str(tmp_path / "cp.db"),
        builder_fn_path="x",
        builder_fn_name="y",
        project_root=str(tmp_path),
    )
    return h


def _events(tmp_path: Path, age_s: float) -> None:
    ev = tmp_path / ".coresmith" / "pipeline_events.jsonl"
    ev.parent.mkdir(parents=True, exist_ok=True)
    ev.write_text("{}\n")
    old = time.time() - age_s
    os.utime(ev, (old, old))


class TestWedgeSignature:
    def test_not_suspected_when_not_running(self, tmp_path):
        h = _handle(tmp_path)
        h.status = "interrupted"
        assert h._wedge_suspected() is False

    async def test_not_suspected_with_fresh_events(self, tmp_path, monkeypatch):
        h = _handle(tmp_path)
        h.status = "running"
        h.task = asyncio.create_task(asyncio.sleep(30))
        _events(tmp_path, age_s=10)
        monkeypatch.setattr(GraphLifecycle, "_daemon_has_children",
                            staticmethod(lambda: False))
        try:
            assert h._wedge_suspected() is False
        finally:
            h.task.cancel()

    async def test_not_suspected_with_live_children(self, tmp_path, monkeypatch):
        h = _handle(tmp_path)
        h.status = "running"
        h.task = asyncio.create_task(asyncio.sleep(30))
        _events(tmp_path, age_s=5000)
        monkeypatch.setattr(GraphLifecycle, "_daemon_has_children",
                            staticmethod(lambda: True))
        try:
            assert h._wedge_suspected() is False
        finally:
            h.task.cancel()

    async def test_suspected_on_full_signature(self, tmp_path, monkeypatch):
        h = _handle(tmp_path)
        h.status = "running"
        h.task = asyncio.create_task(asyncio.sleep(30))
        _events(tmp_path, age_s=5000)
        monkeypatch.setattr(GraphLifecycle, "_daemon_has_children",
                            staticmethod(lambda: False))
        try:
            assert h._wedge_suspected() is True
        finally:
            h.task.cancel()

    async def test_threshold_env(self, tmp_path, monkeypatch):
        h = _handle(tmp_path)
        h.status = "running"
        h.task = asyncio.create_task(asyncio.sleep(30))
        _events(tmp_path, age_s=100)
        monkeypatch.setattr(GraphLifecycle, "_daemon_has_children",
                            staticmethod(lambda: False))
        monkeypatch.setenv("CORESMITH_WEDGE_TIMEOUT_S", "50")
        try:
            assert h._wedge_suspected() is True
        finally:
            h.task.cancel()


class TestHeal:
    async def test_heal_cancels_and_resumes(self, tmp_path, monkeypatch):
        h = _handle(tmp_path)
        h.status = "running"
        _events(tmp_path, age_s=5000)
        monkeypatch.setattr(GraphLifecycle, "_daemon_has_children",
                            staticmethod(lambda: False))

        resumed = []

        async def fake_run_task(resume_input, config):
            resumed.append((resume_input, config))
            h.status = "interrupted"

        monkeypatch.setattr(h, "run_task", fake_run_task)
        h.task = asyncio.create_task(asyncio.sleep(600))  # the zombie
        h._last_config = {"configurable": {"thread_id": "testgraph"}}

        action = await h._watchdog_heal_once()
        assert action == "healed"
        await asyncio.sleep(0.05)  # let the new task run
        assert resumed == [(None, {"configurable": {"thread_id": "testgraph"}})]
        assert h.status == "interrupted"

    async def test_heal_limit_gives_up(self, tmp_path, monkeypatch):
        h = _handle(tmp_path)
        h.status = "running"
        _events(tmp_path, age_s=5000)
        monkeypatch.setattr(GraphLifecycle, "_daemon_has_children",
                            staticmethod(lambda: False))
        h._watchdog_heals = 3  # already at the limit
        h.task = asyncio.create_task(asyncio.sleep(600))
        action = await h._watchdog_heal_once()
        assert action == "gave_up"
        assert h.status == "error"
        assert "wedge watchdog" in h.error_message

    async def test_noop_when_healthy(self, tmp_path, monkeypatch):
        h = _handle(tmp_path)
        h.status = "interrupted"  # parked normally
        action = await h._watchdog_heal_once()
        assert action == "noop"

    def test_kill_switch(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_WEDGE_WATCHDOG", "0")
        h = _handle(tmp_path)
        h._arm_watchdog()
        assert h._watchdog is None
