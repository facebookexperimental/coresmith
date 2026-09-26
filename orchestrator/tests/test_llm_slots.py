# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""C1-4: CORESMITH_LLM_SLOTS caps concurrent CLI workers with DB leases."""
from __future__ import annotations

import threading
import time

from orchestrator.langchain.agents import coresmith_llm as cl
from orchestrator.state_store.project_db import open_project


def _llm(timeout=5.0):
    obj = cl.ClaudeLLM.__new__(cl.ClaudeLLM)
    obj.timeout = timeout
    return obj


def test_unset_means_unlimited_and_unslotted(tmp_path, monkeypatch):
    monkeypatch.delenv("CORESMITH_LLM_SLOTS", raising=False)
    calls = []
    monkeypatch.setattr(cl.ClaudeLLM, "_run_cli_unslotted",
                        lambda self, *a, **k: calls.append(a) or ("ok", "", 0, 0.1, False, False, {}))
    out = _llm()._run_cli_with_watchdog(["x"], "p", str(tmp_path), "m", 0.0)
    assert out[0] == "ok" and calls
    assert open_project(tmp_path).leases() == []


def test_slot_is_held_for_the_call_and_released(tmp_path, monkeypatch):
    monkeypatch.setenv("CORESMITH_LLM_SLOTS", "1")
    db = open_project(tmp_path)
    seen = {}

    def fake(self, *a, **k):
        seen["lease"] = db.lease("llm_slot:0")
        return ("ok", "", 0, 0.1, False, False, {})
    monkeypatch.setattr(cl.ClaudeLLM, "_run_cli_unslotted", fake)
    _llm()._run_cli_with_watchdog(["x"], "p", str(tmp_path), "m", 0.0)
    assert seen["lease"] is not None and seen["lease"]["meta"]["model"] == "m"
    assert db.leases() == []


def test_two_workers_serialize_on_one_slot(tmp_path, monkeypatch):
    monkeypatch.setenv("CORESMITH_LLM_SLOTS", "1")
    order = []

    def fake(self, cmd, *a, **k):
        order.append(f"{cmd[0]}-in")
        time.sleep(0.3)
        order.append(f"{cmd[0]}-out")
        return ("ok", "", 0, 0.1, False, False, {})
    monkeypatch.setattr(cl.ClaudeLLM, "_run_cli_unslotted", fake)
    ts = [threading.Thread(target=lambda c: _llm()._run_cli_with_watchdog([c], "p", str(tmp_path), "m", 0.0),
                           args=(c,)) for c in ("a", "b")]
    ts[0].start()
    time.sleep(0.05)
    ts[1].start()
    [t.join() for t in ts]
    assert order == ["a-in", "a-out", "b-in", "b-out"]


def test_two_slots_run_in_parallel(tmp_path, monkeypatch):
    monkeypatch.setenv("CORESMITH_LLM_SLOTS", "2")
    active, peak = [0], [0]
    lock = threading.Lock()

    def fake(self, cmd, *a, **k):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.3)
        with lock:
            active[0] -= 1
        return ("ok", "", 0, 0.1, False, False, {})
    monkeypatch.setattr(cl.ClaudeLLM, "_run_cli_unslotted", fake)
    ts = [threading.Thread(target=lambda: _llm()._run_cli_with_watchdog(["x"], "p", str(tmp_path), "m", 0.0))
          for _ in range(2)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert peak[0] == 2
