# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Engine defects the third MCU+FFT evaluation run exposed
(``coresmith-runs/mcufft-cli-tactics-v3-20260930``).

1. cluster workers shared one lease (livelock);
2. revise prose re-targeted blocks; interrupt ids collided across tiers and
   answered rows stayed ``pending``;
3. chip-level numbers (integration/validation DV, flat synth cells, chip WNS)
   never reached the FRD; worker must-answer questions went nowhere;
4. the backend was out of the architect's reach (parks, CLI resume, OpenROAD
   preflight, gate-sim replay cap, P&R script / deadline);
5. the daemon inherited ``CORESMITH_ROLE``; daemon-client verbs were not in
   ``actions``;
6. the false ERS warning, the missing integration_check accept park, the stale
   signoff scorecard.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi import HTTPException

from orchestrator.architect.cluster import ClusterSession
from orchestrator.architect.session import ArchitectSession
from orchestrator.harness.tools import block as bt
from orchestrator.state_store.project_db import open_project

ROOT = Path(__file__).resolve().parents[2]
CS = str(ROOT / "bin" / "coresmith")


# ---------------------------------------------------------------------------
# A fake LangGraph pipeline for the daemon's resume/backend endpoints: parks are
# (lg_id, payload) pairs; a resume removes the ones it answers.
# ---------------------------------------------------------------------------
class _Intr:
    def __init__(self, iid, value):
        self.id, self.value = iid, value


class _Task:
    def __init__(self, intr):
        self.interrupts = [intr]


class _Snap:
    def __init__(self, parks):
        self.tasks = [_Task(_Intr(i, v)) for i, v in parks]
        self.next = ("node",) if parks else ()
        self.values = {}


class _Graph:
    def __init__(self, pipe):
        self.pipe = pipe

    async def aget_state(self, cfg):
        return _Snap(list(self.pipe.parks))


class _Pipeline:
    thread_id = "t-1"
    status = "interrupted"

    def __init__(self):
        self.parks: list[tuple[str, dict]] = []
        self.task = None
        self.resumes: list = []
        self.graph = _Graph(self)

    async def ensure_graph(self):
        return None

    async def safe_resume(self, cmd, cfg):
        self.resumes.append(cmd)
        val = getattr(cmd, "resume", None)
        if isinstance(val, dict) and all(k in {i for i, _ in self.parks} for k in val):
            self.parks = [(i, v) for i, v in self.parks if i not in val]
        else:
            self.parks = []


@pytest.fixture
def env(tmp_path, monkeypatch):
    from orchestrator.daemon import server as ds
    monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))
    monkeypatch.delenv("CORESMITH_ROLE", raising=False)
    monkeypatch.setattr(ds, "_PROJECT_ROOT", str(tmp_path))
    pipe = _Pipeline()
    monkeypatch.setattr(ds, "_pipeline", pipe)
    monkeypatch.setattr(ds, "_consumed_interrupt_ids", set())
    db = open_project(tmp_path)
    db.begin_run("r1")

    def park(kind, block="blk", actions=("retry", "skip", "abort"), ts=None):
        payload = {"type": kind, "block_name": block, "supported_actions": list(actions),
                   "previous_error": f"{kind} failed", "attempt": len(pipe.parks) + 1}
        iid, payload = db.park_interrupt(payload, graph="pipeline", node=kind)
        if ts is not None:
            with db._tx() as c:
                c.execute("UPDATE interrupts SET ts=? WHERE id=?", (ts, iid))
        lg = f"lg-{iid}"
        pipe.parks.append((lg, payload))
        return iid, lg

    return ds, db, pipe, park



# ===========================================================================
# 1. leases: cluster workers vs the on-call session
# ===========================================================================
def test_cluster_sessions_have_their_own_lease(tmp_path):
    a = ArchitectSession(tmp_path)
    c = ClusterSession(tmp_path, "accel", ["fft64"])
    assert a.lease_name() == "native_session"          # the base class; no Architect lease exists
    assert c.lease_name() == "cluster:accel"
    assert ClusterSession(tmp_path, "periph", ["uart"]).lease_name() == "cluster:periph"
    # the cluster prompt is per instance (no module-global swap that concurrent
    # clusters would race on)
    assert c.system_prompt() != a.system_prompt()


def test_a_foreign_lease_does_not_block_a_cluster_worker(tmp_path, monkeypatch):
    db = open_project(tmp_path)
    tok = db.acquire_lease("native_session", 120, meta={"holder": "another session"})
    seen = {}

    def _run(self):
        seen["lease"] = (open_project(tmp_path).lease(self.lease_name()) or {}).get("name")
        return self._write_status(state="done", sittings=1)
    monkeypatch.setattr(ClusterSession, "_run", _run)
    st = ClusterSession(tmp_path, "accel", ["fft64"]).run()
    assert st["state"] == "done" and "busy" not in str(st.get("stop_reason"))
    assert seen["lease"] == "cluster:accel"
    # a session's own lease still guards it
    assert ArchitectSession(tmp_path, claude_path="/bin/false", max_sittings=1).run()["state"] == "busy"
    db.release_lease("native_session", tok)


def test_two_clusters_run_concurrently_and_one_cluster_never_twice(tmp_path, monkeypatch):
    open_project(tmp_path)
    both_inside = threading.Barrier(2, timeout=20)
    results = {}

    def _run(self):
        both_inside.wait()          # only passes when BOTH clusters hold their lease at once
        return self._write_status(state="done", sittings=1)
    monkeypatch.setattr(ClusterSession, "_run", _run)

    def go(name):
        results[name] = ClusterSession(tmp_path, name, [f"{name}_blk"]).run()
    ts = [threading.Thread(target=go, args=(n,)) for n in ("accel", "periph")]
    for t in ts:
        t.start()
    for t in ts:
        t.join(30)
    assert {n: r["state"] for n, r in results.items()} == {"accel": "done", "periph": "done"}

    db = open_project(tmp_path)
    tok = db.acquire_lease("cluster:accel", 120)
    st = ClusterSession(tmp_path, "accel", ["accel_blk"]).run()
    assert st["state"] == "busy" and st["stop_reason"].startswith("cluster worker busy")
    db.release_lease("cluster:accel", tok)


def test_cluster_sit_attributes_its_verbs_to_the_worker(tmp_path, monkeypatch):
    seen = {}

    def sit(self, prompt, *, resume="", index=1, name="", timeout_s=None, max_turns=None, extra_env=None):
        seen.update(extra_env or {})
        return {"ok": True}
    monkeypatch.setattr(ArchitectSession, "sit", sit)
    ClusterSession(tmp_path, "cpu", ["mcu"]).sit("p", index=1)
    assert seen["CORESMITH_ACTOR"] == "worker:cpu"


# ===========================================================================
# 2. revise targeting, interrupt ids, consumed rows
# ===========================================================================
def _published(tmp_path, names):
    from orchestrator.langgraph import pipeline_graph as pg
    db = open_project(tmp_path)
    (tmp_path / "arch" / "uarch_specs").mkdir(parents=True, exist_ok=True)
    for n in names:
        (tmp_path / "arch" / "uarch_specs" / f"{n}.md").write_text(f"# {n}\n")
        db.set_result(n, "best", {"done": True, "sim_passed": True})
    return pg, db


RUN3_FEEDBACK = "No spec change. fft64 never ran; uart/gpio/sram are published; re-run fft64 only."


def test_prose_never_targets(tmp_path, monkeypatch):
    monkeypatch.delenv("CORESMITH_REVISE_PROSE_TARGETS", raising=False)
    names = ["fft64", "uart", "gpio", "sram"]
    pg, db = _published(tmp_path, names)
    resp = {"action": "revise", "affected_blocks": ["fft64"], "feedback": RUN3_FEEDBACK}
    assert pg._revise_named_blocks(resp, names) == ["fft64"]
    plan = pg._plan_targeted_revise(str(tmp_path), resp, names, [], {}, [], "", 1)
    assert list(plan) == ["fft64"]
    for n in ("uart", "gpio", "sram"):           # untargeted: published result kept, no feedback
        assert db.result(n, "best")["done"] is True
        assert not (tmp_path / ".coresmith" / "blocks" / n / "gate_feedback.txt").exists()
    assert db.result("fft64", "best") is None
    assert "re-run fft64 only" in (tmp_path / ".coresmith" / "blocks" / "fft64" / "gate_feedback.txt").read_text()


def test_prose_targets_env_restores_the_old_rule(tmp_path, monkeypatch):
    monkeypatch.setenv("CORESMITH_REVISE_PROSE_TARGETS", "1")
    names = ["fft64", "uart", "gpio", "sram"]
    pg, db = _published(tmp_path, names)
    resp = {"action": "revise", "affected_blocks": ["fft64"], "feedback": RUN3_FEEDBACK}
    assert pg._revise_named_blocks(resp, names) == names
    pg._plan_targeted_revise(str(tmp_path), resp, names, [], {}, [], "", 1)
    assert all(db.result(n, "best") is None for n in names)       # the run-3 wipe


def test_block_actions_keys_target_and_keep_protects_even_an_unscoped_revise(tmp_path, monkeypatch):
    monkeypatch.delenv("CORESMITH_REVISE_PROSE_TARGETS", raising=False)
    names = ["fft64", "uart", "gpio", "sram"]
    pg, db = _published(tmp_path, names)
    resp = {"action": "revise", "block_actions": json.dumps({"fft64": "keep", "uart": "retry"}),
            "feedback": "fft64 and gpio are fine"}
    assert pg._revise_named_blocks(resp, names) == ["uart"]
    # nothing targeted but a keep: the whole tier re-runs EXCEPT the kept block
    plan = pg._plan_targeted_revise(str(tmp_path), {"action": "revise", "block_actions": {"fft64": "keep"}},
                                    names, ["fft64"], {}, ["fft64"], "summary", 1)
    assert "fft64" not in plan and set(plan) == {"uart", "gpio", "sram"}
    assert db.result("fft64", "best")["done"] is True


def test_interrupt_ids_carry_tier_and_round(tmp_path, monkeypatch):
    from orchestrator.state_store.interrupts import interrupt_id_for
    monkeypatch.delenv("CORESMITH_INTERRUPT_ID_LEGACY", raising=False)
    p = {"type": "uarch_integration_review", "tier": 0}
    kw = {"graph": "pipeline", "node": "uarch_integration_review", "run_id": "r"}
    t0 = interrupt_id_for(p, **kw, task="integration_review:aaa")
    t1 = interrupt_id_for({**p, "tier": 1}, **kw, task="integration_review:aaa")
    r2 = interrupt_id_for(p, **kw, task="integration_review:bbb")
    assert len({t0, t1, r2}) == 3
    assert interrupt_id_for(p, **kw, task="integration_review:aaa") == t0     # re-execution: same id
    monkeypatch.setenv("CORESMITH_INTERRUPT_ID_LEGACY", "1")
    assert interrupt_id_for(p, **kw, task="x") == interrupt_id_for({**p, "tier": 1}, **kw, task="y")


def test_answered_park_stays_consumed_and_the_next_round_gets_a_new_row(tmp_path, monkeypatch):
    """A real LangGraph loop: two integration-review parks of one tier. The
    re-execution after the answer keeps the row consumed; round two parks
    under a new id."""
    from typing import TypedDict

    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command

    from orchestrator.langgraph import pipeline_graph as pg
    monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))
    monkeypatch.delenv("CORESMITH_INTERRUPT_ID_LEGACY", raising=False)
    monkeypatch.delenv("CORESMITH_INTERRUPT_WAIT_S", raising=False)
    db = open_project(tmp_path)
    db.begin_run("r1")

    class S(TypedDict):
        n: int

    def review(s):
        res = pg._park({"type": "uarch_integration_review", "tier": 1,
                        "supported_actions": ["approve", "revise"]})
        assert res.get("action") in ("approve", "revise")
        return {"n": s["n"] + 1}

    g = StateGraph(S)
    g.add_node("review", review)
    g.add_edge(START, "review")
    g.add_conditional_edges("review", lambda s: END if s["n"] >= 2 else "review", ["review", END])
    app = g.compile(checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "t"}}

    def answer(action):
        rows = db.interrupts(status="pending")
        assert len(rows) == 1
        iid = rows[0]["id"]
        db.resolve_interrupt(iid, {"action": action}, resolved_by="architect")
        db.consume_interrupt(iid)
        app.invoke(Command(resume={"action": action}), cfg)
        return iid

    app.invoke({"n": 0}, cfg)
    first = answer("revise")
    assert db.interrupt(first)["status"] == "consumed"          # NOT re-opened by the re-execution
    second = answer("approve")
    assert second != first
    assert {r["id"]: r["status"] for r in db.interrupts()} == {first: "consumed", second: "consumed"}
    assert db.interrupt(second)["resolved_by"] == "architect"


def test_without_a_task_a_consumed_row_is_reopened_as_before(tmp_path):
    db = open_project(tmp_path)
    db.begin_run("r1")
    iid, _ = db.park_interrupt({"type": "x"}, graph="pipeline", node="x")
    db.resolve_interrupt(iid, {"action": "retry"})
    db.consume_interrupt(iid)
    db.park_interrupt({"type": "x"}, graph="pipeline", node="x")
    assert db.interrupt(iid)["status"] == "pending"


# ===========================================================================
# 3. chip-level numbers into the FRD; worker questions
# ===========================================================================
_CHIP_XML = """<testsuites><testsuite name="all">
<testcase classname="test_chip_top" name="test_fft_latency" time="1"/>
<testcase classname="test_chip_top" name="test_uart_echo" time="1"/>
<testcase classname="test_chip_top" name="test_irq" time="1"><failure message="x"/></testcase>
</testsuite></testsuites>"""


@pytest.fixture
def chip_items(tmp_path):
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs" / "task.yaml").write_text("top: chip_top\n")
    db = open_project(tmp_path)
    db.import_block_diagram({"blocks": [{"name": "fft64", "tier": 1}, {"name": "sram", "tier": 1}],
                             "connections": []})
    db.ensure_db_artifact("frd")
    for iid in ("PERF-001", "PERF-004", "INV-IRQ-1", "TIME-001", "TIME-002", "ERS-sram-3", "TEST-009"):
        db.upsert_item("frd", {"id": iid, "text": iid, "priority": "must_have"})
    db.edit_item("PERF-001", metric="latency_cycles", bound_max=400, unit="cycles")
    db.edit_item("PERF-004", metric="rx_overruns", bound_max=0, unit="count")
    db.edit_item("TIME-001", metric="cells", bound_max=60000, unit="cells")
    db.edit_item("TIME-002", metric="wns_ns", bound_min=0, unit="ns")
    db.edit_item("ERS-sram-3", metric="cells", bound_max=500, unit="cells")
    db.link_items("TIME-001", "block:chip_top", "owned_by")
    db.link_items("TIME-002", "block:chip_top", "owned_by")
    db.link_items("ERS-sram-3", "block:sram", "owned_by")
    tb = tmp_path / "tb" / "integration" / "test_chip_top.py"
    tb.parent.mkdir(parents=True)
    tb.write_text("# chip tb\n")
    db.add_verifier("PERF-001", "chip", entry="test_fft_latency")                 # path-less chip verifier
    db.add_verifier("PERF-004", "cocotb", path="tb/integration/test_chip_top.py", entry="test_uart_echo")
    db.add_verifier("INV-IRQ-1", "chip", path=str(tb), entry="test_irq")
    db.add_verifier("TEST-009", "chip", entry="test_only_in_validation")
    sim = tmp_path / "sim_build" / "integration"
    sim.mkdir(parents=True)
    (sim / "results.xml").write_text(_CHIP_XML)
    (sim / "measurements.jsonl").write_text(
        json.dumps({"item": "PERF-001", "value": 216, "unit": "cycles", "test": "test_fft_latency"}) + "\n")
    return db, tmp_path, tb, sim


def test_chip_dv_stamps_items_like_block_done(chip_items, monkeypatch):
    from orchestrator.langgraph.integration_helpers import _stamp_chip_item_checks
    monkeypatch.delenv("CORESMITH_CHIP_ITEM_CHECKS", raising=False)
    db, root, tb, sim = chip_items
    res = _stamp_chip_item_checks(root, "integration", str(tb), sim)
    latest = {c["item_id"]: c for c in db.checks(kind="integration_dv", latest=True)}
    assert latest["PERF-001"]["status"] == "pass" and latest["PERF-001"]["value"] == 216    # measured
    assert latest["PERF-004"]["status"] == "tool_error"     # bounded, passed, but no number recorded
    assert "harness.measure.record" in latest["PERF-004"]["evidence"]
    assert latest["INV-IRQ-1"]["status"] == "fail"
    assert "TEST-009" not in latest                          # its test lives in the validation TB
    assert res["unmeasured_items"] == ["PERF-004"] and set(res["failed_items"]) == {"INV-IRQ-1", "PERF-004"}
    assert db.item("PERF-001")["status"] == "verified" and db.item("INV-IRQ-1")["status"] == "failed"


def test_chip_dv_stamping_scopes_and_env(chip_items, monkeypatch):
    from orchestrator.langgraph.integration_helpers import _stamp_chip_item_checks
    db, root, tb, sim = chip_items
    assert _stamp_chip_item_checks(root, "agent_integration", str(tb), sim) is None   # scratch runs: nothing
    monkeypatch.setenv("CORESMITH_CHIP_ITEM_CHECKS", "0")
    assert _stamp_chip_item_checks(root, "integration", str(tb), sim) is None
    assert db.checks(kind="integration_dv") == []
    monkeypatch.delenv("CORESMITH_CHIP_ITEM_CHECKS")
    (sim / "results.xml").unlink()                        # no verdict: nothing stamped
    assert _stamp_chip_item_checks(root, "integration", str(tb), sim) is None


def test_validation_scope_stamps_validation_dv(chip_items):
    from orchestrator.langgraph.integration_helpers import _stamp_chip_item_checks
    db, root, tb, _ = chip_items
    vtb = root / "tb" / "integration" / "test_chip_top_validation.py"
    vtb.write_text("#\n")
    vsim = root / "sim_build" / "validation"
    vsim.mkdir(parents=True)
    (vsim / "results.xml").write_text(
        '<testsuites><testsuite><testcase classname="v" name="test_only_in_validation"/></testsuite></testsuites>')
    _stamp_chip_item_checks(root, "validation", str(vtb), vsim)
    latest = {c["item_id"]: c["status"] for c in db.checks(kind="validation_dv", latest=True)}
    assert latest == {"TEST-009": "pass"}


def test_flat_synth_and_chip_sta_numbers_reach_chip_items(chip_items, monkeypatch):
    from orchestrator.langgraph import backend_graph as bg
    monkeypatch.delenv("CORESMITH_CHIP_ITEM_CHECKS", raising=False)
    db, root, _, _ = chip_items
    state = {"project_root": str(root)}
    rows = bg._record_chip_checks(state, "synth", bt.CELL_METRICS, 23305, "flat synth")
    assert rows == [{"item": "TIME-001", "status": "pass", "value": 23305.0}]   # ERS-sram-3 is sram's
    assert bg._record_chip_checks(state, "sta", bt.WNS_METRICS, -0.4, "chip sta")[0]["status"] == "fail"
    assert db.item("TIME-002")["status"] == "failed"
    assert db.checks("ERS-sram-3") == []
    monkeypatch.setenv("CORESMITH_CHIP_ITEM_CHECKS", "0")
    assert bg._record_chip_checks(state, "synth", bt.CELL_METRICS, 1, "x") == []


def test_chip_verifier_kind_via_cli(tmp_path):
    db = open_project(tmp_path)
    db.ensure_db_artifact("frd")
    db.upsert_item("frd", {"id": "PERF-001", "text": "x", "priority": "must_have"})
    base = {"CORESMITH_PROJECT_ROOT": str(tmp_path), "PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT)}
    bad = subprocess.run([sys.executable, CS, "frd", "verifier", "PERF-001", "--kind", "chip"],
                         capture_output=True, text=True, env=base, timeout=120)
    assert bad.returncode != 0 and "FRD_VERIFIER_INCOMPLETE" in bad.stdout + bad.stderr
    ok = subprocess.run([sys.executable, CS, "frd", "verifier", "PERF-001", "--kind", "chip",
                         "--entry", "test_fft_latency"], capture_output=True, text=True, env=base, timeout=120)
    assert ok.returncode == 0, ok.stderr
    assert db.verifiers(item_id="PERF-001")[0]["kind"] == "chip"


def test_question_add_and_actions_take_the_worker_actor(tmp_path):
    open_project(tmp_path)
    env_ = {"CORESMITH_PROJECT_ROOT": str(tmp_path), "PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT),
            "CORESMITH_ACTOR": "worker:cpu"}
    p = subprocess.run([sys.executable, CS, "question", "add", "which reset?"], capture_output=True, text=True,
                       env=env_, timeout=120)
    assert p.returncode == 0, p.stderr
    db = open_project(tmp_path)
    assert db.questions()[0]["asked_by"] == "worker:cpu"
    assert db.actions()[-1]["actor"] == "worker:cpu"


# ===========================================================================
# 4. the backend in the architect's reach
# ===========================================================================
class _BIntr:
    def __init__(self, iid, value):
        self.id, self.value = iid, value


class _BTask:
    def __init__(self, i):
        self.interrupts = [i]


class _BSnap:
    def __init__(self, parks):
        self.tasks = [_BTask(_BIntr(i, v)) for i, v in parks]


class _BackendHandle:
    thread_id = "backend-1"
    task = None
    status = "interrupted"

    def __init__(self):
        self.parks: list = []
        outer = self

        class _G:
            async def aget_state(self, cfg):
                return _BSnap(list(outer.parks))
        self.graph = _G()

    async def ensure_graph(self):
        return None


class _FakeMcp:
    def __init__(self, handle):
        self._backend = handle
        self.calls = []

    async def resume_backend(self, action, constraint=""):
        self.calls.append((action, constraint))
        self._backend.parks = []
        return json.dumps({"resumed": True, "action": action})


def _backend_park(db, handle, *, actions=("retry", "skip", "abort", "accept")):
    payload = {"type": "human_intervention_needed", "graph": "backend", "block_name": "chip_top",
               "phase": "pnr", "supported_actions": list(actions), "error": "PnR deadline"}
    iid, payload = db.park_interrupt(payload, graph="backend", node="ask_human")
    handle.parks.append((f"lg-{iid}", payload))
    return iid


def test_backend_resume_validates_the_action(env, monkeypatch):  # noqa: F811 - the imported on-call fixture
    ds, db, pipe, park = env
    handle = _BackendHandle()
    mcp = _FakeMcp(handle)
    monkeypatch.setattr(ds, "_backend_lifecycle", lambda: handle)
    monkeypatch.setattr(ds, "_backend_handle", lambda: mcp)
    _backend_park(db, handle, actions=("retry", "abort"))
    with pytest.raises(HTTPException) as ei:
        asyncio.run(ds.backend_resume(ds.BackendResumeRequest(action="accept")))
    assert ei.value.status_code == 400 and mcp.calls == []
    with pytest.raises(HTTPException) as ei:
        asyncio.run(ds.backend_resume(ds.BackendResumeRequest(action="retry", interrupt_id="int-nope")))
    assert ei.value.status_code == 404


def test_backend_ask_human_parks_through_the_interrupts_table(tmp_path, monkeypatch):
    from orchestrator.langgraph import backend_graph as bg
    monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))
    db = open_project(tmp_path)
    db.begin_run("r1")
    raised = []
    monkeypatch.setattr(bg, "interrupt", lambda p: raised.append(p) or {"action": "retry"})
    out = bg._backend_park({"type": "human_intervention_needed", "graph": "backend", "block_name": "chip_top",
                            "supported_actions": ["retry", "abort"]})
    assert out == {"action": "retry"} and raised[0]["interrupt_id"].startswith("int-")
    assert db.interrupt(raised[0]["interrupt_id"])["graph"] == "backend"


def test_cli_backend_resume_verb():
    p = subprocess.run([sys.executable, CS, "backend", "resume", "--help"], capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": str(ROOT)}, timeout=120)
    assert p.returncode == 0
    for flag in ("--action", "--feedback", "--interrupt-id", "--actor", "--rationale"):
        assert flag in p.stdout


def test_openroad_native_fallback_and_backend_preflight(tmp_path, monkeypatch):
    from orchestrator.pdk.deployments import sky130
    wrapper = str(ROOT / "scripts" / "openroad-nix.sh")
    monkeypatch.delenv("CORESMITH_BACKEND_OPENROAD", raising=False)
    monkeypatch.delenv("CORESMITH_BACKEND_NATIVE_FALLBACK", raising=False)
    monkeypatch.setattr(sky130.Path, "home", classmethod(lambda cls: tmp_path))     # no local build
    which = {"openroad": "/opt/or/bin/openroad"}
    monkeypatch.setattr(sky130.shutil, "which", lambda n, *a, **k: which.get(n))
    assert sky130.openroad_for_backend(wrapper) == ("/opt/or/bin/openroad", "native")
    which.pop("openroad")
    assert sky130.openroad_for_backend(wrapper)[1] == "unreachable"
    pre = sky130.backend_tools_preflight()
    assert not pre["ok"] and "CORESMITH_BACKEND_OPENROAD" in pre["errors"][0]
    which["nix"] = "/usr/bin/nix"                                   # nix present: the wrapper works
    assert sky130.openroad_for_backend(wrapper) == (wrapper, "configured")
    monkeypatch.setenv("CORESMITH_BACKEND_OPENROAD", "/x/openroad")
    assert sky130.openroad_for_backend(wrapper) == ("/x/openroad", "env")
    monkeypatch.delenv("CORESMITH_BACKEND_OPENROAD")
    which.pop("nix")
    which["openroad"] = "/opt/or/bin/openroad"
    monkeypatch.setenv("CORESMITH_BACKEND_NATIVE_FALLBACK", "0")    # old: the wrapper, unreachable
    assert sky130.openroad_for_backend(wrapper) == (wrapper, "unreachable")


def test_backend_start_full_refuses_412_without_openroad(env, monkeypatch):  # noqa: F811 - the imported on-call fixture
    ds, db, pipe, park = env
    launched = []

    class _M:
        async def launch_backend(self, **kw):
            launched.append(kw)
            return {"status": "running"}
    monkeypatch.setattr(ds, "_backend_handle", lambda: _M())
    monkeypatch.setattr(ds, "_backend_preflight", lambda: {
        "ok": False, "errors": ["OpenROAD unreachable: ... Remedy: export CORESMITH_BACKEND_OPENROAD"],
        "warnings": ["magic not found"]})
    with pytest.raises(HTTPException) as ei:
        asyncio.run(ds.backend_start(ds.BackendStartRequest(full=True)))
    assert ei.value.status_code == 412 and "CORESMITH_BACKEND_OPENROAD" in json.dumps(ei.value.detail)
    assert launched == []
    out = asyncio.run(ds.backend_start(ds.BackendStartRequest(full=False)))   # flat synth + gate-sim only
    assert out["status"] == "running" and launched and out["preflight_warnings"]


def test_gate_sim_replays_the_whole_reference_by_default(tmp_path, monkeypatch):
    from orchestrator.harness import gate_sim as gs
    from orchestrator.tests.test_gate_sim import _BLOCK, _good_ref, _runner_for, _stub_env
    monkeypatch.delenv("CORESMITH_GATE_SIM_MAX_CYCLES", raising=False)
    net, pdk = _stub_env(monkeypatch, tmp_path)
    monkeypatch.setattr(gs, "build_and_run_gate_sim", lambda **_: {
        "ok": True, "cycles_compared": 5, "output_bits_compared": 45, "diverged": False})
    res = gs.check_gate_sim(_BLOCK, str(net), "r.v", "tb.py", sim_runner=_runner_for(_good_ref(tmp_path)),
                            pdk_root=pdk, work_root=tmp_path / "work")
    assert res.status == "pass" and res.detail["recorded_cycles"] == res.detail["reference_cycles"]
    # the time budget bounds the replay instead: bounded, never a fail
    monkeypatch.setattr(gs, "build_and_run_gate_sim", lambda **_: {
        "ok": False, "stage": "run", "time_budget_exceeded": True, "timeout_s": 7, "error": "exceeded"})
    res = gs.check_gate_sim(_BLOCK, str(net), "r.v", "tb.py", sim_runner=_runner_for(_good_ref(tmp_path)),
                            pdk_root=pdk, work_root=tmp_path / "work2")
    assert res.status == "bounded" and "time budget" in res.reason and not res.ok


def test_pnr_script_sets_threads_first_and_places_macros_by_odb_name(tmp_path, monkeypatch):
    from orchestrator.langgraph import backend_helpers as bh
    monkeypatch.setenv("CORESMITH_PNR_THREADS", "12")
    net = tmp_path / "top.v"
    net.write_text("module chip_top(input clk); endmodule\n")
    sdc = tmp_path / "top.sdc"
    sdc.write_text("create_clock -period 20 [get_ports clk]\n")
    tcl = Path(bh.prepare_pnr_working_copy("chip_top", str(net), str(sdc), str(tmp_path / "out"))).read_text()
    assert tcl.splitlines()[0] == "set_thread_count 12"
    ref = (ROOT / "orchestrator" / "pdk_templates" / "sky130" / "pnr_reference.tcl").read_text()
    place = ref[ref.index("# --- place SRAM macros"):ref.index("place_pins")]
    code = "\n".join(ln for ln in place.splitlines() if not ln.lstrip().startswith("#"))
    assert "get_full_name" not in code and "getInsts" in place and "[$_inst getName]" in place
    assert "never drop a macro" in code
    monkeypatch.delenv("CORESMITH_PNR_THREADS")
    assert bh.pnr_thread_count() == max(1, os.cpu_count() or 1)


def test_pnr_deadline_scales_with_the_floorplan(tmp_path, monkeypatch):
    from orchestrator.langgraph import backend_helpers as bh
    for k in ("CORESMITH_PNR_DEADLINE_S", "CORESMITH_PNR_TIMEOUT"):
        monkeypatch.delenv(k, raising=False)
    tcl = tmp_path / "pnr.tcl"
    places = " ".join("{%d 10 N}" % (i * 100) for i in range(14))
    tcl.write_text(f'set macro_place [list {places}]\nset macro_die_area "0 0 3100 3100"\n')
    big = bh.pnr_deadline_s(tcl)
    assert 2.5 * 3600 <= big <= 4 * 3600                          # run 3: 3.1 mm, 14 macros
    assert bh.pnr_deadline_s(None, gate_count=2000) == 1800       # a small block keeps 30 min
    monkeypatch.setenv("CORESMITH_PNR_TIMEOUT", "2400")           # legacy override still honoured
    assert bh.pnr_deadline_s(tcl) == 2400
    monkeypatch.setenv("CORESMITH_PNR_DEADLINE_S", "9000")
    assert bh.pnr_deadline_s(tcl) == 9000


# ===========================================================================
# 5. the daemon never runs under a CLI role; lifecycle verbs are logged
# ===========================================================================
def test_daemon_clears_the_role_for_itself_and_its_children(monkeypatch):
    from orchestrator.daemon import server as ds
    monkeypatch.setenv("CORESMITH_ROLE", "watchdog")
    monkeypatch.setenv("CORESMITH_ACTOR", "worker:x")
    monkeypatch.delenv("CORESMITH_DAEMON_KEEP_ROLE", raising=False)
    assert ds._clear_role_env("test") == ["CORESMITH_ROLE", "CORESMITH_ACTOR"]
    assert "CORESMITH_ROLE" not in os.environ and "CORESMITH_ACTOR" not in os.environ
    monkeypatch.setenv("CORESMITH_ROLE", "watchdog")
    monkeypatch.setenv("CORESMITH_DAEMON_KEEP_ROLE", "1")
    assert ds._clear_role_env("test") == [] and os.environ["CORESMITH_ROLE"] == "watchdog"
    src = (ROOT / "bin" / "coresmith").read_text()
    assert 'env.pop(k, None) is not None]' in src and '("CORESMITH_ROLE", "CORESMITH_ACTOR")' in src


def test_daemon_client_verbs_land_in_actions(tmp_path):
    open_project(tmp_path)
    base = {"CORESMITH_PROJECT_ROOT": str(tmp_path), "PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT),
            "CORESMITH_ROLE": "watchdog"}
    subprocess.run([sys.executable, CS, "daemon", "status"], capture_output=True, text=True, env=base, timeout=120)
    p = subprocess.run([sys.executable, CS, "resume", "--action", "approve"], capture_output=True, text=True,
                       env=base, timeout=120)
    assert p.returncode != 0                              # no daemon
    q = subprocess.run([sys.executable, CS, "backend", "resume", "--action", "retry"], capture_output=True,
                       text=True, env={**base, "CORESMITH_ROLE": "architect"}, timeout=120)
    assert q.returncode != 0
    rows = open_project(tmp_path).actions()
    got = [(r["argv"][:2], r["rc"], r["actor"]) for r in rows]
    assert (["daemon", "status"], 0, "watchdog") in got
    assert (["resume", "--action"], 1, "watchdog") in got
    assert (["backend", "resume"], 1, "architect") in got
    assert any("no daemon" in (r.get("summary") or "") for r in rows)


# ===========================================================================
# 6. ERS warning, integration_check accept park, signoff scorecard
# ===========================================================================
@pytest.mark.parametrize("ers", ["arch/ers_spec.json", ".coresmith/ers_spec.json", "arch/ers_spec.md"])
def test_ers_json_satisfies_the_architecture_check(tmp_path, monkeypatch, ers):
    from orchestrator.daemon.server import _check_architecture_artifacts
    monkeypatch.delenv("CORESMITH_SKIP_ARCH_WARN", raising=False)
    for rel in (".coresmith/prd_spec.json", ".coresmith/block_diagram.json", ers):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("{}")
    assert _check_architecture_artifacts(str(tmp_path)) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("action,expect", [("accept", "accepted_check"), ("retry", "retry_requested"),
                                           ("abort", "aborted")])
async def test_clean_integration_check_parks_for_accept(monkeypatch, action, expect):
    from orchestrator.langgraph import pipeline_graph
    from orchestrator.tests.test_integration_compat_wiring import _wire_two_block_design
    monkeypatch.delenv("CORESMITH_INTEGRATION_CHECK_PARK", raising=False)
    monkeypatch.setenv("CORESMITH_DETERMINISTIC_INTEGRATION_CHECK", "1")
    state = _wire_two_block_design(monkeypatch, src_width=8, dst_width=8)
    parked = []
    monkeypatch.setattr(pipeline_graph, "interrupt", lambda p: parked.append(p) or {"action": action})
    out = await pipeline_graph.integration_check_node(state)
    assert [p["type"] for p in parked] == ["integration_check"]
    assert parked[0]["supported_actions"] == ["accept", "retry", "abort"] and parked[0]["lint_clean"] is True
    assert out["integration_result"].get(expect) is True
    route = pipeline_graph.route_after_integration({**state, "integration_result": out["integration_result"]})
    assert route == {"accept": "integration_dv", "retry": "integration_check", "abort": pipeline_graph.END}[action]


@pytest.mark.asyncio
async def test_integration_check_park_env_off_advances(monkeypatch):
    from orchestrator.langgraph import pipeline_graph
    from orchestrator.tests.test_integration_compat_wiring import _wire_two_block_design
    monkeypatch.setenv("CORESMITH_INTEGRATION_CHECK_PARK", "0")
    state = _wire_two_block_design(monkeypatch, src_width=8, dst_width=8)
    parked = []
    monkeypatch.setattr(pipeline_graph, "interrupt", lambda p: parked.append(p) or {"action": "accept"})
    out = await pipeline_graph.integration_check_node(state)
    assert parked == [] and not out["integration_result"].get("accepted_check")


class _StaleScore:
    """The run-3 scoreboard: the latest dv_results rows are a worker's earlier
    `verify rtl` debug runs (fft64 19/20, uart 0/0)."""

    def __init__(self, ts):
        self.ts = ts

    def latest_dv(self, block, scope):
        rows = {"fft64": {"passed": 0, "tests_passed": 19, "tests_total": 20, "source": "agent", "ts": self.ts},
                "uart": {"passed": 0, "tests_passed": 0, "tests_total": 0, "source": "agent", "ts": self.ts}}
        return [rows[block]] if block in rows else []

    def coverage_latest(self, *a, **k):
        return None

    def latest_ppa(self, *a, **k):
        return None


def test_scorecard_reads_the_latest_dv_verdict(tmp_path, monkeypatch):
    from orchestrator.langgraph import final_report as fr
    monkeypatch.delenv("CORESMITH_SCORECARD_PUBLISHED_DV", raising=False)
    db = open_project(tmp_path)
    old = time.time() - 600
    for b, n in (("fft64", 20), ("uart", 17)):
        d = tmp_path / "sim_build" / b
        d.mkdir(parents=True)
        (d / "results.xml").write_text("<testsuites><testsuite>" + "".join(
            f'<testcase classname="t" name="t{i}"/>' for i in range(n)) + "</testsuite></testsuites>")
        from orchestrator.harness.sim_evidence import capture, input_hashes
        rtl = tmp_path / f"{b}.v"
        rtl.write_text(f"module {b}; endmodule")
        evidence = capture(d / "results.xml", input_hashes([rtl]))
        db.set_result(b, "dv_best", {"sim_passed": True, "ts": time.time(), "source": "cluster",
                                      "simulation_evidence": evidence})
    state = {"block_queue": [{"name": "fft64"}, {"name": "uart"}],
             "completed_blocks": [{"name": "fft64", "success": True}, {"name": "uart", "success": True}],
             "target_clock_mhz": 50}
    rep = fr.build_final_report(state, str(tmp_path), scoreboard=_StaleScore(old))
    dv = {b["name"]: b["dv"] for b in rep["blocks"]}
    assert dv["fft64"]["passed"] is True and (dv["fft64"]["tests_passed"], dv["fft64"]["tests_total"]) == (20, 20)
    assert dv["uart"]["passed"] is True and dv["uart"]["tests_total"] == 17
    assert rep["signoff"]["blocks_passed"] == 2
    # a dv_results row NEWER than the published record still wins
    rep = fr.build_final_report(state, str(tmp_path), scoreboard=_StaleScore(time.time() + 60))
    assert {b["name"]: b["dv"]["passed"] for b in rep["blocks"]} == {"fft64": False, "uart": False}
    monkeypatch.setenv("CORESMITH_SCORECARD_PUBLISHED_DV", "0")          # old: the stale row decides
    rep = fr.build_final_report(state, str(tmp_path), scoreboard=_StaleScore(old))
    assert rep["signoff"]["blocks_passed"] == 0
