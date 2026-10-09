# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Projects recorded under earlier stage sequences: the legacy ``arch_model``
/ ``model_eval`` rows are recognised (``current()`` reads past them) and
never rewritten, no model rows are manufactured, and a fresh project still
owes its requirements, interfaces and uArch. What a stored ``done`` row
means is re-evaluated against the current contract: the blocks exit needs
recorded builds, a done stage whose structural criteria no longer hold is
reported as regressed, and ``/run/start`` is refused before ``blocks`` --
``--force`` keeps only its meaning of replacing an existing run, and a
project without a stage machine is at ``requirements``, not exempt."""
from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import HTTPException

from orchestrator.state_store import stages as st
from orchestrator.state_store.project_db import open_project


def _stalled_sol_project(tmp_path):
    """Stored rows shaped like the stalled October run: five legacy stages
    done, ``model_eval`` active with its blockers, three blocks, no model rows."""
    db = open_project(tmp_path)
    for i, name in enumerate(("requirements", "arch_model", "decomposition", "interfaces", "uarch")):
        db.stage_set(name, i, "done")
    db.stage_set("model_eval", 5, "active", blocked_by=[{"code": "MODEL_NOT_BUILT", "ids": ["rv_core"]}])
    db.import_block_diagram({"blocks": [{"name": b, "tier": 1} for b in ("rv_core", "sdram_ctrl", "uart")],
                             "connections": []})
    db.upsert_item("frd", {"id": "PERF-001", "kind": "PERF", "text": "boot", "priority": "must_have", "bound_max": 10.0})
    db.link_items("PERF-001", "block:rv_core", "owned_by")
    return db


def test_stage_sequence_has_no_model_stages():
    assert st.STAGES == ("requirements", "decomposition", "interfaces", "uarch",
                         "blocks", "integration", "acceptance", "backend")
    assert set(st.LEGACY_STAGES) == {"arch_model", "model_eval"}
    assert not hasattr(st, "_entry_arch_model") and not hasattr(st, "_entry_model_eval")


def test_a_project_stalled_at_model_eval_reads_as_blocks_but_its_done_rows_are_re_evaluated(tmp_path):
    db = _stalled_sol_project(tmp_path)
    assert st.current(db) == "blocks"
    s = st.status(db, tmp_path)
    assert s["stage"] == "blocks" and s["index"] == st.STAGES.index("blocks")
    assert s["legacy_rows"] == ["arch_model", "model_eval"]
    assert set(s["done"]) >= {"requirements", "decomposition", "interfaces", "uarch"}
    codes = {b["code"] for b in s["blocked_by"]}
    assert "MODEL_NOT_BUILT" not in codes and "FRD_EVAL_MISSING" not in codes
    assert db.models() == []                                 # nothing manufactured
    # the blocks gate demands recorded builds
    assert codes >= {"BLOCKS_UNPUBLISHED"}
    # the stored done rows are history: their criteria are re-evaluated now
    # (this project registered no documents, bound no targets, built no model)
    regressed = {r["stage"]: r["ids"] for r in s["regressed"]}
    assert set(regressed) == {"requirements", "decomposition", "interfaces", "uarch"}
    assert "MISSING_ARTIFACT" in regressed["requirements"] and "TARGET_UNBOUND" in regressed["uarch"]
    assert s["can_advance"] is False
    rows = {r["name"]: r["status"] for r in db.stage_rows()}
    assert rows["model_eval"] == "active" and rows["arch_model"] == "done"   # old rows retained, not rewritten


def test_blocks_gate_demands_recorded_builds_not_claims(tmp_path, monkeypatch):
    from orchestrator.tests.build_fixtures import complete_build, ready_project
    db = ready_project(tmp_path, monkeypatch, stage="blocks", with_rtl=True)
    from orchestrator.state_store import builds as B
    for b in ("tiny", "sink"):
        db.set_result(b, "best", {"done": True})                    # the old shape: a claim
    # a current model pass, value and all (bound to the model + the requirement it judged)
    db.add_check("PERF-001", "model_eval", "pass", value=5.0, sha=B.model_check_sha(db, tmp_path, "PERF-001"))
    codes = {b["code"]: b["ids"] for b in st.entry(db, tmp_path, "blocks")}
    assert set(codes) == {"BLOCK_NOT_BUILT"} and len(codes["BLOCK_NOT_BUILT"]) == 2
    for b in ("tiny", "sink"):
        complete_build(db, tmp_path, b)
    codes = {b["code"]: b["ids"] for b in st.entry(db, tmp_path, "blocks")}
    # PERF-001 is measured at chip scope in the fixture (deferred); FUNC-001 is owed at block level
    assert codes["OWNED_ITEM_UNVERIFIED"] == ["FUNC-001"]
    db.add_check("FUNC-001", "block_dv", "pass")
    db.add_check("PERF-001", "block_dv", value=7.0)
    res = st.advance(db, tmp_path)
    assert res["advanced"] and res["done"] == "blocks" and res["stage"] == "integration"


def test_fresh_project_still_owes_requirements_interfaces_and_uarch(tmp_path):
    db = open_project(tmp_path)
    assert st.current(db) == "requirements"
    assert [b["code"] for b in st.entry(db, tmp_path, "requirements")] == ["MISSING_ARTIFACT", "MISSING_ARTIFACT"]
    assert "FRD_NO_MODEL_CHECK" not in json.dumps(st.entry(db, tmp_path, "requirements"))
    for stage in ("decomposition", "interfaces"):
        assert any(b["code"] == "MISSING_ARTIFACT" for b in st.entry(db, tmp_path, stage)), stage
    with pytest.raises(ValueError):
        st.entry(db, tmp_path, "model_eval")


def test_model_only_failure_is_advisory_and_rtl_failure_blocks(tmp_path, monkeypatch):
    monkeypatch.delenv("CORESMITH_STAGE_FAIL_BLOCKS", raising=False)
    db = _stalled_sol_project(tmp_path)
    db.add_check("PERF-001", "model_eval", value=12.0)           # 12 > max 10: a model-level fail
    assert db.item("PERF-001")["status"] == "failed"            # the derived status is kept
    for stage in st.STAGES:
        assert not any(b["code"] == "MUST_HAVE_FAILED" for b in st.entry(db, tmp_path, stage)), stage
    assert st.advisories(db) == [{"code": "MODEL_ONLY_FAILED", "ids": ["PERF-001"], "count": 1,
                                  "text": st.advisories(db)[0]["text"]}]
    assert "advisory" in st.advisories(db)[0]["text"]
    # a model pass cannot override an RTL fail (authority rule unchanged) ...
    db.add_check("PERF-001", "block_dv", value=11.0)
    db.add_check("PERF-001", "model_eval", value=3.0)
    assert db.item("PERF-001")["status"] == "failed"
    # ... and a real RTL failure blocks every stage's exit (but is not a regression)
    for stage in st.STAGES:
        codes = {b["code"]: b["ids"] for b in st.entry(db, tmp_path, stage)}
        assert codes["MUST_HAVE_FAILED"] == ["PERF-001"], stage
    assert not any("MUST_HAVE_FAILED" in r["ids"] for r in st.regressed_stages(db, tmp_path))
    assert st.advisories(db) == []
    # an RTL pass supersedes
    db.add_check("PERF-001", "block_dv", value=9.0)
    assert db.item("PERF-001")["status"] == "verified"
    assert not any(b["code"] == "MUST_HAVE_FAILED" for b in st.entry(db, tmp_path, "blocks"))


def test_record_entered_still_refuses_unmet_stages(tmp_path):
    db = open_project(tmp_path)
    db.stage_set("requirements", 0, "active")
    r = st.record_entered(db, tmp_path, "blocks")
    assert r["recorded"] is False and r["blocked_stage"] == "requirements"
    assert {x["name"]: x["status"] for x in db.stage_rows()} == {"requirements": "active"}


# ---------------------------------------------------------------------------
# /run/start before blocks: refused with the blockers; force never bypasses
# ---------------------------------------------------------------------------
class _Snap:
    values: dict = {}


class _Graph:
    async def aget_state(self, cfg):
        return _Snap()


class _Pipeline:
    thread_id = "t"
    task = None
    graph = _Graph()

    async def ensure_graph(self):
        return None


class _Build:
    thread_id = "build"
    task = None


@pytest.fixture
def daemon(tmp_path, monkeypatch):
    from orchestrator.daemon import server as ds
    monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(ds, "_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(ds, "_pipeline", _Pipeline())
    monkeypatch.setattr(ds, "_build", _Build())
    monkeypatch.setattr(ds, "_consumed_interrupt_ids", set())
    monkeypatch.setattr(ds, "_apply_run_env", lambda where: {})
    return ds


def test_run_start_is_refused_before_blocks_without_touching_tools_or_state(daemon, tmp_path, monkeypatch):
    db = open_project(tmp_path)
    db.stage_set("requirements", 0, "done")
    db.stage_set("decomposition", 1, "active")
    touched = []
    monkeypatch.setattr(daemon, "_load_block_queue", lambda f: touched.append("queue") or [])
    monkeypatch.setattr(daemon, "_preflight_or_400", lambda: touched.append("preflight"))
    resp = asyncio.run(daemon.run_start(daemon.StartRequest()))
    assert resp.status_code == 409
    body = json.loads(resp.body)
    assert body["error"] == "STAGE_BEFORE_BLOCKS" and body["stage"] == "decomposition"
    assert [b["code"] for b in body["blocked_by"]] == ["MISSING_ARTIFACT"] and "stage next" in body["hint"]
    assert touched == []                                       # nothing launched, nothing reset
    assert {r["name"]: r["status"] for r in db.stage_rows()} == {"requirements": "done", "decomposition": "active"}


def test_a_refused_run_start_leaves_the_consumed_interrupts_and_stall_clock_alone(daemon, tmp_path, monkeypatch):
    """A refused request starts nothing, so it must not reset the state of
    the run that is there (its answered interrupts, its stall clock)."""
    db = open_project(tmp_path)
    db.stage_set("requirements", 0, "done")
    db.stage_set("decomposition", 1, "active")
    monkeypatch.setattr(daemon, "_consumed_interrupt_ids", {"iid-old"})
    monkeypatch.setattr(daemon, "_last_resume_ts", 1234.5)
    resp = asyncio.run(daemon.run_start(daemon.StartRequest()))          # 409 STAGE_BEFORE_BLOCKS
    assert resp.status_code == 409
    assert daemon._consumed_interrupt_ids == {"iid-old"} and daemon._last_resume_ts == 1234.5
    monkeypatch.setattr(_Snap, "values", {"pipeline_done": True, "completed_blocks": ["a"]})
    with pytest.raises(HTTPException) as ei:                             # 409 a run already exists
        asyncio.run(daemon.run_start(daemon.StartRequest()))
    assert ei.value.status_code == 409 and "already exists" in ei.value.detail
    assert daemon._consumed_interrupt_ids == {"iid-old"} and daemon._last_resume_ts == 1234.5
    monkeypatch.setattr(_Snap, "values", {})
    resp = asyncio.run(daemon.run_start(daemon.StartRequest(force=True)))   # force: still the stage refusal
    assert resp.status_code == 409 and json.loads(resp.body)["error"] == "STAGE_BEFORE_BLOCKS"
    assert daemon._consumed_interrupt_ids == {"iid-old"} and daemon._last_resume_ts == 1234.5


def test_run_start_force_does_not_bypass_the_stage_gate_and_marks_no_stage_done(daemon, tmp_path, monkeypatch):
    db = open_project(tmp_path)
    db.stage_set("requirements", 0, "done")
    db.stage_set("decomposition", 1, "active")
    monkeypatch.setattr(daemon, "_load_block_queue", lambda f: (_ for _ in ()).throw(AssertionError("queue loaded")))
    resp = asyncio.run(daemon.run_start(daemon.StartRequest(force=True)))
    assert resp.status_code == 409 and json.loads(resp.body)["error"] == "STAGE_BEFORE_BLOCKS"
    assert {r["name"]: r["status"] for r in db.stage_rows()} == {"requirements": "done", "decomposition": "active"}


def test_run_start_without_a_stage_machine_is_at_requirements_not_exempt(daemon, tmp_path, monkeypatch):
    open_project(tmp_path)                                      # a DB, but no stage rows
    monkeypatch.setattr(daemon, "_load_block_queue", lambda f: (_ for _ in ()).throw(AssertionError("queue loaded")))
    resp = asyncio.run(daemon.run_start(daemon.StartRequest()))
    assert resp.status_code == 409 and json.loads(resp.body)["error"] == "STAGE_MACHINE_UNUSED"
    assert daemon._stage_before_blocks(str(tmp_path))["error"] == "STAGE_MACHINE_UNUSED"


def test_run_start_at_blocks_passes_only_when_the_done_rows_hold(daemon, tmp_path, monkeypatch):
    from orchestrator.tests.build_fixtures import ready_project
    db = open_project(tmp_path)
    for i, s in enumerate(st.STAGES[:st.STAGES.index("blocks")]):
        db.stage_set(s, i, "done")
    db.stage_set("blocks", st.STAGES.index("blocks"), "active")
    # rows alone are not enough: the done stages are re-evaluated
    assert daemon._stage_before_blocks(str(tmp_path))["error"] == "STAGE_REGRESSED"
    legacy = open_project(tmp_path / "legacy")
    for i, s in enumerate(("requirements", "arch_model", "decomposition", "interfaces", "uarch")):
        legacy.stage_set(s, i, "done")
    legacy.stage_set("model_eval", 5, "active")
    assert daemon._stage_before_blocks(str(tmp_path / "legacy"))["error"] == "STAGE_REGRESSED"   # current() is blocks
    # a project whose stages genuinely hold passes the gate
    ready_project(tmp_path / "ready", monkeypatch, stage="blocks")
    assert daemon._stage_before_blocks(str(tmp_path / "ready")) is None
