# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Stage-machine enforcement: the /run/start requirements gate, graph-side
stage recording, and pipeline events log rotation."""
from __future__ import annotations

import asyncio
import json
import types

import pytest

from orchestrator.daemon import server
from orchestrator.harness.tools.register import register
from orchestrator.state_store import stages as st
from orchestrator.state_store.project_db import open_project

_PRD = {"prd": {
    "title": "t",
    "functional_requirements": ["FR-TOP-1: The top is soc_top."],
    "validation_kpis": [{"id": "KPI-FPS-1", "metric": "fps", "threshold": ">=30", "test_method": "throughput.py"}],
    "constraints": ["Sky130 only"],
}}

_FRD = """# FRD

## Performance Requirements

1. **ID**: PERF-001
   - **Requirement**: Mean cycles/frame <= 2,133,333 [HARD, KPI-FPS-1].
   - **Acceptance criteria**: throughput.py mean <= 2133333.
   - **Priority**: must_have
   - **Model check**: arch model cycle accounting.
"""

_ALL = ["PRD_NOT_REGISTERED", "FRD_NOT_REGISTERED", "FRD_NO_MUST_HAVE"]


def _register_requirements(root):
    (root / ".coresmith").mkdir(parents=True, exist_ok=True)
    (root / "arch").mkdir(exist_ok=True)
    (root / ".coresmith" / "prd_spec.json").write_text(json.dumps(_PRD))
    (root / "arch" / "frd_spec.md").write_text(_FRD)
    db = open_project(root)
    assert register(db, root, "prd", ".coresmith/prd_spec.json")["ok"]
    assert register(db, root, "frd", "arch/frd_spec.md")["ok"]
    return db


@pytest.fixture()
def gate_env(monkeypatch):
    monkeypatch.delenv("CORESMITH_SKIP_ARCH_WARN", raising=False)
    monkeypatch.delenv("CORESMITH_REQUIRE_REQUIREMENTS", raising=False)
    return monkeypatch


# -- requirements_registered -------------------------------------------------

def test_requirements_registered_empty_project(tmp_path):
    ok, missing = server.requirements_registered(tmp_path)
    assert ok is False
    assert missing == _ALL


def test_requirements_registered_ok(tmp_path):
    _register_requirements(tmp_path)
    assert server.requirements_registered(tmp_path) == (True, [])


# -- /run/start gate -----------------------------------------------------------

class _PastGate(Exception):
    """Raised by the first step after the gate: the gate let the start through."""


def _call_run_start(monkeypatch, root):
    monkeypatch.setattr(server, "_PROJECT_ROOT", str(root))
    monkeypatch.setattr(server, "_pipeline", types.SimpleNamespace(task=None))
    monkeypatch.setattr(server, "_build", types.SimpleNamespace(task=None, thread_id="build"))
    monkeypatch.setattr(server, "_apply_run_env", lambda where: [])
    monkeypatch.setattr(server, "_load_block_queue", lambda f: [{"name": "a", "tier": 1}])
    monkeypatch.setattr(server, "_preflight_or_400", lambda: None)

    def _past(_root):
        raise _PastGate()
    monkeypatch.setattr(server, "_check_architecture_artifacts", _past)
    return asyncio.run(server.run_start(server.StartRequest(force=True)))


def test_run_start_refused_without_a_stage_machine(tmp_path, gate_env):
    """A project with no stage rows starts at requirements: the stage gate
    refuses before the requirements gate, and ``--force`` does not help."""
    resp = _call_run_start(gate_env, tmp_path)
    assert resp.status_code == 409
    body = json.loads(resp.body)
    assert body["error"] == "STAGE_MACHINE_UNUSED"
    assert "stage next" in body["hint"]


def test_run_start_refused_before_blocks_even_with_requirements(tmp_path, gate_env):
    _register_requirements(tmp_path)
    db = open_project(tmp_path)
    db.stage_set("requirements", 0, "active")
    resp = _call_run_start(gate_env, tmp_path)
    assert resp.status_code == 409
    assert json.loads(resp.body)["error"] == "STAGE_BEFORE_BLOCKS"


def test_run_start_passes_the_stage_gate_at_blocks(tmp_path, gate_env):
    from orchestrator.tests.build_fixtures import ready_project
    ready_project(tmp_path, gate_env, stage="blocks")
    with pytest.raises(_PastGate):
        _call_run_start(gate_env, tmp_path)


@pytest.mark.parametrize("var,value", [("CORESMITH_SKIP_ARCH_WARN", "1"),
                                       ("CORESMITH_REQUIRE_REQUIREMENTS", "0"),
                                       ("CORESMITH_REQUIRE_REQUIREMENTS", "off")])
def test_requirements_gate_bypass_does_not_bypass_the_stage_gate(tmp_path, gate_env, var, value):
    gate_env.setenv(var, value)
    resp = _call_run_start(gate_env, tmp_path)
    assert resp.status_code == 409 and json.loads(resp.body)["error"] == "STAGE_MACHINE_UNUSED"


def test_requirements_gate_still_applies_after_the_stage_gate(tmp_path, gate_env):
    """The requirements gate is reached only once the stage machine is at
    blocks; it is unchanged (and its env bypass still only bypasses it)."""
    from orchestrator.tests.build_fixtures import ready_project
    ready_project(tmp_path, gate_env, stage="blocks")
    gate_env.delenv("CORESMITH_SKIP_ARCH_WARN", raising=False)
    gate_env.setattr(server, "requirements_registered", lambda root: (False, _ALL))
    resp = _call_run_start(gate_env, tmp_path)
    assert resp.status_code == 409 and json.loads(resp.body)["error"] == "requirements_not_registered"
    gate_env.setenv("CORESMITH_REQUIRE_REQUIREMENTS", "0")
    with pytest.raises(_PastGate):
        _call_run_start(gate_env, tmp_path)


def test_gate_default_on_and_skip_warn_zero_does_not_bypass(gate_env):
    assert server._requirements_gate_enabled() is True
    gate_env.setenv("CORESMITH_SKIP_ARCH_WARN", "0")
    gate_env.setenv("CORESMITH_REQUIRE_REQUIREMENTS", "1")
    assert server._requirements_gate_enabled() is True


# -- record_entered ------------------------------------------------------------

def test_record_entered_noop_without_stage_machine(tmp_path):
    db = open_project(tmp_path)
    r = st.record_entered(db, tmp_path, "integration")
    assert r == {"recorded": False, "reason": "stage machine unused"}
    assert db.stage_rows() == []
    assert not (tmp_path / ".coresmith" / "pipeline_events.jsonl").exists()


def test_record_entered_advances_and_is_idempotent(tmp_path, monkeypatch):
    from orchestrator.tests.build_fixtures import ready_project
    from orchestrator.tests.test_build_identity import _complete_build
    db = ready_project(tmp_path, monkeypatch, stage="blocks", with_rtl=True)
    for m in ("tiny", "sink"):
        _complete_build(db, tmp_path, m)          # the blocks exit holds on recorded builds
    db.add_check("FUNC-001", "block_dv", "pass")
    db.add_check("PERF-001", "block_dv", None, value=50.0)

    r = st.record_entered(db, tmp_path, "integration")
    assert r == {"recorded": True, "stage": "integration", "newly_done": ["blocks"]}
    rows = {x["name"]: x["status"] for x in db.stage_rows()}
    assert rows["blocks"] == "done" and rows["integration"] == "active"
    assert st.current(db) == "integration"

    events = tmp_path / ".coresmith" / "pipeline_events.jsonl"
    lines = [json.loads(ln) for ln in events.read_text().splitlines() if ln.strip()]
    entered = [ln for ln in lines if "stage_entered" in json.dumps(ln)]
    assert len(entered) == 1

    again = st.record_entered(db, tmp_path, "integration")
    assert again == {"recorded": True, "stage": "integration", "newly_done": []}
    assert {x["name"]: x["status"] for x in db.stage_rows()} == rows
    # a revise that loops back to the tier loop does not regress the machine
    assert st.record_entered(db, tmp_path, "blocks")["newly_done"] == []
    assert st.current(db) == "integration"
    lines = [ln for ln in events.read_text().splitlines() if "stage_entered" in ln]
    assert len(lines) == 1


# -- events log rotation -------------------------------------------------------

def test_rotate_events_default_renames(tmp_path, monkeypatch):
    monkeypatch.delenv("CORESMITH_ROTATE_EVENTS", raising=False)
    path = tmp_path / ".coresmith" / "pipeline_events.jsonl"
    path.parent.mkdir()
    path.write_text('{"event_type": "stage_done"}\n')
    rotated = server._rotate_events(path)
    assert path.exists() and path.read_text() == ""
    olds = list(path.parent.glob("pipeline_events.*.jsonl"))
    assert olds == [rotated]
    assert "stage_done" in rotated.read_text()


def test_rotate_events_empty_file_not_rotated(tmp_path, monkeypatch):
    monkeypatch.delenv("CORESMITH_ROTATE_EVENTS", raising=False)
    path = tmp_path / "pipeline_events.jsonl"
    assert server._rotate_events(path) is None
    assert path.read_text() == ""
    assert list(tmp_path.glob("pipeline_events.*.jsonl")) == []


def test_rotate_events_disabled_truncates(tmp_path, monkeypatch):
    monkeypatch.setenv("CORESMITH_ROTATE_EVENTS", "0")
    path = tmp_path / "pipeline_events.jsonl"
    path.write_text('{"event_type": "stage_done"}\n')
    assert server._rotate_events(path) is None
    assert path.read_text() == ""
    assert list(tmp_path.glob("pipeline_events.*.jsonl")) == []


def test_graph_stage_recording_skips_projects_without_a_db(tmp_path):
    from orchestrator.langgraph import pipeline_graph
    assert st.project_db_exists(tmp_path) is False
    pipeline_graph._record_stage(str(tmp_path), "integration")
    assert not (tmp_path / ".coresmith").exists()


def test_graph_stage_recording_never_marks_unmet_stages_done(tmp_path, monkeypatch):
    """The graph records the stage it entered; it never certifies the
    stages before it. Without recorded builds the ``blocks`` exit does not
    hold, so entering ``acceptance`` is refused at ``blocks``."""
    from orchestrator.langgraph import pipeline_graph
    from orchestrator.tests.build_fixtures import ready_project
    db = ready_project(tmp_path, monkeypatch, stage="blocks")
    pipeline_graph._record_stage(str(tmp_path), "blocks")
    assert st.current(db) == "blocks"
    pipeline_graph._record_stage(str(tmp_path), "acceptance")
    rows = {x["name"]: x["status"] for x in db.stage_rows()}
    assert rows["blocks"] == "active" and "acceptance" not in rows and "integration" not in rows
    refused = [json.loads(ln) for ln in (tmp_path / ".coresmith" / "pipeline_events.jsonl").read_text().splitlines()
               if "stage_entry_refused" in ln]
    assert refused and refused[-1]["blocked_stage"] == "blocks"
