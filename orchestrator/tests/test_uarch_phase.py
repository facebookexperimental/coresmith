# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""B2: the uArch phase authors every spec and delivers the SystemC SoC model
before the first RTL tier."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from orchestrator.langgraph import pipeline_graph as pg
from orchestrator.state_store.project_db import open_project
from orchestrator.systemc_model import detect
from orchestrator.tests.test_systemc_model import (
    _BLOCKS,
    _EDGES,
    _PADS,
    _REQ,
    _RSP,
    _SINK,
    _extra_members,
)

_BODIES = {"req": _REQ, "rsp": _RSP, "sink": _SINK, "pads": _PADS}


def _project(tmp_path):
    (tmp_path / "inputs").mkdir(exist_ok=True)
    (tmp_path / "inputs" / "task.yaml").write_text("top: tiny\n")
    db = open_project(tmp_path)
    db.import_block_diagram({"blocks": [{"name": b, "tier": 1} for b in _BLOCKS], "connections": []})
    db.import_contracts({"contracts": _EDGES})
    return db


def _state(tmp_path):
    return {"project_root": str(tmp_path), "block_queue": [{"name": b, "tier": 1} for b in _BLOCKS]}


async def _fake_specs(blocks, feedback_by_block=None):
    root = Path(pg._pr({"project_root": _fake_specs.root}))
    d = root / "arch" / "uarch_specs"
    d.mkdir(parents=True, exist_ok=True)
    for b in blocks:
        (d / f"{b['name']}.md").write_text(f"# {b['name']}\n### 4a. Cross-Block Semantic Invariants\n- INV-{b['name'].upper()}-001 x\n")
    return {"written": [b["name"] for b in blocks], "missing": [], "session_id": "s"}


class _FakeAgent:
    calls: list = []

    def __init__(self, *a, **k):
        pass

    async def generate(self, block_name, *, project_root, header_path, compiler_log="", attempt=1):
        _FakeAgent.calls.append((block_name, attempt))
        md = Path(project_root) / "model"
        h = md / f"{block_name}_model.h"
        h.write_text(h.read_text().replace("  void run();", _extra_members(block_name) + "  void run();"))
        (md / f"{block_name}_model.cpp").write_text(_BODIES[block_name])
        return {"files_written": [f"model/{block_name}_model.cpp"], "written": True}


def test_flag_off_is_a_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("CORESMITH_UARCH_PHASE", "0")
    assert asyncio.run(pg.uarch_phase_node(_state(tmp_path))) == {}


def test_graph_wires_the_phase_and_fan_out_reuses_specs(tmp_path, monkeypatch):
    monkeypatch.setenv("CORESMITH_UARCH_PHASE", "1")
    monkeypatch.setattr(pg, "_stale_specs", lambda pr, names: [])
    monkeypatch.setattr(pg, "_uarch_single_context_enabled", lambda: False)
    state = {"project_root": str(tmp_path), "target_clock_mhz": 50.0, "max_attempts": 3,
             "block_queue": [{"name": "req", "tier": 1}], "tier_list": [1], "current_tier_index": 0}
    assert pg.fan_out_tier(state)[0].arg["reuse_spec"] is True


def test_toolchain_missing_is_recorded_not_fatal(tmp_path, monkeypatch):
    monkeypatch.setenv("CORESMITH_UARCH_PHASE", "1")
    monkeypatch.setenv("CORESMITH_SYSTEM_MODEL", "1")
    _project(tmp_path)
    _fake_specs.root = str(tmp_path)
    import orchestrator.langgraph.pipeline_helpers as ph
    import orchestrator.systemc_model as scm
    monkeypatch.setattr(ph, "generate_uarch_specs_single_context", _fake_specs)
    monkeypatch.setattr(scm, "detect", lambda: {"ok": False, "reason": "no systemc", "cxx": None, "systemc_home": ""})
    out = asyncio.run(pg.uarch_phase_node(_state(tmp_path)))
    r = out["uarch_phase"]
    assert sorted(r["specs"]["written"]) == sorted(_BLOCKS)
    assert (tmp_path / "arch/uarch_specs/rsp.md").exists()
    assert "system_model_toolchain_missing" in r["system_model"]["parked_reason"]
    assert json.loads((tmp_path / ".coresmith/system_model.json").read_text())["system_model"]["build_ok"] is None


@pytest.mark.slow
@pytest.mark.skipif(not detect()["ok"], reason="SystemC toolchain not available")
def test_phase_builds_and_smokes_the_soc_model(tmp_path, monkeypatch):
    monkeypatch.setenv("CORESMITH_UARCH_PHASE", "1")
    monkeypatch.setenv("CORESMITH_SYSTEM_MODEL", "1")
    db = _project(tmp_path)
    _fake_specs.root = str(tmp_path)
    _FakeAgent.calls = []
    import orchestrator.langchain.agents.systemc_model_generator as gen
    import orchestrator.langgraph.pipeline_helpers as ph
    monkeypatch.setattr(ph, "generate_uarch_specs_single_context", _fake_specs)
    monkeypatch.setattr(gen, "SystemCModelGenerator", _FakeAgent)
    out = asyncio.run(pg.uarch_phase_node(_state(tmp_path)))
    sm = out["uarch_phase"]["system_model"]
    assert sm["build_ok"] is True and sm["smoke_ok"] is True, sm.get("build_log", "")[-2000:]
    assert sorted(b for b, _ in _FakeAgent.calls) == sorted(_BLOCKS)
    assert (tmp_path / "model" / "soc_model").exists() and (tmp_path / "model" / "soc_model.cpp").exists()
    rows = {m["block"]: m for m in db.models()}
    assert set(rows) == set(_BLOCKS) and all(r["build_ok"] == 1 and r["smoke_ok"] == 1 for r in rows.values())
    assert "reads=4" in sm["smoke_log"]


def _gate_run(tmp_path, monkeypatch, sm):
    monkeypatch.setenv("CORESMITH_UARCH_PHASE", "1")
    monkeypatch.setenv("CORESMITH_SYSTEM_MODEL", "1")
    _project(tmp_path)
    parked = []

    async def _specs(pr, blocks):
        return {"written": [], "missing": []}

    async def _models(pr, blocks):
        return dict(sm, enabled=True)
    monkeypatch.setattr(pg, "_uarch_phase_specs", _specs)
    monkeypatch.setattr(pg, "_uarch_phase_models", _models)
    monkeypatch.setattr(pg, "_park", lambda payload, **k: parked.append((payload, k)))
    asyncio.run(pg.uarch_phase_node(_state(tmp_path)))
    return parked


def test_gate_parks_when_the_model_smoke_fails(tmp_path, monkeypatch):
    monkeypatch.delenv("CORESMITH_UARCH_PHASE_GATE", raising=False)   # default on
    parked = _gate_run(tmp_path, monkeypatch, {"build_ok": True, "smoke_ok": False})
    assert len(parked) == 1 and parked[0][0]["type"] == "uarch_phase_failed"
    assert parked[0][1]["node"] == "uarch_phase" and "retry" in parked[0][0]["supported_actions"]


def test_gate_parks_when_the_model_does_not_build(tmp_path, monkeypatch):
    monkeypatch.setenv("CORESMITH_UARCH_PHASE_GATE", "1")
    assert len(_gate_run(tmp_path, monkeypatch, {"build_ok": False, "smoke_ok": None})) == 1


def test_gate_off_records_and_continues(tmp_path, monkeypatch):
    monkeypatch.setenv("CORESMITH_UARCH_PHASE_GATE", "0")
    assert _gate_run(tmp_path, monkeypatch, {"build_ok": True, "smoke_ok": False}) == []


def test_gate_ignores_a_missing_toolchain_and_a_passing_model(tmp_path, monkeypatch):
    monkeypatch.delenv("CORESMITH_UARCH_PHASE_GATE", raising=False)
    assert _gate_run(tmp_path, monkeypatch, {"build_ok": None, "smoke_ok": None,
                                             "parked_reason": "system_model_toolchain_missing"}) == []
    assert _gate_run(tmp_path, monkeypatch, {"build_ok": True, "smoke_ok": True}) == []
