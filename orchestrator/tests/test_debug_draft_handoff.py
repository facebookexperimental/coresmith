# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Debug workers author drafts; the engine owns persistent diagnosis views."""
import asyncio
import json
import re
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from orchestrator.langchain.agents.debug_agent import DebugAgent
from orchestrator.state_store.project_db import open_project


@pytest.mark.asyncio
async def test_readonly_view_is_not_worker_output_and_stale_output_is_not_adopted(tmp_path):
    db = open_project(tmp_path)
    db.set_diagnosis("leaf", {"diagnosis": "old", "category": "LOGIC_ERROR"})
    view = tmp_path / ".coresmith/blocks/leaf/diagnosis.json"
    before = view.read_bytes()
    assert view.stat().st_mode & 0o222 == 0
    outputs = []

    async def call(**kw):
        path = Path(re.search(r"(/[^\s]+/drafts/debug/[^\s]+/diagnosis\.json)", kw["prompt"]).group(1))
        assert not path.exists()
        outputs.append(path)
        if len(outputs) == 1:
            path.write_text(json.dumps({"diagnosis": "fresh", "category": "TESTBENCH_BUG"}))
        return "done"

    agent = DebugAgent.__new__(DebugAgent)
    agent.llm = type("FakeLLM", (), {"call": staticmethod(call)})()
    result = await agent._debug_analysis("leaf", "sim", str(tmp_path))
    assert result["category"] == "TESTBENCH_BUG"
    assert view.read_bytes() == before
    missing = await agent._debug_analysis("leaf", "sim", str(tmp_path))
    assert missing["category"] == "AGENT_ERROR"
    assert outputs[0] != outputs[1]


@pytest.mark.asyncio
async def test_concurrent_same_block_diagnoses_have_isolated_outputs(tmp_path):
    outputs = []

    async def call(**kw):
        path = Path(re.search(r"(/[^\s]+/drafts/debug/[^\s]+/diagnosis\.json)", kw["prompt"]).group(1))
        outputs.append(path)
        await asyncio.sleep(0)
        path.write_text(json.dumps({"diagnosis": str(path), "category": "LOGIC_ERROR"}))
        return "done"

    agent = DebugAgent.__new__(DebugAgent)
    agent.llm = type("FakeLLM", (), {"call": staticmethod(call)})()
    results = await asyncio.gather(*(agent._debug_analysis("leaf", "sim", str(tmp_path)) for _ in range(2)))
    assert len(set(outputs)) == 2
    assert {r["diagnosis"] for r in results} == set(map(str, outputs))


@pytest.mark.asyncio
async def test_native_diagnose_imports_constraints_without_replacing_prior_rules(tmp_path, monkeypatch):
    from orchestrator.langgraph import pipeline_graph as pg
    db = open_project(tmp_path)
    db.add_constraint("leaf", "Keep reset polarity", source="operator")
    diag = {"diagnosis": "wrong compare", "category": "LOGIC_ERROR", "confidence": 0.9,
            "suggested_fix": "Use the expected signed comparison", "constraints": [
                {"rule": "Keep reset polarity"},
                {"rule": "Compare signed operands", "code_snippet": "$signed(a) < $signed(b)"},
                {"rule": "Compare signed operands"}, None,
            ]}
    monkeypatch.setattr(pg, "diagnose_failure", AsyncMock(return_value=diag))
    state = {"current_block": {"name": "leaf"}, "project_root": str(tmp_path),
             "phase": "sim", "attempt": 1, "max_attempts": 5}
    await pg.diagnose_node(state)
    rules = db.constraints("leaf")
    assert len(rules) == 2
    assert rules[0]["source"] == "operator"
    assert rules[1]["source"] == "debug_agent"
    assert rules[1]["code_snippet"] == "$signed(a) < $signed(b)"
    db.export_block_views("leaf")
    assert json.loads((tmp_path / ".coresmith/blocks/leaf/constraints.json").read_text()) == rules
    assert db.diagnosis("leaf")["diagnosis"] == "wrong compare"


@pytest.mark.asyncio
@pytest.mark.parametrize("category", ["INFRASTRUCTURE_ERROR", "SIM_TIMEOUT", "AGENT_ERROR"])
async def test_missing_hardware_verdict_cannot_install_design_constraints(tmp_path, monkeypatch, category):
    from orchestrator.langgraph import pipeline_graph as pg
    db = open_project(tmp_path)
    db.add_constraint("leaf", "Keep reset polarity", source="operator")
    before = db.constraints("leaf")
    diag = {"category": category, "diagnosis": "Timing repair did not complete",
            "constraints": [{"rule": "MUST bank the memory into at least 12 instances"}]}
    monkeypatch.setattr(pg, "diagnose_failure", AsyncMock(return_value=diag))
    await pg.diagnose_node({"current_block": {"name": "leaf"}, "project_root": str(tmp_path),
                            "phase": "synth", "attempt": 1, "max_attempts": 3})
    assert db.diagnosis("leaf")["constraints"] == diag["constraints"]
    assert db.constraints("leaf") == before
