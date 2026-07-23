# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the RTL-generator pipeline-synthesizability discipline wiring.

The codec RD-search failure was an RTL-fidelity gap: the uArch spec correctly
described an N-stage pipeline, but the ``rtl_generator`` agent never saw the
``pipeline_contract`` skill or the PDK timing budget, so it collapsed the
datapath into one combinational cloud that walls the synth gate. These tests
pin the fix: the skill is in the RTL system prompt, the prompt no longer
*permits* unrolling a search into a cloud, and the gated PDK-budget helper
behaves correctly (off by default, populated when characterized).
"""
from __future__ import annotations

import importlib

import pytest

from orchestrator.langchain.agents import rtl_generator as rg
from orchestrator.langchain.prompts.skills import load_skills


def test_pipeline_contract_skill_is_in_rtl_system_prompt():
    """The RTL agent's system prompt must carry the pipeline_contract skill."""
    sp = rg.SYSTEM_PROMPT
    assert "combinational cloud" in sp
    assert "registered boundary" in sp
    # the RTL-implementer subsection (added to the shared skill) must be present
    assert "At RTL time" in sp
    assert "always @(posedge clk)" in sp
    # the skill is appended under its own MANDATORY header, not just referenced
    assert "synthesizable-pipeline discipline" in sp


def test_skill_text_matches_loaded_skill():
    """The appended text is exactly the on-disk pipeline_contract skill."""
    skill = load_skills("pipeline_contract")
    assert skill, "pipeline_contract.md must exist and be non-empty"
    assert skill.strip() in rg.SYSTEM_PROMPT


def test_rtl_prompt_no_longer_permits_unrolling_a_search():
    """The old permissive 'or combinational unrolling' line must be gone."""
    sp = rg.SYSTEM_PROMPT
    norm = " ".join(sp.split())  # whitespace-robust (prompt is line-wrapped)
    # the exact permissive phrasing that licensed the cloud
    assert "FSMs with counters or combinational unrolling" not in norm
    # and the replacement forbids the cloud explicitly
    assert "UNSYNTHESIZABLE" in norm
    assert "NEVER unroll a multi-op" in norm


def test_pdk_budget_fragment_empty_when_stage_disabled(monkeypatch):
    """Gated off (default): the budget helper returns '' and never raises."""
    monkeypatch.delenv("CORESMITH_PDK_CHAR", raising=False)
    assert rg._pdk_budget_fragment() == ""


def test_pdk_budget_fragment_fail_open_on_import_error(monkeypatch):
    """If the budget machinery errors, the helper degrades to '' (never blocks)."""
    import orchestrator.langgraph.pdk_characterize as pdkc

    monkeypatch.setattr(pdkc, "stage_enabled", lambda: True)
    monkeypatch.setattr(pdkc, "is_characterized", lambda: True)

    def _boom(*a, **k):
        raise RuntimeError("characterization machinery exploded")

    import orchestrator.langgraph.pipeline_scheduler as ps

    monkeypatch.setattr(ps, "pdk_budget_section", _boom)
    assert rg._pdk_budget_fragment() == ""


def test_pdk_budget_fragment_populates_when_characterized(monkeypatch):
    """When enabled + characterized, the helper returns the real budget text."""
    import orchestrator.langgraph.pdk_characterize as pdkc
    import orchestrator.langgraph.pipeline_scheduler as ps

    monkeypatch.setattr(pdkc, "stage_enabled", lambda: True)
    monkeypatch.setattr(pdkc, "is_characterized", lambda: True)
    monkeypatch.setattr(
        ps, "pdk_budget_section", lambda mhz=50.0, pdk=None: "BUDGET-ROWS-HERE"
    )
    assert rg._pdk_budget_fragment() == "BUDGET-ROWS-HERE"


def test_rtl_md_prompt_file_forbids_cloud():
    """The on-disk rtl_generator.md (the real prompt) must carry the rule."""
    from pathlib import Path

    md = (
        Path(rg.__file__).resolve().parent.parent
        / "prompts"
        / "rtl_generator.md"
    )
    text = md.read_text()
    assert "FSMs with counters or combinational unrolling" not in text
    assert "REGISTERED" in text and "combinational cloud" in text


def test_module_reimport_is_stable():
    """Re-importing must not double-append the skill (idempotent at import)."""
    before = rg.SYSTEM_PROMPT.count("At RTL time")
    importlib.reload(rg)
    after = rg.SYSTEM_PROMPT.count("At RTL time")
    assert before == after == 1
