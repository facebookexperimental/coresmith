# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The RTL worker receives measured timing context and a compact job contract."""
from __future__ import annotations

from orchestrator.langchain.agents import rtl_generator as rg


def test_worker_prompt_names_inputs_and_checks_without_a_rule_catalogue():
    prompt = rg.SYSTEM_PROMPT
    assert len(prompt) < 5000
    for required in ("uArch spec", "reference model", "acceptance testbench", "authoritative port table",
                     "synthesis/timing", '"$CS" tool run_lint', "unchanged"):
        assert required in prompt


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
