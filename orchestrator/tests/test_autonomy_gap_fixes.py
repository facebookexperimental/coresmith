# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Two fail-opens observed on the raster validation run.

D2  A contract audit describing a PREVIOUS failure sat at the stage-derived
    (therefore stable) audit path next to a new, different failure, and was
    quoted as its diagnosis at 0.99 confidence.
D1  All 8 uArch calls carried an EMPTY `--- Python Golden Model ---` block --
    the section was appended unconditionally while every reader of the golden
    returned "" on failure.
"""

from __future__ import annotations

import json

import pytest

from orchestrator.langgraph import pipeline_graph as pg


# ===========================================================================
# D2 -- contract-audit staleness
# ===========================================================================
def _write_ctx(tmp_path, body: dict) -> str:
    d = tmp_path / ".coresmith" / "contract_audit"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "integration_dv_failure_context.json"
    p.write_text(json.dumps(body), encoding="utf-8")
    return str(p)


def test_audit_stamped_for_this_context_is_not_stale(tmp_path):
    ctx = _write_ctx(tmp_path, {"sim_log_tail": "assertion X failed"})
    audit = {"category": "TESTBENCH_BUG",
             pg.CONTRACT_AUDIT_STAMP_KEY: pg._context_fingerprint(ctx)}
    assert pg.contract_audit_staleness(audit, ctx) == ""


def test_audit_stamped_for_a_different_failure_is_flagged(tmp_path):
    """THE BUG: same path, different failure. A 0.99-confidence verdict about
    an already-fixed crash must not read as this failure's diagnosis."""
    ctx = _write_ctx(tmp_path, {"sim_log_tail": "the OLD crash"})
    audit = {"category": "TESTBENCH_BUG", "confidence": 0.99,
             pg.CONTRACT_AUDIT_STAMP_KEY: pg._context_fingerprint(ctx)}
    _write_ctx(tmp_path, {"sim_log_tail": "a NEW and different failure"})
    stale = pg.contract_audit_staleness(audit, ctx)
    assert stale.startswith("STALE CONTRACT AUDIT")
    assert "recommended_action" in stale


def test_unstamped_audit_is_flagged_as_unverified(tmp_path):
    ctx = _write_ctx(tmp_path, {"sim_log_tail": "x"})
    assert "STALE?" in pg.contract_audit_staleness({"category": "X"}, ctx)


def test_no_audit_is_not_a_staleness_claim(tmp_path):
    assert pg.contract_audit_staleness(None, "") == ""
    assert pg.contract_audit_staleness({}, "") == ""


def test_retry_context_leads_with_the_staleness_warning(tmp_path):
    ctx = _write_ctx(tmp_path, {"sim_log_tail": "OLD"})
    audit = {
        "category": "TESTBENCH_BUG", "recommended_action": "fix_tb",
        "confidence": 0.99, "context_path": ctx,
        pg.CONTRACT_AUDIT_STAMP_KEY: pg._context_fingerprint(ctx),
    }
    fresh = pg._format_dv_retry_context({"contract_audit": audit})
    assert not fresh.startswith("!!")
    _write_ctx(tmp_path, {"sim_log_tail": "NEW"})
    stale = pg._format_dv_retry_context({"contract_audit": audit})
    assert stale.splitlines()[0].startswith("!! STALE CONTRACT AUDIT")


def test_fingerprint_of_an_unreadable_context_is_empty_not_matching(tmp_path):
    assert pg._context_fingerprint(str(tmp_path / "nope.json")) == {}


@pytest.mark.asyncio
async def test_agent_does_not_adopt_a_file_from_a_previous_call(tmp_path):
    """The root cause: `if out.exists()` was satisfied by the PREVIOUS audit
    sitting at the stage-derived path before this call even started."""
    from orchestrator.langchain.agents.contract_audit_agent import ContractAuditAgent

    out = tmp_path / "integration_dv_contract_audit.json"
    out.write_text(json.dumps({
        "category": "TESTBENCH_BUG", "confidence": 0.99,
        "outer_agent_summary": "verdict about the PREVIOUS failure",
    }), encoding="utf-8")
    ctx = tmp_path / "ctx.json"
    ctx.write_text("{}", encoding="utf-8")

    agent = ContractAuditAgent.__new__(ContractAuditAgent)

    class _LLM:
        async def call(self, **_kw):
            return '```json\n{"category": "RTL_BUG", "confidence": 0.4}\n```'

    agent.llm = _LLM()
    res = await agent.analyze(
        stage="integration_dv", project_root=str(tmp_path),
        context_path=str(ctx), output_path=str(out),
    )
    assert res["category"] == "RTL_BUG", "adopted the previous call's verdict"
    assert "PREVIOUS failure" not in json.dumps(res)


# ===========================================================================
# D1 -- a missing golden is never silent
# ===========================================================================
def test_uarch_prompt_never_ships_an_empty_golden_fence():
    from orchestrator.langchain.agents import uarch_spec_generator as u
    import inspect

    src = inspect.getsource(u.UarchSpecGenerator.generate)
    assert "if python_source.strip():" in src
    assert "NONE SUPPLIED" in src


def test_declared_but_unreadable_golden_is_loud_and_carried(tmp_path, monkeypatch):
    """A block that DECLARES python_source promised a transcription target.
    Resolving it to "" and appending an empty fence is the fail-open: the spec
    author designs with no target and invents the math."""
    import orchestrator.langgraph.pipeline_helpers as ph

    recorded: list[dict] = []
    monkeypatch.setattr(
        pg, "record_carried_forward_defect",
        lambda root, defect: recorded.append(defect))
    lines: list[str] = []
    monkeypatch.setattr(ph, "log", lambda msg, color="": lines.append(msg))

    ph._report_uarch_golden("blk", "inputs/golden.py:compute", "")
    assert any("resolves to" in ln and "NOTHING" in ln for ln in lines)
    assert any("inputs/golden.py:compute" in ln for ln in lines)
    assert recorded and recorded[0]["kind"] == "declared_golden_unreadable"
    assert "EMPTY golden model" in recorded[0]["unmodeled"]


def test_absent_golden_is_logged_and_carried_forward(tmp_path, monkeypatch):
    """No ref at all is legitimate -- 6 of 8 raster blocks -- but the spec is
    then unconstrained by any reference and the report must say so."""
    import orchestrator.langgraph.pipeline_helpers as ph

    recorded: list[dict] = []
    monkeypatch.setattr(
        pg, "record_carried_forward_defect",
        lambda root, defect: recorded.append(defect))
    lines: list[str] = []
    monkeypatch.setattr(ph, "log", lambda msg, color="": lines.append(msg))

    ph._report_uarch_golden("framebuffer_sram", "", "")     # must NOT raise
    assert any("NO reference golden model" in ln for ln in lines)
    assert recorded and recorded[0]["kind"] == "no_reference_golden"
    assert recorded[0]["first_divergence_block"] == "framebuffer_sram"


def test_present_golden_says_nothing(monkeypatch):
    import orchestrator.langgraph.pipeline_helpers as ph

    called: list = []
    monkeypatch.setattr(
        pg, "record_carried_forward_defect",
        lambda root, defect: called.append(defect))
    ph._report_uarch_golden("blk", "inputs/g.py", "def compute(): pass\n")
    assert called == []


def test_promised_hardware_golden_that_cannot_be_read_fails_rtl_generation():
    from orchestrator.langchain.agents import rtl_generator as rg
    import inspect

    src = inspect.getsource(rg.RTLGeneratorAgent.generate)
    assert "if not _hw_src.strip():" in src
    assert "CORESMITH_RTL_FROM_HW_GOLDEN" in src
