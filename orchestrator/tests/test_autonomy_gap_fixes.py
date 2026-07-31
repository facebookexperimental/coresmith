# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""A fail-open observed on the raster validation run.

D2  A contract audit describing a PREVIOUS failure sat at the stage-derived
    (therefore stable) audit path next to a new, different failure, and was
    quoted as its diagnosis at 0.99 confidence.
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
