# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Chip lead = the architect resumed (architect sitting, step 5).

The chip-lead decisions (integration review, spec re-verification, parked
block questions) used to be a fresh policy with a 20 KB payload and ten prior
decisions as its whole memory. With the sitting on, the same architect
session that wrote the architecture is resumed (``--resume``, cache warm)
with the interrupt payload and answers in the chip-lead decision schema; the
engine validates the action against ``supported_actions`` exactly as before,
so nothing downstream changes. No session id (the sitting never ran) or a
malformed answer falls back to the chip-lead agent.
"""
from __future__ import annotations

import json
from pathlib import Path

from .session import ArchitectSession

_DECISION_SCHEMA = '''Reply with ONE fenced ```json block:
{"action": "<one of the payload's supported_actions>",
 "reasoning": "<why, citing item ids / block names / evidence>",
 "feedback": "<for retry or revise: concrete guidance that changes the next attempt>",
 "block_actions": {"<block>": "retry|skip|revise"}   (optional, per-block)
}'''


def session_id(project_root) -> str:
    try:
        from orchestrator.state_store.project_db import open_project
        return open_project(project_root).get_setting("architect_session_id") or ""
    except Exception:  # noqa: BLE001
        return ""


def consult_prompt(payload: dict, prior_decisions: list[str]) -> str:
    kind = str(payload.get("type") or "decision")
    lines = [f"The engine needs YOUR decision as the architect (chip lead) on a `{kind}` interrupt.",
             "You wrote this architecture; decide against it, not against a summary. Read any file you need",
             "(specs, contracts, `coresmith status --json`, `coresmith item show <id>`, block results) before answering.",
             "", "## Interrupt payload", "```json", json.dumps(payload, indent=2, default=str)[:20000], "```"]
    if prior_decisions:
        lines += ["", "## Your prior decisions (most recent last)", *[f"- {d[:600]}" for d in prior_decisions[-10:]]]
    lines += ["", "## Answer", _DECISION_SCHEMA,
              "Rules: `abort` is never yours to choose (a ruling can); prefer `approve` when the evidence holds and `revise`",
              "with block-specific feedback when it does not; a primitive block is regenerated, never hand-fixed."]
    return "\n".join(lines)


def consult(project_root, payload: dict, prior_decisions: list[str] | None = None, *, max_turns: int = 40,
            session: ArchitectSession | None = None) -> dict | None:
    """Resume the architect session for one decision. Returns the decision dict
    (with ``decided_by: "architect"``) or ``None`` to fall back."""
    sid = session_id(project_root)
    if not sid and session is None:
        return None
    sess = session or ArchitectSession(project_root, max_turns=max_turns, max_sittings=1)
    sess.dir = Path(project_root) / ".coresmith" / "architect"
    sess.dir.mkdir(parents=True, exist_ok=True)
    n = len(list(sess.dir.glob("consult-*.md"))) + 1
    prompt = consult_prompt(payload, prior_decisions or [])
    (sess.dir / f"consult-{n}.md").write_text(prompt)
    res = sess.sit(prompt, resume=sid or "", index=1000 + n)
    from orchestrator.langchain.agents.chip_lead_agent import _parse_decision
    try:
        decision = _parse_decision(res.get("text") or "")
    except ValueError:
        return None
    if not isinstance(decision, dict) or not decision.get("action"):
        return None
    decision["decided_by"] = "architect"
    decision["session_id"] = res.get("session_id") or sid
    (sess.dir / f"consult-{n}.decision.json").write_text(json.dumps(decision, indent=2, default=str))
    return decision
