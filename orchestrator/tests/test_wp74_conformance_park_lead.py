# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""WP-74: the contract-conformance park goes through ``_resolve_interrupt`` (it
parks; the architect on call can repair the RTL or the contract and answer
`retry`), and the on-call contract documents the park."""
from __future__ import annotations

import asyncio

from orchestrator.langgraph import pipeline_graph as pg


def test_park_asks_the_chip_lead_and_returns_its_decision(tmp_path, monkeypatch):
    seen: dict = {}

    async def fake_resolve(payload):
        seen.update(payload)
        return {"action": "retry", "rationale": "fixed the contract on disk"}

    monkeypatch.setattr(pg, "_resolve_interrupt", fake_resolve)
    state = {"project_root": str(tmp_path)}
    record = {"deviations": ["ambiguous channel 'irq_in (irq)'"], "renames": {},
              "feedback": "expected ports: irq_in"}
    decision = asyncio.run(pg._park_conformance_unrepairable(state, "user_project_wrapper", record, 2))
    assert decision["action"] == "retry"
    assert seen["type"] == "contract_conformance_unrepairable"
    assert seen["block_name"] == "user_project_wrapper"
    assert set(seen["supported_actions"]) >= {"retry", "proceed", "abort"}
    assert seen["deviations"] == record["deviations"]


def test_park_without_a_decision_returns_an_empty_dict(tmp_path, monkeypatch):
    async def fake_resolve(payload):
        return None  # a resolver that parked (interrupt) would raise instead

    monkeypatch.setattr(pg, "_resolve_interrupt", fake_resolve)
    decision = asyncio.run(pg._park_conformance_unrepairable(
        {"project_root": str(tmp_path)}, "blk", {}, 2))
    assert decision == {}
