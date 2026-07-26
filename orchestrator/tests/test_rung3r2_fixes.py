# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the rung-3 repeat-2 video_codec live-run engine fixes [rung3r2-fixes-1].

Defect 1 -- escalate_constraints_node (+ escalate_exhausted_node, its
            route_after_constraints router, and the shared _block_diagram_summary
            helper) AttributeError on present-but-None optional stage outputs.
            The daemon SEEDS memory_map/clock_tree/register_spec/block_diagram/
            constraint_result as explicit None (memory-map stage disabled by
            default), and ``dict.get(key, {})`` does NOT apply the default for a
            present-but-None key -> the escalation payload crashed. Two shapes are
            covered: the run1 crash (only memory_map None, a real constraint_result)
            and the run2 crash (all optional docs None).

Defect 2 -- interface_definition_node wrote constraint_result ONLY on the
            violation path. A clean re-emit after an operator serialization fix
            left the STALE structural result (source=interface_definition) in
            state, and route_after_interface_definition re-diverted to Escalate
            Constraints FOREVER. The clean path now ALWAYS records an explicit
            interface-definition verdict, clearing its own stale result while
            PRESERVING a constraint_result owned by another source.

Hermetic -- the interface LLM specialist is monkeypatched and ``interrupt`` is
replaced with a canned response. Compatible with
`-m "not live_llm and not requires_nix and not e2e"`.
"""
from __future__ import annotations

import asyncio

import pytest

from orchestrator.langgraph import architecture_graph as ag


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Defect 1 -- escalation nodes are None-safe on seeded optional stage outputs
# ---------------------------------------------------------------------------

def _all_none_optional_docs_state(tmp_path) -> dict:
    """A state carrying explicit None for EVERY optional doc, as the daemon
    seeds it when the memory-map/clock-tree/register-spec stages are disabled."""
    return {
        "round": 2,
        "max_rounds": 3,
        "project_root": str(tmp_path),
        "block_diagram": None,
        "memory_map": None,
        "clock_tree": None,
        "register_spec": None,
        "constraint_result": None,
        "violations_history": [],
    }


class TestBlockDiagramSummaryNoneSafe:
    def test_present_but_none_block_diagram(self):
        # dict.get("block_diagram", {}) returns None (not {}) here -> the old
        # code AttributeError'd on None.get("blocks").
        summary = ag._block_diagram_summary({"block_diagram": None})
        assert summary["block_count"] == 0
        assert summary["block_names"] == []
        assert summary["connection_count"] == 0


class TestEscalateConstraintsNodeNoneSafe:
    @pytest.mark.asyncio
    async def test_all_optional_docs_none_builds_payload(self, tmp_path, monkeypatch):
        # run2 shape: block_diagram / memory_map / constraint_result all None.
        captured: dict = {}

        def _fake_interrupt(payload):
            captured["payload"] = payload
            return {"action": "abort"}

        monkeypatch.setattr(ag, "interrupt", _fake_interrupt)

        update = await ag.escalate_constraints_node(
            _all_none_optional_docs_state(tmp_path)
        )
        # No AttributeError -> node reached interrupt() and returned.
        assert update["human_response"] == {"action": "abort"}
        payload = captured["payload"]
        assert payload["memory_map_summary"]["peripheral_count"] == 0
        assert payload["block_diagram_summary"]["block_count"] == 0
        assert payload["violations"] == []

    @pytest.mark.asyncio
    async def test_run1_shape_only_memory_map_none(self, tmp_path, monkeypatch):
        # run1 crash shape: memory_map None but a real constraint_result present
        # (the live crash site was the memory_map peripheral_count deref).
        captured: dict = {}
        monkeypatch.setattr(
            ag, "interrupt",
            lambda payload: captured.update(payload=payload) or {"action": "accept"},
        )
        state = {
            "round": 1, "max_rounds": 3, "project_root": str(tmp_path),
            "block_diagram": {"blocks": [{"name": "a"}], "connections": []},
            "memory_map": None,
            "constraint_result": {
                "violations": [
                    {"category": "structural", "violation": "peripheral mismatch"},
                ],
            },
            "violations_history": [],
        }
        update = await ag.escalate_constraints_node(state)
        assert update["human_response"] == {"action": "accept"}
        payload = captured["payload"]
        assert payload["memory_map_summary"]["peripheral_count"] == 0
        assert len(payload["structural_violations"]) == 1


class TestEscalateExhaustedNodeNoneSafe:
    @pytest.mark.asyncio
    async def test_all_optional_docs_none_builds_payload(self, tmp_path, monkeypatch):
        captured: dict = {}
        monkeypatch.setattr(
            ag, "interrupt",
            lambda payload: captured.update(payload=payload) or {"action": "abort"},
        )
        update = await ag.escalate_exhausted_node(
            _all_none_optional_docs_state(tmp_path)
        )
        assert update["human_response"] == {"action": "abort"}
        assert captured["payload"]["violations"] == []
        assert captured["payload"]["block_diagram_summary"]["block_count"] == 0


class TestRouteAfterConstraintsNoneSafe:
    def test_present_but_none_constraint_result_routes_to_iteration(self):
        # The router derefs constraint_result; a seeded None must not crash and
        # must fall through to the auto-fix iteration (no pass, no structural).
        assert (
            ag.route_after_constraints({"constraint_result": None, "round": 1})
            == "Constraint Iteration"
        )


# ---------------------------------------------------------------------------
# Defect 2 -- clean interface re-emit clears its own stale structural result
# ---------------------------------------------------------------------------

def _patch_specialist(monkeypatch, contract_violations):
    import orchestrator.architecture.specialists.interface_definition as ifd_mod

    async def fake_analyze(**kw):
        return {
            "result": {
                "contracts": [{"producer_block": "a", "consumer_block": "b"}],
                "contract_violations": list(contract_violations),
                "open_questions": [],
            },
            "questions": [],
        }

    monkeypatch.setattr(ifd_mod, "analyze_interface_definition", fake_analyze)


def _base_state(tmp_path, **extra) -> dict:
    state = {
        "project_root": str(tmp_path),
        "round": 2,
        "block_diagram": {
            "blocks": [{"name": "a"}, {"name": "b"}],
            "connections": [{"from": "a", "to": "b"}],
        },
        "requirements": "",
    }
    state.update(extra)
    return state


class TestCleanReemitClearsStaleInterfaceResult:
    def test_violation_round_routes_to_escalation(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CORESMITH_INTERFACE_CONTRACT_GATE", "1")
        viol = [{"edge": "a->b", "type": "over_wide_bus", "category": "structural",
                 "severity": "error", "violation": "too wide",
                 "source": "interface_definition"}]
        _patch_specialist(monkeypatch, viol)

        out = _run(ag.interface_definition_node(_base_state(tmp_path)))
        cr = out["constraint_result"]
        assert cr["has_structural"] is True and cr["source"] == "interface_definition"
        state_after = _base_state(tmp_path)
        state_after.update(out)
        assert ag.route_after_interface_definition(state_after) == "Escalate Constraints"

    def test_clean_reemit_after_stale_interface_result_routes_forward(
        self, monkeypatch, tmp_path
    ):
        """THE loop fix: a clean re-emit following a violation round must clear
        the stale interface-definition structural result and route forward to
        Memory Map instead of re-diverting to Escalate Constraints forever."""
        monkeypatch.setenv("CORESMITH_INTERFACE_CONTRACT_GATE", "1")
        _patch_specialist(monkeypatch, [])  # clean re-emit

        stale = {
            "all_pass": False, "has_structural": True,
            "violations": [{"violation": "round-1 over_wide_bus"}],
            "source": "interface_definition",
        }
        out = _run(ag.interface_definition_node(
            _base_state(tmp_path, constraint_result=stale)
        ))
        cr = out["constraint_result"]
        assert cr["all_pass"] is True
        assert cr["has_structural"] is False
        assert cr["violations"] == []
        assert cr["source"] == "interface_definition"

        state_after = _base_state(tmp_path, constraint_result=stale)
        state_after.update(out)
        assert ag.route_after_interface_definition(state_after) == "Memory Map"

    def test_clean_reemit_preserves_other_source_violations(
        self, monkeypatch, tmp_path
    ):
        """A constraint_result owned by ANOTHER source (Constraint Check) is
        preserved untouched -- only the interface router's own source is
        filtered/cleared."""
        monkeypatch.setenv("CORESMITH_INTERFACE_CONTRACT_GATE", "1")
        _patch_specialist(monkeypatch, [])

        other = {
            "all_pass": False, "has_structural": True,
            "violations": [{"violation": "peripheral mismatch"}],
            "source": "constraint_check",
        }
        out = _run(ag.interface_definition_node(
            _base_state(tmp_path, constraint_result=other)
        ))
        # node did not clobber the other-source result
        assert "constraint_result" not in out
        state_after = _base_state(tmp_path, constraint_result=other)
        state_after.update(out)
        # interface router only diverts on source==interface_definition
        assert ag.route_after_interface_definition(state_after) == "Memory Map"
        # the other-source violations are still in state for the downstream gate
        assert state_after["constraint_result"]["source"] == "constraint_check"
        assert len(state_after["constraint_result"]["violations"]) == 1

    def test_gate_off_clean_is_noop(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CORESMITH_INTERFACE_CONTRACT_GATE", "0")
        _patch_specialist(monkeypatch, [])
        stale = {
            "all_pass": False, "has_structural": True,
            "violations": [{"violation": "x"}],
            "source": "interface_definition",
        }
        out = _run(ag.interface_definition_node(
            _base_state(tmp_path, constraint_result=stale)
        ))
        # gate off -> node must not touch constraint_result at all
        assert "constraint_result" not in out
