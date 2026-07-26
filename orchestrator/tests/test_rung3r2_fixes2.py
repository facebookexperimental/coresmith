# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the rung-3 repeat-2 escalation-retry Doc-Fix routing livelock
[rung3r2-fixes-2].

Engine defect #3 from the run2 video_codec feasibility rerun: on operator feedback at a
constraint escalation, a MIXED violation set (structural/diagram + doc-sourced)
looped feedback -> Doc Fix (SAD/FRD regen) -> Constraint Check -> escalate
forever. Block Diagram was unreachable, so diagram-directed operator feedback
(port names, tiers) could never land, and 13 byte-identical structural
violations reproduced across 3 rounds. Root causes + fixes:

- Fix 1 (mixed-set routing): ``_escalation_retry_target`` (+ the shared
  ``route_after_constraints`` constraint-loop partition) route to Doc Fix ONLY
  when the violation set is doc-sourced ONLY; any structural/diagram violation
  present -> Block Diagram (threading the operator feedback + violation strings).
- Fix 2 (budget semantics): ``doc_fix_attempts`` is re-armed at most once per
  DISTINCT operator feedback (keyed on a hash of the feedback text), not
  unconditionally -- repeated identical feedback exhausts to Block Diagram.
- Fix 3 (PRD-derived classes): auto_fixable violations rooted in the PRD /
  requirements / ERS are classified escalation-only (Doc Fix re-derives from the
  PRD; the diagram loop cannot clear them either) and surfaced with a clear
  amendment/supersession note in the escalation payload.

Hermetic -- ``interrupt`` is monkeypatched with a canned response; no LLM.
Compatible with ``-m "not live_llm and not requires_nix and not e2e"``.
"""
from __future__ import annotations

import asyncio

import pytest

from orchestrator.langgraph import architecture_graph as ag


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _v(category: str, source_doc: str | None = None, violation: str = "x") -> dict:
    d: dict = {"category": category, "violation": violation}
    if source_doc is not None:
        d["source_doc"] = source_doc
    return d


def _doc() -> dict:
    return _v("auto_fixable", "sad", "SAD pinout summary is stale")


def _frd() -> dict:
    return _v("auto_fixable", "frd", "FRD latency claim is stale")


def _prd() -> dict:
    return _v("auto_fixable", "prd", "PRD mandates a full-frame store")


def _structural() -> dict:
    return _v("structural", "block_diagram", "port pixel_axis undeclared")


def _diagram_autofix() -> dict:
    return _v("auto_fixable", "block_diagram", "tier direction backwards")


# ---------------------------------------------------------------------------
# Fix 3 -- three-way classification (doc-fixable / escalation-only / other)
# ---------------------------------------------------------------------------

class TestClassifyConstraintViolations:
    def test_doc_sources_are_doc_fixable(self):
        doc, esc, oth = ag._classify_constraint_violations([_doc(), _frd()])
        assert len(doc) == 2 and esc == [] and oth == []

    def test_prd_requirements_ers_are_escalation_only(self):
        vs = [
            _v("auto_fixable", "prd"),
            _v("auto_fixable", "requirements"),
            _v("auto_fixable", "ers"),
        ]
        doc, esc, oth = ag._classify_constraint_violations(vs)
        assert doc == [] and len(esc) == 3 and oth == []

    def test_structural_prd_is_other_not_escalation_only(self):
        # escalation-only is auto_fixable ONLY; a structural PRD violation is a
        # normal structural escalation (has_structural drives it), not this class.
        doc, esc, oth = ag._classify_constraint_violations(
            [_v("structural", "prd")]
        )
        assert doc == [] and esc == [] and len(oth) == 1

    def test_diagram_and_unknown_autofix_are_other(self):
        doc, esc, oth = ag._classify_constraint_violations(
            [_diagram_autofix(), _v("auto_fixable", ""), _v("auto_fixable")]
        )
        assert doc == [] and esc == [] and len(oth) == 3

    def test_partition_backcompat_folds_escalation_only_into_other(self):
        # _partition_constraint_violations keeps its 2-way (doc_sourced, other)
        # shape; escalation-only (PRD) folds into "other" (it is NOT doc-fixable).
        doc_sourced, other = ag._partition_constraint_violations(
            [_doc(), _prd(), _structural()]
        )
        assert len(doc_sourced) == 1
        assert len(other) == 2  # prd + structural
        assert ag._partition_constraint_violations(None) == ([], [])


# ---------------------------------------------------------------------------
# Fix 1 -- mixed-set routing (Doc Fix ONLY for doc-sourced-only sets)
# ---------------------------------------------------------------------------

class TestMixedSetRouting:
    def _rac(self, violations, **extra):
        st = {
            "constraint_result": {
                "all_pass": False, "has_structural": False,
                "violations": violations,
            },
            "round": 1,
        }
        st.update(extra)
        return ag.route_after_constraints(st)

    def _retry(self, violations, **extra):
        st = {"constraint_result": {"violations": violations}}
        st.update(extra)
        return ag._escalation_retry_target(st)

    # -- escalation-retry path --
    def test_doc_only_retry_goes_to_doc_fix(self):
        assert self._retry([_doc(), _frd()], doc_fix_attempts=0) == "Doc Fix"

    def test_mixed_doc_plus_structural_retry_goes_to_block_diagram(self):
        # THE defect: a doc-sourced violation coexisting with a structural
        # (diagram) violation must NOT route to Doc Fix.
        assert self._retry([_doc(), _structural()], doc_fix_attempts=0) == "Block Diagram"

    def test_mixed_doc_plus_diagram_autofix_retry_goes_to_block_diagram(self):
        assert self._retry(
            [_doc(), _diagram_autofix()], doc_fix_attempts=0
        ) == "Block Diagram"

    def test_pure_structural_retry_still_block_diagram(self):
        assert self._retry([_structural()]) == "Block Diagram"

    def test_doc_only_over_budget_retry_falls_through_to_block_diagram(self):
        assert self._retry(
            [_doc()], doc_fix_attempts=ag._DOC_FIX_MAX_ATTEMPTS
        ) == "Block Diagram"

    # -- constraint-loop path (route_after_constraints) --
    def test_doc_only_rac_goes_to_doc_fix(self):
        assert self._rac([_doc()], doc_fix_attempts=0) == "Doc Fix"

    def test_mixed_doc_plus_diagram_rac_goes_to_iteration(self):
        # Fix 1 in the constraint-loop partition: a mixed non-structural set
        # spins the diagram loop first instead of dead-ending at Doc Fix.
        assert self._rac(
            [_doc(), _diagram_autofix()], doc_fix_attempts=0
        ) == "Constraint Iteration"

    def test_diagram_only_rac_goes_to_iteration(self):
        assert self._rac([_diagram_autofix()], doc_fix_attempts=0) == "Constraint Iteration"


# ---------------------------------------------------------------------------
# Fix 3 -- PRD-rooted escalation-only routing + payload note
# ---------------------------------------------------------------------------

class TestEscalationOnlyRouting:
    def test_prd_only_rac_escalates_instead_of_spinning_diagram(self):
        # A PRD-rooted-only auto_fixable set can never be cleared by the diagram
        # loop -> surface to the operator (has an Escalate Constraints edge),
        # rather than burning max_rounds to exhaustion.
        st = {
            "constraint_result": {
                "all_pass": False, "has_structural": False,
                "violations": [_prd()],
            },
            "round": 1, "doc_fix_attempts": 0,
        }
        assert ag.route_after_constraints(st) == "Escalate Constraints"

    def test_prd_only_retry_goes_to_block_diagram_no_self_edge(self):
        # On the escalation-retry path there is no Escalate->Escalate edge, so a
        # PRD-only set threads the operator feedback through Block Diagram.
        st = {"constraint_result": {"violations": [_prd()]}, "doc_fix_attempts": 0}
        assert ag._escalation_retry_target(st) == "Block Diagram"

    @pytest.mark.asyncio
    async def test_escalation_payload_notes_prd_rooted_violations(
        self, tmp_path, monkeypatch
    ):
        captured: dict = {}
        monkeypatch.setattr(
            ag, "interrupt",
            lambda payload: captured.update(payload=payload) or {"action": "abort"},
        )
        state = {
            "round": 2, "max_rounds": 3, "project_root": str(tmp_path),
            "memory_map": {}, "violations_history": [],
            "block_diagram": {"blocks": [{"name": "a"}], "connections": []},
            "constraint_result": {"violations": [_prd(), _structural()]},
        }
        await ag.escalate_constraints_node(state)
        payload = captured["payload"]
        assert len(payload["escalation_only_violations"]) == 1
        assert "PRD amendment" in payload["escalation_note"]

    @pytest.mark.asyncio
    async def test_escalation_payload_no_note_when_no_prd_rooted(
        self, tmp_path, monkeypatch
    ):
        captured: dict = {}
        monkeypatch.setattr(
            ag, "interrupt",
            lambda payload: captured.update(payload=payload) or {"action": "abort"},
        )
        state = {
            "round": 2, "max_rounds": 3, "project_root": str(tmp_path),
            "memory_map": {}, "violations_history": [],
            "block_diagram": {"blocks": [{"name": "a"}], "connections": []},
            "constraint_result": {"violations": [_doc(), _structural()]},
        }
        await ag.escalate_constraints_node(state)
        payload = captured["payload"]
        assert payload["escalation_only_violations"] == []
        assert "escalation_note" not in payload


# ---------------------------------------------------------------------------
# Fix 2 -- doc-fix budget re-armed at most once per DISTINCT feedback
# ---------------------------------------------------------------------------

def _escalate_once(monkeypatch, state, feedback):
    """Drive one Escalate Constraints round with a canned feedback response and
    return the state merged with the node's update (as LangGraph would)."""
    monkeypatch.setattr(
        ag, "interrupt",
        lambda payload: {"action": "feedback", "feedback": feedback},
    )
    update = _run(ag.escalate_constraints_node(state))
    merged = dict(state)
    merged.update(update)
    return merged, update


class TestBudgetResetSemantics:
    def _state(self, tmp_path, **extra):
        st = {
            "round": 3, "max_rounds": 3, "project_root": str(tmp_path),
            "memory_map": {}, "violations_history": [],
            "block_diagram": {"blocks": [{"name": "a"}], "connections": []},
            "constraint_result": {"violations": [_doc()]},
            "doc_fix_attempts": 2,
        }
        st.update(extra)
        return st

    def test_first_feedback_rearms_budget(self, tmp_path, monkeypatch):
        st = self._state(tmp_path)
        _merged, update = _escalate_once(monkeypatch, st, "fix the SAD summary")
        assert update["doc_fix_attempts"] == 0
        assert update["doc_fix_reset_key"]  # stamped

    def test_identical_feedback_does_not_rearm_budget(self, tmp_path, monkeypatch):
        st = self._state(tmp_path)
        merged, first = _escalate_once(monkeypatch, st, "fix the SAD summary")
        # carry the reset key forward (as the checkpoint would) + the budget it
        # would have been decremented to by an intervening Doc Fix attempt
        merged["doc_fix_attempts"] = 2
        _merged2, second = _escalate_once(monkeypatch, merged, "fix the SAD summary")
        assert "doc_fix_attempts" not in second   # NOT re-armed on identical feedback
        assert "doc_fix_reset_key" not in second

    def test_distinct_feedback_rearms_again(self, tmp_path, monkeypatch):
        st = self._state(tmp_path)
        merged, _first = _escalate_once(monkeypatch, st, "fix the SAD summary")
        merged["doc_fix_attempts"] = 2
        _merged2, second = _escalate_once(monkeypatch, merged, "different guidance now")
        assert second["doc_fix_attempts"] == 0   # new guidance -> fresh budget

    @pytest.mark.asyncio
    async def test_retry_no_feedback_rearms_once_only(self, tmp_path, monkeypatch):
        # A bare 'retry' (empty feedback) re-arms the FIRST time, not the second.
        monkeypatch.setattr(ag, "interrupt", lambda payload: {"action": "retry"})
        st = self._state(tmp_path)
        u1 = await ag.escalate_constraints_node(st)
        assert u1["doc_fix_attempts"] == 0
        st2 = dict(st)
        st2.update(u1)
        st2["doc_fix_attempts"] = 2
        u2 = await ag.escalate_constraints_node(st2)
        assert "doc_fix_attempts" not in u2   # same (empty) feedback -> no re-arm


# ---------------------------------------------------------------------------
# Livelock regression -- 3 rounds of a MIXED set + identical feedback terminate
# at Block Diagram, never a Doc Fix loop
# ---------------------------------------------------------------------------

class TestLivelockRegression:
    def test_three_rounds_mixed_feedback_never_doc_fix(self, tmp_path, monkeypatch):
        """The run2 defect reproduction: a mixed set (13 structural port-name
        violations + doc-staleness) with the operator sending the SAME feedback
        each round. Pre-fix this looped feedback -> Doc Fix -> Constraint Check
        -> escalate forever and Block Diagram was never reached. Post-fix every
        round routes to Block Diagram (threading the feedback) and the budget is
        re-armed at most once."""
        mixed = [_structural() for _ in range(13)] + [_doc()]
        feedback = "declare pixel_axis on both endpoints; re-tier the monitor"

        state = {
            "round": 2, "max_rounds": 3, "project_root": str(tmp_path),
            "memory_map": {}, "violations_history": [],
            "block_diagram": {
                "blocks": [{"name": "a"}, {"name": "b"}], "connections": [],
            },
            "constraint_result": {
                "all_pass": False, "has_structural": True, "violations": mixed,
            },
            "doc_fix_attempts": 0,
        }

        targets: list[str] = []
        budget_resets = 0
        for _round in range(3):
            merged, update = _escalate_once(monkeypatch, state, feedback)
            if "doc_fix_attempts" in update and update["doc_fix_attempts"] == 0:
                budget_resets += 1
            # feedback is threaded into the state Block Diagram reads
            assert merged["human_feedback"] == feedback
            target = ag.route_after_constraint_escalation(merged)
            targets.append(target)
            # next round: BD "failed" to clear the mixed set -> same violations
            # persist (the defect's byte-identical reproduction); carry state.
            state = merged
            state["constraint_result"] = {
                "all_pass": False, "has_structural": True, "violations": mixed,
            }

        assert targets == ["Block Diagram", "Block Diagram", "Block Diagram"]
        assert "Doc Fix" not in targets           # livelock class eliminated
        assert budget_resets == 1                 # re-armed once, not every round

    def test_mixed_set_reaches_block_diagram_which_owns_the_feedback(self):
        # Sanity: the target Block Diagram is the node that threads human_feedback
        # + constraint violation strings (verified structurally by the router;
        # block_diagram_node reads state.human_feedback + constraint_result).
        st = {
            "human_response": {"action": "feedback", "feedback": "x"},
            "constraint_result": {"violations": [_doc(), _structural()]},
            "doc_fix_attempts": 0,
        }
        assert ag.route_after_constraint_escalation(st) == "Block Diagram"
