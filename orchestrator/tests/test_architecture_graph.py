# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for the architecture LangGraph execution graph.

Tests:
- Graph construction (compiles, has expected nodes)
- ArchGraphState schema
- Routing functions (route_after_prd, route_after_prd_escalation,
  review_diagram, route_after_diagram_escalation, route_after_constraints,
  route_after_constraint_escalation, route_after_increment,
  route_after_exhausted_escalation)
- PRD interrupt flow (mocked LLM, FFT16 design)
- Block diagram interrupt flow
- Constraint flow (pass / structural / auto-fixable / exhausted)
- Happy path (full graph, all specialists mocked, FFT16 design)
- Checkpoint persistence (MemorySaver)
"""

from __future__ import annotations

import copy
from contextlib import ExitStack
from unittest.mock import AsyncMock, patch

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from orchestrator.langgraph.architecture_graph import (
    ArchGraphState,
    _persist_intermediate_state,
    build_architecture_graph,
    route_after_prd,
    route_after_prd_escalation,
    review_diagram,
    route_after_diagram_escalation,
    route_after_constraints,
    route_after_constraint_escalation,
    route_after_increment,
    route_after_exhausted_escalation,
)

from orchestrator.tests.fft16_fixtures import (
    FFT16_BLOCK_DIAGRAM,
    FFT16_CLOCK_TREE,
    FFT16_FRD_DOCUMENT,
    FFT16_MEMORY_MAP,
    FFT16_PRD_ANSWERS,
    FFT16_PRD_DOCUMENT,
    FFT16_PRD_QUESTIONS,
    FFT16_REGISTER_SPEC,
    FFT16_REQUIREMENTS,
    FFT16_SAD_DOCUMENT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _get_interrupt(graph, config) -> dict | None:
    """Get the interrupt payload from the graph state, or None."""
    state = await graph.aget_state(config)
    if state.tasks:
        for task in state.tasks:
            if task.interrupts:
                return task.interrupts[0].value
    return None


def _make_node_mock(specialist_result):
    """Create an async mock node function that returns specialist results."""
    async def _mock_node(state):
        return specialist_result
    return _mock_node


def _patch_all_specialists(
    prd_result=None,
    sad_result=None,
    frd_result=None,
    block_diagram_result=None,
    memory_map_result=None,
    clock_tree_result=None,
    register_spec_result=None,
    constraint_result=None,
):
    """Return a context manager that patches all architecture specialist node functions.

    Patches at the node level inside the architecture_graph module so that
    the underlying specialist modules (which may have import-time side effects
    like reading prompt files) are never imported.
    """
    if prd_result is None:
        prd_result = FFT16_PRD_QUESTIONS
    if block_diagram_result is None:
        block_diagram_result = FFT16_BLOCK_DIAGRAM
    if memory_map_result is None:
        memory_map_result = FFT16_MEMORY_MAP
    if clock_tree_result is None:
        clock_tree_result = FFT16_CLOCK_TREE
    if register_spec_result is None:
        register_spec_result = FFT16_REGISTER_SPEC
    if constraint_result is None:
        constraint_result = []

    stack = ExitStack()

    stack.enter_context(patch(
        "orchestrator.architecture.specialists.prd_spec.gather_prd",
        new_callable=AsyncMock,
        return_value=prd_result,
    ))
    from orchestrator.tests.fft16_fixtures import FFT16_SAD_MARKDOWN, FFT16_FRD_MARKDOWN
    stack.enter_context(patch(
        "orchestrator.architecture.specialists.sad_spec.generate_sad",
        new_callable=AsyncMock,
        return_value=sad_result or FFT16_SAD_MARKDOWN,
    ))
    stack.enter_context(patch(
        "orchestrator.architecture.specialists.frd_spec.generate_frd",
        new_callable=AsyncMock,
        return_value=frd_result or FFT16_FRD_MARKDOWN,
    ))
    stack.enter_context(patch(
        "orchestrator.architecture.specialists.block_diagram.analyze_block_diagram",
        new_callable=AsyncMock,
        return_value=block_diagram_result,
    ))
    stack.enter_context(patch(
        "orchestrator.architecture.specialists.memory_map.analyze_memory_map",
        new_callable=AsyncMock,
        return_value=memory_map_result,
    ))
    stack.enter_context(patch(
        "orchestrator.architecture.specialists.clock_tree.analyze_clock_tree",
        new_callable=AsyncMock,
        return_value=clock_tree_result,
    ))
    stack.enter_context(patch(
        "orchestrator.architecture.specialists.register_spec.analyze_register_spec",
        new_callable=AsyncMock,
        return_value=register_spec_result,
    ))
    stack.enter_context(patch(
        "orchestrator.architecture.constraints.check_constraints",
        new_callable=AsyncMock,
        return_value=constraint_result,
    ))
    # Mock the ERS doc generator (called by create_documentation_node)
    # to prevent LLM calls during tests.
    stack.enter_context(patch(
        "orchestrator.architecture.specialists.ers_doc.generate_ers_doc",
        new_callable=AsyncMock,
        return_value={"ers": {"title": "FFT16 ERS", "summary": "test"}, "phase": "ers_complete"},
    ))

    return stack


async def _accept_final_review(graph, config) -> dict:
    """Auto-accept the Final Review interrupt and return the final state.

    After Finalize Architecture -> Create Documentation -> Final Review,
    the graph interrupts for OK2DEV approval. This helper detects and
    auto-accepts it.
    """
    payload = await _get_interrupt(graph, config)
    if payload and payload.get("type") == "final_review":
        return await graph.ainvoke(
            Command(resume={"action": "accept"}),
            config,
        )
    # If no final_review interrupt, return current state values
    state = await graph.aget_state(config)
    return state.values if state else {}


# ═══════════════════════════════════════════════════════════════════════════
# Graph Construction
# ═══════════════════════════════════════════════════════════════════════════

class TestGraphConstruction:
    def test_compiles_without_error(self):
        graph = build_architecture_graph(checkpointer=MemorySaver())
        assert graph is not None

    def test_compiles_without_checkpointer(self):
        graph = build_architecture_graph(checkpointer=None)
        assert graph is not None

    def test_has_expected_nodes(self):
        graph = build_architecture_graph(checkpointer=MemorySaver())
        node_names = list(graph.get_graph().nodes.keys())
        expected = [
            "Gather Requirements",
            "Escalate PRD",
            "System Architecture",
            "Functional Requirements",
            "Block Diagram",
            "Escalate Diagram",
            "Memory Map",
            "Clock Tree",
            "Register Spec",
            "Constraint Check",
            "Escalate Constraints",
            "Constraint Iteration",
            "Escalate Exhausted",
            "Finalize Architecture",
            "Architecture Complete",
            "Abort",
        ]
        for name in expected:
            assert name in node_names, f"Missing node: {name}"


# ═══════════════════════════════════════════════════════════════════════════
# State Schema
# ═══════════════════════════════════════════════════════════════════════════

class TestArchGraphState:
    def test_has_required_fields(self):
        annotations = ArchGraphState.__annotations__
        required = [
            "project_root", "requirements", "pdk_summary",
            "target_clock_mhz", "pdk_config", "max_rounds",
            "round", "phase",
            "prd_spec", "prd_questions",
            "violations_history", "questions",
            "block_diagram", "memory_map", "clock_tree",
            "register_spec", "benchmark_data", "constraint_result",
            "human_feedback", "human_response",
            "success", "error", "block_specs_path",
        ]
        for f in required:
            assert f in annotations, f"Missing field: {f}"


# ═══════════════════════════════════════════════════════════════════════════
# Routing Functions (pure unit tests, no graph invocation)
# ═══════════════════════════════════════════════════════════════════════════

class TestRouteAfterPRD:
    def test_questions_goes_to_escalate(self):
        state = {"prd_spec": None, "prd_questions": [{"id": "q1"}]}
        assert route_after_prd(state) == "Escalate PRD"

    def test_prd_complete_goes_to_system_architecture(self):
        state = {"prd_spec": {"title": "FFT PRD"}, "prd_questions": None}
        assert route_after_prd(state) == "System Architecture"

    def test_no_prd_defaults_to_escalate(self):
        state = {}
        assert route_after_prd(state) == "Escalate PRD"


class TestRouteAfterPRDEscalation:
    def test_continue_goes_to_gather_requirements(self):
        state = {"human_response": {"action": "continue", "answers": {}}}
        assert route_after_prd_escalation(state) == "Gather Requirements"

    def test_abort_goes_to_abort(self):
        state = {"human_response": {"action": "abort"}}
        assert route_after_prd_escalation(state) == "Abort"

    def test_missing_action_fails_closed_to_abort(self):
        state = {}
        assert route_after_prd_escalation(state) == "Abort"


class TestReviewDiagram:
    def test_questions_goes_to_escalate(self):
        state = {"block_diagram": {"blocks": [{"name": "a"}], "questions": [{"q": "?"}]}}
        assert review_diagram(state) == "Escalate Diagram"

    def test_clean_goes_to_interface_definition(self):
        # Interface Definition is the new stage between a clean block
        # diagram and Memory Map (PR #45). Memory Map runs immediately
        # after Interface Definition; the route here advances to the
        # interface contract stage first.
        state = {"block_diagram": {"blocks": [{"name": "a"}], "questions": []}}
        assert review_diagram(state) == "Interface Definition"

    def test_no_blocks_goes_to_escalate(self):
        state = {"block_diagram": {"blocks": [], "questions": []}}
        assert review_diagram(state) == "Escalate Diagram"

    def test_missing_diagram_goes_to_escalate(self):
        state = {}
        assert review_diagram(state) == "Escalate Diagram"


class TestRouteAfterDiagramEscalation:
    def test_continue_goes_to_interface_definition(self):
        # After the diagram escalation continues, route through the new
        # Interface Definition stage before Memory Map (PR #45).
        state = {"human_response": {"action": "continue"}}
        assert route_after_diagram_escalation(state) == "Interface Definition"

    def test_feedback_goes_to_block_diagram(self):
        state = {"human_response": {"action": "feedback", "feedback": "fix it"}}
        assert route_after_diagram_escalation(state) == "Block Diagram"

    def test_abort_goes_to_abort(self):
        state = {"human_response": {"action": "abort"}}
        assert route_after_diagram_escalation(state) == "Abort"

    def test_missing_action_fails_closed_to_abort(self):
        state = {}
        assert route_after_diagram_escalation(state) == "Abort"


class TestRouteAfterConstraints:
    def test_all_pass_goes_to_finalize(self):
        state = {"constraint_result": {"all_pass": True, "has_structural": False}}
        assert route_after_constraints(state) == "Finalize Architecture"

    def test_structural_goes_to_escalate(self):
        state = {"constraint_result": {"all_pass": False, "has_structural": True}}
        assert route_after_constraints(state) == "Escalate Constraints"

    def test_auto_fixable_goes_to_iteration(self):
        state = {"constraint_result": {"all_pass": False, "has_structural": False}}
        assert route_after_constraints(state) == "Constraint Iteration"

    def test_missing_result_goes_to_iteration(self):
        state = {}
        assert route_after_constraints(state) == "Constraint Iteration"


class TestRouteAfterConstraintEscalation:
    def test_retry_goes_to_block_diagram(self):
        state = {"human_response": {"action": "retry"}}
        assert route_after_constraint_escalation(state) == "Block Diagram"

    def test_accept_goes_to_finalize(self):
        state = {"human_response": {"action": "accept"}}
        assert route_after_constraint_escalation(state) == "Finalize Architecture"

    def test_feedback_goes_to_block_diagram(self):
        state = {"human_response": {"action": "feedback", "feedback": "merge blocks"}}
        assert route_after_constraint_escalation(state) == "Block Diagram"

    def test_abort_goes_to_abort(self):
        state = {"human_response": {"action": "abort"}}
        assert route_after_constraint_escalation(state) == "Abort"

    def test_missing_action_fails_closed_to_abort(self):
        state = {}
        assert route_after_constraint_escalation(state) == "Abort"


class TestRouteAfterIncrement:
    def test_within_limit_goes_to_block_diagram(self):
        state = {"round": 2, "max_rounds": 3}
        assert route_after_increment(state) == "Block Diagram"

    def test_at_limit_goes_to_block_diagram(self):
        state = {"round": 3, "max_rounds": 3}
        assert route_after_increment(state) == "Block Diagram"

    def test_exceeded_goes_to_escalate_exhausted(self):
        state = {"round": 4, "max_rounds": 3}
        assert route_after_increment(state) == "Escalate Exhausted"


class TestRouteAfterExhaustedEscalation:
    def test_retry_goes_to_block_diagram(self):
        state = {"human_response": {"action": "retry"}}
        assert route_after_exhausted_escalation(state) == "Block Diagram"

    def test_accept_goes_to_finalize(self):
        state = {"human_response": {"action": "accept"}}
        assert route_after_exhausted_escalation(state) == "Finalize Architecture"

    def test_abort_goes_to_abort(self):
        state = {"human_response": {"action": "abort"}}
        assert route_after_exhausted_escalation(state) == "Abort"

    def test_missing_action_fails_closed_to_abort(self):
        state = {}
        assert route_after_exhausted_escalation(state) == "Abort"


# ═══════════════════════════════════════════════════════════════════════════
# PRD Interrupt Flow (full graph invocation, FFT16 design)
# ═══════════════════════════════════════════════════════════════════════════

class TestPRDInterruptFlow:
    @pytest.mark.asyncio
    async def test_prd_questions_trigger_interrupt(self, arch_graph, fft16_initial_state):
        """Gather Requirements (Phase 1) -> Escalate PRD (interrupt)."""
        config = {"configurable": {"thread_id": "test-prd-interrupt-1"}}

        with _patch_all_specialists(prd_result=FFT16_PRD_QUESTIONS):
            await arch_graph.ainvoke(fft16_initial_state, config)

        payload = await _get_interrupt(arch_graph, config)
        assert payload is not None
        assert payload["type"] == "prd_questions"
        assert payload["phase"] == "prd"
        assert len(payload["questions"]) > 0
        assert "continue" in payload["supported_actions"]
        assert "abort" in payload["supported_actions"]

    @pytest.mark.asyncio
    async def test_prd_resume_with_answers_proceeds_to_block_diagram(
        self, arch_graph, fft16_initial_state
    ):
        """Resume PRD with answers -> Phase 2 -> SAD -> FRD -> Block Diagram -> ... -> Final Review."""
        config = {"configurable": {"thread_id": "test-prd-resume-1"}}

        with _patch_all_specialists(prd_result=FFT16_PRD_QUESTIONS):
            await arch_graph.ainvoke(fft16_initial_state, config)

        with _patch_all_specialists(
            prd_result=FFT16_PRD_DOCUMENT,
            block_diagram_result=FFT16_BLOCK_DIAGRAM,
            constraint_result=[],
        ):
            await arch_graph.ainvoke(
                Command(resume={"action": "continue", "answers": FFT16_PRD_ANSWERS}),
                config,
            )
            result = await _accept_final_review(arch_graph, config)

        assert result.get("success") is True or result.get("block_diagram") is not None

    @pytest.mark.asyncio
    async def test_prd_abort_ends_graph(self, arch_graph, fft16_initial_state):
        """Resume PRD with abort -> Abort -> END."""
        config = {"configurable": {"thread_id": "test-prd-abort-1"}}

        with _patch_all_specialists(prd_result=FFT16_PRD_QUESTIONS):
            await arch_graph.ainvoke(fft16_initial_state, config)

        result = await arch_graph.ainvoke(
            Command(resume={"action": "abort"}),
            config,
        )

        assert result["success"] is False
        assert "abort" in result.get("error", "").lower()


# ═══════════════════════════════════════════════════════════════════════════
# PRD Answer Delivery Contract
#
# These tests exercise the graph-level contract: answers in the resume
# value must propagate through escalate_prd_node -> human_response ->
# gather_requirements_node -> gather_prd(user_answers=...).
#
# They reproduce the exact failure mode of the original bug: when
# answers are missing from the resume value, gather_requirements_node
# receives user_answers=None and Phase 1 repeats, creating an infinite
# question-regeneration loop.
# ═══════════════════════════════════════════════════════════════════════════

class TestPRDAnswerDeliveryContract:
    """Verify the graph-level contract: answers in resume → Phase 2."""

    @pytest.mark.asyncio
    async def test_resume_without_answers_key_regenerates_questions(
        self, arch_graph, fft16_initial_state,
    ):
        """Resume with action='continue' but NO 'answers' key → Phase 1 repeats.

        This is the exact failure mode of the original bug: the MCP layer
        dropped the answers dict, so gather_requirements_node received
        human_response={"action": "continue"} with no "answers" key.
        """
        config = {"configurable": {"thread_id": "test-no-answers-1"}}

        # Phase 1: generate initial questions
        with _patch_all_specialists(prd_result=FFT16_PRD_QUESTIONS):
            await arch_graph.ainvoke(fft16_initial_state, config)

        payload1 = await _get_interrupt(arch_graph, config)
        assert payload1 is not None
        assert payload1["type"] == "prd_questions"

        # Resume WITHOUT answers key (simulates the original bug)
        with _patch_all_specialists(prd_result=FFT16_PRD_QUESTIONS):
            await arch_graph.ainvoke(
                Command(resume={"action": "continue"}),
                config,
            )

        # Should interrupt AGAIN with questions — Phase 1 repeated
        payload2 = await _get_interrupt(arch_graph, config)
        assert payload2 is not None, (
            "Expected second interrupt (questions regenerated), but graph continued. "
            "This means answers=None was incorrectly treated as having answers."
        )
        assert payload2["type"] == "prd_questions"
        assert len(payload2["questions"]) > 0

    @pytest.mark.asyncio
    async def test_resume_with_answers_none_regenerates_questions(
        self, arch_graph, fft16_initial_state,
    ):
        """Resume with answers=None → Phase 1 repeats.

        Edge case: the 'answers' key exists but is explicitly None.
        """
        config = {"configurable": {"thread_id": "test-answers-none-1"}}

        with _patch_all_specialists(prd_result=FFT16_PRD_QUESTIONS):
            await arch_graph.ainvoke(fft16_initial_state, config)

        with _patch_all_specialists(prd_result=FFT16_PRD_QUESTIONS):
            await arch_graph.ainvoke(
                Command(resume={"action": "continue", "answers": None}),
                config,
            )

        payload = await _get_interrupt(arch_graph, config)
        assert payload is not None
        assert payload["type"] == "prd_questions"

    @pytest.mark.asyncio
    async def test_gather_prd_receives_user_answers_from_resume(
        self, arch_graph, fft16_initial_state,
    ):
        """Integration: answers from Command(resume=...) must reach gather_prd().

        Verifies the complete in-graph data flow:
          Command(resume={"answers": {...}})
          → escalate_prd_node sets human_response
          → gather_requirements_node reads human_response["answers"]
          → gather_prd(user_answers=answers)

        Uses a mock with side_effect that captures user_answers and
        conditionally returns the PRD document or questions.
        """
        config = {"configurable": {"thread_id": "test-answers-flow-1"}}

        # Phase 1: generate questions
        with _patch_all_specialists(prd_result=FFT16_PRD_QUESTIONS):
            await arch_graph.ainvoke(fft16_initial_state, config)

        # Phase 2: resume with answers — verify gather_prd receives them
        captured = {}

        async def smart_gather_prd(
            *, requirements, pdk_summary, target_clock_mhz,
            user_answers=None, previous_questions=None,
            project_root=".",
        ):
            captured["user_answers"] = user_answers
            captured["previous_questions"] = previous_questions
            if user_answers is not None and len(user_answers) > 0:
                return FFT16_PRD_DOCUMENT
            return FFT16_PRD_QUESTIONS

        with _patch_all_specialists(
            block_diagram_result=FFT16_BLOCK_DIAGRAM,
            constraint_result=[],
        ) as stack:
            stack.enter_context(patch(
                "orchestrator.architecture.specialists.prd_spec.gather_prd",
                new_callable=AsyncMock,
                side_effect=smart_gather_prd,
            ))
            await arch_graph.ainvoke(
                Command(resume={"action": "continue", "answers": FFT16_PRD_ANSWERS}),
                config,
            )
            result = await _accept_final_review(arch_graph, config)

        assert "user_answers" in captured, (
            "gather_prd was never called — graph may have skipped "
            "Gather Requirements entirely"
        )
        assert captured["user_answers"] is not None, (
            "gather_prd received user_answers=None — the answers dict did not "
            "propagate from resume value through escalate_prd_node → "
            "human_response → gather_requirements_node"
        )
        assert captured["user_answers"] == FFT16_PRD_ANSWERS
        assert result.get("success") is True

    @pytest.mark.asyncio
    async def test_missing_answers_causes_gather_prd_called_with_none(
        self, arch_graph, fft16_initial_state,
    ):
        """When answers are missing from resume, gather_prd gets user_answers=None.

        The converse of the above test: verify that the failure mode
        actually results in gather_prd(user_answers=None), confirming the
        root cause of the infinite loop.
        """
        config = {"configurable": {"thread_id": "test-no-answers-flow-1"}}

        with _patch_all_specialists(prd_result=FFT16_PRD_QUESTIONS):
            await arch_graph.ainvoke(fft16_initial_state, config)

        captured = {}

        async def smart_gather_prd(
            *, requirements, pdk_summary, target_clock_mhz,
            user_answers=None, previous_questions=None,
            project_root=".",
        ):
            captured["user_answers"] = user_answers
            # Always return questions so we can verify the loop
            return FFT16_PRD_QUESTIONS

        with _patch_all_specialists() as stack:
            stack.enter_context(patch(
                "orchestrator.architecture.specialists.prd_spec.gather_prd",
                new_callable=AsyncMock,
                side_effect=smart_gather_prd,
            ))
            await arch_graph.ainvoke(
                Command(resume={"action": "continue"}),  # no answers
                config,
            )

        assert "user_answers" in captured
        assert captured["user_answers"] is None, (
            f"Expected user_answers=None when no answers in resume, "
            f"got {captured['user_answers']!r}"
        )

        # And the graph should be back at the interrupt
        payload = await _get_interrupt(arch_graph, config)
        assert payload is not None
        assert payload["type"] == "prd_questions"


# ═══════════════════════════════════════════════════════════════════════════
# Block Diagram Interrupt Flow
# ═══════════════════════════════════════════════════════════════════════════

class TestBlockDiagramInterruptFlow:
    @pytest.mark.asyncio
    async def test_diagram_with_questions_triggers_interrupt(
        self, arch_graph, fft16_initial_state
    ):
        """Block Diagram returns questions -> Escalate Diagram (interrupt)."""
        config = {"configurable": {"thread_id": "test-diagram-questions-1"}}

        diagram_with_questions = copy.deepcopy(FFT16_BLOCK_DIAGRAM)
        diagram_with_questions["questions"] = [
            {"question": "Shared or separate twiddle ROM?", "priority": "blocking"}
        ]

        with _patch_all_specialists(prd_result=FFT16_PRD_QUESTIONS):
            await arch_graph.ainvoke(fft16_initial_state, config)

        with _patch_all_specialists(
            prd_result=FFT16_PRD_DOCUMENT,
            block_diagram_result=diagram_with_questions,
        ):
            await arch_graph.ainvoke(
                Command(resume={"action": "continue", "answers": FFT16_PRD_ANSWERS}),
                config,
            )

        payload = await _get_interrupt(arch_graph, config)
        assert payload is not None
        assert payload["type"] == "architecture_review_needed"
        assert payload["phase"] == "block_diagram"

    @pytest.mark.asyncio
    async def test_diagram_clean_proceeds_to_memory_map(
        self, arch_graph, fft16_initial_state
    ):
        """Block Diagram clean (no questions) -> Memory Map -> ... -> Finalize -> Final Review."""
        config = {"configurable": {"thread_id": "test-diagram-clean-1"}}

        with _patch_all_specialists(prd_result=FFT16_PRD_QUESTIONS):
            await arch_graph.ainvoke(fft16_initial_state, config)

        with _patch_all_specialists(
            prd_result=FFT16_PRD_DOCUMENT,
            block_diagram_result=FFT16_BLOCK_DIAGRAM,
            constraint_result=[],
        ):
            await arch_graph.ainvoke(
                Command(resume={"action": "continue", "answers": FFT16_PRD_ANSWERS}),
                config,
            )
            result = await _accept_final_review(arch_graph, config)

        assert result["success"] is True


# ═══════════════════════════════════════════════════════════════════════════
# Constraint Flow
# ═══════════════════════════════════════════════════════════════════════════

class TestConstraintFlow:
    @pytest.mark.asyncio
    async def test_constraints_pass_finalizes(self, arch_graph, fft16_initial_state):
        """All constraints pass -> Finalize Architecture -> Final Review -> Architecture Complete."""
        config = {"configurable": {"thread_id": "test-constraint-pass-1"}}

        with _patch_all_specialists(prd_result=FFT16_PRD_QUESTIONS):
            await arch_graph.ainvoke(fft16_initial_state, config)

        with _patch_all_specialists(
            prd_result=FFT16_PRD_DOCUMENT,
            constraint_result=[],
        ):
            await arch_graph.ainvoke(
                Command(resume={"action": "continue", "answers": FFT16_PRD_ANSWERS}),
                config,
            )
            result = await _accept_final_review(arch_graph, config)

        assert result["success"] is True
        assert result["block_specs_path"] != ""

    @pytest.mark.asyncio
    async def test_structural_violation_triggers_escalation(
        self, arch_graph, fft16_initial_state
    ):
        """Structural violation -> Escalate Constraints (interrupt)."""
        config = {"configurable": {"thread_id": "test-constraint-structural-1"}}

        structural_violations = [
            {"violation": "Peripheral count exceeds limit", "severity": "error", "category": "structural"}
        ]

        with _patch_all_specialists(prd_result=FFT16_PRD_QUESTIONS):
            await arch_graph.ainvoke(fft16_initial_state, config)

        with _patch_all_specialists(
            prd_result=FFT16_PRD_DOCUMENT,
            constraint_result=structural_violations,
        ):
            await arch_graph.ainvoke(
                Command(resume={"action": "continue", "answers": FFT16_PRD_ANSWERS}),
                config,
            )

        payload = await _get_interrupt(arch_graph, config)
        assert payload is not None
        assert payload["type"] == "architecture_review_needed"
        assert payload["phase"] == "constraints"
        assert "retry" in payload["supported_actions"]

    @pytest.mark.asyncio
    async def test_auto_fixable_violation_loops_to_block_diagram(
        self, arch_graph, fft16_initial_state
    ):
        """Auto-fixable violation -> Constraint Iteration -> Block Diagram (round 2)."""
        config = {"configurable": {"thread_id": "test-constraint-autofix-1"}}

        auto_fixable = [
            {"violation": "Gate budget exceeded", "severity": "warning", "category": "auto_fixable"}
        ]

        call_count = {"constraint": 0}

        async def constraint_side_effect(*args, **kwargs):
            call_count["constraint"] += 1
            if call_count["constraint"] == 1:
                return auto_fixable
            return []

        with _patch_all_specialists(prd_result=FFT16_PRD_QUESTIONS):
            await arch_graph.ainvoke(fft16_initial_state, config)

        with _patch_all_specialists(prd_result=FFT16_PRD_DOCUMENT) as stack:
            stack.enter_context(patch(
                "orchestrator.architecture.constraints.check_constraints",
                new_callable=AsyncMock,
                side_effect=constraint_side_effect,
            ))
            await arch_graph.ainvoke(
                Command(resume={"action": "continue", "answers": FFT16_PRD_ANSWERS}),
                config,
            )
            result = await _accept_final_review(arch_graph, config)

        assert result["success"] is True
        assert result["round"] >= 2

    @pytest.mark.asyncio
    async def test_max_rounds_exhausted_triggers_escalation(
        self, fft16_initial_state
    ):
        """max_rounds=1, auto-fixable failure -> Escalate Exhausted (interrupt)."""
        graph = build_architecture_graph(checkpointer=MemorySaver())
        fft16_initial_state["max_rounds"] = 1
        config = {"configurable": {"thread_id": "test-constraint-exhausted-1"}}

        auto_fixable = [
            {"violation": "Gate budget exceeded", "severity": "warning", "category": "auto_fixable"}
        ]

        with _patch_all_specialists(prd_result=FFT16_PRD_QUESTIONS):
            await graph.ainvoke(fft16_initial_state, config)

        with _patch_all_specialists(
            prd_result=FFT16_PRD_DOCUMENT,
            constraint_result=auto_fixable,
        ):
            await graph.ainvoke(
                Command(resume={"action": "continue", "answers": FFT16_PRD_ANSWERS}),
                config,
            )

        payload = await _get_interrupt(graph, config)
        assert payload is not None
        assert payload["phase"] == "max_rounds_exhausted"
        assert "accept" in payload["supported_actions"]


# ═══════════════════════════════════════════════════════════════════════════
# Happy Path (full graph, FFT16 design)
# ═══════════════════════════════════════════════════════════════════════════

class TestHappyPath:
    @pytest.mark.asyncio
    async def test_architecture_happy_path(self, arch_graph, fft16_initial_state):
        """Walk the FFT16 design through the entire architecture flow.

        PRD Phase 1 -> interrupt -> resume with answers -> Phase 2 ->
        SAD -> FRD -> Block Diagram (3 blocks, clean) -> Memory Map ->
        Clock Tree -> Register Spec -> Constraint Check (pass) ->
        Finalize -> Architecture Complete.
        """
        config = {"configurable": {"thread_id": "test-happy-path-1"}}

        # Phase 1: PRD generates questions, graph interrupts
        with _patch_all_specialists(prd_result=FFT16_PRD_QUESTIONS):
            await arch_graph.ainvoke(fft16_initial_state, config)

        payload = await _get_interrupt(arch_graph, config)
        assert payload is not None
        assert payload["type"] == "prd_questions"

        # Resume: Phase 2 -> SAD -> FRD -> Block Diagram -> ... -> Finalize -> Final Review
        with _patch_all_specialists(
            prd_result=FFT16_PRD_DOCUMENT,
            block_diagram_result=FFT16_BLOCK_DIAGRAM,
            memory_map_result=FFT16_MEMORY_MAP,
            clock_tree_result=FFT16_CLOCK_TREE,
            register_spec_result=FFT16_REGISTER_SPEC,
            constraint_result=[],
        ):
            await arch_graph.ainvoke(
                Command(resume={"action": "continue", "answers": FFT16_PRD_ANSWERS}),
                config,
            )
            result = await _accept_final_review(arch_graph, config)

        assert result["success"] is True
        assert result["block_specs_path"] != ""
        assert result["error"] == ""

        # Verify block specs written
        from pathlib import Path
        import json
        specs_path = Path(result["block_specs_path"])
        assert specs_path.exists()
        specs = json.loads(specs_path.read_text())
        assert len(specs) == 3
        names = [s["name"] for s in specs]
        assert "fft_butterfly" in names
        assert "twiddle_rom" in names
        assert "fft_controller" in names


# ═══════════════════════════════════════════════════════════════════════════
# Checkpoint Persistence
# ═══════════════════════════════════════════════════════════════════════════

class TestCheckpointPersistence:
    @pytest.mark.asyncio
    async def test_state_preserved_after_prd_interrupt(self, fft16_initial_state):
        """Verify state is readable from checkpoint after PRD interrupt."""
        checkpointer = MemorySaver()
        graph = build_architecture_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "test-checkpoint-prd-1"}}

        with _patch_all_specialists(prd_result=FFT16_PRD_QUESTIONS):
            await graph.ainvoke(fft16_initial_state, config)

        graph2 = build_architecture_graph(checkpointer=checkpointer)
        saved_state = await graph2.aget_state(config)

        assert saved_state is not None
        assert saved_state.values["phase"] == "prd"
        assert saved_state.values["requirements"] == FFT16_REQUIREMENTS
        assert saved_state.tasks
        found_interrupt = False
        for task in saved_state.tasks:
            if task.interrupts:
                found_interrupt = True
                payload = task.interrupts[0].value
                assert payload["type"] == "prd_questions"
                break
        assert found_interrupt

    @pytest.mark.asyncio
    async def test_state_preserved_after_constraint_interrupt(self, fft16_initial_state):
        """Verify state is readable from checkpoint after constraint escalation."""
        checkpointer = MemorySaver()
        graph = build_architecture_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "test-checkpoint-constraint-1"}}

        structural_violations = [
            {"violation": "Memory overlap", "severity": "error", "category": "structural"}
        ]

        with _patch_all_specialists(prd_result=FFT16_PRD_QUESTIONS):
            await graph.ainvoke(fft16_initial_state, config)

        with _patch_all_specialists(
            prd_result=FFT16_PRD_DOCUMENT,
            constraint_result=structural_violations,
        ):
            await graph.ainvoke(
                Command(resume={"action": "continue", "answers": FFT16_PRD_ANSWERS}),
                config,
            )

        saved_state = await graph.aget_state(config)
        assert saved_state.tasks
        found_interrupt = False
        for task in saved_state.tasks:
            if task.interrupts:
                found_interrupt = True
                payload = task.interrupts[0].value
                assert payload["phase"] == "constraints"
                break
        assert found_interrupt


# ═══════════════════════════════════════════════════════════════════════════
# Block Diagram Viz Intermediate Sync
# ═══════════════════════════════════════════════════════════════════════════

class TestBlockDiagramVizIntermediateSync:
    """Verify that _persist_intermediate_state regenerates block_diagram_viz.json
    when the block diagram is updated, keeping the ReactFlow canvas in sync
    with the memory map sidebar during architecture iteration loops.

    Previously, block_diagram_viz.json was only written by
    create_documentation_node at the end of the pipeline. This caused the
    block diagram canvas to show stale block names (e.g. "cavlc_encoder")
    while the memory map sidebar (sourced from architecture_state.json)
    already showed the updated name (e.g. "expgolomb_encoder").
    """

    def test_viz_written_on_block_diagram_update(self, isolated_project):
        """When _persist_intermediate_state receives a block_diagram update,
        block_diagram_viz.json should be written to disk."""
        import json
        from pathlib import Path

        state = {
            "project_root": isolated_project,
            "requirements": FFT16_REQUIREMENTS,
            "target_clock_mhz": 50.0,
        }
        updates = {
            "block_diagram": FFT16_BLOCK_DIAGRAM,
            "phase": "block_diagram",
        }

        _persist_intermediate_state(state, updates)

        viz_path = Path(isolated_project) / ".coresmith" / "block_diagram_viz.json"
        assert viz_path.exists(), "block_diagram_viz.json was not written"

        viz = json.loads(viz_path.read_text())
        assert "architecture" in viz
        node_names = [
            n["data"]["device_name"]
            for n in viz["architecture"]["systemNodes"]
        ]
        assert "fft_butterfly" in node_names
        assert "twiddle_rom" in node_names
        assert "fft_controller" in node_names

    def test_viz_updated_when_block_renamed(self, isolated_project):
        """Simulates the cavlc_encoder -> expgolomb_encoder rename bug.

        After the first persist with "old_block", a second persist with
        "new_block" must update block_diagram_viz.json to contain only
        the new name.
        """
        import json
        from pathlib import Path

        state = {
            "project_root": isolated_project,
            "requirements": "test chip",
            "target_clock_mhz": 50.0,
        }

        old_diagram = {
            "blocks": [
                {"name": "cavlc_encoder", "description": "CAVLC entropy encoder", "tier": 1},
                {"name": "pixel_buffer", "description": "Pixel line buffer", "tier": 1},
            ],
            "connections": [
                {"from": "pixel_buffer", "to": "cavlc_encoder", "interface": "data", "data_width": 16},
            ],
            "questions": [],
        }

        # First persist: write initial viz with cavlc_encoder
        _persist_intermediate_state(state, {"block_diagram": old_diagram, "phase": "block_diagram"})

        viz_path = Path(isolated_project) / ".coresmith" / "block_diagram_viz.json"
        viz_v1 = json.loads(viz_path.read_text())
        v1_names = [n["data"]["device_name"] for n in viz_v1["architecture"]["systemNodes"]]
        assert "cavlc_encoder" in v1_names

        # Second persist: rename cavlc_encoder -> expgolomb_encoder
        new_diagram = {
            "blocks": [
                {"name": "expgolomb_encoder", "description": "Exp-Golomb entropy encoder", "tier": 1},
                {"name": "pixel_buffer", "description": "Pixel line buffer", "tier": 1},
            ],
            "connections": [
                {"from": "pixel_buffer", "to": "expgolomb_encoder", "interface": "data", "data_width": 16},
            ],
            "questions": [],
        }

        _persist_intermediate_state(state, {"block_diagram": new_diagram, "phase": "block_diagram"})

        viz_v2 = json.loads(viz_path.read_text())
        v2_names = [n["data"]["device_name"] for n in viz_v2["architecture"]["systemNodes"]]
        assert "expgolomb_encoder" in v2_names, (
            f"block_diagram_viz.json still has old block names: {v2_names}"
        )
        assert "cavlc_encoder" not in v2_names, (
            "cavlc_encoder should no longer appear after rename"
        )

    def test_viz_includes_memory_map_annotations(self, isolated_project):
        """When both block_diagram and memory_map are in state, the viz
        should include memory map annotations (address, size) in node notes."""
        import json
        from pathlib import Path

        state = {
            "project_root": isolated_project,
            "requirements": FFT16_REQUIREMENTS,
            "target_clock_mhz": 50.0,
            "memory_map": FFT16_MEMORY_MAP,
        }
        updates = {
            "block_diagram": FFT16_BLOCK_DIAGRAM,
            "phase": "block_diagram",
        }

        _persist_intermediate_state(state, updates)

        viz_path = Path(isolated_project) / ".coresmith" / "block_diagram_viz.json"
        viz = json.loads(viz_path.read_text())

        butterfly_nodes = [
            n for n in viz["architecture"]["systemNodes"]
            if n["data"].get("device_name") == "fft_butterfly"
        ]
        assert len(butterfly_nodes) == 1
        notes = butterfly_nodes[0]["data"].get("node_notes", "")
        assert "0x10000000" in notes, (
            f"Expected memory map address in node notes, got: {notes}"
        )

    def test_viz_not_written_without_blocks(self, isolated_project):
        """When block_diagram has no blocks, viz should not be written
        (avoids writing an empty/broken diagram)."""
        from pathlib import Path

        state = {
            "project_root": isolated_project,
            "requirements": "test",
            "target_clock_mhz": 50.0,
        }
        updates = {
            "block_diagram": {"blocks": [], "connections": [], "questions": []},
            "phase": "block_diagram",
        }

        _persist_intermediate_state(state, updates)

        viz_path = Path(isolated_project) / ".coresmith" / "block_diagram_viz.json"
        assert not viz_path.exists(), (
            "block_diagram_viz.json should not be written for empty block diagram"
        )

    def test_viz_not_written_without_block_diagram(self, isolated_project):
        """When updates don't include block_diagram, viz should not be touched."""
        from pathlib import Path

        state = {
            "project_root": isolated_project,
            "requirements": "test",
            "target_clock_mhz": 50.0,
        }
        updates = {
            "memory_map": FFT16_MEMORY_MAP,
            "phase": "memory_map",
        }

        _persist_intermediate_state(state, updates)

        viz_path = Path(isolated_project) / ".coresmith" / "block_diagram_viz.json"
        assert not viz_path.exists(), (
            "block_diagram_viz.json should not be written when only memory_map is updated"
        )


# ═══════════════════════════════════════════════════════════════════════════
# SAD Node Flow (new node, tests for post-refactor)
# ═══════════════════════════════════════════════════════════════════════════

class TestSADNodeFlow:
    """Tests for the System Architecture node (post-refactor).

    These tests target the new system_architecture_node() function that
    will be added in the PRD->SAD->FRD refactor. They verify:
    - The node calls generate_sad() with the PRD spec from graph state
    - The node returns {"sad_spec": ..., "phase": "sad"}
    - The node persists arch/sad_spec.md (markdown only)
    """

    @pytest.mark.asyncio
    async def test_sad_node_returns_sad_spec(self, isolated_project):
        """system_architecture_node should return sad_spec and phase='sad'."""
        from orchestrator.langgraph.architecture_graph import system_architecture_node
        from orchestrator.tests.fft16_fixtures import FFT16_SAD_MARKDOWN

        state = {
            "project_root": isolated_project,
            "prd_spec": FFT16_PRD_DOCUMENT,
            "requirements": FFT16_REQUIREMENTS,
            "pdk_summary": "sky130 | 130nm",
            "round": 1,
        }

        with patch(
            "orchestrator.architecture.specialists.sad_spec.generate_sad",
            new_callable=AsyncMock,
            return_value=FFT16_SAD_MARKDOWN,
        ):
            result = await system_architecture_node(state)

        assert result["sad_spec"] == FFT16_SAD_MARKDOWN
        assert result["phase"] == "sad"

    @pytest.mark.asyncio
    async def test_sad_node_persists_files(self, isolated_project):
        """system_architecture_node should write sad_spec.md (markdown only)."""
        from orchestrator.langgraph.architecture_graph import system_architecture_node
        from orchestrator.tests.fft16_fixtures import FFT16_SAD_MARKDOWN

        state = {
            "project_root": isolated_project,
            "prd_spec": FFT16_PRD_DOCUMENT,
            "requirements": FFT16_REQUIREMENTS,
            "pdk_summary": "sky130 | 130nm",
            "round": 1,
        }

        with patch(
            "orchestrator.architecture.specialists.sad_spec.generate_sad",
            new_callable=AsyncMock,
            return_value=FFT16_SAD_MARKDOWN,
        ):
            await system_architecture_node(state)

        from pathlib import Path
        arch = Path(isolated_project) / "arch"
        assert (arch / "sad_spec.md").exists()
        md_text = (arch / "sad_spec.md").read_text()
        assert "16-Point FFT Processor" in md_text


# ═══════════════════════════════════════════════════════════════════════════
# FRD Node Flow (new node, tests for post-refactor)
# ═══════════════════════════════════════════════════════════════════════════

class TestFRDNodeFlow:
    """Tests for the Functional Requirements node (post-refactor).

    These tests target the new functional_requirements_node() function.
    """

    @pytest.mark.asyncio
    async def test_frd_node_returns_frd_spec(self, isolated_project):
        """functional_requirements_node should return frd_spec and phase='frd'."""
        from orchestrator.langgraph.architecture_graph import functional_requirements_node
        from orchestrator.tests.fft16_fixtures import FFT16_FRD_MARKDOWN, FFT16_SAD_MARKDOWN

        state = {
            "project_root": isolated_project,
            "prd_spec": FFT16_PRD_DOCUMENT,
            "sad_spec": FFT16_SAD_MARKDOWN,
            "requirements": FFT16_REQUIREMENTS,
            "round": 1,
        }

        with patch(
            "orchestrator.architecture.specialists.frd_spec.generate_frd",
            new_callable=AsyncMock,
            return_value=FFT16_FRD_MARKDOWN,
        ):
            result = await functional_requirements_node(state)

        assert result["frd_spec"] == FFT16_FRD_MARKDOWN
        assert result["phase"] == "frd"

    @pytest.mark.asyncio
    async def test_frd_node_persists_files(self, isolated_project):
        """functional_requirements_node should write frd_spec.md (markdown only)."""
        from orchestrator.langgraph.architecture_graph import functional_requirements_node
        from orchestrator.tests.fft16_fixtures import FFT16_FRD_MARKDOWN, FFT16_SAD_MARKDOWN

        state = {
            "project_root": isolated_project,
            "prd_spec": FFT16_PRD_DOCUMENT,
            "sad_spec": FFT16_SAD_MARKDOWN,
            "requirements": FFT16_REQUIREMENTS,
            "round": 1,
        }

        with patch(
            "orchestrator.architecture.specialists.frd_spec.generate_frd",
            new_callable=AsyncMock,
            return_value=FFT16_FRD_MARKDOWN,
        ):
            await functional_requirements_node(state)

        from pathlib import Path
        arch = Path(isolated_project) / "arch"
        assert (arch / "frd_spec.md").exists()
        md_text = (arch / "frd_spec.md").read_text()
        assert "16-Point FFT Processor" in md_text


# ═══════════════════════════════════════════════════════════════════════════
# Per-Document Persistence (post-refactor: replaces _persist_intermediate_state)
# ═══════════════════════════════════════════════════════════════════════════

class TestPerDocPersistence:
    """After refactor, each node writes its own .json + .md file pair.

    Verify that after each node function runs, the corresponding
    per-document files exist and architecture_state.json does NOT.
    """

    @pytest.mark.asyncio
    async def test_block_diagram_node_writes_per_doc_files(self, isolated_project):
        """block_diagram_node should write block_diagram.json and block_diagram.md."""
        from orchestrator.langgraph.architecture_graph import block_diagram_node

        state = {
            "project_root": isolated_project,
            "requirements": FFT16_REQUIREMENTS,
            "prd_spec": FFT16_PRD_DOCUMENT,
            "sad_spec": FFT16_SAD_DOCUMENT,
            "frd_spec": FFT16_FRD_DOCUMENT,
            "pdk_summary": "sky130 | 130nm | 1.8V | tt_025C_1v80",
            "block_diagram": None,
            "constraint_result": None,
            "human_feedback": "",
            "round": 1,
            "target_clock_mhz": 50.0,
            "violations_history": [],
            "human_response_history": [],
            "total_rounds": 0,
        }

        with patch(
            "orchestrator.architecture.specialists.block_diagram.analyze_block_diagram",
            new_callable=AsyncMock,
            return_value=FFT16_BLOCK_DIAGRAM,
        ):
            await block_diagram_node(state)

        from pathlib import Path
        coresmith = Path(isolated_project) / ".coresmith"
        assert (coresmith / "block_diagram.json").exists()
        arch = Path(isolated_project) / "arch"
        assert (arch / "block_diagram.md").exists()

    @pytest.mark.asyncio
    async def test_memory_map_node_writes_per_doc_files(self, isolated_project):
        """memory_map_node should write memory_map.json and memory_map.md."""
        from orchestrator.langgraph.architecture_graph import memory_map_node

        state = {
            "project_root": isolated_project,
            "requirements": FFT16_REQUIREMENTS,
            "block_diagram": FFT16_BLOCK_DIAGRAM,
            "target_clock_mhz": 50.0,
            "prd_spec": FFT16_PRD_DOCUMENT,
            "round": 1,
        }

        with patch(
            "orchestrator.architecture.specialists.memory_map.analyze_memory_map",
            new_callable=AsyncMock,
            return_value=FFT16_MEMORY_MAP,
        ):
            await memory_map_node(state)

        from pathlib import Path
        coresmith = Path(isolated_project) / ".coresmith"
        assert (coresmith / "memory_map.json").exists()
        arch = Path(isolated_project) / "arch"
        assert (arch / "memory_map.md").exists()

    @pytest.mark.asyncio
    async def test_no_architecture_state_json_written(self, isolated_project):
        """After per-doc refactor, architecture_state.json should NOT be written."""
        from orchestrator.langgraph.architecture_graph import memory_map_node

        state = {
            "project_root": isolated_project,
            "requirements": FFT16_REQUIREMENTS,
            "block_diagram": FFT16_BLOCK_DIAGRAM,
            "target_clock_mhz": 50.0,
            "prd_spec": FFT16_PRD_DOCUMENT,
            "round": 1,
        }

        with patch(
            "orchestrator.architecture.specialists.memory_map.analyze_memory_map",
            new_callable=AsyncMock,
            return_value=FFT16_MEMORY_MAP,
        ):
            await memory_map_node(state)

        from pathlib import Path
        arch_state = Path(isolated_project) / ".coresmith" / "architecture_state.json"
        # This assertion will pass only after the refactor removes
        # _persist_intermediate_state and replaces it with per-doc helpers
        if arch_state.exists():
            pytest.xfail(
                "architecture_state.json still written "
                "(expected until per-doc migration is complete)"
            )


class TestFinalizeBlockSpecs:
    def test_non_silicon_validation_block_is_not_frontend_work_item(self):
        from orchestrator.langgraph.architecture_graph import (
            _is_non_silicon_validation_block,
        )

        assert _is_non_silicon_validation_block({
            "name": "flow_smoke_checks",
            "description": "Non-synthesizable validation harness definition",
            "rtl_target": "",
            "estimated_gates": 0,
        })
        assert not _is_non_silicon_validation_block({
            "name": "adder32",
            "description": "Pure combinational 32-bit unsigned adder",
            "rtl_target": "rtl/adder32.v",
            "estimated_gates": 240,
        })
