# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Integration tests for the architecture MCP loop.

These tests exercise the full MCP tool -> GraphLifecycle -> LangGraph ->
interrupt -> poll -> resume -> proceed cycle.  Unlike the unit tests in
test_architecture_graph.py (which call ``graph.ainvoke()`` directly) and
test_mcp_server.py (which mock away ``run_task``), these tests:

  1. Call the real MCP tool functions (start_architecture, get_architecture_state,
     resume_architecture).
  2. Let GraphLifecycle create real asyncio.Tasks with AsyncSqliteSaver.
  3. Mock only the specialist functions (gather_prd, analyze_block_diagram, etc.)
     so no LLM calls are made.
  4. Poll the runner status with a timeout to wait for interrupts.
  5. Resume through interrupts and verify the graph proceeds.

Specialist mocks use ``side_effect`` functions for phase-aware behavior
(e.g. ERS Phase 1 returns questions, Phase 2 returns the ERS document).
"""

from __future__ import annotations

import asyncio
import json
from contextlib import ExitStack
from unittest.mock import AsyncMock, patch

import pytest

from orchestrator.tests.conftest import enter_arch_llm_node_patches, wait_for_status
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


async def _wait_after_resume(runner, target_statuses, timeout=30):
    """Wait for the graph to reach a new stopping point after a resume.

    After ``resume_architecture`` creates a new background task, the mocked
    specialists may complete so quickly that the status transitions from
    "interrupted" -> "running" -> "interrupted" within a single poll
    interval.  This helper waits for the background *task* to finish
    (``task.done()``) rather than polling the status string, avoiding
    that race entirely.
    """
    import time

    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if runner.task is not None and runner.task.done():
            break
        await asyncio.sleep(0.1)

    if runner.status in target_statuses:
        return runner.status

    raise TimeoutError(
        f"Runner '{runner.name}' status is '{runner.status}', "
        f"expected one of {target_statuses} within {timeout}s"
    )


# ---------------------------------------------------------------------------
# Phase-aware specialist mocks
# ---------------------------------------------------------------------------

def _make_smart_gather_prd():
    """PRD mock that returns questions on Phase 1, document on Phase 2."""
    async def _gather_prd(*, requirements, pdk_summary, target_clock_mhz,
                          user_answers=None, previous_questions=None,
                          project_root="."):
        if user_answers and len(user_answers) > 0:
            return FFT16_PRD_DOCUMENT
        return FFT16_PRD_QUESTIONS
    return _gather_prd


def _patch_specialists_for_lifecycle(*, constraint_side_effect=None):
    """Context manager that patches all specialists for a full lifecycle run.

    Uses a phase-aware PRD mock and simple return values for the other
    specialists.  Optionally accepts a ``constraint_side_effect`` callable
    to control constraint checker behavior across iterations.
    """
    stack = ExitStack()

    stack.enter_context(patch(
        "orchestrator.architecture.specialists.prd_spec.gather_prd",
        new_callable=AsyncMock,
        side_effect=_make_smart_gather_prd(),
    ))
    stack.enter_context(patch(
        "orchestrator.architecture.specialists.sad_spec.generate_sad",
        new_callable=AsyncMock,
        return_value=FFT16_SAD_DOCUMENT,
    ))
    stack.enter_context(patch(
        "orchestrator.architecture.specialists.frd_spec.generate_frd",
        new_callable=AsyncMock,
        return_value=FFT16_FRD_DOCUMENT,
    ))
    stack.enter_context(patch(
        "orchestrator.architecture.specialists.ers_doc.generate_ers_doc",
        new_callable=AsyncMock,
        return_value={"ers": {"title": "Test ERS"}, "phase": "prd_complete"},
    ))
    stack.enter_context(patch(
        "orchestrator.architecture.specialists.block_diagram.analyze_block_diagram",
        new_callable=AsyncMock,
        return_value=FFT16_BLOCK_DIAGRAM,
    ))
    stack.enter_context(patch(
        "orchestrator.architecture.specialists.memory_map.analyze_memory_map",
        new_callable=AsyncMock,
        return_value=FFT16_MEMORY_MAP,
    ))
    stack.enter_context(patch(
        "orchestrator.architecture.specialists.clock_tree.analyze_clock_tree",
        new_callable=AsyncMock,
        return_value=FFT16_CLOCK_TREE,
    ))
    stack.enter_context(patch(
        "orchestrator.architecture.specialists.register_spec.analyze_register_spec",
        new_callable=AsyncMock,
        return_value=FFT16_REGISTER_SPEC,
    ))

    if constraint_side_effect is not None:
        stack.enter_context(patch(
            "orchestrator.architecture.constraints.check_constraints",
            new_callable=AsyncMock,
            side_effect=constraint_side_effect,
        ))
    else:
        stack.enter_context(patch(
            "orchestrator.architecture.constraints.check_constraints",
            new_callable=AsyncMock,
            return_value=[],
        ))

    # Interface Definition and Output Contract Review are graph nodes rather
    # than specialists, so they were missing from the list above and reached a
    # live LLM. Here that showed up as `wait_for_status` timing out at 30s --
    # the runner was still 'running' because a real API call was in flight.
    enter_arch_llm_node_patches(stack)

    return stack


# ---------------------------------------------------------------------------
# Cleanup fixture
# ---------------------------------------------------------------------------

@pytest.fixture
async def arch_cleanup(reset_mcp_state):
    """Yield the MCP module and clean up background tasks afterward."""
    import orchestrator.mcp_server as mcp

    yield mcp

    if mcp._architecture.task and not mcp._architecture.task.done():
        mcp._architecture.task.cancel()
        try:
            await mcp._architecture.task
        except (asyncio.CancelledError, Exception):
            pass
    await mcp._architecture.cleanup()


# ═══════════════════════════════════════════════════════════════════════════
# TestArchitectureMCPLifecycle
#
# Full integration: MCP tool functions -> GraphLifecycle -> LangGraph
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.mcp
class TestArchitectureMCPLifecycle:
    """Tests that exercise the complete MCP -> graph -> interrupt -> resume cycle."""

    @pytest.mark.asyncio
    async def test_start_reaches_ers_interrupt(self, arch_cleanup):
        """start_architecture -> graph runs -> ERS questions interrupt.

        Verifies that:
        - start_architecture returns without error
        - The background task reaches "interrupted" status
        - get_architecture_state surfaces the ERS questions payload
        """
        mcp = arch_cleanup

        with _patch_specialists_for_lifecycle():
            result_json = await mcp.start_architecture(
                requirements=FFT16_REQUIREMENTS,
                target_clock_mhz=50.0,
            )
            result = json.loads(result_json)
            assert "error" not in result, f"start_architecture failed: {result}"

            status = await wait_for_status(
                mcp._architecture, {"interrupted", "error"}, timeout=15,
            )
            assert status == "interrupted", (
                f"Expected 'interrupted', got '{status}'. "
                f"Error: {mcp._architecture.error_message}"
            )

            state_json = await mcp.get_architecture_state()
            state = json.loads(state_json)

        assert state["status"] == "interrupted"
        assert state["human_input_needed"] is True
        assert state["interrupt_payload"] is not None
        assert state["interrupt_payload"]["type"] == "prd_questions"
        assert len(state["interrupt_payload"]["questions"]) > 0

    @pytest.mark.asyncio
    async def test_ers_resume_proceeds_past_ers(self, arch_cleanup):
        """Resume ERS with answers -> graph proceeds past ERS phase.

        After providing answers, the graph should reach either:
        - Another interrupt (block diagram, constraints, or final review)
        - Completion (done)
        It must NOT loop back to ERS questions.
        """
        mcp = arch_cleanup

        with _patch_specialists_for_lifecycle():
            await mcp.start_architecture(
                requirements=FFT16_REQUIREMENTS,
                target_clock_mhz=50.0,
            )
            await wait_for_status(
                mcp._architecture, {"interrupted"}, timeout=15,
            )

            # Resume with ERS answers
            resume_json = await mcp.resume_architecture(
                action="continue",
                feedback=json.dumps(FFT16_PRD_ANSWERS),
            )
            resume_result = json.loads(resume_json)
            assert "error" not in resume_result, (
                f"resume_architecture failed: {resume_result}"
            )

            # Wait for the graph to reach its next stopping point
            status = await _wait_after_resume(
                mcp._architecture, {"interrupted", "done", "error"}, timeout=30,
            )
            assert status != "error", (
                f"Graph errored after ERS resume: {mcp._architecture.error_message}"
            )

            state_json = await mcp.get_architecture_state()
            state = json.loads(state_json)

        # The graph must have advanced past ERS
        assert state["prd_complete"] is True
        # If interrupted, it should be at a later phase (not ERS again)
        if state["status"] == "interrupted":
            payload = state["interrupt_payload"]
            assert payload["type"] != "prd_questions", (
                "Graph looped back to ERS questions -- answers were not delivered"
            )

    @pytest.mark.asyncio
    async def test_happy_path_full_cycle(self, arch_cleanup):
        """Full lifecycle: start -> ERS Q&A -> specialists -> final review -> accept.

        Walks the entire architecture loop through the MCP layer:
        1. start_architecture
        2. Poll until ERS interrupt
        3. Resume with answers
        4. Poll until final review interrupt (constraints pass, so no other interrupts)
        5. Accept at final review
        6. Poll until done
        7. Verify success=True and block_specs_path populated
        """
        mcp = arch_cleanup

        with _patch_specialists_for_lifecycle():
            # Step 1: Start
            result_json = await mcp.start_architecture(
                requirements=FFT16_REQUIREMENTS,
                target_clock_mhz=50.0,
            )
            result = json.loads(result_json)
            assert "error" not in result

            # Step 2: Wait for ERS interrupt
            await wait_for_status(
                mcp._architecture, {"interrupted"}, timeout=15,
            )
            state = json.loads(await mcp.get_architecture_state())
            assert state["interrupt_payload"]["type"] == "prd_questions"

            # Step 3: Resume with ERS answers
            await mcp.resume_architecture(
                action="continue",
                feedback=json.dumps(FFT16_PRD_ANSWERS),
            )

            # Step 4: Wait for final review interrupt
            # (constraints pass, so graph goes straight through to final review)
            status = await _wait_after_resume(
                mcp._architecture, {"interrupted", "done", "error"}, timeout=30,
            )
            assert status != "error", (
                f"Graph errored: {mcp._architecture.error_message}"
            )

            state = json.loads(await mcp.get_architecture_state())

            # Should be at final review
            if state["status"] == "interrupted":
                assert state["interrupt_payload"]["type"] == "final_review", (
                    f"Expected final_review, got {state['interrupt_payload']['type']}"
                )

                # Step 5: Accept at final review (OK2DEV)
                await mcp.resume_architecture(action="accept")

                # Step 6: Wait for completion
                status = await _wait_after_resume(
                    mcp._architecture, {"done", "error"}, timeout=15,
                )
                assert status == "done", (
                    f"Expected 'done', got '{status}'. "
                    f"Error: {mcp._architecture.error_message}"
                )

            # Step 7: Verify final state
            state = json.loads(await mcp.get_architecture_state())

        assert state["success"] is True
        assert state["block_specs_path"] != ""
        assert state["block_count"] == 3
        assert "fft_butterfly" in state["block_names"]

    @pytest.mark.asyncio
    async def test_ers_abort_stops_graph(self, arch_cleanup):
        """Aborting at ERS interrupt -> graph completes with success=False."""
        mcp = arch_cleanup

        with _patch_specialists_for_lifecycle():
            await mcp.start_architecture(
                requirements=FFT16_REQUIREMENTS,
                target_clock_mhz=50.0,
            )
            await wait_for_status(
                mcp._architecture, {"interrupted"}, timeout=15,
            )

            # Abort
            await mcp.resume_architecture(action="abort")

            await _wait_after_resume(
                mcp._architecture, {"done", "error"}, timeout=15,
            )

            state = json.loads(await mcp.get_architecture_state())

        assert state["success"] is False

    @pytest.mark.asyncio
    async def test_constraint_autofix_loops_and_succeeds(self, arch_cleanup):
        """Auto-fixable constraint violation -> Block Diagram re-run -> pass.

        Uses a side_effect on check_constraints that fails on the first call
        (auto_fixable) and passes on the second. Verifies the graph loops
        through Constraint Iteration and eventually completes.
        """
        mcp = arch_cleanup

        call_count = {"n": 0}
        auto_fixable = [
            {"violation": "Gate budget exceeded", "severity": "warning",
             "category": "auto_fixable"}
        ]

        async def constraint_side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return auto_fixable
            return []

        with _patch_specialists_for_lifecycle(
            constraint_side_effect=constraint_side_effect,
        ):
            await mcp.start_architecture(
                requirements=FFT16_REQUIREMENTS,
                target_clock_mhz=50.0,
            )
            await wait_for_status(
                mcp._architecture, {"interrupted"}, timeout=15,
            )

            # Resume ERS
            await mcp.resume_architecture(
                action="continue",
                feedback=json.dumps(FFT16_PRD_ANSWERS),
            )

            # Wait for next stopping point (final review after constraint loop)
            status = await _wait_after_resume(
                mcp._architecture, {"interrupted", "done", "error"}, timeout=30,
            )
            assert status != "error", (
                f"Graph errored: {mcp._architecture.error_message}"
            )

            state = json.loads(await mcp.get_architecture_state())

            # Accept final review if we reached it
            if (state["status"] == "interrupted" and
                    state.get("interrupt_payload", {}).get("type") == "final_review"):
                await mcp.resume_architecture(action="accept")
                await _wait_after_resume(
                    mcp._architecture, {"done", "error"}, timeout=15,
                )

            state = json.loads(await mcp.get_architecture_state())

        assert state["success"] is True
        # Constraint checker was called at least twice (fail then pass)
        assert call_count["n"] >= 2
        # Round should have advanced past 1
        assert state["round"] >= 2

    @pytest.mark.asyncio
    async def test_get_state_returns_interrupt_payload_fields(self, arch_cleanup):
        """get_architecture_state surfaces all expected interrupt fields."""
        mcp = arch_cleanup

        with _patch_specialists_for_lifecycle():
            await mcp.start_architecture(
                requirements=FFT16_REQUIREMENTS,
                target_clock_mhz=50.0,
            )
            await wait_for_status(
                mcp._architecture, {"interrupted"}, timeout=15,
            )

            state = json.loads(await mcp.get_architecture_state())

        assert state["status"] == "interrupted"
        assert state["human_input_needed"] is True
        assert state["interrupt_payload"] is not None
        assert state["interrupt_type"] == "prd_questions"
        assert isinstance(state["interrupt_actions"], list)
        assert len(state["interrupt_actions"]) > 0
        assert "continue" in state["interrupt_actions"]
        assert state["phase"] == "prd"
        assert state["thread_id"] == "architecture"


# ═══════════════════════════════════════════════════════════════════════════
# TestGraphLifecycleIntegration
#
# Lifecycle management: reset, status self-heal
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.mcp
class TestGraphLifecycleIntegration:
    """Tests for GraphLifecycle management through the MCP layer."""

    @pytest.mark.asyncio
    async def test_reset_for_new_run_clears_previous_state(self, arch_cleanup):
        """After reaching an interrupt, reset_for_new_run wipes checkpoint state."""
        mcp = arch_cleanup

        with _patch_specialists_for_lifecycle():
            # Start and reach ERS interrupt
            await mcp.start_architecture(
                requirements=FFT16_REQUIREMENTS,
                target_clock_mhz=50.0,
            )
            await wait_for_status(
                mcp._architecture, {"interrupted"}, timeout=15,
            )

            # Verify we have state
            state = json.loads(await mcp.get_architecture_state())
            assert state["interrupt_payload"] is not None

        # Reset
        await mcp._architecture.reset_for_new_run()

        # State should be wiped
        state = json.loads(await mcp.get_architecture_state())
        assert state["status"] == "idle"
        assert state.get("interrupt_payload") is None

    @pytest.mark.asyncio
    async def test_status_self_heal_on_poll(self, arch_cleanup):
        """Status self-heals when task is done but status still says 'running'."""
        mcp = arch_cleanup

        with _patch_specialists_for_lifecycle():
            # Start and wait for ERS interrupt
            await mcp.start_architecture(
                requirements=FFT16_REQUIREMENTS,
                target_clock_mhz=50.0,
            )
            await wait_for_status(
                mcp._architecture, {"interrupted"}, timeout=15,
            )

            # Simulate the race: manually set status to "running"
            # even though the task is done (has an interrupt)
            mcp._architecture.status = "running"

            # get_architecture_state should self-heal
            state = json.loads(await mcp.get_architecture_state())

        assert state["status"] == "interrupted"
        assert mcp._architecture.status == "interrupted"
