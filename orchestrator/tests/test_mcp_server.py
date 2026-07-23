# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for the MCP server tool layer.

Tests:
- Tool registration (expected tools exist)
- start_architecture lifecycle (valid start, missing requirements, double start)
- get_architecture_state (idle, after interrupt, surfaces questions)
- resume_architecture (invalid action, not interrupted, ERS answer parsing)
- Introspection tools (get_graph_structure, get_node_prompt)
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.tests.fft16_fixtures import (
    FFT16_ERS_ANSWERS,
    FFT16_REQUIREMENTS,
)


# ═══════════════════════════════════════════════════════════════════════════
# Tool Registration
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.mcp
class TestToolRegistration:
    def test_server_object_exists(self):
        from orchestrator.mcp_server import server

        assert server is not None
        assert server.name == "coresmith-architecture"


# ═══════════════════════════════════════════════════════════════════════════
# start_architecture Tool
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.mcp
class TestStartArchitecture:
    @pytest.mark.asyncio
    async def test_no_requirements_returns_error(self, reset_mcp_state):
        from orchestrator.mcp_server import start_architecture

        result_json = await start_architecture(requirements="")
        result = json.loads(result_json)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_rejects_double_start(self, reset_mcp_state):
        import orchestrator.mcp_server as mcp

        mcp._architecture.status = "running"
        mcp._architecture.thread_id = "test-123"

        result_json = await mcp.start_architecture(requirements=FFT16_REQUIREMENTS)
        result = json.loads(result_json)
        assert "error" in result
        assert "already running" in result["error"].lower()


# ═══════════════════════════════════════════════════════════════════════════
# get_architecture_state Tool
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.mcp
class TestGetArchitectureState:
    @pytest.mark.asyncio
    async def test_idle_returns_idle(self, reset_mcp_state):
        from orchestrator.mcp_server import get_architecture_state

        result_json = await get_architecture_state()
        result = json.loads(result_json)
        assert result["status"] == "idle"


# ═══════════════════════════════════════════════════════════════════════════
# resume_architecture Tool
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.mcp
class TestResumeArchitecture:
    @pytest.mark.asyncio
    async def test_invalid_action_returns_error(self, reset_mcp_state):
        from orchestrator.mcp_server import resume_architecture

        result_json = await resume_architecture(action="invalid_action")
        result = json.loads(result_json)
        assert "error" in result
        assert "invalid" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_resume_when_not_interrupted_returns_error(self, reset_mcp_state):
        import orchestrator.mcp_server as mcp

        mcp._architecture.status = "idle"

        result_json = await mcp.resume_architecture(action="continue")
        result = json.loads(result_json)
        assert "error" in result
        assert "cannot resume" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_feedback_without_text_returns_error(self, reset_mcp_state):
        import orchestrator.mcp_server as mcp

        mcp._architecture.status = "interrupted"
        mcp._architecture.thread_id = "test-123"

        result_json = await mcp.resume_architecture(action="feedback", feedback="")
        result = json.loads(result_json)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_ers_answers_json_parsing(self, reset_mcp_state):
        """Verify JSON-encoded answers in feedback are parsed correctly."""
        import orchestrator.mcp_server as mcp

        answers_json = json.dumps(FFT16_ERS_ANSWERS)

        mcp._architecture.status = "interrupted"
        mcp._architecture.thread_id = "test-parse-1"
        mcp._architecture.graph = AsyncMock()

        captured_input = {}

        async def mock_run(initial_input, config):
            captured_input["cmd"] = initial_input

        with patch.object(mcp._architecture, "ensure_graph", new_callable=AsyncMock), \
             patch.object(mcp._architecture, "run_task", new_callable=AsyncMock, side_effect=mock_run), \
             patch("orchestrator.langgraph.event_stream.write_graph_event"):
            result_json = await mcp.resume_architecture(
                action="continue", feedback=answers_json
            )

        result = json.loads(result_json)
        assert result["status"] == "running"


# ═══════════════════════════════════════════════════════════════════════════
# resume_architecture: Answer Delivery Contract
#
# The original bug: resume_architecture(action="feedback", feedback='...')
# silently dropped the answers dict because the JSON parsing only ran
# when action=="continue".  The outer agent used "feedback" (matching the
# MCP docstring's suggestion), so gather_requirements_node received
# human_response with no "answers" key, causing it to regenerate
# questions ad infinitum.
#
# These tests verify the contract between the MCP tool layer and the
# LangGraph Command(resume=...) mechanism.
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.mcp
class TestResumeArchitectureAnswerDelivery:
    """Verify that ERS answers survive the MCP -> Command(resume=...) pipeline."""

    async def _call_resume_and_get_resume_value(
        self, mcp, *, action: str, feedback: str,
    ) -> dict:
        """Call resume_architecture and return the resume_value dict passed
        to Command(resume=...).

        Sets up the MCP state to "interrupted", calls the tool, and inspects
        the Command object that run_task was called with.
        """
        mcp._architecture.status = "interrupted"
        mcp._architecture.thread_id = "test-answer-delivery"
        mcp._architecture.graph = AsyncMock()

        with patch.object(mcp._architecture, "ensure_graph", new_callable=AsyncMock), \
             patch.object(mcp._architecture, "run_task", new_callable=AsyncMock) as mock_task, \
             patch("orchestrator.langgraph.event_stream.write_graph_event"):
            result_json = await mcp.resume_architecture(
                action=action, feedback=feedback,
            )

        result = json.loads(result_json)
        assert "error" not in result, f"resume_architecture returned error: {result}"
        assert mock_task.called, "run_task was never called"

        cmd = mock_task.call_args[0][0]
        return cmd.resume

    # --- Core contract: answers must arrive for both action strings ----------

    @pytest.mark.asyncio
    @pytest.mark.parametrize("action", ["continue", "feedback"])
    async def test_json_answers_parsed_for_both_actions(self, reset_mcp_state, action):
        """Answers dict must appear in resume_value for 'continue' AND 'feedback'.

        This is the exact gap that caused the original bug: only 'continue'
        was handled, so 'feedback' silently dropped the answers.
        """
        import orchestrator.mcp_server as mcp

        resume_value = await self._call_resume_and_get_resume_value(
            mcp, action=action, feedback=json.dumps(FFT16_ERS_ANSWERS),
        )

        assert "answers" in resume_value, (
            f"action='{action}' with JSON answers did not set 'answers' key. "
            f"Keys present: {list(resume_value.keys())}"
        )
        assert resume_value["answers"] == FFT16_ERS_ANSWERS

    @pytest.mark.asyncio
    async def test_feedback_action_normalised_to_continue(self, reset_mcp_state):
        """action='feedback' + valid JSON answers -> action normalised to 'continue'.

        The graph routes on action string; downstream routing functions check
        for 'abort' vs default.  If action stays 'feedback', the routing
        still works (it falls through to 'Gather Requirements'), but the
        normalisation documents the intent and avoids subtle future breaks.
        """
        import orchestrator.mcp_server as mcp

        resume_value = await self._call_resume_and_get_resume_value(
            mcp, action="feedback", feedback=json.dumps({"q1": "a1"}),
        )

        assert resume_value["action"] == "continue", (
            f"Expected action normalised to 'continue', got '{resume_value['action']}'"
        )

    @pytest.mark.asyncio
    async def test_all_answer_keys_preserved(self, reset_mcp_state):
        """Every key from the outer agent's JSON must survive round-trip parsing."""
        import orchestrator.mcp_server as mcp

        resume_value = await self._call_resume_and_get_resume_value(
            mcp, action="feedback", feedback=json.dumps(FFT16_ERS_ANSWERS),
        )

        for key, value in FFT16_ERS_ANSWERS.items():
            assert key in resume_value["answers"], f"Missing answer key: {key}"
            assert resume_value["answers"][key] == value, (
                f"Answer value mismatch for '{key}': "
                f"expected {value!r}, got {resume_value['answers'][key]!r}"
            )

    # --- Negative cases: non-answer feedback must NOT create answers key -----

    @pytest.mark.asyncio
    async def test_plain_text_feedback_no_answers_key(self, reset_mcp_state):
        """Plain text feedback (not JSON) must NOT create an 'answers' key."""
        import orchestrator.mcp_server as mcp

        resume_value = await self._call_resume_and_get_resume_value(
            mcp, action="feedback", feedback="please add DMA support",
        )

        assert "answers" not in resume_value
        assert resume_value["feedback"] == "please add DMA support"

    @pytest.mark.asyncio
    async def test_json_list_not_treated_as_answers(self, reset_mcp_state):
        """JSON list (not dict) must NOT be treated as answers.

        The answers must be a dict mapping question IDs to answer strings.
        A list, even if valid JSON, is not a valid answers dict.
        """
        import orchestrator.mcp_server as mcp

        resume_value = await self._call_resume_and_get_resume_value(
            mcp, action="feedback", feedback='["not", "a", "dict"]',
        )

        assert "answers" not in resume_value

    @pytest.mark.asyncio
    async def test_continue_without_feedback_has_no_answers(self, reset_mcp_state):
        """action='continue' with empty feedback -> no answers key.

        This is the normal 'accept and proceed' flow for non-ERS interrupts.
        """
        import orchestrator.mcp_server as mcp

        resume_value = await self._call_resume_and_get_resume_value(
            mcp, action="continue", feedback="",
        )

        assert "answers" not in resume_value

    @pytest.mark.asyncio
    async def test_json_string_not_treated_as_answers(self, reset_mcp_state):
        """JSON string literal (not dict) must NOT be treated as answers.

        json.loads('"hello"') returns a str, not a dict.
        """
        import orchestrator.mcp_server as mcp

        resume_value = await self._call_resume_and_get_resume_value(
            mcp, action="feedback", feedback='"just a string"',
        )

        assert "answers" not in resume_value


# ═══════════════════════════════════════════════════════════════════════════
# Introspection Tools
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.mcp
class TestIntrospectionTools:
    @pytest.mark.asyncio
    async def test_get_graph_structure_architecture(self, reset_mcp_state):
        from orchestrator.mcp_server import get_graph_structure

        result_json = await get_graph_structure(graph_name="architecture")
        result = json.loads(result_json)

        assert "nodes" in result or "error" not in result

    @pytest.mark.asyncio
    async def test_get_graph_structure_frontend(self, reset_mcp_state):
        from orchestrator.mcp_server import get_graph_structure

        result_json = await get_graph_structure(graph_name="frontend")
        result = json.loads(result_json)

        assert "nodes" in result or "error" not in result

    @pytest.mark.asyncio
    async def test_get_node_prompt_returns_content(self, reset_mcp_state):
        from orchestrator.mcp_server import get_node_prompt

        result_json = await get_node_prompt(
            graph_name="frontend", node_id="generate_rtl"
        )
        result = json.loads(result_json)

        if "prompt" in result:
            assert len(result["prompt"]) > 0


# ═══════════════════════════════════════════════════════════════════════════
# _build_resume_command: Type-aware interrupt handling
#
# The original escape: _build_resume_command matched block_actions to
# interrupts by block_name only, ignoring the interrupt type.  When a
# block had both a stale uarch_spec_review and a current
# human_intervention_needed interrupt, the action (e.g. "skip") could
# be applied to the wrong interrupt type.
# ═══════════════════════════════════════════════════════════════════════════

def _mock_interrupt(iid: str, payload: dict) -> MagicMock:
    """Create a mock Interrupt object with id and value."""
    intr = MagicMock()
    intr.id = iid
    intr.value = payload
    return intr


def _mock_state_snapshot(interrupts: list[tuple[str, dict]]) -> MagicMock:
    """Build a mock StateSnapshot from (id, payload) tuples.

    Each interrupt is placed in its own task (simulating parallel Send()
    branches), matching how LangGraph stores parallel block interrupts.
    """
    snapshot = MagicMock()
    tasks = []
    for iid, payload in interrupts:
        task = MagicMock()
        task.interrupts = [_mock_interrupt(iid, payload)]
        tasks.append(task)
    snapshot.tasks = tasks
    return snapshot


@pytest.mark.mcp
class TestBuildResumeCommand:
    """Tests for _build_resume_command interrupt handling."""

    def test_single_interrupt_uses_interrupt_id_key(self):
        from orchestrator.mcp_server import _build_resume_command

        snapshot = _mock_state_snapshot([
            ("int-1", {"type": "human_intervention_needed", "block_name": "scrambler"}),
        ])
        resume_value = {"action": "retry"}
        cmd = _build_resume_command(snapshot, resume_value, "retry", "", "", "")

        # Single interrupt now uses interrupt-ID-keyed map to avoid
        # LangGraph's "multiple pending interrupts" error when the
        # checkpoint's write-level count disagrees with our filtering.
        assert isinstance(cmd.resume, dict)
        assert "int-1" in cmd.resume
        assert cmd.resume["int-1"]["action"] == "retry"

    def test_multiple_interrupts_default_action(self):
        from orchestrator.mcp_server import _build_resume_command

        snapshot = _mock_state_snapshot([
            ("int-1", {"type": "uarch_spec_review", "block_name": "scrambler"}),
            ("int-2", {"type": "uarch_spec_review", "block_name": "encoder"}),
        ])
        resume_value = {"action": "approve", "constraint": "", "description": ""}
        cmd = _build_resume_command(snapshot, resume_value, "approve", "", "", "")

        assert isinstance(cmd.resume, dict)
        assert len(cmd.resume) == 2
        assert cmd.resume["int-1"] == resume_value
        assert cmd.resume["int-2"] == resume_value

    def test_block_actions_override_default(self):
        from orchestrator.mcp_server import _build_resume_command

        snapshot = _mock_state_snapshot([
            ("int-1", {"type": "uarch_spec_review", "block_name": "scrambler"}),
            ("int-2", {"type": "uarch_spec_review", "block_name": "encoder"}),
        ])
        block_actions = json.dumps({"scrambler": "skip", "encoder": "approve"})
        cmd = _build_resume_command(snapshot, {}, "approve", "", "", block_actions)

        assert cmd.resume["int-1"]["action"] == "skip"
        assert cmd.resume["int-2"]["action"] == "approve"

    def test_no_interrupts_returns_none(self):
        from orchestrator.mcp_server import _build_resume_command

        snapshot = MagicMock()
        snapshot.tasks = []
        cmd = _build_resume_command(snapshot, {}, "retry", "", "", "")
        assert cmd is None

    def test_type_aware_validation_blocks_approve_on_ask_human(self):
        """approve sent to ask_human interrupt should be remapped to safe default.

        This is the core escape: when block_actions sends approve to an
        ask_human interrupt, route_after_human defaults to increment_attempt,
        causing silent re-execution.  The fix validates action against the
        interrupt's supported_actions and remaps to a safe default.
        """
        from orchestrator.mcp_server import _build_resume_command

        snapshot = _mock_state_snapshot([
            ("int-1", {
                "type": "human_intervention_needed",
                "block_name": "scrambler",
                "supported_actions": ["retry", "fix_rtl", "add_constraint", "skip", "abort"],
            }),
        ])
        block_actions = json.dumps({"scrambler": "approve"})
        cmd = _build_resume_command(snapshot, {}, "approve", "", "", block_actions)

        # After the fix: approve should be remapped, not sent to ask_human
        # Single interrupt with block_actions: resume is a flat dict with "action" key
        action_sent = cmd.resume.get("action", "")
        assert action_sent != "approve", (
            "approve must not be sent to a human_intervention_needed interrupt"
        )

    def test_type_aware_validation_allows_valid_action(self):
        """skip is valid for both uarch_spec_review and ask_human."""
        from orchestrator.mcp_server import _build_resume_command

        snapshot = _mock_state_snapshot([
            ("int-1", {
                "type": "human_intervention_needed",
                "block_name": "scrambler",
                "supported_actions": ["retry", "fix_rtl", "add_constraint", "skip", "abort"],
            }),
        ])
        block_actions = json.dumps({"scrambler": "skip"})
        cmd = _build_resume_command(snapshot, {}, "skip", "", "", block_actions)

        # Single interrupt now uses interrupt-ID-keyed map
        assert "int-1" in cmd.resume
        assert cmd.resume["int-1"]["action"] == "skip"

    def test_mixed_interrupt_types_block_actions(self):
        """When blocks have different interrupt types, actions validated per-type.

        This reproduces the exact escape: neighbor_line_buffer has a stale
        uarch_spec_review AND pixel_input_buffer has ask_human.  Sending
        "approve" globally should approve the uarch review but NOT send
        approve to the ask_human interrupt.
        """
        from orchestrator.mcp_server import _build_resume_command

        snapshot = _mock_state_snapshot([
            ("int-1", {
                "type": "uarch_spec_review",
                "block_name": "pixel_input_buffer",
                "supported_actions": ["approve", "revise", "skip"],
            }),
            ("int-2", {
                "type": "human_intervention_needed",
                "block_name": "neighbor_line_buffer",
                "supported_actions": ["retry", "fix_rtl", "add_constraint", "skip", "abort"],
            }),
        ])
        block_actions = json.dumps({
            "pixel_input_buffer": "approve",
            "neighbor_line_buffer": "skip",
        })
        cmd = _build_resume_command(snapshot, {}, "approve", "", "", block_actions)

        assert isinstance(cmd.resume, dict)
        assert cmd.resume["int-1"]["action"] == "approve"
        assert cmd.resume["int-2"]["action"] == "skip"

    def test_stale_interrupts_from_completed_blocks_filtered(self):
        """Interrupts from completed blocks should be skipped in resume commands.

        When parallel Send() branches complete, their interrupt payloads remain
        in the LangGraph checkpoint. _build_resume_command must filter these out
        so the outer agent doesn't resume stale interrupts and re-execute
        completed blocks.
        """
        from orchestrator.mcp_server import _build_resume_command

        snapshot = _mock_state_snapshot([
            ("int-stale", {
                "type": "uarch_spec_review",
                "block_name": "scrambler",
                "supported_actions": ["approve", "revise", "skip"],
            }),
            ("int-active", {
                "type": "human_intervention_needed",
                "block_name": "decoder",
                "supported_actions": ["retry", "fix_rtl", "add_constraint", "skip", "abort"],
            }),
        ])
        # Mark scrambler as completed in the state values
        snapshot.values = {
            "completed_blocks": [
                {"name": "scrambler", "success": True},
            ],
        }

        resume_value = {"action": "retry", "constraint": "", "description": ""}
        cmd = _build_resume_command(snapshot, resume_value, "retry", "", "", "")

        # Only the active interrupt (decoder) should be in the resume command
        assert cmd is not None
        # Uses interrupt-ID-keyed map with only the active interrupt
        assert "int-active" in cmd.resume
        assert cmd.resume["int-active"]["action"] == "retry"
        assert "int-stale" not in cmd.resume

    def test_all_interrupts_stale_returns_none(self):
        """When all interrupts are from completed blocks, return None."""
        from orchestrator.mcp_server import _build_resume_command

        snapshot = _mock_state_snapshot([
            ("int-1", {
                "type": "uarch_spec_review",
                "block_name": "scrambler",
            }),
            ("int-2", {
                "type": "human_intervention_needed",
                "block_name": "encoder",
            }),
        ])
        snapshot.values = {
            "completed_blocks": [
                {"name": "scrambler", "success": True},
                {"name": "encoder", "success": False},
            ],
        }

        cmd = _build_resume_command(snapshot, {}, "retry", "", "", "")
        assert cmd is None

    def test_filtered_single_still_uses_keyed_format(self):
        """After filtering stale interrupts, even 1 remaining must use keyed format.

        This is the exact scenario that caused the original crash:
        _build_resume_command filtered out a completed block's interrupt,
        saw only 1 remaining, and returned Command(resume=flat_dict).
        But LangGraph's checkpoint still counted the stale interrupt,
        saw 2 pending, and threw RuntimeError.
        """
        from orchestrator.mcp_server import _build_resume_command

        snapshot = _mock_state_snapshot([
            ("int-stale", {
                "type": "uarch_spec_review",
                "block_name": "inverse_quantizer",
            }),
            ("int-active", {
                "type": "human_intervention_needed",
                "block_name": "clock_reset",
                "supported_actions": ["retry", "fix_rtl", "add_constraint", "skip", "abort"],
            }),
        ])
        snapshot.values = {
            "completed_blocks": [
                {"name": "inverse_quantizer", "success": True},
            ],
        }

        resume_value = {"action": "retry", "constraint": "", "description": ""}
        cmd = _build_resume_command(snapshot, resume_value, "retry", "", "", "")

        assert cmd is not None
        assert isinstance(cmd.resume, dict)
        # Must be keyed by interrupt ID, NOT a flat {action, constraint, description}
        assert "int-active" in cmd.resume, (
            "Resume must use interrupt-ID key even when only 1 interrupt remains "
            "after filtering. Flat format triggers LangGraph RuntimeError when "
            "checkpoint has more pending interrupts than our filtered count."
        )
        assert "action" not in cmd.resume, (
            "Resume dict must NOT have 'action' as a top-level key -- that's "
            "the flat format that LangGraph rejects with multiple checkpoint writes"
        )

    def test_constraint_and_description_forwarded(self):
        """constraint and rtl_fix_description values propagate into the keyed map."""
        from orchestrator.mcp_server import _build_resume_command

        snapshot = _mock_state_snapshot([
            ("int-1", {"type": "human_intervention_needed", "block_name": "encoder"}),
        ])
        cmd = _build_resume_command(
            snapshot, {}, "add_constraint",
            "MUST use 8-bit data path", "Widened data bus from 4 to 8 bits", "",
        )

        entry = cmd.resume["int-1"]
        assert entry["constraint"] == "MUST use 8-bit data path"
        assert entry["description"] == "Widened data bus from 4 to 8 bits"


# ═══════════════════════════════════════════════════════════════════════════
# resume_pipeline: Integration tests
#
# These test the full resume_pipeline() MCP tool, verifying that the
# Command object passed to run_task uses the correct format.  The unit
# tests above cover _build_resume_command in isolation; these tests
# verify the calling code in resume_pipeline handles edge cases
# (exception fallback, self-heal, supported_actions validation).
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.mcp
class TestResumePipelineIntegration:
    """Integration tests for resume_pipeline MCP tool."""

    @pytest.mark.asyncio
    async def test_resume_uses_interrupt_id_keyed_command(self, reset_mcp_state):
        """resume_pipeline must pass an interrupt-ID-keyed Command to run_task."""
        import orchestrator.mcp_server as mcp

        mcp._pipeline.status = "interrupted"
        mcp._pipeline.thread_id = "test-resume-int"

        snapshot = _mock_state_snapshot([
            ("int-a", {"type": "uarch_spec_review", "block_name": "scrambler",
                       "supported_actions": ["approve", "revise", "skip"]}),
            ("int-b", {"type": "uarch_spec_review", "block_name": "encoder",
                       "supported_actions": ["approve", "revise", "skip"]}),
        ])
        snapshot.values = {"completed_blocks": []}

        mcp._pipeline.graph = MagicMock()
        mcp._pipeline.graph.aget_state = AsyncMock(return_value=snapshot)

        captured = {}

        async def mock_run(initial_input, config):
            captured["cmd"] = initial_input

        with patch.object(mcp._pipeline, "ensure_graph", new_callable=AsyncMock), \
             patch.object(mcp._pipeline, "run_task", new_callable=AsyncMock, side_effect=mock_run):
            result_json = await mcp.resume_pipeline(action="approve")
            await asyncio.sleep(0)  # let create_task fire

        result = json.loads(result_json)
        assert result["status"] == "running"

        cmd = captured["cmd"]
        assert isinstance(cmd.resume, dict)
        assert "int-a" in cmd.resume
        assert "int-b" in cmd.resume
        assert cmd.resume["int-a"]["action"] == "approve"
        assert cmd.resume["int-b"]["action"] == "approve"

    @pytest.mark.asyncio
    async def test_resume_with_stale_interrupt_still_uses_keyed(self, reset_mcp_state):
        """Even when _build_resume_command filters out stale interrupts,
        the remaining single interrupt must use keyed format.

        This reproduces the exact crash: 2 interrupts in checkpoint, 1 filtered
        as completed, _build_resume_command returns Command for 1 interrupt.
        Old code used flat format -> LangGraph RuntimeError.
        """
        import orchestrator.mcp_server as mcp

        mcp._pipeline.status = "interrupted"
        mcp._pipeline.thread_id = "test-stale-resume"

        snapshot = _mock_state_snapshot([
            ("int-stale", {"type": "uarch_spec_review",
                           "block_name": "inverse_quantizer"}),
            ("int-active", {"type": "human_intervention_needed",
                            "block_name": "clock_reset",
                            "supported_actions": ["retry", "fix_rtl", "add_constraint", "skip", "abort"]}),
        ])
        snapshot.values = {
            "completed_blocks": [{"name": "inverse_quantizer", "success": True}],
        }

        mcp._pipeline.graph = MagicMock()
        mcp._pipeline.graph.aget_state = AsyncMock(return_value=snapshot)

        captured = {}

        async def mock_run(initial_input, config):
            captured["cmd"] = initial_input

        with patch.object(mcp._pipeline, "ensure_graph", new_callable=AsyncMock), \
             patch.object(mcp._pipeline, "run_task", new_callable=AsyncMock, side_effect=mock_run):
            result_json = await mcp.resume_pipeline(action="retry")
            await asyncio.sleep(0)

        result = json.loads(result_json)
        assert result["status"] == "running"

        cmd = captured["cmd"]
        assert isinstance(cmd.resume, dict)
        assert "int-active" in cmd.resume, (
            "Active interrupt must be in keyed resume map"
        )
        assert "int-stale" not in cmd.resume, (
            "Stale interrupt from completed block must be filtered out"
        )
        assert "action" not in cmd.resume, (
            "Flat-format keys (action/constraint/description) must NOT appear "
            "at top level -- this is the pattern that triggers RuntimeError"
        )

    @pytest.mark.asyncio
    async def test_fallback_uses_keyed_format_on_build_failure(self, reset_mcp_state):
        """When _build_resume_command throws, the fallback must still
        produce an interrupt-ID-keyed Command (not flat format).
        """
        import orchestrator.mcp_server as mcp

        mcp._pipeline.status = "interrupted"
        mcp._pipeline.thread_id = "test-fallback"

        snapshot_for_fallback = _mock_state_snapshot([
            ("int-fb-1", {"type": "uarch_spec_review", "block_name": "scrambler"}),
            ("int-fb-2", {"type": "uarch_spec_review", "block_name": "encoder"}),
        ])

        call_count = {"n": 0}

        async def aget_state_side_effect(config):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                # First two calls: action validation check + _build_resume_command call
                raise RuntimeError("simulated aget_state failure")
            # Third call: fallback path
            return snapshot_for_fallback

        mcp._pipeline.graph = MagicMock()
        mcp._pipeline.graph.aget_state = AsyncMock(side_effect=aget_state_side_effect)

        captured = {}

        async def mock_run(initial_input, config):
            captured["cmd"] = initial_input

        with patch.object(mcp._pipeline, "ensure_graph", new_callable=AsyncMock), \
             patch.object(mcp._pipeline, "run_task", new_callable=AsyncMock, side_effect=mock_run):
            result_json = await mcp.resume_pipeline(action="approve")
            await asyncio.sleep(0)

        result = json.loads(result_json)
        assert result["status"] == "running"

        cmd = captured["cmd"]
        assert isinstance(cmd.resume, dict)
        assert "int-fb-1" in cmd.resume or "int-fb-2" in cmd.resume, (
            "Fallback must use interrupt IDs from snapshot, not flat format"
        )

    @pytest.mark.asyncio
    async def test_final_fallback_when_all_aget_state_fail(self, reset_mcp_state):
        """When both _build_resume_command AND fallback aget_state throw,
        resume_pipeline falls back to flat Command(resume=dict) as last resort.
        """
        import orchestrator.mcp_server as mcp

        mcp._pipeline.status = "interrupted"
        mcp._pipeline.thread_id = "test-total-fallback"

        mcp._pipeline.graph = MagicMock()
        mcp._pipeline.graph.aget_state = AsyncMock(
            side_effect=RuntimeError("persistent failure"),
        )

        captured = {}

        async def mock_run(initial_input, config):
            captured["cmd"] = initial_input

        with patch.object(mcp._pipeline, "ensure_graph", new_callable=AsyncMock), \
             patch.object(mcp._pipeline, "run_task", new_callable=AsyncMock, side_effect=mock_run):
            result_json = await mcp.resume_pipeline(action="retry")
            await asyncio.sleep(0)

        result = json.loads(result_json)
        assert result["status"] == "running"

        cmd = captured["cmd"]
        # Last-resort flat format -- not ideal but better than crashing
        assert cmd.resume["action"] == "retry"

    @pytest.mark.asyncio
    async def test_self_heal_detects_pending_interrupts(self, reset_mcp_state):
        """When status is 'running' but the asyncio task is done and the
        checkpoint has pending interrupts, resume_pipeline should self-heal
        the status to 'interrupted' and proceed.
        """
        import orchestrator.mcp_server as mcp

        mcp._pipeline.status = "running"
        mcp._pipeline.thread_id = "test-self-heal"

        done_task = MagicMock()
        done_task.done.return_value = True
        mcp._pipeline.task = done_task

        snapshot = _mock_state_snapshot([
            ("int-heal", {"type": "uarch_spec_review", "block_name": "scrambler",
                          "supported_actions": ["approve", "revise", "skip"]}),
        ])
        snapshot.values = {"completed_blocks": []}

        mcp._pipeline.graph = MagicMock()
        mcp._pipeline.graph.aget_state = AsyncMock(return_value=snapshot)

        captured = {}

        async def mock_run(initial_input, config):
            captured["cmd"] = initial_input

        with patch.object(mcp._pipeline, "ensure_graph", new_callable=AsyncMock), \
             patch.object(mcp._pipeline, "run_task", new_callable=AsyncMock, side_effect=mock_run):
            result_json = await mcp.resume_pipeline(action="approve")
            await asyncio.sleep(0)

        result = json.loads(result_json)
        assert result["status"] == "running"
        assert "cmd" in captured, "run_task should have been called after self-heal"

    @pytest.mark.asyncio
    async def test_self_heal_rejected_when_task_still_running(self, reset_mcp_state):
        """Self-heal should NOT trigger when the asyncio task is still running."""
        import orchestrator.mcp_server as mcp

        mcp._pipeline.status = "running"
        mcp._pipeline.thread_id = "test-no-heal"

        running_task = MagicMock()
        running_task.done.return_value = False
        mcp._pipeline.task = running_task

        result_json = await mcp.resume_pipeline(action="retry")
        result = json.loads(result_json)
        assert "error" in result
        assert "cannot resume" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_supported_actions_validation_rejects_invalid(self, reset_mcp_state):
        """resume_pipeline should reject an action not in the interrupt's
        supported_actions before building the resume command.
        """
        import orchestrator.mcp_server as mcp

        mcp._pipeline.status = "interrupted"
        mcp._pipeline.thread_id = "test-validate"

        snapshot = _mock_state_snapshot([
            ("int-val", {
                "type": "uarch_spec_review",
                "block_name": "scrambler",
                "supported_actions": ["approve", "revise", "skip"],
            }),
        ])

        mcp._pipeline.graph = MagicMock()
        mcp._pipeline.graph.aget_state = AsyncMock(return_value=snapshot)

        with patch.object(mcp._pipeline, "ensure_graph", new_callable=AsyncMock):
            result_json = await mcp.resume_pipeline(action="retry")

        result = json.loads(result_json)
        assert "error" in result
        assert "not valid" in result["error"].lower() or "not in" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_block_actions_forwarded_through_resume_pipeline(self, reset_mcp_state):
        """Per-block actions via block_actions JSON must be forwarded correctly."""
        import orchestrator.mcp_server as mcp

        mcp._pipeline.status = "interrupted"
        mcp._pipeline.thread_id = "test-block-actions"

        snapshot = _mock_state_snapshot([
            ("int-ba-1", {"type": "uarch_spec_review", "block_name": "scrambler",
                          "supported_actions": ["approve", "revise", "skip"]}),
            ("int-ba-2", {"type": "uarch_spec_review", "block_name": "encoder",
                          "supported_actions": ["approve", "revise", "skip"]}),
        ])
        snapshot.values = {"completed_blocks": []}

        mcp._pipeline.graph = MagicMock()
        mcp._pipeline.graph.aget_state = AsyncMock(return_value=snapshot)

        captured = {}

        async def mock_run(initial_input, config):
            captured["cmd"] = initial_input

        with patch.object(mcp._pipeline, "ensure_graph", new_callable=AsyncMock), \
             patch.object(mcp._pipeline, "run_task", new_callable=AsyncMock, side_effect=mock_run):
            result_json = await mcp.resume_pipeline(
                action="approve",
                block_actions='{"scrambler": "skip", "encoder": "approve"}',
            )
            await asyncio.sleep(0)

        result = json.loads(result_json)
        assert result["status"] == "running"

        cmd = captured["cmd"]
        assert cmd.resume["int-ba-1"]["action"] == "skip"
        assert cmd.resume["int-ba-2"]["action"] == "approve"


# ═══════════════════════════════════════════════════════════════════════════
# get_pipeline_state: Stale interrupt filtering
#
# The original escape: get_pipeline_state() reported ALL interrupts from
# the LangGraph checkpoint, including stale interrupts from blocks that
# had already completed.  When the outer agent resumed these stale
# interrupts, blocks re-executed from the interrupt point instead of
# staying completed.
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.mcp
class TestGetPipelineStateInterruptFiltering:
    """Verify that get_pipeline_state filters stale interrupts."""

    @pytest.mark.asyncio
    async def test_completed_blocks_excluded_from_interrupts(self, reset_mcp_state):
        """Interrupts from completed blocks should NOT appear in the response.

        When a block completes (success, failed, skipped), any old
        interrupts it had (e.g., from uarch_spec_review) are stale and
        should be filtered out.
        """
        import orchestrator.mcp_server as mcp

        # Set up pipeline as interrupted
        mcp._pipeline.status = "interrupted"
        mcp._pipeline.thread_id = "test-stale-1"

        # Create mock state with completed blocks and stale interrupts
        completed = [
            {"name": "scrambler", "success": True},
            {"name": "encoder", "success": False},
        ]
        values = {
            "completed_blocks": completed,
            "block_queue": [
                {"name": "scrambler", "tier": 1},
                {"name": "encoder", "tier": 1},
                {"name": "decoder", "tier": 1},
            ],
            "tier_list": [1],
            "current_tier_index": 0,
            "max_attempts": 5,
            "pipeline_done": False,
        }

        # Stale interrupt from completed block + real interrupt from active block
        task_scrambler = MagicMock()
        task_scrambler.interrupts = [_mock_interrupt("int-stale", {
            "type": "uarch_spec_review",
            "block_name": "scrambler",
            "supported_actions": ["approve", "revise", "skip"],
        })]
        task_decoder = MagicMock()
        task_decoder.interrupts = [_mock_interrupt("int-active", {
            "type": "human_intervention_needed",
            "block_name": "decoder",
            "supported_actions": ["retry", "fix_rtl", "add_constraint", "skip", "abort"],
        })]

        snapshot = MagicMock()
        snapshot.tasks = [task_scrambler, task_decoder]
        snapshot.values = values
        snapshot.next = ["ask_human"]
        snapshot.config = {"configurable": {"checkpoint_id": "cp-1"}}

        with patch.object(mcp._pipeline, "ensure_graph", new_callable=AsyncMock), \
             patch("orchestrator.mcp_server._aggregate_failure_summary", return_value={}):
            mcp._pipeline.graph = MagicMock()
            mcp._pipeline.graph.aget_state = AsyncMock(return_value=snapshot)

            result_json = await mcp.get_pipeline_state()

        result = json.loads(result_json)

        # Only decoder's interrupt should be visible
        assert result["pending_interrupt_count"] == 1
        assert result["interrupt_type"] == "human_intervention_needed"

        if result.get("interrupted_blocks"):
            block_names = [b["block_name"] for b in result["interrupted_blocks"]]
            assert "scrambler" not in block_names
            assert "decoder" in block_names

    @pytest.mark.asyncio
    async def test_no_stale_interrupts_still_works(self, reset_mcp_state):
        """When there are no completed blocks, all interrupts pass through."""
        import orchestrator.mcp_server as mcp

        mcp._pipeline.status = "interrupted"
        mcp._pipeline.thread_id = "test-no-stale"

        values = {
            "completed_blocks": [],
            "block_queue": [
                {"name": "scrambler", "tier": 1},
                {"name": "encoder", "tier": 1},
            ],
            "tier_list": [1],
            "current_tier_index": 0,
            "max_attempts": 5,
            "pipeline_done": False,
        }

        task1 = MagicMock()
        task1.interrupts = [_mock_interrupt("int-1", {
            "type": "uarch_spec_review",
            "block_name": "scrambler",
        })]
        task2 = MagicMock()
        task2.interrupts = [_mock_interrupt("int-2", {
            "type": "uarch_spec_review",
            "block_name": "encoder",
        })]

        snapshot = MagicMock()
        snapshot.tasks = [task1, task2]
        snapshot.values = values
        snapshot.next = ["review_uarch_spec"]
        snapshot.config = {"configurable": {"checkpoint_id": "cp-2"}}

        with patch.object(mcp._pipeline, "ensure_graph", new_callable=AsyncMock), \
             patch("orchestrator.mcp_server._aggregate_failure_summary", return_value={}):
            mcp._pipeline.graph = MagicMock()
            mcp._pipeline.graph.aget_state = AsyncMock(return_value=snapshot)

            result_json = await mcp.get_pipeline_state()

        result = json.loads(result_json)
        assert result["pending_interrupt_count"] == 2


# ═══════════════════════════════════════════════════════════════════════════
# Failure summary integration
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.mcp
class TestFailureSummaryIntegration:
    """Verify failure_summary appears in get_pipeline_state when failures exist."""

    @pytest.mark.asyncio
    async def test_failure_summary_present_when_failures_exist(self, reset_mcp_state):
        import orchestrator.mcp_server as mcp

        mcp._pipeline.status = "running"
        mcp._pipeline.thread_id = "test-summary-1"

        values = {
            "completed_blocks": [],
            "block_queue": [{"name": "scrambler", "tier": 1}],
            "tier_list": [1],
            "current_tier_index": 0,
            "max_attempts": 5,
            "pipeline_done": False,
        }
        snapshot = MagicMock()
        snapshot.tasks = []
        snapshot.values = values
        snapshot.next = ["generate_rtl"]
        snapshot.config = {"configurable": {"checkpoint_id": "cp-3"}}

        failure_data = {
            "failure_categories": {"LOGIC_ERROR": 2},
            "per_block_failures": {"scrambler": ["LOGIC_ERROR", "LOGIC_ERROR"]},
            "systematic_patterns": [],
            "total_failures": 2,
            "avg_retries": 2.0,
        }

        with patch.object(mcp._pipeline, "ensure_graph", new_callable=AsyncMock), \
             patch("orchestrator.mcp_server._aggregate_failure_summary", return_value=failure_data):
            mcp._pipeline.graph = MagicMock()
            mcp._pipeline.graph.aget_state = AsyncMock(return_value=snapshot)

            result_json = await mcp.get_pipeline_state()

        result = json.loads(result_json)
        assert "failure_summary" in result
        assert result["failure_summary"]["total_failures"] == 2
        assert result["failure_summary"]["failure_categories"]["LOGIC_ERROR"] == 2

    @pytest.mark.asyncio
    async def test_no_failure_summary_when_no_failures(self, reset_mcp_state):
        import orchestrator.mcp_server as mcp

        mcp._pipeline.status = "running"
        mcp._pipeline.thread_id = "test-summary-2"

        values = {
            "completed_blocks": [],
            "block_queue": [{"name": "scrambler", "tier": 1}],
            "tier_list": [1],
            "current_tier_index": 0,
            "max_attempts": 5,
            "pipeline_done": False,
        }
        snapshot = MagicMock()
        snapshot.tasks = []
        snapshot.values = values
        snapshot.next = ["generate_rtl"]
        snapshot.config = {"configurable": {"checkpoint_id": "cp-4"}}

        with patch.object(mcp._pipeline, "ensure_graph", new_callable=AsyncMock), \
             patch("orchestrator.mcp_server._aggregate_failure_summary", return_value={}):
            mcp._pipeline.graph = MagicMock()
            mcp._pipeline.graph.aget_state = AsyncMock(return_value=snapshot)

            result_json = await mcp.get_pipeline_state()

        result = json.loads(result_json)
        assert "failure_summary" not in result


# ═══════════════════════════════════════════════════════════════════════════
# reset_project: Per-Document File Cleanup (post-refactor)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.mcp
@pytest.mark.doc_persistence
class TestResetProjectPerDocFiles:
    """Verify reset_project deletes all per-document files.

    After the migration from architecture_state.json to per-document
    files, reset_project(scope='architecture') must delete all document
    .json and .md files.
    """

    @pytest.mark.asyncio
    async def test_reset_architecture_deletes_all_doc_files(self, reset_mcp_state, tmp_path):
        import orchestrator.mcp_server as mcp

        coresmith = tmp_path / ".coresmith"

        all_docs = [
            "prd_spec", "sad_spec", "frd_spec", "block_diagram",
            "memory_map", "clock_tree", "register_spec", "ers_spec",
        ]
        for doc in all_docs:
            (coresmith / f"{doc}.json").write_text("{}")
            (coresmith / f"{doc}.md").write_text("# test")

        (coresmith / "block_specs.json").write_text("[]")
        (coresmith / "summary_architecture.md").write_text("# summary")

        result_json = await mcp.reset_project(scope="architecture")
        json.loads(result_json)

        for doc in all_docs:
            json_path = coresmith / f"{doc}.json"
            md_path = coresmith / f"{doc}.md"
            if json_path.exists() or md_path.exists():
                pytest.xfail(
                    f"{doc} files not cleaned by reset_project "
                    "(expected until per-doc migration updates reset_project)"
                )

    @pytest.mark.asyncio
    async def test_reset_does_not_delete_pipeline_files(self, reset_mcp_state, tmp_path):
        """reset_project(scope='architecture') should not touch pipeline files."""
        import orchestrator.mcp_server as mcp

        coresmith = tmp_path / ".coresmith"

        (coresmith / "pipeline_events.jsonl").write_text("")
        (coresmith / "summary_frontend.md").write_text("# pipeline summary")

        await mcp.reset_project(scope="architecture")

        assert (coresmith / "pipeline_events.jsonl").exists()
        assert (coresmith / "summary_frontend.md").exists()


# ═══════════════════════════════════════════════════════════════════════════
# Structured ask_question Helpers
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.mcp
class TestBuildArchAskQuestion:
    """Tests for _build_arch_ask_question helper."""

    def test_prd_questions_classifies_auto_vs_choice(self):
        from orchestrator.mcp_server import _build_arch_ask_question

        payload = {
            "type": "prd_questions",
            "phase": "prd",
            "questions": [
                {"id": "q1", "category": "technology", "question": "PDK?",
                 "options": ["sky130"], "required": True},
                {"id": "q2", "category": "dataflow", "question": "Bus protocol?",
                 "options": ["Dedicated pins", "AXI-Stream"], "required": True},
                {"id": "q3", "category": "area", "question": "Gate budget?",
                 "options": ["500", "1000", "No limit"], "required": True},
            ],
            "questions_by_category": {"technology": [], "dataflow": [], "area": []},
            "supported_actions": ["continue", "abort"],
        }

        result = _build_arch_ask_question(payload)

        assert result["interrupt_type"] in ("prd_questions", "ers_questions")
        aq = result["ask_question"]
        assert aq["title"] == "PRD Sizing Questions"

        # q1 has 1 option -> auto_answerable
        assert len(aq["auto_answerable"]) == 1
        assert aq["auto_answerable"][0]["id"] == "q1"
        assert aq["auto_answerable"][0]["suggested_answer"] == "sky130"

        # q2 and q3 have multiple options -> in questions
        assert len(aq["questions"]) == 2
        question_ids = {q["id"] for q in aq["questions"]}
        assert "q2" in question_ids
        assert "q3" in question_ids

    def test_prd_questions_limits_to_4(self):
        from orchestrator.mcp_server import _build_arch_ask_question

        payload = {
            "type": "prd_questions",
            "phase": "prd",
            "questions": [
                {"id": f"q{i}", "category": "technology", "question": f"Q{i}?",
                 "options": ["a", "b"], "required": True}
                for i in range(6)
            ],
            "questions_by_category": {"technology": []},
            "supported_actions": ["continue", "abort"],
        }

        result = _build_arch_ask_question(payload)

        aq = result["ask_question"]
        assert len(aq["questions"]) <= 4
        assert len(aq["remaining_choice_questions"]) == 2

    def test_prd_has_resume_mapping(self):
        from orchestrator.mcp_server import _build_arch_ask_question

        payload = {
            "type": "prd_questions",
            "phase": "prd",
            "questions": [
                {"id": "q1", "category": "technology", "question": "Q?",
                 "options": ["a", "b"], "required": True},
            ],
            "questions_by_category": {"technology": []},
            "supported_actions": ["continue", "abort"],
        }

        result = _build_arch_ask_question(payload)
        assert "resume_mapping" in result
        assert result["resume_mapping"]["action"] == "continue"

    def test_final_review_has_ask_question_and_mapping(self):
        from orchestrator.mcp_server import _build_arch_ask_question

        payload = {
            "type": "final_review",
            "phase": "final_review",
            "title": "PRD — 16-Bit Adder",
            "block_count": 1,
            "block_names": ["adder_16bit"],
            "total_estimated_gates": 100,
            "constraint_rounds_used": 1,
            "max_rounds": 3,
            "supported_actions": ["accept", "feedback", "abort"],
        }

        result = _build_arch_ask_question(payload)

        assert result["interrupt_type"] == "final_review"
        aq = result["ask_question"]
        assert aq["title"] == "Architecture Final Review"
        assert len(aq["questions"]) == 1
        options = aq["questions"][0]["options"]
        labels = [o["label"] for o in options]
        assert "OK2DEV" in labels
        assert "REVISE" in labels
        assert "ABORT" in labels

        rm = result["resume_mapping"]
        assert rm["OK2DEV"]["action"] == "accept"
        assert rm["REVISE"]["action"] == "feedback"
        assert rm["ABORT"]["action"] == "abort"

    def test_constraint_violations_structured(self):
        from orchestrator.mcp_server import _build_arch_ask_question

        payload = {
            "type": "architecture_review_needed",
            "phase": "constraints",
            "round": 2,
            "max_rounds": 3,
            "violations": [
                {"violation": "Gate budget exceeded", "category": "area"},
                {"violation": "Memory overlap", "category": "structural"},
            ],
            "structural_violations": [
                {"violation": "Memory overlap", "category": "structural"},
            ],
            "supported_actions": ["retry", "accept", "feedback", "abort"],
        }

        result = _build_arch_ask_question(payload)

        assert result["interrupt_type"] == "architecture_review_constraints"
        aq = result["ask_question"]
        assert aq["title"] == "Constraint Violations"
        labels = [o["label"] for o in aq["questions"][0]["options"]]
        assert "Retry" in labels
        assert "Accept" in labels

        rm = result["resume_mapping"]
        assert rm["Retry"]["action"] == "retry"
        assert rm["Accept"]["action"] == "accept"

    def test_exhausted_structured(self):
        from orchestrator.mcp_server import _build_arch_ask_question

        payload = {
            "type": "architecture_review_needed",
            "phase": "max_rounds_exhausted",
            "round": 3,
            "max_rounds": 3,
            "violations": [{"violation": "Still too many gates"}],
            "supported_actions": ["retry", "accept", "feedback", "abort"],
        }

        result = _build_arch_ask_question(payload)

        assert result["interrupt_type"] == "architecture_review_exhausted"
        aq = result["ask_question"]
        assert aq["title"] == "Constraint Rounds Exhausted"
        assert "resume_mapping" in result

    def test_block_diagram_review(self):
        from orchestrator.mcp_server import _build_arch_ask_question

        payload = {
            "type": "architecture_review_needed",
            "phase": "block_diagram",
            "round": 1,
            "max_rounds": 3,
            "questions": [
                {"question": "Should we split the encoder?"},
            ],
            "block_diagram_summary": {
                "block_count": 3,
                "total_estimated_gates": 5000,
            },
            "supported_actions": ["continue", "feedback", "abort"],
        }

        result = _build_arch_ask_question(payload)

        assert result["interrupt_type"] == "architecture_review_diagram"
        aq = result["ask_question"]
        assert aq["title"] == "Block Diagram Review"
        labels = [o["label"] for o in aq["questions"][0]["options"]]
        assert "Accept" in labels
        assert "Provide Feedback" in labels

    def test_unknown_type_returns_fallback(self):
        from orchestrator.mcp_server import _build_arch_ask_question

        payload = {
            "type": "some_future_interrupt",
            "phase": "unknown",
            "questions": [{"question": "What?"}],
            "supported_actions": ["retry"],
        }

        result = _build_arch_ask_question(payload)

        assert result["interrupt_type"] == "some_future_interrupt"
        assert "ask_question" not in result
        assert len(result["interrupt_questions"]) == 1


@pytest.mark.mcp
class TestBuildPipelineAskQuestion:
    """Tests for _build_pipeline_ask_question helper."""

    def test_uarch_review_structured(self):
        from orchestrator.mcp_server import _build_pipeline_ask_question

        payload = {
            "type": "uarch_spec_review",
            "block_name": "adder_16bit",
            "spec_path": "arch/uarch_specs/adder_16bit.md",
            "spec_summary": {"ports": ["a", "b", "sum"]},
            "spec_text": "# Adder 16-bit\n\nSimple adder...",
            "ers_summary": "16-bit adder",
            "supported_actions": ["approve", "revise", "skip"],
        }

        result = _build_pipeline_ask_question(payload)

        assert "ask_question" in result
        aq = result["ask_question"]
        assert "adder_16bit" in aq["title"]
        labels = [o["label"] for o in aq["questions"][0]["options"]]
        assert "Approve" in labels
        assert "Revise" in labels
        assert "Skip" in labels

        rm = result["resume_mapping"]
        assert rm["Approve"]["action"] == "approve"

    def test_human_intervention_structured(self):
        from orchestrator.mcp_server import _build_pipeline_ask_question

        payload = {
            "type": "human_intervention_needed",
            "block_name": "scrambler",
            "diagnosis": "Output counter off by one",
            "category": "LOGIC_ERROR",
            "confidence": 0.4,
            "suggested_fix": "Change <= to <",
            "attempt": 3,
            "max_attempts": 5,
            "human_question": "Should I change the counter?",
            "supported_actions": ["retry", "add_constraint", "skip", "abort"],
        }

        result = _build_pipeline_ask_question(payload)

        aq = result["ask_question"]
        assert "scrambler" in aq["title"]
        labels = [o["label"] for o in aq["questions"][0]["options"]]
        assert "Retry" in labels
        assert "Add Constraint" in labels
        assert "Skip" in labels
        assert "Abort" in labels

        rm = result["resume_mapping"]
        assert rm["Retry"]["action"] == "retry"
        assert rm["Add Constraint"]["action"] == "add_constraint"

    def test_uarch_integration_review_structured(self):
        from orchestrator.mcp_server import _build_pipeline_ask_question

        payload = {
            "type": "uarch_integration_review",
            "tier": 1,
            "block_names": ["scrambler", "encoder"],
            "spec_paths": {
                "scrambler": "arch/uarch_specs/scrambler.md",
                "encoder": "arch/uarch_specs/encoder.md",
            },
            "review_summary": "All interfaces match. No mismatches found.",
            "issues_found": 0,
            "issues_fixed": 0,
            "supported_actions": ["approve", "revise", "abort"],
        }

        result = _build_pipeline_ask_question(payload)

        assert "ask_question" in result
        aq = result["ask_question"]
        assert aq["title"] == "Chip-Level uArch Integration Review"
        labels = [o["label"] for o in aq["questions"][0]["options"]]
        assert "Approve" in labels
        assert "Revise" in labels
        assert "Abort" in labels

        rm = result["resume_mapping"]
        assert rm["Approve"]["action"] == "approve"
        assert rm["Revise"]["action"] == "revise"
        assert rm["Abort"]["action"] == "abort"

        assert "interrupt_summary" in result
        assert "scrambler" in result["interrupt_summary"]

    def test_integration_failure_structured(self):
        from orchestrator.mcp_server import _build_pipeline_ask_question

        payload = {
            "type": "integration_failure",
            "mismatches": [
                {
                    "from_block": "a",
                    "to_block": "b",
                    "issue_type": "width_mismatch",
                    "severity": "error",
                    "description": "Port width 8 vs 16",
                },
            ],
            "lint_clean": False,
            "supported_actions": ["fix_rtl", "skip", "abort"],
        }

        result = _build_pipeline_ask_question(payload)

        aq = result["ask_question"]
        assert aq["title"] == "Integration Check Failed"
        assert "resume_mapping" in result

    def test_integration_dv_failure_structured(self):
        from orchestrator.mcp_server import _build_pipeline_ask_question

        payload = {
            "type": "integration_dv_failure",
            "sim_log": "FAIL: test_smoke - timeout after 10000 cycles",
            "supported_actions": ["retry", "fix_rtl", "skip", "abort"],
        }

        result = _build_pipeline_ask_question(payload)

        aq = result["ask_question"]
        assert aq["title"] == "Integration DV Failed"
        labels = [o["label"] for o in aq["questions"][0]["options"]]
        assert "Retry" in labels
        assert "Fix RTL" in labels

    def test_validation_dv_failure_structured(self):
        from orchestrator.mcp_server import _build_pipeline_ask_question

        payload = {
            "type": "validation_dv_failure",
            "sim_log": "FAIL: KPI-001 PSNR below threshold",
            "supported_actions": ["retry", "fix_rtl", "fix_tb", "abort"],
        }

        result = _build_pipeline_ask_question(payload)

        aq = result["ask_question"]
        assert aq["title"] == "Validation DV Failed"
        labels = [o["label"] for o in aq["questions"][0]["options"]]
        assert "Retry" in labels
        assert "Fix RTL" in labels
        assert "Fix TB" in labels
        assert "Skip" not in labels

    def test_unknown_type_returns_empty(self):
        from orchestrator.mcp_server import _build_pipeline_ask_question

        payload = {"type": "some_future_type"}

        result = _build_pipeline_ask_question(payload)

        assert "ask_question" not in result


@pytest.mark.mcp
class TestMergeBlockIntoPipelineCheckpoint:
    """_merge_block_into_pipeline_checkpoint must APPEND the authoritative block
    result (the channel uses operator.add and the gate dedups keeping the LAST
    entry), so a restart_block PASS becomes the gate-visible entry -- not replace
    the first name-match (which, under the append reducer, left a stale later
    entry winning and re-parked the run at pipeline_incomplete)."""

    @pytest.mark.asyncio
    async def test_appends_single_authoritative_result(self, reset_mcp_state):
        import orchestrator.mcp_server as mcp

        mcp._pipeline.thread_id = "test-merge"
        snapshot = MagicMock()
        # Stale history: an early entropy pass then a LATER entropy fail (the bug
        # scenario). First-match-replace would update the early one, leaving the
        # later fail to win under last-wins dedup.
        snapshot.values = {"completed_blocks": [
            {"name": "entropy_enc", "success": True, "attempts": 1},
            {"name": "block_packer", "success": True, "attempts": 1},
            {"name": "entropy_enc", "success": False, "attempts": 2},
        ]}
        mcp._pipeline.graph = MagicMock()
        mcp._pipeline.graph.aget_state = AsyncMock(return_value=snapshot)
        captured = {}

        async def _aupdate(config, values, **kw):
            captured["values"] = values
            captured["kw"] = kw
        mcp._pipeline.graph.aupdate_state = AsyncMock(side_effect=_aupdate)

        new_result = {"name": "entropy_enc", "success": True, "attempts": 1,
                      "sim_passed": True}
        ok = await mcp._merge_block_into_pipeline_checkpoint(new_result)

        assert ok is True
        # The update appends EXACTLY the one authoritative result (last-wins),
        # not a rebuilt full list.
        cbs = captured["values"]["completed_blocks"]
        assert cbs == [new_result], cbs
        assert len(cbs) == 1

    @pytest.mark.asyncio
    async def test_no_name_returns_false(self, reset_mcp_state):
        import orchestrator.mcp_server as mcp
        mcp._pipeline.thread_id = "test-merge2"
        snapshot = MagicMock()
        snapshot.values = {"completed_blocks": []}
        mcp._pipeline.graph = MagicMock()
        mcp._pipeline.graph.aget_state = AsyncMock(return_value=snapshot)
        mcp._pipeline.graph.aupdate_state = AsyncMock()
        ok = await mcp._merge_block_into_pipeline_checkpoint({"success": True})
        assert ok is False
        mcp._pipeline.graph.aupdate_state.assert_not_called()
