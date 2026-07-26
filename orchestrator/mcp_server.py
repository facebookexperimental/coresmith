# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
MCP Server -- exposes architecture, pipeline, and backend tools via MCP.

This is the primary entry point for Claude CLI / Cursor users. All three
LangGraph state machines (architecture, frontend pipeline, backend) run
as autonomous background tasks with interrupt-based human-in-the-loop.

Architecture lifecycle (same pattern as pipeline):
  1. start_architecture    -- launch background task
  2. get_architecture_state -- poll for progress / detect interrupts
  3. resume_architecture   -- respond to interrupts (feedback, accept, abort)
  4. pause_architecture    -- cancel background task
  5. restart_architecture_node -- fork from historical checkpoint

The architecture graph runs autonomously through:
  Gather Requirements (PRD) -> Escalate PRD (user answers sizing questions)
  -> System Architecture (SAD) -> Functional Requirements (FRD)
  -> Block Diagram (LLM) -> review -> Memory Map -> Clock Tree -> Register Spec
  -> Constraint Check -> route -> Finalize / Escalate / Iterate

Four interrupt points escalate to the outer agent:
  - PRD sizing questions (user must answer before architecture proceeds)
  - Block diagram questions/ambiguities
  - Structural constraint violations
  - Max iteration rounds exhausted

Usage:
    # Start the MCP server (stdio transport for Claude CLI)
    python -m orchestrator.mcp_server

    # Or add to Claude CLI config:
    # ~/.claude/config.json -> mcpServers -> coresmith-architecture
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

# Architecture, pipeline, and backend graphs all run as autonomous background
# tasks. The outer agent monitors via get_*_state() and responds to interrupts
# via resume_*().

# ---------------------------------------------------------------------------
# Resolve project root & initialise telemetry (must run before graph imports)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.environ.get(
    "CORESMITH_PROJECT_ROOT",
    str(Path(__file__).resolve().parent.parent),
)
os.environ["CORESMITH_PROJECT_ROOT"] = _PROJECT_ROOT
_TELEMETRY_ROOT = os.environ.get("CORESMITH_TELEMETRY_ROOT", _PROJECT_ROOT)

# A-Fix 1: seed profile flag defaults before importing graph code.
from orchestrator.profile import apply as _apply_profile

_apply_profile()

from orchestrator.architecture.state import ARCH_DOC_DIR
from orchestrator.telemetry import init_telemetry

init_telemetry(_TELEMETRY_ROOT)

# Register the observability LLM hook (fires after every graph_node_exit)
from orchestrator.langgraph.event_stream import register_exit_hook
from orchestrator.langgraph.observer import observer_hook

register_exit_hook(observer_hook)


def _project_root() -> str:
    return _PROJECT_ROOT


def _get_diagnostics(graph_filter: str = "") -> dict:
    """Query traces.db for failure diagnostics to include in MCP responses."""
    from orchestrator.telemetry.reader import get_failure_diagnostics
    db_path = os.path.join(_project_root(), ".coresmith", "traces.db")
    if not os.path.exists(db_path):
        return {}
    try:
        return get_failure_diagnostics(db_path, graph_filter=graph_filter)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# MCP Server setup
# ---------------------------------------------------------------------------

server = FastMCP("coresmith-architecture")


# ---------------------------------------------------------------------------
# GraphLifecycle -- unified lifecycle manager for all three graphs
#
# Extracted to orchestrator/graph_lifecycle.py so the FastAPI daemon
# (orchestrator/daemon/server.py) can reuse the same start/resume/pause
# semantics. The MCP server pins ``project_root`` to ``_PROJECT_ROOT`` via
# the thin subclass below.
# ---------------------------------------------------------------------------

from orchestrator.graph_lifecycle import GraphLifecycle as _GraphLifecycle


class GraphLifecycle(_GraphLifecycle):
    """MCP-flavoured subclass: defaults ``project_root`` to ``_PROJECT_ROOT``."""

    def __init__(self, name: str, checkpoint_db: str, builder_fn_path: str, builder_fn_name: str):
        super().__init__(
            name=name,
            checkpoint_db=checkpoint_db,
            builder_fn_path=builder_fn_path,
            builder_fn_name=builder_fn_name,
            project_root=_PROJECT_ROOT,
        )


# --- Architecture runner ---
_architecture = GraphLifecycle(
    name="architecture",
    checkpoint_db=os.path.join(_PROJECT_ROOT, ".coresmith", "architecture_checkpoint.db"),
    builder_fn_path="orchestrator.langgraph.architecture_graph",
    builder_fn_name="build_architecture_graph",
)
_ARCH_CHECKPOINT_DB = _architecture.checkpoint_db

# --- Frontend (pipeline) runner ---
_pipeline = GraphLifecycle(
    name="pipeline",
    checkpoint_db=os.path.join(_PROJECT_ROOT, ".coresmith", "pipeline_checkpoint.db"),
    builder_fn_path="orchestrator.langgraph.pipeline_graph",
    builder_fn_name="build_pipeline_graph",
)
_CHECKPOINT_DB = _pipeline.checkpoint_db

# --- Backend runner ---
_backend = GraphLifecycle(
    name="backend",
    checkpoint_db=os.path.join(_PROJECT_ROOT, ".coresmith", "backend_checkpoint.db"),
    builder_fn_path="orchestrator.langgraph.backend_graph",
    builder_fn_name="build_backend_graph",
)
_BACKEND_CHECKPOINT_DB = _backend.checkpoint_db

# --- Tapeout runner ---
_tapeout = GraphLifecycle(
    name="tapeout",
    checkpoint_db=os.path.join(_PROJECT_ROOT, ".coresmith", "tapeout_checkpoint.db"),
    builder_fn_path="orchestrator.langgraph.tapeout_graph",
    builder_fn_name="build_tapeout_graph",
)
_TAPEOUT_CHECKPOINT_DB = _tapeout.checkpoint_db


# ═══════════════════════════════════════════════════════════════════════════
# ARCHITECTURE TOOLS -- autonomous background task with interrupt escalation
#
# The architecture graph runs autonomously. The outer agent monitors via
# get_architecture_state() and responds to interrupts via resume_architecture().
# ═══════════════════════════════════════════════════════════════════════════


# ---------------------------------------------------------------------------
# Tool: start_architecture
# ---------------------------------------------------------------------------


@server.tool()
async def start_architecture(
    requirements: str = "",
    target_clock_mhz: float = 50.0,
    pdk_config_path: str = "",
    max_rounds: int = 3,
) -> str:
    """Start the architecture graph as a background task.

    Loads requirements and PDK config, builds the initial state, and launches
    the LangGraph architecture graph. The graph runs autonomously -- it only
    pauses when an escalation point fires for human review (interrupt).

    Call get_architecture_state() to monitor progress.

    Args:
        requirements: High-level ASIC requirements text.
        target_clock_mhz: Target clock frequency in MHz (default 50.0).
        pdk_config_path: Path to PDK YAML config (relative to project root).
        max_rounds: Maximum constraint iteration rounds (default 3).
    """
    if _architecture.status == "running":
        return json.dumps({
            "error": "Architecture graph already running",
            "thread_id": _architecture.thread_id,
            "status": _architecture.status,
        })

    await _architecture.ensure_graph()

    from orchestrator.architecture.state import load_state, save_state
    from orchestrator.pdk import PDKConfig

    # Load or initialize architecture state from disk
    arch_state = load_state(_project_root())

    if requirements:
        arch_state.requirements = requirements
    if not arch_state.requirements:
        return json.dumps({"error": "No requirements provided."})

    arch_state.target_clock_mhz = target_clock_mhz

    # Load PDK config
    pdk_summary = "No PDK configured"
    if pdk_config_path:
        pdk_path = Path(_project_root()) / pdk_config_path
        if pdk_path.exists():
            pdk = PDKConfig.from_yaml(str(pdk_path))
            arch_state.pdk_config = pdk.to_dict()
    if arch_state.pdk_config:
        pdk = PDKConfig.from_dict(arch_state.pdk_config)
        pdk_summary = pdk.to_summary()

    save_state(arch_state, _project_root())

    await _architecture.reset_for_new_run()

    graph_config = {
        "configurable": {"thread_id": _architecture.thread_id},
    }

    initial_state = {
        "project_root": _project_root(),
        "requirements": arch_state.requirements,
        "pdk_summary": pdk_summary,
        "target_clock_mhz": target_clock_mhz,
        "pdk_config": arch_state.pdk_config or {},
        "max_rounds": max_rounds,
        "round": 1,
        "phase": "prd",
        "prd_spec": None,
        "prd_questions": None,
        "sad_spec": None,
        "frd_spec": None,
        "ers_spec": None,
        "violations_history": [],
        "questions": [],
        "block_diagram": None,
        "memory_map": None,
        "clock_tree": None,
        "register_spec": None,
        "benchmark_data": arch_state.benchmark_results or None,
        "constraint_result": None,
        "human_feedback": arch_state.human_feedback or "",
        "human_response": None,
        "success": False,
        "error": "",
        "block_specs_path": "",
    }

    _architecture.task = asyncio.create_task(
        _architecture.run_task(initial_state, graph_config)
    )
    await asyncio.sleep(0.1)

    result = {
        "status": _architecture.status,
        "thread_id": _architecture.thread_id,
        "requirements_length": len(arch_state.requirements),
        "target_clock_mhz": target_clock_mhz,
        "pdk_summary": pdk_summary,
        "max_rounds": max_rounds,
    }
    if _architecture.error_message:
        result["error_message"] = _architecture.error_message
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Helpers: structured ask_question for architecture & pipeline interrupts
# ---------------------------------------------------------------------------


def _build_arch_ask_question(payload: dict) -> dict:
    """Build structured ask_question + resume_mapping for an architecture interrupt.

    Returns a dict with keys: ask_question, resume_mapping, interrupt_type,
    interrupt_summary, interrupt_questions.  The outer agent uses ask_question
    to present decisions via AskUserQuestion, and resume_mapping to translate
    the user's selection into a resume_architecture() call.
    """
    payload_type = payload.get("type", "")
    phase = payload.get("phase", "")
    out: dict[str, Any] = {
        "interrupt_type": payload_type,
        "interrupt_questions": [],
        "interrupt_summary": "",
        "interrupt_actions": payload.get("supported_actions", []),
    }

    # ---- prd_questions ----
    if payload_type in ("prd_questions", "ers_questions"):
        questions = payload.get("questions", [])
        by_category = payload.get("questions_by_category", {})

        # Classify: single-option (auto-fill) vs multi-option (user choice)
        auto_answerable: list[dict] = []
        needs_choice: list[dict] = []

        for q in questions:
            options = q.get("options", [])
            if len(options) <= 1:
                auto_answerable.append({
                    "id": q.get("id", ""),
                    "question": q.get("question", ""),
                    "suggested_answer": options[0] if options else "",
                    "category": q.get("category", ""),
                })
            else:
                needs_choice.append(q)

        # Build AskQuestion data: up to 4 highest-priority choice questions
        category_priority = [
            "technology", "speed_and_feeds", "dataflow", "area", "power",
        ]
        sorted_qs = sorted(
            needs_choice,
            key=lambda q: (
                category_priority.index(q.get("category", ""))
                if q.get("category", "") in category_priority
                else 99
            ),
        )

        ask_questions = []
        for q in sorted_qs[:4]:
            opts = q.get("options", [])[:4]
            ask_questions.append({
                "id": q.get("id", ""),
                "question": q.get("question", ""),
                "header": q.get("category", "").replace("_", " ").title(),
                "options": [
                    {"label": opt, "description": q.get("context", "")}
                    for opt in opts
                ],
                "multiSelect": False,
            })

        out["ask_question"] = {
            "title": "PRD Sizing Questions",
            "questions": ask_questions,
            "auto_answerable": auto_answerable,
            "remaining_choice_questions": [
                {
                    "id": q.get("id", ""),
                    "question": q.get("question", ""),
                    "options": q.get("options", []),
                    "category": q.get("category", ""),
                }
                for q in sorted_qs[4:]
            ],
        }
        out["resume_mapping"] = {
            "action": "continue",
            "feedback_format": "JSON dict mapping question IDs to answer strings",
            "example": '{"target_technology": "sky130", "bus_protocol": "Dedicated pins"}',
            "note": (
                "Collect answers for ALL questions (auto_answerable + user choices). "
                "Pass as feedback param to resume_architecture(action='continue', feedback='{...}')"
            ),
        }
        out["interrupt_questions"] = questions
        out["interrupt_summary"] = (
            f"PRD sizing: {len(questions)} question(s) across "
            f"{len(by_category)} categories. "
            f"{len(auto_answerable)} auto-answerable, "
            f"{len(needs_choice)} need user choice."
        )

    # ---- architecture_review_needed: block_diagram ----
    elif payload_type == "architecture_review_needed" and phase == "block_diagram":
        bd_summary = payload.get("block_diagram_summary", {})
        diagram_questions = payload.get("questions", [])

        q_text = ""
        if diagram_questions:
            q_text += "The block diagram specialist has questions:\n\n"
            for i, q in enumerate(diagram_questions[:5], 1):
                qt = q.get("question", str(q)) if isinstance(q, dict) else str(q)
                q_text += f"{i}. {qt}\n"
            q_text += "\n"
        q_text += (
            f"Current design: {bd_summary.get('block_count', 0)} blocks, "
            f"~{bd_summary.get('total_estimated_gates', 0)} gates"
        )

        out["interrupt_type"] = "architecture_review_diagram"
        out["ask_question"] = {
            "title": "Block Diagram Review",
            "questions": [
                {
                    "id": "diagram_action",
                    "question": q_text,
                    "header": "Block Diagram",
                    "options": [
                        {"label": "Accept", "description": "Accept diagram and proceed to memory map"},
                        {"label": "Provide Feedback", "description": "Send revision notes to regenerate"},
                        {"label": "Abort", "description": "Stop the architecture run"},
                    ],
                    "multiSelect": False,
                }
            ],
        }
        out["resume_mapping"] = {
            "Accept": {"action": "continue"},
            "Provide Feedback": {"action": "feedback", "feedback": "<revision notes>"},
            "Abort": {"action": "abort"},
        }
        out["interrupt_questions"] = diagram_questions
        out["interrupt_summary"] = (
            f"Block diagram has {len(diagram_questions)} question(s). "
            f"{bd_summary.get('block_count', 0)} blocks, "
            f"~{bd_summary.get('total_estimated_gates', 0)} estimated gates."
        )

    # ---- architecture_review_needed: constraints ----
    elif payload_type == "architecture_review_needed" and phase == "constraints":
        violations = payload.get("violations", [])
        structural = payload.get("structural_violations", [])

        v_text = (
            f"{len(violations)} constraint violation(s) "
            f"({len(structural)} structural):\n\n"
        )
        for v in violations[:5]:
            vt = v.get("violation", str(v)) if isinstance(v, dict) else str(v)
            cat = v.get("category", "") if isinstance(v, dict) else ""
            v_text += f"- [{cat.upper()}] {vt}\n"
        if len(violations) > 5:
            v_text += f"\n...and {len(violations) - 5} more"

        out["interrupt_type"] = "architecture_review_constraints"
        out["ask_question"] = {
            "title": "Constraint Violations",
            "questions": [
                {
                    "id": "constraint_action",
                    "question": v_text,
                    "header": "Constraints",
                    "options": [
                        {"label": "Retry", "description": "Re-run block diagram with violations as feedback"},
                        {"label": "Accept", "description": "Accept architecture despite violations"},
                        {"label": "Provide Feedback", "description": "Send specific revision notes"},
                        {"label": "Abort", "description": "Stop the architecture run"},
                    ],
                    "multiSelect": False,
                }
            ],
        }
        out["resume_mapping"] = {
            "Retry": {"action": "retry"},
            "Accept": {"action": "accept"},
            "Provide Feedback": {"action": "feedback", "feedback": "<revision notes>"},
            "Abort": {"action": "abort"},
        }
        out["interrupt_summary"] = (
            f"Constraint check: {len(violations)} violation(s), "
            f"{len(structural)} structural. "
            f"Round {payload.get('round', '?')}/{payload.get('max_rounds', '?')}."
        )

    # ---- architecture_review_needed: max_rounds_exhausted ----
    elif payload_type == "architecture_review_needed" and phase == "max_rounds_exhausted":
        violations = payload.get("violations", [])

        v_text = (
            f"Max constraint rounds ({payload.get('max_rounds', '?')}) exhausted. "
            f"{len(violations)} violation(s) remain:\n\n"
        )
        for v in violations[:5]:
            vt = v.get("violation", str(v)) if isinstance(v, dict) else str(v)
            v_text += f"- {vt}\n"

        out["interrupt_type"] = "architecture_review_exhausted"
        out["ask_question"] = {
            "title": "Constraint Rounds Exhausted",
            "questions": [
                {
                    "id": "exhausted_action",
                    "question": v_text,
                    "header": "Exhausted",
                    "options": [
                        {"label": "Retry", "description": "Reset round counter and try again"},
                        {"label": "Accept", "description": "Accept despite remaining violations"},
                        {"label": "Provide Feedback", "description": "Reset and retry with guidance"},
                        {"label": "Abort", "description": "Stop the architecture run"},
                    ],
                    "multiSelect": False,
                }
            ],
        }
        out["resume_mapping"] = {
            "Retry": {"action": "retry"},
            "Accept": {"action": "accept"},
            "Provide Feedback": {"action": "feedback", "feedback": "<revision notes>"},
            "Abort": {"action": "abort"},
        }
        out["interrupt_summary"] = (
            f"Constraint iteration exhausted after {payload.get('max_rounds', '?')} rounds. "
            f"{len(violations)} violation(s) remain."
        )

    # ---- final_review ----
    elif payload_type == "final_review":
        out["interrupt_type"] = "final_review"
        out["interrupt_summary"] = (
            "Architecture complete. Architect must approve (OK2DEV) "
            "or request revisions (REVISE) before proceeding to RTL."
        )
        out["ask_question"] = {
            "title": "Architecture Final Review",
            "questions": [
                {
                    "id": "final_review_decision",
                    "question": (
                        f"Architecture for \"{payload.get('title', 'Design')}\" "
                        f"is complete.\n\n"
                        f"{payload.get('block_count', 0)} block(s): "
                        f"{', '.join(payload.get('block_names', []))}\n"
                        f"Estimated gates: {payload.get('total_estimated_gates', 'N/A')}\n"
                        f"Constraint rounds used: {payload.get('constraint_rounds_used', '?')}"
                        f"/{payload.get('max_rounds', '?')}\n\n"
                        f"Review the architecture summary and decide:"
                    ),
                    "header": "Final Review",
                    "options": [
                        {"label": "OK2DEV", "description": "Approve and proceed to RTL generation"},
                        {"label": "REVISE", "description": "Request changes (provide feedback)"},
                        {"label": "ABORT", "description": "Cancel the architecture run"},
                    ],
                    "multiSelect": False,
                }
            ],
        }
        out["resume_mapping"] = {
            "OK2DEV": {"action": "accept"},
            "REVISE": {"action": "feedback", "feedback": "<revision notes>"},
            "ABORT": {"action": "abort"},
        }

    # ---- fallback: unknown interrupt type ----
    else:
        out["interrupt_type"] = payload_type
        out["interrupt_questions"] = payload.get("questions", [])
        out["interrupt_summary"] = (
            f"Architecture paused: {len(payload.get('questions', []))} "
            f"question(s) need architect review "
            f"(phase: {payload.get('phase', 'unknown')})"
        )

    return out


def _build_pipeline_ask_question(payload: dict) -> dict:
    """Build structured ask_question + resume_mapping for a pipeline interrupt.

    Returns a dict with keys: ask_question, resume_mapping, interrupt_summary.
    The outer agent uses ask_question to present decisions via AskUserQuestion,
    and resume_mapping to translate the user's selection into a
    resume_pipeline() call.
    """
    payload_type = payload.get("type", "")
    out: dict[str, Any] = {}

    # ---- uarch_spec_review ----
    if payload_type == "uarch_spec_review":
        block_name = payload.get("block_name", "")
        payload.get("spec_summary", {})
        spec_text_preview = payload.get("spec_text", "")[:500]

        q_text = f"Microarchitecture spec for '{block_name}' is ready for review.\n\n"
        if spec_text_preview:
            q_text += f"{spec_text_preview}\n\n"
        q_text += f"Spec file: {payload.get('spec_path', '')}"

        out["ask_question"] = {
            "title": f"uArch Spec Review: {block_name}",
            "questions": [
                {
                    "id": "uarch_review_decision",
                    "question": q_text,
                    "header": "uArch Review",
                    "options": [
                        {"label": "Approve", "description": "Accept spec and proceed to RTL generation"},
                        {"label": "Revise", "description": "Provide feedback to regenerate the spec"},
                        {"label": "Skip", "description": "Skip this block entirely"},
                    ],
                    "multiSelect": False,
                }
            ],
        }
        out["resume_mapping"] = {
            "Approve": {"action": "approve"},
            "Revise": {"action": "revise", "feedback": "<revision notes>"},
            "Skip": {"action": "skip"},
        }
        out["interrupt_summary"] = (
            f"uArch spec for '{block_name}' needs review before RTL generation."
        )

    # ---- human_intervention_needed ----
    elif payload_type == "human_intervention_needed":
        block_name = payload.get("block_name", payload.get("block", ""))
        diagnosis = payload.get("diagnosis", "")
        category = payload.get("category", "UNKNOWN")
        confidence = payload.get("confidence", 0.5)
        suggested_fix = payload.get("suggested_fix", "")
        attempt = payload.get("attempt", 0)
        max_attempts = payload.get("max_attempts", 5)

        q_text = (
            f"Block '{block_name}' failed (attempt {attempt}/{max_attempts}).\n\n"
            f"Category: {category}\n"
            f"Confidence: {confidence:.0%}\n"
            f"Diagnosis: {diagnosis[:300]}\n"
        )
        if suggested_fix:
            q_text += f"\nSuggested fix: {suggested_fix[:200]}"
        if payload.get("human_question"):
            q_text += f"\n\nDebug agent asks: {payload['human_question']}"

        out["ask_question"] = {
            "title": f"Pipeline Failure: {block_name}",
            "questions": [
                {
                    "id": "failure_action",
                    "question": q_text,
                    "header": "Failure",
                    "options": [
                        {"label": "Retry", "description": "Re-generate RTL and retry"},
                        {"label": "Add Constraint", "description": "Inject a design constraint to guide generation"},
                        {"label": "Skip", "description": "Skip this block and continue"},
                        {"label": "Abort", "description": "Stop the entire pipeline"},
                    ],
                    "multiSelect": False,
                }
            ],
        }
        out["resume_mapping"] = {
            "Retry": {"action": "retry"},
            "Add Constraint": {"action": "add_constraint", "constraint": "<constraint text>"},
            "Skip": {"action": "skip"},
            "Abort": {"action": "abort"},
        }
        out["interrupt_summary"] = (
            f"Block '{block_name}' failed: {category} "
            f"(attempt {attempt}/{max_attempts}, confidence {confidence:.0%})."
        )

    # ---- uarch_integration_review ----
    elif payload_type == "uarch_integration_review":
        block_names = payload.get("block_names", [])
        issues_found = payload.get("issues_found", 0)
        issues_fixed = payload.get("issues_fixed", 0)
        review_summary = payload.get("review_summary", "")[:600]

        q_text = (
            f"Chip-level uArch integration review (tier {payload.get('tier', '?')}).\n\n"
            f"Blocks reviewed: {', '.join(block_names)}\n"
            f"Issues found: {issues_found}, Issues fixed: {issues_fixed}\n\n"
        )
        if review_summary:
            q_text += f"Summary:\n{review_summary}\n"

        out["ask_question"] = {
            "title": "Chip-Level uArch Integration Review",
            "questions": [
                {
                    "id": "integration_review_decision",
                    "question": q_text,
                    "header": "Integration Review",
                    "options": [
                        {"label": "Approve", "description": "Accept all specs and proceed to RTL generation"},
                        {"label": "Revise", "description": "Restart affected blocks to regenerate specs"},
                        {"label": "Abort", "description": "Stop the pipeline"},
                    ],
                    "multiSelect": False,
                }
            ],
        }
        out["resume_mapping"] = {
            "Approve": {"action": "approve"},
            "Revise": {"action": "revise"},
            "Abort": {"action": "abort"},
        }
        out["interrupt_summary"] = (
            f"Chip-level uArch review: {issues_found} issue(s) found, "
            f"{issues_fixed} fixed. Blocks: {', '.join(block_names)}."
        )

    # ---- uarch_feasibility ----
    elif payload_type == "uarch_feasibility":
        block_name = payload.get("block_name", payload.get("block", ""))
        issues = payload.get("blocking_issues", []) or []
        guidance = payload.get("outer_agent_guidance", "")
        issues_txt = "\n".join(f"  - {i}" for i in issues)

        q_text = (
            f"Block '{block_name}' is INFEASIBLE as budgeted.\n\n"
            f"The microarchitecture step reports it cannot be built byte-exactly "
            f"against the golden. Feasibility spans four dimensions and each "
            f"blocker below LEADS WITH ITS CATEGORY: [interface] a port can't "
            f"carry data the golden reads, [area] it won't fit the area budget, "
            f"[timing] it can't hit the clock/throughput budget, or [capability] "
            f"it's fundamentally not realizable from this golden slice (e.g. needs "
            f"runtime-built tables like 80 Huffman trees). [area] and [timing] "
            f"have later dedicated gates; [capability] has NO backstop, so take it "
            f"seriously here. This is the engine's OWN diagnosis; do NOT have it "
            f"emit a stub.\n\n"
            f"Blocking issues:\n{issues_txt}\n\n"
            f"uArch spec: {payload.get('uarch_spec_path', '')}"
        )
        if guidance:
            q_text += f"\n\n{guidance}"

        out["ask_question"] = {
            "title": f"uArch Feasibility Block: {block_name}",
            "questions": [
                {
                    "id": "uarch_feasibility_decision",
                    "question": q_text,
                    "header": "Feasibility",
                    "options": [
                        {"label": "Fix & re-spec", "description": "Fix the root cause per the blocker's category -- widen the interface, repartition/shrink storage (area), pipeline it (timing), or decompose the block (capability) -- then re-spec against the corrected design"},
                        {"label": "Override", "description": "Proceed to RTL anyway -- ONLY for a verified false alarm; never override a [capability] blocker"},
                        {"label": "Abort", "description": "End this block without emitting RTL"},
                    ],
                    "multiSelect": False,
                }
            ],
        }
        out["resume_mapping"] = {
            "Fix & re-spec": {"action": "revise_interface"},
            "Override": {"action": "override"},
            "Abort": {"action": "abort"},
        }
        out["interrupt_summary"] = (
            f"Block '{block_name}' reports {len(issues)} feasibility blocker(s) "
            f"(interface/area/timing/capability); fix the root cause before RTL."
        )

    # ---- integration_failure ----
    elif payload_type == "integration_failure":
        mismatches = payload.get("mismatches", [])
        lint_clean = payload.get("lint_clean", False)
        error_count = len([m for m in mismatches if isinstance(m, dict) and m.get("severity") == "error"])
        warning_count = len([m for m in mismatches if isinstance(m, dict) and m.get("severity") == "warning"])

        q_text = (
            f"Integration check found issues:\n"
            f"- {error_count} error(s), {warning_count} warning(s)\n"
            f"- Lint clean: {'Yes' if lint_clean else 'No'}\n"
        )
        if mismatches:
            q_text += "\nMismatches:\n"
            for m in mismatches[:5]:
                if isinstance(m, dict):
                    q_text += (
                        f"  - [{m.get('issue_type', '?')}] "
                        f"{m.get('from_block', '?')} -> {m.get('to_block', '?')}: "
                        f"{str(m.get('description', ''))[:100]}\n"
                    )

        out["ask_question"] = {
            "title": "Integration Check Failed",
            "questions": [
                {
                    "id": "integration_action",
                    "question": q_text,
                    "header": "Integration",
                    "options": [
                        {"label": "Fix and Retry", "description": "Edit RTL then re-run integration check"},
                        {"label": "Skip", "description": "Proceed without integration verification"},
                        {"label": "Abort", "description": "Stop the pipeline"},
                    ],
                    "multiSelect": False,
                }
            ],
        }
        out["resume_mapping"] = {
            "Fix and Retry": {"action": "fix_rtl", "rtl_fix_description": "<describe fix>"},
            "Skip": {"action": "skip"},
            "Abort": {"action": "abort"},
        }
        out["interrupt_summary"] = (
            f"Integration check: {error_count} error(s), {warning_count} warning(s). "
            f"Lint clean: {'Yes' if lint_clean else 'No'}."
        )

    # ---- integration_dv_failure ----
    elif payload_type == "integration_dv_failure":
        sim_log = payload.get("sim_log", "")[-500:]
        supported = payload.get("supported_actions", [])

        q_text = (
            f"Integration DV simulation failed.\n\n"
            f"Sim log (last 500 chars):\n{sim_log}"
        )

        out["ask_question"] = {
            "title": "Integration DV Failed",
            "questions": [
                {
                    "id": "integration_dv_action",
                    "question": q_text,
                    "header": "Integration DV",
                    "options": [
                        {"label": "Retry", "description": "Regenerate testbench and re-simulate"},
                        {"label": "Fix RTL", "description": "RTL was edited; re-run sim only"},
                        *(
                            [{"label": "Skip", "description": "Proceed without integration DV"}]
                            if "skip" in supported else []
                        ),
                        {"label": "Abort", "description": "Stop the pipeline"},
                    ],
                    "multiSelect": False,
                }
            ],
        }
        out["resume_mapping"] = {
            "Retry": {"action": "retry"},
            "Fix RTL": {"action": "fix_rtl", "rtl_fix_description": "<describe fix>"},
            "Abort": {"action": "abort"},
        }
        if "skip" in supported:
            out["resume_mapping"]["Skip"] = {"action": "skip"}
        out["interrupt_summary"] = "Integration DV simulation failed."

    # ---- validation_dv_failure ----
    elif payload_type == "validation_dv_failure":
        sim_log = payload.get("sim_log", "")[-500:]
        supported = payload.get("supported_actions", [])

        q_text = (
            f"Validation DV failed while checking ERS/KPI requirements.\n\n"
            f"Sim log (last 500 chars):\n{sim_log}"
        )

        out["ask_question"] = {
            "title": "Validation DV Failed",
            "questions": [
                {
                    "id": "validation_dv_action",
                    "question": q_text,
                    "header": "Validation DV",
                    "options": [
                        {"label": "Retry", "description": "Regenerate validation testbench and re-simulate"},
                        {"label": "Fix RTL", "description": "RTL was edited; re-run validation sim"},
                        {"label": "Fix TB", "description": "Validation testbench was edited; re-run validation sim"},
                        *(
                            [{"label": "Skip", "description": "Proceed without validation DV"}]
                            if "skip" in supported else []
                        ),
                        {"label": "Abort", "description": "Stop the pipeline"},
                    ],
                    "multiSelect": False,
                }
            ],
        }
        out["resume_mapping"] = {
            "Retry": {"action": "retry"},
            "Fix RTL": {"action": "fix_rtl", "rtl_fix_description": "<describe fix>"},
            "Fix TB": {"action": "fix_tb", "rtl_fix_description": "<describe fix>"},
            "Abort": {"action": "abort"},
        }
        if "skip" in supported:
            out["resume_mapping"]["Skip"] = {"action": "skip"}
        out["interrupt_summary"] = "Validation DV failed."

    return out


# ---------------------------------------------------------------------------
# Tool: get_architecture_state
# ---------------------------------------------------------------------------


@server.tool()
async def get_architecture_state() -> str:
    """Get the current architecture graph state from the checkpoint.

    Returns status, current phase, round, completed blocks,
    interrupt payload (if interrupted), and next nodes.
    """
    await _architecture.ensure_graph()

    if not _architecture.thread_id:
        return json.dumps({"status": "idle", "message": "No architecture run started"})

    config = {"configurable": {"thread_id": _architecture.thread_id}}

    try:
        state_snapshot = await _architecture.graph.aget_state(config)
    except Exception as e:
        return json.dumps({
            "status": _architecture.status,
            "error": f"Failed to read state: {e}",
        })

    if not state_snapshot or not state_snapshot.values:
        return json.dumps({
            "status": _architecture.status,
            "thread_id": _architecture.thread_id,
        })

    values = state_snapshot.values

    # Extract interrupt payload if present
    interrupt_payload = None
    if state_snapshot.tasks:
        for task in state_snapshot.tasks:
            if task.interrupts:
                interrupt_payload = task.interrupts[0].value
                break

    # Self-heal: if checkpoint has pending interrupts but in-memory status
    # hasn't caught up yet, correct it so the outer agent can resume.
    # Only self-heal when the asyncio task has finished -- if it's still
    # running, the checkpoint data may be stale (race after resume).
    if interrupt_payload and _architecture.status == "running":
        if _architecture.task is None or _architecture.task.done():
            _architecture.status = "interrupted"

    # Build block diagram summary
    bd = values.get("block_diagram") or {}
    blocks = bd.get("blocks", [])
    block_names = [b.get("name", "") for b in blocks]

    # Current violations
    cr = values.get("constraint_result") or {}
    violations = cr.get("violations", [])

    # PRD status
    prd_spec = values.get("prd_spec")
    prd_questions = values.get("prd_questions") or []

    result = {
        "status": _architecture.status,
        "thread_id": _architecture.thread_id,
        "phase": values.get("phase", ""),
        "round": values.get("round", 0),
        "max_rounds": values.get("max_rounds", 3),
        "next_nodes": list(state_snapshot.next) if state_snapshot.next else [],
        "prd_complete": prd_spec is not None,
        "prd_question_count": len(prd_questions),
        "block_count": len(blocks),
        "block_names": block_names,
        "constraint_result": cr,
        "violation_count": len(violations),
        "violations": [
            v["violation"] if isinstance(v, dict) else v
            for v in violations[:5]
        ],
        "questions": [
            q.get("question", "") if isinstance(q, dict) else q
            for q in (bd.get("questions") or [])
        ],
        "success": values.get("success", False),
        "block_specs_path": values.get("block_specs_path", ""),
        "interrupt_payload": interrupt_payload,
        "checkpoint_id": (
            state_snapshot.config.get("configurable", {}).get("checkpoint_id")
            if state_snapshot.config
            else None
        ),
        # Human-in-the-loop signals for the outer agent
        "human_input_needed": _architecture.status == "interrupted",
    }

    # When interrupted, surface structured question data so the outer agent
    # can present them via AskQuestion without parsing the raw payload.
    if interrupt_payload:
        arch_q = _build_arch_ask_question(interrupt_payload)
        result.update(arch_q)

    if _architecture.error_message:
        result["error_message"] = _architecture.error_message

    diag = _get_diagnostics(graph_filter="architecture")
    if diag and diag.get("failure_count", 0) > 0:
        result["diagnostics"] = diag

    return json.dumps(result, indent=2, default=str)


# ---------------------------------------------------------------------------
# Tool: resume_architecture
# ---------------------------------------------------------------------------


@server.tool()
async def resume_architecture(
    action: str,
    feedback: str = "",
) -> str:
    """Resume the architecture graph after an interrupt or pause.

    The graph must be in 'interrupted' or 'paused' status. The action
    tells the graph what to do next.

    Args:
        action: One of 'continue', 'retry', 'accept', 'feedback', 'abort'.
            - 'continue': Proceed (used with ERS answers as JSON in feedback).
            - 'accept': Approve current state (OK2DEV at final review).
            - 'feedback': Provide revision notes (REVISE) in feedback param.
            - 'retry': Re-run the current phase.
            - 'abort': Stop the architecture run.
        feedback: Architect's feedback text (used when action is 'feedback').
            For ERS interrupts, pass JSON-encoded answers dict as feedback,
            e.g. '{"target_technology": "sky130", "input_data_rate": "270 Mbps"}'.
    """
    valid_actions = {"continue", "retry", "accept", "feedback", "abort"}
    if action not in valid_actions:
        return json.dumps({
            "error": f"Invalid action: {action}. Must be one of: {sorted(valid_actions)}",
        })

    if _architecture.status not in ("interrupted", "paused"):
        # Self-heal: check checkpoint for pending interrupts that the
        # in-memory status hasn't caught up with yet (race condition).
        # Only self-heal when the asyncio task has finished -- if it's
        # still running, the checkpoint data may be stale.
        if _architecture.status == "running" and _architecture.thread_id:
            if _architecture.task is None or _architecture.task.done():
                try:
                    await _architecture.ensure_graph()
                    _config = {"configurable": {"thread_id": _architecture.thread_id}}
                    _snap = await _architecture.graph.aget_state(_config)
                    if _snap and _snap.tasks:
                        for _t in _snap.tasks:
                            if _t.interrupts:
                                _architecture.status = "interrupted"
                                break
                except Exception:
                    pass

        if _architecture.status not in ("interrupted", "paused"):
            return json.dumps({
                "error": f"Cannot resume: architecture status is '{_architecture.status}'",
                "hint": "Architecture must be 'interrupted' or 'paused' to resume.",
            })

    if action == "feedback" and not feedback:
        return json.dumps({
            "error": "feedback parameter is required when action is 'feedback'",
        })

    await _architecture.ensure_graph()

    from langgraph.types import Command

    resume_value: dict[str, Any] = {
        "action": action,
        "feedback": feedback,
    }

    # For ERS interrupts, try to parse feedback as JSON answers dict.
    # Accept answers from both "continue" and "feedback" actions so
    # the outer agent doesn't silently lose answers by using the
    # wrong action string.
    if feedback and action in ("continue", "feedback"):
        try:
            answers = json.loads(feedback)
            if isinstance(answers, dict):
                resume_value["answers"] = answers
                # Normalise to "continue" so downstream routing works
                resume_value["action"] = "continue"
        except (json.JSONDecodeError, TypeError):
            pass

    # Log what the outer agent sent so trajectories are debuggable
    from orchestrator.langgraph.event_stream import write_graph_event
    write_graph_event(_project_root(), "resume_architecture", "escalation_response", {
        "graph": "architecture",
        "action": resume_value["action"],
        "has_answers": "answers" in resume_value,
        "answer_keys": list(resume_value["answers"].keys()) if "answers" in resume_value else [],
        "feedback_length": len(feedback),
        "feedback": feedback[:2000] if feedback else "",
    })

    config = {"configurable": {"thread_id": _architecture.thread_id}}

    if _architecture.status == "paused":
        # Check if checkpoint has a pending interrupt that needs a Command.
        _has_pending_interrupt = False
        try:
            _snap = await _architecture.graph.aget_state(config)
            if _snap and _snap.tasks:
                for _t in _snap.tasks:
                    if _t.interrupts:
                        _has_pending_interrupt = True
                        break
        except Exception:
            pass

        if _has_pending_interrupt:
            resume_input = Command(resume=resume_value)
        else:
            resume_input = None
        _architecture.task = asyncio.create_task(
            _architecture.run_task(resume_input, config)
        )
    else:
        _architecture.task = asyncio.create_task(
            _architecture.run_task(Command(resume=resume_value), config)
        )

    result = {
        "status": "running",
        "action": action,
        "thread_id": _architecture.thread_id,
    }
    if _architecture.error_message:
        result["error_message"] = _architecture.error_message
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Tool: pause_architecture
# ---------------------------------------------------------------------------


@server.tool()
async def pause_architecture() -> str:
    """Pause the running architecture graph.

    Cancels the background task and kills any stuck Claude CLI
    subprocesses. The graph state is preserved at the last completed
    node boundary (via checkpoint). Call resume_architecture() to
    continue from that point.
    """
    if _architecture.status != "running":
        return json.dumps({
            "error": f"Cannot pause: architecture status is '{_architecture.status}'",
        })

    # Kill any stuck CLI subprocesses first so the task can actually cancel
    from orchestrator.langchain.agents.coresmith_llm import kill_active_cli_processes
    kill_active_cli_processes()

    if _architecture.task and not _architecture.task.done():
        _architecture.task.cancel()
        try:
            await _architecture.task
        except asyncio.CancelledError:
            pass

    _architecture.status = "paused"

    # Read current state for the response
    await _architecture.ensure_graph()
    config = {"configurable": {"thread_id": _architecture.thread_id}}
    try:
        state_snapshot = await _architecture.graph.aget_state(config)
        values = state_snapshot.values if state_snapshot else {}
        phase = values.get("phase", "")
        round_num = values.get("round", 0)
    except Exception:
        phase = ""
        round_num = 0

    result = {
        "status": "paused",
        "phase": phase,
        "round": round_num,
        "thread_id": _architecture.thread_id,
        "message": "Architecture paused. Call resume_architecture() to continue.",
    }
    if _architecture.error_message:
        result["error_message"] = _architecture.error_message
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Tool: restart_architecture_node
# ---------------------------------------------------------------------------


@server.tool()
async def restart_architecture_node(node_name: str) -> str:
    """Re-execute from a specific node by forking from its checkpoint.

    Walks the checkpoint history to find the state just before the
    target node, then re-invokes the graph from that point.

    The architecture graph must be 'interrupted' or 'paused'.

    Args:
        node_name: The node to restart from (e.g., 'Block Diagram',
            'Memory Map', 'Constraint Check', 'Escalate Constraints').
    """
    if _architecture.status not in ("interrupted", "paused"):
        return json.dumps({
            "error": f"Cannot restart: architecture status is '{_architecture.status}'",
            "hint": "Pause or wait for interrupt first.",
        })

    await _architecture.ensure_graph()
    config = {"configurable": {"thread_id": _architecture.thread_id}}

    # Walk history to find the checkpoint where node_name is next
    found_config = None
    async for state_snapshot in _architecture.graph.aget_state_history(config):
        if state_snapshot.next and node_name in state_snapshot.next:
            found_config = state_snapshot.config
            break

    if not found_config:
        return json.dumps({
            "error": f"No checkpoint found with '{node_name}' as next node",
            "hint": "Available nodes: Block Diagram, Memory Map, Clock Tree, "
                    "Register Spec, Constraint Check, Finalize Architecture, "
                    "Escalate Diagram, Escalate Constraints, Escalate Exhausted, "
                    "Architecture Complete, Constraint Iteration, Abort",
        })

    fork_config = {
        "configurable": {
            "thread_id": _architecture.thread_id,
            "checkpoint_id": found_config["configurable"]["checkpoint_id"],
        },
    }

    _architecture.task = asyncio.create_task(
        _architecture.run_task(None, fork_config)
    )

    return json.dumps({
        "status": "running",
        "restarting_from": node_name,
        "thread_id": _architecture.thread_id,
        "checkpoint_id": found_config["configurable"]["checkpoint_id"],
    })


# ═══════════════════════════════════════════════════════════════════════════
# ARCHITECTURE BENCHMARK TOOLS -- optional pre-work before starting graph
# ═══════════════════════════════════════════════════════════════════════════


# ---------------------------------------------------------------------------
# Tool: run_benchmark
# ---------------------------------------------------------------------------


@server.tool()
async def run_benchmark(
    component: str = "multiplier",
    width: int = 16,
    depth: int = 64,
    radix: int = 2,
    target_clock_mhz: float = 50.0,
) -> str:
    """Synthesize a micro-benchmark to get real gate count and timing.

    Available components: multiplier, fifo, sram_array, fft_butterfly, counter.

    Args:
        component: Benchmark type.
        width: Data width in bits (all components).
        depth: Memory depth (fifo, sram_array only).
        radix: FFT radix (fft_butterfly only).
        target_clock_mhz: Target clock frequency.
    """
    from orchestrator.architecture.benchmarks.runner import (
        run_benchmark as _run,
    )
    from orchestrator.architecture.state import load_state, save_state
    from orchestrator.pdk import PDKConfig

    state = load_state(_project_root())

    if not state.pdk_config:
        return "Error: No PDK configured. Call start_architecture with pdk_config_path first."

    pdk = PDKConfig.from_dict(state.pdk_config)

    # Build params based on component type
    params: dict[str, Any] = {"width": width}
    if component in ("fifo", "sram_array"):
        params["depth"] = depth
    if component == "fft_butterfly":
        params["radix"] = radix

    result = await _run(
        component=component,
        params=params,
        pdk_config=pdk,
        target_clock_mhz=target_clock_mhz,
        project_root=_project_root(),
    )

    # Save benchmark result to state
    key = f"{component}_{params}"
    state.benchmark_results[key] = result
    save_state(state, _project_root())

    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Tool: characterize_pdk
# ---------------------------------------------------------------------------


@server.tool()
async def characterize_pdk(
    target_clock_mhz: float = 50.0,
) -> str:
    """Run a standard benchmark suite against the configured PDK.

    Synthesizes ~10 micro-benchmarks (multipliers, FIFOs, counters, etc.)
    to empirically measure what the PDK can do. Results are cached.

    Args:
        target_clock_mhz: Target clock frequency for the benchmarks.
    """
    from orchestrator.architecture.benchmarks.runner import (
        characterize_pdk as _characterize,
    )
    from orchestrator.architecture.state import load_state, save_state
    from orchestrator.pdk import PDKConfig

    state = load_state(_project_root())

    if not state.pdk_config:
        return "Error: No PDK configured. Call start_architecture with pdk_config_path first."

    pdk = PDKConfig.from_dict(state.pdk_config)

    results = await _characterize(
        pdk_config=pdk,
        target_clock_mhz=target_clock_mhz,
        project_root=_project_root(),
    )

    # Save all results to state
    state.benchmark_results.update(results)
    save_state(state, _project_root())

    # Summary with honest timing reporting
    lines = [f"PDK Characterization: {pdk.to_summary()}", ""]
    for key, result in results.items():
        gc = result.get("gate_count", 0)
        mc = result.get("max_clock_mhz")
        err = result.get("error", "")
        skipped = result.get("sta_skipped", "")
        if err:
            lines.append(f"  {key}: ERROR - {err}")
        elif mc is not None:
            lines.append(f"  {key}: {gc:,} gates, max {mc} MHz")
        elif skipped:
            lines.append(f"  {key}: {gc:,} gates, timing: N/A ({skipped})")
        else:
            lines.append(f"  {key}: {gc:,} gates, timing: N/A")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE OBSERVABILITY TOOLS
# ═══════════════════════════════════════════════════════════════════════════


@server.tool()
async def get_pipeline_status(last_n: int = 25) -> str:
    """Get a formatted summary of recent pipeline events.

    Shows graph transitions (node enter/exit), LLM calls (model, prompt
    size, elapsed time, output size), and block progress. Reads from
    the JSONL event log at .coresmith/pipeline_events.jsonl.

    Also includes:
    - Escalation response events (what the outer agent sent back)
    - Warnings when escalation answers are missing
    - Trajectory file pointers for debugging LLM prompts/completions

    Args:
        last_n: Number of recent events to show (default 25).
    """
    from orchestrator.langgraph.event_stream import format_event_summary, read_events

    root = _project_root()
    events = read_events(root)
    total = len(events)
    recent = events[-last_n:] if last_n < total else events

    # Count blocks and phases for a quick summary header
    blocks_seen = set()
    completed_blocks = set()
    current_block = None
    escalation_events = []
    warning_events = []
    for ev in events:
        block = ev.get("block", "")
        if ev.get("event") == "block_start" and block:
            blocks_seen.add(block)
            current_block = block
        if ev.get("event") == "graph_node_exit" and ev.get("node") == "Synthesize":
            completed_blocks.add(ev.get("block", ""))
        if ev.get("event") == "escalation_response":
            escalation_events.append(ev)
        if ev.get("event") == "escalation_warning":
            warning_events.append(ev)

    header = (
        f"Pipeline: {len(completed_blocks)}/{len(blocks_seen)} blocks complete"
        f" | {total} events total"
    )
    if current_block:
        header += f" | current: {current_block}"

    body = format_event_summary(recent)

    # Trajectory file pointers for debugging
    trajectory_section = _trajectory_debug_info(root)

    # Escalation summary
    escalation_section = ""
    if escalation_events or warning_events:
        lines = ["\n--- ESCALATION HISTORY ---"]
        for ev in escalation_events:
            ts = ev.get("iso", "")
            action = ev.get("action", "?")
            has_ans = ev.get("has_answers", False)
            keys = ev.get("answer_keys", [])
            lines.append(
                f"  [{ts}] Outer agent resumed: action={action}, "
                f"has_answers={has_ans}, keys={keys}"
            )
        for ev in warning_events:
            ts = ev.get("iso", "")
            warning = ev.get("warning", "")
            hint = ev.get("hint", "")
            lines.append(f"  [{ts}] WARNING: {warning}")
            if hint:
                lines.append(f"           HINT: {hint}")
        escalation_section = "\n".join(lines)

    # Diagnostic guidance when pipeline is interrupted
    diagnostic_section = ""
    if _pipeline.status == "interrupted":
        diagnostic_section = _diagnostic_guidance(root)

    return f"{header}\n\n{body}{escalation_section}{diagnostic_section}{trajectory_section}"


def _diagnostic_guidance(project_root: str) -> str:
    """Build diagnostic guidance for the outer agent when pipeline is interrupted."""
    lines = ["\n\n--- DIAGNOSTIC GUIDANCE (for outer agent) ---"]
    lines.append("You are the OUTER-LOOP DIAGNOSTIC AGENT. Do not blindly")
    lines.append("retry, auto-accept, or relay failures to the user.")
    lines.append("Read the evidence, classify the root cause, then choose")
    lines.append("one explicit supported action with rationale:")
    lines.append("")
    lines.append("1. Call get_pipeline_state() to read the interrupt_payload")
    lines.append("2. Check OTEL events, step logs, RTL, TB, VCD/WaveKit audit,")
    lines.append("   uarch spec, and ERS contract before deciding")
    lines.append("3. Check 'category' field:")
    lines.append("   - INTERFACE_MISMATCH / UARCH_SPEC_ERROR: read RTL + spec,")
    lines.append("     inject constraint via resume_pipeline(action='add_constraint',")
    lines.append("     constraint='MUST ...')")
    lines.append("4. Check 'category_counts' for repeated failures:")
    lines.append("   - Same category 2+ times: inner loop is stuck. Read the RTL")
    lines.append("     at 'rtl_path', diagnose it yourself, edit on disk, then")
    lines.append("     resume_pipeline(action='fix_rtl', rtl_fix_description='...')")
    lines.append("5. Only use resume_pipeline(action='retry') after explaining why")
    lines.append("   retrying is expected to change the outcome")
    lines.append("6. Check 'needs_human' and 'confidence':")
    lines.append("   - needs_human=true or confidence < 0.5: escalate to user")
    lines.append("     with your own diagnosis appended")
    lines.append("")
    lines.append("Files to read for diagnosis:")

    from pathlib import Path
    root = Path(project_root)
    uarch_dir = root / ARCH_DOC_DIR / "uarch_specs"
    if uarch_dir.exists():
        specs = list(uarch_dir.glob("*.md"))
        for s in specs:
            lines.append(f"  uArch spec: {s}")

    rtl_dir = root / "rtl"
    if rtl_dir.exists():
        rtl_files = list(rtl_dir.rglob("*.v")) + list(rtl_dir.rglob("*.sv"))
        for r in rtl_files[:5]:
            lines.append(f"  RTL source: {r}")

    # Integration-specific guidance
    integration_rtl = root / "rtl" / "integration"
    if integration_rtl.exists():
        for f in integration_rtl.glob("*.v"):
            lines.append(f"  Integration RTL: {f}")

    lines.append("")
    lines.append("For INTEGRATION_FAILURE interrupts:")
    lines.append("  - Read the 'mismatches' list in the interrupt payload")
    lines.append("  - WIDTH_MISMATCH: edit the block RTL to fix port widths,")
    lines.append("    then run_step('lint', block_name) to verify")
    lines.append("  - MISSING_PORT: add the port to the block RTL")
    lines.append("  - LINT_ERRORS: read the lint log and fix the top-level RTL")
    lines.append("  - After fixing: resume_pipeline(action='fix_rtl',")
    lines.append("    rtl_fix_description='...')")
    lines.append("")
    lines.append("REMEMBER: Only escalate to the user if you CANNOT diagnose")
    lines.append("the issue yourself. You have full file read/write access.")
    return "\n".join(lines)


def _trajectory_debug_info(project_root: str) -> str:
    """Build trajectory file pointers for outer-agent debugging."""
    from pathlib import Path
    root = Path(project_root)
    lines = ["\n\n--- TRAJECTORY DEBUG FILES ---"]
    lines.append("Use these files to inspect full LLM prompts and completions:")

    llm_log = root / ".coresmith" / "llm_calls.jsonl"
    if llm_log.exists():
        with open(llm_log) as f:
            count = sum(1 for _ in f)
        size_kb = llm_log.stat().st_size / 1024
        lines.append(f"  LLM calls:       {llm_log}  ({count} calls, {size_kb:.0f} KB)")
        lines.append(f"    -> Read last call: tail -1 {llm_log} | python -m json.tool")
    else:
        lines.append(f"  LLM calls:       {llm_log}  (not yet created)")

    events_log = root / ".coresmith" / "pipeline_events.jsonl"
    if events_log.exists():
        with open(events_log) as f:
            count = sum(1 for _ in f)
        size_kb = events_log.stat().st_size / 1024
        lines.append(f"  Pipeline events: {events_log}  ({count} events, {size_kb:.0f} KB)")
    else:
        lines.append(f"  Pipeline events: {events_log}  (not yet created)")

    traces_db = root / ".coresmith" / "traces.db"
    if traces_db.exists():
        size_kb = traces_db.stat().st_size / 1024
        lines.append(f"  OTel traces DB:  {traces_db}  ({size_kb:.0f} KB)")
        lines.append(f"    -> Query: sqlite3 {traces_db} \"SELECT name, json_extract(attributes, '$.has_answers') FROM spans ORDER BY start_ns DESC LIMIT 10\"")

    lines.append("")
    lines.append("To debug escalation flow, look for 'escalation_response' and")
    lines.append("'escalation_warning' events in pipeline_events.jsonl.")
    lines.append("If Gather Requirements keeps regenerating questions, check that")
    lines.append("resume_architecture was called with action='continue' (not 'feedback')")
    lines.append("and that the feedback parameter contains valid JSON answers dict.")

    # Step log files (full untruncated EDA tool output)
    from orchestrator.langgraph.pipeline_helpers import _LOG_DIR
    log_dir = _LOG_DIR
    if log_dir.exists():
        lines.append("")
        lines.append("--- STEP LOGS (full stdout/stderr) ---")
        for block_dir in sorted(log_dir.iterdir()):
            if block_dir.is_dir():
                for log_file in sorted(block_dir.iterdir()):
                    try:
                        size = log_file.stat().st_size
                        lines.append(f"  {log_file}  ({size:,} bytes)")
                    except OSError:
                        pass
        lines.append("Use Read tool to inspect any log file for full untruncated output.")

    return "\n".join(lines)


@server.tool()
async def get_pipeline_events(
    after_timestamp: float = 0,
    event_type: str = "",
    block_name: str = "",
    limit: int = 50,
) -> str:
    """Query pipeline events with optional filters.

    Returns raw JSON events from .coresmith/pipeline_events.jsonl, useful
    for programmatic analysis or debugging specific blocks/phases.

    Args:
        after_timestamp: Only return events after this Unix timestamp.
        event_type: Filter by event type (e.g. "llm_end", "block_start",
            "graph_node_enter"). Empty string returns all types.
        block_name: Filter by block name (e.g. "scrambler"). Empty string
            returns all blocks.
        limit: Maximum number of events to return (default 50).
    """
    from orchestrator.langgraph.event_stream import read_events

    events = read_events(_project_root(), after_ts=after_timestamp)

    if event_type:
        events = [e for e in events if e.get("event") == event_type]
    if block_name:
        events = [e for e in events if e.get("block") == block_name]

    events = events[-limit:]
    return json.dumps(events, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE CONTROL TOOLS -- start, inspect, pause, resume, restart
#
# Claude Code (or any MCP client) drives the pipeline through these tools.
# The graph runs autonomously and only pauses at ask_human interrupts or
# when pause_pipeline() is called.
# ═══════════════════════════════════════════════════════════════════════════


@server.tool()
async def start_pipeline(
    max_attempts: int = 5,
    target_clock_mhz: float = 50.0,
    blocks_file: str = "",
) -> str:
    """Start the RTL pipeline graph as a background task.

    Loads config, builds the tier-sorted block queue, and launches the
    LangGraph pipeline.  The graph runs autonomously -- it only pauses
    when the debug agent flags a failure for human review (interrupt).

    Call get_pipeline_state() to monitor progress.

    Args:
        max_attempts: Maximum retry attempts per block (default 5).
        target_clock_mhz: Target clock frequency in MHz (default 50.0).
        blocks_file: Path to external YAML block registry (optional).
            If provided, blocks are loaded from this file instead of
            config.yaml or block_specs.json.
    """
    if _pipeline.status == "running":
        return json.dumps({
            "error": "Pipeline already running",
            "thread_id": _pipeline.thread_id,
            "status": _pipeline.status,
        })

    # Auto-pause architecture if it's still running (mirrors pause_architecture)
    if _architecture.status == "running":
        from orchestrator.langchain.agents.coresmith_llm import kill_active_cli_processes
        if _architecture.task and not _architecture.task.done():
            _architecture.task.cancel()
            try:
                await _architecture.task
            except (asyncio.CancelledError, Exception):
                pass
        kill_active_cli_processes()
        _architecture.status = "paused"

    await _pipeline.ensure_graph()

    block_queue: list[dict] = []

    # Priority 1: explicit blocks_file parameter
    if blocks_file:
        import yaml
        bf_path = Path(_project_root()) / blocks_file
        if bf_path.exists():
            with open(bf_path) as f:
                bf_data = yaml.safe_load(f) or {}
            from orchestrator.langgraph.pipeline_helpers import get_sorted_block_queue
            block_queue = get_sorted_block_queue(bf_data)

    # Priority 2: architecture-generated block specs
    if not block_queue:
        specs_path = Path(_project_root()) / ".coresmith" / "block_specs.json"
        if specs_path.exists():
            block_queue = json.loads(specs_path.read_text())

    # Priority 3: config.yaml blocks section
    if not block_queue:
        from orchestrator.langgraph.pipeline_helpers import (
            get_sorted_block_queue,
            load_config,
        )
        config = load_config()
        block_queue = get_sorted_block_queue(config)

    if not block_queue:
        return json.dumps({
            "error": "No blocks found. Run start_architecture() first, "
                     "add a blocks: section to config.yaml, or pass "
                     "blocks_file='path/to/blocks.yaml'.",
        })

    # Preflight: validate PDK/EDA tools exist before resetting checkpoint
    from orchestrator.langgraph.pipeline_helpers import preflight_check
    check = preflight_check(["pipeline"])
    if not check["ok"]:
        return json.dumps({
            "error": "Preflight failed — required tools/PDK files missing",
            "details": check["errors"],
            "warnings": check.get("warnings", []),
            "hint": "Fix the missing dependencies before starting the pipeline.",
        })

    await _pipeline.reset_for_new_run()

    graph_config = {
        "configurable": {"thread_id": _pipeline.thread_id},
    }

    import time as _time

    initial_state = {
        "project_root": _project_root(),
        "target_clock_mhz": target_clock_mhz,
        "max_attempts": max_attempts,
        "block_queue": block_queue,
        "tier_list": [],
        "current_tier_index": 0,
        "completed_blocks": [],
        "integration_result": None,
        "pipeline_done": False,
        "pipeline_run_start": _time.time(),  # Fix #11: stale file detection
    }

    # Clear previous frontend/pipeline events but preserve architecture
    # and backend events so they remain visible in the timeline.
    events_path = Path(_project_root()) / ".coresmith" / "pipeline_events.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    if events_path.exists():
        kept_lines = []
        for line in events_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("graph") in ("architecture", "backend"):
                kept_lines.append(line)
        events_path.write_text(
            "\n".join(kept_lines) + "\n" if kept_lines else ""
        )

    _pipeline.task = asyncio.create_task(
        _pipeline.run_task(initial_state, graph_config)
    )
    await asyncio.sleep(0.1)

    result = {
        "status": _pipeline.status,
        "thread_id": _pipeline.thread_id,
        "blocks": len(block_queue),
        "block_names": [b["name"] for b in block_queue],
        "max_attempts": max_attempts,
        "target_clock_mhz": target_clock_mhz,
    }
    if _pipeline.error_message:
        result["error_message"] = _pipeline.error_message
    return json.dumps(result)


def _aggregate_failure_summary() -> dict:
    """Aggregate failure categories from pipeline events.

    Wrapper around :func:`aggregate_failure_categories` that reads
    events from disk and filters to frontend events.

    Returns:
        Failure summary dict, or empty dict if no events.
    """
    from orchestrator.langgraph.event_stream import (
        aggregate_failure_categories,
        read_events,
    )
    events = read_events(_project_root())
    frontend_events = [
        e for e in events
        if e.get("graph", "") not in ("architecture", "backend")
    ]
    if not frontend_events:
        return {}
    return aggregate_failure_categories(frontend_events)


@server.tool()
async def get_pipeline_state() -> str:
    """Get the current pipeline state from the checkpoint.

    Returns status, current block, phase, attempt, completed blocks,
    interrupt payload (if interrupted), and next nodes.
    """
    await _pipeline.ensure_graph()

    if not _pipeline.thread_id:
        return json.dumps({"status": "idle", "message": "No pipeline started"})

    config = {"configurable": {"thread_id": _pipeline.thread_id}}

    try:
        state_snapshot = await _pipeline.graph.aget_state(config)
    except Exception as e:
        return json.dumps({
            "status": _pipeline.status,
            "error": f"Failed to read state: {e}",
        })

    if not state_snapshot or not state_snapshot.values:
        return json.dumps({
            "status": _pipeline.status,
            "thread_id": _pipeline.thread_id,
        })

    values = state_snapshot.values
    completed = values.get("completed_blocks", [])
    block_queue = values.get("block_queue", [])
    tier_list = values.get("tier_list", [])
    current_tier_index = values.get("current_tier_index", 0)

    # Extract interrupt payloads (parallel blocks may have multiple)
    # Fix #10: also build interrupted_blocks list for per-block actions
    # Fix #11: filter stale interrupts from completed blocks
    completed_names = {b.get("name") for b in completed if b.get("name")}
    interrupt_payload = None
    interrupt_payloads = []
    interrupted_blocks = []
    if state_snapshot.tasks:
        for task in state_snapshot.tasks:
            for intr in task.interrupts:
                payload = intr.value
                # Skip stale interrupts from blocks that have already completed
                if isinstance(payload, dict):
                    block_name = payload.get("block", payload.get("block_name", ""))
                    if block_name and block_name in completed_names:
                        continue
                interrupt_payloads.append(payload)
                if isinstance(payload, dict):
                    interrupted_blocks.append({
                        "block_name": payload.get("block", payload.get("block_name", "")),
                        "error_summary": str(payload.get("previous_error", ""))[:300],
                        "interrupt_id": intr.id,
                        "phase": payload.get("phase", ""),
                    })
        if interrupt_payloads:
            interrupt_payload = interrupt_payloads[0]

    # Self-heal: if checkpoint has pending interrupts but in-memory status
    # hasn't caught up yet (race between ainvoke return and aget_state),
    # correct the status so the outer agent can resume immediately.
    # Only self-heal when the asyncio task has finished -- if it's still
    # running, the checkpoint data may be stale (race after resume).
    if interrupt_payloads and _pipeline.status == "running":
        if _pipeline.task is None or _pipeline.task.done():
            _pipeline.status = "interrupted"

    # Current tier info
    current_tier = tier_list[current_tier_index] if current_tier_index < len(tier_list) else None
    tier_blocks = [b for b in block_queue if b.get("tier", 1) == current_tier] if current_tier else []

    # Derive completed tiers
    total_tiers = len(tier_list) if tier_list else len(set(b.get("tier", 1) for b in block_queue))

    result = {
        "status": _pipeline.status,
        "thread_id": _pipeline.thread_id,
        "current_tier": current_tier,
        "current_tier_index": current_tier_index,
        "total_tiers": total_tiers,
        "tier_list": tier_list,
        "current_tier_blocks": [b.get("name", "") for b in tier_blocks],
        "max_attempts": values.get("max_attempts", 5),
        "next_nodes": list(state_snapshot.next) if state_snapshot.next else [],
        "completed_blocks": [
            {
                "name": b.get("name"),
                "success": b.get("success"),
                "step_log_paths": b.get("step_log_paths", {}),
            }
            for b in completed
        ],
        "completed_count": len(completed),
        "total_blocks": len(block_queue),
        "remaining_count": len(block_queue) - len(completed),
        "pipeline_done": values.get("pipeline_done", False),
        "interrupt_payload": interrupt_payload,
        "interrupt_payloads": interrupt_payloads if len(interrupt_payloads) > 1 else None,
        "interrupted_blocks": interrupted_blocks if interrupted_blocks else None,
        "pending_interrupt_count": len(interrupt_payloads),
        "interrupt_type": (
            interrupt_payload.get("type", "")
            if isinstance(interrupt_payload, dict) else None
        ),
        "interrupt_actions": (
            interrupt_payload.get("supported_actions", [])
            if isinstance(interrupt_payload, dict) else []
        ),
        "checkpoint_id": (
            state_snapshot.config.get("configurable", {}).get("checkpoint_id")
            if state_snapshot.config
            else None
        ),
    }

    # Add integration check results if available
    integration_result = values.get("integration_result")
    if integration_result:
        result["integration_result"] = {
            "design_name": integration_result.get("design_name"),
            "top_module": integration_result.get("top_module"),
            "top_rtl_path": integration_result.get("top_rtl_path"),
            "block_count": integration_result.get("block_count"),
            "wire_count": integration_result.get("wire_count"),
            "error_count": integration_result.get("error_count", 0),
            "warning_count": integration_result.get("warning_count", 0),
            "lint_clean": integration_result.get("lint_clean"),
            "skipped": integration_result.get("skipped", False),
            "skipped_reason": integration_result.get("reason"),
        }

    integration_dv_result = values.get("integration_dv_result")
    if integration_dv_result:
        result["integration_dv_result"] = {
            "passed": integration_dv_result.get("passed"),
            "skipped": integration_dv_result.get("skipped", False),
            "skipped_by_user": integration_dv_result.get("skipped_by_user", False),
            "test_count": integration_dv_result.get("test_count"),
            "testbench_path": integration_dv_result.get("testbench_path"),
            "sim_log_path": integration_dv_result.get("sim_log_path"),
        }

    validation_dv_result = values.get("validation_dv_result")
    if validation_dv_result:
        result["validation_dv_result"] = {
            "passed": validation_dv_result.get("passed"),
            "skipped": validation_dv_result.get("skipped", False),
            "skipped_by_user": validation_dv_result.get("skipped_by_user", False),
            "test_count": validation_dv_result.get("test_count"),
            "requirement_count": validation_dv_result.get("requirement_count"),
            "testbench_path": validation_dv_result.get("testbench_path"),
            "sim_log_path": validation_dv_result.get("sim_log_path"),
        }

    # Add failure trend summary if there are failures
    failure_summary = _aggregate_failure_summary()
    if failure_summary and failure_summary.get("total_failures", 0) > 0:
        result["failure_summary"] = failure_summary

    # Surface structured ask_question for pipeline interrupts
    if interrupt_payload and isinstance(interrupt_payload, dict):
        pipe_q = _build_pipeline_ask_question(interrupt_payload)
        result.update(pipe_q)

    if _pipeline.error_message:
        result["error_message"] = _pipeline.error_message

    diag = _get_diagnostics(graph_filter="pipeline")
    if diag and diag.get("failure_count", 0) > 0:
        result["diagnostics"] = diag

    # Guide the outer agent on what to do next when pipeline finishes
    if values.get("pipeline_done") and _pipeline.status == "done":
        all_passed = all(
            b.get("success") for b in completed
        ) if completed else False
        total = len(block_queue)
        passed = sum(1 for b in completed if b.get("success"))

        if all_passed and passed == total and total > 0:
            integration = values.get("integration_result") or {}
            integration_ok = (
                not integration.get("skipped", False)
                and integration.get("error_count", 0) == 0
            )
            integration_skipped_single_block = (
                integration.get("skipped", False)
                and total == passed
            )
            integration_dv = values.get("integration_dv_result") or {}
            validation_dv = values.get("validation_dv_result") or {}
            integration_dv_ok = integration_dv.get("passed") is True
            validation_dv_ok = validation_dv.get("passed") is True
            if (
                (integration_ok or integration_skipped_single_block)
                and integration_dv_ok
                and validation_dv_ok
            ):
                result["next_action"] = "start_backend"
                result["next_action_reason"] = (
                    f"All {passed}/{total} blocks passed frontend "
                    f"(lint + sim + synth). "
                    + (
                        "Integration check passed."
                        if integration_ok
                        else "Integration check skipped "
                             "(single-block design)."
                    )
                    + " Integration DV and Validation DV passed. "
                    "Call start_backend() to proceed to PnR/DRC/LVS."
                )

    return json.dumps(result, indent=2, default=str)


def _build_resume_command(state_snapshot, resume_value, action, constraint,
                          rtl_fix_description, block_actions):
    """Build a Command(resume=...) that handles multiple pending interrupts.

    When parallel blocks produce multiple pending interrupts, LangGraph
    requires a resume dict keyed by interrupt ID.  This helper inspects
    the state snapshot, enumerates all pending interrupts, and returns
    a properly-keyed Command.  Used by both the 'paused' and
    'interrupted' resume paths.

    Type-aware validation (Fix #11): when ``block_actions`` specifies a
    per-block action, the action is validated against the interrupt's
    ``supported_actions``.  If the action is not supported (e.g.
    ``approve`` sent to an ``ask_human`` interrupt), it is remapped to
    the type-appropriate default to prevent silent re-execution.
    """
    from langgraph.types import Command

    # Collect completed block names (any success status) to filter stale
    # interrupts.  Once a block lands in completed_blocks the pipeline has
    # moved past it, so any lingering interrupt from the parallel Send()
    # checkpoint is stale -- including failed-block interrupts, which the
    # pipeline already routed around.
    completed_names: set[str] = set()
    if state_snapshot and hasattr(state_snapshot, "values"):
        for b in (state_snapshot.values or {}).get("completed_blocks", []):
            name = b.get("name", "")
            if name:
                completed_names.add(name)

    # (interrupt_id, block_name, supported_actions)
    interrupt_info: list[tuple[str, str, list[str]]] = []
    if state_snapshot and state_snapshot.tasks:
        for task in state_snapshot.tasks:
            for intr in task.interrupts:
                payload = intr.value if hasattr(intr, "value") else {}
                block_name = ""
                supported = []
                if isinstance(payload, dict):
                    block_name = payload.get("block", payload.get("block_name", ""))
                    supported = payload.get("supported_actions", [])
                # Skip stale interrupts from completed blocks
                if block_name and block_name in completed_names:
                    continue
                interrupt_info.append((intr.id, block_name, supported))

    if not interrupt_info:
        return None  # No pending interrupts, resume with None

    [iid for iid, _, _ in interrupt_info]

    per_block: dict[str, str] = {}
    if block_actions:
        try:
            per_block = json.loads(block_actions)
        except (json.JSONDecodeError, TypeError):
            pass

    # Always build an interrupt-ID-keyed resume map, even for a single
    # interrupt.  LangGraph's _pending_interrupts() counts checkpoint
    # writes which may disagree with our stale-block filtering.  Using
    # the keyed format avoids the "multiple pending interrupts" error.
    resume_map = {}
    for iid, bname, supported in interrupt_info:
        block_action = action
        if per_block and bname in per_block:
            block_action = per_block[bname]
        # Validate action against supported_actions for the interrupt
        if supported and block_action not in supported:
            # Remap to type-appropriate default
            block_action = supported[0] if supported else action
            import logging
            logging.getLogger(__name__).warning(
                "Action '%s' not in supported_actions %s for block '%s'; "
                "remapped to '%s'",
                per_block.get(bname, action) if per_block else action,
                supported, bname, block_action,
            )
        resume_map[iid] = {
            "action": block_action,
            "constraint": constraint,
            "description": rtl_fix_description,
        }
    return Command(resume=resume_map)


@server.tool()
async def resume_pipeline(
    action: str,
    constraint: str = "",
    rtl_fix_description: str = "",
    block_actions: str = "",
) -> str:
    """Resume the pipeline after an interrupt or pause.

    The graph must be in 'interrupted' or 'paused' status.  The action
    tells the graph what to do next.

    Args:
        action: One of 'retry', 'fix_rtl', 'add_constraint', 'skip', 'abort'.
        constraint: Constraint text (required if action is 'add_constraint').
        rtl_fix_description: Description of the RTL fix applied on disk
            (informational, used if action is 'fix_rtl').
        block_actions: Optional JSON dict mapping block_name -> action for
            per-block interrupt handling (e.g. '{"scrambler": "retry",
            "viterbi_decoder": "skip"}'). When provided, each interrupted
            block gets its own action instead of using the global *action*.
    """
    valid_actions = {"retry", "fix_rtl", "fix_tb", "add_constraint", "skip", "abort",
                     "approve", "revise"}
    if action not in valid_actions:
        return json.dumps({
            "error": f"Invalid action: {action}. Must be one of: {sorted(valid_actions)}",
        })

    if _pipeline.status not in ("interrupted", "paused"):
        # Self-heal: check checkpoint for pending interrupts that the
        # in-memory status hasn't caught up with yet (race condition).
        # Only self-heal when the asyncio task has finished -- if it's
        # still running, the checkpoint data may be stale.
        if _pipeline.status == "running" and _pipeline.thread_id:
            if _pipeline.task is None or _pipeline.task.done():
                try:
                    await _pipeline.ensure_graph()
                    _config = {"configurable": {"thread_id": _pipeline.thread_id}}
                    _snap = await _pipeline.graph.aget_state(_config)
                    if _snap and _snap.tasks:
                        for _t in _snap.tasks:
                            if _t.interrupts:
                                _pipeline.status = "interrupted"
                                break
                except Exception:
                    pass

        if _pipeline.status not in ("interrupted", "paused"):
            return json.dumps({
                "error": f"Cannot resume: pipeline status is '{_pipeline.status}'",
                "hint": "Pipeline must be 'interrupted' or 'paused' to resume.",
            })

    if action == "add_constraint" and not constraint:
        return json.dumps({
            "error": "constraint parameter is required when action is 'add_constraint'",
        })

    await _pipeline.ensure_graph()

    from langgraph.types import Command

    config = {"configurable": {"thread_id": _pipeline.thread_id}}

    # Validate action against the interrupt's supported_actions to prevent
    # sending e.g. "retry" to a uarch_spec_review interrupt or "approve"
    # to a human_intervention interrupt.
    if _pipeline.status == "interrupted":
        try:
            _check_snap = await _pipeline.graph.aget_state(config)
            if _check_snap and _check_snap.tasks:
                for _t in _check_snap.tasks:
                    for _intr in _t.interrupts:
                        _payload = _intr.value
                        if isinstance(_payload, dict):
                            _supported = _payload.get("supported_actions", [])
                            if _supported and action not in _supported:
                                return json.dumps({
                                    "error": (
                                        f"Action '{action}' is not valid for interrupt "
                                        f"type '{_payload.get('type', 'unknown')}'. "
                                        f"Supported actions: {_supported}"
                                    ),
                                })
                        break  # only check first interrupt
                    break
        except Exception:
            pass  # Non-fatal -- proceed without validation

    resume_value = {
        "action": action,
        "constraint": constraint,
        "description": rtl_fix_description,
    }

    config = {"configurable": {"thread_id": _pipeline.thread_id}}

    # Unified resume path for both "paused" and "interrupted" states.
    # Handles multiple pending interrupts from parallel block processing.
    try:
        _snap = await _pipeline.graph.aget_state(config)
        resume_input = _build_resume_command(
            _snap, resume_value, action, constraint,
            rtl_fix_description, block_actions,
        )
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(
            "Failed to build interrupt-ID-keyed resume command: %s. "
            "Falling back to interrupt-ID-keyed format using snapshot tasks.",
            exc,
        )
        # Fallback: try to get interrupt IDs from the snapshot directly
        # and build a keyed map.  Only use the flat format as a last resort.
        from langgraph.types import Command
        try:
            _snap2 = await _pipeline.graph.aget_state(config)
            iids = []
            if _snap2 and _snap2.tasks:
                for _t in _snap2.tasks:
                    for _intr in _t.interrupts:
                        iids.append(_intr.id)
            if iids:
                resume_input = Command(resume={iid: resume_value for iid in iids})
            else:
                resume_input = Command(resume=resume_value)
        except Exception:
            resume_input = Command(resume=resume_value)

    _pipeline.task = asyncio.create_task(
        _pipeline.run_task(resume_input, config)
    )

    result = {
        "status": "running",
        "action": action,
        "thread_id": _pipeline.thread_id,
    }
    if _pipeline.error_message:
        result["error_message"] = _pipeline.error_message
    return json.dumps(result)


@server.tool()
async def pause_pipeline() -> str:
    """Pause the running pipeline.

    Cancels the background task and kills any stuck Claude CLI
    subprocesses.  The graph state is preserved at the last completed
    node boundary (via checkpoint).  Call resume_pipeline() to continue
    from that point.
    """
    if _pipeline.status != "running":
        return json.dumps({
            "error": f"Cannot pause: pipeline status is '{_pipeline.status}'",
        })

    # Kill any stuck CLI subprocesses first so the task can actually cancel
    from orchestrator.langchain.agents.coresmith_llm import kill_active_cli_processes
    kill_active_cli_processes()

    if _pipeline.task and not _pipeline.task.done():
        _pipeline.task.cancel()
        try:
            await _pipeline.task
        except asyncio.CancelledError:
            pass

    _pipeline.status = "paused"

    # Read current state for the response
    await _pipeline.ensure_graph()
    config = {"configurable": {"thread_id": _pipeline.thread_id}}
    try:
        state_snapshot = await _pipeline.graph.aget_state(config)
        values = state_snapshot.values if state_snapshot else {}
        completed_count = len(values.get("completed_blocks", []))
        total_blocks = len(values.get("block_queue", []))
        current_tier = None
        tier_list = values.get("tier_list", [])
        idx = values.get("current_tier_index", 0)
        if tier_list and idx < len(tier_list):
            current_tier = tier_list[idx]
    except Exception:
        completed_count = 0
        total_blocks = 0
        current_tier = None

    result = {
        "status": "paused",
        "current_tier": current_tier,
        "completed_count": completed_count,
        "total_blocks": total_blocks,
        "thread_id": _pipeline.thread_id,
        "message": "Pipeline paused. Call resume_pipeline() to continue.",
    }
    if _pipeline.error_message:
        result["error_message"] = _pipeline.error_message
    return json.dumps(result)


@server.tool()
async def restart_node(node_name: str) -> str:
    """Re-execute from a specific node by forking from its checkpoint.

    Walks the checkpoint history to find the state just before the
    target node, then re-invokes the graph from that point.  Useful
    after editing RTL on disk to re-run lint without regenerating.

    The pipeline must be 'interrupted' or 'paused'.

    Args:
        node_name: The node to restart from (e.g., 'lint', 'simulate',
            'generate_rtl').
    """
    if _pipeline.status == "running":
        return json.dumps({
            "error": "Cannot restart: pipeline is currently running",
            "hint": "Pause the pipeline first with pause_pipeline(), then restart.",
        })
    if not _pipeline.thread_id:
        return json.dumps({
            "error": "Cannot restart: no pipeline run exists",
            "hint": "Start a pipeline first with start_pipeline().",
        })

    await _pipeline.ensure_graph()
    config = {"configurable": {"thread_id": _pipeline.thread_id}}

    # Walk history to find the checkpoint where node_name is next
    found_config = None
    async for state_snapshot in _pipeline.graph.aget_state_history(config):
        if state_snapshot.next and node_name in state_snapshot.next:
            found_config = state_snapshot.config
            break

    if not found_config:
        return json.dumps({
            "error": f"No checkpoint found with '{node_name}' as next node",
            "hint": "Orchestrator nodes: init_tier, process_block, "
                    "advance_tier, pipeline_complete. "
                    "Block subgraph nodes: init_block, generate_uarch_spec, "
                    "review_uarch_spec, generate_rtl, "
                    "generate_testbench, synthesize, diagnose, "
                    "decide, ask_human, block_done",
        })

    # Fork from the found checkpoint
    fork_config = {
        "configurable": {
            "thread_id": _pipeline.thread_id,
            "checkpoint_id": found_config["configurable"]["checkpoint_id"],
        },
    }

    _pipeline.task = asyncio.create_task(
        _pipeline.run_task(None, fork_config)
    )

    return json.dumps({
        "status": "running",
        "restarting_from": node_name,
        "thread_id": _pipeline.thread_id,
        "checkpoint_id": found_config["configurable"]["checkpoint_id"],
    })


@server.tool()
async def skip_block() -> str:
    """Skip the current block and advance to the next one.

    Convenience wrapper: if interrupted, resumes with action='skip'.
    If paused, resumes and then the graph will continue from wherever
    it was checkpointed.
    """
    if _pipeline.status == "interrupted":
        return await resume_pipeline(action="skip")
    elif _pipeline.status == "paused":
        # For paused state, we need to update state to skip
        return await resume_pipeline(action="skip")
    else:
        return json.dumps({
            "error": f"Cannot skip: pipeline status is '{_pipeline.status}'",
            "hint": "Pipeline must be interrupted or paused.",
        })


@server.tool()
async def restart_block(
    block_name: str,
    from_node: str = "generate_rtl",
    uarch_feedback: str = "",
    max_attempts: int = 3,
) -> str:
    """Restart a single block from a specific node in its lifecycle.

    Runs the block through a standalone block subgraph independent of
    the main pipeline checkpoint.  Useful after integration check finds
    a cross-block mismatch (clock polarity, reset naming, port width) --
    edit the uArch spec or RTL on disk, then restart just that block.

    After completion, call restart_node('integration_check') to re-verify
    the integrated design.

    The uArch review interrupt is auto-approved (the spec was already
    reviewed in the original pipeline run).  If the block fails during
    its lifecycle and the debug agent escalates to ask_human, the failure
    payload is returned so the outer agent can diagnose and retry.

    Args:
        block_name: Name of the block to restart (e.g. 'fft_butterfly').
        from_node: Where to start the lifecycle.  One of:
            'generate_uarch_spec' -- regenerate uArch spec, then RTL/lint/sim/synth
            'generate_rtl' -- regenerate RTL from existing uArch spec (default)
        uarch_feedback: Feedback for uArch spec regeneration (only used
            when from_node is 'generate_uarch_spec').
        max_attempts: Max retry attempts for the block lifecycle (default 3).
    """
    import time as _time

    valid_nodes = ("generate_uarch_spec", "generate_rtl")
    if from_node not in valid_nodes:
        return json.dumps({
            "error": f"Invalid from_node '{from_node}'",
            "hint": f"Must be one of: {', '.join(valid_nodes)}",
        })

    root = Path(_project_root())

    # ── Load block spec from block_specs.json ─────────────────────────
    specs_path = root / ".coresmith" / "block_specs.json"
    block_spec = None
    if specs_path.exists():
        try:
            specs = json.loads(specs_path.read_text())
            for spec in specs:
                if spec.get("name") == block_name:
                    block_spec = spec
                    break
        except (json.JSONDecodeError, TypeError):
            pass

    if not block_spec:
        return json.dumps({
            "error": f"Block '{block_name}' not found in .coresmith/block_specs.json",
            "hint": "The block must exist in the architecture output.",
        })

    # ── Build standalone block subgraph ───────────────────────────────
    from langgraph.checkpoint.memory import MemorySaver

    from orchestrator.langgraph.pipeline_graph import build_block_subgraph

    checkpointer = MemorySaver()
    block_graph = build_block_subgraph().compile(checkpointer=checkpointer)

    thread_id = f"restart-{block_name}-{int(_time.time())}"
    config = {"configurable": {"thread_id": thread_id}}

    # ── Inherit target_clock_mhz from the main pipeline if available ──
    target_clock = 50.0
    effective_max = max_attempts
    if _pipeline.thread_id and _pipeline.graph:
        try:
            await _pipeline.ensure_graph()
            p_config = {"configurable": {"thread_id": _pipeline.thread_id}}
            p_snap = await _pipeline.graph.aget_state(p_config)
            if p_snap and p_snap.values:
                target_clock = p_snap.values.get(
                    "target_clock_mhz", 50.0
                )
        except Exception:
            pass

    # ── Build initial BlockState ──────────────────────────────────────
    initial_state: dict[str, Any] = {
        "project_root": str(root),
        "target_clock_mhz": target_clock,
        "max_attempts": effective_max,
        "pipeline_run_start": _time.time(),
        "current_block": block_spec,
        "attempt": 1,
        "phase": "init",
        "constraints": [],
        "attempt_history": [],
        "previous_error": "",
        "uarch_spec": None,
        "uarch_approved": False,
        "uarch_feedback": uarch_feedback,
        "rtl_result": None,
        "lint_result": None,
        "tb_result": None,
        "sim_result": None,
        "synth_result": None,
        "debug_result": None,
        "human_response": None,
        "completed_blocks": [],
        "step_log_paths": {},
        "preserve_testbench": False,
    }

    # ── Position the graph at the desired starting node ───────────────
    run_input: Any = initial_state

    if from_node == "generate_rtl":
        uarch_spec_path = root / ARCH_DOC_DIR / "uarch_specs" / f"{block_name}.md"
        uarch_spec: dict[str, Any] = {}
        if uarch_spec_path.exists():
            spec_text = uarch_spec_path.read_text(encoding="utf-8")
            uarch_spec = {
                "spec_text": spec_text,
                "spec_path": str(uarch_spec_path),
                "spec_summary": {},
                "block_name": block_name,
            }
        initial_state["uarch_spec"] = uarch_spec
        initial_state["uarch_approved"] = True
        initial_state["human_response"] = {"action": "approve"}
        initial_state["phase"] = "rtl"
        await block_graph.aupdate_state(
            config, initial_state, as_node="review_uarch_spec",
        )
        run_input = None

    # else: from_node == "generate_uarch_spec" -- start from the top

    # ── Run the block lifecycle ───────────────────────────────────────
    from langgraph.errors import GraphInterrupt

    max_interrupt_loops = 10
    for _ in range(max_interrupt_loops):
        try:
            await block_graph.ainvoke(run_input, config)
        except GraphInterrupt:
            pass

        snap = await block_graph.aget_state(config)
        if not snap or not snap.next:
            break

        # Handle pending interrupts
        handled = False
        if snap.tasks:
            for task in snap.tasks:
                for intr in task.interrupts:
                    payload = intr.value
                    if not isinstance(payload, dict):
                        continue
                    itype = payload.get("type", "")

                    if itype == "uarch_spec_review":
                        from langgraph.types import Command
                        run_input = Command(
                            resume={intr.id: {"action": "approve"}},
                        )
                        handled = True
                        break

                    if itype == "human_intervention_needed":
                        return json.dumps({
                            "status": "needs_intervention",
                            "block_name": block_name,
                            "from_node": from_node,
                            "interrupt_payload": payload,
                            "hint": (
                                "The block failed during its lifecycle. "
                                "Edit the RTL or uArch spec on disk and "
                                "call restart_block() again."
                            ),
                        })

                if handled:
                    break

        if not handled:
            break

    # ── Collect final result ──────────────────────────────────────────
    final_snap = await block_graph.aget_state(config)
    final_values = final_snap.values if final_snap else {}
    completed = final_values.get("completed_blocks", [])

    block_result: dict[str, Any] = {}
    for b in completed:
        if b.get("name") == block_name:
            block_result = b
            break

    # Merge result back into the main pipeline checkpoint
    merged = False
    if block_result:
        merged = await _merge_block_into_pipeline_checkpoint(block_result)

    next_steps = []
    if block_result.get("success"):
        next_steps.append(
            "Block passed all steps. Call restart_node('integration_check') "
            "to re-verify the integrated design."
        )
    else:
        error_hint = block_result.get("error", "unknown failure")
        next_steps.append(
            f"Block did not pass ({error_hint}). Inspect the step logs, "
            "edit RTL on disk, and call restart_block() again with "
            "from_node='lint'."
        )

    return json.dumps({
        "status": "completed",
        "block_name": block_name,
        "from_node": from_node,
        "result": block_result,
        "success": block_result.get("success", False),
        "checkpoint_merged": merged,
        "next_steps": next_steps,
    })


async def _merge_block_into_pipeline_checkpoint(block_result: dict) -> bool:
    """Merge a block result back into the pipeline checkpoint's completed_blocks.

    Reads the pipeline checkpoint, updates (or appends) the entry for the
    given block, and writes back via ``aupdate_state``.

    Returns True on success, False on any failure (non-fatal).
    """
    if not _pipeline.graph or not _pipeline.thread_id:
        return False
    try:
        config = {"configurable": {"thread_id": _pipeline.thread_id}}
        snap = await _pipeline.graph.aget_state(config)
        if not snap or not snap.values:
            return False

        block_name = block_result.get("name", "")
        if not block_name:
            return False

        # ``completed_blocks`` uses an ``operator.add`` (append) reducer, and the
        # downstream gates (pipeline_complete / _current_phase_completed) dedup by
        # name keeping the LAST entry. So the authoritative way to record a
        # restart_block result is to APPEND it -- it then becomes the last entry
        # for that name and last-wins dedup surfaces it. The previous
        # implementation read the list, replaced the FIRST name-match, and wrote
        # the whole list back; because the reducer APPENDS, that both duplicated
        # the list AND left the genuine result at a non-last slot (a later stale
        # entry for the same block could still win), so a restart_block PASS was
        # invisible to the gate and the run re-parked at pipeline_incomplete.
        await _pipeline.graph.aupdate_state(
            config,
            {"completed_blocks": [block_result]},
            as_node="process_block",
        )
        return True
    except Exception:
        logging.getLogger(__name__).warning(
            "Failed to merge block '%s' into pipeline checkpoint",
            block_result.get("name", "?"),
            exc_info=True,
        )
        return False


async def _merge_block_into_backend_checkpoint(block_result: dict) -> bool:
    """Merge a block result into the backend checkpoint's completed_blocks.

    Mirrors ``_merge_block_into_pipeline_checkpoint`` but operates on
    the backend graph checkpoint so that ``run_backend_step`` results
    are visible to downstream gates (tapeout readiness, etc.).
    """
    if not _backend.graph or not _backend.thread_id:
        return False
    try:
        config = {"configurable": {"thread_id": _backend.thread_id}}
        snap = await _backend.graph.aget_state(config)
        if not snap or not snap.values:
            return False

        completed = list(snap.values.get("completed_blocks", []))
        block_name = block_result.get("name", "")
        if not block_name:
            return False

        replaced = False
        for i, b in enumerate(completed):
            if b.get("name") == block_name:
                completed[i] = block_result
                replaced = True
                break
        if not replaced:
            completed.append(block_result)

        await _backend.graph.aupdate_state(
            config,
            {"completed_blocks": completed},
            as_node="advance_block",
        )
        return True
    except Exception:
        logging.getLogger(__name__).warning(
            "Failed to merge block '%s' into backend checkpoint",
            block_result.get("name", "?"),
            exc_info=True,
        )
        return False


@server.tool()
async def run_step(
    step: str,
    block_name: str,
    rtl_path: str = "",
    target_clock_mhz: float = 50.0,
) -> str:
    """Run a single EDA step on a block without the full pipeline.

    Runs lint, simulate, or synthesize directly on the specified block's
    RTL. Useful for quick iteration after editing RTL on disk. Results
    include full log file paths in .coresmith/step_logs/.

    This does NOT use the pipeline graph or checkpoints -- it's a direct
    tool invocation. Use restart_node() if you need full graph state
    (constraints, attempt history, etc.).

    Args:
        step: One of 'lint', 'simulate', 'synthesize'.
        block_name: Block name (e.g. 'scrambler', 'adder_16bit').
        rtl_path: Path to RTL file. If empty, auto-resolved from
            block_specs.json, config.yaml, or rtl/<block_name>/<block_name>.v.
        target_clock_mhz: Target clock for synthesis (default 50.0).
    """
    valid_steps = ("lint", "simulate", "synthesize")
    if step not in valid_steps:
        return json.dumps({
            "error": f"Invalid step '{step}'",
            "hint": f"Must be one of: {', '.join(valid_steps)}",
        })

    root = Path(_project_root())

    # ── Resolve RTL path ──────────────────────────────────────────────
    resolved_rtl = rtl_path
    block_dict: dict = {"name": block_name}

    if not resolved_rtl:
        # Priority 1: block_specs.json (architecture output)
        specs_path = root / ".coresmith" / "block_specs.json"
        if specs_path.exists():
            try:
                specs = json.loads(specs_path.read_text())
                for spec in specs:
                    if spec.get("name") == block_name:
                        resolved_rtl = spec.get("rtl_target", "")
                        block_dict = {**block_dict, **spec}
                        break
            except (json.JSONDecodeError, TypeError):
                pass

        # Priority 2: config.yaml blocks section
        if not resolved_rtl:
            from orchestrator.langgraph.pipeline_helpers import load_config
            config = load_config()
            blocks = config.get("blocks", {})
            if block_name in blocks:
                spec = blocks[block_name]
                resolved_rtl = spec.get("rtl_target", "")
                block_dict = {**block_dict, **spec}

        # Priority 3: convention rtl/<block_name>/<block_name>.v
        if not resolved_rtl:
            candidate = root / "rtl" / block_name / f"{block_name}.v"
            if candidate.exists():
                resolved_rtl = str(candidate)

    # Make absolute if relative
    if resolved_rtl and not os.path.isabs(resolved_rtl):
        resolved_rtl = str(root / resolved_rtl)

    if not resolved_rtl or not os.path.isfile(resolved_rtl):
        return json.dumps({
            "error": f"RTL file not found for block '{block_name}'",
            "searched": [
                ".coresmith/block_specs.json",
                "config.yaml blocks section",
                f"rtl/{block_name}/{block_name}.v",
            ],
            "hint": "Provide rtl_path explicitly, e.g. run_step(step='lint', "
                    f"block_name='{block_name}', rtl_path='rtl/path/to/file.v')",
        })

    # ── Run the step ──────────────────────────────────────────────────
    from orchestrator.langgraph.event_stream import write_graph_event
    from orchestrator.langgraph.pipeline_helpers import (
        lint_rtl,
        run_simulation,
        synthesize_block,
    )

    # Map step names to the node names the webview timeline expects
    _step_node_names = {
        "lint": "Lint Check",
        "simulate": "Simulate",
        "synthesize": "Synthesize",
    }
    node_label = _step_node_names[step]
    project_root = str(root)

    try:
        if step == "lint":
            write_graph_event(project_root, node_label, "graph_node_enter", {
                "block": block_name,
                "source": "run_step",
            })
            result = await asyncio.to_thread(lint_rtl, resolved_rtl, block_name)
            lint_output = result.get("errors", "") or result.get("warnings", "")
            write_graph_event(project_root, node_label, "graph_node_exit", {
                "block": block_name,
                "clean": result.get("clean", False),
                "tool_stdout": lint_output[:2000] if lint_output else "",
                "log_path": result.get("log_path", ""),
                "source": "run_step",
            })
            merged = False
            if result.get("clean", False):
                merged = await _merge_block_into_pipeline_checkpoint({
                    "name": block_name,
                    "success": True,
                    "lint_clean": True,
                })
            return json.dumps({
                "step": "lint",
                "block_name": block_name,
                "rtl_path": resolved_rtl,
                "checkpoint_merged": merged,
                **result,
            })

        elif step == "simulate":
            # Resolve testbench path
            tb_path = block_dict.get("testbench", "")
            if not tb_path:
                tb_candidate = root / "tb" / "cocotb" / f"test_{block_name}.py"
                if tb_candidate.exists():
                    tb_path = str(tb_candidate)
            if tb_path and not os.path.isabs(tb_path):
                tb_path = str(root / tb_path)
            if not tb_path or not os.path.isfile(tb_path):
                return json.dumps({
                    "error": f"Testbench not found for block '{block_name}'",
                    "searched": [
                        "block_specs.json 'testbench' field",
                        f"tb/cocotb/test_{block_name}.py",
                    ],
                    "hint": "Simulate requires a cocotb testbench file.",
                })
            write_graph_event(project_root, node_label, "graph_node_enter", {
                "block": block_name,
                "source": "run_step",
            })
            result = await asyncio.to_thread(
                run_simulation, block_dict, resolved_rtl, tb_path,
            )
            sim_log = result.get("log", "")
            write_graph_event(project_root, node_label, "graph_node_exit", {
                "block": block_name,
                "passed": result.get("passed", False),
                "tool_stdout": sim_log[-1500:] if sim_log else "",
                "log_path": result.get("log_path", ""),
                "source": "run_step",
            })
            merged = False
            if result.get("passed", False):
                merged = await _merge_block_into_pipeline_checkpoint({
                    "name": block_name,
                    "success": True,
                    "sim_passed": True,
                })
            return json.dumps({
                "step": "simulate",
                "block_name": block_name,
                "rtl_path": resolved_rtl,
                "testbench_path": tb_path,
                "checkpoint_merged": merged,
                **result,
            })

        else:  # synthesize
            write_graph_event(project_root, node_label, "graph_node_enter", {
                "block": block_name,
                "source": "run_step",
            })
            result = await asyncio.to_thread(
                synthesize_block, block_dict, resolved_rtl,
                target_clock_mhz=target_clock_mhz,
            )
            synth_log = result.get("log", "") or result.get("errors", "")
            write_graph_event(project_root, node_label, "graph_node_exit", {
                "block": block_name,
                "success": result.get("success", False),
                "gate_count": result.get("gate_count", 0),
                "chip_area_um2": result.get("chip_area_um2", 0.0),
                "tool_stdout": synth_log[:2000] if synth_log else "",
                "log_path": result.get("log_path", ""),
                "source": "run_step",
            })
            # Best-effort: merge successful synthesis into pipeline checkpoint
            merged = False
            if result.get("success"):
                merged = await _merge_block_into_pipeline_checkpoint({
                    "name": block_name,
                    "success": True,
                    "gate_count": result.get("gate_count", 0),
                    "chip_area_um2": result.get("chip_area_um2", 0.0),
                    "synth_success": True,
                    "sim_passed": True,
                })
            return json.dumps({
                "step": "synthesize",
                "block_name": block_name,
                "rtl_path": resolved_rtl,
                "target_clock_mhz": target_clock_mhz,
                "checkpoint_merged": merged,
                **result,
            })

    except Exception as e:
        return json.dumps({
            "error": f"Step '{step}' failed with exception: {e}",
            "step": step,
            "block_name": block_name,
            "rtl_path": resolved_rtl,
        })


# ═══════════════════════════════════════════════════════════════════════════
# BACKEND CONTROL TOOLS -- start, inspect, pause, resume the backend graph
#
# Mirrors the frontend pipeline control tools but drives the backend
# physical design graph (floorplan -> PnR -> DRC -> LVS -> timing -> power).
# ═══════════════════════════════════════════════════════════════════════════


@server.tool()
async def start_backend(
    max_attempts: int = 3,
    target_clock_mhz: float = 50.0,
) -> str:
    """Start the backend physical design graph as a background task.

    Loads blocks from the frontend's completed results and launches
    the LangGraph backend graph.  The graph runs autonomously -- it
    only pauses when failures need human review (interrupt).

    Call get_backend_state() to monitor progress.

    Args:
        max_attempts: Maximum retry attempts per block (default 3).
        target_clock_mhz: Target clock frequency in MHz (default 50.0).
    """
    if _backend.status == "running":
        return json.dumps({
            "error": "Backend already running",
            "thread_id": _backend.thread_id,
            "status": _backend.status,
        })

    await _backend.ensure_graph()

    # Load frontend completed_blocks from pipeline checkpoint if available
    frontend_blocks: list[dict] = []
    try:
        if _pipeline.graph and _pipeline.thread_id:
            await _pipeline.ensure_graph()
            p_config = {"configurable": {"thread_id": _pipeline.thread_id}}
            p_snap = await _pipeline.graph.aget_state(p_config)
            if p_snap and p_snap.values:
                frontend_blocks = p_snap.values.get("completed_blocks", [])
    except Exception:
        pass

    # Fall back to block_specs.json
    specs_path = Path(_project_root()) / ".coresmith" / "block_specs.json"
    if specs_path.exists():
        block_queue = json.loads(specs_path.read_text())
    else:
        from orchestrator.langgraph.pipeline_helpers import (
            get_sorted_block_queue,
            load_config,
        )
        config = load_config()
        block_queue = get_sorted_block_queue(config)

    if not frontend_blocks:
        frontend_blocks = block_queue

    if not block_queue:
        return json.dumps({"error": "No blocks found in block_specs.json or config.yaml"})

    # Load architecture connections
    from orchestrator.langgraph.integration_helpers import load_architecture_connections
    architecture_connections, design_name = load_architecture_connections(_project_root())

    # Gate: verify ALL blocks have RTL and synthesis artifacts on disk
    root = Path(_project_root())
    missing_rtl = []
    missing_synth = []
    for bspec in block_queue:
        bname = bspec["name"] if isinstance(bspec, dict) else bspec
        rtl_target = bspec.get("rtl_target", "") if isinstance(bspec, dict) else ""
        rtl_found = False
        if rtl_target and (root / rtl_target).exists():
            rtl_found = True
        else:
            for candidate in (
                root / "rtl" / bname / f"{bname}.v",
                root / "rtl" / f"{bname}.v",
            ):
                if candidate.exists():
                    rtl_found = True
                    break
            if not rtl_found:
                rtl_dir = root / "rtl"
                if rtl_dir.is_dir():
                    for sub in rtl_dir.iterdir():
                        if sub.is_dir() and (sub / f"{bname}.v").exists():
                            rtl_found = True
                            break
        if not rtl_found:
            missing_rtl.append(bname)

        netlist = root / "syn" / "output" / bname / f"{bname}_netlist.v"
        if not netlist.exists():
            missing_synth.append(bname)

    if missing_rtl or missing_synth:
        return json.dumps({
            "error": "Backend gate failed: not all blocks have required artifacts",
            "missing_rtl": missing_rtl,
            "missing_synthesis": missing_synth,
            "hint": "All blocks must pass frontend (RTL + synthesis) before backend. "
                    "Re-run the pipeline or restart failed blocks.",
        })

    # Preflight: validate backend PDK/EDA tools exist before resetting checkpoint
    from orchestrator.langgraph.pipeline_helpers import preflight_check
    check = preflight_check(["backend"])
    if not check["ok"]:
        return json.dumps({
            "error": "Preflight failed — required backend tools/PDK files missing",
            "details": check["errors"],
            "warnings": check.get("warnings", []),
            "hint": "Fix the missing dependencies before starting the backend.",
        })

    await _backend.reset_for_new_run()

    graph_config = {
        "configurable": {"thread_id": _backend.thread_id},
    }

    initial_state = {
        "project_root": _project_root(),
        "target_clock_mhz": target_clock_mhz,
        "max_attempts": max_attempts,
        "block_queue": block_queue,
        # Backend Lead fields
        "frontend_blocks": frontend_blocks,
        "architecture_connections": architecture_connections,
        "design_name": design_name,
        "block_rtl_paths": {},
        "glue_blocks": [],
        "integration_top_path": "",
        "flat_netlist_path": "",
        "flat_sdc_path": "",
        "synth_gate_count": 0,
        "synth_area_um2": 0.0,
        # Legacy compat
        "current_block_index": 0,
        "current_block": {},
        "attempt": 1,
        "phase": "init",
        "constraints": [],
        "attempt_history": [],
        "previous_error": "",
        "floorplan_result": None,
        "place_result": None,
        "cts_result": None,
        "route_result": None,
        "drc_result": None,
        "lvs_result": None,
        "timing_result": None,
        "power_result": None,
        "debug_result": None,
        "completed_blocks": [],
        "human_response": None,
        "backend_done": False,
        "routed_def_path": "",
        "pnr_verilog_path": "",
        "pwr_verilog_path": "",
        "spef_path": "",
        "gds_path": "",
        "spice_path": "",
        "step_log_paths": {},
    }

    _backend.task = asyncio.create_task(
        _backend.run_task(initial_state, graph_config)
    )
    await asyncio.sleep(0.1)

    result = {
        "status": _backend.status,
        "thread_id": _backend.thread_id,
        "blocks": len(block_queue),
        "block_names": [b["name"] for b in block_queue],
        "max_attempts": max_attempts,
        "target_clock_mhz": target_clock_mhz,
    }
    if _backend.error_message:
        result["error_message"] = _backend.error_message
    return json.dumps(result)


@server.tool()
async def get_backend_state() -> str:
    """Get the current backend graph state from the checkpoint.

    Returns status, current block, phase, attempt, completed blocks,
    interrupt payload (if interrupted), and next nodes.
    """
    await _backend.ensure_graph()

    if not _backend.thread_id:
        return json.dumps({"status": "idle", "message": "No backend started"})

    config = {"configurable": {"thread_id": _backend.thread_id}}

    try:
        state_snapshot = await _backend.graph.aget_state(config)
    except Exception as e:
        return json.dumps({
            "status": _backend.status,
            "error": f"Failed to read state: {e}",
        })

    if not state_snapshot or not state_snapshot.values:
        return json.dumps({
            "status": _backend.status,
            "thread_id": _backend.thread_id,
        })

    values = state_snapshot.values
    current_block = values.get("current_block") or {}
    completed = values.get("completed_blocks", [])
    block_queue = values.get("block_queue", [])

    interrupt_payload = None
    if state_snapshot.tasks:
        for task in state_snapshot.tasks:
            if task.interrupts:
                interrupt_payload = task.interrupts[0].value
                break

    # Self-heal: if checkpoint has pending interrupts but in-memory status
    # hasn't caught up yet, correct it so the outer agent can resume.
    # Only self-heal when the asyncio task has finished -- if it's still
    # running, the checkpoint data may be stale (race after resume).
    if interrupt_payload and _backend.status == "running":
        if _backend.task is None or _backend.task.done():
            _backend.status = "interrupted"

    result = {
        "status": _backend.status,
        "thread_id": _backend.thread_id,
        "current_block": current_block.get("name", ""),
        "current_block_tier": current_block.get("tier", 0),
        "phase": values.get("phase", ""),
        "attempt": values.get("attempt", 0),
        "max_attempts": values.get("max_attempts", 3),
        "next_nodes": list(state_snapshot.next) if state_snapshot.next else [],
        "previous_error": (values.get("previous_error") or "")[:1000],
        "attempt_history": values.get("attempt_history", []),
        "constraints": values.get("constraints", []),
        "completed_blocks": [
            {"name": b.get("name"), "success": b.get("success")}
            for b in completed
        ],
        "completed_count": len(completed),
        "total_blocks": len(block_queue),
        "remaining_count": len(block_queue) - values.get("current_block_index", 0),
        "backend_done": values.get("backend_done", False),
        "interrupt_payload": interrupt_payload,
        "checkpoint_id": (
            state_snapshot.config.get("configurable", {}).get("checkpoint_id")
            if state_snapshot.config
            else None
        ),
    }

    if _backend.error_message:
        result["error_message"] = _backend.error_message

    diag = _get_diagnostics(graph_filter="backend")
    if diag and diag.get("failure_count", 0) > 0:
        result["diagnostics"] = diag

    return json.dumps(result, indent=2, default=str)


@server.tool()
async def resume_backend(
    action: str,
    constraint: str = "",
) -> str:
    """Resume the backend graph after an interrupt or pause.

    The graph must be in 'interrupted' or 'paused' status.

    Args:
        action: One of 'retry', 'skip', 'abort'.
        constraint: Optional constraint text.
    """
    valid_actions = {"retry", "skip", "abort"}
    if action not in valid_actions:
        return json.dumps({
            "error": f"Invalid action: {action}. Must be one of: {sorted(valid_actions)}",
        })

    if _backend.status not in ("interrupted", "paused"):
        # Self-heal: check checkpoint for pending interrupts.  A fresh MCP
        # process can have status "idle" even when the persisted backend
        # thread is interrupted and resumable.
        if _backend.thread_id:
            if _backend.task is None or _backend.task.done():
                try:
                    await _backend.ensure_graph()
                    _config = {"configurable": {"thread_id": _backend.thread_id}}
                    _snap = await _backend.graph.aget_state(_config)
                    if _snap and _snap.tasks:
                        for _t in _snap.tasks:
                            if _t.interrupts:
                                _backend.status = "interrupted"
                                break
                except Exception:
                    pass

        if _backend.status not in ("interrupted", "paused"):
            return json.dumps({
                "error": f"Cannot resume: backend status is '{_backend.status}'",
                "hint": "Backend must be 'interrupted' or 'paused' to resume.",
            })

    await _backend.ensure_graph()

    from langgraph.types import Command

    resume_value = {
        "action": action,
        "constraint": constraint,
    }

    config = {"configurable": {"thread_id": _backend.thread_id}}

    if _backend.status == "paused":
        # Check if checkpoint has a pending interrupt that needs a Command.
        _has_pending_interrupt = False
        try:
            _snap = await _backend.graph.aget_state(config)
            if _snap and _snap.tasks:
                for _t in _snap.tasks:
                    if _t.interrupts:
                        _has_pending_interrupt = True
                        break
        except Exception:
            pass

        if _has_pending_interrupt:
            resume_input = Command(resume=resume_value)
        else:
            resume_input = None
        _backend.task = asyncio.create_task(
            _backend.run_task(resume_input, config)
        )
    else:
        _backend.task = asyncio.create_task(
            _backend.run_task(Command(resume=resume_value), config)
        )

    result = {
        "status": "running",
        "action": action,
        "thread_id": _backend.thread_id,
    }
    if _backend.error_message:
        result["error_message"] = _backend.error_message
    return json.dumps(result)


@server.tool()
async def pause_backend() -> str:
    """Pause the running backend graph.

    Cancels the background task and kills any stuck Claude CLI
    subprocesses.  The graph state is preserved at the last completed
    node boundary.  Call resume_backend() to continue.
    """
    if _backend.status != "running":
        return json.dumps({
            "error": f"Cannot pause: backend status is '{_backend.status}'",
        })

    # Kill any stuck CLI subprocesses first so the task can actually cancel
    from orchestrator.langchain.agents.coresmith_llm import kill_active_cli_processes
    kill_active_cli_processes()

    if _backend.task and not _backend.task.done():
        _backend.task.cancel()
        try:
            await _backend.task
        except asyncio.CancelledError:
            pass

    _backend.status = "paused"

    await _backend.ensure_graph()
    config = {"configurable": {"thread_id": _backend.thread_id}}
    try:
        state_snapshot = await _backend.graph.aget_state(config)
        values = state_snapshot.values if state_snapshot else {}
        current_block = (values.get("current_block") or {}).get("name", "")
        phase = values.get("phase", "")
        attempt = values.get("attempt", 0)
    except Exception:
        current_block = ""
        phase = ""
        attempt = 0

    result = {
        "status": "paused",
        "current_block": current_block,
        "phase": phase,
        "attempt": attempt,
        "thread_id": _backend.thread_id,
        "message": "Backend paused. Call resume_backend() to continue.",
    }
    if _backend.error_message:
        result["error_message"] = _backend.error_message
    return json.dumps(result)


@server.tool()
async def skip_backend_block() -> str:
    """Skip the current block in the backend and advance to the next one.

    Convenience wrapper: if interrupted, resumes with action='skip'.
    """
    if _backend.status == "interrupted" or _backend.status == "paused":
        return await resume_backend(action="skip")
    else:
        return json.dumps({
            "error": f"Cannot skip: backend status is '{_backend.status}'",
            "hint": "Backend must be interrupted or paused.",
        })


@server.tool()
async def run_backend_step(
    step: str,
    block_name: str,
    netlist_path: str = "",
    sdc_path: str = "",
    target_clock_mhz: float = 50.0,
) -> str:
    """Run a single backend EDA step on a block without the full backend graph.

    Runs pnr, drc, or lvs directly on the specified block. Useful for
    quick iteration after editing constraints or netlist on disk.

    Args:
        step: One of 'pnr', 'drc', 'lvs'.
        block_name: Block name (e.g. 'scrambler', 'adder_16bit').
        netlist_path: Path to synthesized netlist (for pnr). Auto-resolved if empty.
        sdc_path: Path to SDC file (for pnr). Auto-resolved if empty.
        target_clock_mhz: Target clock for STA (default 50.0).
    """
    valid_steps = ("pnr", "drc", "lvs")
    if step not in valid_steps:
        return json.dumps({
            "error": f"Invalid step '{step}'",
            "hint": f"Must be one of: {', '.join(valid_steps)}",
        })

    root = Path(_project_root())
    output_dir = str(root / "syn" / "output" / block_name / "pnr")

    from orchestrator.langgraph.event_stream import write_graph_event

    _step_labels = {"pnr": "Run PnR", "drc": "DRC", "lvs": "LVS"}
    node_label = _step_labels[step]

    try:
        if step == "pnr":
            from orchestrator.langgraph.backend_helpers import run_pnr_flow

            # Resolve netlist and SDC
            if not netlist_path:
                candidate = root / "syn" / "output" / block_name / f"{block_name}_netlist.v"
                if candidate.exists():
                    netlist_path = str(candidate)
            if not sdc_path:
                candidate = root / "syn" / "output" / block_name / f"{block_name}.sdc"
                if candidate.exists():
                    sdc_path = str(candidate)

            if not netlist_path or not os.path.isfile(netlist_path):
                return json.dumps({
                    "error": f"Netlist not found for '{block_name}'",
                    "hint": "Provide netlist_path or ensure "
                            f"syn/output/{block_name}/{block_name}_netlist.v exists",
                })

            if not sdc_path or not os.path.isfile(sdc_path):
                # Auto-generate SDC
                sdc_dir = root / "syn" / "output" / block_name
                sdc_dir.mkdir(parents=True, exist_ok=True)
                sdc_path = str(sdc_dir / f"{block_name}.sdc")
                period_ns = 1000.0 / target_clock_mhz
                # Discover the netlist's ACTUAL clock port (a hardcoded `clk`
                # never binds on wb_clk_i-style tops -> CTS finds zero clock
                # nets and STA is meaningless).
                _clk_port = "clk"
                try:
                    from orchestrator.langgraph.pipeline_helpers import (
                        _detect_clock_port,
                    )
                    _clk_port = (_detect_clock_port(
                        Path(netlist_path).read_text(errors="ignore")) or "clk")
                except Exception:
                    _clk_port = "clk"
                Path(sdc_path).write_text(
                    f"create_clock -name clk -period {period_ns} [get_ports {_clk_port}]\n"
                    f"set_input_delay {period_ns * 0.2:.1f} -clock clk [all_inputs]\n"
                    f"set_output_delay {period_ns * 0.2:.1f} -clock clk [all_outputs]\n"
                )

            write_graph_event(str(root), node_label, "graph_node_enter", {
                "block": block_name, "source": "run_backend_step", "graph": "backend",
            })

            result = await asyncio.to_thread(
                run_pnr_flow, block_name, netlist_path, sdc_path, output_dir,
            )

            write_graph_event(str(root), node_label, "graph_node_exit", {
                "block": block_name, "success": result.get("success", False),
                "source": "run_backend_step", "graph": "backend",
            })

            merged = False
            if result.get("success"):
                merged = await _merge_block_into_backend_checkpoint({
                    "name": block_name,
                    "success": True,
                    "gate_count": result.get("gate_count", 0),
                    "chip_area_um2": result.get("chip_area_um2", 0.0),
                    "wns_ns": result.get("wns_ns", 0.0),
                    "pnr_passed": True,
                })
            return json.dumps({
                "step": "pnr", "block_name": block_name,
                "checkpoint_merged": merged, **result,
            }, indent=2, default=str)

        elif step == "drc":
            from orchestrator.langgraph.backend_helpers import run_drc_flow

            routed_def = str(
                root / "syn" / "output" / block_name / "pnr"
                / f"{block_name}_routed.def"
            )
            if not os.path.isfile(routed_def):
                return json.dumps({
                    "error": f"Routed DEF not found: {routed_def}",
                    "hint": "Run pnr step first.",
                })

            write_graph_event(str(root), node_label, "graph_node_enter", {
                "block": block_name, "source": "run_backend_step", "graph": "backend",
            })

            result = await asyncio.to_thread(
                run_drc_flow, block_name, routed_def, output_dir,
            )

            write_graph_event(str(root), node_label, "graph_node_exit", {
                "block": block_name, "clean": result.get("clean", False),
                "source": "run_backend_step", "graph": "backend",
            })

            merged = False
            if result.get("clean"):
                merged = await _merge_block_into_backend_checkpoint({
                    "name": block_name,
                    "success": True,
                    "drc_violations": 0,
                    "drc_clean": True,
                })
            return json.dumps({
                "step": "drc", "block_name": block_name,
                "checkpoint_merged": merged, **result,
            }, indent=2, default=str)

        elif step == "lvs":
            from orchestrator.langgraph.backend_helpers import run_lvs_flow

            spice_path = str(
                root / "syn" / "output" / block_name / "pnr"
                / f"{block_name}.spice"
            )
            pwr_v = str(
                root / "syn" / "output" / block_name / "pnr"
                / f"{block_name}_pwr.v"
            )

            if not os.path.isfile(spice_path):
                return json.dumps({
                    "error": f"SPICE not found: {spice_path}",
                    "hint": "Run drc step first (generates SPICE via Magic).",
                })
            if not os.path.isfile(pwr_v):
                return json.dumps({
                    "error": f"Power Verilog not found: {pwr_v}",
                    "hint": "Run pnr step first (generates power-aware Verilog).",
                })

            write_graph_event(str(root), node_label, "graph_node_enter", {
                "block": block_name, "source": "run_backend_step", "graph": "backend",
            })

            result = await asyncio.to_thread(
                run_lvs_flow, block_name, spice_path, pwr_v, output_dir,
            )

            write_graph_event(str(root), node_label, "graph_node_exit", {
                "block": block_name, "match": result.get("match", False),
                "source": "run_backend_step", "graph": "backend",
            })

            merged = False
            if result.get("match"):
                merged = await _merge_block_into_backend_checkpoint({
                    "name": block_name,
                    "success": True,
                    "lvs_clean": True,
                })
            return json.dumps({
                "step": "lvs", "block_name": block_name,
                "checkpoint_merged": merged, **result,
            }, indent=2, default=str)

    except Exception as e:
        return json.dumps({
            "error": f"run_backend_step failed: {e}",
            "step": step,
            "block_name": block_name,
        })

    return json.dumps({"error": "unreachable"})


# ═══════════════════════════════════════════════════════════════════════════
# TAPEOUT CONTROL TOOLS -- OpenFrame tapeout submission pipeline
#
# Drives the tapeout graph: wrapper generation -> wrapper PnR -> wrapper DRC
# -> wrapper LVS -> native MPW precheck -> tapeout complete.
# ═══════════════════════════════════════════════════════════════════════════


@server.tool()
async def start_tapeout(
    gpio_mapping: str = "",
    target_clock_mhz: float = 50.0,
    max_attempts: int = 2,
) -> str:
    """Start the OpenFrame tapeout pipeline after backend completes.

    Generates the OpenFrame wrapper RTL, runs wrapper-level PnR/DRC/LVS,
    and performs native MPW precheck (no Docker required).

    The backend graph must have completed with at least one passing block.

    Args:
        gpio_mapping: Optional JSON dict mapping block ports to GPIO pads.
                     Auto-generated if empty.
        target_clock_mhz: Target clock frequency (default 50.0).
        max_attempts: Max retry attempts for wrapper PnR (default 2).
    """
    if _tapeout.status == "running":
        return json.dumps({
            "error": "Tapeout already running",
            "thread_id": _tapeout.thread_id,
            "status": _tapeout.status,
        })

    await _tapeout.ensure_graph()

    # Gather completed backend blocks
    root = Path(_project_root())

    # Try to get backend results from the backend lifecycle
    completed_backend_blocks: list[dict] = []
    blocks: list[dict] = []

    if _backend.graph and _backend.thread_id:
        try:
            config = {"configurable": {"thread_id": _backend.thread_id}}
            snap = await _backend.graph.aget_state(config)
            if snap and snap.values:
                completed_backend_blocks = snap.values.get("completed_blocks", [])
                blocks = snap.values.get("block_queue", [])
        except Exception:
            pass

    # Fallback: load block specs from disk
    if not blocks:
        specs_path = root / ".coresmith" / "block_specs.json"
        if specs_path.exists():
            blocks = json.loads(specs_path.read_text())

    passing_blocks = [b for b in completed_backend_blocks if b.get("success")]
    if not passing_blocks:
        return json.dumps({
            "error": "No passing backend blocks found. Run start_backend() first.",
            "completed_blocks": len(completed_backend_blocks),
            "hint": "Backend must complete with at least one passing block.",
        })

    # Parse GPIO mapping
    parsed_gpio = None
    if gpio_mapping:
        try:
            parsed_gpio = json.loads(gpio_mapping)
        except json.JSONDecodeError as e:
            return json.dumps({
                "error": f"Invalid gpio_mapping JSON: {e}",
                "hint": 'Provide a JSON dict, e.g. {"adder_16bit": {"sum": {"start": 2, "width": 16, "direction": "output"}}}',
            })

    await _tapeout.reset_for_new_run()

    graph_config = {
        "configurable": {"thread_id": _tapeout.thread_id},
    }

    initial_state = {
        "project_root": _project_root(),
        "target_clock_mhz": target_clock_mhz,
        "blocks": blocks,
        "completed_backend_blocks": passing_blocks,
        "gpio_mapping": parsed_gpio,
        "phase": "init",
        "attempt": 1,
        "max_attempts": max_attempts,
        "previous_error": "",
        "wrapper_result": None,
        "wrapper_pnr_result": None,
        "wrapper_drc_result": None,
        "wrapper_lvs_result": None,
        "precheck_result": None,
        "submission_result": None,
        "wrapper_rtl_path": "",
        "wrapper_netlist_path": "",
        "wrapper_routed_def": "",
        "wrapper_gds_path": "",
        "wrapper_spice_path": "",
        "submission_dir": "",
        "step_log_paths": {},
        "human_response": None,
        "tapeout_done": False,
    }

    _tapeout.task = asyncio.create_task(
        _tapeout.run_task(initial_state, graph_config)
    )
    await asyncio.sleep(0.1)

    result = {
        "status": _tapeout.status,
        "thread_id": _tapeout.thread_id,
        "blocks": len(blocks),
        "passing_blocks": len(passing_blocks),
        "block_names": [b.get("name", "") for b in passing_blocks],
        "target_clock_mhz": target_clock_mhz,
    }
    if _tapeout.error_message:
        result["error_message"] = _tapeout.error_message
    return json.dumps(result)


@server.tool()
async def get_tapeout_state() -> str:
    """Get the current tapeout graph state.

    Returns status, phase, wrapper/DRC/LVS/precheck results,
    interrupt payload (if interrupted), and submission directory.
    """
    await _tapeout.ensure_graph()

    if not _tapeout.thread_id:
        return json.dumps({"status": "idle", "message": "No tapeout started"})

    config = {"configurable": {"thread_id": _tapeout.thread_id}}

    try:
        state_snapshot = await _tapeout.graph.aget_state(config)
    except Exception as e:
        return json.dumps({
            "status": _tapeout.status,
            "error": f"Failed to read state: {e}",
        })

    if not state_snapshot or not state_snapshot.values:
        return json.dumps({
            "status": _tapeout.status,
            "thread_id": _tapeout.thread_id,
        })

    values = state_snapshot.values

    interrupt_payload = None
    if state_snapshot.tasks:
        for task in state_snapshot.tasks:
            if task.interrupts:
                interrupt_payload = task.interrupts[0].value
                break

    if interrupt_payload and _tapeout.status == "running":
        if _tapeout.task is None or _tapeout.task.done():
            _tapeout.status = "interrupted"

    wrapper_drc = values.get("wrapper_drc_result") or {}
    wrapper_lvs = values.get("wrapper_lvs_result") or {}
    precheck = values.get("precheck_result") or {}

    result = {
        "status": _tapeout.status,
        "thread_id": _tapeout.thread_id,
        "phase": values.get("phase", ""),
        "tapeout_done": values.get("tapeout_done", False),
        "next_nodes": list(state_snapshot.next) if state_snapshot.next else [],
        "previous_error": (values.get("previous_error") or "")[:1000],
        "wrapper_drc_clean": wrapper_drc.get("clean"),
        "wrapper_lvs_match": wrapper_lvs.get("match"),
        "precheck_pass": precheck.get("pass"),
        "precheck_checks": {
            k: v.get("pass") for k, v in precheck.get("checks", {}).items()
        },
        "submission_dir": values.get("submission_dir", ""),
        "wrapper_gds_path": values.get("wrapper_gds_path", ""),
        "interrupt_payload": interrupt_payload,
    }

    if _tapeout.error_message:
        result["error_message"] = _tapeout.error_message

    return json.dumps(result, indent=2, default=str)


@server.tool()
async def resume_tapeout(action: str) -> str:
    """Resume the tapeout graph after an interrupt.

    The graph must be in 'interrupted' status.

    Args:
        action: One of 'retry', 'fix_pnr', 'skip', 'abort'.
            Use 'fix_pnr' to re-run PnR with adjusted parameters read from
            .coresmith/pnr_overrides.json.  The outer agent should write that file
            (via the Write tool) before calling resume with 'fix_pnr'.
    """
    valid_actions = {"retry", "fix_pnr", "skip", "abort"}
    if action not in valid_actions:
        return json.dumps({
            "error": f"Invalid action: {action}. Must be one of: {sorted(valid_actions)}",
        })

    if _tapeout.status not in ("interrupted", "paused"):
        if _tapeout.status == "running" and _tapeout.thread_id:
            if _tapeout.task is None or _tapeout.task.done():
                try:
                    await _tapeout.ensure_graph()
                    _config = {"configurable": {"thread_id": _tapeout.thread_id}}
                    _snap = await _tapeout.graph.aget_state(_config)
                    if _snap and _snap.tasks:
                        for _t in _snap.tasks:
                            if _t.interrupts:
                                _tapeout.status = "interrupted"
                                break
                except Exception:
                    pass

        if _tapeout.status not in ("interrupted", "paused"):
            return json.dumps({
                "error": f"Cannot resume: tapeout status is '{_tapeout.status}'",
                "hint": "Tapeout must be 'interrupted' or 'paused' to resume.",
            })

    await _tapeout.ensure_graph()

    from langgraph.types import Command

    resume_value = {"action": action}
    config = {"configurable": {"thread_id": _tapeout.thread_id}}

    _tapeout.task = asyncio.create_task(
        _tapeout.run_task(Command(resume=resume_value), config)
    )

    return json.dumps({
        "status": "running",
        "action": action,
        "thread_id": _tapeout.thread_id,
    })


# ═══════════════════════════════════════════════════════════════════════════
# PROJECT MANAGEMENT TOOLS -- reset state, inspect on-disk collateral
#
# reset_project()   -- wipe state with scope control (all, architecture, ...)
# get_project_info() -- inventory of on-disk files, sizes, runner statuses
#
# Multi-project workflow:
#   Each MCP server instance is bound to a single CORESMITH_PROJECT_ROOT. To run
#   two projects in parallel, add a second entry in .cursor/mcp.json with
#   a different CORESMITH_PROJECT_ROOT (and optionally a different server name).
#   All .coresmith/ state, generated RTL, sim builds, and synthesis output are
#   scoped to the project root, so they won't collide.
# ═══════════════════════════════════════════════════════════════════════════


async def _stop_runner(runner: GraphLifecycle) -> None:
    """Cancel a running graph task and tear down its checkpointer.

    After this call the runner is back to idle state and the next
    ``ensure_graph()`` call will re-create everything from scratch.
    """
    if runner.task and not runner.task.done():
        runner.task.cancel()
        try:
            await runner.task
        except asyncio.CancelledError:
            pass
    await runner.cleanup()
    runner.status = "idle"
    runner.error_message = ""
    runner.task = None


def _remove_file(path: Path, deleted: list[str], errors: list[str],
                 root: Path) -> None:
    """Delete a single file, recording the result."""
    if not path.exists():
        return
    try:
        path.unlink()
        deleted.append(str(path.relative_to(root)))
    except OSError as exc:
        errors.append(f"{path.relative_to(root)}: {exc}")


def _remove_dir(path: Path, deleted: list[str], errors: list[str],
                root: Path) -> None:
    """Recursively remove a directory, recording the result."""
    if not path.exists():
        return
    try:
        shutil.rmtree(path)
        deleted.append(str(path.relative_to(root)) + "/")
    except OSError as exc:
        errors.append(f"{path.relative_to(root)}/: {exc}")


def _remove_sqlite(db_path: Path, deleted: list[str], errors: list[str],
                   root: Path) -> None:
    """Delete a SQLite DB and its WAL / SHM / journal companion files."""
    for suffix in ("", "-wal", "-shm", "-journal"):
        companion = db_path.parent / (db_path.name + suffix)
        _remove_file(companion, deleted, errors, root)


@server.tool()
async def reset_project(
    scope: str = "all",
    include_generated: bool = False,
) -> str:
    """Reset project state to start fresh.

    Stops any running graphs, closes database connections, and removes
    state files.  Use this when restarting a project from scratch.

    After reset the next start_architecture / start_pipeline /
    start_backend call will create fresh state.

    Args:
        scope: What to reset. Options:
            "all"          -- Everything in .coresmith/ (default).
            "architecture" -- Architecture checkpoint, ERS, block specs,
                              uarch specs, architecture summary.
            "pipeline"     -- Pipeline checkpoint, events, results,
                              frontend summary.
            "backend"      -- Backend checkpoint and summary.
            "benchmarks"   -- Benchmark cache DB and generated benchmark RTL.
            "traces"       -- Telemetry traces DB (truncated, not deleted,
                              so the live OTel exporter stays healthy).
        include_generated: If True, also remove generated RTL (rtl/),
            simulation builds (sim_build/), and synthesis output (syn/).
            Only applies when scope is "all" or "pipeline".
    """
    valid_scopes = {"all", "architecture", "pipeline", "backend",
                    "tapeout", "benchmarks", "traces"}
    if scope not in valid_scopes:
        return json.dumps({
            "error": f"Invalid scope '{scope}'",
            "valid_scopes": sorted(valid_scopes),
        })

    root = Path(_project_root())
    coresmith_dir = root / ".coresmith"
    deleted: list[str] = []
    errors: list[str] = []

    # --- Architecture ---
    if scope in ("all", "architecture"):
        await _stop_runner(_architecture)
        _remove_sqlite(coresmith_dir / "architecture_checkpoint.db",
                        deleted, errors, root)
        for name in ("architecture_state.json",
                      "prd_spec.json",
                      "sad_spec.json",   # .json is legacy; clean up if present
                      "frd_spec.json",   # .json is legacy; clean up if present
                      "ers_spec.json",
                      "block_specs.json"):
            _remove_file(coresmith_dir / name, deleted, errors, root)
        # Clean arch/ directory (markdown docs + uarch specs + dashboard)
        arch_dir = root / ARCH_DOC_DIR
        for name in ("prd_spec.md", "sad_spec.md", "frd_spec.md",
                      "ers_spec.md", "memory_map.md", "clock_tree.md",
                      "summary_architecture.md", "dashboard.html"):
            _remove_file(arch_dir / name, deleted, errors, root)
        _remove_dir(arch_dir / "uarch_specs", deleted, errors, root)

    # --- Pipeline ---
    if scope in ("all", "pipeline"):
        await _stop_runner(_pipeline)
        _remove_sqlite(coresmith_dir / "pipeline_checkpoint.db",
                        deleted, errors, root)
        for name in ("pipeline_events.jsonl", "pipeline_results.json"):
            _remove_file(coresmith_dir / name, deleted, errors, root)
        _remove_file(root / ARCH_DOC_DIR / "summary_frontend.md",
                      deleted, errors, root)

    # --- Backend ---
    if scope in ("all", "backend"):
        await _stop_runner(_backend)
        _remove_sqlite(coresmith_dir / "backend_checkpoint.db",
                        deleted, errors, root)
        _remove_file(root / ARCH_DOC_DIR / "summary_backend.md",
                      deleted, errors, root)

    # --- Tapeout ---
    if scope in ("all", "tapeout"):
        await _stop_runner(_tapeout)
        _remove_sqlite(coresmith_dir / "tapeout_checkpoint.db",
                        deleted, errors, root)
        _remove_dir(root / "openframe_submission",
                     deleted, errors, root)

    # --- Benchmarks ---
    if scope in ("all", "benchmarks"):
        _remove_sqlite(coresmith_dir / "benchmark_cache.db",
                        deleted, errors, root)
        _remove_dir(coresmith_dir / "benchmarks", deleted, errors, root)

    # --- Traces ---
    if scope in ("all", "traces"):
        traces_db = coresmith_dir / "traces.db"
        if traces_db.exists():
            # Truncate the table rather than deleting the file so the
            # live OTel BatchSpanProcessor keeps a valid connection.
            def _truncate_traces():
                import sqlite3
                conn = sqlite3.connect(str(traces_db))
                conn.execute("DELETE FROM spans")
                conn.execute("VACUUM")
                conn.commit()
                conn.close()
            try:
                await asyncio.to_thread(_truncate_traces)
                deleted.append(".coresmith/traces.db (truncated)")
            except Exception:
                _remove_sqlite(traces_db, deleted, errors, root)

    # --- Generated outputs (optional) ---
    if include_generated and scope in ("all", "pipeline"):
        for dirname in ("rtl", "sim_build", "syn"):
            _remove_dir(root / dirname, deleted, errors, root)

    result: dict[str, Any] = {
        "status": "reset_complete",
        "scope": scope,
        "include_generated": include_generated,
        "deleted_count": len(deleted),
        "deleted": deleted,
    }
    if errors:
        result["errors"] = errors
    return json.dumps(result, indent=2)


def _human_size(size_bytes: int) -> str:
    """Format byte count as a human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            if unit == "B":
                return f"{size_bytes} B"
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024  # type: ignore[assignment]
    return f"{size_bytes:.1f} TB"


@server.tool()
async def get_project_info() -> str:
    """Get an overview of the current project's on-disk state.

    Shows the project root, .coresmith/ contents with sizes and timestamps,
    graph runner statuses, and generated output directory sizes.
    Useful before calling reset_project() to see what exists.
    """
    root = Path(_project_root())
    coresmith_dir = root / ".coresmith"

    def _scan_filesystem():
        # --- Collect .coresmith/ file inventory ---
        state_files: dict[str, dict[str, Any]] = {}
        total_state_size = 0
        if coresmith_dir.exists():
            for f in sorted(coresmith_dir.rglob("*")):
                if not f.is_file():
                    continue
                try:
                    stat = f.stat()
                    rel = str(f.relative_to(root))
                    state_files[rel] = {
                        "size": _human_size(stat.st_size),
                        "size_bytes": stat.st_size,
                        "modified": datetime.fromtimestamp(
                            stat.st_mtime
                        ).strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    total_state_size += stat.st_size
                except OSError:
                    pass

        # --- Generated output directories ---
        generated: dict[str, dict[str, Any]] = {}
        total_generated_size = 0
        for dirname in ("rtl", "sim_build", "syn"):
            d = root / dirname
            if not d.exists():
                continue
            file_count = 0
            dir_size = 0
            for fp in d.rglob("*"):
                if fp.is_file():
                    file_count += 1
                    try:
                        dir_size += fp.stat().st_size
                    except OSError:
                        pass
            generated[dirname] = {
                "file_count": file_count,
                "size": _human_size(dir_size),
                "size_bytes": dir_size,
            }
            total_generated_size += dir_size

        return state_files, total_state_size, generated, total_generated_size

    state_files, total_state_size, generated, total_generated_size = (
        await asyncio.to_thread(_scan_filesystem)
    )

    # --- Graph runner statuses ---
    runners: dict[str, dict[str, Any]] = {}
    for label, runner in [
        ("architecture", _architecture),
        ("pipeline", _pipeline),
        ("backend", _backend),
    ]:
        entry: dict[str, Any] = {
            "status": runner.status,
            "thread_id": runner.thread_id or None,
        }
        if runner.error_message:
            entry["error_message"] = runner.error_message
        runners[label] = entry

    return json.dumps({
        "project_root": str(root),
        "state_directory": str(coresmith_dir),
        "state_file_count": len(state_files),
        "state_total_size": _human_size(total_state_size),
        "state_files": state_files,
        "generated_outputs": generated,
        "generated_total_size": _human_size(total_generated_size),
        "graph_runners": runners,
        "reset_hint": "Call reset_project(scope='all') to wipe state and start fresh.",
    }, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# GRAPH INTROSPECTION TOOLS -- inspect LangGraph structure for visualization
# ═══════════════════════════════════════════════════════════════════════════


# Node-to-type and node-to-prompt mappings for each graph module.
# These are static because the graph topology is defined in code.

_ARCH_NODE_META: dict[str, dict[str, Any]] = {
    "Gather Requirements": {
        "type": "agent",
        "prompt_file": "orchestrator/langchain/prompts/prd_spec.md",
        "uses_interrupt": False,
        "display_name": "Gather Requirements",
        "description": "Analyzes input requirements and generates a Product Requirements Document (PRD)",
    },
    "Escalate PRD": {
        "type": "human_review",
        "prompt_file": None,
        "uses_interrupt": True,
        "display_name": "Review PRD",
        "description": "Pauses for human review of the product requirements",
    },
    "System Architecture": {
        "type": "agent",
        "prompt_file": "orchestrator/langchain/prompts/sad_spec.md",
        "uses_interrupt": False,
        "display_name": "Design System Architecture",
        "description": "Generates the System Architecture Document (SAD) from the PRD",
    },
    "Functional Requirements": {
        "type": "agent",
        "prompt_file": "orchestrator/langchain/prompts/frd_spec.md",
        "uses_interrupt": False,
        "display_name": "Define Functional Requirements",
        "description": "Generates the Functional Requirements Document (FRD) from PRD + SAD",
    },
    "Block Diagram": {
        "type": "agent",
        "prompt_file": "orchestrator/langchain/prompts/block_diagram.md",
        "uses_interrupt": False,
        "display_name": "Design Block Diagram",
        "description": "Generates a candidate block diagram from the architecture requirements",
    },
    "Memory Map": {
        "type": "agent",
        "prompt_file": "orchestrator/langchain/prompts/memory_map.md",
        "uses_interrupt": False,
        "display_name": "Design Memory Map",
        "description": "Designs the memory map and address space allocation",
    },
    "Clock Tree": {
        "type": "agent",
        "prompt_file": "orchestrator/langchain/prompts/clock_tree.md",
        "uses_interrupt": False,
        "display_name": "Plan Clock Tree",
        "description": "Plans clock domains and clock tree structure",
    },
    "Register Spec": {
        "type": "agent",
        "prompt_file": "orchestrator/langchain/prompts/register_spec.md",
        "uses_interrupt": False,
        "display_name": "Define Registers",
        "description": "Generates the register specification and CSR map",
    },
    "Constraint Check": {
        "type": "agent",
        "prompt_file": "orchestrator/langchain/prompts/constraint_check.md",
        "uses_interrupt": False,
        "display_name": "Check Constraints",
        "description": "Validates the architecture against physical and timing constraints",
    },
    "Finalize Architecture": {
        "type": "internal",
        "prompt_file": None,
        "uses_interrupt": False,
        "display_name": "Finalize Design",
        "description": "Packages the final architecture deliverables",
    },
    "Create Documentation": {
        "type": "agent",
        "prompt_file": None,
        "uses_interrupt": False,
        "display_name": "Create Documentation",
        "description": "Generates block diagram visualization JSON for the Block Diagram tab",
    },
    "Architecture Complete": {
        "type": "terminal",
        "prompt_file": None,
        "uses_interrupt": False,
        "display_name": "Architecture Complete",
        "description": "Terminal state \u2014 architecture phase is done",
    },
    "Constraint Iteration": {
        "type": "internal",
        "prompt_file": None,
        "uses_interrupt": False,
        "display_name": "Iterate Constraints",
        "description": "Increments the constraint iteration counter",
    },
    "Escalate Diagram": {
        "type": "human_review",
        "prompt_file": None,
        "uses_interrupt": True,
        "display_name": "Review Block Diagram",
        "description": "Pauses for human review of the block diagram",
    },
    "Escalate Constraints": {
        "type": "human_review",
        "prompt_file": None,
        "uses_interrupt": True,
        "display_name": "Review Constraints",
        "description": "Pauses for human review of constraint violations",
    },
    "Escalate Exhausted": {
        "type": "human_review",
        "prompt_file": None,
        "uses_interrupt": True,
        "display_name": "Review Iteration Limit",
        "description": "Max iterations reached \u2014 escalates to human for decision",
    },
    "Final Review": {
        "type": "human_review",
        "prompt_file": None,
        "uses_interrupt": True,
        "display_name": "Approve Architecture",
        "description": "OK2DEV gate \u2014 architect approves or revises architecture before RTL handoff",
    },
    "Abort": {
        "type": "terminal",
        "prompt_file": None,
        "uses_interrupt": False,
        "display_name": "Abort",
        "description": "Terminal state \u2014 architecture was aborted",
    },
}

_PIPELINE_NODE_META: dict[str, dict[str, Any]] = {
    # Orchestrator-level nodes
    "init_tier": {
        "type": "internal",
        "prompt_file": None,
        "uses_interrupt": False,
        "display_name": "Initialize Tier",
        "description": "Computes tier list and logs current tier info",
    },
    "process_block": {
        "type": "subgraph",
        "prompt_file": None,
        "uses_interrupt": True,
        "display_name": "Process Block",
        "description": "Block lifecycle subgraph (runs N in parallel per tier via Send)",
    },
    "advance_tier": {
        "type": "internal",
        "prompt_file": None,
        "uses_interrupt": False,
        "display_name": "Advance Tier",
        "description": "Increments the tier index after all blocks in a tier complete",
    },
    "pipeline_complete": {
        "type": "internal",
        "prompt_file": None,
        "uses_interrupt": False,
        "display_name": "Pipeline Complete",
        "description": "All blocks processed \u2014 proceeds to integration check",
    },
    "integration_review": {
        "type": "agent",
        "prompt_file": None,
        "uses_interrupt": True,
        "display_name": "Review Integration",
        "description": (
            "Integration Agent checks cross-block interface coherence after "
            "each tier completes. Edits uArch specs on disk to fix mismatches."
        ),
    },
    "integration_check": {
        "type": "agent",
        "prompt_file": "orchestrator/langchain/prompts/integration_lead.md",
        "uses_interrupt": True,
        "display_name": "Check Integration",
        "description": (
            "Integration Lead agent analyzes cross-block compatibility, "
            "generates top-level RTL, and lints the integrated design. "
            "Interrupts on mismatches."
        ),
    },
    "integration_dv": {
        "type": "agent",
        "prompt_file": "orchestrator/langchain/prompts/integration_testbench.md",
        "uses_interrupt": True,
        "display_name": "Verify Integration",
        "description": (
            "Lead DV agent generates and runs a cocotb integration testbench "
            "against the top-level integrated design."
        ),
    },
    "validation_dv": {
        "type": "agent",
        "prompt_file": "orchestrator/langchain/prompts/validation_dv.md",
        "uses_interrupt": True,
        "display_name": "Validate ERS/KPIs",
        "description": (
            "Lead Validation DV agent generates and runs a cocotb testbench "
            "that verifies ERS requirements and measurable application KPIs "
            "after smoke/integration DV."
        ),
    },
    # Block subgraph nodes (nested inside process_block)
    "init_block": {
        "type": "internal",
        "prompt_file": None,
        "uses_interrupt": False,
        "display_name": "Initialize Block",
        "description": "Sets up the block and resets per-block state",
    },
    "generate_uarch_spec": {
        "type": "agent",
        "prompt_file": "orchestrator/langchain/prompts/uarch_spec_generator.md",
        "uses_interrupt": False,
        "display_name": "Generate uArch Spec",
        "description": "Generates a micro-architecture specification for the current block",
    },
    "review_uarch_spec": {
        "type": "human_review",
        "prompt_file": None,
        "uses_interrupt": True,
        "display_name": "Review Microarchitecture",
        "description": "Pauses for human review of the micro-architecture spec",
    },
    "generate_rtl": {
        "type": "agent",
        "prompt_file": "orchestrator/langchain/prompts/rtl_generator.md",
        "uses_interrupt": False,
        "display_name": "Generate RTL + Lint",
        "description": "Generates Verilog RTL, then runs Verilator lint with local LLM fix loop",
    },
    "generate_testbench": {
        "type": "agent",
        "prompt_file": "orchestrator/langchain/prompts/testbench_generator.md",
        "uses_interrupt": False,
        "display_name": "Generate Testbench + Simulate",
        "description": "Generates cocotb testbench, runs simulation, fixes TB bugs locally; escalates RTL bugs to diagnose",
    },
    "synthesize": {
        "type": "agent",
        "prompt_file": "orchestrator/langchain/prompts/synth_fixer.md",
        "uses_interrupt": False,
        "display_name": "Run Synthesis",
        "description": "Runs synthesis with local LLM fix loop before escalating to diagnose",
    },
    "diagnose": {
        "type": "agent",
        "prompt_file": "orchestrator/langchain/prompts/debug_agent.md",
        "uses_interrupt": False,
        "display_name": "Diagnose Failure",
        "description": "Debug agent that analyzes failures and proposes fixes",
    },
    "decide": {
        "type": "decide",
        "prompt_file": "orchestrator/langchain/prompts/decide.md",
        "uses_interrupt": False,
        "possible_outcomes": ["retry_rtl", "retry_tb", "retry_synth", "ask_human", "escalate"],
        "display_name": "Decide Next Action",
        "description": "Routes to the next action based on diagnosis results",
    },
    "ask_human": {
        "type": "human_review",
        "prompt_file": None,
        "uses_interrupt": True,
        "display_name": "Ask Human",
        "description": "Escalates to human when automated fixes are exhausted",
    },
    "block_done": {
        "type": "internal",
        "prompt_file": None,
        "uses_interrupt": False,
        "display_name": "Finish Block",
        "description": "Records block result and terminates the block subgraph",
    },
}

_BACKEND_NODE_META: dict[str, dict[str, Any]] = {
    "init_design": {
        "type": "internal",
        "prompt_file": None,
        "uses_interrupt": False,
        "display_name": "Initialize Design",
        "description": "Discovers the integration top-level RTL and all block RTL files",
    },
    "flat_top_synthesis": {
        "type": "agent",
        "prompt_file": "orchestrator/langchain/prompts/backend_synthesis.md",
        "uses_interrupt": False,
        "display_name": "Synthesize Flat Top",
        "description": "LLM adapts the synthesis script for the flat top-level design, then runs synthesis",
    },
    "run_pnr": {
        "type": "agent",
        "prompt_file": "orchestrator/langchain/prompts/backend_pnr.md",
        "uses_interrupt": False,
        "display_name": "Run Place & Route",
        "description": "LLM adapts the PnR TCL (floorplan, placement, CTS, routing, STA), then runs PnR",
    },
    "drc": {
        "type": "agent",
        "prompt_file": "orchestrator/langchain/prompts/backend_drc.md",
        "uses_interrupt": False,
        "display_name": "Run DRC + GDS",
        "description": "LLM adapts the Magic DRC script, then runs DRC, GDS generation, and SPICE extraction",
    },
    "lvs": {
        "type": "agent",
        "prompt_file": "orchestrator/langchain/prompts/backend_lvs.md",
        "uses_interrupt": False,
        "display_name": "Run LVS",
        "description": "LLM analyzes LVS setup and applies pre-processing, then runs Netgen verification",
    },
    "timing_signoff": {
        "type": "agent",
        "prompt_file": "orchestrator/langchain/prompts/backend_timing_signoff.md",
        "uses_interrupt": False,
        "display_name": "Sign Off Timing",
        "description": "LLM analyzes post-route timing results and provides expert sign-off assessment",
    },
    "mpw_precheck": {
        "type": "agent",
        "prompt_file": "orchestrator/langchain/prompts/backend_mpw_precheck.md",
        "uses_interrupt": False,
        "display_name": "Run MPW Precheck",
        "description": "LLM analyzes Efabless MPW precheck results and assesses shuttle submission readiness",
    },
    "diagnose": {
        "type": "agent",
        "prompt_file": "orchestrator/langchain/prompts/tapeout_diagnosis.md",
        "uses_interrupt": False,
        "display_name": "Diagnose Failure",
        "description": "LLM agent that analyzes PnR/DRC/LVS failures and proposes fixes",
    },
    "decide": {
        "type": "decide",
        "prompt_file": None,
        "uses_interrupt": False,
        "possible_outcomes": ["retry_pnr", "ask_human", "escalate"],
        "display_name": "Decide Next Action",
        "description": "Routes to the next action based on diagnosis results",
    },
    "ask_human": {
        "type": "human_review",
        "prompt_file": None,
        "uses_interrupt": True,
        "display_name": "Ask Human",
        "description": "Escalates to human when automated fixes are exhausted",
    },
    "increment_attempt": {
        "type": "internal",
        "prompt_file": None,
        "uses_interrupt": False,
        "display_name": "Increment Attempt",
        "description": "Bumps the retry counter for the current block",
    },
    "advance_block": {
        "type": "internal",
        "prompt_file": None,
        "uses_interrupt": False,
        "display_name": "Record Result",
        "description": "Records block result (DRC/LVS/timing/precheck pass/fail) for downstream reporting",
    },
    "backend_complete": {
        "type": "terminal",
        "prompt_file": None,
        "uses_interrupt": False,
        "display_name": "Backend Complete",
        "description": "Terminal state \u2014 persists results and proceeds to visualization",
    },
    "generate_3d_view": {
        "type": "activity",
        "prompt_file": None,
        "uses_interrupt": False,
        "display_name": "Generate Layout Views",
        "description": "Generates 3D and 2D GDS layout viewers from the routed design",
    },
    "final_report": {
        "type": "agent",
        "prompt_file": "orchestrator/langchain/prompts/chip_finish_template.html",
        "uses_interrupt": False,
        "display_name": "Generate Report",
        "description": "LLM generates a chip-finish HTML dashboard summarizing the full design flow",
    },
}

_GRAPH_MODULES = {
    "architecture": {
        "module": "orchestrator.langgraph.architecture_graph",
        "builder": "build_architecture_graph",
        "node_meta": _ARCH_NODE_META,
    },
    "frontend": {
        "module": "orchestrator.langgraph.pipeline_graph",
        "builder": "build_pipeline_graph",
        "node_meta": _PIPELINE_NODE_META,
    },
    "backend": {
        "module": "orchestrator.langgraph.backend_graph",
        "builder": "build_backend_graph",
        "node_meta": _BACKEND_NODE_META,
    },
}

# Edge label -> visual style mapping
_EDGE_STYLES: dict[str, str] = {
    "PASS": "flow",
    "CLEAN": "flow",
    "MATCH": "flow",
    "MET": "flow",
    "FAIL": "fail",
    "APPROVED": "flow",
    "REVISE": "retry",
    "RETRY": "retry",
    "RETRY RTL": "retry",
    "RETRY TB": "retry",
    "RETRY SYNTH": "retry",
    "RETRY PNR": "retry",
    "SUCCESS": "flow",
    "CONTINUE": "retry",
    "ESCALATE": "fail",
    "EXHAUSTED": "fail",
    "VIOLATED": "fail",
    "ASK HUMAN": "agent",
    "FIX RTL": "retry",
    "SKIP": "fail",
    "ABORT": "fail",
    "ABORT / REVISE": "fail",
    "NEXT BLOCK": "flow",
    "NEXT TIER": "flow",
    "ALL DONE": "flow",
    "FAN OUT": "flow",
    "DV": "flow",
    "DONE": "flow",
    "VIOLATIONS": "fail",
    "QUESTIONS": "flow",
    "PRD_COMPLETE": "flow",
}


def _resolve_prompt(prompt_file: str | None, root: str) -> tuple[str, str]:
    """Read a prompt .md file and return (preview, full_text)."""
    if not prompt_file:
        return ("", "")
    path = Path(root) / prompt_file
    if not path.exists():
        return ("", "")
    text = path.read_text()
    preview = text[:200].replace("\n", " ").strip()
    return (preview, text)


def _get_original_func(runnable: Any) -> Any:
    """Extract the original callable from a LangGraph-wrapped Runnable."""
    # RunnableLambda/RunnableCallable: sync in .func, async in .afunc
    func = getattr(runnable, "func", None)
    if func is not None:
        return func
    afunc = getattr(runnable, "afunc", None)
    if afunc is not None:
        return afunc
    # Fallback: check .bound for RunnableBinding
    bound = getattr(runnable, "bound", None)
    if bound is not None:
        return _get_original_func(bound)
    return None


def _introspect_graph(graph_key: str, root: str) -> dict:
    """Introspect a LangGraph StateGraph and return structured JSON.

    For the frontend graph, inlines subgraph nodes (e.g. the block
    lifecycle inside ``process_block``) so the webview renders the
    full orchestrator + block pipeline as a single unified graph.
    """
    import importlib

    spec = _GRAPH_MODULES[graph_key]
    mod = importlib.import_module(spec["module"])
    builder_fn = getattr(mod, spec["builder"])

    from langgraph.graph import StateGraph as SG

    captured_graphs: list = []
    original_compile = SG.compile

    def capturing_compile(self, **kwargs):
        captured_graphs.append(self)
        return original_compile(self, **kwargs)

    SG.compile = capturing_compile
    try:
        builder_fn(checkpointer=None)
    finally:
        SG.compile = original_compile

    sg = captured_graphs[-1]
    node_meta = spec["node_meta"]

    subgraphs: dict[str, Any] = {}
    if len(captured_graphs) > 1:
        for sub_sg in captured_graphs[:-1]:
            for name in sg.nodes:
                meta = node_meta.get(name, {})
                if meta.get("type") == "subgraph":
                    subgraphs[name] = sub_sg
                    break

    nodes_out = []
    edges_out = []

    for name in sg.nodes:
        if name in subgraphs:
            _inline_subgraph(
                name, subgraphs[name], node_meta, root,
                nodes_out, edges_out, sg,
            )
        else:
            meta = node_meta.get(name, {
                "type": "internal", "prompt_file": None, "uses_interrupt": False,
            })
            func = _get_original_func(sg.nodes[name].runnable)
            func_name = func.__name__ if func else ""
            preview, full_text = _resolve_prompt(meta.get("prompt_file"), root)

            node_info: dict[str, Any] = {
                "id": name,
                "type": meta["type"],
                "label": meta.get("display_name", name),
                "function": func_name,
                "prompt_file": meta.get("prompt_file"),
                "prompt_preview": preview,
                "prompt_full": full_text,
                "uses_interrupt": meta.get("uses_interrupt", False),
                "description": meta.get("description", ""),
                "metadata": {},
            }
            if "possible_outcomes" in meta:
                node_info["possible_outcomes"] = meta["possible_outcomes"]
            nodes_out.append(node_info)

    _collect_edges(sg, edges_out, subgraphs)

    return {"nodes": nodes_out, "edges": edges_out}


def _inline_subgraph(
    parent_name: str,
    sub_sg: Any,
    node_meta: dict,
    root: str,
    nodes_out: list,
    edges_out: list,
    parent_sg: Any,
) -> None:
    """Inline a subgraph's nodes into the parent graph output.

    Subgraph nodes get ``group`` set to ``parent_name`` so the webview
    can render them as a visual cluster.  Edges within the subgraph are
    emitted directly; edges crossing the subgraph boundary (parent ->
    subgraph START, subgraph END -> parent) are rewired to the actual
    entry/exit nodes.
    """
    entry_nodes = _find_start_targets(sub_sg)
    exit_nodes = _find_end_sources(sub_sg)

    for name in sub_sg.nodes:
        meta = node_meta.get(name, {
            "type": "internal", "prompt_file": None, "uses_interrupt": False,
        })
        func = _get_original_func(sub_sg.nodes[name].runnable)
        func_name = func.__name__ if func else ""
        preview, full_text = _resolve_prompt(meta.get("prompt_file"), root)

        node_info: dict[str, Any] = {
            "id": name,
            "type": meta["type"],
            "label": meta.get("display_name", name),
            "function": func_name,
            "prompt_file": meta.get("prompt_file"),
            "prompt_preview": preview,
            "prompt_full": full_text,
            "uses_interrupt": meta.get("uses_interrupt", False),
            "description": meta.get("description", ""),
            "group": parent_name,
            "metadata": {},
        }
        if "possible_outcomes" in meta:
            node_info["possible_outcomes"] = meta["possible_outcomes"]
        nodes_out.append(node_info)

    _collect_edges(sub_sg, edges_out, {})

    _rewire_parent_edges(parent_sg, parent_name, entry_nodes, exit_nodes, edges_out)


def _find_start_targets(sg: Any) -> list[str]:
    """Find nodes that __start__ connects to (entry points)."""
    targets = []
    for source, target in sg.edges:
        if source == "__start__":
            targets.append(target)
    for source, branches in sg.branches.items():
        if source == "__start__":
            for _bn, bs in branches.items():
                if bs.ends:
                    targets.extend(bs.ends.values())
    return targets or ["__start__"]


def _find_end_sources(sg: Any) -> list[str]:
    """Find nodes that connect to __end__ (exit points)."""
    sources = []
    for source, target in sg.edges:
        if target == "__end__":
            sources.append(source)
    for source, branches in sg.branches.items():
        for _bn, bs in branches.items():
            func = _get_original_func(bs.path)
            labels = getattr(func, "__edge_labels__", {}) if func else {}
            if bs.ends:
                for _k, t in bs.ends.items():
                    if t == "__end__":
                        sources.append(source)
                        break
            elif "__end__" in labels:
                sources.append(source)
    return list(dict.fromkeys(sources)) or ["__end__"]


def _rewire_parent_edges(
    parent_sg: Any,
    subgraph_name: str,
    entry_nodes: list[str],
    exit_nodes: list[str],
    edges_out: list[dict],
) -> None:
    """Rewire parent-level edges that pointed to/from the subgraph node.

    - Edges TO subgraph_name get rewritten to point at each entry node.
    - Edges FROM subgraph_name get rewritten to come from each exit node.
    """
    for source, target in parent_sg.edges:
        if target == subgraph_name:
            src = source if source != "__start__" else "START"
            for entry in entry_nodes:
                edges_out.append({
                    "source": src,
                    "target": entry,
                    "type": "normal",
                    "label": None,
                    "style": "flow",
                })
        elif source == subgraph_name:
            tgt = target if target != "__end__" else "END"
            for ex in exit_nodes:
                edges_out.append({
                    "source": ex,
                    "target": tgt,
                    "type": "normal",
                    "label": None,
                    "style": "flow",
                })

    for src, branches in parent_sg.branches.items():
        for _bn, bs in branches.items():
            func = _get_original_func(bs.path)
            labels = getattr(func, "__edge_labels__", {}) if func else {}
            if bs.ends:
                for _k, tgt in bs.ends.items():
                    if tgt == subgraph_name:
                        label = labels.get(tgt)
                        for entry in entry_nodes:
                            edges_out.append({
                                "source": src if src != "__start__" else "START",
                                "target": entry,
                                "type": "conditional",
                                "label": label,
                                "style": _EDGE_STYLES.get(label, "flow") if label else "flow",
                            })
            elif labels:
                for tgt, label in labels.items():
                    if tgt == subgraph_name:
                        for entry in entry_nodes:
                            edges_out.append({
                                "source": src if src != "__start__" else "START",
                                "target": entry,
                                "type": "conditional",
                                "label": label,
                                "style": _EDGE_STYLES.get(label, "flow"),
                            })


def _collect_edges(
    sg: Any,
    edges_out: list[dict],
    subgraphs: dict[str, Any],
) -> None:
    """Collect normal and conditional edges from a StateGraph.

    Edges that touch a subgraph node are skipped here (handled by
    ``_rewire_parent_edges`` instead).
    """
    for source, target in sg.edges:
        if source in subgraphs or target in subgraphs:
            continue
        if source == "__start__" or target == "__start__":
            continue
        edges_out.append({
            "source": source if source != "__start__" else "START",
            "target": target if target != "__end__" else "END",
            "type": "normal",
            "label": None,
            "style": "flow",
        })

    for source, branches in sg.branches.items():
        if source in subgraphs:
            continue
        for _branch_name, branch_spec in branches.items():
            routing_func = _get_original_func(branch_spec.path)
            edge_labels = getattr(routing_func, "__edge_labels__", {}) if routing_func else {}

            if branch_spec.ends:
                for _key, target in branch_spec.ends.items():
                    if target in subgraphs:
                        continue
                    label = edge_labels.get(target)
                    edges_out.append({
                        "source": source if source != "__start__" else "START",
                        "target": target if target != "__end__" else "END",
                        "type": "conditional",
                        "label": label,
                        "style": _EDGE_STYLES.get(label, "flow") if label else "flow",
                    })
            elif edge_labels:
                for target, label in edge_labels.items():
                    if target in subgraphs:
                        continue
                    edges_out.append({
                        "source": source if source != "__start__" else "START",
                        "target": target if target != "__end__" else "END",
                        "type": "conditional",
                        "label": label,
                        "style": _EDGE_STYLES.get(label, "flow"),
                    })



@server.tool()
async def mark_block_passed(
    block_name: str,
    gate_count: int = 0,
    chip_area_um2: float = 0.0,
) -> str:
    """Register a manual verification result in the pipeline checkpoint.

    Injects a success entry for the named block into the pipeline's
    completed_blocks list. Useful when the outer agent has manually
    verified RTL (e.g., via run_step) and wants the pipeline_complete_node
    to see the block as passing.

    Note: the pipeline_complete_node deduplicates by block name (keeps
    last entry), so this overrides any previous failure for the same block.

    Args:
        block_name: Name of the block to mark as passed.
        gate_count: Gate count from synthesis (informational, default 0).
        chip_area_um2: Chip area from synthesis in um^2 (default 0.0).
    """
    if not _pipeline.thread_id or not _pipeline.graph:
        return json.dumps({
            "error": "No pipeline has been run yet.",
            "hint": "Call start_pipeline() first.",
        })

    try:
        await _pipeline.ensure_graph()
        config = {"configurable": {"thread_id": _pipeline.thread_id}}
        snap = await _pipeline.graph.aget_state(config)
        if not snap or not snap.values:
            return json.dumps({
                "error": "Pipeline checkpoint not found.",
                "hint": "The pipeline may not have started or was reset.",
            })

        import time as _time
        entry = {
            "name": block_name,
            "success": True,
            "gate_count": gate_count,
            "chip_area_um2": chip_area_um2,
            "sim_passed": True,
            "synth_success": True,
            "attempts": 0,
            "skipped": False,
            "aborted": False,
            "escalated": False,
            "error": "",
            "marked_manually": True,
            "marked_at": _time.time(),
        }

        await _pipeline.graph.aupdate_state(
            config,
            {"completed_blocks": [entry]},
            as_node="block_done",
        )

        return json.dumps({
            "status": "ok",
            "block_name": block_name,
            "message": f"Block '{block_name}' marked as passed in pipeline checkpoint.",
        })
    except Exception as e:
        return json.dumps({
            "error": f"Failed to mark block: {e}",
            "hint": "Check that the pipeline checkpoint DB exists.",
        })


@server.tool()
async def get_graph_structure(graph_name: str = "frontend") -> str:
    """Introspect a LangGraph state machine and return its structure as JSON.

    Returns node metadata (type, prompt file, function name) and edges
    (normal and conditional with labels) for visualization in the
    workflow editor VS Code extension.

    Args:
        graph_name: Which graph to introspect. Options:
            "architecture", "frontend", or "backend". Default: "frontend".
    """
    if graph_name not in _GRAPH_MODULES:
        available = ", ".join(_GRAPH_MODULES.keys())
        return json.dumps({"error": f"Unknown graph: {graph_name}. Available: {available}"})

    result = _introspect_graph(graph_name, _project_root())
    return json.dumps(result, indent=2)


@server.tool()
async def get_node_prompt(graph_name: str, node_id: str) -> str:
    """Get the full prompt text for a specific graph node.

    Returns the markdown prompt file content for nodes that have
    associated LLM prompts (agents, decide nodes).

    Args:
        graph_name: Which graph ("architecture", "frontend", or "backend").
        node_id: The node ID/name (e.g., "generate_rtl", "diagnose").
    """
    if graph_name not in _GRAPH_MODULES:
        return json.dumps({"error": f"Unknown graph: {graph_name}"})

    node_meta = _GRAPH_MODULES[graph_name]["node_meta"]
    meta = node_meta.get(node_id)
    if not meta:
        available = ", ".join(node_meta.keys())
        return json.dumps({"error": f"Unknown node: {node_id}. Available: {available}"})

    prompt_file = meta.get("prompt_file")
    if not prompt_file:
        return json.dumps({"node_id": node_id, "prompt_file": None, "content": ""})

    path = Path(_project_root()) / prompt_file
    content = path.read_text() if path.exists() else ""
    return json.dumps({
        "node_id": node_id,
        "prompt_file": prompt_file,
        "content": content,
    }, indent=2)


if __name__ == "__main__":
    import asyncio
    import atexit

    async def _shutdown():
        await _architecture.cleanup()
        await _pipeline.cleanup()
        await _backend.cleanup()

    def _sync_shutdown():
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(_shutdown())
            else:
                loop.run_until_complete(_shutdown())
        except Exception:
            pass

    atexit.register(_sync_shutdown)
    server.run(transport="stdio")
