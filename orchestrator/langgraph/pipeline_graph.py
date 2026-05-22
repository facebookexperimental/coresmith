# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
LangGraph StateGraph for the full ASIC pipeline.

Two-level architecture:
  1. **Block Subgraph** (``BlockState``) -- self-contained lifecycle for a
     single block: uarch spec -> RTL (with lint) -> testbench (with sim) ->
     synthesize, with a diagnose/retry loop and human escalation.
  2. **Orchestrator Graph** (``OrchestratorState``) -- iterates through
     tiers and uses ``Send()`` to fan out all blocks within each tier
     for parallel execution.

Block lifecycle (simplified)::

    init -> uarch_spec -> review -> generate_rtl (lint built-in)
         -> generate_testbench (sim + local TB fix loop)
         -> synthesize -> block_done
                 |
              diagnose -> decide -> generate_rtl (direct retry)

Key design decisions:
  - Lint is folded into generate_rtl: run Verilator lint after RTL
    generation, with a local LLM fix loop before escalating.
  - Simulate is folded into generate_testbench: run cocotb sim after
    TB generation, with a local LLM fix loop for testbench bugs.
    Only escalates to diagnose for serious RTL bugs.
  - decide routes directly to generate_rtl (no intermediate
    increment_attempt node).

Tier N+1 does not start until every block in tier N completes.  Interrupts
in any block pause the entire graph (natural LangGraph behaviour).

Within a tier, blocks run in parallel: ``fan_out_tier`` emits one
``Send("process_block", ...)`` per block and LangGraph schedules every
async branch concurrently via ``asyncio.gather``.  Each per-block
``ClaudeLLM.call`` then dispatches the blocking CLI subprocess into the
default thread executor (``loop.run_in_executor`` in ``call``), so two
concurrent blocks do not serialise on the GIL or on a single Popen --
verified empirically: 3 parallel CLI calls finish in 1× wall-time.

Usage::

    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    async with AsyncSqliteSaver.from_conn_string(".socmate/pipeline_checkpoint.db") as cp:
        graph = build_pipeline_graph(checkpointer=cp)
        result = await graph.ainvoke(initial_state, config)
"""

from __future__ import annotations

import asyncio
import json
import operator
import os
import re
from pathlib import Path
from typing import Annotated, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt
from opentelemetry import trace

from orchestrator.langgraph.event_stream import write_graph_event
from orchestrator.utils import smart_truncate
from orchestrator.langgraph.integration_helpers import (
    discover_block_rtl,
    generate_integration_testbench,
    generate_validation_testbench,
    lint_top_level,
    load_architecture_connections,
    parse_verilog_ports,
    run_integration_simulation,
)
from orchestrator.langgraph.pipeline_helpers import (
    PROJECT_ROOT,
    create_golden_model_wrapper,
    diagnose_failure,
    fix_lint_errors,
    fix_synth_errors,
    fix_testbench_errors,
    generate_rtl,
    generate_testbench,
    generate_uarch_spec,
    lint_rtl,
    log,
    run_simulation,
    synthesize_block,
    CYAN,
    GREEN,
    RED,
    YELLOW,
)

_tracer = trace.get_tracer("socmate.langgraph.pipeline_graph")

# Maximum local LLM fix attempts before escalating to diagnose.
# Each agent node (lint, synthesize) tries to self-heal up to this
# many times before giving up and routing to the diagnose lead.
MAX_LOCAL_RETRIES = 2


def _normalize_constraint(text: str) -> str:
    """Normalize constraint text for dedup comparison.

    Fix #13: Lowercases, strips punctuation, collapses whitespace so
    semantically identical constraints worded differently are deduplicated.
    """
    import re as _re
    text = text.lower().strip()
    text = _re.sub(r"\s+", " ", text)
    text = _re.sub(r"[^\w\s]", "", text)
    return text


def _normalize_ws(text: str) -> str:
    """Collapse all whitespace to single spaces."""
    import re as _re
    return _re.sub(r"\s+", " ", text.strip())


def _fuzzy_replace(
    spec: str, original: str, replacement: str
) -> tuple[str, str]:
    """Replace *original* in *spec* with *replacement* using progressively
    looser matching.

    Fix #12: Handles LLM whitespace variations and minor paraphrasing.

    Returns:
        ``(new_spec, method)`` where *method* is ``"exact"``, ``"whitespace"``,
        ``"fuzzy"`` or ``""`` (no match found).
    """
    # 1. Exact match
    if original in spec:
        return spec.replace(original, replacement, 1), "exact"

    # 2. Whitespace-normalised match via sliding window
    norm_orig = _normalize_ws(original)
    lines = spec.split("\n")
    orig_line_count = original.count("\n") + 1
    for i in range(len(lines) - orig_line_count + 1):
        window = "\n".join(lines[i : i + orig_line_count])
        if _normalize_ws(window) == norm_orig:
            return spec.replace(window, replacement, 1), "whitespace"

    # 3. difflib fuzzy match (ratio > 0.85)
    import difflib
    best_ratio = 0.0
    best_start = -1
    best_end = -1
    for window_size in range(orig_line_count - 1, orig_line_count + 2):
        if window_size < 1 or window_size > len(lines):
            continue
        for i in range(len(lines) - window_size + 1):
            window = "\n".join(lines[i : i + window_size])
            ratio = difflib.SequenceMatcher(None, original, window).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = i
                best_end = i + window_size
    if best_ratio > 0.85 and best_start >= 0:
        old_window = "\n".join(lines[best_start:best_end])
        return spec.replace(old_window, replacement, 1), "fuzzy"

    return spec, ""


def _file_is_fresh(path: Path, state: dict) -> bool:
    """Check if *path* was written during the current pipeline run.

    Fix #11: prevents reuse of stale RTL/TB files from previous runs.
    Returns True if the file's mtime is newer than the pipeline start time.
    """
    try:
        run_start = state.get("pipeline_run_start", 0.0)
        if not run_start:
            return True  # no start time recorded -> assume fresh (backwards compat)
        return path.stat().st_mtime >= run_start
    except OSError:
        return False


def _last(a, b):
    """Reducer that keeps the latest value.

    Used for config keys (``project_root``, ``target_clock_mhz``, etc.)
    that are shared between the orchestrator and block subgraph states.
    Without a reducer, parallel ``Send()`` branches would conflict when
    merging their (identical) config values back into the parent state.
    """
    return b


# ---------------------------------------------------------------------------
# State -- Block Subgraph
# ---------------------------------------------------------------------------

class BlockState(TypedDict):
    """Per-block state for the block lifecycle subgraph.

    DISK-FIRST ARCHITECTURE: Graph state carries ONLY routing metadata.
    All content (RTL, testbenches, specs, constraints, diagnosis, error
    logs) lives on disk.  Specialist agents read/write files directly
    via tool use (claude CLI with Read/Write/Edit tools enabled).

    Per-block transient state on disk:
      .socmate/blocks/<block>/constraints.json    -- accumulated constraints
      .socmate/blocks/<block>/diagnosis.json      -- latest debug diagnosis
      .socmate/blocks/<block>/attempt_history.json -- attempt history
      .socmate/blocks/<block>/previous_error.txt  -- latest error context

    Existing artifact locations (unchanged):
      arch/uarch_specs/<block>.md              -- uArch spec
      rtl/<rtl_target>                         -- generated RTL
      tb/cocotb/test_<block>.py                -- testbench
      .socmate/step_logs/<block>/*.log            -- EDA tool logs
    """

    # Config (injected via Send from orchestrator) ──────────────────────────
    project_root: str
    target_clock_mhz: float
    max_attempts: int
    pipeline_run_start: float

    # The block being processed ─────────────────────────────────────────────
    current_block: dict

    # Lifecycle tracking ────────────────────────────────────────────────────
    attempt: int
    phase: str  # "init" | "uarch" | "rtl" | "lint" | "tb" | "sim" | "synth"

    # Routing-only flags (no content -- agents read/write disk directly) ────
    uarch_approved: bool
    lint_clean: bool
    sim_passed: bool
    synth_success: bool
    synth_gate_count: int

    # File paths (set by nodes, consumed by routing and downstream nodes) ───
    rtl_path: str          # path to generated Verilog file
    tb_path: str           # path to generated testbench file

    # Debug routing (set by diagnose_node after reading diagnosis.json) ─────
    debug_action: str      # "retry_rtl" | "retry_tb" | "ask_human" | "escalate" | ...

    # Step log file paths ──────────────────────────────────────────────────
    step_log_paths: Annotated[dict, _last]  # {step: log_path}

    # Testbench control flags ──────────────────────────────────────────────
    preserve_testbench: bool
    force_regen_tb: bool

    # Human interaction ─────────────────────────────────────────────────────
    human_response: Optional[dict]

    # Output (reducer -- flows back to orchestrator) ────────────────────────
    completed_blocks: Annotated[list[dict], operator.add]


# ---------------------------------------------------------------------------
# State -- Orchestrator Graph
# ---------------------------------------------------------------------------

class OrchestratorState(TypedDict):
    """Top-level orchestrator state for tier-based parallel execution.

    The orchestrator iterates through tiers and fans out blocks within
    each tier via ``Send()``.  Results accumulate in ``completed_blocks``.

    Config keys shared with ``BlockState`` use the ``_last`` reducer so
    that parallel ``Send()`` branches can merge without conflict.
    """

    # Config (set once) ─────────────────────────────────────────────────────
    # Reducers on config keys prevent InvalidUpdateError when multiple
    # Send() branches write the same (unchanged) config values back.
    project_root: Annotated[str, _last]
    target_clock_mhz: Annotated[float, _last]
    max_attempts: Annotated[int, _last]
    block_queue: Annotated[list[dict], _last]
    pipeline_run_start: Annotated[float, _last]  # Fix #11: epoch time of pipeline start

    # Tier tracking ─────────────────────────────────────────────────────────
    tier_list: list[int]          # sorted unique tiers, e.g. [1, 2, 3]
    current_tier_index: int

    # Results (accumulated via reducer from all Send branches) ──────────────
    completed_blocks: Annotated[list[dict], operator.add]

    # Integration review decision (set by integration_review_node) ────────
    integration_review_action: Optional[str]

    # Integration check results ────────────────────────────────────────────
    integration_result: Optional[dict]  # set by integration_check node

    # Integration DV results ───────────────────────────────────────────────
    integration_dv_result: Optional[dict]  # set by integration_dv node

    # Validation DV results ────────────────────────────────────────────────
    validation_dv_result: Optional[dict]  # set by validation_dv node

    # Top-level contract audit results ─────────────────────────────────────
    contract_audit_result: Optional[dict]  # set by integration/validation DV failure triage

    # Terminal ──────────────────────────────────────────────────────────────
    pipeline_done: bool
    pipeline_aborted: bool  # set by pipeline_complete_node on abort resume


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _block_name(state: BlockState) -> str:
    block = state.get("current_block")
    if block:
        return block.get("name", "unknown")
    return "unknown"


def _pr(state: BlockState) -> str:
    return state.get("project_root", str(PROJECT_ROOT))


def _callbacks(state: BlockState) -> list:
    """Return an empty callback list (event writing is now internal to ClaudeLLM)."""
    return []


# ---------------------------------------------------------------------------
# Node: init_block  (block subgraph)
# ---------------------------------------------------------------------------

async def init_block_node(state: BlockState) -> dict:
    """Set up the block and reset per-block state.

    In the subgraph model, ``current_block`` is already populated by the
    orchestrator's ``Send()`` call.  This node creates the golden model
    wrapper, logs, and resets lifecycle fields.
    """
    block = state["current_block"]
    block_name = block["name"]

    with _tracer.start_as_current_span(f"Init Block [{block_name}]") as span:
        span.set_attribute("block_name", block_name)
        span.set_attribute("tier", block.get("tier", 0))

    write_graph_event(_pr(state), "Init Block", "graph_node_enter", {
        "block": block_name,
    })

    create_golden_model_wrapper(block_name, block.get("python_source", ""))

    log(f"\n{'='*60}", CYAN)
    log(f"  Block: {block_name} | Tier {block.get('tier', '?')}", CYAN)
    log(f"{'='*60}", CYAN)

    write_graph_event(_pr(state), "Init Block", "graph_node_exit", {
        "block": block_name,
    })

    # Initialize per-block disk state directory
    block_dir = Path(_pr(state)) / ".socmate" / "blocks" / block_name
    block_dir.mkdir(parents=True, exist_ok=True)
    # Reset transient files for a fresh block lifecycle
    for fname in ("constraints.json", "diagnosis.json",
                  "attempt_history.json", "previous_error.txt"):
        fpath = block_dir / fname
        if fname.endswith(".json"):
            fpath.write_text("[]" if "history" in fname or "constraint" in fname else "{}")
        else:
            fpath.write_text("")

    return {
        "attempt": 1,
        "phase": "init",
        "uarch_approved": False,
        "lint_clean": False,
        "sim_passed": False,
        "synth_success": False,
        "synth_gate_count": 0,
        "rtl_path": "",
        "tb_path": "",
        "debug_action": "",
        "human_response": None,
        "step_log_paths": {},
    }


# ---------------------------------------------------------------------------
# Node: generate_uarch_spec
# ---------------------------------------------------------------------------

async def generate_uarch_spec_node(state: BlockState) -> dict:
    """Generate (or revise) a microarchitecture spec for the current block.

    Disk-first: the agent reads all context from disk and writes the spec
    to arch/uarch_specs/<block>.md.  No content flows through state.
    """
    block = state["current_block"]
    block_name = block["name"]

    write_graph_event(_pr(state), "Generate Uarch Spec", "graph_node_enter", {
        "block": block_name,
    })

    with _tracer.start_as_current_span(
        f"Generate Uarch Spec [{block_name}]"
    ) as span:
        span.set_attribute("block_name", block_name)

        # Read feedback from human_response if this is a revision
        feedback = ""
        response = state.get("human_response") or {}
        if response.get("action") == "revise":
            feedback = response.get("feedback", "")

        spec_path = Path(_pr(state)) / "arch" / "uarch_specs" / f"{block_name}.md"
        previous_spec = ""
        if feedback and spec_path.exists():
            previous_spec = spec_path.read_text()
            log(f"  [UARCH] Revising spec for {block_name} with feedback...", YELLOW)
        else:
            log(f"  [UARCH] Generating microarchitecture spec for {block_name}...", YELLOW)

        result = await generate_uarch_spec(
            block, feedback=feedback, previous_spec=previous_spec,
            constraints=[],
            callbacks=_callbacks(state),
        )

        if "error" in result:
            log(f"  [UARCH] FAILED: {result['error']}", RED)
            span.set_attribute("error", result["error"])
        else:
            chars = len(result.get("spec_text", ""))
            log(f"  [UARCH] Generated spec ({chars} chars)", GREEN)
            span.set_attribute("chars", chars)

    write_graph_event(_pr(state), "Generate Uarch Spec", "graph_node_exit", {
        "block": block_name,
    })

    return {
        "uarch_approved": False,
        "phase": "uarch",
    }


# ---------------------------------------------------------------------------
# Node: review_uarch_spec  (INTERRUPT -- human-in-the-loop)
# ---------------------------------------------------------------------------

async def review_uarch_spec_node(state: BlockState) -> dict:
    """Auto-approve the uArch spec at the per-block level.

    Cross-block interface coherence is handled by the Integration Agent
    at the orchestrator level (``integration_review_node``), which runs
    after all blocks in a tier generate their specs and fires a single
    chip-level interrupt for user approval.
    """
    block = state["current_block"]
    block_name = block["name"]

    write_graph_event(_pr(state), "Review Uarch Spec", "graph_node_enter", {
        "block": block_name,
    })

    log(f"  [UARCH] Auto-approve {block_name} "
        f"(chip-level review after tier completes)", GREEN)

    write_graph_event(_pr(state), "Review Uarch Spec", "graph_node_exit", {
        "block": block_name, "action": "approve (deferred to integration review)",
    })

    return {"human_response": {"action": "approve"},
            "uarch_approved": True}


# ---------------------------------------------------------------------------
# Node: generate_rtl  (with lint built-in)
# ---------------------------------------------------------------------------

async def generate_rtl_node(state: BlockState) -> dict:
    """Generate RTL, then run lint with local LLM fix loop.

    Disk-first: the agent reads all context from disk (uarch spec, ERS,
    constraints, previous error, golden model) and writes the Verilog
    to disk.  After generation, runs Verilator lint and attempts local
    LLM fixes before escalating to the diagnose lead.

    Regression guard: if a previous attempt passed simulation, skip
    RTL regeneration entirely and force testbench regeneration instead.
    """
    block = state["current_block"]
    block_name = block["name"]
    attempt = state["attempt"]
    rtl_path_obj = Path(state["project_root"]) / block["rtl_target"]

    write_graph_event(_pr(state), "Generate RTL", "graph_node_enter", {
        "block": block_name, "attempt": attempt,
    })

    best_result_path = (
        Path(_pr(state)) / ".socmate" / "blocks" / block_name / "best_result.json"
    )

    with _tracer.start_as_current_span(
        f"Generate RTL [{block_name}] attempt {attempt}"
    ) as span:
        span.set_attribute("block_name", block_name)
        span.set_attribute("attempt", attempt)

        # --- Regression guard ---
        if attempt > 1 and rtl_path_obj.exists() and best_result_path.exists():
            try:
                best = json.loads(best_result_path.read_text())
                if best.get("sim_passed"):
                    log(f"  [RTL] SKIP regeneration -- attempt {best.get('attempt')} "
                        f"passed sim ({best.get('tests_passed')}/{best.get('tests_total')} tests). "
                        f"Forcing testbench regeneration instead.", YELLOW)
                    span.set_attribute("skipped_regen", True)
                    write_graph_event(_pr(state), "Generate RTL", "graph_node_exit", {
                        "block": block_name, "attempt": attempt,
                        "action": "skip_regen (previous sim passed)",
                    })
                    return {
                        "rtl_path": str(rtl_path_obj),
                        "phase": "rtl",
                        "lint_clean": True,
                        "force_regen_tb": True,
                    }
            except (json.JSONDecodeError, OSError):
                pass

        if attempt == 1 and rtl_path_obj.exists() and _file_is_fresh(rtl_path_obj, state):
            log(f"  [RTL] Using existing (fresh): {block['rtl_target']}", GREEN)
        else:
            log(f"  [RTL] Generating Verilog for {block_name}...", YELLOW)
            rtl_result = await generate_rtl(
                block, attempt,
                callbacks=_callbacks(state),
            )
            if "error" in rtl_result:
                log(f"  [RTL] FAILED: {rtl_result['error']}", RED)
                span.set_attribute("error", rtl_result["error"])

                write_graph_event(_pr(state), "Generate RTL", "graph_node_exit", {
                    "block": block_name, "attempt": attempt, "error": rtl_result["error"],
                })
                block_dir = Path(_pr(state)) / ".socmate" / "blocks" / block_name
                block_dir.mkdir(parents=True, exist_ok=True)
                (block_dir / "previous_error.txt").write_text(
                    f"RTL generation failed: {rtl_result['error']}"
                )
                return {"rtl_path": str(rtl_path_obj), "phase": "lint", "lint_clean": False}
            else:
                log(f"  [RTL] Generated to {block['rtl_target']}", GREEN)

    # --- Lint with local fix loop ---
    rtl_path = str(rtl_path_obj)
    lint_clean = False
    lint_result = None
    existing_logs = dict(state.get("step_log_paths") or {})

    if not rtl_path_obj.exists():
        error_msg = "RTL generation failed (no file on disk)"
        log(f"  [LINT] Skipped -- {error_msg}", RED)
        block_dir = Path(_pr(state)) / ".socmate" / "blocks" / block_name
        block_dir.mkdir(parents=True, exist_ok=True)
        (block_dir / "previous_error.txt").write_text(error_msg)
        write_graph_event(_pr(state), "Generate RTL", "graph_node_exit", {
            "block": block_name, "attempt": attempt, "lint_clean": False,
        })
        return {"rtl_path": rtl_path, "phase": "lint", "lint_clean": False,
                "step_log_paths": existing_logs}

    try:
        rtl_source = rtl_path_obj.read_text()
    except OSError:
        rtl_source = ""

    if rtl_source and not re.search(r"^\s*module\s+\w+", rtl_source, re.MULTILINE):
        corrupt_msg = "RTL file is corrupt (not valid Verilog). Needs regeneration."
        log(f"  [LINT] {corrupt_msg}", RED)
        block_dir = Path(_pr(state)) / ".socmate" / "blocks" / block_name
        block_dir.mkdir(parents=True, exist_ok=True)
        (block_dir / "previous_error.txt").write_text(corrupt_msg)
        write_graph_event(_pr(state), "Generate RTL", "graph_node_exit", {
            "block": block_name, "attempt": attempt, "lint_clean": False,
        })
        return {"rtl_path": rtl_path, "phase": "lint", "lint_clean": False,
                "step_log_paths": existing_logs}

    with _tracer.start_as_current_span(f"Lint [{block_name}]") as lint_span:
        lint_span.set_attribute("block_name", block_name)

        for local_attempt in range(1 + MAX_LOCAL_RETRIES):
            log(f"  [LINT] Running Verilator lint"
                f"{f' (local fix #{local_attempt})' if local_attempt > 0 else ''}...",
                YELLOW)
            lint_result = await asyncio.to_thread(lint_rtl, rtl_path, block_name, attempt)

            if lint_result["clean"]:
                lint_clean = True
                log(f"  [LINT] Clean"
                    f"{f' (after {local_attempt} local fix(es))' if local_attempt > 0 else ''}",
                    GREEN)
                lint_span.set_attribute("clean", True)
                lint_span.set_attribute("local_fixes", local_attempt)
                break

            log("  [LINT] Errors found", RED)
            log(f"    {lint_result.get('errors', '')[:200]}", RED)

            if local_attempt < MAX_LOCAL_RETRIES:
                log(f"  [LINT] Attempting local LLM fix ({local_attempt + 1}/{MAX_LOCAL_RETRIES})...", YELLOW)
                write_graph_event(_pr(state), "Lint Fix", "llm_start", {
                    "block": block_name, "local_attempt": local_attempt + 1,
                })

                fixed_rtl = await fix_lint_errors(
                    block_name, rtl_path, lint_result.get("log_path", ""),
                    callbacks=_callbacks(state),
                )

                write_graph_event(_pr(state), "Lint Fix", "llm_end", {
                    "block": block_name, "local_attempt": local_attempt + 1,
                    "fix_produced": fixed_rtl is not None,
                })

                if fixed_rtl:
                    log("  [LINT] Local fix applied, re-linting...", YELLOW)
                else:
                    log("  [LINT] LLM could not produce a fix, escalating to diagnose", RED)
                    break
            else:
                log("  [LINT] Local retries exhausted, escalating to diagnose", RED)

        lint_span.set_attribute("clean", lint_clean)

    if lint_result and lint_result.get("log_path"):
        existing_logs["lint"] = lint_result["log_path"]

    if not lint_clean and lint_result:
        lint_output = lint_result.get("errors", "") or lint_result.get("warnings", "")
        block_dir = Path(_pr(state)) / ".socmate" / "blocks" / block_name
        block_dir.mkdir(parents=True, exist_ok=True)
        (block_dir / "previous_error.txt").write_text(lint_output[-5000:])

    write_graph_event(_pr(state), "Generate RTL", "graph_node_exit", {
        "block": block_name, "attempt": attempt, "lint_clean": lint_clean,
    })

    return {
        "rtl_path": rtl_path,
        "phase": "rtl" if lint_clean else "lint",
        "lint_clean": lint_clean,
        "step_log_paths": existing_logs,
    }


# ---------------------------------------------------------------------------
# Helpers: testbench bug detection
# ---------------------------------------------------------------------------

_TB_BUG_PATTERNS = [
    # Python framework / import problems
    "AttributeError", "has no attribute",
    "ModuleNotFoundError", "ImportError",
    "SyntaxError", "NameError",
    "TypeError: 'NoneType'",
    "TypeError: int() argument",
    # cocotb timing / API misuse
    "Timer(0)", "Timer( 0",
    "cocotb.result.SimFailure",
    "start_fork",                       # removed in cocotb 2.0
    "units=",                           # cocotb 2.0 wants unit= (singular)
    "unexpected keyword argument 'unit'",
    # Compile-time port/signal mismatches
    "Cannot find signal",
    "No such signal",
    "Verilator: %Error",
]


def _is_likely_testbench_bug(sim_log: str) -> bool:
    """Heuristic: returns True if sim failure looks like a TB framework bug
    (Python errors, missing signals, cocotb API misuse) rather than an RTL
    logic bug. Bare assertion failures against a Python reference model are
    NOT treated as TB bugs — they could be either a wrong reference or a
    real RTL miscompute, and the diagnose agent is far better at telling
    them apart than this string-match heuristic.
    """
    return any(p in sim_log for p in _TB_BUG_PATTERNS)


# ---------------------------------------------------------------------------
# Node: generate_testbench  (with simulation + local TB fix loop)
# ---------------------------------------------------------------------------

async def generate_testbench_node(state: BlockState) -> dict:
    """Generate testbench, run simulation, and fix TB locally on failure.

    After generating (or reusing) the testbench, runs cocotb simulation.
    If simulation fails and the error looks like a testbench bug (import
    error, wrong port names, timing issues), calls an LLM to fix the TB
    and re-runs -- up to MAX_LOCAL_RETRIES times.

    Only escalates to the diagnose lead for failures that appear to be
    RTL bugs (wrong computation, stuck signals, etc.).
    """
    block = state["current_block"]
    block_name = block["name"]
    attempt = state["attempt"]
    tb_path_obj = Path(state["project_root"]) / block["testbench"]
    rtl_path = state.get("rtl_path", "")

    write_graph_event(_pr(state), "Generate Testbench", "graph_node_enter", {
        "block": block_name,
    })

    existing_logs = dict(state.get("step_log_paths") or {})

    # --- Guard: RTL must exist ---
    if not rtl_path or not Path(rtl_path).exists():
        log("  [TB+SIM] Skipped -- RTL file not found", RED)
        write_graph_event(_pr(state), "Generate Testbench", "graph_node_exit", {
            "block": block_name, "sim_passed": False, "reason": "no_rtl",
        })
        return {"tb_path": str(tb_path_obj), "sim_passed": False,
                "phase": "sim", "force_regen_tb": False, "step_log_paths": existing_logs}

    with _tracer.start_as_current_span(
        f"Generate Testbench + Sim [{block_name}]"
    ) as span:
        span.set_attribute("block_name", block_name)

        # --- Step 1: Generate or reuse testbench ---
        force_regen = state.get("force_regen_tb", False)
        if not force_regen and (
            (state.get("preserve_testbench") and tb_path_obj.exists()) or
            (attempt == 1 and tb_path_obj.exists() and _file_is_fresh(tb_path_obj, state))
        ):
            log(f"  [TB] Using existing (fresh): {block['testbench']}", GREEN)
        else:
            log("  [TB] Generating cocotb testbench...", YELLOW)
            try:
                tb_result = await generate_testbench(
                    block,
                    callbacks=_callbacks(state),
                )
            except RuntimeError as exc:
                # The agent now raises if claude CLI failed to write
                # a usable testbench. Fall through to the SIM-skipped
                # path (preserves the existing retry semantics) but
                # log the actual reason instead of a misleading
                # "Generated (N tests)" / "testbench file not found"
                # mirage.
                log(f"  [TB] Generation failed: {exc}", RED)
                tb_result = {"test_count": 0}
            else:
                test_count = tb_result.get("test_count", "?")
                log(f"  [TB] Generated ({test_count} tests)", GREEN)

        tb_path = str(tb_path_obj)

        # --- Step 2: Simulate with local TB fix loop ---
        sim_passed = False
        sim_result = None
        block_dir = Path(_pr(state)) / ".socmate" / "blocks" / block_name
        block_dir.mkdir(parents=True, exist_ok=True)

        for sim_attempt in range(1 + MAX_LOCAL_RETRIES):
            if not tb_path_obj.exists():
                log("  [SIM] Skipped -- testbench file not found", RED)
                break

            log(f"  [SIM] Running cocotb simulation"
                f"{f' (TB fix #{sim_attempt})' if sim_attempt > 0 else ''}...",
                YELLOW)
            sim_result = await asyncio.to_thread(
                run_simulation, block, rtl_path, tb_path, attempt
            )

            if sim_result["passed"]:
                sim_passed = True
                log(f"  [SIM] PASSED"
                    f"{f' (after {sim_attempt} TB fix(es))' if sim_attempt > 0 else ''}",
                    GREEN)
                span.set_attribute("passed", True)
                span.set_attribute("tb_fixes", sim_attempt)

                best_path = block_dir / "best_result.json"
                best_path.write_text(json.dumps({
                    "sim_passed": True,
                    "attempt": attempt,
                    "tests_passed": sim_result.get("tests_passed", 0),
                    "tests_total": sim_result.get("tests_total", 0),
                }))
                break

            sim_log = sim_result.get("log", "")
            log("  [SIM] FAILED", RED)
            for line in sim_log.split("\n")[-5:]:
                if line.strip():
                    log(f"    {line.strip()}", RED)

            is_tb_bug = _is_likely_testbench_bug(sim_log)

            # Only run the local TB-fix loop when the heuristic actually
            # matches. Previously the orchestrator forced a TB-fix LLM call
            # on every first failure (`is_tb_bug or sim_attempt == 0`),
            # which burned ~5 minutes of compute on assertion failures that
            # were genuinely RTL bugs (or, as in mcu3, TB logic bugs that
            # required spec-level reasoning the fix-loop prompt cannot do).
            if sim_attempt < MAX_LOCAL_RETRIES and is_tb_bug:
                log(f"  [SIM] TB framework bug detected -- attempting "
                    f"local fix ({sim_attempt + 1}/{MAX_LOCAL_RETRIES})...", YELLOW)
                write_graph_event(_pr(state), "TB Fix", "llm_start", {
                    "block": block_name, "sim_attempt": sim_attempt + 1,
                    "is_tb_bug": is_tb_bug,
                })

                fixed = await fix_testbench_errors(
                    block_name, rtl_path, tb_path,
                    sim_result.get("log_path", ""),
                    callbacks=_callbacks(state),
                )

                write_graph_event(_pr(state), "TB Fix", "llm_end", {
                    "block": block_name, "sim_attempt": sim_attempt + 1,
                    "fix_produced": fixed is not None,
                })

                if fixed:
                    log("  [SIM] TB fix applied, re-simulating...", YELLOW)
                else:
                    log("  [SIM] LLM could not fix TB, escalating to diagnose", RED)
                    break
            else:
                # Don't pre-classify here -- the diagnose agent does that
                # well (see attempt_history.json / diagnosis.json), and a
                # wrong "Likely RTL bug" line above a real TESTBENCH_BUG
                # diagnosis is misleading.
                if is_tb_bug:
                    log("  [SIM] TB fix retries exhausted, escalating to diagnose", RED)
                else:
                    log("  [SIM] Sim failed -- escalating to diagnose for classification", RED)
                break

        span.set_attribute("sim_passed", sim_passed)

    # Write sim error for diagnose if failed
    if not sim_passed and sim_result:
        sim_log = sim_result.get("log", "")
        (block_dir / "previous_error.txt").write_text(sim_log[-5000:])

    if sim_result and sim_result.get("log_path"):
        existing_logs["simulate"] = sim_result["log_path"]

    # Don't dump multi-KB sim stdout into the event log -- log_path already
    # points to the full file on disk. Keep just enough to grep on (last
    # error line) so the JSONL stays tail-able.
    sim_log_out = sim_result.get("log", "") if sim_result else ""
    last_err = ""
    if sim_log_out and not sim_passed:
        for line in reversed(sim_log_out.splitlines()):
            if line.strip() and ("Error" in line or "FAIL" in line or "Assert" in line):
                last_err = line.strip()[:200]
                break
    write_graph_event(_pr(state), "Generate Testbench", "graph_node_exit", {
        "block": block_name,
        "sim_passed": sim_passed,
        "tb_fixes_attempted": min(sim_attempt + 1, MAX_LOCAL_RETRIES) if sim_result and not sim_passed else 0,
        "last_error": last_err,
        "log_path": sim_result.get("log_path", "") if sim_result else "",
    })

    return {
        "tb_path": tb_path,
        "sim_passed": sim_passed,
        "phase": "sim" if not sim_passed else "tb",
        "force_regen_tb": False,
        "step_log_paths": existing_logs,
    }


# ---------------------------------------------------------------------------
# Node: synthesize  (agent -- local LLM iteration)
# ---------------------------------------------------------------------------

async def synthesize_node(state: BlockState) -> dict:
    """Run Yosys synthesis with local LLM fix loop.

    If synthesis fails, calls an LLM to fix the RTL for synthesizability
    and re-runs -- up to ``MAX_LOCAL_RETRIES`` times.  After local fixes
    are exhausted, the routing function sends failures to the diagnose
    lead for deeper analysis.
    """
    block = state["current_block"]
    block_name = block["name"]

    # Honor SOCMATE_SKIP_SYNTH=1 so hosts with no Sky130 PDK can still
    # complete RTL + sim.  Treat as a no-op success.
    import os as _os
    if _os.environ.get("SOCMATE_SKIP_SYNTH") == "1":
        log(f"  [SYNTH] Skipped (SOCMATE_SKIP_SYNTH=1) for {block_name}", YELLOW)
        return {"synth_success": True, "synth_gate_count": 0, "phase": "synth"}

    rtl_path = state.get("rtl_path", "")
    if not rtl_path or not Path(rtl_path).exists():
        log("  [SYNTH] Skipped -- RTL file not found", RED)
        return {"synth_success": False, "synth_gate_count": 0, "phase": "synth"}

    write_graph_event(_pr(state), "Synthesize", "graph_node_enter", {
        "block": block_name,
    })

    result = None
    synth_ok = False
    gate_count = 0

    with _tracer.start_as_current_span(f"Synthesize [{block_name}]") as span:
        span.set_attribute("block_name", block_name)

        for local_attempt in range(1 + MAX_LOCAL_RETRIES):
            log(f"  [SYNTH] Running Yosys synthesis"
                f"{f' (local fix #{local_attempt})' if local_attempt > 0 else ''}...",
                YELLOW)
            result = await asyncio.to_thread(
                synthesize_block,
                block, rtl_path,
                target_clock_mhz=state.get("target_clock_mhz", 50.0),
                attempt=state["attempt"],
            )

            if result["success"]:
                synth_ok = True
                gate_count = result.get("gate_count", 0)
                area = result.get("chip_area_um2", 0.0)
                area_str = f", {area:,.1f} µm²" if area else ""
                log(f"  [SYNTH] SUCCESS: {gate_count:,} cells{area_str}"
                    f"{f' (after {local_attempt} local fix(es))' if local_attempt > 0 else ''}",
                    GREEN)
                span.set_attribute("success", True)
                span.set_attribute("gate_count", gate_count)
                span.set_attribute("chip_area_um2", area)
                span.set_attribute("local_fixes", local_attempt)
                break

            log("  [SYNTH] FAILED", RED)
            log(f"    {result.get('log', '')[:200]}", RED)

            if local_attempt < MAX_LOCAL_RETRIES:
                log(f"  [SYNTH] Attempting local LLM fix ({local_attempt + 1}/{MAX_LOCAL_RETRIES})...", YELLOW)
                write_graph_event(_pr(state), "Synth Fix", "llm_start", {
                    "block": block_name, "local_attempt": local_attempt + 1,
                })

                fixed_rtl = await fix_synth_errors(
                    block_name, rtl_path, result.get("log_path", ""),
                    callbacks=_callbacks(state),
                )

                write_graph_event(_pr(state), "Synth Fix", "llm_end", {
                    "block": block_name, "local_attempt": local_attempt + 1,
                    "fix_produced": fixed_rtl is not None,
                })

                if fixed_rtl:
                    log("  [SYNTH] Local fix applied, re-synthesizing...", YELLOW)
                else:
                    log("  [SYNTH] LLM could not produce a fix, escalating to diagnose", RED)
                    break
            else:
                log("  [SYNTH] Local retries exhausted, escalating to diagnose", RED)

        span.set_attribute("success", synth_ok)
        span.set_attribute("gate_count", gate_count)

    synth_log = ""
    if result:
        synth_log = result.get("log", "") or result.get("errors", "")

    if not synth_ok and result:
        block_dir = Path(_pr(state)) / ".socmate" / "blocks" / block_name
        block_dir.mkdir(parents=True, exist_ok=True)
        (block_dir / "previous_error.txt").write_text(synth_log[-5000:])

    write_graph_event(_pr(state), "Synthesize", "graph_node_exit", {
        "block": block_name,
        "success": synth_ok,
        "gate_count": gate_count,
        "chip_area_um2": result.get("chip_area_um2", 0.0) if result else 0.0,
        "local_fixes_attempted": min(local_attempt + 1, MAX_LOCAL_RETRIES) if result and not synth_ok else 0,
        "tool_stdout": smart_truncate(synth_log, 2000, "head_tail") if synth_log else "",
        "log_path": result.get("log_path", "") if result else "",
    })

    existing_logs = dict(state.get("step_log_paths") or {})
    if result and result.get("log_path"):
        existing_logs["synthesize"] = result["log_path"]

    return {
        "synth_success": synth_ok,
        "synth_gate_count": gate_count,
        "phase": "synth",
        "step_log_paths": existing_logs,
    }


# ---------------------------------------------------------------------------
# Node: diagnose
# ---------------------------------------------------------------------------

async def diagnose_node(state: BlockState) -> dict:
    """Run DebugAgent to analyze the most recent failure."""
    block = state["current_block"]
    block_name = block["name"]
    phase = state.get("phase", "unknown")

    write_graph_event(_pr(state), "Diagnose Failure", "graph_node_enter", {
        "block": block_name, "phase": phase,
    })

    block_dir = Path(_pr(state)) / ".socmate" / "blocks" / block_name
    block_dir.mkdir(parents=True, exist_ok=True)

    error_file = block_dir / "previous_error.txt"
    error_log = error_file.read_text() if error_file.exists() else "Unknown failure"

    # Short-circuit: detect infrastructure failures (LLM timeout/crash)
    # and skip the debug LLM call which would likely also fail.
    _INFRA_MARKERS = ("[ClaudeLLM error:", "timed out", "exit_code=-9",
                      "circuit breaker open")
    if any(m in error_log for m in _INFRA_MARKERS):
        log("  [DIAGNOSE] Infrastructure failure detected, skipping debug LLM", YELLOW)
        infra_diag = {
            "category": "INFRASTRUCTURE_ERROR",
            "confidence": 0.0,
            "diagnosis": "LLM infrastructure failure (timeout/crash), not a code bug.",
            "suggested_fix": "Retry after backoff.",
            "needs_human": False,
            "is_testbench_bug": False,
            "escalate": False,
            "constraints": [],
            "affected_blocks": [],
        }
        import json as _json
        _ah_path = block_dir / "attempt_history.json"
        history = _json.loads(_ah_path.read_text()) if _ah_path.exists() else []
        history.append({
            "attempt": state["attempt"],
            "phase": phase,
            "error": error_log[:500],
            "category": "INFRASTRUCTURE_ERROR",
        })
        _ah_path.write_text(_json.dumps(history, indent=2))
        (block_dir / "diagnosis.json").write_text(_json.dumps(infra_diag, indent=2))
        write_graph_event(_pr(state), "Diagnose Failure", "graph_node_exit", {
            "block": block_name, "category": "INFRASTRUCTURE_ERROR",
            "confidence": 0.0, "needs_human": False,
        })
        return {"debug_action": "retry_rtl"}

    # Fast-path: detect known testbench bugs via regex to skip expensive
    # opus diagnosis call (~80-100s per invocation).
    import re as _re
    _fast_diag = None
    if phase == "sim":
        if "has no attribute" in error_log or "AttributeError" in error_log:
            _fast_diag = {
                "category": "TESTBENCH_BUG",
                "confidence": 1.0,
                "diagnosis": "Testbench references a DUT port that does not exist.",
                "suggested_fix": "Regenerate testbench with correct port names from RTL.",
                "needs_human": False,
                "is_testbench_bug": True,
                "escalate": False,
                "constraints": [],
                "affected_blocks": [],
            }
        elif "ModuleNotFoundError" in error_log or "ImportError" in error_log:
            _fast_diag = {
                "category": "TESTBENCH_BUG",
                "confidence": 1.0,
                "diagnosis": "Testbench has a missing Python import.",
                "suggested_fix": "Regenerate testbench without external dependencies.",
                "needs_human": False,
                "is_testbench_bug": True,
                "escalate": False,
                "constraints": [],
                "affected_blocks": [],
            }
        elif _re.search(r"cocotb\.result\.TestFail.*Timer\(0\)", error_log):
            _fast_diag = {
                "category": "TESTBENCH_BUG",
                "confidence": 0.95,
                "diagnosis": "Testbench uses Timer(0) causing Verilator delta-cycle race.",
                "suggested_fix": "Regenerate testbench; use FallingEdge/RisingEdge instead of Timer(0).",
                "needs_human": False,
                "is_testbench_bug": True,
                "escalate": False,
                "constraints": [],
                "affected_blocks": [],
            }
    elif phase == "lint":
        if "Module not found" in error_log or "Cannot find file" in error_log:
            _fast_diag = {
                "category": "INFRASTRUCTURE_ERROR",
                "confidence": 1.0,
                "diagnosis": "RTL file missing or module name mismatch.",
                "suggested_fix": "Regenerate RTL.",
                "needs_human": False,
                "is_testbench_bug": False,
                "escalate": False,
                "constraints": [],
                "affected_blocks": [],
            }

    if _fast_diag:
        log(f"  [DIAGNOSE] Fast-path: {_fast_diag['category']} "
            f"(skipped opus LLM call)", GREEN)
        import json as _json
        _ah_path = block_dir / "attempt_history.json"
        history = _json.loads(_ah_path.read_text()) if _ah_path.exists() else []
        history.append({
            "attempt": state["attempt"],
            "phase": phase,
            "error": error_log[:500],
            "category": _fast_diag["category"],
        })
        _ah_path.write_text(_json.dumps(history, indent=2))
        (block_dir / "diagnosis.json").write_text(_json.dumps(_fast_diag, indent=2))
        fast_action = "retry_tb" if _fast_diag.get("is_testbench_bug") else "retry_rtl"
        write_graph_event(_pr(state), "Diagnose Failure", "graph_node_exit", {
            "block": block_name, "category": _fast_diag["category"],
            "confidence": _fast_diag["confidence"], "needs_human": False,
            "fast_path": True,
        })
        return {"debug_action": fast_action}

    with _tracer.start_as_current_span(f"Diagnose [{block_name}]") as span:
        span.set_attribute("block_name", block_name)
        span.set_attribute("failed_phase", phase)

        diag = await diagnose_failure(
            block_name=block_name,
            phase=phase,
            project_root=_pr(state),
            callbacks=_callbacks(state),
        )

        category = diag.get("category", "UNKNOWN")
        span.set_attribute("category", category)
        span.set_attribute("needs_human", diag.get("needs_human", False))

    import json as _json

    (block_dir / "diagnosis.json").write_text(_json.dumps(diag, indent=2))

    _ah_path = block_dir / "attempt_history.json"
    history = _json.loads(_ah_path.read_text()) if _ah_path.exists() else []
    history.append({
        "attempt": state["attempt"],
        "phase": phase,
        "error": error_log[:500],
        "category": category,
    })
    _ah_path.write_text(_json.dumps(history, indent=2))

    action = _route_decision(
        debug_result=diag,
        attempt_history=history,
        attempt=state["attempt"],
        max_attempts=state["max_attempts"],
        phase=phase,
    )

    write_graph_event(_pr(state), "Diagnose Failure", "graph_node_exit", {
        "block": block_name,
        "category": category,
        "confidence": diag.get("confidence", 0),
        "needs_human": diag.get("needs_human", False),
        "suggested_fix": str(diag.get("suggested_fix", ""))[:300],
        "diagnosis_preview": str(diag.get("diagnosis", ""))[:300],
    })

    return {"debug_action": action}


# ---------------------------------------------------------------------------
# Node: decide (deterministic -- no LLM call)
# ---------------------------------------------------------------------------

def _route_decision(debug_result: dict, attempt_history: list[dict],
                    attempt: int, max_attempts: int, phase: str) -> str:
    """Deterministic failure routing based on debug agent output."""
    category = debug_result.get("category", "UNKNOWN")
    confidence = debug_result.get("confidence", 0.5)
    needs_human = debug_result.get("needs_human", False)
    escalate = debug_result.get("escalate", False)

    # High-confidence machine-applicable fix -> auto-retry instead of asking
    # a human.  The LLM debug agent often hedges with needs_human=True even
    # when it has a concrete suggested_fix (e.g. the codec run's
    # transform_select fp16-static-function bug came back at confidence=0.92
    # with a precise `code_snippet`, but `needs_human=True` triggered an
    # ask_human escalation that resolved to instant retry with no new
    # context).  Skip the round-trip: the diagnosis.json is already on disk
    # and the next retry's prompt reads it.
    #   Tunable via SOCMATE_AUTO_FIX_CONFIDENCE (default 0.85).
    suggested_fix = str(debug_result.get("suggested_fix") or "").strip()
    local_fix_possible = debug_result.get("local_fix_possible", True)
    try:
        auto_fix_threshold = float(os.environ.get("SOCMATE_AUTO_FIX_CONFIDENCE", "0.85"))
    except ValueError:
        auto_fix_threshold = 0.85
    if (
        needs_human
        and confidence >= auto_fix_threshold
        and local_fix_possible
        and len(suggested_fix) >= 50
    ):
        needs_human = False  # overridden -- the retry path injects the fix

    # Count how many times each category has occurred
    category_counts: dict[str, int] = {}
    for entry in attempt_history:
        cat = entry.get("category", "UNKNOWN")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    # Rule 0: Infrastructure errors get special handling -- escalate on 2+
    if category == "INFRASTRUCTURE_ERROR":
        if category_counts.get("INFRASTRUCTURE_ERROR", 0) >= 2:
            return "ask_human"
        return "retry_rtl"

    # Rule 1: Same category 3+ times -> stuck in a loop, escalate
    if category_counts.get(category, 0) >= 3:
        return "escalate"

    # Rule 2: Explicit escalation or human-needed flag
    if escalate:
        return "escalate"
    if needs_human:
        return "ask_human"

    # Rule 3: Out of retries
    if attempt >= max_attempts:
        return "escalate"

    # Rule 4: Low confidence -> human should look
    if confidence < 0.3:
        return "ask_human"

    # Rule 5: Testbench bug -> regenerate testbench, not RTL
    if debug_result.get("is_testbench_bug"):
        return "retry_tb"

    # Rule 6: Route based on failed phase
    if phase == "sim":
        return "retry_rtl"  # sim failure -> regenerate RTL
    if phase == "synth":
        return "retry_rtl"  # synth failure -> regenerate RTL
    if phase == "lint":
        return "retry_rtl"  # lint failure -> regenerate RTL

    # Default: retry
    return "retry_rtl"


async def decide_node(state: BlockState) -> dict:
    """Deterministic failure routing with attempt management.

    Reads debug_action from diagnose_node.  For RTL retries, increments
    the attempt counter and checks max_attempts (overriding to escalate
    if exhausted).  For TB retries, sets force_regen_tb.  Handles
    infrastructure backoff.
    """
    block = state["current_block"]
    block_name = block["name"]
    action = state.get("debug_action", "retry_rtl")

    block_title = block_name.replace("_", " ").title()

    with _tracer.start_as_current_span(f"Route Decision [{block_title}]") as span:
        span.set_attribute("block_name", block_name)
        span.set_attribute("attempt", state["attempt"])
        span.set_attribute("decision", action)

        update: dict = {}

        if action == "retry_tb":
            update["force_regen_tb"] = True

        elif action == "retry_rtl":
            new_attempt = state["attempt"] + 1
            if new_attempt > state["max_attempts"]:
                log(f"  [DECIDE] Retries exhausted ({state['max_attempts']} max), escalating", RED)
                action = "escalate"
                update["debug_action"] = "escalate"
            else:
                update["attempt"] = new_attempt
                log(f"  [RETRY] Attempt {new_attempt}/{state['max_attempts']}", YELLOW)

                block_dir = Path(_pr(state)) / ".socmate" / "blocks" / block_name
                diag_path = block_dir / "diagnosis.json"
                if diag_path.exists():
                    try:
                        diag = json.loads(diag_path.read_text())
                        if diag.get("category") == "INFRASTRUCTURE_ERROR":
                            backoff_s = min(30 * (2 ** (new_attempt - 1)), 120)
                            log(f"  [RETRY] Backing off {backoff_s}s after infra failure", YELLOW)
                            await asyncio.sleep(backoff_s)
                    except (json.JSONDecodeError, OSError):
                        pass

        span.set_attribute("final_decision", action)

        write_graph_event(_pr(state), "Route Decision", "graph_node_exit", {
            "block": block_name,
            "decision": action,
            "attempt": state["attempt"],
        })

        return update


# ---------------------------------------------------------------------------
# Node: ask_human  (INTERRUPT)
# ---------------------------------------------------------------------------

async def ask_human_node(state: BlockState) -> dict:
    """Pause the graph and surface failure details to the outer agent.

    One of two nodes that call ``interrupt()`` (the other is
    ``review_uarch_spec_node``).  The outer agent (Claude Code via MCP
    tools) inspects the payload and resumes with
    ``Command(resume={"action": "...", ...})``.
    """
    block = state["current_block"]
    block_name = block["name"]
    state.get("debug_result", {})

    write_graph_event(_pr(state), "Ask Human", "graph_node_enter", {
        "block": block_name, "attempt": state["attempt"],
    })

    log(f"  [HUMAN] Intervention needed for {block_name}", YELLOW)

    import json as _json
    block_dir = Path(_pr(state)) / ".socmate" / "blocks" / block_name
    block_dir.mkdir(parents=True, exist_ok=True)

    diag_path = block_dir / "diagnosis.json"
    diag = _json.loads(diag_path.read_text()) if diag_path.exists() else {}

    ah_path = block_dir / "attempt_history.json"
    attempt_history = _json.loads(ah_path.read_text()) if ah_path.exists() else []

    error_path = block_dir / "previous_error.txt"
    error_text = error_path.read_text() if error_path.exists() else ""

    constr_path = block_dir / "constraints.json"
    constraints = _json.loads(constr_path.read_text()) if constr_path.exists() else []

    category_counts: dict[str, int] = {}
    for entry in attempt_history:
        cat = entry.get("category", "UNKNOWN")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    payload = {
        "type": "human_intervention_needed",
        "block_name": block_name,
        "attempt": state["attempt"],
        "max_attempts": state.get("max_attempts", 5),
        "phase": state.get("phase", ""),
        "error": error_text[:2000],
        "diagnosis": diag.get("diagnosis", ""),
        "category": diag.get("category", ""),
        "suggested_fix": diag.get("suggested_fix", ""),
        "confidence": diag.get("confidence", 0.5),
        "needs_human": diag.get("needs_human", False),
        "human_question": diag.get("human_question", ""),
        "attempt_history": attempt_history[-5:],
        "category_counts": category_counts,
        "constraints": constraints,
        # File paths for outer-agent diagnosis
        "rtl_path": str(
            Path(state["project_root"]) / block.get("rtl_target", "")
        ),
        "uarch_spec_path": str(
            Path(state["project_root"]) / "arch" / "uarch_specs"
            / f"{block_name}.md"
        ),
        # Step log file paths for outer-agent diagnosis
        "step_log_paths": dict(state.get("step_log_paths") or {}),
        # Testbench path
        "testbench_path": str(
            Path(state["project_root"]) / block.get("testbench", "")
        ),
        # Project-root-relative paths for all artifacts
        "relative_paths": {
            "rtl": block.get("rtl_target", ""),
            "testbench": block.get("testbench", ""),
            "uarch_spec": f"arch/uarch_specs/{block_name}.md",
            "ers": ".socmate/ers_spec.json",
        },
        "supported_actions": [
            "retry", "fix_rtl", "fix_tb", "add_constraint", "skip", "abort",
        ],
        # Guidance for the outer agent
        "outer_agent_guidance": (
            "You are the outer-loop diagnostic agent. Do not auto-accept or "
            "blindly retry. Read the OTEL events, step logs, RTL, uarch spec, "
            "testbench, VCD/WaveKit audit, and ERS contract before choosing an "
            "action:\n"
            "1. Classify the root cause and cite concrete evidence.\n"
            "2. If the failure is infrastructure or testbench-only, fix that "
            "shared issue first, then explicitly choose retry or fix_tb.\n"
            "3. If the failure is RTL/spec behavior, edit the relevant RTL or "
            "add a precise constraint, then resume with fix_rtl or "
            "add_constraint.\n"
            "4. If the measurable ERS KPI cannot be verified or the evidence "
            "is inconclusive, escalate to a human with the missing facts.\n"
            "5. Record a rationale with every decision."
        ),
    }

    # Add ERS summary context (non-fatal if missing)
    try:
        import json as _json
        ers_path = Path(state["project_root"]) / ".socmate" / "ers_spec.json"
        if ers_path.exists():
            ers_data = _json.loads(ers_path.read_text(encoding="utf-8"))
            ers_doc = ers_data.get("ers", {})
            ers_info = {
                "summary": ers_doc.get("summary", "")[:2000],
                "bus_protocol": ers_doc.get("dataflow", {}).get("bus_protocol", ""),
                "data_width_bits": ers_doc.get("dataflow", {}).get("data_width_bits", 0),
            }
            payload["ers_summary"] = ers_info
    except Exception:
        pass

    # Add RTL snippet (first 100 lines, non-fatal if missing)
    try:
        rtl_file = Path(state["project_root"]) / block.get("rtl_target", "")
        if rtl_file.exists():
            rtl_lines = rtl_file.read_text(encoding="utf-8").splitlines()[:100]
            payload["rtl_snippet"] = "\n".join(rtl_lines)[:3000]
    except Exception:
        pass

    response = interrupt(payload)

    write_graph_event(_pr(state), "Ask Human", "graph_node_exit", {
        "block": block_name, "action": response.get("action", "unknown"),
    })

    action = response.get("action", "abort")
    updated: dict = {"human_response": response}

    if action == "add_constraint" and response.get("constraint"):
        constraints.append({
            "rule": response["constraint"],
            "source": "human",
            "attempt": state["attempt"],
        })
        constr_path.write_text(_json.dumps(constraints, indent=2))

    if action == "fix_rtl" and response.get("description"):
        constraints.append({
            "rule": f"Outer-agent RTL fix applied: {response['description']}",
            "source": "human",
            "attempt": state["attempt"],
        })
        constr_path.write_text(_json.dumps(constraints, indent=2))

    return updated


# ---------------------------------------------------------------------------
# Node: block_done  (terminal node in the block subgraph)
# ---------------------------------------------------------------------------

async def block_done_node(state: BlockState) -> dict:
    """Record block result.  This is the terminal node of the block subgraph.

    Replaces the old ``advance_block_node`` -- no longer advances a queue
    index; instead the result flows back to the orchestrator via the
    ``completed_blocks`` reducer.
    """
    block = state["current_block"]
    block_name = block["name"]
    attempt = state["attempt"]

    sim_passed = state.get("sim_passed", False)
    synth_success = state.get("synth_success", False)
    gate_count = state.get("synth_gate_count", 0)

    human_resp = state.get("human_response") or {}
    is_skip = human_resp.get("action") == "skip"
    is_abort = human_resp.get("action") == "abort"
    is_escalate = state.get("debug_action") == "escalate"

    all_passed = (
        sim_passed and synth_success
        and not is_skip and not is_abort and not is_escalate
    )

    step_log_paths = dict(state.get("step_log_paths") or {})

    block_dir = Path(_pr(state)) / ".socmate" / "blocks" / block_name
    constr_path = block_dir / "constraints.json"
    import json as _json
    constraints = _json.loads(constr_path.read_text()) if constr_path.exists() else []

    if all_passed:
        result = {
            "name": block_name,
            "success": True,
            "attempts": attempt,
            "gate_count": gate_count,
            "synth_success": True,
            "constraints_learned": len(constraints),
            "step_log_paths": step_log_paths,
        }
        log(f"  [{block_name}] PASSED (attempt {attempt})", GREEN)
    else:
        error_path = block_dir / "previous_error.txt"
        error_text = error_path.read_text()[:500] if error_path.exists() else ""
        result = {
            "name": block_name,
            "success": False,
            "attempts": attempt,
            "error": error_text,
            "constraints_learned": len(constraints),
            "skipped": is_skip,
            "escalated": is_escalate,
            "aborted": is_abort,
            "sim_passed": sim_passed,
            "synth_success": synth_success,
            "step_log_paths": step_log_paths,
        }
        reason = (
            "aborted" if is_abort
            else "skipped" if is_skip
            else "escalated" if is_escalate
            else "failed"
        )
        log(f"  [{block_name}] {reason.upper()} after {attempt} attempts", RED)

    write_graph_event(_pr(state), "Block Done", "graph_node_exit", {
        "block": block_name, "success": result["success"],
    })

    return {
        "completed_blocks": [result],
    }


# ---------------------------------------------------------------------------
# Block-level routing functions
# ---------------------------------------------------------------------------

def route_after_uarch_review(state: BlockState) -> str:
    """Route after uarch spec review: approve -> generate_rtl, revise -> regenerate, skip -> block_done."""
    response = state.get("human_response") or {}
    action = response.get("action", "abort")
    if action == "approve":
        return "generate_rtl"
    elif action == "revise":
        return "generate_uarch_spec"
    elif action == "skip":
        return "block_done"
    return "generate_rtl"


route_after_uarch_review.__edge_labels__ = {
    "generate_rtl": "APPROVED",
    "generate_uarch_spec": "REVISE",
    "block_done": "SKIP",
}


def route_after_rtl(state: BlockState) -> str:
    """Route after RTL generation + lint: CLEAN -> testbench, FAIL -> diagnose."""
    return "generate_testbench" if state.get("lint_clean") else "diagnose"


route_after_rtl.__edge_labels__ = {
    "generate_testbench": "LINT CLEAN",
    "diagnose": "LINT FAIL",
}


def route_after_tb(state: BlockState) -> str:
    """Route after testbench generation + simulation: PASS -> synthesize, FAIL -> diagnose."""
    return "synthesize" if state.get("sim_passed") else "diagnose"


route_after_tb.__edge_labels__ = {
    "synthesize": "SIM PASS",
    "diagnose": "SIM FAIL (RTL bug)",
}


def route_after_synth(state: BlockState) -> str:
    """Route after synthesis: SUCCESS -> block_done, FAIL -> diagnose."""
    return "block_done" if state.get("synth_success") else "diagnose"


route_after_synth.__edge_labels__ = {
    "block_done": "SUCCESS",
    "diagnose": "FAIL",
}


def route_decision(state: BlockState) -> str:
    """Route after decide: directly to generate_rtl, generate_testbench, etc."""
    action = state.get("debug_action", "retry_rtl")
    mapping = {
        "retry_rtl": "generate_rtl",
        "retry_tb": "generate_testbench",
        "retry_synth": "synthesize",
        "ask_human": "ask_human",
        "escalate": "block_done",
    }
    return mapping.get(action, "generate_rtl")


route_decision.__edge_labels__ = {
    "generate_rtl": "RETRY RTL",
    "generate_testbench": "RETRY TB",
    "synthesize": "RETRY SYNTH",
    "ask_human": "ASK HUMAN",
    "block_done": "ESCALATE",
}


def route_after_human(state: BlockState) -> str:
    """Route based on the human's resume action."""
    action = (state.get("human_response") or {}).get("action", "retry")
    mapping = {
        "retry": "generate_rtl",
        "fix_rtl": "generate_rtl",
        "fix_tb": "generate_testbench",
        "add_constraint": "generate_rtl",
        "skip": "block_done",
        "abort": "block_done",
    }
    return mapping.get(action, "generate_rtl")


route_after_human.__edge_labels__ = {
    "generate_rtl": "RETRY / FIX RTL",
    "generate_testbench": "FIX TB",
    "block_done": "SKIP / ABORT",
}


# ---------------------------------------------------------------------------
# Block subgraph builder
# ---------------------------------------------------------------------------

def build_block_subgraph():
    """Build the block lifecycle subgraph (uncompiled StateGraph).

    Contains the full lifecycle for a single block:
      init -> uarch spec -> review
        -> generate_rtl (with lint)
        -> generate_testbench (with sim + local TB fix)
        -> synthesize -> done

    Plus the diagnose/decide/retry failure loop, where decide routes
    directly back to generate_rtl (no intermediate increment node).

    Returns:
        Uncompiled ``StateGraph(BlockState)`` -- the caller compiles it
        (with or without a checkpointer) before adding it as a node.
    """
    graph = StateGraph(BlockState)

    # Nodes (10 -- lint, simulate, increment_attempt are folded in)
    graph.add_node("init_block", init_block_node)
    graph.add_node("generate_uarch_spec", generate_uarch_spec_node)
    graph.add_node("review_uarch_spec", review_uarch_spec_node)
    graph.add_node("generate_rtl", generate_rtl_node)
    graph.add_node("generate_testbench", generate_testbench_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("diagnose", diagnose_node)
    graph.add_node("decide", decide_node)
    graph.add_node("ask_human", ask_human_node)
    graph.add_node("block_done", block_done_node)

    # Happy path
    graph.add_edge(START, "init_block")
    graph.add_edge("init_block", "generate_uarch_spec")
    graph.add_edge("generate_uarch_spec", "review_uarch_spec")
    graph.add_conditional_edges("review_uarch_spec", route_after_uarch_review)
    graph.add_conditional_edges("generate_rtl", route_after_rtl)
    graph.add_conditional_edges("generate_testbench", route_after_tb)
    graph.add_conditional_edges("synthesize", route_after_synth)

    # Failure path
    graph.add_edge("diagnose", "decide")
    graph.add_conditional_edges("decide", route_decision)
    graph.add_conditional_edges("ask_human", route_after_human)

    # Terminal
    graph.add_edge("block_done", END)

    return graph


# ---------------------------------------------------------------------------
# Orchestrator nodes
# ---------------------------------------------------------------------------

async def init_tier_node(state: OrchestratorState) -> dict:
    """Compute the tier list (once) and log the current tier."""
    block_queue = state["block_queue"]
    tier_list = state.get("tier_list") or sorted(
        set(b.get("tier", 1) for b in block_queue)
    )
    current_idx = state.get("current_tier_index", 0)

    tier = tier_list[current_idx]
    tier_blocks = [b for b in block_queue if b.get("tier", 1) == tier]

    pr = state.get("project_root", str(PROJECT_ROOT))
    write_graph_event(pr, "Init Tier", "graph_node_enter", {
        "tier": tier, "tier_index": current_idx,
        "block_count": len(tier_blocks),
        "block_names": [b["name"] for b in tier_blocks],
    })

    log(f"\n{'='*60}", CYAN)
    log(f"  Tier {tier}: {len(tier_blocks)} blocks "
        f"({', '.join(b['name'] for b in tier_blocks)}) | "
        f"Tier {current_idx + 1}/{len(tier_list)}", CYAN)
    log(f"{'='*60}", CYAN)

    write_graph_event(pr, "Init Tier", "graph_node_exit", {
        "tier": tier,
    })

    return {"tier_list": tier_list}


def fan_out_tier(state: OrchestratorState) -> list[Send]:
    """Fan out all blocks in the current tier for parallel execution.

    Returns a list of ``Send("process_block", block_state)`` -- one per
    block.  LangGraph runs all branches concurrently and collects results
    via the ``completed_blocks`` reducer before continuing.
    """
    block_queue = state["block_queue"]
    tier_list = state["tier_list"]
    current_idx = state.get("current_tier_index", 0)
    tier = tier_list[current_idx]

    tier_blocks = [b for b in block_queue if b.get("tier", 1) == tier]

    sends = []
    for block in tier_blocks:
        sends.append(Send("process_block", {
            "project_root": state["project_root"],
            "target_clock_mhz": state["target_clock_mhz"],
            "max_attempts": state["max_attempts"],
            "pipeline_run_start": state.get("pipeline_run_start", 0.0),
            "current_block": block,
            "attempt": 1,
            "phase": "init",
            "constraints": [],
            "attempt_history": [],
            "previous_error": "",
            "uarch_spec": None,
            "uarch_approved": False,
            "uarch_feedback": "",
            "rtl_result": None,
            "lint_result": None,
            "tb_result": None,
            "sim_result": None,
            "synth_result": None,
            "debug_result": None,
            "human_response": None,
            "completed_blocks": [],
            "step_log_paths": {},
        }))

    return sends


fan_out_tier.__edge_labels__ = {
    "process_block": "FAN OUT",
}


async def integration_review_node(state: OrchestratorState) -> dict:
    """Run the Integration Agent to check cross-block interface coherence.

    After all blocks in a tier generate their uArch specs and complete
    RTL/sim/synth, the Integration Agent reads all Section 9 stubs,
    cross-checks against the block diagram, and edits specs on disk to
    fix mismatches.  Then fires ONE chip-level interrupt for user
    approval of the full uArch.
    """
    pr = state.get("project_root", str(PROJECT_ROOT))
    state.get("completed_blocks", [])
    block_queue = state.get("block_queue", [])

    tier_list = state.get("tier_list", [])
    current_idx = state.get("current_tier_index", 0)
    tier = tier_list[current_idx] if current_idx < len(tier_list) else 1
    tier_blocks = [b for b in block_queue if b.get("tier", 1) == tier]
    block_names = [b["name"] for b in tier_blocks]

    write_graph_event(pr, "Integration Review", "graph_node_enter", {
        "tier": tier, "block_names": block_names,
    })

    if not block_names:
        write_graph_event(pr, "Integration Review", "graph_node_exit", {
            "action": "skip (no blocks)",
        })
        return {}

    try:
        from orchestrator.langchain.agents.integration_review_agent import (
            IntegrationReviewAgent,
        )
        from orchestrator.langchain.agents.socmate_llm import DEFAULT_MODEL
        agent = IntegrationReviewAgent(model=DEFAULT_MODEL, temperature=0.1)
        result = await agent.review(
            block_names=block_names,
            project_root=pr,
        )
        review_summary = result.get("summary", "No issues found.")
        issues_found = result.get("issues_found", 0)
        issues_fixed = result.get("issues_fixed", 0)
    except Exception as exc:
        review_summary = f"Integration review failed: {exc}"
        issues_found = 1
        issues_fixed = 0
        review_failed = True
    else:
        review_failed = False

    completed_by_name = {
        b.get("name"): b
        for b in state.get("completed_blocks", [])
        if isinstance(b, dict) and b.get("name")
    }
    failed_tier_blocks = [
        name
        for name in block_names
        if name in completed_by_name and not completed_by_name[name].get("success")
    ]
    if failed_tier_blocks:
        failure_note = (
            "Blocking tier failure: uArch integration review cannot be approved "
            "because these current-tier blocks have not passed their lifecycle: "
            f"{', '.join(failed_tier_blocks)}."
        )
        review_summary = f"{failure_note}\n\n{review_summary}"
        issues_found = int(issues_found or 0) + len(failed_tier_blocks)
        review_failed = True
    if issues_fixed:
        stale_artifact_note = (
            "Blocking uArch edits: integration review modified current-tier "
            "uArch specs after RTL/testbench artifacts were generated. The "
            "affected blocks must be regenerated from uArch before this tier "
            "can be approved; otherwise stale RTL can falsely pass against the "
            "old contract."
        )
        review_summary = f"{stale_artifact_note}\n\n{review_summary}"
        review_failed = True

    log(f"  [INTEGRATION REVIEW] {review_summary[:200]}", GREEN if issues_found == 0 else YELLOW)

    spec_paths = {
        name: str(Path(pr) / "arch" / "uarch_specs" / f"{name}.md")
        for name in block_names
    }

    payload = {
        "type": "uarch_integration_review",
        "tier": tier,
        "block_names": block_names,
        "spec_paths": spec_paths,
        "review_summary": review_summary,
        "issues_found": issues_found,
        "issues_fixed": issues_fixed,
        "review_failed": review_failed,
        "supported_actions": ["approve", "revise", "abort"],
        "outer_agent_guidance": (
            "The Integration Agent has reviewed all uArch specs for "
            "cross-block interface coherence. Present this as a CHIP-LEVEL "
            "review to the user. The user approves or rejects ALL specs at "
            "once. If the Integration Agent fixed mismatches, summarize "
            "what was changed. If the user wants revisions, use "
            "restart_block(from_node='generate_uarch_spec') for affected blocks."
        ),
    }

    response = interrupt(payload)
    action = response.get("action", "abort")
    if action == "approve" and failed_tier_blocks:
        log(
            "  [INTEGRATION REVIEW] Approval rejected because current-tier "
            "blocks failed; treating as revise",
            YELLOW,
        )
        action = "revise"
    if action == "approve" and issues_fixed:
        # NOTE: The integration_review agent edits specs on every run, even
        # cosmetically, so this auto-revise creates an infinite loop:
        # revise -> restart_block -> integration_review edits again -> revise...
        # When the outer agent (human or run_top_headless auto-approve)
        # explicitly approves, trust that decision; the integration_check
        # node at RTL level will catch any real cross-block lint/wiring
        # mismatch and surface it as a normal failure. Setting
        # SOCMATE_STRICT_INTEGRATION_REVIEW=1 restores the old auto-revise.
        import os as _os
        if _os.environ.get("SOCMATE_STRICT_INTEGRATION_REVIEW") == "1":
            log(
                "  [INTEGRATION REVIEW] Approval rejected because uArch specs "
                "were edited after block artifacts were generated; treating as revise "
                "(SOCMATE_STRICT_INTEGRATION_REVIEW=1)",
                YELLOW,
            )
            action = "revise"
        else:
            log(
                "  [INTEGRATION REVIEW] Spec edits were made; honoring explicit "
                "approve (integration_check at RTL level will catch real mismatches). "
                "Export SOCMATE_STRICT_INTEGRATION_REVIEW=1 to force revise.",
                YELLOW,
            )
    if action == "revise" and issues_found == 0 and not review_failed:
        log(
            "  [INTEGRATION REVIEW] Clean review returned revise; "
            "treating as approve",
            YELLOW,
        )
        action = "approve"

    write_graph_event(pr, "Integration Review", "graph_node_exit", {
        "action": action, "issues_found": issues_found,
        "review_failed": review_failed,
    })

    if action == "abort":
        log("  [INTEGRATION REVIEW] Aborted by user/agent", RED)
    elif action == "revise":
        log("  [INTEGRATION REVIEW] Revision requested — "
            "use restart_block to re-generate affected specs", YELLOW)

    return {
        "integration_review_action": action,
        "integration_review_failed": review_failed,
    }


async def advance_tier_node(state: OrchestratorState) -> dict:
    """Advance the tier index after all blocks in the current tier complete."""
    new_idx = state.get("current_tier_index", 0) + 1

    completed = state.get("completed_blocks", [])
    passed = sum(1 for b in completed if b.get("success"))
    total = len(completed)

    pr = state.get("project_root", str(PROJECT_ROOT))
    write_graph_event(pr, "Advance Tier", "graph_node_exit", {
        "new_tier_index": new_idx, "completed_so_far": total,
        "passed_so_far": passed,
    })

    return {"current_tier_index": new_idx}


# ---------------------------------------------------------------------------
# Orchestrator routing functions
# ---------------------------------------------------------------------------

def route_after_integration_review(state: OrchestratorState) -> str:
    """Route based on the user's integration review decision.

    approve → advance_tier (continue normally)
    abort   → END (terminate the pipeline)
    revise  → init_tier (rerun the current tier from the revised uArch specs)
    """
    action = state.get("integration_review_action", "approve")
    if action == "abort":
        return END
    if action == "revise":
        return "init_tier"
    return "advance_tier"


route_after_integration_review.__edge_labels__ = {
    "advance_tier": "APPROVED",
    "init_tier": "REVISE",
    END: "ABORT",
}


def route_next_tier(state: OrchestratorState) -> str:
    """Route after tier advancement: more tiers -> init_tier, done -> pipeline_complete."""
    completed = state.get("completed_blocks", [])
    if any(b.get("aborted") for b in completed):
        return "pipeline_complete"

    tier_list = state.get("tier_list", [])
    current_idx = state.get("current_tier_index", 0)
    if current_idx < len(tier_list):
        return "init_tier"
    return "pipeline_complete"


route_next_tier.__edge_labels__ = {
    "init_tier": "NEXT TIER",
    "pipeline_complete": "ALL DONE",
}


# ---------------------------------------------------------------------------
# Node: pipeline_complete  (orchestrator terminal)
# ---------------------------------------------------------------------------

async def pipeline_complete_node(state: OrchestratorState) -> dict:
    """Mark the pipeline as done, interrupting if any blocks failed.

    All blocks must succeed (sim + synth) before the pipeline can
    proceed to integration check and backend.  If any block failed,
    this node fires a ``pipeline_incomplete`` interrupt so the outer
    agent can diagnose each failure and restart blocks with fixes.
    """
    completed = state.get("completed_blocks", [])
    block_queue = state.get("block_queue", [])

    # Deduplicate completed_blocks by name (keep last entry so that
    # mark_block_passed overrides a previous failure entry)
    seen: dict[str, dict] = {}
    for b in completed:
        name = b.get("name")
        if name:
            seen[name] = b
    completed = list(seen.values())

    expected = len(block_queue) if block_queue else len(completed)
    passed = sum(1 for b in completed if b.get("success"))
    total = len(completed)

    log(f"\n{'#'*60}", CYAN)
    log(f"  PIPELINE COMPLETE: {passed}/{expected} blocks passed", CYAN)
    log(f"{'#'*60}\n", CYAN)

    pr = state.get("project_root", str(PROJECT_ROOT))
    write_graph_event(pr, "Pipeline Complete", "graph_node_exit", {
        "passed": passed, "expected": expected, "total": total,
    })

    # --- Gate: ALL blocks must succeed before proceeding ---
    if passed < expected:
        failed_blocks = []
        for b in completed:
            if not b.get("success"):
                failed_blocks.append({
                    "name": b.get("name", "unknown"),
                    "error": b.get("error", ""),
                    "skipped": b.get("skipped", False),
                    "aborted": b.get("aborted", False),
                    "escalated": b.get("escalated", False),
                    "sim_passed": b.get("sim_passed", False),
                    "synth_success": b.get("synth_success", False),
                    "attempts": b.get("attempts", 0),
                    "step_log_paths": b.get("step_log_paths", {}),
                })

        # Also identify blocks that were expected but never completed
        completed_names = {b.get("name") for b in completed}
        missing_blocks = [
            bq.get("name", "unknown")
            for bq in block_queue
            if bq.get("name") not in completed_names
        ]

        failed_names = [fb["name"] for fb in failed_blocks]

        log(f"  [PIPELINE] {expected - passed} block(s) did not pass: "
            f"{failed_names + missing_blocks}", RED)

        payload = {
            "type": "pipeline_incomplete",
            "passed": passed,
            "expected": expected,
            "failed_blocks": failed_blocks,
            "missing_blocks": missing_blocks,
            "message": (
                f"All blocks must succeed before backend can begin. "
                f"{passed}/{expected} blocks passed. "
                f"Failed: {failed_names}. "
                f"Missing: {missing_blocks}. "
                f"Diagnose each failure (read sim logs, compare RTL against "
                f"testbench expectations, check for timing mismatches) and "
                f"restart blocks with fixes."
            ),
            "supported_actions": ["retry", "abort"],
            "outer_agent_guidance": (
                "As the outer-loop diagnostic agent, you MUST:\n"
                "1. Read step_log_paths for each failed block\n"
                "2. Read the RTL and testbench for each failed block\n"
                "3. Diagnose the root cause of each failure\n"
                "4. Restart each failed block with corrective constraints or RTL fixes\n"
                "5. Do NOT proceed to backend until all blocks pass\n"
                "6. Do NOT use run_step() to bypass this gate -- it does not "
                "register results in the pipeline checkpoint"
            ),
        }

        write_graph_event(pr, "Pipeline Incomplete", "pipeline_gate", {
            "passed": passed, "expected": expected,
            "failed_blocks": failed_names,
            "missing_blocks": missing_blocks,
        })

        resume = interrupt(payload)

        action = resume.get("action") if isinstance(resume, dict) else "abort"
        if action == "retry":
            log(
                f"  [PIPELINE] Retry requested at incomplete gate with "
                f"{passed}/{expected} blocks passed. Stopping graph so the "
                f"outer controller can restart failed blocks; not proceeding "
                f"to integration.",
                YELLOW,
            )
        else:
            log(
                f"  [PIPELINE] Aborted at gate with {passed}/{expected} "
                f"blocks passed; not proceeding to integration.",
                RED,
            )
        return {"pipeline_done": False, "pipeline_aborted": True}

    return {"pipeline_done": True}


# ---------------------------------------------------------------------------
# Node: integration_check  (orchestrator -- verifies cross-block wiring)
# ---------------------------------------------------------------------------

async def integration_check_node(state: OrchestratorState) -> dict:
    """Run the Integration Lead agent to check compatibility and generate top-level RTL.

    After all blocks complete, this node:
    1. Loads architecture connections (block diagram)
    2. Discovers and reads all completed block RTL sources
    3. Calls the IntegrationLeadAgent to analyze compatibility and
       generate the top-level integration module
    4. Writes the generated Verilog to disk
    5. Lints the integrated design

    If errors are found, fires an interrupt with structured mismatch data
    so the outer agent can diagnose and fix.
    """
    import asyncio
    from orchestrator.langchain.agents.integration_lead import IntegrationLeadAgent

    pr = state.get("project_root", str(PROJECT_ROOT))
    completed = state.get("completed_blocks", [])
    passed_blocks = [b for b in completed if b.get("success")]
    block_queue = state.get("block_queue", [])
    expected_blocks = len(block_queue) if block_queue else len(completed)

    write_graph_event(pr, "Integration Check", "graph_node_enter", {
        "total_blocks": len(completed),
        "passed_blocks": len(passed_blocks),
        "expected_blocks": expected_blocks,
    })

    with _tracer.start_as_current_span("Integration Check") as span:
        span.set_attribute("total_blocks", len(completed))
        span.set_attribute("passed_blocks", len(passed_blocks))
        span.set_attribute("expected_blocks", expected_blocks)

        if expected_blocks and len(passed_blocks) < expected_blocks:
            failed_names = [
                b.get("name", "unknown") for b in completed
                if not b.get("success")
            ]
            completed_names = {b.get("name") for b in completed}
            missing_names = [
                b.get("name", "unknown") for b in block_queue
                if b.get("name") not in completed_names
            ]
            log(
                "  [INTEGRATION] Refusing partial integration: "
                f"{len(passed_blocks)}/{expected_blocks} blocks passed; "
                f"failed={failed_names}, missing={missing_names}",
                RED,
            )
            result = {
                "aborted": True,
                "skipped": True,
                "reason": (
                    f"Refusing partial integration: "
                    f"{len(passed_blocks)}/{expected_blocks} blocks passed; "
                    f"failed={failed_names}, missing={missing_names}"
                ),
                "error": "partial_block_set",
                "error_count": max(1, expected_blocks - len(passed_blocks)),
                "passed_blocks": len(passed_blocks),
                "expected_blocks": expected_blocks,
                "failed_blocks": failed_names,
                "missing_blocks": missing_names,
            }
            write_graph_event(pr, "Integration Check", "graph_node_exit", result)
            return {"integration_result": result}

        connections, design_name = await asyncio.to_thread(
            load_architecture_connections, pr
        )

        if not connections and len(passed_blocks) < 1:
            log("  [INTEGRATION] No architecture connections found -- "
                "skipping integration check", YELLOW)
            write_graph_event(pr, "Integration Check", "graph_node_exit", {
                "skipped": True,
                "reason": "no_connections",
            })
            return {"integration_result": {
                "skipped": True,
                "reason": "No architecture connections found",
            }}

        log(f"  [INTEGRATION] Found {len(connections)} connections, "
            f"design: {design_name}", CYAN)
        span.set_attribute("connection_count", len(connections))

        rtl_paths = await asyncio.to_thread(
            discover_block_rtl, pr, passed_blocks
        )

        modules = {}
        block_rtl_sources: dict[str, str] = {}
        for block_name, rtl_path in rtl_paths.items():
            mod = await asyncio.to_thread(parse_verilog_ports, rtl_path)
            if mod.name:
                modules[block_name] = mod
                try:
                    block_rtl_sources[block_name] = Path(rtl_path).read_text(
                        encoding="utf-8", errors="replace"
                    )
                except OSError:
                    block_rtl_sources[block_name] = ""
                log(f"  [INTEGRATION] Parsed {block_name}: "
                    f"{len(mod.ports)} ports", GREEN)
            else:
                log(f"  [INTEGRATION] Failed to parse {block_name} "
                    f"at {rtl_path}", RED)

        span.set_attribute("parsed_blocks", len(modules))

        if not modules:
            log("  [INTEGRATION] No block RTL could be parsed", RED)
            write_graph_event(pr, "Integration Check", "graph_node_exit", {
                "error": "no_rtl_parsed",
            })
            return {"integration_result": {
                "skipped": True,
                "reason": "No block RTL could be parsed",
            }}

        block_port_summaries = []
        for name, mod in sorted(modules.items()):
            block_port_summaries.append({
                "name": name,
                "port_count": len(mod.ports),
                "ports": [p.to_dict() for p in mod.ports],
            })

        prd_summary = ""
        for prd_name in ("prd_spec.json", "ers_spec.json"):
            prd_path = Path(pr) / ".socmate" / prd_name
            if prd_path.exists():
                try:
                    prd_data = json.loads(prd_path.read_text(encoding="utf-8"))
                    doc = prd_data.get("prd", prd_data.get("ers", {}))
                    prd_summary = doc.get("summary", "")
                    if doc.get("speed_and_feeds"):
                        sf = doc["speed_and_feeds"]
                        prd_summary += (
                            f"\nTarget clock: {sf.get('target_clock_mhz', '?')} MHz"
                        )
                    if doc.get("dataflow"):
                        df = doc["dataflow"]
                        prd_summary += (
                            f"\nBus protocol: {df.get('bus_protocol', '?')}"
                            f", Data width: {df.get('data_width_bits', '?')} bits"
                        )
                except (OSError, json.JSONDecodeError, KeyError):
                    pass
                break

        rtl_dir = Path(pr) / "rtl" / "integration"
        rtl_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', design_name).lower()
        if not safe_name or safe_name[0].isdigit():
            safe_name = f"top_{safe_name}"
        output_path = str(rtl_dir / f"{safe_name}.v")

        # Single-block designs: generate a passthrough wrapper that
        # instantiates the block and wires all ports to the top level.
        # This ensures the backend always has an integration top-level
        # module regardless of block count.
        if len(modules) == 1:
            solo_name, solo_mod = next(iter(modules.items()))
            top_name = f"{safe_name}_top" if not safe_name.endswith("_top") else safe_name
            lines = [f"module {top_name} ("]
            port_decls = []
            for p in solo_mod.ports:
                width_str = f"[{p.msb}:{p.lsb}] " if p.width > 1 else ""
                port_decls.append(f"    {p.direction} wire {width_str}{p.name}")
            lines.append(",\n".join(port_decls))
            lines.append(");")
            lines.append("")
            inst_conns = [f"        .{p.name}({p.name})" for p in solo_mod.ports]
            lines.append(f"    {solo_mod.name} u_{solo_name} (")
            lines.append(",\n".join(inst_conns))
            lines.append("    );")
            lines.append("")
            lines.append("endmodule")
            wrapper_src = "\n".join(lines) + "\n"
            Path(output_path).write_text(wrapper_src, encoding="utf-8")

            log(f"  [INTEGRATION] Single-block design: generated wrapper "
                f"{top_name} for {solo_name}", GREEN)

            solo_rtl_path = list(rtl_paths.values())[0]
            lint_result = await asyncio.to_thread(
                lint_top_level, output_path, [solo_rtl_path], top_name
            )
            lint_clean = lint_result.get("clean", False)
            log(f"  [INTEGRATION] Lint: {'CLEAN' if lint_clean else 'ERRORS'}",
                GREEN if lint_clean else RED)

            integration_result = {
                "design_name": design_name,
                "top_module": top_name,
                "top_rtl_path": output_path,
                "block_count": 1,
                "wire_count": len(solo_mod.ports),
                "skipped_connections": [],
                "mismatches": [],
                "error_count": 0,
                "warning_count": 0,
                "lint_clean": lint_clean,
                "lint_errors": lint_result.get("errors", ""),
                "block_rtl_paths": rtl_paths,
                "single_block_wrapper": True,
            }

            write_graph_event(pr, "Integration Check", "graph_node_exit", {
                "success": True,
                "top_module": top_name,
                "block_count": 1,
                "single_block_wrapper": True,
            })
            return {"integration_result": integration_result}

        log("  [INTEGRATION] Calling Integration Lead agent...", YELLOW)
        agent = IntegrationLeadAgent()
        try:
            agent_result = await agent.integrate(
                design_name=design_name,
                block_rtl_sources=block_rtl_sources,
                block_port_summaries=block_port_summaries,
                connections=connections,
                prd_summary=prd_summary,
                output_path=output_path,
            )
        except Exception as e:
            log(f"  [INTEGRATION] Agent failed: {e}", RED)
            write_graph_event(pr, "Integration Check", "graph_node_exit", {
                "error": str(e), "phase": "agent_call",
            })
            return {"integration_result": {
                "skipped": True,
                "reason": f"Integration Lead agent failed: {e}",
            }}

        if agent_result.get("parse_error"):
            log("  [INTEGRATION] Agent returned unparseable response", RED)
            write_graph_event(pr, "Integration Check", "graph_node_exit", {
                "error": "parse_error",
            })
            return {"integration_result": {
                "skipped": True,
                "reason": "Integration Lead agent returned unparseable response",
                "notes": agent_result.get("notes", ""),
            }}

        mismatches = agent_result.get("mismatches", [])
        module_name = agent_result.get("module_name", design_name)
        top_rtl_path = agent_result.get("rtl_path", output_path)

        log(f"  [INTEGRATION] Agent generated {module_name}: "
            f"{len(modules)} blocks, "
            f"{agent_result.get('wire_count', 0)} wires", GREEN)
        span.set_attribute("top_module", module_name)

        block_rtl_list = list(rtl_paths.values())
        lint_result = await asyncio.to_thread(
            lint_top_level, top_rtl_path, block_rtl_list,
            design_name
        )

        lint_clean = lint_result.get("clean", False)
        log(f"  [INTEGRATION] Lint: {'CLEAN' if lint_clean else 'ERRORS'}",
            GREEN if lint_clean else RED)
        span.set_attribute("lint_clean", lint_clean)

        errors = [m for m in mismatches if m.get("severity") == "error"]
        warnings = [m for m in mismatches if m.get("severity") == "warning"]

        integration_result = {
            "design_name": design_name,
            "top_module": module_name,
            "top_rtl_path": top_rtl_path,
            "block_count": len(modules),
            "wire_count": agent_result.get("wire_count", 0),
            "skipped_connections": agent_result.get("skipped_connections", []),
            "mismatches": mismatches,
            "error_count": len(errors),
            "warning_count": len(warnings),
            "lint_clean": lint_clean,
            "lint_errors": lint_result.get("errors", ""),
            "lint_log_path": lint_result.get("log_path", ""),
            "block_rtl_paths": rtl_paths,
            "agent_notes": agent_result.get("notes", ""),
            "parsed_modules": {
                name: {
                    "port_count": len(mod.ports),
                    "inputs": len(mod.inputs()),
                    "outputs": len(mod.outputs()),
                }
                for name, mod in modules.items()
            },
        }

        has_issues = len(errors) > 0 or not lint_clean
        if has_issues:
            log("  [INTEGRATION] Issues found -- interrupting for review", YELLOW)

            payload = {
                "type": "integration_failure",
                "design_name": design_name,
                "top_rtl_path": top_rtl_path,
                "block_count": len(modules),
                "error_count": len(errors),
                "warning_count": len(warnings),
                "lint_clean": lint_clean,
                "mismatches": mismatches,
                "lint_errors": lint_result.get("errors", "")[:3000],
                "lint_log_path": lint_result.get("log_path", ""),
                "block_rtl_paths": rtl_paths,
                "skipped_connections": agent_result.get("skipped_connections", []),
                "supported_actions": [
                    "retry",
                    "fix_rtl",
                    "skip",
                    "abort",
                ],
                "outer_agent_guidance": (
                    "Integration Lead agent found issues. As the outer-loop "
                    "diagnostic agent, diagnose and fix before escalating:\n"
                    "1. WIDTH_MISMATCH: Read both block RTL files. Edit the RTL "
                    "on disk, then resume_pipeline(action='fix_rtl', "
                    "rtl_fix_description='Fixed width ...')\n"
                    "2. MISSING_PORT: Edit the block RTL to add it.\n"
                    "3. DIRECTION_ERROR: Fix the port direction.\n"
                    "4. LINT_ERRORS: Read the lint log and edit "
                    f"{top_rtl_path} directly.\n"
                    "5. After fixing, resume_pipeline(action='fix_rtl').\n"
                    "6. Only escalate for architectural issues."
                ),
                "reference_files": {
                    "top_rtl": top_rtl_path,
                    "architecture": ".socmate/architecture_state.json",
                    "block_diagram": ".socmate/block_diagram_viz.json",
                    "lint_log": lint_result.get("log_path", ""),
                },
            }

            response = interrupt(payload)

            action = response.get("action", "abort")
            write_graph_event(pr, "Integration Check", "graph_node_exit", {
                "action": action,
                "error_count": len(errors),
                "lint_clean": lint_clean,
            })

            if action == "skip":
                integration_result["skipped_by_user"] = True
                log("  [INTEGRATION] Skipped by user/agent", YELLOW)
            elif action == "abort":
                integration_result["aborted"] = True
                log("  [INTEGRATION] Aborted", RED)
            elif action in ("retry", "fix_rtl"):
                fix_desc = response.get("rtl_fix_description", "")
                log(f"  [INTEGRATION] Fix applied: {fix_desc}", GREEN)
                integration_result["fix_applied"] = fix_desc

            return {"integration_result": integration_result}

        log(f"\n{'='*60}", GREEN)
        log("  INTEGRATION CHECK PASSED", GREEN)
        log(f"  Top module: {module_name}", GREEN)
        log(f"  {len(modules)} blocks, "
            f"{agent_result.get('wire_count', 0)} wires", GREEN)
        if warnings:
            log(f"  {len(warnings)} warnings (non-blocking)", YELLOW)
        log(f"{'='*60}\n", GREEN)

        write_graph_event(pr, "Integration Check", "graph_node_exit", {
            "success": True,
            "top_module": module_name,
            "block_count": len(modules),
            "wire_count": agent_result.get("wire_count", 0),
            "warnings": len(warnings),
        })

        return {"integration_result": integration_result}


def route_after_integration(state: OrchestratorState) -> str:
    """Route after integration check: proceed to DV or END."""
    result = state.get("integration_result") or {}
    if result.get("aborted"):
        return END
    if result.get("skipped") or result.get("skipped_by_user"):
        return END
    if result.get("lint_clean") is False:
        return END
    if int(result.get("error_count", 0) or 0) > 0:
        return END
    return "integration_dv"


route_after_integration.__edge_labels__ = {
    END: "DONE",
    "integration_dv": "DV",
}


def _format_dv_retry_context(previous_result: dict | None) -> str:
    """Format prior top-level DV failure/audit context for retry prompts."""
    if not previous_result:
        return ""
    audit = previous_result.get("contract_audit") or {}
    if not isinstance(audit, dict) or not audit:
        bits = []
        if previous_result.get("error"):
            bits.append(f"Previous error: {previous_result.get('error')}")
        if previous_result.get("sim_log_path"):
            bits.append(f"Previous sim log: {previous_result.get('sim_log_path')}")
        return "\n".join(bits)

    first = audit.get("first_divergence") or {}
    if not isinstance(first, dict):
        first = {"summary": str(first)}
    evidence = audit.get("evidence") or []
    if isinstance(evidence, list):
        evidence_text = "\n".join(f"- {item}" for item in evidence[:8])
    else:
        evidence_text = str(evidence)

    parts = [
        f"Stage: {audit.get('stage', previous_result.get('phase', 'unknown'))}",
        f"Category: {audit.get('category', 'UNKNOWN')}",
        f"Recommended action: {audit.get('recommended_action', '')}",
        f"Contract failure: {audit.get('contract_failure', False)}",
        f"First divergence: {first.get('summary', '')}",
        f"RTL observation: {first.get('rtl_observation', '')}",
        f"Golden/expected observation: {first.get('golden_observation', '')}",
        f"Suggested fix: {audit.get('suggested_fix', '')}",
        f"Outer-agent summary: {audit.get('outer_agent_summary', '')}",
        f"Audit path: {previous_result.get('contract_audit_path', audit.get('audit_path', ''))}",
        f"Failure context path: {audit.get('context_path', '')}",
        f"Sim log path: {previous_result.get('sim_log_path', '')}",
    ]
    if evidence_text:
        parts.append("Evidence:\n" + evidence_text)
    return smart_truncate("\n".join(p for p in parts if p), 12000, "head_tail")


# ---------------------------------------------------------------------------
# Node: integration_dv  (Lead DV -- generates + runs integration testbench)
# ---------------------------------------------------------------------------

async def integration_dv_node(state: OrchestratorState) -> dict:
    """Generate and run an integration-level cocotb testbench.

    This node is the Lead DV (AI) step that:
    1. Calls the IntegrationTestbenchGenerator LLM to produce a cocotb
       testbench exercising the top-level integrated design
    2. Runs the testbench via Verilator against all block RTL
    3. On failure, fires an interrupt so the outer agent can diagnose

    Runs after integration_check passes (lint-clean top-level RTL exists).
    """
    import json as _json

    pr = state.get("project_root", str(PROJECT_ROOT))
    integration_result = state.get("integration_result") or {}

    top_rtl_path = integration_result.get("top_rtl_path", "")
    design_name = integration_result.get("design_name", "chip_top")
    block_rtl_paths = integration_result.get("block_rtl_paths", {})

    write_graph_event(pr, "Integration DV", "graph_node_enter", {
        "design_name": design_name,
        "block_count": len(block_rtl_paths),
    })

    with _tracer.start_as_current_span("Integration DV") as span:
        span.set_attribute("design_name", design_name)

        if not top_rtl_path or not Path(top_rtl_path).exists():
            log("  [INTEG-DV] Skipping -- no top-level RTL found", YELLOW)
            write_graph_event(pr, "Integration DV", "graph_node_exit", {
                "skipped": True, "reason": "no_top_rtl",
            })
            return {"integration_dv_result": {
                "skipped": True,
                "reason": "No top-level RTL available",
            }}

        if len(block_rtl_paths) < 1:
            log("  [INTEG-DV] Skipping -- no block RTL found", YELLOW)
            write_graph_event(pr, "Integration DV", "graph_node_exit", {
                "skipped": True, "reason": "no_blocks",
            })
            return {"integration_dv_result": {
                "skipped": True,
                "reason": "No block RTL files found",
            }}

        # Load connections and PRD summary for context
        connections, _ = await asyncio.to_thread(
            load_architecture_connections, pr
        )

        prd_summary = ""
        for prd_name in ("prd_spec.json", "ers_spec.json"):
            prd_path = Path(pr) / ".socmate" / prd_name
            if prd_path.exists():
                try:
                    prd_data = _json.loads(prd_path.read_text(encoding="utf-8"))
                    doc = prd_data.get("prd", prd_data.get("ers", {}))
                    prd_summary = doc.get("summary", "")
                    if doc.get("speed_and_feeds"):
                        sf = doc["speed_and_feeds"]
                        prd_summary += (
                            f"\nTarget clock: {sf.get('target_clock_mhz', '?')} MHz"
                            f", Data width: {sf.get('input_data_rate_mbps', '?')} Mbps"
                        )
                    if doc.get("dataflow"):
                        df = doc["dataflow"]
                        prd_summary += (
                            f"\nBus protocol: {df.get('bus_protocol', '?')}"
                            f", Data width: {df.get('data_width_bits', '?')} bits"
                        )
                except (OSError, _json.JSONDecodeError, KeyError):
                    pass
                break

        # Re-parse modules so the LLM gets block port details
        modules = {}
        for block_name, rtl_path in block_rtl_paths.items():
            mod = await asyncio.to_thread(parse_verilog_ports, rtl_path)
            if mod.name:
                modules[block_name] = mod

        previous_dv = state.get("integration_dv_result") or {}
        previous_action = previous_dv.get("action_taken", "")
        previous_tb_path = previous_dv.get("testbench_path", "")
        reuse_existing_tb = (
            previous_action in ("fix_rtl", "fix_tb")
            and previous_tb_path
            and Path(previous_tb_path).exists()
        )

        generation_error: Exception | None = None
        if reuse_existing_tb:
            log(
                "  [INTEG-DV] Reusing existing testbench after "
                f"{previous_action}: {previous_tb_path}",
                YELLOW,
            )
            tb_result = {
                "testbench_path": previous_tb_path,
                "tb_path": previous_tb_path,
                "test_count": previous_dv.get("test_count", 0),
            }
        else:
            # 1. Generate integration testbench
            log("  [INTEG-DV] Generating integration testbench...", YELLOW)
            try:
                tb_result = await generate_integration_testbench(
                    design_name=design_name,
                    top_rtl_path=top_rtl_path,
                    modules=modules,
                    connections=connections,
                    block_rtl_paths=block_rtl_paths,
                    prd_summary=prd_summary,
                    prior_failure=_format_dv_retry_context(previous_dv),
                )
            except Exception as e:
                tb_result = None
                generation_error = e
        if not reuse_existing_tb and tb_result is None:
            if generation_error is None:
                generation_error = RuntimeError("Integration testbench generation returned no result")
            e = generation_error
            log(f"  [INTEG-DV] Testbench generation failed: {e}", RED)
            error_msg = f"Integration testbench generation failed: {e}"
            contract_audit = await _run_top_level_contract_audit(
                stage="integration_dv_generation",
                project_root=pr,
                design_name=design_name,
                top_rtl_path=top_rtl_path,
                testbench_path="",
                test_count=0,
                sim_log=error_msg,
                sim_log_path="",
                block_rtl_paths=block_rtl_paths,
            )
            payload = {
                "type": "integration_dv_failure",
                "phase": "tb_generation",
                "design_name": design_name,
                "top_rtl_path": top_rtl_path,
                "testbench_path": "",
                "test_count": 0,
                "sim_log": error_msg,
                "sim_log_path": "",
                "block_rtl_paths": block_rtl_paths,
                "contract_audit": contract_audit,
                "contract_audit_path": contract_audit.get("audit_path", ""),
                "supported_actions": [
                    "retry",
                    "fix_rtl",
                    "fix_tb",
                    "abort",
                ],
                "outer_agent_guidance": (
                    "Integration DV could not generate a usable cocotb "
                    "testbench. As the outer-loop diagnostic agent, inspect "
                    "the top-level RTL, block port contracts, generator prompt, "
                    "and any partially written testbench. Use action='fix_tb' "
                    "when the testbench generator or prompt needs repair, "
                    "action='fix_rtl' when the top-level contract is invalid, "
                    "or action='retry' after an external fix. Do not mark the "
                    "pipeline complete until Integration DV runs.\n\n"
                    "Contract audit result: "
                    f"{contract_audit.get('category', 'UNKNOWN')} -- "
                    f"{contract_audit.get('outer_agent_summary', '')}"
                ),
                "reference_files": {
                    "top_rtl": top_rtl_path,
                    "contract_audit": contract_audit.get("audit_path", ""),
                },
            }
            response = interrupt(payload)
            action = response.get("action", "abort")
            write_graph_event(pr, "Integration DV", "graph_node_exit", {
                "error": str(e),
                "phase": "tb_generation",
                "action": action,
            })
            dv_result = {
                "passed": False,
                "error": error_msg,
                "phase": "tb_generation",
                "test_count": 0,
                "testbench_path": "",
                "design_name": design_name,
                "action_taken": action,
                "contract_audit": contract_audit,
                "contract_audit_path": contract_audit.get("audit_path", ""),
            }
            if action == "abort":
                dv_result["aborted"] = True
                log("  [INTEG-DV] Aborted", RED)
            elif action in ("retry", "fix_rtl", "fix_tb"):
                fix_desc = response.get("rtl_fix_description", "")
                dv_result["fix_applied"] = fix_desc
                log(f"  [INTEG-DV] Fix applied: {fix_desc}", GREEN)
            return {
                "integration_dv_result": dv_result,
                "pipeline_done": False,
                "pipeline_aborted": action == "abort",
            }

        tb_path = tb_result.get("testbench_path", "")
        test_count = tb_result.get("test_count", 0)
        log(f"  [INTEG-DV] Generated ({test_count} tests): {tb_path}", GREEN)
        span.set_attribute("test_count", test_count)

        # 2. Run integration simulation
        log("  [INTEG-DV] Running integration simulation...", YELLOW)
        sim_result = await asyncio.to_thread(
            run_integration_simulation,
            design_name, top_rtl_path, block_rtl_paths, tb_path,
        )

        passed = sim_result.get("passed", False)
        sim_log = sim_result.get("log", "")

        if passed:
            log(f"\n{'='*60}", GREEN)
            log("  INTEGRATION DV PASSED", GREEN)
            log(f"  {test_count} tests, all passing", GREEN)
            log(f"{'='*60}\n", GREEN)
            span.set_attribute("passed", True)

            write_graph_event(pr, "Integration DV", "graph_node_exit", {
                "passed": True,
                "test_count": test_count,
                "log_path": sim_result.get("log_path", ""),
            })

            return {
                "integration_dv_result": {
                    "passed": True,
                    "test_count": test_count,
                    "testbench_path": tb_path,
                    "sim_log_path": sim_result.get("log_path", ""),
                    "design_name": design_name,
                },
                "pipeline_done": False,
            }

        # 3. Simulation failed -- interrupt for outer agent diagnosis
        log("  [INTEG-DV] FAILED", RED)
        for line in sim_log.split("\n")[-10:]:
            if line.strip():
                log(f"    {line.strip()}", RED)

        span.set_attribute("passed", False)

        contract_audit = await _run_top_level_contract_audit(
            stage="integration_dv",
            project_root=pr,
            design_name=design_name,
            top_rtl_path=top_rtl_path,
            testbench_path=tb_path,
            test_count=test_count,
            sim_log=sim_log,
            sim_log_path=sim_result.get("log_path", ""),
            block_rtl_paths=block_rtl_paths,
        )

        payload = {
            "type": "integration_dv_failure",
            "design_name": design_name,
            "top_rtl_path": top_rtl_path,
            "testbench_path": tb_path,
            "test_count": test_count,
            "sim_log": sim_log[-3000:],
            "sim_log_path": sim_result.get("log_path", ""),
            "block_rtl_paths": block_rtl_paths,
            "contract_audit": contract_audit,
            "contract_audit_path": contract_audit.get("audit_path", ""),
            "supported_actions": [
                "retry",        # regenerate testbench + re-simulate
                "fix_rtl",      # outer agent fixed RTL, re-run sim only
                "fix_tb",       # outer agent fixed testbench, re-run sim only
                "abort",        # stop the pipeline
            ],
            "outer_agent_guidance": (
                "Integration DV (top-level simulation) failed. As the outer-loop "
                "diagnostic agent, read the sim log and testbench to diagnose:\n"
                "1. TESTBENCH BUG: If the testbench has incorrect port names, "
                "wrong timing, or bad assumptions, edit the testbench at "
                f"{tb_path} and resume with action='fix_tb'.\n"
                "2. RTL WIRING BUG: If the top-level wiring is wrong (e.g., "
                "signals crossed, wrong widths), edit the top-level RTL at "
                f"{top_rtl_path} and resume with action='fix_rtl'.\n"
                "3. BLOCK BUG: If a specific block's output is wrong, this may "
                "need per-block debugging. Note which block and escalate.\n"
                "4. TIMEOUT: If the sim timed out, check for combinational "
                "loops or missing clock/reset connections.\n"
                "5. After fixing, resume_pipeline(action='fix_rtl' or 'fix_tb') "
                "to re-run integration DV.\n"
                "6. Only escalate to the user for architectural issues."
                "\n\nContract audit result: "
                f"{contract_audit.get('category', 'UNKNOWN')} -- "
                f"{contract_audit.get('outer_agent_summary', '')}"
            ),
            "reference_files": {
                "top_rtl": top_rtl_path,
                "testbench": tb_path,
                "sim_log": sim_result.get("log_path", ""),
                "contract_audit": contract_audit.get("audit_path", ""),
            },
        }

        if os.environ.get("SOCMATE_ALLOW_SKIP_INTEGRATION_DV", "").lower() in (
            "1",
            "true",
            "yes",
        ):
            payload["supported_actions"].insert(-1, "skip")

        response = interrupt(payload)

        action = response.get("action", "abort")
        write_graph_event(pr, "Integration DV", "graph_node_exit", {
            "action": action,
            "passed": False,
            "test_count": test_count,
        })

        dv_result = {
            "passed": False,
            "test_count": test_count,
            "testbench_path": tb_path,
            "sim_log_path": sim_result.get("log_path", ""),
            "design_name": design_name,
            "action_taken": action,
            "contract_audit": contract_audit,
            "contract_audit_path": contract_audit.get("audit_path", ""),
        }

        if action == "skip":
            dv_result["skipped_by_user"] = True
            log("  [INTEG-DV] Skipped by user/agent", YELLOW)
        elif action == "abort":
            dv_result["aborted"] = True
            log("  [INTEG-DV] Aborted", RED)
        elif action in ("retry", "fix_rtl", "fix_tb"):
            fix_desc = response.get("rtl_fix_description", "")
            log(f"  [INTEG-DV] Fix applied: {fix_desc}", GREEN)
            dv_result["fix_applied"] = fix_desc

        return {
            "integration_dv_result": dv_result,
            "pipeline_done": False,
            "pipeline_aborted": action == "abort",
        }


def _load_ers_validation_context(project_root: str) -> tuple[str, int]:
    """Load ERS context for validation DV and count likely RTL-checkable reqs."""
    ers_path = Path(project_root) / ".socmate" / "ers_spec.json"
    if not ers_path.exists():
        return "", 0

    raw = ers_path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw, 0

    ers = data.get("ers", data)
    req_count = 0

    def _count_value(value) -> None:
        nonlocal req_count
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    req_count += 1
                elif isinstance(item, dict):
                    if item.get("requirement") or item.get("id"):
                        req_count += 1
                    _count_value(item)
        elif isinstance(value, dict):
            for nested in value.values():
                _count_value(nested)

    for key in (
        "functional_requirements",
        "per_block_requirements",
        "verification_requirements",
        "validation_dv_requirements",
        "validation_kpis",
    ):
        _count_value(ers.get(key))

    return json.dumps(data, indent=2), req_count


async def _run_top_level_contract_audit(
    *,
    stage: str,
    project_root: str,
    design_name: str,
    top_rtl_path: str,
    testbench_path: str,
    test_count: int,
    requirement_count: int = 0,
    sim_log: str = "",
    sim_log_path: str = "",
    block_rtl_paths: dict[str, str] | None = None,
) -> dict:
    """Run contract audit for a top-level DV failure.

    The audit is deliberately pipeline-owned: validation/integration failures
    are first classified as TB/local RTL/top wiring/contract before the outer
    agent is interrupted.
    """
    from orchestrator.langchain.agents.contract_audit_agent import ContractAuditAgent

    root = Path(project_root)
    audit_dir = root / ".socmate" / "contract_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    safe_stage = re.sub(r"[^a-zA-Z0-9_]+", "_", stage).strip("_") or "unknown"
    context_path = audit_dir / f"{safe_stage}_failure_context.json"
    output_path = audit_dir / f"{safe_stage}_contract_audit.json"

    context = {
        "stage": stage,
        "design_name": design_name,
        "top_rtl_path": top_rtl_path,
        "testbench_path": testbench_path,
        "test_count": test_count,
        "requirement_count": requirement_count,
        "sim_log_tail": sim_log[-12000:],
        "sim_log_path": sim_log_path,
        "block_rtl_paths": block_rtl_paths or {},
        "reference_files": {
            "ers_json": str(root / ".socmate" / "ers_spec.json"),
            "prd_json": str(root / ".socmate" / "prd_spec.json"),
            "block_diagram": str(root / ".socmate" / "block_diagram.json"),
            "integration_vcd": str(root / "sim_build" / "integration" / "dump.vcd"),
            "integration_wavekit_audit": str(
                root / "sim_build" / "integration" / "wavekit_audit.json"
            ),
        },
    }
    context_path.write_text(json.dumps(context, indent=2), encoding="utf-8")

    log(f"  [CONTRACT-AUDIT] Auditing {stage} failure...", YELLOW)
    agent = ContractAuditAgent(temperature=0.1)
    result = await agent.analyze(
        stage=stage,
        project_root=project_root,
        context_path=str(context_path),
        output_path=str(output_path),
    )

    write_graph_event(project_root, "Contract Audit", "graph_node_exit", {
        "stage": stage,
        "category": result.get("category", "UNKNOWN"),
        "contract_failure": result.get("contract_failure", False),
        "recommended_action": result.get("recommended_action", ""),
        "confidence": result.get("confidence", 0),
        "audit_path": str(output_path),
    })
    log(
        "  [CONTRACT-AUDIT] "
        f"{result.get('category', 'UNKNOWN')} "
        f"action={result.get('recommended_action', 'ask_human')} "
        f"confidence={result.get('confidence', 0)}",
        RED if result.get("contract_failure") else YELLOW,
    )
    result["audit_path"] = str(output_path)
    result["context_path"] = str(context_path)
    return result


def route_after_integration_dv(state: OrchestratorState) -> str:
    """Route after smoke/integration DV into ERS/KPI validation DV."""
    result = state.get("integration_dv_result") or {}
    if result.get("passed") is True:
        return "validation_dv"
    if result.get("action_taken") in ("retry", "fix_rtl", "fix_tb"):
        return "integration_dv"
    return END


route_after_integration_dv.__edge_labels__ = {
    "validation_dv": "Validation DV",
    "integration_dv": "Retry",
    END: "DONE",
}


# ---------------------------------------------------------------------------
# Node: validation_dv  (Lead Validation DV -- verifies ERS + KPIs)
# ---------------------------------------------------------------------------

async def validation_dv_node(state: OrchestratorState) -> dict:
    """Generate and run an ERS/KPI validation-level cocotb testbench.

    This stage follows smoke/integration DV. It validates measurable
    application intent preserved in the ERS and records requirement coverage.
    """
    pr = state.get("project_root", str(PROJECT_ROOT))
    integration_result = state.get("integration_result") or {}

    top_rtl_path = integration_result.get("top_rtl_path", "")
    design_name = integration_result.get("design_name", "chip_top")
    block_rtl_paths = integration_result.get("block_rtl_paths", {})

    write_graph_event(pr, "Validation DV", "graph_node_enter", {
        "design_name": design_name,
        "block_count": len(block_rtl_paths),
    })

    with _tracer.start_as_current_span("Validation DV") as span:
        span.set_attribute("design_name", design_name)

        if not top_rtl_path or not Path(top_rtl_path).exists():
            msg = "No top-level RTL available for Validation DV"
            log(f"  [VALIDATION-DV] FAILED -- {msg}", RED)
            return {"validation_dv_result": {
                "passed": False,
                "error": msg,
                "phase": "preflight",
                "aborted": True,
            }}

        if len(block_rtl_paths) < 1:
            msg = "No block RTL files available for Validation DV"
            log(f"  [VALIDATION-DV] FAILED -- {msg}", RED)
            return {"validation_dv_result": {
                "passed": False,
                "error": msg,
                "phase": "preflight",
                "aborted": True,
            }}

        ers_context, requirement_count = _load_ers_validation_context(pr)
        if not ers_context:
            msg = "No ERS found; Validation DV cannot verify requirements"
            log(f"  [VALIDATION-DV] FAILED -- {msg}", RED)
            return {"validation_dv_result": {
                "passed": False,
                "error": msg,
                "phase": "missing_ers",
                "aborted": True,
            }}

        connections, _ = await asyncio.to_thread(load_architecture_connections, pr)

        modules = {}
        for block_name, rtl_path in block_rtl_paths.items():
            mod = await asyncio.to_thread(parse_verilog_ports, rtl_path)
            if mod.name:
                modules[block_name] = mod

        previous_dv = state.get("validation_dv_result") or {}
        previous_action = previous_dv.get("action_taken", "")
        previous_tb_path = previous_dv.get("testbench_path", "")
        reuse_existing_tb = (
            previous_action in ("fix_rtl", "fix_tb")
            and previous_tb_path
            and Path(previous_tb_path).exists()
        )

        generation_error: Exception | None = None
        if reuse_existing_tb:
            log(
                "  [VALIDATION-DV] Reusing existing testbench after "
                f"{previous_action}: {previous_tb_path}",
                YELLOW,
            )
            tb_result = {
                "testbench_path": previous_tb_path,
                "tb_path": previous_tb_path,
                "test_count": previous_dv.get("test_count", 0),
            }
        else:
            log("  [VALIDATION-DV] Generating ERS/KPI validation testbench...", YELLOW)
            try:
                tb_result = await generate_validation_testbench(
                    design_name=design_name,
                    top_rtl_path=top_rtl_path,
                    modules=modules,
                    connections=connections,
                    block_rtl_paths=block_rtl_paths,
                    ers_context=smart_truncate(ers_context, 30000),
                    prior_failure=_format_dv_retry_context(previous_dv),
                )
            except Exception as e:
                tb_result = None
                generation_error = e
        if not reuse_existing_tb and tb_result is None:
            if generation_error is None:
                generation_error = RuntimeError("Validation testbench generation returned no result")
            e = generation_error
            log(f"  [VALIDATION-DV] Testbench generation failed: {e}", RED)
            error_msg = f"Validation testbench generation failed: {e}"
            contract_audit = await _run_top_level_contract_audit(
                stage="validation_dv_generation",
                project_root=pr,
                design_name=design_name,
                top_rtl_path=top_rtl_path,
                testbench_path="",
                test_count=0,
                requirement_count=requirement_count,
                sim_log=error_msg,
                sim_log_path="",
                block_rtl_paths=block_rtl_paths,
            )
            payload = {
                "type": "validation_dv_failure",
                "phase": "tb_generation",
                "design_name": design_name,
                "top_rtl_path": top_rtl_path,
                "testbench_path": "",
                "test_count": 0,
                "requirement_count": requirement_count,
                "sim_log": error_msg,
                "sim_log_path": "",
                "block_rtl_paths": block_rtl_paths,
                "contract_audit": contract_audit,
                "contract_audit_path": contract_audit.get("audit_path", ""),
                "supported_actions": [
                    "retry",
                    "fix_rtl",
                    "fix_tb",
                    "abort",
                ],
                "outer_agent_guidance": (
                    "Validation DV could not generate a usable cocotb "
                    "testbench for the measurable ERS/KPI requirements. "
                    "Diagnose whether the failure is missing ERS/KPI detail, "
                    "an invalid top-level contract, or a validation testbench "
                    "generation bug. Use action='fix_tb' when the validation "
                    "testbench prompt/generator needs repair, action='fix_rtl' "
                    "when RTL/top contracts must change, or action='retry' "
                    "after applying an external fix. Do not mark the pipeline "
                    "complete until Validation DV runs and verifies every ERS "
                    "requirement.\n\nContract audit result: "
                    f"{contract_audit.get('category', 'UNKNOWN')} -- "
                    f"{contract_audit.get('outer_agent_summary', '')}"
                ),
                "reference_files": {
                    "top_rtl": top_rtl_path,
                    "ers": str(Path(pr) / ".socmate" / "ers_spec.json"),
                    "contract_audit": contract_audit.get("audit_path", ""),
                },
            }
            response = interrupt(payload)
            action = response.get("action", "abort")
            write_graph_event(pr, "Validation DV", "graph_node_exit", {
                "error": str(e),
                "phase": "tb_generation",
                "action": action,
            })
            dv_result = {
                "passed": False,
                "error": error_msg,
                "phase": "tb_generation",
                "requirement_count": requirement_count,
                "test_count": 0,
                "testbench_path": "",
                "design_name": design_name,
                "action_taken": action,
                "contract_audit": contract_audit,
                "contract_audit_path": contract_audit.get("audit_path", ""),
            }
            if action == "abort":
                dv_result["aborted"] = True
                log("  [VALIDATION-DV] Aborted", RED)
            elif action in ("retry", "fix_rtl", "fix_tb"):
                fix_desc = response.get("rtl_fix_description", "")
                dv_result["fix_applied"] = fix_desc
                log(f"  [VALIDATION-DV] Fix applied: {fix_desc}", GREEN)
            return {
                "validation_dv_result": dv_result,
                "pipeline_done": False,
                "pipeline_aborted": action == "abort",
            }

        tb_path = tb_result.get("testbench_path", "")
        test_count = tb_result.get("test_count", 0)
        log(f"  [VALIDATION-DV] Generated ({test_count} tests): {tb_path}", GREEN)
        span.set_attribute("test_count", test_count)
        span.set_attribute("requirement_count", requirement_count)

        log("  [VALIDATION-DV] Running validation simulation...", YELLOW)
        sim_result = await asyncio.to_thread(
            run_integration_simulation,
            design_name, top_rtl_path, block_rtl_paths, tb_path,
        )

        passed = sim_result.get("passed", False)
        sim_log = sim_result.get("log", "")

        if passed:
            log(f"\n{'='*60}", GREEN)
            log("  VALIDATION DV PASSED", GREEN)
            log(f"  {test_count} tests, ERS requirements covered", GREEN)
            log(f"{'='*60}\n", GREEN)
            write_graph_event(pr, "Validation DV", "graph_node_exit", {
                "passed": True,
                "test_count": test_count,
                "requirement_count": requirement_count,
                "log_path": sim_result.get("log_path", ""),
            })
            return {
                "validation_dv_result": {
                    "passed": True,
                    "test_count": test_count,
                    "requirement_count": requirement_count,
                    "testbench_path": tb_path,
                    "sim_log_path": sim_result.get("log_path", ""),
                    "design_name": design_name,
                },
                "pipeline_done": True,
            }

        log("  [VALIDATION-DV] FAILED", RED)
        for line in sim_log.split("\n")[-10:]:
            if line.strip():
                log(f"    {line.strip()}", RED)

        contract_audit = await _run_top_level_contract_audit(
            stage="validation_dv",
            project_root=pr,
            design_name=design_name,
            top_rtl_path=top_rtl_path,
            testbench_path=tb_path,
            test_count=test_count,
            requirement_count=requirement_count,
            sim_log=sim_log,
            sim_log_path=sim_result.get("log_path", ""),
            block_rtl_paths=block_rtl_paths,
        )

        payload = {
            "type": "validation_dv_failure",
            "design_name": design_name,
            "top_rtl_path": top_rtl_path,
            "testbench_path": tb_path,
            "test_count": test_count,
            "requirement_count": requirement_count,
            "sim_log": sim_log[-3000:],
            "sim_log_path": sim_result.get("log_path", ""),
            "block_rtl_paths": block_rtl_paths,
            "contract_audit": contract_audit,
            "contract_audit_path": contract_audit.get("audit_path", ""),
            "supported_actions": [
                "retry",
                "fix_rtl",
                "fix_tb",
                "abort",
            ],
            "outer_agent_guidance": (
                "Validation DV failed after smoke/integration DV passed. "
                "Diagnose whether the failure is a real ERS/KPI miss, an RTL "
                "bug, or an over/under-constrained validation testbench. Fix "
                "RTL with action='fix_rtl' or fix the generated validation "
                "testbench with action='fix_tb'. Do not skip this stage unless "
                "the pipeline is explicitly configured to permit validation "
                "skips.\n\nContract audit result: "
                f"{contract_audit.get('category', 'UNKNOWN')} -- "
                f"{contract_audit.get('outer_agent_summary', '')}"
            ),
            "reference_files": {
                "top_rtl": top_rtl_path,
                "testbench": tb_path,
                "sim_log": sim_result.get("log_path", ""),
                "ers": str(Path(pr) / ".socmate" / "ers_spec.json"),
                "contract_audit": contract_audit.get("audit_path", ""),
            },
        }

        if os.environ.get("SOCMATE_ALLOW_SKIP_VALIDATION_DV", "").lower() in (
            "1",
            "true",
            "yes",
        ):
            payload["supported_actions"].insert(-1, "skip")

        response = interrupt(payload)
        action = response.get("action", "abort")
        write_graph_event(pr, "Validation DV", "graph_node_exit", {
            "action": action,
            "passed": False,
            "test_count": test_count,
            "requirement_count": requirement_count,
        })

        dv_result = {
            "passed": False,
            "test_count": test_count,
            "requirement_count": requirement_count,
            "testbench_path": tb_path,
            "sim_log_path": sim_result.get("log_path", ""),
            "design_name": design_name,
            "action_taken": action,
            "contract_audit": contract_audit,
            "contract_audit_path": contract_audit.get("audit_path", ""),
        }

        if action == "skip":
            dv_result["skipped_by_user"] = True
            log("  [VALIDATION-DV] Skipped by explicit configuration", YELLOW)
        elif action == "abort":
            dv_result["aborted"] = True
            log("  [VALIDATION-DV] Aborted", RED)
        elif action in ("retry", "fix_rtl", "fix_tb"):
            fix_desc = response.get("rtl_fix_description", "")
            log(f"  [VALIDATION-DV] Fix applied: {fix_desc}", GREEN)
            dv_result["fix_applied"] = fix_desc

        return {
            "validation_dv_result": dv_result,
            "pipeline_done": False,
            "pipeline_aborted": action == "abort",
        }


def route_after_validation_dv(state: OrchestratorState) -> str:
    """Route after validation DV: terminal frontend pipeline."""
    result = state.get("validation_dv_result") or {}
    if result.get("action_taken") in ("retry", "fix_rtl", "fix_tb"):
        return "validation_dv"
    return END


route_after_validation_dv.__edge_labels__ = {
    "validation_dv": "Retry",
    END: "DONE",
}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_block_subgraph_compiled(checkpointer=None):
    """Build and compile the block lifecycle subgraph standalone.

    Used by the graph introspection / web UI visualizer so the frontend
    graph view shows the full block lifecycle pipeline (10 nodes) rather
    than the thin orchestrator wrapper (4 nodes).
    """
    return build_block_subgraph().compile(checkpointer=checkpointer)


def build_pipeline_graph(checkpointer=None):
    """Build and compile the orchestrator pipeline graph.

    The orchestrator fans out blocks within each tier for parallel
    execution via ``Send()``.  Each block runs through the full block
    lifecycle subgraph autonomously.

    Args:
        checkpointer: LangGraph checkpointer for state persistence.
            Use ``MemorySaver`` for tests, ``AsyncSqliteSaver`` for
            production.

    Returns:
        Compiled StateGraph ready for ``ainvoke`` / ``astream``.
    """
    block_subgraph = build_block_subgraph().compile()

    orchestrator = StateGraph(OrchestratorState)

    # Nodes
    orchestrator.add_node("init_tier", init_tier_node)
    orchestrator.add_node("process_block", block_subgraph)
    orchestrator.add_node("integration_review", integration_review_node)
    orchestrator.add_node("advance_tier", advance_tier_node)
    orchestrator.add_node("pipeline_complete", pipeline_complete_node)
    orchestrator.add_node("integration_check", integration_check_node)
    orchestrator.add_node("integration_dv", integration_dv_node)
    orchestrator.add_node("validation_dv", validation_dv_node)

    # Edges
    orchestrator.add_edge(START, "init_tier")
    orchestrator.add_conditional_edges("init_tier", fan_out_tier)
    orchestrator.add_edge("process_block", "integration_review")
    orchestrator.add_conditional_edges("integration_review", route_after_integration_review)
    orchestrator.add_conditional_edges("advance_tier", route_next_tier)
    orchestrator.add_conditional_edges(
        "pipeline_complete",
        lambda s: END if s.get("pipeline_aborted") else "integration_check",
        {END: END, "integration_check": "integration_check"},
    )
    orchestrator.add_conditional_edges("integration_check", route_after_integration)
    orchestrator.add_conditional_edges("integration_dv", route_after_integration_dv)
    orchestrator.add_conditional_edges("validation_dv", route_after_validation_dv)

    return orchestrator.compile(checkpointer=checkpointer)
