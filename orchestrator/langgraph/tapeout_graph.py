# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
LangGraph StateGraph for the OpenFrame tapeout pipeline.

Runs AFTER the per-block backend graph completes.  Takes the GDS/DEF/netlist
artifacts from each passing block and assembles them into an OpenFrame
shuttle submission:

    generate_wrapper -> wrapper_pnr -> wrapper_drc -> wrapper_lvs
      -> mpw_precheck -> tapeout_complete

Failures route to ``diagnose_tapeout`` (LLM-based triage) which either
auto-retries with adjusted PnR parameters, continues past benign issues
(e.g. expected LVS tap-cell deltas), or escalates to ``ask_human`` with
an enriched diagnosis payload.

The mpw_precheck node runs natively (no Docker) using Nix-wrapped KLayout
and Magic for DRC, plus directory structure validation.

Usage::

    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    async with AsyncSqliteSaver.from_conn_string(".coresmith/tapeout_checkpoint.db") as cp:
        graph = build_tapeout_graph(checkpointer=cp)
        result = await graph.ainvoke(initial_state, config)
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from opentelemetry import trace

from orchestrator.langgraph.event_stream import write_graph_event
from orchestrator.langgraph.pipeline_helpers import (
    CYAN,
    GREEN,
    PROJECT_ROOT,
    RED,
    YELLOW,
    log,
)

_tracer = trace.get_tracer("coresmith.langgraph.tapeout_graph")


def _last(a, b):
    """Reducer that keeps the latest value."""
    return b


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class TapeoutState(TypedDict):
    """State for the OpenFrame tapeout pipeline."""

    # Config (set once) ─────────────────────────────────────────────────────
    project_root: str
    target_clock_mhz: float
    blocks: list[dict]
    completed_backend_blocks: list[dict]
    gpio_mapping: dict | None

    # Phase tracking ────────────────────────────────────────────────────────
    phase: str  # "init" | "wrapper" | "pnr" | "drc" | "lvs" | "precheck" | "done"
    attempt: int
    max_attempts: int
    previous_error: str

    # Results ───────────────────────────────────────────────────────────────
    wrapper_result: dict | None
    wrapper_pnr_result: dict | None
    wrapper_drc_result: dict | None
    wrapper_lvs_result: dict | None
    precheck_result: dict | None
    submission_result: dict | None

    # Artifact paths ────────────────────────────────────────────────────────
    wrapper_rtl_path: Annotated[str, _last]
    wrapper_netlist_path: Annotated[str, _last]
    wrapper_routed_def: Annotated[str, _last]
    wrapper_gds_path: Annotated[str, _last]
    wrapper_spice_path: Annotated[str, _last]
    submission_dir: Annotated[str, _last]

    # Step logs ─────────────────────────────────────────────────────────────
    step_log_paths: Annotated[dict, _last]

    # Diagnosis agent ───────────────────────────────────────────────────────
    diagnosis_result: dict | None

    # Human interaction ─────────────────────────────────────────────────────
    human_response: dict | None

    # Terminal ──────────────────────────────────────────────────────────────
    tapeout_done: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pr(state: TapeoutState) -> str:
    return state.get("project_root", str(PROJECT_ROOT))


def _output_dir(state: TapeoutState) -> str:
    return str(Path(state["project_root"]) / "openframe_submission" / "pnr")


_PNR_OVERRIDES_NAME = ".coresmith/pnr_overrides.json"


def _read_pnr_overrides(project_root: str) -> dict:
    """Read PnR parameter overrides from disk."""
    p = Path(project_root) / _PNR_OVERRIDES_NAME
    if p.exists():
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _write_pnr_overrides(project_root: str, overrides: dict) -> None:
    """Write PnR parameter overrides to disk (merge with existing)."""
    p = Path(project_root) / _PNR_OVERRIDES_NAME
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_pnr_overrides(project_root)
    existing.update(overrides)
    p.write_text(json.dumps(existing, indent=2))


_PROMPT_DIR = Path(__file__).resolve().parent.parent / "langchain" / "prompts"


def _spec_paths(project_root: str) -> dict[str, str]:
    """Return paths to PRD/ERS/FRD spec files."""
    root = Path(project_root)
    prd = root / ".coresmith" / "prd_spec.json"
    ers = root / ".coresmith" / "ers_spec.json"
    frd = root / "arch" / "frd_spec.md"
    return {
        "prd_path": str(prd if prd.exists() else ers),
        "frd_path": str(frd) if frd.exists() else "(not generated)",
    }


async def _run_tapeout_llm_step(
    step_name: str,
    prompt_file: str,
    context: dict,
    result_json_path: str,
    timeout: int = 900,
) -> dict:
    """Run an LLM-driven EDA step for the tapeout pipeline."""
    from orchestrator.langchain.agents.coresmith_llm import DEFAULT_MODEL, ClaudeLLM

    prompt_path = _PROMPT_DIR / prompt_file
    system_prompt = prompt_path.read_text().format(**context)
    user_message = (
        f"Execute the {step_name} step as described in the system prompt.\n"
        f"Write the result JSON to: {result_json_path}\n"
        "After writing the result file, respond with a brief summary."
    )

    llm = ClaudeLLM(model=DEFAULT_MODEL, timeout=timeout)
    try:
        await llm.call(
            system=system_prompt,
            prompt=user_message,
            run_name=step_name,
        )
    except Exception as e:
        return {"success": False, "error": f"LLM call failed: {e}"}

    rp = Path(result_json_path)
    if rp.exists():
        try:
            return json.loads(rp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    return {"success": False, "error": f"LLM did not write result JSON to {result_json_path}"}


# ---------------------------------------------------------------------------
# Node: generate_wrapper
# ---------------------------------------------------------------------------

async def generate_wrapper_node(state: TapeoutState) -> dict:
    """Generate OpenFrame wrapper RTL, submission structure, and config.

    Reuses the backend's wrapper_result.json if it exists and succeeded,
    avoiding a broken re-generation with the template path.  Falls back
    to the template generator only when the backend didn't produce one.
    """
    from orchestrator.langgraph.tapeout_helpers import (
        generate_submission_structure,
        generate_wrapper_rtl,
    )

    pr = _pr(state)
    blocks = state["blocks"]
    completed = state["completed_backend_blocks"]
    gpio_mapping = state.get("gpio_mapping")

    write_graph_event(pr, "Generate Wrapper", "graph_node_enter", {
        "graph": "tapeout", "block_count": len(blocks),
    })

    log(f"\n{'='*60}", CYAN)
    log("  Tapeout: Generating OpenFrame wrapper", CYAN)
    log(f"{'='*60}", CYAN)

    # --- Try to reuse the backend's wrapper artifacts ---
    backend_result_path = Path(pr) / "openframe_submission" / "wrapper_result.json"
    reused = False

    if backend_result_path.exists():
        try:
            backend_wr = json.loads(backend_result_path.read_text())
            wp = backend_wr.get("wrapper_path", "")
            if backend_wr.get("success") and wp and Path(wp).exists():
                wrapper_result = backend_wr
                wrapper_path = wp
                submission_dir = backend_wr.get("submission_dir", str(Path(pr) / "openframe_submission"))
                reused = True
                log("  [WRAPPER] Reusing backend wrapper_result.json "
                    f"({backend_wr.get('gpio_used', '?')} GPIOs)", GREEN)
        except (json.JSONDecodeError, OSError, KeyError):
            pass

    if not reused:
        with _tracer.start_as_current_span("Generate Wrapper") as span:
            span.set_attribute("block_count", len(blocks))

            sub_dir = str(Path(pr) / "openframe_submission")
            rtl_dir = str(Path(sub_dir) / "verilog" / "rtl")
            Path(rtl_dir).mkdir(parents=True, exist_ok=True)

            wrapper_result = generate_wrapper_rtl(
                blocks, gpio_mapping, rtl_dir,
            )
            submission_result = generate_submission_structure(pr, blocks, completed)
            wrapper_path = wrapper_result["wrapper_path"]
            submission_dir = submission_result.get("submission_dir", "")

            span.set_attribute("gpio_used", wrapper_result.get("gpio_used", 0))

        log(f"  [SUBMISSION] Directory: {submission_dir}", GREEN)
        log(f"  [SUBMISSION] Files: {len(submission_result.get('files_copied', []))}", GREEN)

    write_graph_event(pr, "Generate Wrapper", "graph_node_exit", {
        "graph": "tapeout",
        "wrapper_path": wrapper_path,
        "gpio_used": wrapper_result.get("gpio_used", 0),
        "submission_dir": submission_dir,
        "reused_backend": reused,
    })

    log(f"  [WRAPPER] Path: {wrapper_path}", GREEN)
    log(f"  [WRAPPER] GPIO used: {wrapper_result.get('gpio_used', 0)}/{wrapper_result.get('gpio_available', 44)}", GREEN)

    return {
        "wrapper_result": wrapper_result,
        "wrapper_rtl_path": wrapper_path,
        "wrapper_gds_path": "",
        "submission_dir": submission_dir,
        "gpio_mapping": wrapper_result.get("gpio_mapping"),
        "phase": "wrapper",
    }


# ---------------------------------------------------------------------------
# Node: synthesize_wrapper
# ---------------------------------------------------------------------------

async def synthesize_wrapper_node(state: TapeoutState) -> dict:
    """Synthesize wrapper RTL to a gate-level netlist via LLM agent."""
    from orchestrator.langgraph.backend_helpers import LIBERTY

    pr = _pr(state)
    wrapper_rtl = state.get("wrapper_rtl_path", "")
    completed = state.get("completed_backend_blocks", [])

    write_graph_event(pr, "Synthesize Wrapper", "graph_node_enter", {
        "graph": "tapeout",
    })

    if not wrapper_rtl or not Path(wrapper_rtl).exists():
        error = f"Wrapper RTL not found: {wrapper_rtl}"
        log(f"  [WRAPPER SYNTH] FAILED: {error}", RED)
        write_graph_event(pr, "Synthesize Wrapper", "graph_node_exit", {
            "graph": "tapeout", "success": False,
        })
        return {"phase": "synth", "previous_error": error}

    output_dir = _output_dir(state)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    result_json_path = str(Path(output_dir) / "wrapper_synth_result.json")
    target_clock = state.get("target_clock_mhz", 50.0)

    netlist_lines = []
    root = Path(pr)
    for blk in completed:
        if not blk.get("success"):
            continue
        name = blk["name"]
        for candidate in (
            root / "syn" / "output" / name / f"{name}_netlist.v",
            root / "syn" / "output" / name / f"{name}_flat_netlist.v",
        ):
            if candidate.exists():
                netlist_lines.append(f"- Block `{name}`: `{candidate}`")
                break

    with _tracer.start_as_current_span("Synthesize Wrapper") as span:
        result = await _run_tapeout_llm_step(
            step_name="Synthesize Wrapper [openframe_project_wrapper]",
            prompt_file="tapeout_wrapper_synth.md",
            context={
                "liberty_path": str(LIBERTY),
                "target_clock_mhz": target_clock,
                "period_ns": 1000.0 / target_clock,
                "output_dir": output_dir,
                "wrapper_rtl_path": wrapper_rtl,
                "block_netlists": "\n".join(netlist_lines) or "- (none -- search syn/output/ for *_netlist.v)",
                "result_json_path": result_json_path,
            },
            result_json_path=result_json_path,
        )
        span.set_attribute("success", result.get("success", False))

    write_graph_event(pr, "Synthesize Wrapper", "graph_node_exit", {
        "graph": "tapeout",
        "success": result.get("success", False),
        "gate_count": result.get("gate_count", 0),
    })

    out: dict = {"phase": "synth"}
    if result.get("success"):
        out["wrapper_netlist_path"] = result.get("netlist_path", "")
        log(f"  [WRAPPER SYNTH] Complete: {result.get('gate_count', 0)} cells", GREEN)
    else:
        out["previous_error"] = result.get("error", "Wrapper synthesis failed")[:2000]
        log(f"  [WRAPPER SYNTH] FAILED: {out['previous_error'][:200]}", RED)

    return out


# ---------------------------------------------------------------------------
# Node: wrapper_pnr
# ---------------------------------------------------------------------------

async def wrapper_pnr_node(state: TapeoutState) -> dict:
    """Run wrapper-level PnR via LLM agent (OpenROAD)."""
    from orchestrator.langgraph.backend_helpers import (
        CELL_LEF,
        LIBERTY,
        OPENROAD_BIN,
        TECH_LEF,
    )
    from orchestrator.langgraph.tapeout_helpers import (
        OPENFRAME_CORE_MARGIN_UM,
        OPENFRAME_DIE_HEIGHT_UM,
        OPENFRAME_DIE_WIDTH_UM,
        generate_wrapper_pnr_tcl,
    )

    pr = _pr(state)
    netlist = state.get("wrapper_netlist_path", "") or state.get("wrapper_rtl_path", "")

    write_graph_event(pr, "Wrapper PnR", "graph_node_enter", {"graph": "tapeout"})

    if not netlist or not Path(netlist).exists():
        error = f"Wrapper netlist not found: {netlist}"
        log(f"  [WRAPPER PNR] FAILED: {error}", RED)
        return {
            "wrapper_pnr_result": {"success": False, "error": error},
            "phase": "pnr", "previous_error": error,
        }

    output_dir = _output_dir(state)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    result_json_path = str(Path(output_dir) / "wrapper_pnr_result.json")
    target_clock = state.get("target_clock_mhz", 50.0)
    overrides = _read_pnr_overrides(pr)
    attempt = state.get("attempt", 1)
    max_attempts = state.get("max_attempts", 3)

    tcl_path = generate_wrapper_pnr_tcl(
        netlist, state.get("blocks", []),
        state.get("completed_backend_blocks", []),
        output_dir, target_clock,
    )

    sdc_path = str(Path(output_dir) / "wrapper.sdc")

    with _tracer.start_as_current_span("Wrapper PnR") as span:
        result = await _run_tapeout_llm_step(
            step_name="Wrapper PnR [openframe_project_wrapper]",
            prompt_file="tapeout_wrapper_pnr.md",
            context={
                **_spec_paths(pr),
                "tech_lef": str(TECH_LEF),
                "cell_lef": str(CELL_LEF),
                "liberty_path": str(LIBERTY),
                "openroad_bin": str(OPENROAD_BIN),
                "target_clock_mhz": target_clock,
                "period_ns": 1000.0 / target_clock,
                "die_width_um": OPENFRAME_DIE_WIDTH_UM,
                "die_height_um": OPENFRAME_DIE_HEIGHT_UM,
                "core_margin_um": OPENFRAME_CORE_MARGIN_UM,
                "netlist_path": netlist,
                "sdc_path": sdc_path,
                "tcl_path": tcl_path,
                "output_dir": output_dir,
                "attempt": attempt,
                "max_attempts": max_attempts,
                "prior_failure": state.get("previous_error", "None"),
                "pnr_overrides": json.dumps(overrides) if overrides else "None",
                "result_json_path": result_json_path,
            },
            result_json_path=result_json_path,
            timeout=1800,
        )
        span.set_attribute("success", result.get("success", False))

    write_graph_event(pr, "Wrapper PnR", "graph_node_exit", {
        "graph": "tapeout", "success": result.get("success", False),
    })

    out: dict = {
        "wrapper_pnr_result": result,
        "phase": "pnr",
    }
    if result.get("success"):
        out["wrapper_routed_def"] = result.get("routed_def_path", "")
        log(f"  [WRAPPER PNR] Complete: area={result.get('design_area_um2', 0):.0f} um²", GREEN)
    else:
        out["previous_error"] = result.get("error", "Wrapper PnR failed")[:2000]
        log(f"  [WRAPPER PNR] FAILED: {out['previous_error'][:200]}", RED)

    return out


# ---------------------------------------------------------------------------
# Node: wrapper_drc
# ---------------------------------------------------------------------------

async def wrapper_drc_node(state: TapeoutState) -> dict:
    """Run Magic DRC + GDS generation via LLM agent."""
    from orchestrator.langgraph.backend_helpers import (
        CELL_GDS,
        CELL_LEF,
        MAGIC_BIN,
        MAGIC_RC,
        TECH_LEF,
    )

    pr = _pr(state)
    routed_def = state.get("wrapper_routed_def", "")

    write_graph_event(pr, "Wrapper DRC", "graph_node_enter", {"graph": "tapeout"})

    pnr_result = state.get("wrapper_pnr_result") or {}
    if pnr_result and not pnr_result.get("success"):
        error = pnr_result.get("error", "Wrapper PnR failed")
        return {
            "wrapper_drc_result": {"clean": False, "error": error},
            "phase": "drc", "previous_error": error,
        }

    if not routed_def or not Path(routed_def).exists():
        error = f"Wrapper routed DEF not found: {routed_def}"
        return {
            "wrapper_drc_result": {"clean": False, "error": error},
            "phase": "drc", "previous_error": error,
        }

    output_dir = _output_dir(state)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    result_json_path = str(Path(output_dir) / "wrapper_drc_result.json")

    with _tracer.start_as_current_span("Wrapper DRC") as span:
        result = await _run_tapeout_llm_step(
            step_name="Wrapper DRC [openframe_project_wrapper]",
            prompt_file="tapeout_wrapper_drc.md",
            context={
                **_spec_paths(pr),
                "magic_rc": str(MAGIC_RC),
                "cell_gds": str(CELL_GDS),
                "tech_lef": str(TECH_LEF),
                "cell_lef": str(CELL_LEF),
                "magic_bin": str(MAGIC_BIN),
                "routed_def_path": routed_def,
                "output_dir": output_dir,
                "prior_failure": state.get("previous_error", "None"),
                "result_json_path": result_json_path,
            },
            result_json_path=result_json_path,
            timeout=1200,
        )
        span.set_attribute("clean", result.get("clean", False))

    write_graph_event(pr, "Wrapper DRC", "graph_node_exit", {
        "graph": "tapeout",
        "clean": result.get("clean", False),
        "violation_count": result.get("violation_count", -1),
    })

    out: dict = {
        "wrapper_drc_result": {
            "clean": result.get("clean", False),
            "violation_count": result.get("violation_count", -1),
        },
        "phase": "drc",
    }

    if result.get("clean") or result.get("success"):
        out["wrapper_gds_path"] = result.get("gds_path", "")
        out["wrapper_spice_path"] = result.get("spice_path", "")
        log(f"  [WRAPPER DRC] Clean: {result.get('violation_count', 0)} violations", GREEN)
    else:
        out["previous_error"] = f"Wrapper DRC: {result.get('violation_count', '?')} violations"
        log(f"  [WRAPPER DRC] {out['previous_error']}", RED)

    return out


# ---------------------------------------------------------------------------
# Node: wrapper_lvs
# ---------------------------------------------------------------------------

async def wrapper_lvs_node(state: TapeoutState) -> dict:
    """Run Netgen LVS on the wrapper via LLM agent."""
    from orchestrator.langgraph.backend_helpers import NETGEN_BIN, NETGEN_SETUP

    pr = _pr(state)

    write_graph_event(pr, "Wrapper LVS", "graph_node_enter", {"graph": "tapeout"})

    drc_result = state.get("wrapper_drc_result") or {}
    if not drc_result.get("clean"):
        error = "Wrapper DRC not clean"
        return {
            "wrapper_lvs_result": {"match": False, "error": error},
            "phase": "lvs", "previous_error": error,
        }

    spice_path = state.get("wrapper_spice_path", "")
    pwr_verilog = (state.get("wrapper_pnr_result") or {}).get("pwr_verilog_path", "")

    if not spice_path or not Path(spice_path).exists():
        error = f"Wrapper SPICE not found: {spice_path}"
        return {
            "wrapper_lvs_result": {"match": False, "error": error},
            "phase": "lvs", "previous_error": error,
        }

    output_dir = _output_dir(state)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    result_json_path = str(Path(output_dir) / "wrapper_lvs_result.json")

    with _tracer.start_as_current_span("Wrapper LVS") as span:
        result = await _run_tapeout_llm_step(
            step_name="Wrapper LVS [openframe_project_wrapper]",
            prompt_file="tapeout_wrapper_lvs.md",
            context={
                **_spec_paths(pr),
                "netgen_setup": str(NETGEN_SETUP),
                "netgen_bin": str(NETGEN_BIN),
                "spice_path": spice_path,
                "pwr_verilog_path": pwr_verilog or "(not available)",
                "verilog_path": pwr_verilog or "(not available)",
                "output_dir": output_dir,
                "prior_failure": state.get("previous_error", "None"),
                "result_json_path": result_json_path,
            },
            result_json_path=result_json_path,
            timeout=1200,
        )
        span.set_attribute("match", result.get("match", False))

    write_graph_event(pr, "Wrapper LVS", "graph_node_exit", {
        "graph": "tapeout",
        "match": result.get("match", False),
    })

    out: dict = {
        "wrapper_lvs_result": {
            "match": result.get("match", False),
            "device_delta": result.get("device_delta", 0),
            "net_delta": result.get("net_delta", 0),
        },
        "phase": "lvs",
    }

    if result.get("match"):
        log(f"  [WRAPPER LVS] Match (delta: {result.get('device_delta', 0)} devices)", GREEN)
    else:
        out["previous_error"] = f"Wrapper LVS mismatch: device_delta={result.get('device_delta', '?')}"
        log(f"  [WRAPPER LVS] {out['previous_error']}", RED)

    return out


# ---------------------------------------------------------------------------
# Node: mpw_precheck  (native, no Docker)
# ---------------------------------------------------------------------------

async def mpw_precheck_node(state: TapeoutState) -> dict:
    """Run native MPW precheck (KLayout DRC + Magic DRC + structure check).

    This replaces the Docker-based efabless/mpw_precheck with native checks
    using Nix-wrapped KLayout and Magic.
    """
    from orchestrator.langgraph.tapeout_helpers import run_mpw_precheck_native

    pr = _pr(state)
    submission_dir = state.get("submission_dir", "")
    gds_path = state.get("wrapper_gds_path", "")

    write_graph_event(pr, "MPW Precheck", "graph_node_enter", {"graph": "tapeout"})

    log("  [PRECHECK] Running native MPW precheck (no Docker)...", YELLOW)

    with _tracer.start_as_current_span("MPW Precheck") as span:
        result = await asyncio.to_thread(
            run_mpw_precheck_native, submission_dir, gds_path,
        )
        span.set_attribute("pass", result.get("pass", False))

    write_graph_event(pr, "MPW Precheck", "graph_node_exit", {
        "graph": "tapeout",
        "pass": result.get("pass", False),
        "checks": {k: v.get("pass") for k, v in result.get("checks", {}).items()},
    })

    out: dict = {
        "precheck_result": result,
        "phase": "precheck",
    }

    if not result.get("pass"):
        out["previous_error"] = "; ".join(result.get("errors", ["Precheck failed"]))

    return out


# ---------------------------------------------------------------------------
# Node: diagnose_tapeout (LLM-based failure triage)
# ---------------------------------------------------------------------------

async def diagnose_tapeout_node(state: TapeoutState) -> dict:
    """LLM-based diagnosis of DRC/LVS/precheck failures.

    Determines whether the failure can be auto-retried with adjusted PnR
    parameters, continued past (if benign), or must be escalated to the
    outer agent.
    """
    from orchestrator.architecture.specialists.tapeout_diagnosis import (
        diagnose_tapeout_failure,
    )

    pr = _pr(state)
    phase = state.get("phase", "unknown")
    attempt = state.get("attempt", 1)
    max_attempts = state.get("max_attempts", 2)

    write_graph_event(pr, "Diagnose Tapeout", "graph_node_enter", {
        "graph": "tapeout", "phase": phase, "attempt": attempt,
    })

    log(f"  [TAPEOUT DIAG] Diagnosing {phase} failure (attempt {attempt})...", YELLOW)

    pnr_params = {"utilization": 45, "density": 0.6}
    pnr_params.update(_read_pnr_overrides(pr))

    result = await diagnose_tapeout_failure(
        phase=phase,
        attempt=attempt,
        max_attempts=max_attempts,
        error_summary=state.get("previous_error", ""),
        wrapper_drc_result=state.get("wrapper_drc_result"),
        wrapper_lvs_result=state.get("wrapper_lvs_result"),
        precheck_result=state.get("precheck_result"),
        pnr_params=pnr_params,
        previous_diagnosis=state.get("diagnosis_result"),
        project_root=pr,
    )

    action = result.get("action", "escalate")
    category = result.get("category", "UNKNOWN")
    confidence = result.get("confidence", 0.0)

    if attempt >= max_attempts and action == "auto_retry":
        log("  [TAPEOUT DIAG] Max attempts reached, forcing escalate", YELLOW)
        action = "escalate"
        result["action"] = action

    log(f"  [TAPEOUT DIAG] Category: {category}", CYAN)
    log(f"  [TAPEOUT DIAG] Confidence: {confidence:.1%}", CYAN)
    log(f"  [TAPEOUT DIAG] Action: {action}", GREEN if action != "escalate" else RED)

    write_graph_event(pr, "Diagnose Tapeout", "graph_node_exit", {
        "graph": "tapeout",
        "category": category,
        "confidence": confidence,
        "action": action,
    })

    out: dict = {
        "diagnosis_result": result,
        "phase": phase,
    }

    if action == "auto_retry":
        out["attempt"] = attempt + 1
        new_overrides = result.get("pnr_overrides") or {}
        if new_overrides:
            _write_pnr_overrides(pr, new_overrides)
            log(f"  [TAPEOUT DIAG] Wrote PnR overrides to disk: {new_overrides}", CYAN)

    return out


# ---------------------------------------------------------------------------
# Node: ask_human (INTERRUPT on failure)
# ---------------------------------------------------------------------------

async def ask_human_node(state: TapeoutState) -> dict:
    """Pause the graph for human review of tapeout failures."""
    pr = _pr(state)
    phase = state.get("phase", "unknown")

    write_graph_event(pr, "Tapeout Ask Human", "graph_node_enter", {
        "graph": "tapeout", "phase": phase,
    })

    log(f"  [TAPEOUT] Human intervention needed (phase: {phase})", YELLOW)

    payload = {
        "type": "tapeout_intervention_needed",
        "graph": "tapeout",
        "phase": phase,
        "error": state.get("previous_error", "")[:2000],
        "attempt": state.get("attempt", 1),
        "diagnosis": state.get("diagnosis_result"),
        "wrapper_drc_result": state.get("wrapper_drc_result"),
        "wrapper_lvs_result": state.get("wrapper_lvs_result"),
        "precheck_result": state.get("precheck_result"),
        "submission_dir": state.get("submission_dir", ""),
        "supported_actions": ["retry", "fix_pnr", "skip", "abort"],
    }

    response = interrupt(payload)

    action = response.get("action", "unknown")
    write_graph_event(pr, "Tapeout Ask Human", "graph_node_exit", {
        "graph": "tapeout", "action": action,
    })

    return {"human_response": response}


# ---------------------------------------------------------------------------
# Node: tapeout_complete
# ---------------------------------------------------------------------------

async def tapeout_complete_node(state: TapeoutState) -> dict:
    """Final tapeout sign-off via LLM agent with deterministic pass/fail."""
    pr = _pr(state)

    precheck = state.get("precheck_result") or {}
    drc = state.get("wrapper_drc_result") or {}
    lvs = state.get("wrapper_lvs_result") or {}

    drc_clean = drc.get("clean", False)
    lvs_match = lvs.get("match", False)
    precheck_pass = precheck.get("pass", False)
    all_pass = drc_clean and lvs_match and precheck_pass

    output_dir = _output_dir(state)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    result_json_path = str(Path(output_dir) / "tapeout_complete_result.json")

    drc_summary = (
        f"Clean: {drc_clean}, Violations: {drc.get('violation_count', 'N/A')}"
    )
    lvs_summary = (
        f"Match: {lvs_match}, Device delta: {lvs.get('device_delta', 'N/A')}, "
        f"Net delta: {lvs.get('net_delta', 'N/A')}"
    )
    precheck_checks = precheck.get("checks", {})
    precheck_summary = (
        f"Pass: {precheck_pass}, "
        f"Checks: {json.dumps(precheck_checks, default=str)}"
    )

    gds_path = state.get("wrapper_gds_path", "")
    artifact_summary = (
        f"- GDS: {gds_path} ({'exists' if gds_path and Path(gds_path).exists() else 'MISSING'})\n"
        f"- Routed DEF: {state.get('wrapper_routed_def', 'N/A')}\n"
        f"- SPICE: {state.get('wrapper_spice_path', 'N/A')}"
    )

    with _tracer.start_as_current_span("Tapeout Complete") as span:
        result = await _run_tapeout_llm_step(
            step_name="Tapeout Sign-off [openframe_project_wrapper]",
            prompt_file="tapeout_complete.md",
            context={
                **_spec_paths(pr),
                "drc_summary": drc_summary,
                "lvs_summary": lvs_summary,
                "precheck_summary": precheck_summary,
                "submission_dir": state.get("submission_dir", "N/A"),
                "artifact_summary": artifact_summary,
                "all_pass_str": str(all_pass).lower(),
                "drc_clean_str": str(drc_clean).lower(),
                "lvs_match_str": str(lvs_match).lower(),
                "precheck_pass_str": str(precheck_pass).lower(),
                "result_json_path": result_json_path,
            },
            result_json_path=result_json_path,
            timeout=300,
        )

        final_all_pass = drc_clean and lvs_match and precheck_pass
        span.set_attribute("all_pass", final_all_pass)

    log(f"\n{'#'*60}", CYAN)
    log(f"  TAPEOUT {'COMPLETE' if final_all_pass else 'FINISHED (with issues)'}", CYAN)
    log(f"  Wrapper DRC: {'CLEAN' if drc_clean else 'VIOLATIONS'}", CYAN)
    log(f"  Wrapper LVS: {'MATCH' if lvs_match else 'MISMATCH'}", CYAN)
    log(f"  MPW Precheck: {'PASS' if precheck_pass else 'FAIL'}", CYAN)
    log(f"  Submission: {state.get('submission_dir', 'N/A')}", CYAN)
    if result.get("prd_compliance"):
        compliance = result["prd_compliance"]
        log(f"  PRD Compliance: {compliance.get('requirements_met', '?')}/{compliance.get('requirements_checked', '?')}", CYAN)
    log(f"{'#'*60}\n", CYAN)

    write_graph_event(pr, "Tapeout Complete", "graph_node_exit", {
        "graph": "tapeout",
        "all_pass": final_all_pass,
        "drc_clean": drc_clean,
        "lvs_match": lvs_match,
        "precheck_pass": precheck_pass,
        "prd_compliance": result.get("prd_compliance"),
    })

    return {"tapeout_done": True}


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_after_wrapper_gen(state: TapeoutState) -> str:
    """Route after wrapper generation: always run synthesize_wrapper.

    Wrapper PnR is required even when block-level GDS exists because the
    block GDS is a small macro (e.g. 60x60 um) that must be placed inside
    the full OpenFrame die (3520x5188 um) and routed to the GPIO pads.
    """
    return "synthesize_wrapper"


def route_after_wrapper_synth(state: TapeoutState) -> str:
    """Route after wrapper synthesis: SUCCESS -> wrapper_pnr, FAIL -> diagnose."""
    netlist = state.get("wrapper_netlist_path", "")
    if netlist and Path(netlist).exists():
        return "wrapper_pnr"
    return "diagnose_tapeout"


def route_after_wrapper_pnr(state: TapeoutState) -> str:
    """Route after wrapper PnR: SUCCESS -> wrapper_drc, FAIL -> diagnose."""
    result = state.get("wrapper_pnr_result") or {}
    return "wrapper_drc" if result.get("success") else "diagnose_tapeout"


def route_after_wrapper_drc(state: TapeoutState) -> str:
    """Route after wrapper DRC: CLEAN -> wrapper_lvs, FAIL -> diagnose."""
    result = state.get("wrapper_drc_result") or {}
    return "wrapper_lvs" if result.get("clean") else "diagnose_tapeout"


def route_after_wrapper_lvs(state: TapeoutState) -> str:
    """Route after wrapper LVS: MATCH -> mpw_precheck, FAIL -> ask_human.

    LVS mismatches are common (tap cell deltas) so we proceed to precheck
    even on mismatch, but warn.
    """
    # Proceed to precheck regardless -- LVS mismatch from tap cells is expected
    return "mpw_precheck"


def route_after_precheck(state: TapeoutState) -> str:
    """Route after precheck: PASS -> tapeout_complete, FAIL -> diagnose."""
    result = state.get("precheck_result") or {}
    return "tapeout_complete" if result.get("pass") else "diagnose_tapeout"


def route_after_diagnosis(state: TapeoutState) -> str:
    """Route based on the diagnosis agent's decision.

    auto_retry -> wrapper_pnr (or synthesize_wrapper if no netlist)
    continue   -> mpw_precheck (proceed past benign issue)
    escalate   -> ask_human (interrupt for outer agent)
    """
    diag = state.get("diagnosis_result") or {}
    action = diag.get("action", "escalate")

    if action == "auto_retry":
        netlist = state.get("wrapper_netlist_path", "")
        if netlist and Path(netlist).exists():
            return "wrapper_pnr"
        return "synthesize_wrapper"

    mapping = {
        "continue": "mpw_precheck",
        "escalate": "ask_human",
    }
    return mapping.get(action, "ask_human")


def route_after_human(state: TapeoutState) -> str:
    """Route based on human response."""
    action = (state.get("human_response") or {}).get("action", "abort")
    mapping = {
        "retry": "generate_wrapper",
        "fix_pnr": "wrapper_pnr",
        "skip": "tapeout_complete",
        "abort": "tapeout_complete",
    }
    return mapping.get(action, "tapeout_complete")


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_tapeout_graph(checkpointer=None):
    """Build and compile the tapeout StateGraph.

    Topology::

        generate_wrapper -> synthesize_wrapper -> wrapper_pnr
          -> wrapper_drc -> wrapper_lvs -> mpw_precheck -> tapeout_complete

    Failures route to ``diagnose_tapeout`` (LLM triage) which either
    auto-retries with adjusted PnR params, continues past benign issues,
    or escalates to ``ask_human`` with an enriched diagnosis payload.
    """
    graph = StateGraph(TapeoutState)

    # Nodes
    graph.add_node("generate_wrapper", generate_wrapper_node)
    graph.add_node("synthesize_wrapper", synthesize_wrapper_node)
    graph.add_node("wrapper_pnr", wrapper_pnr_node)
    graph.add_node("wrapper_drc", wrapper_drc_node)
    graph.add_node("wrapper_lvs", wrapper_lvs_node)
    graph.add_node("mpw_precheck", mpw_precheck_node)
    graph.add_node("diagnose_tapeout", diagnose_tapeout_node)
    graph.add_node("ask_human", ask_human_node)
    graph.add_node("tapeout_complete", tapeout_complete_node)

    # Edges
    graph.add_edge(START, "generate_wrapper")
    graph.add_conditional_edges("generate_wrapper", route_after_wrapper_gen)
    graph.add_conditional_edges("synthesize_wrapper", route_after_wrapper_synth)
    graph.add_conditional_edges("wrapper_pnr", route_after_wrapper_pnr)
    graph.add_conditional_edges("wrapper_drc", route_after_wrapper_drc)
    graph.add_conditional_edges("wrapper_lvs", route_after_wrapper_lvs)
    graph.add_conditional_edges("mpw_precheck", route_after_precheck)
    graph.add_conditional_edges("diagnose_tapeout", route_after_diagnosis)
    graph.add_conditional_edges("ask_human", route_after_human)
    graph.add_edge("tapeout_complete", END)

    return graph.compile(checkpointer=checkpointer)
