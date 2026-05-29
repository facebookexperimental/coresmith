# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
LangGraph StateGraph for the ASIC backend (physical design) pipeline.

Backend Lead architecture: operates on the flat integrated design rather
than iterating over individual blocks.

Flow:
  init_design -> flat_top_synthesis -> run_pnr -> DRC -> LVS ->
  timing_signoff -> generate_wrapper -> mpw_precheck -> advance_block ->
  backend_complete -> generate_3d_view -> final_report -> END

The ``init_design`` node discovers the integration top-level RTL and all
block RTL files.  ``flat_top_synthesis`` runs Yosys on the flat top-level
design (all blocks in one synthesis run).  The rest of the flow operates
on the resulting flat netlist.

Each EDA node uses an LLM agent (``BackendEDAAgent``) to review and adapt
the TCL/script before executing the EDA tool.  The LLM receives a
template-generated baseline script plus design context, prior failures,
and constraints, and returns a modified script optimized for the design.

Only the ``ask_human`` node uses ``interrupt()`` to pause the graph and
surface failures to the outer agent (Claude Code via MCP tools).

Usage::

    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    async with AsyncSqliteSaver.from_conn_string(".coresmith/backend_checkpoint.db") as cp:
        graph = build_backend_graph(checkpointer=cp)
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
from langgraph.types import interrupt
from opentelemetry import trace

from orchestrator.langgraph.event_stream import write_graph_event
from orchestrator.langgraph.pipeline_helpers import (
    PROJECT_ROOT,
    log,
    CYAN,
    GREEN,
    RED,
    YELLOW,
)

_tracer = trace.get_tracer("coresmith.langgraph.backend_graph")


def _eda_timeout(env: str, default: int) -> int:
    """Per-stage EDA-step timeout (seconds), overridable by env var.

    Large designs (≳300K cells) routinely need longer than the historic
    defaults for PnR and signoff: detailed routing alone can run well past
    30 min. Gate the ceiling on an env var so big-design runs don't get
    their OpenROAD child killed mid-route while small designs keep the
    snappy default.
    """
    try:
        v = int(os.environ.get(env, "").strip())
        return v if v > 0 else default
    except (ValueError, AttributeError):
        return default


def _last(a, b):
    """Reducer that keeps the latest value."""
    return b


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class BackendState(TypedDict):
    """Full backend state: Backend Lead flat-design physical flow."""

    # Config (set once) ─────────────────────────────────────────────────────
    project_root: str
    target_clock_mhz: float
    max_attempts: int
    block_queue: list[dict]

    # Backend Lead fields (set by init_design, consumed downstream) ────────
    frontend_blocks: list[dict]           # completed blocks from pipeline
    architecture_connections: list[dict]  # block diagram connections
    design_name: str                      # e.g. "h264_encoder_top"
    block_rtl_paths: dict                 # {block_name: rtl_path}
    glue_blocks: list[dict]              # detected glue block needs
    integration_top_path: str            # rtl/integration/<design>_top.v
    flat_netlist_path: str               # syn/output/<design>/<design>_netlist.v
    flat_sdc_path: str                   # syn/output/<design>/<design>.sdc
    synth_gate_count: int
    synth_area_um2: float

    # Current block tracking ────────────────────────────────────────────────
    current_block_index: int
    current_block: dict
    attempt: int
    phase: str  # "init" | "synth" | "pnr" | "drc" | "lvs" | "signoff"

    # Per-block state (plain fields, reset by init_block) ──────────────────
    constraints: list[dict]
    attempt_history: list[dict]
    previous_error: str

    # Phase results (overwritten each cycle) ───────────────────────────────
    # run_pnr produces floorplan+place+cts+route+timing+power in one shot
    floorplan_result: Optional[dict]
    place_result: Optional[dict]
    cts_result: Optional[dict]
    route_result: Optional[dict]
    drc_result: Optional[dict]
    lvs_result: Optional[dict]
    timing_result: Optional[dict]
    power_result: Optional[dict]
    debug_result: Optional[dict]
    precheck_result: Optional[dict]

    # Artifact paths (set by run_pnr, consumed by drc/lvs)
    routed_def_path: Annotated[str, _last]
    pnr_verilog_path: Annotated[str, _last]
    pwr_verilog_path: Annotated[str, _last]
    spef_path: Annotated[str, _last]
    gds_path: Annotated[str, _last]
    spice_path: Annotated[str, _last]

    # Tapeout wrapper (generated before MPW precheck) ───────────────────
    wrapper_rtl_path: Annotated[str, _last]
    wrapper_result: Optional[dict]
    submission_dir: Annotated[str, _last]

    # Step log file paths ──────────────────────────────────────────────────
    step_log_paths: Annotated[dict, _last]

    # Global accumulators (reducer) ────────────────────────────────────────
    completed_blocks: Annotated[list[dict], operator.add]

    # Human interaction ────────────────────────────────────────────────────
    human_response: Optional[dict]

    # Terminal ─────────────────────────────────────────────────────────────
    backend_done: bool

    # 3D viewer / 2D layout / final report ────────────────────────────────
    viewer_3d_path: Annotated[str, _last]
    layout_2d_png_path: Annotated[str, _last]
    final_report_path: Annotated[str, _last]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROMPT_DIR = Path(__file__).resolve().parent.parent / "langchain" / "prompts"


async def _run_llm_eda_step(
    step_name: str,
    prompt_file: str,
    context: dict,
    result_json_path: str,
    timeout: int = 1200,
) -> dict:
    """Run an EDA step entirely within the inner Claude LLM.

    The LLM has Bash/Write/Read tool access (via ClaudeLLM with
    disable_tools=False). It writes, runs, and debugs EDA tool
    scripts autonomously, then writes a structured result JSON file.

    Args:
        step_name: Human label for logging/tracing (e.g. "Flat Top Synthesis").
        prompt_file: Filename in orchestrator/langchain/prompts/ (e.g. "backend_synth_llm.md").
        context: Dict of template variables to fill into the prompt.
        result_json_path: Path where the LLM must write the result JSON.
        timeout: Max seconds for the LLM call.

    Returns:
        Parsed result dict from the JSON file, or a failure dict.
    """
    from orchestrator.langchain.agents.coresmith_llm import DEFAULT_MODEL, ClaudeLLM

    prompt_path = _PROMPT_DIR / prompt_file
    system_prompt = prompt_path.read_text().format(**context)

    user_message = (
        f"Execute the {step_name} step as described in the system prompt.\n"
        f"Write the result JSON to: {result_json_path}\n"
        f"After writing the result file, respond with a brief summary."
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

    result_path = Path(result_json_path)
    if result_path.exists():
        try:
            return json.loads(result_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    return {"success": False, "error": f"LLM did not write result JSON to {result_json_path}"}


def _block_name(state: BackendState) -> str:
    block = state.get("current_block")
    if block:
        return block.get("name", "unknown")
    return "unknown"


def _pr(state: BackendState) -> str:
    return state.get("project_root", str(PROJECT_ROOT))


def _output_dir(state: BackendState) -> str:
    """Return the PnR output directory for the current block."""
    block_name = _block_name(state)
    return str(Path(state["project_root"]) / "syn" / "output" / block_name / "pnr")


def _resolve_netlist(state: BackendState) -> tuple[str, str]:
    """Resolve netlist and SDC paths for the current block.

    Priority order:
      0. Flat netlist from state (Backend Lead path)
      1. Frontend per-block synthesis output
      2. Block spec rtl_target

    Returns (netlist_path, sdc_path).
    """
    # Priority 0: flat netlist from Backend Lead synthesis
    flat_net = state.get("flat_netlist_path", "")
    flat_sdc = state.get("flat_sdc_path", "")
    if flat_net and Path(flat_net).exists():
        return flat_net, flat_sdc if flat_sdc and Path(flat_sdc).exists() else ""

    block = state["current_block"]
    block_name = block["name"]
    root = Path(state["project_root"])

    # Priority 1: frontend synthesis output
    synth_dir = root / "syn" / "output" / block_name
    netlist = synth_dir / f"{block_name}_netlist.v"
    sdc = synth_dir / f"{block_name}.sdc"

    if netlist.exists() and sdc.exists():
        return str(netlist), str(sdc)

    # Priority 2: block spec rtl_target (gate-level netlist)
    rtl_target = block.get("rtl_target", "")
    if rtl_target:
        rtl_path = root / rtl_target
        if rtl_path.exists():
            # Generate a default SDC if missing
            if not sdc.exists():
                synth_dir.mkdir(parents=True, exist_ok=True)
                period_ns = 1000.0 / state.get("target_clock_mhz", 50.0)
                sdc.write_text(
                    f"create_clock -name clk -period {period_ns} [get_ports clk]\n"
                    f"set_input_delay {period_ns * 0.2:.1f} -clock clk [all_inputs]\n"
                    f"set_output_delay {period_ns * 0.2:.1f} -clock clk [all_outputs]\n"
                )
            return str(rtl_path), str(sdc)

    return "", ""


# ---------------------------------------------------------------------------
# Node: init_design  (Backend Lead -- discovers flat integration top)
# ---------------------------------------------------------------------------

async def init_design_node(state: BackendState) -> dict:
    """Discover the integration top-level RTL and all block RTL files.

    Sets ``current_block`` to a synthetic block representing the flat design
    for legacy compatibility with downstream nodes.
    """
    from orchestrator.langgraph.integration_helpers import discover_block_rtl

    pr = _pr(state)
    root = Path(pr)
    design_name = state.get("design_name", "chip_top")
    frontend_blocks = state.get("frontend_blocks") or state.get("block_queue", [])

    write_graph_event(pr, "Init Design", "graph_node_enter", {
        "design_name": design_name, "graph": "backend",
    })

    with _tracer.start_as_current_span(f"Init Design [{design_name}]") as span:
        span.set_attribute("design_name", design_name)

    # Discover all block RTL (source + glue)
    block_rtl = discover_block_rtl(pr, frontend_blocks)

    # Find integration top-level RTL and extract actual module name
    integration_dir = root / "rtl" / "integration"
    integration_top = ""
    if integration_dir.is_dir():
        for f in sorted(integration_dir.glob("*.v")):
            integration_top = str(f)
            # Extract actual module name from the Verilog file so we
            # don't rely on the sanitized PRD title (which produces
            # mangled names like prd___16_point_..._top).
            try:
                _src = f.read_text(encoding="utf-8", errors="replace")
                _mm = re.search(r'^\s*module\s+(\w+)', _src, re.MULTILINE)
                if _mm:
                    design_name = _mm.group(1)
            except OSError:
                pass
            break

    # Single-block designs now always have an integration top-level wrapper
    # generated by integration_check_node, so no special bypass is needed.
    # The flat_top_synthesis_node will synthesize the wrapper + block together.

    # Also pick up glue block .v files from the integration dir
    if integration_dir.is_dir():
        for f in integration_dir.glob("*.v"):
            stem = f.stem
            if stem not in block_rtl and str(f) != integration_top:
                block_rtl[stem] = str(f)

    log(f"\n{'='*60}", CYAN)
    log(f"  Backend Lead: {design_name}", CYAN)
    log(f"  Integration top: {integration_top or '(not found)'}", CYAN)
    log(f"  Block RTL files: {len(block_rtl)}", CYAN)
    log(f"{'='*60}", CYAN)

    out: dict = {
        "current_block": {"name": design_name},
        "integration_top_path": integration_top,
        "block_rtl_paths": block_rtl,
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
        "precheck_result": None,
        "human_response": None,
        "routed_def_path": "",
        "pnr_verilog_path": "",
        "pwr_verilog_path": "",
        "spef_path": "",
        "gds_path": "",
        "spice_path": "",
        "step_log_paths": {},
    }

    if not integration_top:
        out["previous_error"] = (
            f"No integration top-level RTL found in {integration_dir}. "
            "Run the frontend pipeline integration_check first."
        )

    write_graph_event(pr, "Init Design", "graph_node_exit", {
        "design_name": design_name,
        "integration_top": integration_top,
        "block_count": len(block_rtl),
        "graph": "backend",
    })

    return out


def _format_constraints(state: BackendState) -> str:
    """Format constraints list for LLM prompts."""
    constraints = state.get("constraints", [])
    if not constraints:
        return "None"
    return "\n".join(
        f"- {c.get('rule', str(c))}" for c in constraints
    )


# ---------------------------------------------------------------------------
# Node: flat_top_synthesis  (LLM-driven Yosys synthesis)
# ---------------------------------------------------------------------------

async def flat_top_synthesis_node(state: BackendState) -> dict:
    """Run Yosys synthesis entirely within the inner Claude LLM.

    The LLM writes, executes, and debugs the Yosys script autonomously.
    Skips if ``flat_netlist_path`` is already populated (single-block path).
    """
    from orchestrator.langgraph.backend_helpers import LIBERTY

    pr = _pr(state)
    design_name = state.get("design_name", _block_name(state))

    existing_netlist = state.get("flat_netlist_path", "")
    if existing_netlist and Path(existing_netlist).exists():
        log(f"  [FLAT-SYNTH] Using existing netlist: {existing_netlist}", GREEN)
        write_graph_event(pr, "Flat Top Synthesis", "graph_node_exit", {
            "design_name": design_name, "skipped": True, "graph": "backend",
        })
        return {"phase": "synth"}

    integration_top = state.get("integration_top_path", "")
    block_rtl = state.get("block_rtl_paths", {})

    write_graph_event(pr, "Flat Top Synthesis", "graph_node_enter", {
        "design_name": design_name, "graph": "backend",
    })

    if not integration_top or not Path(integration_top).exists():
        error_msg = f"No integration top-level RTL for flat synthesis: {integration_top}"
        log(f"  [FLAT-SYNTH] FAILED: {error_msg}", RED)
        write_graph_event(pr, "Flat Top Synthesis", "graph_node_exit", {
            "design_name": design_name, "success": False, "graph": "backend",
        })
        return {
            "phase": "synth",
            "previous_error": error_msg,
            "flat_netlist_path": "",
            "flat_sdc_path": "",
        }

    target_clock = state.get("target_clock_mhz", 50.0)
    period_ns = 1000.0 / target_clock
    output_dir = str(Path(pr) / "syn" / "output" / design_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    result_json_path = str(Path(output_dir) / "synth_result.json")

    input_lines = [f"- Top-level: `{integration_top}`"]
    for bname, bpath in block_rtl.items():
        if bpath != integration_top and Path(bpath).exists():
            input_lines.append(f"- Block `{bname}`: `{bpath}`")

    with _tracer.start_as_current_span(f"Flat Top Synthesis [{design_name}]") as span:
        span.set_attribute("design_name", design_name)

        result = await _run_llm_eda_step(
            step_name=f"Flat Top Synthesis [{design_name}]",
            prompt_file="backend_synth_llm.md",
            context={
                "design_name": design_name,
                "target_clock_mhz": target_clock,
                "period_ns": period_ns,
                "liberty_path": str(LIBERTY),
                "output_dir": output_dir,
                "input_files": "\n".join(input_lines),
                "input_delay_ns": period_ns * 0.2,
                "output_delay_ns": period_ns * 0.2,
                "attempt": state.get("attempt", 1),
                "prior_failure": state.get("previous_error", "None"),
                "constraints": _format_constraints(state),
                "result_json_path": result_json_path,
            },
            result_json_path=result_json_path,
        )

        span.set_attribute("success", result.get("success", False))
        if result.get("success"):
            span.set_attribute("gate_count", result.get("gate_count", 0))

    write_graph_event(pr, "Flat Top Synthesis", "graph_node_exit", {
        "design_name": design_name,
        "success": result.get("success", False),
        "gate_count": result.get("gate_count", 0),
        "graph": "backend",
    })

    if result.get("success"):
        return {
            "phase": "synth",
            "flat_netlist_path": result.get("netlist_path", ""),
            "flat_sdc_path": result.get("sdc_path", ""),
            "synth_gate_count": result.get("gate_count", 0),
            "synth_area_um2": result.get("area_um2", 0.0),
        }
    else:
        return {
            "phase": "synth",
            "previous_error": result.get("error", "Flat synthesis failed"),
            "flat_netlist_path": "",
            "flat_sdc_path": "",
        }


def route_after_flat_synth(state: BackendState) -> str:
    """Route after flat synthesis: success -> run_pnr, fail -> diagnose."""
    netlist = state.get("flat_netlist_path", "")
    if netlist and Path(netlist).exists():
        return "run_pnr"
    return "diagnose"


route_after_flat_synth.__edge_labels__ = {
    "run_pnr": "SUCCESS",
    "diagnose": "FAIL",
}


# ---------------------------------------------------------------------------
# Node: run_pnr  (LLM-driven OpenROAD PnR)
# ---------------------------------------------------------------------------

async def run_pnr_node(state: BackendState) -> dict:
    """Run OpenROAD PnR entirely within the inner Claude LLM.

    The LLM writes, executes, and debugs the OpenROAD TCL script
    autonomously -- handling floorplan, placement, CTS, routing,
    and timing analysis in a single LLM session.
    """
    from orchestrator.langgraph.backend_helpers import (
        TECH_LEF, CELL_LEF, LIBERTY, OPENROAD_BIN,
        render_layout_image,
    )

    block = state["current_block"]
    block_name = block["name"]
    attempt = state["attempt"]

    write_graph_event(_pr(state), "Run PnR", "graph_node_enter", {
        "block": block_name, "attempt": attempt, "graph": "backend",
    })

    netlist_path, sdc_path = _resolve_netlist(state)

    if not netlist_path:
        error_msg = (
            f"No synthesized netlist found for {block_name}. "
            f"Checked: syn/output/{block_name}/{block_name}_netlist.v"
        )
        log(f"  [PNR] FAILED: {error_msg}", RED)
        write_graph_event(_pr(state), "Run PnR", "graph_node_exit", {
            "block": block_name, "success": False, "error": error_msg,
            "graph": "backend",
        })
        fail_result = {"success": False, "error": error_msg}
        return {
            "floorplan_result": fail_result,
            "place_result": fail_result,
            "cts_result": fail_result,
            "route_result": fail_result,
            "timing_result": {"met": False, "error": error_msg},
            "power_result": fail_result,
            "phase": "pnr",
            "previous_error": error_msg,
        }

    output_dir = _output_dir(state)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    target_clock = state.get("target_clock_mhz", 50.0)
    period_ns = 1000.0 / target_clock
    result_json_path = str(Path(output_dir) / "pnr_result.json")

    utilization = 35
    density = 0.6
    margin = 10
    overrides_path = Path(_pr(state)) / ".coresmith" / "pnr_overrides.json"
    if overrides_path.exists():
        try:
            overrides = json.loads(overrides_path.read_text())
            utilization = overrides.get("utilization", utilization)
            density = overrides.get("density", density)
            margin = overrides.get("margin", margin)
        except (json.JSONDecodeError, OSError):
            pass

    gate_count = state.get("synth_gate_count", 0)

    # Prepare a working copy of the reference PnR TCL template with
    # design-specific variables substituted. The LLM agent can then
    # read, modify, and iterate on this script.
    from orchestrator.langgraph.backend_helpers import prepare_pnr_working_copy
    tcl_path = prepare_pnr_working_copy(
        design_name=block_name,
        netlist_path=netlist_path,
        sdc_path=sdc_path,
        output_dir=output_dir,
        utilization=utilization,
        density=density,
    )

    with _tracer.start_as_current_span(
        f"Run PnR [{block_name}] attempt {attempt}"
    ) as span:
        span.set_attribute("block_name", block_name)
        span.set_attribute("attempt", attempt)

        result = await _run_llm_eda_step(
            step_name=f"Run PnR [{block_name}]",
            prompt_file="backend_pnr_llm.md",
            context={
                "design_name": block_name,
                "target_clock_mhz": target_clock,
                "period_ns": period_ns,
                "gate_count": gate_count,
                "tech_lef": str(TECH_LEF),
                "cell_lef": str(CELL_LEF),
                "liberty_path": str(LIBERTY),
                "openroad_bin": str(OPENROAD_BIN),
                "netlist_path": netlist_path,
                "sdc_path": sdc_path,
                "output_dir": output_dir,
                "utilization": utilization,
                "density": density,
                "margin": margin,
                "attempt": attempt,
                "max_attempts": state.get("max_attempts", 3),
                "prior_failure": state.get("previous_error", "None"),
                "constraints": _format_constraints(state),
                "result_json_path": result_json_path,
                "tcl_path": tcl_path,
            },
            result_json_path=result_json_path,
            timeout=_eda_timeout("CORESMITH_PNR_TIMEOUT", 1800),
        )

        pnr_ok = result.get("success", False)
        span.set_attribute("success", pnr_ok)
        span.set_attribute("design_area_um2", result.get("design_area_um2", 0))
        span.set_attribute("wns_ns", result.get("wns_ns", 0))

    routed_def = result.get("routed_def_path", str(Path(output_dir) / f"{block_name}_routed.def"))
    pnr_verilog = result.get("pnr_verilog_path", str(Path(output_dir) / f"{block_name}_pnr.v"))
    pwr_verilog = result.get("pwr_verilog_path", str(Path(output_dir) / f"{block_name}_pwr.v"))
    spef = result.get("spef_path", str(Path(output_dir) / f"{block_name}.spef"))

    if pnr_ok:
        img_dir = Path(_pr(state)) / ".coresmith" / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        render_layout_image(routed_def, str(img_dir / f"{block_name}_floorplan.png"))

    wns = result.get("wns_ns", 0)
    timing_met = wns >= 0 if isinstance(wns, (int, float)) else False

    write_graph_event(_pr(state), "Run PnR", "graph_node_exit", {
        "block": block_name,
        "success": pnr_ok,
        "design_area_um2": result.get("design_area_um2", 0),
        "wns_ns": wns,
        "total_power_mw": result.get("total_power_mw", 0),
        "graph": "backend",
    })

    if not pnr_ok:
        error = result.get("error", "PnR failed")
        fail_result = {"success": False, "error": error[-1000:]}
        return {
            "floorplan_result": fail_result,
            "place_result": fail_result,
            "cts_result": fail_result,
            "route_result": fail_result,
            "timing_result": {"met": False, "error": error[-1000:]},
            "power_result": fail_result,
            "phase": "pnr",
            "previous_error": error[-3000:],
        }

    return {
        "floorplan_result": {"success": True, "design_area_um2": result.get("design_area_um2", 0)},
        "place_result": {"success": True},
        "cts_result": {"success": True},
        "route_result": {
            "success": True,
            "wire_length_um": result.get("wire_length_um", 0),
            "via_count": result.get("via_count", 0),
        },
        "timing_result": {"met": timing_met, "wns_ns": wns, "tns_ns": result.get("tns_ns", 0)},
        "power_result": {"success": True, "total_power_mw": result.get("total_power_mw", 0)},
        "phase": "pnr",
        "routed_def_path": routed_def,
        "pnr_verilog_path": pnr_verilog,
        "pwr_verilog_path": pwr_verilog,
        "spef_path": spef,
    }


# ---------------------------------------------------------------------------
# Node: drc (LLM-driven Magic DRC + GDS + SPICE)
# ---------------------------------------------------------------------------

async def drc_node(state: BackendState) -> dict:
    """Run Magic DRC, GDS generation, and SPICE extraction entirely
    within the inner Claude LLM."""
    from orchestrator.langgraph.backend_helpers import (
        MAGIC_RC,
        CELL_GDS,
        MAGIC_BIN,
        parse_drc_report,
        render_layout_image,
    )

    block = state["current_block"]
    block_name = block["name"]

    route_result = state.get("route_result") or {}
    if not route_result.get("success"):
        error_msg = route_result.get("error", "PnR/routing failed")
        return {"drc_result": {"clean": False, "errors": error_msg}, "phase": "drc", "previous_error": error_msg}

    routed_def = state.get("routed_def_path", "")
    if not routed_def or not Path(routed_def).exists():
        error_msg = f"Routed DEF not found: {routed_def}"
        return {"drc_result": {"clean": False, "errors": error_msg}, "phase": "drc", "previous_error": error_msg}

    write_graph_event(_pr(state), "DRC", "graph_node_enter", {"block": block_name, "graph": "backend"})

    output_dir = _output_dir(state)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    result_json_path = str(Path(output_dir) / "drc_result.json")

    with _tracer.start_as_current_span(f"DRC [{block_name}]") as span:
        span.set_attribute("block_name", block_name)

        result = await _run_llm_eda_step(
            step_name=f"DRC [{block_name}]",
            prompt_file="backend_drc_llm.md",
            context={
                "design_name": block_name,
                "magic_rc": str(MAGIC_RC),
                "cell_gds": str(CELL_GDS),
                "magic_bin": str(MAGIC_BIN),
                "routed_def_path": routed_def,
                "output_dir": output_dir,
                "attempt": state["attempt"],
                "prior_failure": state.get("previous_error", "None"),
                "constraints": _format_constraints(state),
                "result_json_path": result_json_path,
            },
            result_json_path=result_json_path,
        )

        drc_clean = result.get("clean", False)
        drc_count = result.get("violation_count", 0 if drc_clean else 999)
        gds_path = result.get("gds_path", "")
        spice_path = result.get("spice_path", "")
        report_path = result.get("report_path") or result.get("drc_report_path") or str(Path(output_dir) / "magic_drc.rpt")

        # The EDA agent occasionally writes contradictory JSON, for example
        # clean=false with violation_count=0 after Magic produced an empty
        # DRC report. Trust the tool artifact over the free-form summary.
        if Path(report_path).exists():
            parsed_drc = parse_drc_report(report_path)
            parsed_count = parsed_drc.get("violation_count", -1)
            if parsed_count >= 0:
                drc_count = parsed_count
                drc_clean = bool(parsed_drc.get("clean", False))

        span.set_attribute("clean", drc_clean)
        span.set_attribute("violation_count", drc_count)

    if gds_path and Path(gds_path).exists():
        img_dir = Path(_pr(state)) / ".coresmith" / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        render_layout_image(gds_path, str(img_dir / f"{block_name}_gds.png"))

    write_graph_event(_pr(state), "DRC", "graph_node_exit", {
        "block": block_name, "clean": drc_clean, "violation_count": drc_count, "graph": "backend",
    })

    out: dict = {"drc_result": {"clean": drc_clean, "violation_count": drc_count}, "phase": "drc"}
    if drc_clean:
        out["gds_path"] = gds_path
        out["spice_path"] = spice_path
    else:
        out["previous_error"] = result.get("error", f"DRC: {drc_count} violations")
    return out


# ---------------------------------------------------------------------------
# Node: lvs (LLM-driven Netgen LVS)
# ---------------------------------------------------------------------------

async def lvs_node(state: BackendState) -> dict:
    """Run Netgen LVS comparison entirely within the inner Claude LLM."""
    from orchestrator.langgraph.backend_helpers import NETGEN_SETUP, NETGEN_BIN

    block = state["current_block"]
    block_name = block["name"]

    drc_result = state.get("drc_result") or {}
    if not drc_result.get("clean"):
        error_msg = drc_result.get("errors", "DRC not clean")
        return {"lvs_result": {"match": False, "errors": error_msg}, "phase": "lvs", "previous_error": error_msg}

    spice_path = state.get("spice_path", "")
    pwr_verilog = state.get("pwr_verilog_path", "")

    if not spice_path or not Path(spice_path).exists():
        error_msg = f"SPICE file not found: {spice_path}"
        return {"lvs_result": {"match": False, "errors": error_msg}, "phase": "lvs", "previous_error": error_msg}
    if not pwr_verilog or not Path(pwr_verilog).exists():
        error_msg = f"Power-aware Verilog not found: {pwr_verilog}"
        return {"lvs_result": {"match": False, "errors": error_msg}, "phase": "lvs", "previous_error": error_msg}

    write_graph_event(_pr(state), "LVS", "graph_node_enter", {"block": block_name, "graph": "backend"})

    output_dir = _output_dir(state)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    result_json_path = str(Path(output_dir) / "lvs_result.json")

    with _tracer.start_as_current_span(f"LVS [{block_name}]") as span:
        span.set_attribute("block_name", block_name)

        result = await _run_llm_eda_step(
            step_name=f"LVS [{block_name}]",
            prompt_file="backend_lvs_llm.md",
            context={
                "design_name": block_name,
                "netgen_setup": str(NETGEN_SETUP),
                "netgen_bin": str(NETGEN_BIN),
                "spice_path": spice_path,
                "pwr_verilog_path": pwr_verilog,
                "verilog_path": pwr_verilog,
                "output_dir": output_dir,
                "attempt": state["attempt"],
                "prior_failure": state.get("previous_error", "None"),
                "constraints": _format_constraints(state),
                "result_json_path": result_json_path,
            },
            result_json_path=result_json_path,
        )

        match = result.get("match", False)
        span.set_attribute("match", match)

    write_graph_event(_pr(state), "LVS", "graph_node_exit", {
        "block": block_name, "match": match,
        "device_delta": result.get("device_delta", 0),
        "net_delta": result.get("net_delta", 0),
        "analysis": result.get("analysis", ""),
        "graph": "backend",
    })

    out: dict = {
        "lvs_result": {
            "match": match,
            "device_delta": result.get("device_delta", 0),
            "net_delta": result.get("net_delta", 0),
            "report_path": result.get("report_path", ""),
            "llm_analysis": result.get("analysis", ""),
        },
        "phase": "lvs",
    }
    if not match:
        out["previous_error"] = (
            f"LVS mismatch: device_delta={result.get('device_delta', '?')}, "
            f"net_delta={result.get('net_delta', '?')}"
        )
    return out


# ---------------------------------------------------------------------------
# Node: timing_signoff
# ---------------------------------------------------------------------------

async def timing_signoff_node(state: BackendState) -> dict:
    """LLM-assisted post-route timing sign-off analysis.

    The LLM analyzes timing results from PnR and provides expert assessment:
    whether violations are waivable, which paths are critical, and specific
    recommendations for fixing timing closure issues.
    """
    from orchestrator.langchain.agents.backend_eda_agent import BackendEDAAgent

    block = state["current_block"]
    block_name = block["name"]

    write_graph_event(_pr(state), "Timing Sign-off", "graph_node_enter", {
        "block": block_name, "graph": "backend",
    })

    timing = state.get("timing_result") or {}
    power = state.get("power_result") or {}
    floorplan = state.get("floorplan_result") or {}
    target_mhz = state.get("target_clock_mhz", 50.0)
    period_ns = 1000.0 / target_mhz

    wns = timing.get("wns_ns", 0.0)
    tns = timing.get("tns_ns", 0.0)
    setup_slack = timing.get("setup_slack_ns", 0.0)
    hold_slack = timing.get("hold_slack_ns", 0.0)

    with _tracer.start_as_current_span(f"Timing Sign-off [{block_name}]") as span:
        span.set_attribute("block_name", block_name)

        agent = BackendEDAAgent(step="timing_signoff")
        analysis = await agent.analyze(context={
            "design_name": block_name,
            "target_clock_mhz": target_mhz,
            "period_ns": f"{period_ns:.2f}",
            "gate_count": state.get("synth_gate_count", 0),
            "wns_ns": wns,
            "tns_ns": tns,
            "setup_slack_ns": setup_slack,
            "hold_slack_ns": hold_slack,
            "total_power_mw": power.get("total_power_mw", 0),
            "dynamic_power_mw": power.get("dynamic_power_mw", 0),
            "leakage_power_mw": power.get("leakage_power_mw", 0),
            "design_area_um2": (state.get("place_result") or {}).get("design_area_um2", 0),
            "die_area_um2": floorplan.get("die_area_um2", 0),
            "utilization_pct": floorplan.get("utilization", 0),
            "prior_failure": state.get("previous_error", "None"),
            "constraints": _format_constraints(state),
        })

        sign_off = analysis.get("sign_off", "FAIL")
        met = analysis.get("timing_met", wns >= 0)

        # CONDITIONAL_PASS counts as met (waivable violations)
        if sign_off == "CONDITIONAL_PASS":
            met = True
            log(f"  [STA] Timing CONDITIONAL PASS @ {target_mhz} MHz "
                f"(WNS={wns:.2f} ns) -- {analysis.get('assessment', '')}", YELLOW)
        elif met:
            log(f"  [STA] Timing met @ {target_mhz} MHz (WNS={wns:.2f} ns)", GREEN)
        else:
            log(f"  [STA] Timing VIOLATED: WNS={wns:.2f} ns, TNS={tns:.2f} ns", RED)
            if analysis.get("recommendations"):
                for rec in analysis["recommendations"]:
                    log(f"        → {rec}", YELLOW)

        span.set_attribute("timing_met", met)
        span.set_attribute("wns_ns", wns)
        span.set_attribute("sign_off", sign_off)

    write_graph_event(_pr(state), "Timing Sign-off", "graph_node_exit", {
        "block": block_name, "met": met, "wns_ns": wns, "tns_ns": tns,
        "sign_off": sign_off,
        "assessment": analysis.get("assessment", ""),
        "graph": "backend",
    })

    result = {
        "met": met,
        "wns_ns": wns,
        "tns_ns": tns,
        "setup_slack_ns": setup_slack,
        "hold_slack_ns": hold_slack,
        "max_clock_mhz": target_mhz,
        "sign_off": sign_off,
        "assessment": analysis.get("assessment", ""),
        "power_assessment": analysis.get("power_assessment", ""),
        "recommendations": analysis.get("recommendations", []),
    }

    out: dict = {"timing_result": result, "phase": "signoff"}
    if not met:
        out["previous_error"] = f"Timing violated: WNS={wns:.2f} ns"

    return out


# ---------------------------------------------------------------------------
# Node: generate_wrapper  (LLM-driven OpenFrame wrapper + submission structure)
# ---------------------------------------------------------------------------

async def generate_wrapper_node(state: BackendState) -> dict:
    """Generate OpenFrame wrapper RTL and submission directory entirely
    within the inner Claude LLM.

    Reads the design's gate-level netlist to discover ports, generates the
    openframe_project_wrapper.v, creates the submission directory structure,
    and copies all artifacts.
    """
    pr = _pr(state)
    block_name = _block_name(state)
    target_clock = state.get("target_clock_mhz", 50.0)

    write_graph_event(pr, "Generate Wrapper", "graph_node_enter", {
        "block": block_name, "graph": "backend",
    })

    submission_dir = str(Path(pr) / "openframe_submission")
    Path(submission_dir).mkdir(parents=True, exist_ok=True)
    result_json_path = str(Path(submission_dir) / "wrapper_result.json")

    pnr_verilog = state.get("pnr_verilog_path", "")
    routed_def = state.get("routed_def_path", "")
    gds_path = state.get("gds_path", "")
    spice_path = state.get("spice_path", "")
    sdc_path = state.get("flat_sdc_path", "")
    spef_path = state.get("spef_path", "")

    with _tracer.start_as_current_span(f"Generate Wrapper [{block_name}]") as span:
        span.set_attribute("block_name", block_name)

        result = await _run_llm_eda_step(
            step_name=f"Generate Wrapper [{block_name}]",
            prompt_file="backend_wrapper_llm.md",
            context={
                "design_name": block_name,
                "target_clock_mhz": target_clock,
                "gate_count": state.get("synth_gate_count", 0),
                "project_root": pr,
                "pnr_verilog_path": pnr_verilog,
                "routed_def_path": routed_def,
                "gds_path": gds_path,
                "spice_path": spice_path,
                "sdc_path": sdc_path,
                "spef_path": spef_path,
                "submission_dir": submission_dir,
                "result_json_path": result_json_path,
            },
            result_json_path=result_json_path,
            timeout=_eda_timeout("CORESMITH_WRAPPER_TIMEOUT", 600),
        )

        wrapper_ok = result.get("success", False)
        span.set_attribute("success", wrapper_ok)
        span.set_attribute("gpio_used", result.get("gpio_used", 0))

    write_graph_event(pr, "Generate Wrapper", "graph_node_exit", {
        "block": block_name,
        "success": wrapper_ok,
        "gpio_used": result.get("gpio_used", 0),
        "graph": "backend",
    })

    if wrapper_ok:
        log(f"  [WRAPPER] Generated: {result.get('wrapper_path', '')}", GREEN)
        return {
            "wrapper_result": result,
            "wrapper_rtl_path": result.get("wrapper_path", ""),
            "submission_dir": submission_dir,
            "phase": "wrapper",
        }
    else:
        error = result.get("error", "Wrapper generation failed")
        log(f"  [WRAPPER] FAILED: {error}", RED)
        return {
            "wrapper_result": result,
            "phase": "wrapper",
            "previous_error": error,
        }


def route_after_wrapper(state: BackendState) -> str:
    """Route after wrapper: success -> mpw_precheck, fail -> diagnose."""
    wr = state.get("wrapper_result") or {}
    return "mpw_precheck" if wr.get("success") else "diagnose"


route_after_wrapper.__edge_labels__ = {
    "mpw_precheck": "SUCCESS",
    "diagnose": "FAIL",
}


# ---------------------------------------------------------------------------
# Node: mpw_precheck  (LLM-assisted Efabless MPW submission precheck)
# ---------------------------------------------------------------------------

async def mpw_precheck_node(state: BackendState) -> dict:
    """Run LLM-assisted MPW precheck for shuttle submission readiness.

    Runs the native MPW precheck (directory structure, GDS validation,
    wrapper port names, KLayout/Magic DRC) and then uses an LLM to analyze
    the results and assess submission readiness.
    """
    from orchestrator.langchain.agents.backend_eda_agent import BackendEDAAgent
    from orchestrator.langgraph.tapeout_helpers import run_mpw_precheck_native

    pr = _pr(state)
    block_name = _block_name(state)
    gds_path = state.get("gds_path", "")

    write_graph_event(pr, "MPW Precheck", "graph_node_enter", {
        "block": block_name, "graph": "backend",
    })

    # Build submission directory from backend artifacts
    submission_dir = str(Path(pr) / "openframe_submission")
    sub = Path(submission_dir)
    sub.mkdir(parents=True, exist_ok=True)

    # Ensure required subdirectories exist
    for d in ("gds", "def", "verilog", "sdc", "spef"):
        (sub / d).mkdir(parents=True, exist_ok=True)

    # Copy artifacts into submission structure
    routed_def = state.get("routed_def_path", "")
    pnr_verilog = state.get("pnr_verilog_path", "")
    sdc_path = state.get("flat_sdc_path", "")
    spef_path = state.get("spef_path", "")

    import shutil
    for src, dst_dir in [
        (gds_path, sub / "gds"),
        (routed_def, sub / "def"),
        (pnr_verilog, sub / "verilog"),
        (sdc_path, sub / "sdc"),
        (spef_path, sub / "spef"),
    ]:
        if src and Path(src).exists():
            try:
                shutil.copy2(src, dst_dir)
            except (OSError, shutil.SameFileError):
                pass

    with _tracer.start_as_current_span(f"MPW Precheck [{block_name}]") as span:
        span.set_attribute("block_name", block_name)

        log("  [PRECHECK] Running native MPW precheck...", YELLOW)

        precheck_result = await asyncio.to_thread(
            run_mpw_precheck_native, submission_dir, gds_path,
        )

        span.set_attribute("pass", precheck_result.get("pass", False))

        # LLM analyzes the precheck results
        agent = BackendEDAAgent(step="mpw_precheck")
        analysis = await agent.analyze(context={
            "design_name": block_name,
            "submission_dir": submission_dir,
            "gds_path": gds_path,
            "gate_count": state.get("synth_gate_count", 0),
            "target_clock_mhz": state.get("target_clock_mhz", 50.0),
            "overall_pass": precheck_result.get("pass", False),
            "check_results": json.dumps(
                {k: v for k, v in precheck_result.get("checks", {}).items()},
                indent=2,
            ),
            "errors": "\n".join(precheck_result.get("errors", ["None"])),
            "warnings": "\n".join(precheck_result.get("warnings", ["None"])),
        })

        native_pass = precheck_result.get("pass", False)
        submission_ready = native_pass and analysis.get("submission_ready", native_pass)

        if submission_ready:
            log(f"  [PRECHECK] Submission READY: {analysis.get('assessment', '')}", GREEN)
        else:
            log(f"  [PRECHECK] NOT ready: {analysis.get('assessment', '')}", RED)
            if analysis.get("blocking_issues"):
                for issue in analysis["blocking_issues"]:
                    log(f"        ✗ {issue}", RED)
            if analysis.get("auto_fixable"):
                for fix in analysis["auto_fixable"]:
                    log(f"        ↻ {fix}", YELLOW)

    write_graph_event(pr, "MPW Precheck", "graph_node_exit", {
        "block": block_name,
        "pass": submission_ready,
        "checks": {k: v.get("pass") for k, v in precheck_result.get("checks", {}).items()},
        "assessment": analysis.get("assessment", ""),
        "graph": "backend",
    })

    out: dict = {
        "precheck_result": {
            **precheck_result,
            "llm_analysis": analysis,
        },
        "phase": "precheck",
    }

    if not submission_ready:
        out["previous_error"] = "; ".join(
            analysis.get("blocking_issues", precheck_result.get("errors", ["Precheck failed"]))
        )

    return out


def route_after_precheck(state: BackendState) -> str:
    """Route after MPW precheck: PASS -> advance_block, FAIL -> diagnose.

    Both the native precheck pass AND the LLM's ``submission_ready`` flag
    must be True to advance.  The LLM cannot override a failed native
    precheck (matches the truth table used inside ``mpw_precheck_node``).
    """
    precheck = state.get("precheck_result") or {}
    passed = precheck.get("pass", False)
    llm_analysis = precheck.get("llm_analysis") or {}
    submission_ready = llm_analysis.get("submission_ready", passed)
    if passed and submission_ready:
        return "advance_block"
    return "diagnose"


route_after_precheck.__edge_labels__ = {
    "advance_block": "PASS",
    "diagnose": "FAIL",
}


# ---------------------------------------------------------------------------
# Node: diagnose
# ---------------------------------------------------------------------------

async def diagnose_node(state: BackendState) -> dict:
    """Diagnose physical design failures using LLM-based triage.

    Uses the tapeout diagnosis agent (shared with the tapeout graph) to
    analyze DRC/LVS/PnR failures, classify root causes, and recommend
    parameter adjustments.  Can auto-retry with adjusted PnR params
    instead of always escalating to human.
    """
    from orchestrator.architecture.specialists.tapeout_diagnosis import (
        diagnose_tapeout_failure,
    )

    block = state["current_block"]
    block_name = block["name"]
    phase = state.get("phase", "unknown")

    write_graph_event(_pr(state), "Diagnose Backend", "graph_node_enter", {
        "block": block_name, "phase": phase, "graph": "backend",
    })

    error_log = state.get("previous_error", "Unknown failure")
    if phase == "drc":
        drc = state.get("drc_result") or {}
        error_log = str(drc.get("violations", drc.get("errors", error_log)))
    elif phase == "lvs":
        lvs = state.get("lvs_result") or {}
        error_log = str(lvs.get("mismatches", lvs.get("errors", error_log)))
    elif phase == "signoff":
        timing = state.get("timing_result") or {}
        error_log = f"WNS={timing.get('wns_ns', '?')} ns TNS={timing.get('tns_ns', '?')} ns"

    with _tracer.start_as_current_span(f"Diagnose Backend [{block_name}]") as span:
        span.set_attribute("block_name", block_name)
        span.set_attribute("failed_phase", phase)

        try:
            diag = await diagnose_tapeout_failure(
                phase=phase,
                attempt=state["attempt"],
                max_attempts=state["max_attempts"],
                error_summary=error_log[:2000],
                wrapper_drc_result=state.get("drc_result"),
                wrapper_lvs_result=state.get("lvs_result"),
                pnr_params={"utilization": 45, "density": 0.6},
                project_root=_pr(state),
            )
        except Exception as exc:
            log(f"  [DIAGNOSE-BACKEND] LLM diagnosis failed: {exc}", RED)
            diag = {
                "category": "BACKEND_FAILURE",
                "diagnosis": f"Diagnosis failed: {exc}. Error: {error_log[:500]}",
                "confidence": 0.2,
                "action": "escalate",
                "suggested_fix": "",
                "pnr_overrides": {},
            }

        category = diag.get("category", "BACKEND_FAILURE")
        action = diag.get("action", "escalate")
        confidence = diag.get("confidence", 0.3)

        if action == "auto_retry" and diag.get("pnr_overrides"):
            overrides_path = Path(_pr(state)) / ".coresmith" / "pnr_overrides.json"
            overrides_path.write_text(json.dumps(diag["pnr_overrides"], indent=2))
            log(f"  [DIAGNOSE-BACKEND] Auto-retry with overrides: {diag['pnr_overrides']}", YELLOW)

        span.set_attribute("category", category)
        span.set_attribute("action", action)
        span.set_attribute("confidence", confidence)

    history = list(state.get("attempt_history", []))
    history.append({
        "attempt": state["attempt"],
        "phase": phase,
        "error": error_log[:500],
        "category": category,
    })

    needs_human = action == "escalate" or confidence < 0.3
    diag_result = {
        "category": category,
        "diagnosis": diag.get("diagnosis", ""),
        "needs_human": needs_human,
        "escalate": action == "escalate",
        "suggested_fix": diag.get("suggested_fix", ""),
        "constraints": [],
        "next_action": "ask_human" if needs_human else "retry_pnr",
        "confidence": confidence,
    }

    write_graph_event(_pr(state), "Diagnose Backend", "graph_node_exit", {
        "block": block_name, "category": category,
        "action": action, "confidence": confidence,
        "graph": "backend",
    })

    return {
        "debug_result": diag_result,
        "attempt_history": history,
        "previous_error": error_log[:2000],
    }


# ---------------------------------------------------------------------------
# Node: decide
# ---------------------------------------------------------------------------

async def decide_node(state: BackendState) -> dict:
    """Route decision for backend failures: retry, human, or escalate.

    When PnR already succeeded but a downstream check (DRC/LVS/timing)
    failed, route the retry directly to that step instead of re-running
    PnR from scratch.
    """
    block = state["current_block"]
    block_name = block["name"]
    debug_result = state.get("debug_result", {})
    attempt = state["attempt"]
    max_attempts = state["max_attempts"]

    with _tracer.start_as_current_span(f"Backend Decision [{block_name}]") as span:
        span.set_attribute("block_name", block_name)
        span.set_attribute("attempt", attempt)

        if debug_result.get("escalate") and attempt < max_attempts:
            action = "ask_human"
        elif debug_result.get("escalate"):
            action = "escalate"
        elif attempt >= max_attempts:
            action = "escalate"
        elif debug_result.get("needs_human"):
            action = "ask_human"
        else:
            pnr_ok = (state.get("route_result") or {}).get("success", False)
            drc_clean = (state.get("drc_result") or {}).get("clean", False)
            lvs_match = (state.get("lvs_result") or {}).get("match", False)

            if pnr_ok and not drc_clean:
                action = "retry_drc"
            elif pnr_ok and drc_clean and not lvs_match:
                action = "retry_lvs"
            elif pnr_ok and drc_clean and lvs_match:
                action = "retry_timing"
            else:
                action = "retry_pnr"

        span.set_attribute("decision", action)

    return {
        "debug_result": {**debug_result, "next_action": action},
    }


# ---------------------------------------------------------------------------
# Node: ask_human  (INTERRUPT)
# ---------------------------------------------------------------------------

async def ask_human_node(state: BackendState) -> dict:
    """Pause the graph and surface failure details to the outer agent."""
    block = state["current_block"]
    block_name = block["name"]
    debug_result = state.get("debug_result", {})

    write_graph_event(_pr(state), "Ask Human", "graph_node_enter", {
        "block": block_name, "attempt": state["attempt"], "graph": "backend",
    })

    log(f"  [HUMAN] Backend intervention needed for {block_name}", YELLOW)

    attempt_history = state.get("attempt_history", [])
    category_counts: dict[str, int] = {}
    for entry in attempt_history:
        cat = entry.get("category", "UNKNOWN")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    payload = {
        "type": "human_intervention_needed",
        "graph": "backend",
        "block_name": block_name,
        "attempt": state["attempt"],
        "max_attempts": state.get("max_attempts", 3),
        "phase": state.get("phase", ""),
        "error": state.get("previous_error", "")[:2000],
        "diagnosis": debug_result.get("diagnosis", ""),
        "category": debug_result.get("category", ""),
        "confidence": debug_result.get("confidence", 0.3),
        "suggested_fix": debug_result.get("suggested_fix", ""),
        "needs_human": debug_result.get("needs_human", True),
        "attempt_history": attempt_history[-5:],
        "category_counts": category_counts,
        "routed_def_path": state.get("routed_def_path", ""),
        "gds_path": state.get("gds_path", ""),
        "step_log_paths": state.get("step_log_paths", {}),
        "supported_actions": [
            "retry", "skip", "abort",
        ],
    }

    response = interrupt(payload)

    write_graph_event(_pr(state), "Ask Human", "graph_node_exit", {
        "block": block_name, "action": response.get("action", "unknown"),
        "graph": "backend",
    })

    return {"human_response": response}


# ---------------------------------------------------------------------------
# Node: increment_attempt
# ---------------------------------------------------------------------------

async def increment_attempt_node(state: BackendState) -> dict:
    """Bump the attempt counter."""
    new_attempt = state["attempt"] + 1
    block_name = _block_name(state)

    with _tracer.start_as_current_span(
        f"Backend Retry #{new_attempt - 1} [{block_name}]"
    ) as span:
        span.set_attribute("block_name", block_name)
        span.set_attribute("attempt.new", new_attempt)
        span.set_attribute("max_attempts", state["max_attempts"])

    log(f"  [RETRY] Backend attempt {new_attempt}/{state['max_attempts']}", YELLOW)

    return {"attempt": new_attempt}


# ---------------------------------------------------------------------------
# Node: advance_block
# ---------------------------------------------------------------------------

async def advance_block_node(state: BackendState) -> dict:
    """Record block result and advance the index."""
    block = state["current_block"]
    block_name = block["name"]
    attempt = state["attempt"]

    drc_clean = (state.get("drc_result") or {}).get("clean", False)
    lvs_match = (state.get("lvs_result") or {}).get("match", False)
    timing_met = (state.get("timing_result") or {}).get("met", False)
    precheck = state.get("precheck_result") or {}
    precheck_ok = precheck.get("pass", False)
    all_pass = drc_clean and lvs_match and timing_met and precheck_ok

    power = state.get("power_result") or {}
    step_logs = dict(state.get("step_log_paths") or {})

    if all_pass:
        timing = state.get("timing_result") or {}
        floorplan = state.get("floorplan_result") or {}
        route = state.get("route_result") or {}
        result = {
            "name": block_name,
            "success": True,
            "attempts": attempt,
            "total_power_mw": power.get("total_power_mw", 0),
            "dynamic_power_mw": power.get("dynamic_power_mw", 0),
            "leakage_power_mw": power.get("leakage_power_mw", 0),
            "timing_wns_ns": timing.get("wns_ns", 0),
            "timing_tns_ns": timing.get("tns_ns", 0),
            "setup_slack_ns": timing.get("setup_slack_ns", 0),
            "hold_slack_ns": timing.get("hold_slack_ns", 0),
            "design_area_um2": (state.get("place_result") or {}).get("design_area_um2", 0),
            "die_area_um2": floorplan.get("die_area_um2", 0),
            "utilization_pct": floorplan.get("utilization", 0),
            "wire_length_um": route.get("wire_length_um", 0),
            "via_count": route.get("via_count", 0),
            "drc_clean": drc_clean,
            "lvs_match": lvs_match,
            "timing_met": timing_met,
            "synth_gate_count": state.get("synth_gate_count", 0),
            "gds_path": state.get("gds_path", ""),
            "routed_def_path": state.get("routed_def_path", ""),
            "spef_path": state.get("spef_path", ""),
            "constraints_learned": len(state.get("constraints", [])),
            "step_log_paths": step_logs,
        }
        log(f"  [{block_name}] BACKEND PASSED (attempt {attempt})", GREEN)
    else:
        human_resp = state.get("human_response") or {}
        is_skip = human_resp.get("action") == "skip"
        is_abort = human_resp.get("action") == "abort"

        pnr_ok = (state.get("route_result") or {}).get("success", False)

        result = {
            "name": block_name,
            "success": False,
            "attempts": attempt,
            "error": state.get("previous_error", "")[:500],
            "constraints_learned": len(state.get("constraints", [])),
            "skipped": is_skip,
            "aborted": is_abort,
            "pnr_success": pnr_ok,
            "drc_clean": drc_clean,
            "lvs_match": lvs_match,
            "timing_met": timing_met,
            "precheck_ok": precheck_ok,
            "step_log_paths": step_logs,
            "gds_path": state.get("gds_path", ""),
            "routed_def_path": state.get("routed_def_path", ""),
        }
        reason = (
            "aborted" if is_abort
            else "skipped" if is_skip
            else "failed"
        )
        log(f"  [{block_name}] BACKEND {reason.upper()} after {attempt} attempts", RED)

    write_graph_event(_pr(state), "Advance Block", "graph_node_exit", {
        "block": block_name, "success": result["success"], "graph": "backend",
    })

    return {
        "completed_blocks": [result],
    }


# ---------------------------------------------------------------------------
# Node: backend_complete
# ---------------------------------------------------------------------------

async def backend_complete_node(state: BackendState) -> dict:
    """Mark the backend pipeline as done and persist results for webview."""
    completed = state.get("completed_blocks", [])
    passed = sum(1 for b in completed if b.get("success"))
    total = len(completed)

    log(f"\n{'#'*60}", CYAN)
    log(f"  BACKEND COMPLETE: {passed}/{total} blocks passed", CYAN)
    log(f"{'#'*60}\n", CYAN)

    write_graph_event(_pr(state), "Backend Complete", "graph_node_exit", {
        "passed": passed, "total": total, "graph": "backend",
    })

    # Persist structured results for the webview summary panel
    pr = Path(_pr(state))
    target_clock = state.get("target_clock_mhz", 0)
    results_payload: dict = {
        "passed": passed,
        "total": total,
        "target_clock_mhz": target_clock,
        "blocks": [],
    }
    for blk in completed:
        entry: dict = {
            "name": blk.get("name", ""),
            "success": blk.get("success", False),
            "attempts": blk.get("attempts", 0),
            "pnr_success": blk.get("pnr_success", blk.get("success", False)),
            "drc_clean": blk.get("drc_clean", False),
            "lvs_match": blk.get("lvs_match", False),
            "timing_met": blk.get("timing_met", False),
            "precheck_ok": blk.get("precheck_ok", False),
        }
        if blk.get("success"):
            entry.update({
                "total_power_mw": blk.get("total_power_mw", 0),
                "timing_wns_ns": blk.get("timing_wns_ns", 0),
                "gds_path": blk.get("gds_path", ""),
                "routed_def_path": blk.get("routed_def_path", ""),
            })
            # Read detailed metrics from PnR report files
            name = blk["name"]
            pnr_dir = pr / "syn" / "output" / name / "pnr"
            if pnr_dir.is_dir():
                from orchestrator.langgraph.backend_helpers import (
                    parse_openroad_reports,
                    parse_drc_report,
                )
                pnr_metrics = parse_openroad_reports(str(pnr_dir))
                entry.update({
                    "design_area_um2": pnr_metrics.get("design_area_um2", 0),
                    "die_area_um2": pnr_metrics.get("die_area_um2", 0),
                    "utilization_pct": pnr_metrics.get("utilization_pct", 0),
                    "wns_ns": pnr_metrics.get("wns_ns", 0),
                    "tns_ns": pnr_metrics.get("tns_ns", 0),
                    "setup_slack_ns": pnr_metrics.get("setup_slack_ns", 0),
                    "hold_slack_ns": pnr_metrics.get("hold_slack_ns", 0),
                    "total_power_mw": pnr_metrics.get("total_power_mw", 0),
                    "dynamic_power_mw": pnr_metrics.get("dynamic_power_mw", 0),
                    "leakage_power_mw": pnr_metrics.get("leakage_power_mw", 0),
                    "timing_met": pnr_metrics.get("timing_met", False),
                })
                # DRC report
                drc_rpt = pnr_dir / "magic_drc.rpt"
                if drc_rpt.exists():
                    drc = parse_drc_report(str(drc_rpt))
                    entry["drc_clean"] = drc.get("clean", False)
                    entry["drc_violations"] = drc.get("violation_count", -1)
            # Check for rendered images
            img_dir = pr / ".coresmith" / "images"
            fp_img = img_dir / f"{name}_floorplan.png"
            gds_img = img_dir / f"{name}_gds.png"
            if fp_img.exists():
                entry["floorplan_image"] = str(fp_img)
            if gds_img.exists():
                entry["gds_image"] = str(gds_img)

        results_payload["blocks"].append(entry)

    results_path = pr / ".coresmith" / "backend_results.json"
    try:
        results_path.write_text(json.dumps(results_payload, indent=2))
    except OSError:
        pass

    return {"backend_done": True}


# ---------------------------------------------------------------------------
# Node: generate_3d_view  (3D GDS layout viewer)
# ---------------------------------------------------------------------------

async def generate_3d_view_node(state: BackendState) -> dict:
    """Best-effort: generate 3D and 2D GDS layout viewers.

    Reads the primary block's GDS file and produces:
    - ``chip_finish/3d.html`` -- interactive Three.js 3D viewer
    - ``chip_finish/<block>_layout.svg`` -- full-vector 2D floorplan
    - ``chip_finish/<block>_layout.png`` -- rasterised 2D floorplan

    Never fails the pipeline.
    """
    project_root = _pr(state)
    completed_blocks = state.get("completed_blocks", [])

    write_graph_event(project_root, "Generate 3D View", "graph_node_enter", {
        "graph": "backend",
    })

    viewer_path = ""
    layout_2d_png_path = ""

    with _tracer.start_as_current_span("Generate 3D View") as span:
        # Find primary block with a GDS file
        gds_path = ""
        block_name = "unknown"
        for blk in completed_blocks:
            gp = blk.get("gds_path", "")
            if gp and Path(gp).exists():
                gds_path = gp
                block_name = blk.get("name", "unknown")
                break

        if not gds_path:
            span.set_attribute("skipped", "no_gds")
            log("  [3D] No GDS file found -- skipping 3D viewer", YELLOW)
        else:
            # ── 3D viewer ─────────────────────────────────────────
            try:
                from orchestrator.architecture.specialists.layout_3d import (
                    generate_3d_html,
                )

                html = generate_3d_html(gds_path, block_name, project_root)

                if html:
                    output_dir = Path(project_root) / "chip_finish"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    viewer_file = output_dir / "3d.html"
                    viewer_file.write_text(html, encoding="utf-8")
                    viewer_path = str(viewer_file)
                    span.set_attribute("viewer_path", viewer_path)
                    span.set_attribute("html_size", len(html))
                    log(f"  [3D] Layout viewer: {viewer_file}", GREEN)
                else:
                    log("  [3D] glTF conversion returned empty -- skipping", YELLOW)
            except Exception as exc:
                span.set_attribute("error_3d", str(exc))
                log(f"  [3D] Viewer generation failed (non-fatal): {exc}", YELLOW)

            # ── 2D layout (SVG + PNG) ─────────────────────────────
            try:
                from orchestrator.architecture.specialists.layout_3d import (
                    generate_2d_layout,
                )

                result_2d = generate_2d_layout(gds_path, block_name, project_root)

                if result_2d:
                    svg_path, png_path = result_2d
                    span.set_attribute("layout_svg_path", svg_path)
                    log(f"  [2D] Layout SVG: {svg_path}", GREEN)
                    if png_path:
                        layout_2d_png_path = png_path
                        span.set_attribute("layout_png_path", png_path)
                        log(f"  [2D] Layout PNG: {png_path}", GREEN)
                    else:
                        log("  [2D] PNG skipped (cairosvg not installed)", YELLOW)
                else:
                    log("  [2D] 2D layout generation returned empty -- skipping", YELLOW)
            except Exception as exc:
                span.set_attribute("error_2d", str(exc))
                log(f"  [2D] Layout generation failed (non-fatal): {exc}", YELLOW)

    write_graph_event(project_root, "Generate 3D View", "graph_node_exit", {
        "graph": "backend",
        "viewer_path": viewer_path,
        "layout_2d_png_path": layout_2d_png_path,
    })

    return {
        "viewer_3d_path": viewer_path,
        "layout_2d_png_path": layout_2d_png_path,
    }


# ---------------------------------------------------------------------------
# Node: final_report  (chip finish HTML dashboard)
# ---------------------------------------------------------------------------

async def final_report_node(state: BackendState) -> dict:
    """Generate a self-contained HTML dashboard summarising the design flow.

    Reads architecture docs, backend reports, DEF placements, RTL source,
    and pipeline events.  Calls the LLM to produce a single HTML file at
    ``chip_finish/dashboard.html``.
    """
    from orchestrator.architecture.specialists.chip_finish_dashboard import (
        generate_chip_finish_dashboard,
    )

    project_root = _pr(state)
    completed_blocks = state.get("completed_blocks", [])
    target_clock = state.get("target_clock_mhz", 50.0)

    # Merge frontend per-block results so the dashboard can find
    # testbenches, VCDs, and test results by individual block name.
    # The backend's completed_blocks only has the flat top-level entry.
    frontend_blocks = state.get("frontend_blocks") or state.get("block_queue", [])
    backend_names = {b.get("name") for b in completed_blocks}
    for fb in frontend_blocks:
        if fb.get("name") and fb["name"] not in backend_names:
            completed_blocks = list(completed_blocks) + [fb]

    write_graph_event(project_root, "Final Report", "graph_node_enter", {
        "graph": "backend",
        "block_count": len(completed_blocks),
    })

    with _tracer.start_as_current_span("Final Report") as span:
        span.set_attribute("block_count", len(completed_blocks))

        viewer_3d = state.get("viewer_3d_path", "")
        layout_2d_png = state.get("layout_2d_png_path", "")

        try:
            html = await generate_chip_finish_dashboard(
                completed_blocks=completed_blocks,
                project_root=project_root,
                target_clock_mhz=target_clock,
                viewer_3d_available=bool(viewer_3d and Path(viewer_3d).exists()),
                layout_2d_png_path=layout_2d_png if layout_2d_png and Path(layout_2d_png).exists() else "",
            )
        except TypeError as _e:
            log(f"  [REPORT] Dashboard generation failed ({_e}), "
                f"retrying without optional args", RED)
            html = await generate_chip_finish_dashboard(
                completed_blocks=completed_blocks,
                project_root=project_root,
                target_clock_mhz=target_clock,
            )

        output_dir = Path(project_root) / "chip_finish"
        output_dir.mkdir(parents=True, exist_ok=True)
        dashboard_path = output_dir / "dashboard.html"
        dashboard_path.write_text(html, encoding="utf-8")

        span.set_attribute("dashboard_path", str(dashboard_path))
        span.set_attribute("html_size", len(html))

    log(f"  [REPORT] Chip finish dashboard: {dashboard_path}", GREEN)

    write_graph_event(project_root, "Final Report", "graph_node_exit", {
        "graph": "backend",
        "dashboard_path": str(dashboard_path),
        "html_size": len(html),
    })

    return {"final_report_path": str(dashboard_path)}


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_after_pnr(state: BackendState) -> str:
    """Route after PnR: SUCCESS -> drc, FAIL -> diagnose."""
    pnr_ok = (state.get("route_result") or {}).get("success", False)
    return "drc" if pnr_ok else "diagnose"


route_after_pnr.__edge_labels__ = {
    "drc": "SUCCESS",
    "diagnose": "FAIL",
}


def route_after_drc(state: BackendState) -> str:
    """Route after DRC: CLEAN -> lvs, FAIL -> diagnose."""
    clean = (state.get("drc_result") or {}).get("clean", False)
    return "lvs" if clean else "diagnose"


route_after_drc.__edge_labels__ = {
    "lvs": "CLEAN",
    "diagnose": "FAIL",
}


def route_after_lvs(state: BackendState) -> str:
    """Route after LVS: MATCH -> timing_signoff, FAIL -> diagnose."""
    match = (state.get("lvs_result") or {}).get("match", False)
    return "timing_signoff" if match else "diagnose"


route_after_lvs.__edge_labels__ = {
    "timing_signoff": "MATCH",
    "diagnose": "FAIL",
}


def route_after_timing(state: BackendState) -> str:
    """Route after timing: MET -> generate_wrapper, VIOLATED -> diagnose."""
    met = (state.get("timing_result") or {}).get("met", False)
    return "generate_wrapper" if met else "diagnose"


route_after_timing.__edge_labels__ = {
    "generate_wrapper": "MET",
    "diagnose": "VIOLATED",
}


def route_decision(state: BackendState) -> str:
    """Route after the decision classifier.

    Supports targeted retries: when PnR succeeded but a downstream step
    failed, route directly to that step (via increment_attempt) instead
    of re-running PnR.
    """
    action = (state.get("debug_result") or {}).get("next_action", "retry_pnr")
    mapping = {
        "retry_pnr": "increment_attempt",
        "retry_drc": "increment_attempt",
        "retry_lvs": "increment_attempt",
        "retry_timing": "increment_attempt",
        "ask_human": "ask_human",
        "escalate": "advance_block",
    }
    return mapping.get(action, "increment_attempt")


route_decision.__edge_labels__ = {
    "increment_attempt": "RETRY",
    "ask_human": "ASK HUMAN",
    "advance_block": "ESCALATE",
}


def route_after_human(state: BackendState) -> str:
    """Route based on the human's resume action."""
    action = (state.get("human_response") or {}).get("action", "retry")
    mapping = {
        "retry": "increment_attempt",
        "skip": "advance_block",
        "abort": "backend_complete",
    }
    return mapping.get(action, "increment_attempt")


route_after_human.__edge_labels__ = {
    "increment_attempt": "RETRY",
    "advance_block": "SKIP",
    "backend_complete": "ABORT",
}


def route_after_increment(state: BackendState) -> str:
    """Route after incrementing: within limit -> target step, exhausted -> advance_block.

    When the retry target is a downstream step (DRC/LVS/timing), route
    directly there instead of re-running PnR.
    """
    exhausted = state["attempt"] > state["max_attempts"]
    if exhausted:
        return "advance_block"

    action = (state.get("debug_result") or {}).get("next_action", "retry_pnr")
    target_mapping = {
        "retry_drc": "drc",
        "retry_lvs": "lvs",
        "retry_timing": "timing_signoff",
    }
    return target_mapping.get(action, "run_pnr")


route_after_increment.__edge_labels__ = {
    "run_pnr": "RETRY PNR",
    "drc": "RETRY DRC",
    "lvs": "RETRY LVS",
    "timing_signoff": "RETRY TIMING",
    "advance_block": "EXHAUSTED",
}


def route_after_advance(state: BackendState) -> str:
    """Route after advancing: more blocks -> init_block, done -> backend_complete.

    Legacy routing function retained for backward compatibility.
    """
    idx = state.get("current_block_index", 0)
    queue = state.get("block_queue", [])
    if idx < len(queue):
        return "init_block"
    return "backend_complete"


route_after_advance.__edge_labels__ = {
    "init_block": "NEXT BLOCK",
    "backend_complete": "ALL DONE",
}


def route_after_advance_lead(state: BackendState) -> str:
    """Backend Lead routing: always go to backend_complete (single flat design)."""
    return "backend_complete"


route_after_advance_lead.__edge_labels__ = {
    "backend_complete": "DONE",
}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_backend_graph(checkpointer=None):
    """Build and compile the Backend Lead physical design StateGraph.

    Topology (Backend Lead -- flat design):
      init_design -> flat_top_synthesis -> run_pnr -> drc -> lvs ->
      timing_signoff -> mpw_precheck -> advance_block ->
      backend_complete -> generate_3d_view -> final_report -> END

    Each EDA node uses an LLM agent to adapt TCL scripts before execution.
    The diagnose/decide/ask_human/retry failure loop handles any step failure.

    Args:
        checkpointer: LangGraph checkpointer for state persistence.

    Returns:
        Compiled StateGraph ready for ``ainvoke`` / ``astream``.
    """
    graph = StateGraph(BackendState)

    # Nodes -- Backend Lead physical design flow (LLM-driven)
    graph.add_node("init_design", init_design_node)
    graph.add_node("flat_top_synthesis", flat_top_synthesis_node)
    graph.add_node("run_pnr", run_pnr_node)
    graph.add_node("drc", drc_node)
    graph.add_node("lvs", lvs_node)
    graph.add_node("timing_signoff", timing_signoff_node)
    graph.add_node("generate_wrapper", generate_wrapper_node)
    graph.add_node("mpw_precheck", mpw_precheck_node)

    # Failure handling nodes
    graph.add_node("diagnose", diagnose_node)
    graph.add_node("decide", decide_node)
    graph.add_node("ask_human", ask_human_node)
    graph.add_node("increment_attempt", increment_attempt_node)
    graph.add_node("advance_block", advance_block_node)
    graph.add_node("backend_complete", backend_complete_node)
    graph.add_node("generate_3d_view", generate_3d_view_node)
    graph.add_node("final_report", final_report_node)

    # Happy path: init -> synth -> PnR -> DRC -> LVS -> timing -> precheck -> advance
    graph.add_edge(START, "init_design")
    graph.add_edge("init_design", "flat_top_synthesis")
    graph.add_conditional_edges("flat_top_synthesis", route_after_flat_synth)
    graph.add_conditional_edges("run_pnr", route_after_pnr)

    # Physical verification gates
    graph.add_conditional_edges("drc", route_after_drc)
    graph.add_conditional_edges("lvs", route_after_lvs)
    graph.add_conditional_edges("timing_signoff", route_after_timing)
    graph.add_conditional_edges("generate_wrapper", route_after_wrapper)
    graph.add_conditional_edges("mpw_precheck", route_after_precheck)

    # Failure path
    graph.add_edge("diagnose", "decide")
    graph.add_conditional_edges("decide", route_decision)
    graph.add_conditional_edges("ask_human", route_after_human)
    graph.add_conditional_edges("increment_attempt", route_after_increment)

    # Block advancement -> backend_complete (single flat design)
    graph.add_conditional_edges("advance_block", route_after_advance_lead)
    graph.add_edge("backend_complete", "generate_3d_view")
    graph.add_edge("generate_3d_view", "final_report")
    graph.add_edge("final_report", END)

    return graph.compile(checkpointer=checkpointer)
