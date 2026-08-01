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
    # Stop the backend after flat synthesis + the chip_top gate-sim verdict,
    # WITHOUT entering P&R/DRC/LVS. This is what the frontend->backend handoff
    # asks for: the flat netlist and its gate-sim verdict are the deliverable,
    # and hardening a chip nobody asked to harden costs hours of EDA. Absent /
    # False keeps the full physical flow (every existing caller).
    stop_after_gate_sim: bool

    # Backend Lead fields (set by init_design, consumed downstream) ────────
    frontend_blocks: list[dict]           # completed blocks from pipeline
    architecture_connections: list[dict]  # block diagram connections
    design_name: str                      # e.g. "video_encoder_top"
    block_rtl_paths: dict                 # {block_name: rtl_path}
    glue_blocks: list[dict]              # detected glue block needs
    integration_top_path: str            # rtl/integration/<design>_top.v
    flat_netlist_path: str               # syn/output/<design>/<design>_netlist.v
    # chip_top gate-sim: the flat netlist replayed against the integration-DV
    # vectors. None = did not apply (recorded with a reason, never silent).
    chip_gate_sim_ok: bool | None
    chip_gate_sim_status: str
    chip_gate_sim_reason: str
    flat_sdc_path: str                   # syn/output/<design>/<design>.sdc
    # Per-attempt synthesis failure reasons retained even when a LATER attempt
    # succeeded ([{attempt, error_summary, source, timestamp, unrecorded}]).
    # Empty when nothing failed. Also merged into synth_result.json.
    synth_attempt_history: list[dict]
    synth_gate_count: int
    synth_area_um2: float
    macro_bindings: list                 # Part C: [{name,lef,gds,lib,...}] bound shells

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
    floorplan_result: dict | None
    place_result: dict | None
    cts_result: dict | None
    route_result: dict | None
    drc_result: dict | None
    lvs_result: dict | None
    timing_result: dict | None
    power_result: dict | None
    debug_result: dict | None
    precheck_result: dict | None

    # Artifact paths (set by run_pnr, consumed by drc/lvs)
    routed_def_path: Annotated[str, _last]
    pnr_verilog_path: Annotated[str, _last]
    pwr_verilog_path: Annotated[str, _last]
    spef_path: Annotated[str, _last]
    gds_path: Annotated[str, _last]
    spice_path: Annotated[str, _last]

    # Tapeout wrapper (generated before MPW precheck) ───────────────────
    wrapper_rtl_path: Annotated[str, _last]
    wrapper_result: dict | None
    submission_dir: Annotated[str, _last]

    # Step log file paths ──────────────────────────────────────────────────
    step_log_paths: Annotated[dict, _last]

    # Global accumulators (reducer) ────────────────────────────────────────
    completed_blocks: Annotated[list[dict], operator.add]

    # Human interaction ────────────────────────────────────────────────────
    human_response: dict | None

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

# Brace-safe prompt templating. Backend prompts embed Verilog/tcl snippets that
# contain literal braces -- e.g. `assign io_out = {31'b0, done};` or a Verilog
# replication `{4{oe}}`. A naive `str.format(**context)` treats every `{...}` as
# a replacement field and raises `KeyError`/`ValueError` on such snippets,
# crashing the EDA node for EVERY design (the LVS-prompt regression). `_safe_format`
# substitutes ONLY the recognized `{name}` / `{name:spec}` / `{name!conv}` tokens
# whose bare name is a key in `context`; it unescapes `{{`->`{` and `}}`->`}` like
# `str.format`, and leaves ALL other braces (Verilog/tcl code, unknown names) exactly
# as written. Result: any future code snippet in any prompt is safe by default.
#
# `\{\{` / `\}\}` are matched FIRST (ordered alternation, leftmost match) so an
# escaped `{{name}}` renders to the literal `{name}` and is never substituted --
# byte-identical to `str.format`. A field spec/conversion may not itself contain
# braces (`[^{}]*`), which is true for every backend placeholder and keeps the
# scan single-pass.
_SAFE_FORMAT_RE = re.compile(
    r"\{\{"                              # 1: escaped open brace -> "{"
    r"|\}\}"                             # (no group): escaped close brace -> "}"
    r"|\{(\w+)(?:!([rsa]))?(?::([^{}]*))?\}"  # 2:name 3:conversion 4:format-spec
)


def _safe_format(template: str, context: dict) -> str:
    """Substitute recognized ``{placeholder}`` tokens, leave every other brace
    literal, and unescape ``{{``/``}}`` -- never raising on Verilog/tcl braces.

    Rendering is byte-identical to ``template.format(**context)`` for any prompt
    whose only single-brace ``{name}`` tokens are context keys (all backend
    prompts), while a ``{...}`` code snippet or an unknown ``{name}`` is passed
    through unchanged instead of crashing.
    """
    def _sub(m: re.Match) -> str:
        text = m.group(0)
        if text == "{{":
            return "{"
        if text == "}}":
            return "}"
        name, conv, spec = m.group(1), m.group(2), m.group(3)
        if name not in context:
            return text  # unknown placeholder / not a real field -> leave literal
        value = context[name]
        if conv == "r":
            value = repr(value)
        elif conv == "s":
            value = str(value)
        elif conv == "a":
            value = ascii(value)
        return format(value, spec or "")

    return _SAFE_FORMAT_RE.sub(_sub, template)


async def _run_llm_eda_step(
    step_name: str,
    prompt_file: str,
    context: dict,
    result_json_path: str,
    timeout: int = 1200,
    capture_reply: bool = False,
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
        capture_reply: Attach the step's own summary text under ``_llm_reply``.
            Default OFF so every other EDA step's result dict is byte-identical
            (these dicts land in graph state and the final report). The
            synthesis driver opts in because a step that RETRIED internally says
            so only in that text -- the transcript is otherwise discarded, which
            is how "succeeded on attempt 2" left no record of attempt 1.

    Returns:
        Parsed result dict from the JSON file, or a failure dict.
    """
    from orchestrator.langchain.agents.coresmith_llm import DEFAULT_MODEL, ClaudeLLM
    from orchestrator.langgraph.eda_prompts import (
        merged_prompt_context,
        resolve_prompt_path,
    )

    # Merge in the active deployment's PDK/tool context ({pdk_summary},
    # {tool_notes}, ...); the caller's context keys win on collision. The
    # rollback flag (CORESMITH_TOOL_CLI_PROMPTS=0) selects a .legacy.md sibling.
    context = merged_prompt_context(prompt_file, context)
    prompt_path = resolve_prompt_path(_PROMPT_DIR, prompt_file)
    system_prompt = _safe_format(prompt_path.read_text(), context)

    user_message = (
        f"Execute the {step_name} step as described in the system prompt.\n"
        f"Write the result JSON to: {result_json_path}\n"
        f"After writing the result file, respond with a brief summary."
    )

    llm = ClaudeLLM(model=DEFAULT_MODEL, timeout=timeout)

    reply = ""
    try:
        reply = await llm.call(
            system=system_prompt,
            prompt=user_message,
            run_name=step_name,
        ) or ""
    except Exception as e:
        return {"success": False, "error": f"LLM call failed: {e}"}

    result_path = Path(result_json_path)
    if result_path.exists():
        try:
            parsed = json.loads(result_path.read_text(encoding="utf-8"))
            if capture_reply and isinstance(parsed, dict):
                parsed.setdefault("_llm_reply", str(reply)[:4000])
            return parsed
        except (json.JSONDecodeError, OSError):
            pass

    failed = {
        "success": False,
        "error": f"LLM did not write result JSON to {result_json_path}",
    }
    if capture_reply:
        failed["_llm_reply"] = str(reply)[:4000]
    return failed


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
                # Discover the top's ACTUAL clock port -- a hardcoded
                # `[get_ports clk]` never binds on a Caravel-style top whose
                # clock is wb_clk_i: CTS then sees zero clock nets, the whole
                # tree rides one unbuffered ~1400-fanout net, and STA is
                # meaningless (live run: ~30k shorts from exactly this).
                clk_port = "clk"
                try:
                    from orchestrator.langgraph.pipeline_helpers import (
                        _detect_clock_port,
                    )
                    clk_port = (_detect_clock_port(
                        rtl_path.read_text(errors="ignore")) or "clk")
                except Exception:
                    clk_port = "clk"
                sdc.write_text(
                    f"create_clock -name clk -period {period_ns} [get_ports {clk_port}]\n"
                    f"set_input_delay {period_ns * 0.2:.1f} -clock clk [all_inputs]\n"
                    f"set_output_delay {period_ns * 0.2:.1f} -clock clk [all_outputs]\n"
                )
            return str(rtl_path), str(sdc)

    return "", ""


# ---------------------------------------------------------------------------
# Node: init_design  (Backend Lead -- discovers flat integration top)
# ---------------------------------------------------------------------------

def _select_integration_top(integration_dir: Path) -> tuple[str, str]:
    """Return (top_file, top_module) for the real integration top.

    The top is the module that instantiates other integration modules and is
    itself instantiated by none (no parent). Falls back to the file with the
    most child instantiations, then to sorted-first, so a single-file or
    unparseable dir still yields a top. Returns ("", "") only for an empty dir.
    """
    files = sorted(integration_dir.glob("*.v"))
    if not files:
        return "", ""
    mod_of: dict[str, str] = {}         # file -> its module name
    text_of: dict[str, str] = {}
    for f in files:
        try:
            src = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        text_of[str(f)] = src
        m = re.search(r"^\s*module\s+(\w+)", src, re.MULTILINE)
        if m:
            mod_of[str(f)] = m.group(1)
    if not mod_of:
        return str(files[0]), ""
    all_mods = set(mod_of.values())
    # For each file: which OTHER integration modules does it instantiate, and
    # is its OWN module instantiated by some other file?
    instantiates: dict[str, int] = {}
    instantiated_by_other: set[str] = set()
    for fp, src in text_of.items():
        my_mod = mod_of.get(fp, "")
        cnt = 0
        for other_mod in all_mods:
            if other_mod == my_mod:
                continue
            if re.search(rf"(?<![\w]){re.escape(other_mod)}\s+(?:#\s*\([^;]*?\)\s*)?[\w\\]+\s*\(",
                         src):
                cnt += 1
        instantiates[fp] = cnt
        # a module is "used" if another file names it as an instance type
        for fp2, src2 in text_of.items():
            if fp2 == fp:
                continue
            if re.search(rf"(?<![\w]){re.escape(my_mod)}\s+(?:#\s*\([^;]*?\)\s*)?[\w\\]+\s*\(",
                         src2):
                instantiated_by_other.add(fp)
                break
    # Prefer a root (no parent); among those, the one instantiating the most
    # children. Deterministic tie-break by sorted file order.
    roots = [fp for fp in mod_of if fp not in instantiated_by_other]
    pool = roots or list(mod_of)
    _order = {str(f): i for i, f in enumerate(files)}
    best = max(pool, key=lambda fp: (instantiates.get(fp, 0), -_order.get(fp, 0)))
    return best, mod_of.get(best, "")


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

    # Find integration top-level RTL and extract actual module name. Choose
    # the ACTUAL top -- the module that instantiates the others and is
    # instantiated by none -- not sorted(glob)[0]: the alphabetically-first
    # wrapper is often a leaf GPIO adapter (openframe_project_wrapper) that
    # instantiates nothing, and hardening that empty shell reported
    # "COMPLETE 1/1" while never touching the real design.
    integration_dir = root / "rtl" / "integration"
    integration_top = ""
    if integration_dir.is_dir():
        _top_f, _top_mod = _select_integration_top(integration_dir)
        if _top_f:
            integration_top = str(_top_f)
            if _top_mod:
                design_name = _top_mod

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
        # Return the COMPUTED name. It was logged in the banner and then
        # dropped, so the state kept whatever the caller guessed -- measured:
        # the banner said user_project_wrapper while synthesis ran with a stale
        # name and yosys renamed the top to match it.
        "design_name": design_name,
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

_INTEGRATION_TB_DIRS = ("sim_build/integration", "tb/integration")


def find_integration_tb(root: Path, design_name: str) -> tuple[str, str]:
    """Locate the integration-DV testbench. Returns ``(path, note)``.

    ``path`` is "" when none can be chosen, and ``note`` then explains why.

    Why this is not just ``test_<design_name>.py``: the integration testbench is
    named after the FRONTEND design (``integration_result["design_name"]``),
    while the backend's ``design_name`` is the actual top MODULE that
    ``init_design_node`` read out of the integration RTL -- on a Caravel design
    that is ``user_project_wrapper``. The two are routinely different names for
    the same chip, and the exact-name lookup then found nothing and silently
    reported ``not_run``: the flat netlist that becomes silicon went un-simulated
    because of a filename.

    So: try the design_name form first (unambiguous when it exists), and
    otherwise fall back to the single ``test_*.py`` present. REFUSE on ambiguity
    -- picking one of several testbenches by sort order is how a gate ends up
    grading the wrong stimulus and calling it a pass.
    """
    for rel in _INTEGRATION_TB_DIRS:
        cand = root / rel / f"test_{design_name}.py"
        if cand.is_file():
            return str(cand), ""
    for rel in _INTEGRATION_TB_DIRS:
        d = root / rel
        if not d.is_dir():
            continue
        found = sorted(p for p in d.glob("test_*.py") if p.is_file())
        if len(found) == 1:
            return str(found[0]), (
                f"no test_{design_name}.py; using the only integration "
                f"testbench present, {found[0].name} (the TB is named after the "
                "frontend design, the backend top module is "
                f"'{design_name}')")
        if len(found) > 1:
            names = ", ".join(p.name for p in found)
            return "", (
                f"AMBIGUOUS integration testbench: no test_{design_name}.py, and "
                f"{len(found)} candidates in {rel} ({names}). Refusing to guess "
                "which stimulus is the chip's -- grading the wrong testbench "
                "would report a pass for a netlist nothing verified. Name the "
                f"chip's TB test_{design_name}.py, or remove the others.")
    return "", ("no integration-DV testbench found -- chip_top gate-sim needs "
                "the integration vectors as its reference stimulus")


def _run_chip_top_gate_sim(state: "BackendState", netlist: str) -> tuple:
    """Replay the integration-DV vectors through the FLAT CHIP NETLIST.

    This is the artifact that becomes silicon. Every functional gate upstream
    reads RTL; every PPA gate reads a netlist; until now nothing simulated the
    flat top. A per-block gate-sim cannot close that: a block netlist is an
    intermediate, and its stimulus is that block's own testbench rather than
    real chip traffic.

    Runs here -- after ``flat_top_synthesis``, before ``run_pnr`` -- because
    this is the first point a flat chip netlist exists, and there is no reason
    to spend P&R on a netlist that does not reproduce the verified RTL.

    Returns ``(ok, status, reason)``. ``None`` ok means the gate did not apply
    (disabled, no integration testbench, toolchain absent) and is ALWAYS
    recorded with a reason, so absence never reads as success. Never raises:
    gate plumbing must not crash the backend.
    """
    from orchestrator.harness import gate_sim as _gs

    design = state.get("design_name", "chip_top")
    root = Path(state.get("project_root", "."))

    if not _gs.gate_sim_enabled():
        log(f"  [CHIP-GATE-SIM] not run -- {_gs.GATE_SIM_ENV}=0. The FLAT CHIP "
            f"NETLIST is never simulated; a synthesis-side divergence in the "
            f"assembled design cannot be caught.", YELLOW)
        return (None, _gs.STATUS_DISABLED, f"{_gs.GATE_SIM_ENV}=0")

    # Integration DV is the reference stimulus: real traffic through the
    # assembled chip, and the run that already matched the golden.
    tb, tb_reason = find_integration_tb(root, design)
    if not tb:
        log(f"  [CHIP-GATE-SIM] not run -- {tb_reason}", YELLOW)
        return (None, _gs.STATUS_NOT_RUN, tb_reason)
    if tb_reason:
        log(f"  [CHIP-GATE-SIM] {tb_reason}", YELLOW)

    # The reference stimulus is the integration DV's own source set, rebuilt
    # from the SAME builder that DV used -- never a single path. `top_rtl_path`
    # was read here originally and is NOT a BackendState field: nothing in the
    # flow ever set it, so this gate reported not_run on every real run while
    # its unit test passed by injecting the value itself.
    from orchestrator.langgraph.integration_helpers import chip_rtl_sources

    # Accept EITHER name. `integration_top_path` is what the backend graph's
    # own init_design_node produces; `top_rtl_path` is what the frontend state
    # and .coresmith/integration_result.json call the same artifact. The bug was
    # never a misspelled key -- it was that NEITHER was populated here, so the
    # gate silently self-disabled. Reading both, and failing loudly on neither,
    # is what makes that impossible to reintroduce.
    top_rtl_path = (state.get("integration_top_path", "")
                    or state.get("top_rtl_path", "") or "")
    block_rtl = state.get("block_rtl_paths", {}) or {}
    if not top_rtl_path:
        reason = ("no integration top on disk -- the assembled chip's RTL is "
                  "the gate's reference; without it there is nothing to compare "
                  "the flat netlist against")
        log(f"  [CHIP-GATE-SIM] not run -- {reason}", YELLOW)
        return (None, _gs.STATUS_NOT_RUN, reason)
    # Dedup: on a Caravel design the assembled top and the pad-adapter block
    # both declare `module user_project_wrapper`, which is a MODDUP abort.
    rtl_sources = chip_rtl_sources(
        top_rtl_path, block_rtl, dedup_dir=root / "sim_build" / "chip_gate_sim_srcs")

    log(f"  [CHIP-GATE-SIM] Replaying integration-DV vectors through the FLAT "
        f"chip netlist ({len(rtl_sources)} reference source file(s))...", YELLOW)
    try:
        res = _gs.check_gate_sim(
            block={"name": design, "is_chip_top": True},
            netlist_path=netlist,
            rtl_path=rtl_sources,
            tb_path=tb,
        )
    except Exception as exc:  # noqa: BLE001 - never crash the backend
        reason = f"chip_top gate-sim plumbing error: {type(exc).__name__}: {exc}"
        log(f"  [CHIP-GATE-SIM] {reason}", RED)
        return (None, _gs.STATUS_NOT_RUN, reason)

    if res.status == _gs.STATUS_PASS:
        log(f"  [CHIP-GATE-SIM] PASS -- flat netlist reproduced the verified "
            f"chip RTL ({res.cycles_compared} cycles, "
            f"{res.output_bits_compared} output bits)", GREEN)
        return (True, res.status, res.reason)
    if res.status == _gs.STATUS_FAIL:
        log(f"  [CHIP-GATE-SIM] FAIL -- {res.reason}", RED)
        return (False, res.status, res.reason)
    log(f"  [CHIP-GATE-SIM] not run -- {res.reason}", YELLOW)
    return (None, res.status, res.reason)


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
    _input_paths = [integration_top]
    for bname, bpath in block_rtl.items():
        if bpath != integration_top and Path(bpath).exists():
            input_lines.append(f"- Block `{bname}`: `{bpath}`")
            _input_paths.append(bpath)

    # Part B: the backend/PD synth selects the SRAM MACRO impl on the cs_sram
    # wrappers (gated by CORESMITH_SRAM_MACRO, default ON). If the design
    # instantiates a cs_sram_*/cs_rom_*/cs_mem_* wrapper and no input file
    # already defines those modules, read the shared wrapper library too (so
    # `hierarchy -check` resolves it); then a yosys `chparam ... MEM_IMPL
    # "MACRO"` re-derives the wrappers to the `cs_mem_macro_shell` /
    # `cs_rom_macro_shell` leaf (0 storage flops) instead of a flop array. This
    # is the backend-only flip -- cocotb/Verilator DV keeps the default BEHAV.
    _sram_macro_directive = ""
    _sram_wrapper_lib = ""
    try:
        from orchestrator.langgraph.sram_wrapper import (
            backend_sram_macro_directive as _macro_directive,
        )
        from orchestrator.langgraph.sram_wrapper import (
            uses_wrapper as _uses_wrapper,
        )
        from orchestrator.langgraph.sram_wrapper import (
            wrapper_lib_path as _wrapper_lib_path,
        )
        _combined = "".join(
            Path(p).read_text(errors="ignore") for p in _input_paths
            if p and Path(p).exists()
        )
        if _uses_wrapper(_combined):
            _sram_macro_directive = _macro_directive()
            _lib = _wrapper_lib_path()
            _already_defined = bool(re.search(r"\bmodule\s+cs_(?:sram|mem|rom|fpmem)",
                                              _combined))
            if _lib and Path(_lib).exists() and not _already_defined \
                    and _lib not in _input_paths:
                input_lines.append(f"- SRAM wrapper library: `{_lib}`")
                _sram_wrapper_lib = _lib
    except Exception as _exc:  # noqa: BLE001 - macro selection is best-effort
        log(f"  [FLAT-SYNTH] SRAM-macro directive setup skipped: {_exc!r}", YELLOW)

    # --- Pre-synthesis macro binding ---------------------------------------
    # cs_mem_macro_shell hard-assigns zero ("MACRO is a backend impl, not a sim
    # model") and synthesis faithfully preserves that tie-off, so the flat
    # netlist's memories read zero: PnR is fine (the shell is replaced by
    # LEF/GDS at placement) but the design cannot be gate-simulated -- the
    # first read of real data diverges. Binding used to run on the finished
    # netlist and only recorded collateral for PnR. Resolve each geometry to a
    # concrete macro HERE instead, and swap in a shell that instantiates it, so
    # one artifact serves synthesis (via `read_verilog -lib`), PnR and sim.
    if os.environ.get("CORESMITH_MACRO_PREBIND", "1").strip().lower() \
            not in ("0", "false", "no", "off"):
        try:
            from orchestrator.langgraph import macro_prebind as _mp
            _pb_srcs = [_p for _p in _input_paths if _p]
            if _sram_wrapper_lib:
                _pb_srcs.append(_sram_wrapper_lib)
            _pb = _mp.resolve_prebindings(_pb_srcs, allow_generate=True)
            for _w in _pb.warnings:
                log(f"  [PREBIND] {_w}", YELLOW)
            if _pb.errors or _pb.unresolved:
                # Loud, and deliberately NOT substituted: a memory we cannot
                # bind must not be quietly wired to something that reads zero.
                for _e in _pb.errors:
                    log(f"  [PREBIND] {_e}", RED)
                for _u in _pb.unresolved:
                    log(f"  [PREBIND] unresolved geometry: {_u.describe()}", RED)
            elif _pb.bindings and _sram_wrapper_lib:
                _pbs = _mp.prepare_synth_sources(
                    _sram_wrapper_lib, _pb, Path(pr) / ".coresmith" / "prebind")
                input_lines = [_l for _l in input_lines
                               if "SRAM wrapper library" not in _l]
                _sram_wrapper_lib = _pbs["wrapper_lib"]
                input_lines.append(
                    f"- SRAM wrapper library: `{_sram_wrapper_lib}`")
                input_lines.append(
                    "- Bound macro shell (REPLACES the zero-driving "
                    f"cs_mem_macro_shell; read as a normal source): "
                    f"`{_pbs['bound_shell']}`")
                for _m in _pbs["models"]:
                    input_lines.append(
                        "- Macro model -- read with `read_verilog -lib` so only "
                        f"its interface is taken and it is NOT synthesized: `{_m}`")
                log(f"  [PREBIND] bound {len(_pb.bindings)} memory geometry(ies) "
                    "to concrete macros before synthesis", YELLOW)
        except Exception as _exc:  # noqa: BLE001
            log(f"  [PREBIND] skipped: {_exc!r}", YELLOW)

    with _tracer.start_as_current_span(f"Flat Top Synthesis [{design_name}]") as span:
        span.set_attribute("design_name", design_name)

        # Synthesize against a liberty with the sky130 lpflow_*/probe* cells
        # STRIPPED: dfflibmap/abc otherwise map real logic onto them (live
        # run: 167 lpflow cells in the flat netlist) and they break LVS and
        # skew timing. PnR-level set_dont_use cannot remove already-mapped
        # cells, so the exclusion must happen HERE. Reuses the cached
        # STA dont_use filter; falls back to the full liberty on any error.
        _synth_lib = str(LIBERTY)
        try:
            from orchestrator.langgraph.ppa_check import _sta_dontuse_liberty
            _synth_lib = _sta_dontuse_liberty(str(LIBERTY))
        except Exception:  # noqa: BLE001 - never block synth on the filter
            _synth_lib = str(LIBERTY)
        result = await _run_llm_eda_step(
            step_name=f"Flat Top Synthesis [{design_name}]",
            prompt_file="backend_synth_llm.md",
            context={
                "design_name": design_name,
                "target_clock_mhz": target_clock,
                "period_ns": period_ns,
                "liberty_path": _synth_lib,
                "output_dir": output_dir,
                "input_files": "\n".join(input_lines),
                "input_delay_ns": period_ns * 0.2,
                "output_delay_ns": period_ns * 0.2,
                "attempt": state.get("attempt", 1),
                "prior_failure": state.get("previous_error", "None"),
                "constraints": _format_constraints(state),
                "result_json_path": result_json_path,
                "sram_macro_directive": _sram_macro_directive or "(none)",
                "sram_wrapper_lib": _sram_wrapper_lib or "(already in inputs)",
            },
            result_json_path=result_json_path,
            # the driver retries Yosys internally -- its own summary is the only
            # place that says so (see collect_synth_attempt_history)
            capture_reply=True,
        )

        span.set_attribute("success", result.get("success", False))
        if result.get("success"):
            span.set_attribute("gate_count", result.get("gate_count", 0))

    # SELF-RECOVERY MUST NOT BE SILENT. The driver retries Yosys internally; a
    # live run logged "Synthesis succeeded on attempt 2" and attempt 1's reason
    # survived nowhere -- not in attempt_history, not in previous_error, not in
    # synth_result.json. Retain every attempt's failure reason (driver-reported,
    # harvested from the Yosys logs on disk, and the node's own prior failure),
    # and record EXPLICITLY when the driver claims a retry we cannot account for.
    from orchestrator.langgraph.backend_helpers import (
        collect_synth_attempt_history,
        describe_synth_attempt_history,
        persist_synth_attempt_history,
    )
    _synth_history = collect_synth_attempt_history(
        result,
        output_dir=output_dir,
        llm_reply=str(result.get("_llm_reply", "")),
        prior_error=str(state.get("previous_error", "")),
        node_attempt=int(state.get("attempt", 1) or 1),
    )
    if _synth_history:
        persist_synth_attempt_history(result_json_path, _synth_history)
        log("  [FLAT-SYNTH] " + describe_synth_attempt_history(_synth_history),
            YELLOW)

    write_graph_event(pr, "Flat Top Synthesis", "graph_node_exit", {
        "design_name": design_name,
        "success": result.get("success", False),
        "gate_count": result.get("gate_count", 0),
        "synth_attempt_failures": len(_synth_history),
        "graph": "backend",
    })

    if result.get("success"):
        _netlist = result.get("netlist_path", "")
        # Part C: bind each cs_mem_macro_shell / cs_rom_macro_shell leaf that
        # Part B emitted to a CONCRETE on-disk macro (pre-built, OpenRAM-
        # composed/generated), reusing the frontend resolver + the PDK. A
        # geometry that can be neither matched nor generated is a HARD, reported
        # error -- NEVER a silent fall-back to a flop array. The resolved macro
        # collateral is stashed for the DRC/LVS/PnR LEF/GDS/lib injection.
        _bindings, _bind_err = _bind_macro_shells_for_backend(_netlist)
        if _bind_err:
            log(f"  [FLAT-SYNTH] macro-shell binding FAILED: {_bind_err}", RED)
            write_graph_event(pr, "Flat Top Synthesis", "graph_node_exit", {
                "design_name": design_name, "success": False,
                "macro_binding_error": True, "graph": "backend",
            })
            return {
                "phase": "synth",
                "synth_attempt_history": _synth_history,
                "previous_error": _bind_err,
                "flat_netlist_path": "",
                "flat_sdc_path": "",
            }
        _gs_ok, _gs_status, _gs_reason = _run_chip_top_gate_sim(state, _netlist)
        return {
            "phase": "synth",
            "synth_attempt_history": _synth_history,
            "flat_netlist_path": _netlist,
            "flat_sdc_path": result.get("sdc_path", ""),
            "synth_gate_count": result.get("gate_count", 0),
            "synth_area_um2": result.get("area_um2", 0.0),
            "macro_bindings": _bindings,
            "chip_gate_sim_ok": _gs_ok,
            "chip_gate_sim_status": _gs_status,
            "chip_gate_sim_reason": _gs_reason,
            # Hand diagnose something to work with. Only on a real FAIL: a
            # not_run must not look like an error downstream.
            **({"previous_error":
                f"chip_top gate-sim FAIL: {_gs_reason}"} if _gs_ok is False else {}),
        }
    else:
        # A FAILED synthesis carries its per-attempt history into previous_error
        # too: diagnose reads that string, and "attempt 3 failed" without the two
        # reasons before it is what made the same script defect recur every run.
        _err = result.get("error", "Flat synthesis failed")
        _hist_txt = describe_synth_attempt_history(_synth_history)
        return {
            "phase": "synth",
            "synth_attempt_history": _synth_history,
            "previous_error": (f"{_err}\n\n{_hist_txt}" if _hist_txt else _err),
            "flat_netlist_path": "",
            "flat_sdc_path": "",
        }


def _bind_macro_shells_for_backend(netlist_path: str) -> tuple[list[dict], str]:
    """Resolve every macro shell in the synthesized netlist to a concrete macro.

    Returns ``(bindings, error)``: ``bindings`` is a list of
    ``{name, lef, gds, lib, spice, verilog, shell}`` for the DRC/LVS/PnR
    injection; ``error`` is a non-empty HARD-error string when a shell geometry
    could be neither matched nor generated (never a silent flop fallback). A
    no-shell design (concrete macros / no memory) returns ``([], "")``.

    Best-effort on infrastructure faults (import/read errors are logged and do
    NOT block the backend); only an UNRESOLVABLE geometry is a hard error.
    """
    if not netlist_path or not Path(netlist_path).exists():
        return [], ""
    try:
        from orchestrator.langgraph.sram_wrapper import backend_sram_macro_enabled
        if not backend_sram_macro_enabled():
            return [], ""
        from orchestrator.langgraph.macro_registry import bind_macro_shells
        res = bind_macro_shells(netlist_path, allow_generate=True)
    except Exception as exc:  # noqa: BLE001 - infra fault must not block backend
        log(f"  [FLAT-SYNTH] macro-shell binding skipped (infra): {exc!r}", YELLOW)
        return [], ""
    if res.errors:
        return [], (
            "PPA/backend: macro-shell binding could not resolve a wrapped "
            "memory to a placeable macro (would otherwise silently flop):\n- "
            + "\n- ".join(res.errors)
        )
    bindings = [
        {"name": mi.name, "lef": mi.lef, "gds": mi.gds, "lib": mi.lib,
         "spice": mi.spice, "verilog": mi.verilog, "shell": sp.describe(),
         # shell geometry (matches the netlist shell) + concrete-macro geometry
         # (pin-adapter widths). Carried so PnR can materialize the shell into
         # its concrete macro without re-reading the PDK.
         "kind": sp.kind, "width": sp.width, "depth": sp.depth,
         "nport": sp.nport, "ports": mi.ports,
         "macro_data_bits": mi.data_bits, "macro_words": mi.words,
         "macro_mask_bits": mi.mask_bits}
        for sp, mi in res.resolved
    ]
    if res.plans:
        from orchestrator.langgraph.sram_wrapper import (
            macro_compose_tiles_enabled,
        )
        if macro_compose_tiles_enabled():
            # Materialize each tiled-composition plan into a concrete binding
            # (base tile macro + tile array), so PnR TILES + PLACES it and the
            # memory-absent assertion covers it. A plan that cannot be realized
            # by the shared-control tiling (genuine multi-bank) is a HARD,
            # reported blocker -- never a silently mis-materialized/absent memory.
            comp_bindings, comp_err = _composition_plan_bindings(res.plans)
            if comp_err:
                return [], comp_err
            bindings.extend(comp_bindings)
            if comp_bindings:
                log(f"  [FLAT-SYNTH] materialized {len(comp_bindings)} tiled "
                    f"composition(s): "
                    + "; ".join(b["shell"] for b in comp_bindings), GREEN)
        else:
            log(f"  [FLAT-SYNTH] {len(res.plans)} shell(s) resolve to a tiled "
                f"composition (RTL-level tiling); materialization DISABLED "
                f"(CORESMITH_MACRO_COMPOSE_TILES=0) -- plan dropped", YELLOW)
    if bindings:
        log(f"  [FLAT-SYNTH] bound {len(bindings)} macro shell(s) to concrete "
            f"macros: {', '.join(b['name'] for b in bindings)}", GREEN)
    return bindings, ""


def _composition_plan_bindings(
    plans: list[tuple],
) -> tuple[list[dict], str]:
    """Turn tiled-composition plans (``[(ShellSpec, CompositionPlan), ...]``)
    into ``macro_bindings`` entries the PnR/DRC/memory-absent flow consumes.

    Each entry carries the BASE tile macro's collateral (so the DRC/LVS black
    box + placed-macro assertion key on it) plus a ``composition`` sub-dict
    (tile array shape) the netlist materializer tiles from. Returns
    ``(bindings, error)``; ``error`` is non-empty for a plan the shared-control
    tiling cannot realize (multi-bank depth) -- surfaced as a hard blocker
    rather than a silently-wrong or absent memory.
    """
    out: list[dict] = []
    for sp, plan in plans:
        base = plan.base
        if int(getattr(plan, "tiles_deep", 1)) > 1:
            return [], (
                "PPA/backend: macro-shell "
                f"{sp.describe()} resolves to a MULTI-BANK tiled composition "
                f"({plan.describe()}); the current tiling materializer realizes "
                f"single-bank (width-tile / depth-over-provision) compositions "
                f"only. A genuine multi-bank ({plan.tiles_deep} banks) memory "
                f"needs per-bank select gating + a registered read mux -- it is "
                f"surfaced as a blocker rather than shipped silently wrong or "
                f"absent. Resize the memory to a single-bank geometry or provide "
                f"a deep-enough pre-built macro."
            )
        out.append({
            "name": base.name, "lef": base.lef, "gds": base.gds, "lib": base.lib,
            "spice": base.spice, "verilog": base.verilog,
            "shell": plan.describe(),
            # shell geometry (matches the netlist shell) so the materializer
            # keys the rewrite; base-macro geometry drives the tiling widths.
            "kind": sp.kind, "width": sp.width, "depth": sp.depth,
            "nport": sp.nport, "ports": base.ports or "1rw1r",
            "macro_data_bits": base.data_bits, "macro_words": base.words,
            "macro_mask_bits": base.mask_bits,
            "composition": {
                "tiles_wide": int(plan.tiles_wide),
                "tiles_deep": int(plan.tiles_deep),
                "provisioned_words": int(plan.provisioned_words),
                "provisioned_bits": int(plan.provisioned_bits),
                "base": base.name,
            },
        })
    return out, ""


def route_after_flat_synth(state: BackendState) -> str:
    """Route after flat synthesis: success -> run_pnr, fail -> diagnose.

    A chip_top gate-sim FAIL is a synthesis failure, not an advisory. The flat
    netlist provably does not reproduce the verified RTL, so P&R would harden a
    design that does not work -- and it was recorded in state and ignored, which
    is the one outcome an honest gate must not have.

    ``chip_gate_sim_ok is None`` means the gate did not APPLY (disabled, no
    integration TB, toolchain absent). That is not a verdict and must not block:
    it is already logged with a reason at the point it happened.

    ``stop_after_gate_sim`` ends the graph here in EVERY outcome -- pass, fail
    and did-not-apply alike. The caller asked for the flat netlist plus the
    gate-sim verdict and nothing more, so a FAIL must not silently pull the run
    into the diagnose/retry loop (hours of LLM-driven EDA the caller did not ask
    for). The verdict is in ``chip_gate_sim_ok`` / ``_status`` / ``_reason``,
    and a synth failure is in ``previous_error``; both are read from the final
    state, so ending is not the same as hiding.
    """
    netlist = state.get("flat_netlist_path", "")
    if state.get("stop_after_gate_sim"):
        log("  [BACKEND] stopping after flat synthesis + chip gate-sim "
            f"(gate-sim={state.get('chip_gate_sim_status', 'n/a')}); P&R/DRC/LVS "
            "NOT run -- pass full=true / --full to continue into the physical "
            "flow.", CYAN)
        return END
    if not (netlist and Path(netlist).exists()):
        return "diagnose"
    if state.get("chip_gate_sim_ok") is False:
        return "diagnose"
    return "run_pnr"


route_after_flat_synth.__edge_labels__ = {
    "run_pnr": "SUCCESS",
    "diagnose": "FAIL",
    END: "STOP AFTER GATE-SIM",
}


# ---------------------------------------------------------------------------
# Node: run_pnr  (LLM-driven OpenROAD PnR)
# ---------------------------------------------------------------------------

def memory_absent_pnr_error(
    macro_bindings: list[dict] | None, routed_def_path: str
) -> str | None:
    """Fix: the PnR analogue of the synth memory-as-flops gate.

    If macros were BOUND at synth (`macro_bindings`) but NONE appears in the
    placed layout (post-PnR DEF -- the physical `MacroInstsArea == 0` signal),
    the shell->concrete-macro materialization did not reach PnR and the SRAM is
    physically ABSENT (the chip would read all-zero). Return an actionable
    hard-error string in that case, else None.

    Composition-aware: a shell bound to an N-tile COMPOSITION plan must be
    backed by its FULL set of placed tiles. A plan that was DROPPED (0 tiles) or
    only PARTIALLY materialized (fewer base-macro instances than tiles_wide*
    tiles_deep) is a memory-absent/partial layout -- the exact false-pass this
    closes (`memory_absent_pnr_error` previously checked resolved binding NAMES,
    not composition PLANS, so a dropped plan silently proceeded to DRC/LVS).

    Pure + best-effort: returns None (no false-fail) when there are no bindings
    or the DEF cannot be read, so a design without memories is never blocked.
    """
    bindings = [b for b in (macro_bindings or []) if b.get("name")]
    if not bindings:
        return None
    if not routed_def_path or not Path(routed_def_path).exists():
        return None
    try:
        text = Path(routed_def_path).read_text(errors="ignore")
    except OSError:
        return None

    def _count(nm: str) -> int:
        return len(re.findall(rf"(?<![\w]){re.escape(nm)}(?![\w])", text))

    names = [b["name"] for b in bindings]
    placed = sum(1 for nm in set(names) if _count(nm) > 0)
    if placed == 0:
        return (
            f"Memory-absent layout: {len(names)} SRAM macro(s) were BOUND at "
            f"synth ({', '.join(names)}) but 0 were PLACED in PnR (post-PnR "
            f"MacroInstsArea == 0). The cs_mem_macro_shell -> concrete-macro "
            f"materialization did not reach PnR -- OpenROAD read a shell-stub "
            f"netlist, so the memory is physically ABSENT and the chip would "
            f"read all-zero. Confirm prepare_pnr_working_copy materialized the "
            f"shells (CORESMITH_PNR_MACRO_PLACEMENT) and that read_verilog used "
            f"the <design>_macro.v netlist, then re-run PnR."
        )
    # A composition shell needs its whole tile set placed, not just >=1 macro.
    for b in bindings:
        comp = b.get("composition")
        if not comp:
            continue
        required = max(
            1,
            int(comp.get("tiles_wide") or 1) * int(comp.get("tiles_deep") or 1),
        )
        found = _count(b["name"])
        if found < required:
            return (
                f"Memory-absent composition: the cs_mem_macro_shell "
                f"'{b.get('shell', b['name'])}' resolved to a {required}-tile "
                f"composition of '{b['name']}' but only {found} tile(s) were "
                f"PLACED in PnR. The tiled-composition plan was NOT fully "
                f"materialized into the <design>_macro.v netlist -- so the "
                f"memory is physically ABSENT/partial and the chip would read "
                f"all-zero. Confirm CORESMITH_MACRO_COMPOSE_TILES + "
                f"CORESMITH_PNR_MACRO_PLACEMENT tiled + placed every tile, then "
                f"re-run PnR."
            )
    return None


def pnr_linked_cell_count(routed_def_path: str) -> int | None:
    """Return the placed-instance count from a routed DEF (`COMPONENTS <N> ;`),
    or None when it can't be read. This is the count of cells OpenROAD actually
    linked + placed as the top -- PnR only ADDS physical cells (tap/fill/CTS
    buffers), so it is a lower bound on the synth gate count for the SAME top."""
    if not routed_def_path or not Path(routed_def_path).exists():
        return None
    try:
        text = Path(routed_def_path).read_text(errors="ignore")
    except OSError:
        return None
    m = re.search(r"^\s*COMPONENTS\s+(\d+)\s*;", text, re.MULTILINE)
    return int(m.group(1)) if m else None


def pnr_cell_count_shortfall_error(
    gate_count: int, routed_def_path: str, *, min_ratio: float = 0.5
) -> str | None:
    """Fix 2: hard-fail when the linked/placed design is far smaller than synth.

    A gross deficit (e.g. 154 placed vs 4,682 synth gates, ratio ~0.03)
    means a SUB-BLOCK was linked as the top instead of the full integration
    netlist -- routing + signing off a fragment mislabeled as the chip. Returns
    an actionable hard-error string in that case, else None.

    Pure + best-effort: returns None (never a false-fail) when synth gate_count
    is unknown/zero or the routed DEF can't be read. Defense-in-depth backstop
    independent of the top-name heuristic (Fix 1)."""
    try:
        gc = int(gate_count or 0)
    except (TypeError, ValueError):
        return None
    if gc <= 0:
        return None
    linked = pnr_linked_cell_count(routed_def_path)
    if linked is None:
        return None
    if linked < min_ratio * gc:
        return (
            f"Cell-count shortfall: PnR linked/placed {linked} cell(s) but "
            f"synth reported {gc} gate(s) for this design (ratio "
            f"{linked / gc:.3f} < {min_ratio:.2f}). A sub-block was likely "
            f"linked as the top instead of the full integration netlist -- "
            f"refusing to sign off a {linked}-cell fragment as the "
            f"{gc}-gate chip. Confirm link_design targets the real top "
            f"(defined-but-not-instantiated module), then re-run PnR."
        )
    return None


async def run_pnr_node(state: BackendState) -> dict:
    """Run OpenROAD PnR entirely within the inner Claude LLM.

    The LLM writes, executes, and debugs the OpenROAD TCL script
    autonomously -- handling floorplan, placement, CTS, routing,
    and timing analysis in a single LLM session.
    """
    from orchestrator.langgraph.backend_helpers import (
        CELL_LEF,
        LIBERTY,
        OPENROAD_BIN,
        TECH_LEF,
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
    try:
        tcl_path = prepare_pnr_working_copy(
            design_name=block_name,
            netlist_path=netlist_path,
            sdc_path=sdc_path,
            output_dir=output_dir,
            utilization=utilization,
            density=density,
            # Fix: feed the shell->concrete-macro bindings resolved at flat synth so
            # PnR materializes + PLACES the SRAM (was left on the concrete-name-only
            # path, so shell-bound memories reached PnR as empty stubs).
            macro_bindings=state.get("macro_bindings"),
        )
    except Exception as _prep_exc:
        # Fix 2: macro read/placement is NON-OPTIONAL when memories were bound at
        # synth -- prepare_pnr_working_copy raises (LefDbuError / RuntimeError)
        # rather than emit a macro-less TCL. Convert it into an honest PnR
        # failure (routes to diagnose/park) so the run cannot proceed with the
        # SRAM physically absent.
        error_msg = f"PnR prep failed (macro placement required): {_prep_exc}"
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

    # Fix: memory-absent hard-fail. If memories were bound at synth but the
    # placed layout contains no macro, PnR shipped a logic-only die with the
    # SRAM physically absent -- catch it here instead of downstream. Gated
    # (default ON) so the pre-fix behavior is restorable.
    #
    # Fix 2: evaluate this on BOTH the success AND the failure path. On the
    # failure path a bound-but-not-placed / un-routable memory-absent layout is
    # converted into a clean, honest "memory-absent" diagnosis (the actionable
    # error) instead of the LLM churning on an unrelated routing message -- and
    # closes the reward-hack where the agent dropped the macro LEFs and routed a
    # memory-less die that "passed".
    try:
        from orchestrator.langgraph.sram_wrapper import (
            pnr_macro_placement_enabled,
        )
        if pnr_macro_placement_enabled():
            _mem_err = memory_absent_pnr_error(
                state.get("macro_bindings"), routed_def
            )
            if _mem_err:
                # On the success path, demote to failure. On the failure path,
                # surface the memory-absent diagnosis as the reported error.
                pnr_ok = False
                result["error"] = _mem_err
                log(f"  [PNR] {_mem_err}", RED)
    except Exception as _exc:  # never let the gate itself break PnR
        log(f"  [PNR] memory-absent check skipped: {_exc!r}", YELLOW)

    # Fix 2: cell-count guard. If PnR linked/placed far fewer cells than synth
    # reported for this design, a SUB-BLOCK was linked as the top (a fragment
    # mislabeled as the chip) -- demote to failure with an actionable diagnosis.
    # Evaluated on BOTH the success and the failure path so a fragment can never
    # ship a "clean" signoff. Gated (default ON) so the pre-fix behavior is
    # restorable; independent of the Fix 1 top-name heuristic.
    try:
        from orchestrator.langgraph.sram_wrapper import (
            pnr_cellcount_guard_enabled,
            pnr_cellcount_min_ratio,
        )
        if pnr_cellcount_guard_enabled():
            _cc_err = pnr_cell_count_shortfall_error(
                gate_count, routed_def, min_ratio=pnr_cellcount_min_ratio()
            )
            if _cc_err:
                pnr_ok = False
                result["error"] = _cc_err
                log(f"  [PNR] {_cc_err}", RED)
    except Exception as _exc:  # never let the guard itself break PnR
        log(f"  [PNR] cell-count guard skipped: {_exc!r}", YELLOW)

    # Honest gate: a routed design that OpenROAD left with unresolved detailed-
    # route DRC violations is NOT a passing PnR (same false-pass class as the
    # synth/DRC honest gates). The inner LLM occasionally reports success:true
    # on top of a non-empty route_drc.rpt (e.g. a truncated detailed_route that
    # stopped at thousands of open markers), so trust the tool artifact: count
    # the true violation ENTRIES and demote to failure. Evaluated on BOTH the
    # success and the failure path; gated (default ON, CORESMITH_PNR_ROUTE_DRC_
    # GATE=0 restores the pre-fix behavior). pnr_result.json is rewritten to
    # reflect the demotion so no downstream default re-marks it success.
    try:
        from orchestrator.langgraph.backend_helpers import (
            count_route_drc_violations,
            pnr_route_drc_gate_enabled,
        )
        if pnr_route_drc_gate_enabled():
            _route_drc_path = Path(output_dir) / "route_drc.rpt"
            _rdrc = count_route_drc_violations(_route_drc_path)
            result["route_drc_violations"] = _rdrc
            if _rdrc > 0:
                _rdrc_err = (
                    f"Route-DRC gate: detailed routing left {_rdrc} unresolved "
                    f"DRC violation(s) in {_route_drc_path}. Refusing to report "
                    f"a passing PnR on a die with open detailed-route DRC "
                    f"markers -- re-route to convergence, or disable the gate "
                    f"with CORESMITH_PNR_ROUTE_DRC_GATE=0."
                )
                pnr_ok = False
                result["success"] = False
                if not result.get("error"):
                    result["error"] = _rdrc_err
                log(f"  [PNR] {_rdrc_err}", RED)
                # Rewrite the tool artifact so pnr_result.json reflects the
                # demotion (it was written success:true by the inner LLM).
                try:
                    Path(result_json_path).write_text(
                        json.dumps(result, indent=2)
                    )
                except (OSError, TypeError):
                    pass
    except Exception as _exc:  # never let the gate itself break PnR
        log(f"  [PNR] route-DRC gate skipped: {_exc!r}", YELLOW)

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
        CELL_GDS,
        CELL_LEF,
        MAGIC_BIN,
        MAGIC_RC,
        TECH_LEF,
        macro_bboxes_from_def,
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

    # SRAM-macro awareness for extraction: descending into a hard macro's
    # transistor-level .mag is intractable on a mm-scale die (786 MB .ext,
    # empty result JSON). Detect instantiated macros and pass their LEF
    # abstracts + names so the DRC step reads the LEF and `extract halt`s each
    # macro, comparing it as a black box (proven on matmul's 18 macros).
    _macro_lefs: list[str] = []
    _macro_names: list[str] = []
    try:
        from orchestrator.langgraph.macro_registry import (
            detect_instantiated_macros,
            discover_macros,
        )
        _reg = discover_macros()
        for _mi in detect_instantiated_macros(routed_def, _reg):
            if _mi.lef and _mi.name not in _macro_names:
                _macro_lefs.append(_mi.lef)
                _macro_names.append(_mi.name)
    except Exception:  # noqa: BLE001 - macro awareness is best-effort
        _macro_lefs, _macro_names = [], []
    # Part C: also inject the collateral of the concrete macros the cs_mem/
    # cs_rom shells were bound to at synth (macro_bindings), so a memory that
    # entered as a `cs_mem_macro_shell` leaf is extracted as a black box here
    # too -- not only the macros whose CONCRETE name already appears in the DEF.
    for _b in (state.get("macro_bindings") or []):
        _lef, _nm = _b.get("lef", ""), _b.get("name", "")
        if _lef and _nm and _nm not in _macro_names and Path(_lef).exists():
            _macro_lefs.append(_lef)
            _macro_names.append(_nm)

    write_graph_event(_pr(state), "DRC", "graph_node_enter", {"block": block_name, "graph": "backend"})

    output_dir = _output_dir(state)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    result_json_path = str(Path(output_dir) / "drc_result.json")

    with _tracer.start_as_current_span(f"DRC [{block_name}]") as span:
        span.set_attribute("block_name", block_name)

        result = await _run_llm_eda_step(
            step_name=f"DRC [{block_name}]",
            prompt_file="backend_drc_llm.md",
            # Magic DRC on a mm-scale die legitimately needs >20 min; the
            # hardcoded 1200s default false-timed-out a live run and consumed
            # its attempts. Env-tunable like the PNR step.
            timeout=_eda_timeout("CORESMITH_DRC_TIMEOUT", 2400),
            context={
                "design_name": block_name,
                "magic_rc": str(MAGIC_RC),
                "cell_gds": str(CELL_GDS),
                "cell_lef": str(CELL_LEF),
                "tech_lef": str(TECH_LEF),
                "magic_bin": str(MAGIC_BIN),
                "routed_def_path": routed_def,
                "output_dir": output_dir,
                "attempt": state["attempt"],
                "prior_failure": state.get("previous_error", "None"),
                "constraints": _format_constraints(state),
                "result_json_path": result_json_path,
                "macro_lefs": " ".join(_macro_lefs),
                "macro_names": " ".join(_macro_names),
                "has_macros": "yes" if _macro_names else "no",
            },
            result_json_path=result_json_path,
        )

        drc_clean = result.get("clean", False)
        drc_count = result.get("violation_count", 0 if drc_clean else 999)
        gds_path = result.get("gds_path", "")
        spice_path = result.get("spice_path", "")
        report_path = result.get("report_path") or result.get("drc_report_path") or str(Path(output_dir) / "magic_drc.rpt")

        # Signed-off hard-macro interiors: a hard macro stays a LEF abstract in
        # the top-level Magic DRC, so its sub-min-area LEF pins (e.g. a small
        # OpenRAM SRAM's 0.38x0.38 um met4 signal pins) are DRC'd as top-level
        # met4 even though the real, signed-off macro GDS -- merged into the
        # shipped GDS -- is clean. Build the placed-macro bboxes from the routed
        # DEF (the placement Magic actually DRC'd) so the parser can drop those
        # in-interior met1-4 artifacts. Self-guards on macro presence (returns
        # [] when no registry macro is placed); env-gated default-ON. met5 is
        # never excluded (top PDN runs over macros there).
        macro_bboxes: list = []
        try:
            macro_bboxes = macro_bboxes_from_def(routed_def)
        except Exception:  # noqa: BLE001 - bbox derivation is best-effort
            macro_bboxes = []

        # The EDA agent occasionally writes contradictory JSON, for example
        # clean=false with violation_count=0 after Magic produced an empty
        # DRC report. Trust the tool artifact over the free-form summary.
        if Path(report_path).exists():
            parsed_drc = parse_drc_report(report_path, macro_bboxes=macro_bboxes or None)
            parsed_count = parsed_drc.get("violation_count", -1)
            if parsed_count >= 0:
                drc_count = parsed_count
                drc_clean = bool(parsed_drc.get("clean", False))
            _excl = parsed_drc.get("excluded_count", 0)
            if _excl:
                span.set_attribute("macro_interior_excluded", _excl)
                write_graph_event(_pr(state), "DRC", "macro_interior_exclude", {
                    "block": block_name, "graph": "backend",
                    "excluded_count": _excl,
                    "excluded_detail": parsed_drc.get("excluded_detail", {}),
                })

        span.set_attribute("clean", drc_clean)
        span.set_attribute("violation_count", drc_count)

    if gds_path and Path(gds_path).exists():
        img_dir = Path(_pr(state)) / ".coresmith" / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        render_layout_image(gds_path, str(img_dir / f"{block_name}_gds.png"))

    write_graph_event(_pr(state), "DRC", "graph_node_exit", {
        "block": block_name, "clean": drc_clean, "violation_count": drc_count, "graph": "backend",
    })

    # Salvage conventional artifact paths when the step's result JSON came
    # back empty/partial (live run: Magic completed with a clean report but
    # ext2spice on a 786 MB full-parasitic .ext never finished, the JSON was
    # written as 0 bytes, and the LVS node then hard-failed on an EMPTY
    # spice_path even though the artifacts it needed were derivable). The
    # tool artifacts on disk are the truth; the JSON is only a summary.
    if not gds_path or not Path(gds_path).exists():
        _conv_gds = Path(output_dir) / f"{block_name}.gds"
        if _conv_gds.exists():
            gds_path = str(_conv_gds)
    if not spice_path or not Path(spice_path).exists():
        _conv_spice = Path(output_dir) / f"{block_name}.spice"
        if _conv_spice.exists():
            spice_path = str(_conv_spice)

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
    from orchestrator.langgraph.backend_helpers import NETGEN_BIN, NETGEN_SETUP

    block = state["current_block"]
    block_name = block["name"]

    drc_result = state.get("drc_result") or {}
    if not drc_result.get("clean"):
        error_msg = drc_result.get("errors", "DRC not clean")
        return {"lvs_result": {"match": False, "errors": error_msg}, "phase": "lvs", "previous_error": error_msg}

    spice_path = state.get("spice_path", "")
    pwr_verilog = state.get("pwr_verilog_path", "")

    if not spice_path or not Path(spice_path).exists():
        # The DRC step's result JSON can come back empty (extraction overrun)
        # while the .spice exists at its conventional location -- or can be
        # produced out-of-band. Look before hard-failing a structural error.
        _conv = Path(_output_dir(state)) / f"{_block_name(state)}.spice"
        if _conv.exists():
            spice_path = str(_conv)
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
            timeout=_eda_timeout("CORESMITH_LVS_TIMEOUT", 2400),
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
        tie_analysis = ""
        # Deterministic constant-tie / port-equivalence proof (default ON). When
        # netgen reports a top-pin/net mismatch, only accept it as a match if it
        # is PROVABLY a benign constant-tie/replication (caravel io_out/io_oeb
        # tie-off) or a constant-tied unused macro input -- never an
        # independently-driven real short. Grounds the verdict in the report +
        # reference netlist instead of the inner LLM's free-form judgement, and
        # never flips a netgen "match uniquely" to fail (no regression).
        try:
            from orchestrator.langgraph.macro_backend import (
                classify_lvs_report,
                lvs_verify_ties_enabled,
            )
            if not match and lvs_verify_ties_enabled():
                report_path = result.get("report_path") or str(
                    Path(output_dir) / f"{block_name}_lvs.rpt"
                )
                report_text = ""
                if report_path and Path(report_path).exists():
                    report_text = Path(report_path).read_text(errors="replace")
                ref_v = ""
                if pwr_verilog and Path(pwr_verilog).exists():
                    ref_v = Path(pwr_verilog).read_text(errors="replace")
                verdict = classify_lvs_report(report_text, ref_v)
                if verdict.get("accept"):
                    match = True
                    tie_analysis = verdict.get("analysis", "")
                    log(f"  [LVS] constant-tie proof accepted: {tie_analysis}",
                        GREEN)
        except Exception as _exc:  # never let the proof itself break LVS
            log(f"  [LVS] constant-tie proof skipped: {_exc!r}", YELLOW)
        span.set_attribute("match", match)

    write_graph_event(_pr(state), "LVS", "graph_node_exit", {
        "block": block_name, "match": match,
        "device_delta": result.get("device_delta", 0),
        "net_delta": result.get("net_delta", 0),
        "analysis": tie_analysis or result.get("analysis", ""),
        "graph": "backend",
    })

    out: dict = {
        "lvs_result": {
            "match": match,
            "device_delta": result.get("device_delta", 0),
            "net_delta": result.get("net_delta", 0),
            "report_path": result.get("report_path", ""),
            "llm_analysis": result.get("analysis", ""),
            "tie_analysis": tie_analysis,
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

def _conditional_pass_allowed(wns_ns, *, waiver_exists: bool) -> bool:
    """A-Fix 2h: a CONDITIONAL_PASS timing verdict counts as MET only when

    - the worst negative slack is actually non-negative (``wns_ns >= 0``), OR
    - a recorded timing waiver exists on disk, OR
    - the operator opted in via ``CORESMITH_ALLOW_CONDITIONAL_PASS`` (or the
      global ``CORESMITH_GATE_FAIL_OPEN``).

    Otherwise an LLM that returns ``CONDITIONAL_PASS`` over a genuine timing
    violation is fail-closed and routed to diagnose.
    """
    from orchestrator.langgraph.gate_guard import gate_fail_open_enabled
    from orchestrator.profile import flag_enabled
    if isinstance(wns_ns, (int, float)) and wns_ns >= 0:
        return True
    if waiver_exists:
        return True
    return (
        flag_enabled("CORESMITH_ALLOW_CONDITIONAL_PASS", default=False)
        or gate_fail_open_enabled()
    )


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

        # CONDITIONAL_PASS is met ONLY with non-negative slack, a recorded
        # waiver, or an explicit operator opt-in (A-Fix 2h) -- otherwise it is
        # fail-closed and routes to diagnose.
        if sign_off == "CONDITIONAL_PASS":
            _waiver = (Path(_pr(state)) / ".coresmith" / "waivers"
                       / f"timing_{block_name}.json")
            if _conditional_pass_allowed(wns, waiver_exists=_waiver.exists()):
                met = True
                log(f"  [STA] Timing CONDITIONAL PASS @ {target_mhz} MHz "
                    f"(WNS={wns:.2f} ns) -- {analysis.get('assessment', '')}",
                    YELLOW)
            else:
                met = False
                log(f"  [STA] CONDITIONAL_PASS REJECTED @ {target_mhz} MHz "
                    f"(WNS={wns:.2f} ns < 0, no waiver) -- fail-closed, routing "
                    f"to diagnose", RED)
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
        # Tri-state on purpose: True / False / None, where None is "could not
        # be measured" (see langgraph.drc_verdict). Collapsing None to False
        # would report a check that never ran as a check that FAILED, which is
        # the exact dishonesty the verdict states were introduced to remove.
        "checks": {
            k: (v.get("pass") if v.get("status") != "not_run" else None)
            for k, v in precheck_result.get("checks", {}).items()
        },
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
        elif debug_result.get("escalate") or attempt >= max_attempts:
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

    _exhausted = state["attempt"] > state.get("max_attempts", 3)
    payload = {
        "type": "human_intervention_needed",
        "graph": "backend",
        "block_name": block_name,
        "attempt": state["attempt"],
        "max_attempts": state.get("max_attempts", 3),
        # True when the block hit its attempt budget: a `retry` REOPENS it
        # with a fresh budget (not a dead end). This replaces the old
        # silent advance-to-complete-with-success:false on exhaustion.
        "exhausted": _exhausted,
        "retry_reopens": _exhausted,
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
    """Bump the attempt counter, or RESET it when reopening after exhaustion.

    When the block already exhausted its budget (attempt > max_attempts) and
    the outer agent chose retry at the exhaustion park, re-entering here with
    a bumped counter would immediately re-exhaust -> an unbreakable loop. A
    post-exhaustion retry instead RESETS to attempt 1 (a fresh bounded budget,
    granted only by an explicit human retry), so a design terminal at N/N can
    be reopened and re-driven -- the missing 'reopen last block' path.
    """
    if state["attempt"] > state["max_attempts"]:
        new_attempt = 1
        log(f"  [REOPEN] Backend block reopened after exhaustion -- attempt "
            f"budget reset to 1/{state['max_attempts']}", YELLOW)
    else:
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
                    macro_bboxes_from_def,
                    parse_drc_report,
                    parse_openroad_reports,
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
                # DRC report -- apply the same signed-off hard-macro interior
                # exclusion as the gate so the summary verdict is consistent.
                drc_rpt = pnr_dir / "magic_drc.rpt"
                if drc_rpt.exists():
                    _def = blk.get("routed_def_path", "")
                    _mbb: list = []
                    try:
                        if _def:
                            _mbb = macro_bboxes_from_def(_def)
                    except Exception:  # noqa: BLE001 - best-effort
                        _mbb = []
                    drc = parse_drc_report(str(drc_rpt), macro_bboxes=_mbb or None)
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
    """Route after incrementing: within limit -> target step, exhausted ->
    ask_human.

    When the retry target is a downstream step (DRC/LVS/timing), route
    directly there instead of re-running PnR. On EXHAUSTION, PARK on the
    human interrupt instead of silently advancing to backend_complete with
    success=false: a false DRC-timeout (or any recoverable stall) then
    surfaces as an actionable interrupt, and a retry reopens the block with a
    fresh budget (see increment_attempt_node) rather than dead-ending a
    design that was one methodology fix from closing.
    """
    exhausted = state["attempt"] > state["max_attempts"]
    if exhausted:
        return "ask_human"

    action = (state.get("debug_result") or {}).get("next_action", "retry_pnr")
    target_mapping = {
        "retry_drc": "drc",
        "retry_lvs": "lvs",
        "retry_timing": "timing_signoff",
    }
    return target_mapping.get(action, "run_pnr")


route_after_increment.__edge_labels__ = {
    "ask_human": "EXHAUSTED -> PARK",
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
