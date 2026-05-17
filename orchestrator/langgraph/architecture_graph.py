# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Architecture decision graph -- LangGraph state machine for architecture iteration.

Runs as an autonomous background task via the MCP server, identical to the
pipeline graph pattern. Checkpointed to SQLite for durability across MCP
restarts. Uses interrupt() for human-in-the-loop escalation at five points:

  0. After Gather Requirements -- always interrupts so user can answer sizing questions
  1. After Block Diagram -- if the LLM returns questions or flags ambiguities
  2. After Constraint Check -- if violations are "structural" (require architect input)
  3. After Max Rounds Exhausted -- always escalate instead of silently ending
  4. After Create Documentation -- final OK2DEV/REVISE gate before RTL handoff

Document hierarchy:
    PRD  -- "What functionality is needed?"       (Gather Requirements)
    SAD  -- "How do we get there and why?"         (System Architecture)
    FRD  -- "How well should it work?"             (Functional Requirements)
    ERS  -- "What is needed to enable it?"         (Create Documentation)

Flow:
    START -> Gather Requirements (LLM) -> route_after_prd
          -> [QUESTIONS] -> Escalate PRD (interrupt, user answers) -> route_after_prd_escalation
          -> [PRD_COMPLETE] -> System Architecture -> Functional Requirements
          -> Block Diagram (LLM) -> review_diagram
          -> [QUESTIONS] -> Escalate Diagram -> route_after_diagram_escalation
          -> [CLEAN] -> Memory Map -> Clock Tree -> Register Spec
          -> Constraint Check -> route_after_constraints
          -> [PASS] -> Finalize -> Create Documentation -> Final Review (interrupt)
          -> [OK2DEV] -> Architecture Complete -> END
          -> [REVISE] -> Block Diagram (loop with feedback)
          -> [STRUCTURAL] -> Escalate Constraints -> route_after_constraint_escalation
          -> [AUTO_FIX] -> Constraint Iteration -> route_after_increment
          -> [CONTINUE] -> Block Diagram  (loop)
          -> [EXHAUSTED] -> Escalate Exhausted -> route_after_exhausted_escalation
"""

from __future__ import annotations

import json
import operator
from pathlib import Path
from typing import Annotated, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from opentelemetry import trace

from orchestrator.architecture.state import ARCH_DOC_DIR
from orchestrator.langgraph.event_stream import write_graph_event

_tracer = trace.get_tracer("socmate.langgraph.architecture_graph")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pr(state: dict) -> str:
    """Extract project_root from graph state."""
    return state.get("project_root", ".")


def _event(state: dict, node: str, event_type: str, data: dict | None = None) -> None:
    """Emit a graph event tagged for the architecture graph."""
    merged = {"graph": "architecture"}
    if data:
        merged.update(data)
    write_graph_event(_pr(state), node, event_type, merged)


def _stage_enabled(enable_env: str, legacy_skip_env: str = "") -> bool:
    """Return whether an optional architecture stage should run.

    These stages are bypassed by default for streaming soft-IP exploration.
    Set the SOCMATE_ENABLE_* variable to 1/true/yes/on to run the stage.  The
    older SOCMATE_SKIP_* variables are still honored as an explicit skip.
    """
    import os

    if legacy_skip_env and os.environ.get(legacy_skip_env, "").strip().lower() in {
        "1", "true", "yes", "on",
    }:
        return False
    return os.environ.get(enable_env, "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _optional_stage_payload(value: dict | None) -> dict:
    """Return optional architecture artifact only when the stage actually ran."""
    if not isinstance(value, dict):
        return {}
    if value.get("skipped") is True:
        return {}
    result = value.get("result")
    if isinstance(result, dict) and result.get("skipped") is True:
        return {}
    return value


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class ArchGraphState(TypedDict):
    # Identity (set once)
    project_root: str
    requirements: str
    pdk_summary: str
    target_clock_mhz: float
    pdk_config: dict
    max_rounds: int

    # Progress
    round: int
    total_rounds: int  # Fix #6: lifetime counter -- never reset, prevents infinite loops
    phase: str  # prd | sad | frd | block_diagram | memory_map | clock_tree | register_spec | constraints | finalize | documentation

    # PRD (Product Requirements Document) -- "What functionality is needed?"
    prd_spec: Optional[dict]       # Full PRD document (Phase 2 output)
    prd_questions: Optional[list]  # Sizing questions (Phase 1 output)
    prd_answers: Optional[dict]    # Architect answers used to draft the PRD

    # SAD (System Architecture Document) -- "How do we get there and why?"
    sad_spec: Optional[dict]

    # FRD (Functional Requirements Document) -- "How well should it work?"
    frd_spec: Optional[dict]

    # ERS (Engineering Requirements Specification) -- final synthesized doc
    ers_spec: Optional[dict]

    # Accumulating (reducers -- append, never overwrite)
    violations_history: Annotated[list[dict], operator.add]
    questions: Annotated[list[dict], operator.add]
    human_response_history: Annotated[list[dict], operator.add]  # Fix #7: full escalation history

    # Latest results (overwritten each cycle)
    block_diagram: Optional[dict]
    memory_map: Optional[dict]
    clock_tree: Optional[dict]
    register_spec: Optional[dict]
    benchmark_data: Optional[dict]
    constraint_result: Optional[dict]
    human_feedback: str

    # Block diagram visualization document
    block_diagram_doc: Optional[dict]
    block_diagram_doc_validation_errors: list[str]

    # Human interaction
    human_response: Optional[dict]

    # Terminal state
    success: bool
    error: str
    block_specs_path: str
    block_diagram_doc_path: str


# ---------------------------------------------------------------------------
# Per-document persistence helpers
# ---------------------------------------------------------------------------

def _persist_prd(
    project_root: str,
    prd_result: dict,
    questions: list | None,
    answers: dict | None,
) -> None:
    """Write the PRD to disk as JSON and Markdown immediately after generation.

    Files written:
      .socmate/prd_spec.json  -- structured PRD for programmatic use
      arch/prd_spec.md     -- human-readable Markdown summary
    """
    socmate_dir = Path(project_root) / ".socmate"
    socmate_dir.mkdir(parents=True, exist_ok=True)
    arch_dir = Path(project_root) / ARCH_DOC_DIR
    arch_dir.mkdir(parents=True, exist_ok=True)

    from orchestrator.utils import atomic_write

    json_path = socmate_dir / "prd_spec.json"
    atomic_write(json_path, json.dumps(prd_result, indent=2, default=str))

    prd = prd_result.get("prd", {})
    md_lines = [
        f"# {prd.get('title', 'Product Requirements Document')}",
        f"**Revision:** {prd.get('revision', '1.0')}",
        "",
        "## Summary",
        prd.get("summary", "_(no summary)_"),
        "",
    ]

    tech = prd.get("target_technology", {})
    if tech:
        md_lines += [
            "## Target Technology",
            f"- **PDK:** {tech.get('pdk', 'N/A')}",
            f"- **Process:** {tech.get('process_nm', 'N/A')} nm",
            f"- **Rationale:** {tech.get('rationale', 'N/A')}",
            "",
        ]

    sf = prd.get("speed_and_feeds", {})
    if sf:
        md_lines += [
            "## Speed & Feeds",
            f"- **Input data rate:** {sf.get('input_data_rate_mbps', 'N/A')} Mbps",
            f"- **Output data rate:** {sf.get('output_data_rate_mbps', 'N/A')} Mbps",
            f"- **Target clock:** {sf.get('target_clock_mhz', 'N/A')} MHz",
            f"- **Latency budget:** {sf.get('latency_budget_us', 'N/A')} us",
            f"- **Throughput:** {sf.get('throughput_requirements', 'N/A')}",
            "",
        ]

    area = prd.get("area_budget", {})
    if area:
        md_lines += [
            "## Area Budget",
            f"- **Max gate count:** {area.get('max_gate_count', 'N/A')}",
            f"- **Max die area:** {area.get('max_die_area_mm2', 'N/A')} mm²",
            f"- **Notes:** {area.get('notes', 'N/A')}",
            "",
        ]

    power = prd.get("power_budget", {})
    if power:
        md_lines += [
            "## Power Budget",
            f"- **Total power:** {power.get('total_power_mw', 'N/A')} mW",
            f"- **Power domains:** {', '.join(power.get('power_domains', [])) or 'N/A'}",
            f"- **Leakage budget:** {power.get('leakage_budget_mw', 'N/A')} mW",
            f"- **Notes:** {power.get('notes', 'N/A')}",
            "",
        ]

    df = prd.get("dataflow", {})
    if df:
        md_lines += [
            "## Dataflow",
            f"- **Topology:** {df.get('topology', 'N/A')}",
            f"- **Bus protocol:** {df.get('bus_protocol', 'N/A')}",
            f"- **Data width:** {df.get('data_width_bits', 'N/A')} bits",
            f"- **Buffering:** {df.get('buffering_strategy', 'N/A')}",
            f"- **DMA required:** {df.get('dma_required', 'N/A')}",
            f"- **Notes:** {df.get('notes', 'N/A')}",
            "",
        ]

    func_reqs = prd.get("functional_requirements", [])
    if func_reqs:
        md_lines += ["## Functional Requirements"]
        for req in func_reqs:
            md_lines.append(f"- {req}")
        md_lines.append("")

    constraints = prd.get("constraints", [])
    if constraints:
        md_lines += ["## Constraints"]
        for c in constraints:
            md_lines.append(f"- {c}")
        md_lines.append("")

    open_items = prd.get("open_items", [])
    if open_items:
        md_lines += ["## Open Items"]
        for item in open_items:
            md_lines.append(f"- {item}")
        md_lines.append("")

    if questions and answers:
        md_lines += ["---", "", "## Sizing Q&A (Traceability)"]
        for q in questions:
            qid = q.get("id", "")
            md_lines.append(f"### {q.get('question', qid)}")
            md_lines.append(f"**Category:** {q.get('category', 'N/A')}")
            md_lines.append(f"**Answer:** {answers.get(qid, '_(not answered)_')}")
            md_lines.append("")

    md_path = arch_dir / "prd_spec.md"
    atomic_write(md_path, "\n".join(md_lines))


def _feedback_revision_target(feedback: str) -> str:
    """Route final-review feedback to the earliest document it criticizes."""
    text = str(feedback or "").lower()
    if any(
        token in text
        for token in (
            "prd",
            "product requirement",
            "requirements document",
            "sizing question",
            "not answered",
            "manual answer",
            "target frame",
            "frame rate",
            "golden model path",
        )
    ):
        return "Gather Requirements"
    if any(token in text for token in ("sad", "system architecture")):
        return "System Architecture"
    if any(token in text for token in ("frd", "functional requirement")):
        return "Functional Requirements"
    return "Block Diagram"


def _persist_sad(project_root: str, sad_result: dict) -> None:
    """Write the SAD to disk as Markdown.

    Files written:
      arch/sad_spec.md    -- LLM-produced Markdown (the only copy)
    """
    arch_dir = Path(project_root) / ARCH_DOC_DIR
    arch_dir.mkdir(parents=True, exist_ok=True)

    from orchestrator.utils import atomic_write

    sad_text = sad_result.get("sad_text", "")
    if not sad_text:
        sad_text = "# System Architecture Document\n\n_(empty)_\n"
    atomic_write(arch_dir / "sad_spec.md", sad_text)


def _persist_frd(project_root: str, frd_result: dict) -> None:
    """Write the FRD to disk as Markdown.

    Files written:
      arch/frd_spec.md    -- LLM-produced Markdown (the only copy)
    """
    arch_dir = Path(project_root) / ARCH_DOC_DIR
    arch_dir.mkdir(parents=True, exist_ok=True)

    from orchestrator.utils import atomic_write

    frd_text = frd_result.get("frd_text", "")
    if not frd_text:
        frd_text = "# Functional Requirements Document\n\n_(empty)_\n"
    atomic_write(arch_dir / "frd_spec.md", frd_text)


def _persist_memory_map(project_root: str, mm_result: dict) -> None:
    """Write the memory map to disk as JSON and Markdown.

    Files written:
      .socmate/memory_map.json  -- structured memory map for programmatic use
      arch/memory_map.md     -- human-readable Markdown
    """
    socmate_dir = Path(project_root) / ".socmate"
    socmate_dir.mkdir(parents=True, exist_ok=True)
    arch_dir = Path(project_root) / ARCH_DOC_DIR
    arch_dir.mkdir(parents=True, exist_ok=True)

    from orchestrator.utils import atomic_write

    atomic_write(socmate_dir / "memory_map.json", json.dumps(mm_result, indent=2, default=str))

    mm = mm_result.get("result", mm_result)
    peripherals = mm.get("peripherals", [])
    sram = mm.get("sram")
    top_csr = mm.get("top_csr")
    reasoning = mm.get("reasoning", "")

    md_lines = ["# Memory Map", ""]

    if reasoning:
        md_lines += [reasoning, ""]

    if sram:
        base = sram.get("base_address", "0x00000000")
        size = sram.get("size", 0)
        size_str = f"0x{size:X}" if isinstance(size, int) else str(size)
        md_lines += [
            "## SRAM",
            f"- **Base:** `{base}`",
            f"- **Size:** {size_str}",
            "",
        ]

    if peripherals:
        md_lines += [
            "## Peripherals",
            "",
            "| Peripheral | Base Address | Size |",
            "|---|---|---|",
        ]
        for p in peripherals:
            name = p.get("name", "?")
            base = p.get("base_address", "?")
            size = p.get("size", 0)
            size_str = f"0x{size:X}" if isinstance(size, int) else str(size)
            md_lines.append(f"| {name} | `{base}` | {size_str} |")
        md_lines.append("")

    if top_csr:
        base = top_csr.get("base_address", "?")
        size = top_csr.get("size", 0)
        size_str = f"0x{size:X}" if isinstance(size, int) else str(size)
        md_lines += [
            "## Top-Level CSR",
            f"- **Base:** `{base}`",
            f"- **Size:** {size_str}",
            "",
        ]

    atomic_write(arch_dir / "memory_map.md", "\n".join(md_lines))


def _persist_clock_tree(project_root: str, ct_result: dict) -> None:
    """Write the clock tree to disk as JSON and Markdown.

    Files written:
      .socmate/clock_tree.json  -- structured clock tree for programmatic use
      arch/clock_tree.md     -- human-readable Markdown
    """
    socmate_dir = Path(project_root) / ".socmate"
    socmate_dir.mkdir(parents=True, exist_ok=True)
    arch_dir = Path(project_root) / ARCH_DOC_DIR
    arch_dir.mkdir(parents=True, exist_ok=True)

    from orchestrator.utils import atomic_write

    atomic_write(socmate_dir / "clock_tree.json", json.dumps(ct_result, indent=2, default=str))

    ct = ct_result.get("result", ct_result)
    domains = ct.get("domains", [])
    crossings = ct.get("crossings", [])
    reset_spec = ct.get("reset_spec", {})

    md_lines = ["# Clock Tree", ""]

    if domains:
        md_lines += [
            "## Clock Domains",
            "",
            "| Domain | Frequency | Source |",
            "|---|---|---|",
        ]
        for d in domains:
            name = d.get("name", "?")
            freq = d.get("frequency_mhz", "?")
            src = d.get("source", d.get("parent", "\u2014"))
            md_lines.append(f"| {name} | {freq} MHz | {src} |")
        md_lines.append("")

    if crossings:
        md_lines += ["## Clock Domain Crossings", ""]
        for c in crossings:
            if isinstance(c, dict):
                src = c.get("from", c.get("source", "?"))
                dst = c.get("to", c.get("dest", "?"))
                mech = c.get("mechanism", c.get("type", "2-FF synchronizer"))
                md_lines.append(f"- **{src} \u2192 {dst}**: {mech}")
            else:
                md_lines.append(f"- {c}")
        md_lines.append("")

    if reset_spec:
        strategy = reset_spec.get("strategy", "synchronous")
        rst_domains = reset_spec.get("domains", [])
        md_lines += [
            "## Reset Strategy",
            f"- **Strategy:** {strategy}",
        ]
        if rst_domains:
            md_lines.append(f"- **Domains:** {', '.join(rst_domains)}")
        md_lines.append("")

    cdc_required = ct.get("cdc_required", False)
    md_lines.append(f"**CDC required:** {'Yes' if cdc_required else 'No'}")
    md_lines.append("")

    atomic_write(arch_dir / "clock_tree.md", "\n".join(md_lines))


def _persist_block_diagram(project_root: str, bd_result: dict) -> None:
    """Write the block diagram to disk as JSON and Markdown.

    Files written:
      .socmate/block_diagram.json  -- structured block diagram for programmatic use
      arch/block_diagram.md     -- human-readable Markdown
    """
    socmate_dir = Path(project_root) / ".socmate"
    socmate_dir.mkdir(parents=True, exist_ok=True)
    arch_dir = Path(project_root) / ARCH_DOC_DIR
    arch_dir.mkdir(parents=True, exist_ok=True)

    from orchestrator.utils import atomic_write

    atomic_write(socmate_dir / "block_diagram.json", json.dumps(bd_result, indent=2, default=str))

    blocks = bd_result.get("blocks", [])
    connections = bd_result.get("connections", [])

    md_lines = ["# Block Diagram", ""]

    if blocks:
        md_lines += [
            "## Blocks",
            "",
            "| Block | Description | Tier | Est. Gates |",
            "|---|---|---|---|",
        ]
        for b in blocks:
            name = b.get("name", "?")
            desc = b.get("description", "")
            tier = b.get("tier", "—")
            gates = b.get("estimated_gates", "—")
            md_lines.append(f"| {name} | {desc} | {tier} | {gates} |")
        md_lines.append("")

    if connections:
        md_lines += [
            "## Connections",
            "",
            "| From | To | Interface | Data Width |",
            "|---|---|---|---|",
        ]
        for c in connections:
            src = c.get("from", "?")
            dst = c.get("to", "?")
            iface = c.get("interface", "—")
            width = c.get("data_width", "—")
            md_lines.append(f"| {src} | {dst} | {iface} | {width} |")
        md_lines.append("")

    atomic_write(arch_dir / "block_diagram.md", "\n".join(md_lines))


def _persist_register_spec(project_root: str, rs_result: dict) -> None:
    """Write the register spec to disk as JSON and Markdown.

    Files written:
      .socmate/register_spec.json  -- structured register spec for programmatic use
      arch/register_spec.md     -- human-readable Markdown
    """
    socmate_dir = Path(project_root) / ".socmate"
    socmate_dir.mkdir(parents=True, exist_ok=True)
    arch_dir = Path(project_root) / ARCH_DOC_DIR
    arch_dir.mkdir(parents=True, exist_ok=True)

    from orchestrator.utils import atomic_write

    atomic_write(socmate_dir / "register_spec.json", json.dumps(rs_result, indent=2, default=str))

    rs = rs_result.get("result", rs_result)
    reg_blocks = rs.get("blocks", [])

    md_lines = ["# Register Specification", ""]

    if reg_blocks:
        md_lines += [
            "## Register Blocks",
            "",
            "| Block | Config Regs | Status Regs |",
            "|---|---|---|",
        ]
        for b in reg_blocks:
            name = b.get("name", "?")
            num_cfg = b.get("num_config", 0)
            num_sts = b.get("num_status", 0)
            md_lines.append(f"| {name} | {num_cfg} | {num_sts} |")
        md_lines.append("")

    total = rs.get("total_blocks", len(reg_blocks))
    md_lines.append(f"**Total register blocks:** {total}")
    md_lines.append("")

    atomic_write(arch_dir / "register_spec.md", "\n".join(md_lines))


def _persist_ers(project_root: str, ers_result: dict) -> None:
    """Write the final ERS to disk as JSON and Markdown.

    Files written:
      .socmate/ers_spec.json  -- structured ERS for programmatic use
      arch/ers_spec.md     -- human-readable Markdown
    """
    socmate_dir = Path(project_root) / ".socmate"
    socmate_dir.mkdir(parents=True, exist_ok=True)
    arch_dir = Path(project_root) / ARCH_DOC_DIR
    arch_dir.mkdir(parents=True, exist_ok=True)

    from orchestrator.utils import atomic_write

    atomic_write(socmate_dir / "ers_spec.json", json.dumps(ers_result, indent=2, default=str))

    ers = ers_result.get("ers", {})
    md_lines = [
        f"# {ers.get('title', 'Engineering Requirements Specification')}",
        f"**Revision:** {ers.get('revision', '1.0')}",
        "",
        "## Summary",
        ers.get("summary", "_(no summary)_"),
        "",
    ]

    tech = ers.get("target_technology", {})
    if tech:
        md_lines += [
            "## Target Technology",
            f"- **PDK:** {tech.get('pdk', 'N/A')}",
            f"- **Process:** {tech.get('process_nm', 'N/A')} nm",
            f"- **Rationale:** {tech.get('rationale', 'N/A')}",
            "",
        ]

    sf = ers.get("speed_and_feeds", {})
    if sf:
        md_lines += [
            "## Speed & Feeds",
            f"- **Input data rate:** {sf.get('input_data_rate_mbps', 'N/A')} Mbps",
            f"- **Output data rate:** {sf.get('output_data_rate_mbps', 'N/A')} Mbps",
            f"- **Target clock:** {sf.get('target_clock_mhz', 'N/A')} MHz",
            f"- **Latency budget:** {sf.get('latency_budget_us', 'N/A')} us",
            f"- **Throughput:** {sf.get('throughput_requirements', 'N/A')}",
            "",
        ]

    func_reqs = ers.get("functional_requirements", [])
    if func_reqs:
        md_lines += ["## Functional Requirements"]
        for req in func_reqs:
            md_lines.append(f"- {req}")
        md_lines.append("")

    per_block = ers.get("per_block_requirements", [])
    if per_block:
        md_lines += ["## Per-Block Requirements"]
        for block in per_block:
            if isinstance(block, dict):
                md_lines.append(f"### {block.get('block_name', 'Unknown')}")
                for req in block.get("requirements", []):
                    md_lines.append(f"- {req}")
                if block.get("interface_protocol"):
                    md_lines.append(f"- **Interface:** {block['interface_protocol']}")
            md_lines.append("")

    constraints = ers.get("constraints", [])
    if constraints:
        md_lines += ["## Constraints"]
        for c in constraints:
            md_lines.append(f"- {c}")
        md_lines.append("")

    ver_reqs = ers.get("verification_requirements", [])
    if ver_reqs:
        md_lines += ["## Verification Requirements"]
        for r in ver_reqs:
            md_lines.append(f"- {r}")
        md_lines.append("")

    open_items = ers.get("open_items", [])
    if open_items:
        md_lines += ["## Open Items"]
        for item in open_items:
            md_lines.append(f"- {item}")
        md_lines.append("")

    atomic_write(arch_dir / "ers_spec.md", "\n".join(md_lines))


# ---------------------------------------------------------------------------
# Intermediate state persistence
# ---------------------------------------------------------------------------

def _persist_intermediate_state(state: dict, updates: dict) -> None:
    """Persist architecture state to disk after each specialist node.

    Merges the current graph state with the node's updates into
    ``ArchitectureState`` on disk so the observer (and other file-based
    consumers like the summarizer) can see intermediate results without
    waiting for the ``finalize_node``.

    When the ``block_diagram`` field is present in the updates, the
    block diagram visualization JSON (``block_diagram_viz.json``) is also
    regenerated so the ReactFlow canvas stays in sync with the memory map
    sidebar during architecture iteration loops.  Previously the viz was
    only written by ``create_documentation_node`` at the end of the
    pipeline, causing the canvas to show stale block names while the
    sidebar (sourced from ``architecture_state.json``) already reflected
    the latest data.

    Failures are logged but never propagate -- intermediate persistence
    is best-effort and must not break the graph.
    """
    import logging

    from orchestrator.architecture.state import load_state, save_state

    project_root = state.get("project_root", ".")
    log = logging.getLogger(__name__)

    try:
        arch_state = load_state(project_root)

        # Merge current graph state with the new updates from this node.
        # Updates take precedence (they contain the just-computed results).
        merged = {**state, **updates}

        # Identity / config (always overwrite from latest graph state)
        arch_state.requirements = merged.get("requirements", arch_state.requirements)
        arch_state.target_clock_mhz = merged.get(
            "target_clock_mhz", arch_state.target_clock_mhz,
        )
        if merged.get("pdk_config"):
            arch_state.pdk_config = merged["pdk_config"]
        if merged.get("prd_spec"):
            arch_state.prd_spec = merged["prd_spec"]

        # Specialist outputs (only overwrite if present)
        _FIELD_MAP = {
            "block_diagram": "block_diagram",
            "memory_map": "memory_map",
            "clock_tree": "clock_tree",
            "register_spec": "register_spec",
            "benchmark_data": "benchmark_results",
            "block_diagram_doc": "block_diagram_doc",
        }
        for graph_key, attr_name in _FIELD_MAP.items():
            val = merged.get(graph_key)
            if val is not None:
                setattr(arch_state, attr_name, val)

        save_state(arch_state, project_root)
        log.debug("Intermediate architecture state persisted (phase=%s)", merged.get("phase", "?"))

        # Keep block_diagram_viz.json in sync whenever blocks change.
        # This ensures the ReactFlow canvas reflects the latest block
        # names/connections even before create_documentation_node runs.
        block_diagram = merged.get("block_diagram")
        if block_diagram and block_diagram.get("blocks"):
            try:
                from orchestrator.architecture.specialists.block_diagram_doc import (
                    generate_block_diagram_doc,
                    persist_block_diagram_doc,
                )

                doc = generate_block_diagram_doc(
                    block_diagram=block_diagram,
                    memory_map=merged.get("memory_map"),
                    clock_tree=merged.get("clock_tree"),
                    register_spec=merged.get("register_spec"),
                    ers_spec=merged.get("prd_spec"),
                    design_name=(
                        merged.get("prd_spec", {}) or {}
                    ).get("prd", {}).get("title", ""),
                )
                persist_block_diagram_doc(doc, project_root)
                log.debug("Block diagram viz regenerated during intermediate persist")
            except Exception:
                log.warning(
                    "Failed to regenerate block diagram viz during intermediate persist",
                    exc_info=True,
                )
    except Exception:
        log.warning(
            "Failed to persist intermediate architecture state",
            exc_info=True,
        )


# ---------------------------------------------------------------------------
# Specialist nodes
# ---------------------------------------------------------------------------

async def gather_requirements_node(state: ArchGraphState) -> dict:
    """Gather requirements via the PRD specialist.

    Runs in two modes:
      1. Phase 1 (no answers yet): generates sizing questions for the architect.
      2. Phase 2 (answers provided): drafts the full PRD document.
    """
    from orchestrator.architecture.specialists.prd_spec import gather_prd

    human_response = state.get("human_response")
    has_hr = human_response is not None and isinstance(human_response, dict)

    _event(state, "Gather Requirements", "graph_node_enter", {
        "round": state["round"],
        "has_human_response": has_hr,
        "human_response_keys": list(human_response.keys()) if has_hr else [],
        "has_answers_key": "answers" in human_response if has_hr else False,
    })

    with _tracer.start_as_current_span("Gather Requirements (PRD)") as span:
        user_answers = None
        previous_questions = state.get("prd_questions")
        if has_hr:
            user_answers = human_response.get("answers")
            if (
                not user_answers
                and human_response.get("action") == "feedback"
                and human_response.get("feedback")
            ):
                user_answers = dict(state.get("prd_answers") or {})
                user_answers["architect_final_review_feedback"] = human_response.get("feedback", "")

        span.set_attribute("has_answers", user_answers is not None)
        if user_answers:
            span.set_attribute("answer_count", len(user_answers))

        result = await gather_prd(
            requirements=state["requirements"],
            pdk_summary=state["pdk_summary"],
            target_clock_mhz=state["target_clock_mhz"],
            user_answers=user_answers,
            previous_questions=previous_questions,
        )

        phase = result.get("phase", "questions")
        span.set_attribute("prd_phase", phase)

        update: dict = {"phase": "prd"}

        if phase == "prd_complete":
            prd_doc = result.get("prd", {})
            span.set_attribute("prd_sections", len(prd_doc))

            enriched_requirements = state["requirements"]
            if prd_doc.get("summary"):
                enriched_requirements = (
                    f"{state['requirements']}\n\n"
                    f"--- PRD SUMMARY ---\n{prd_doc['summary']}"
                )

            update["prd_spec"] = result
            if user_answers:
                update["prd_answers"] = user_answers
            update["requirements"] = enriched_requirements

            _persist_prd(state["project_root"], result, previous_questions, user_answers)

            _event(state, "Gather Requirements", "graph_node_exit", {
                "round": state["round"],
                "phase": "prd_complete",
            })
        else:
            questions = result.get("questions", [])
            span.set_attribute("question_count", len(questions))
            update["prd_questions"] = questions

            _event(state, "Gather Requirements", "graph_node_exit", {
                "round": state["round"],
                "phase": "questions",
                "question_count": len(questions),
            })

        return update


async def system_architecture_node(state: ArchGraphState) -> dict:
    """Generate the System Architecture Document (SAD) from the PRD.

    Analyzes the PRD to produce system-level architecture decisions:
    HW/FW/SW partitioning, system flows, technology rationale, and
    architecture decisions with rationale.
    """
    from orchestrator.architecture.specialists.sad_spec import generate_sad

    _event(state, "System Architecture", "graph_node_enter", {
        "round": state["round"],
    })

    with _tracer.start_as_current_span("System Architecture (SAD)") as span:
        span.set_attribute("round", state["round"])

        result = await generate_sad(
            prd_spec=state.get("prd_spec", {}),
            requirements=state["requirements"],
            pdk_summary=state["pdk_summary"],
        )

        sad_text = result.get("sad_text", "")
        span.set_attribute("sad_text_len", len(sad_text))

        _persist_sad(state["project_root"], result)

        _event(state, "System Architecture", "graph_node_exit", {
            "round": state["round"],
            "text_len": len(sad_text),
        })

        update = {"sad_spec": result, "phase": "sad"}
        _persist_intermediate_state(state, update)
        return update


async def functional_requirements_node(state: ArchGraphState) -> dict:
    """Generate the Functional Requirements Document (FRD) from PRD + SAD.

    Produces quantitative, measurable, testable functional requirements
    with acceptance criteria.
    """
    from orchestrator.architecture.specialists.frd_spec import generate_frd

    _event(state, "Functional Requirements", "graph_node_enter", {
        "round": state["round"],
    })

    with _tracer.start_as_current_span("Functional Requirements (FRD)") as span:
        span.set_attribute("round", state["round"])

        result = await generate_frd(
            prd_spec=state.get("prd_spec", {}),
            sad_spec=state.get("sad_spec", {}),
            requirements=state["requirements"],
        )

        frd_text = result.get("frd_text", "")
        span.set_attribute("frd_text_len", len(frd_text))

        _persist_frd(state["project_root"], result)

        _event(state, "Functional Requirements", "graph_node_exit", {
            "round": state["round"],
            "text_len": len(frd_text),
        })

        update = {"frd_spec": result, "phase": "frd"}
        _persist_intermediate_state(state, update)
        return update


async def block_diagram_node(state: ArchGraphState) -> dict:
    """Generate or refine the block diagram via LLM specialist."""
    from orchestrator.architecture.specialists.block_diagram import (
        analyze_block_diagram,
    )

    _event(state, "Block Diagram", "graph_node_enter", {"round": state["round"]})

    round_label = f" - Iteration #{state['round']}" if state["round"] > 1 else ""
    with _tracer.start_as_current_span(f"Block Diagram{round_label}") as span:
        span.set_attribute("round", state["round"])

        # Extract constraint feedback as strings for the LLM prompt
        constraint_feedback = None
        if state.get("constraint_result"):
            violations = state["constraint_result"].get("violations", [])
            if violations:
                constraint_feedback = [
                    v["violation"] if isinstance(v, dict) else v
                    for v in violations
                ]

        result = await analyze_block_diagram(
            requirements=state["requirements"],
            pdk_summary=state["pdk_summary"],
            target_clock_mhz=state["target_clock_mhz"],
            existing_diagram=state.get("block_diagram"),
            constraint_feedback=constraint_feedback,
            benchmark_data=state.get("benchmark_data"),
            human_feedback=state.get("human_feedback", ""),
            project_root=state["project_root"],
            ers_spec=state.get("prd_spec"),
        )

        if result.get("_parse_failed"):
            _event(state, "Block Diagram", "parse_failure", {
                "round": state["round"],
                "hint": "LLM returned non-JSON; proceeding with defaults",
            })

        block_count = len(result.get("blocks", []))
        question_count = len(result.get("questions", []))
        span.set_attribute("block_count", block_count)
        span.set_attribute("question_count", question_count)

        new_questions = list(result.get("questions", []))

        _event(state, "Block Diagram", "graph_node_exit", {
            "round": state["round"],
            "blocks": block_count,
            "questions": question_count,
        })

        update = {
            "block_diagram": result,
            "questions": new_questions,
            "phase": "block_diagram",
        }
        _persist_intermediate_state(state, update)
        _persist_block_diagram(state["project_root"], result)
        return update


async def memory_map_node(state: ArchGraphState) -> dict:
    """Generate memory map via LLM specialist.

    Bypassed by default for streaming designs. Set
    ``SOCMATE_ENABLE_MEMORY_MAP=1`` to run it. The legacy
    ``SOCMATE_SKIP_MEMORY_MAP=1`` still forces a skip.
    """
    if not _stage_enabled("SOCMATE_ENABLE_MEMORY_MAP", "SOCMATE_SKIP_MEMORY_MAP"):
        _event(state, "Memory Map", "graph_node_enter", {"round": state["round"], "skipped": True})
        empty = {"skipped": True,
                 "result": {"skipped": True, "peripheral_count": 0, "peripherals": []},
                 "rationale": "bypassed by default; set SOCMATE_ENABLE_MEMORY_MAP=1 to run"}
        _event(state, "Memory Map", "graph_node_exit", {"round": state["round"], "skipped": True})
        update = {"memory_map": empty, "phase": "memory_map"}
        _persist_intermediate_state(state, update)
        _persist_memory_map(state["project_root"], empty)
        return update

    from orchestrator.architecture.specialists.memory_map import (
        analyze_memory_map,
    )

    _event(state, "Memory Map", "graph_node_enter", {"round": state["round"]})

    round_label = f" - Iteration #{state['round']}" if state["round"] > 1 else ""
    with _tracer.start_as_current_span(f"Memory Map{round_label}") as span:
        span.set_attribute("round", state["round"])
        result = await analyze_memory_map(
            block_diagram=state["block_diagram"],
            target_clock_mhz=state["target_clock_mhz"],
            requirements=state.get("requirements", ""),
            ers_spec=state.get("prd_spec"),
        )
        periph_count = result.get("result", {}).get("peripheral_count", 0)
        span.set_attribute("peripheral_count", periph_count)

        _event(state, "Memory Map", "graph_node_exit", {
            "round": state["round"],
            "peripheral_count": periph_count,
        })

        update = {"memory_map": result, "phase": "memory_map"}
        _persist_intermediate_state(state, update)
        _persist_memory_map(state["project_root"], result)
        return update


async def clock_tree_node(state: ArchGraphState) -> dict:
    """Generate clock tree via LLM specialist.

    Bypassed by default for streaming soft-IP exploration. Set
    ``SOCMATE_ENABLE_CLOCK_TREE=1`` to run it.
    """
    if not _stage_enabled("SOCMATE_ENABLE_CLOCK_TREE", "SOCMATE_SKIP_CLOCK_TREE"):
        _event(state, "Clock Tree", "graph_node_enter", {"round": state["round"], "skipped": True})
        empty = {"skipped": True,
                 "result": {"skipped": True, "num_domains": 0, "domains": []},
                 "rationale": "bypassed by default; set SOCMATE_ENABLE_CLOCK_TREE=1 to run"}
        _event(state, "Clock Tree", "graph_node_exit", {"round": state["round"], "skipped": True})
        update = {"clock_tree": empty, "phase": "clock_tree"}
        _persist_intermediate_state(state, update)
        _persist_clock_tree(state["project_root"], empty)
        return update

    from orchestrator.architecture.specialists.clock_tree import (
        analyze_clock_tree,
    )

    _event(state, "Clock Tree", "graph_node_enter", {"round": state["round"]})

    round_label = f" - Iteration #{state['round']}" if state["round"] > 1 else ""
    with _tracer.start_as_current_span(f"Clock Tree{round_label}") as span:
        span.set_attribute("round", state["round"])
        result = await analyze_clock_tree(
            block_diagram=state["block_diagram"],
            target_clock_mhz=state["target_clock_mhz"],
            requirements=state.get("requirements", ""),
        )
        num_domains = result.get("result", {}).get("num_domains", 0)
        span.set_attribute("num_domains", num_domains)

        _event(state, "Clock Tree", "graph_node_exit", {
            "round": state["round"],
            "num_domains": num_domains,
        })

        update = {"clock_tree": result, "phase": "clock_tree"}
        _persist_intermediate_state(state, update)
        _persist_clock_tree(state["project_root"], result)
        return update


async def register_spec_node(state: ArchGraphState) -> dict:
    """Generate register spec via LLM specialist.

    Bypassed by default for streaming designs with no CSR surface. Set
    ``SOCMATE_ENABLE_REGISTER_SPEC=1`` to run it. The legacy
    ``SOCMATE_SKIP_REGISTER_SPEC=1`` still forces a skip.
    """
    if not _stage_enabled("SOCMATE_ENABLE_REGISTER_SPEC", "SOCMATE_SKIP_REGISTER_SPEC"):
        _event(state, "Register Spec", "graph_node_enter", {"round": state["round"], "skipped": True})
        empty = {"skipped": True,
                 "result": {"skipped": True, "total_blocks": 0, "register_blocks": []},
                 "rationale": "bypassed by default; set SOCMATE_ENABLE_REGISTER_SPEC=1 to run"}
        _event(state, "Register Spec", "graph_node_exit", {"round": state["round"], "skipped": True})
        update = {"register_spec": empty, "phase": "register_spec"}
        _persist_intermediate_state(state, update)
        _persist_register_spec(state["project_root"], empty)
        return update

    from orchestrator.architecture.specialists.register_spec import (
        analyze_register_spec,
    )

    _event(state, "Register Spec", "graph_node_enter", {"round": state["round"]})

    round_label = f" - Iteration #{state['round']}" if state["round"] > 1 else ""
    with _tracer.start_as_current_span(f"Register Spec{round_label}") as span:
        span.set_attribute("round", state["round"])
        result = await analyze_register_spec(
            block_diagram=state["block_diagram"],
            memory_map=state.get("memory_map"),
            requirements=state.get("requirements", ""),
        )
        total_blocks = result.get("result", {}).get("total_blocks", 0)
        span.set_attribute("register_block_count", total_blocks)

        _event(state, "Register Spec", "graph_node_exit", {
            "round": state["round"],
            "register_blocks": total_blocks,
        })

        update = {"register_spec": result, "phase": "register_spec"}
        _persist_intermediate_state(state, update)
        _persist_register_spec(state["project_root"], result)
        return update


async def constraint_check_node(state: ArchGraphState) -> dict:
    """Check constraints via LLM specialist.

    Returns violations as dicts with category classification:
    structural (requires human input) vs auto_fixable (LLM can iterate).
    """
    from orchestrator.architecture.constraints import check_constraints

    _event(state, "Constraint Check", "graph_node_enter", {"round": state["round"]})

    round_label = f" - Iteration #{state['round']}" if state["round"] > 1 else ""
    with _tracer.start_as_current_span(f"Constraint Check{round_label}") as span:
        span.set_attribute("round", state["round"])

        mm = _optional_stage_payload(state.get("memory_map"))
        ct = _optional_stage_payload(state.get("clock_tree"))
        rs = _optional_stage_payload(state.get("register_spec"))

        violations = await check_constraints(
            block_diagram=state["block_diagram"],
            memory_map={"result": mm.get("result", mm)} if mm else {},
            clock_tree={"result": ct.get("result", ct)} if ct else {},
            register_spec={"result": rs.get("result", rs)} if rs else {},
            benchmark_results=state.get("benchmark_data"),
            pdk_config=state.get("pdk_config"),
            requirements=state.get("requirements", ""),
            ers_spec=state.get("prd_spec"),
            project_root=state["project_root"],
        )

        all_pass = len(violations) == 0
        has_structural = any(
            v.get("category") == "structural" for v in violations
        )

        span.set_attribute("violation_count", len(violations))
        span.set_attribute("all_pass", all_pass)
        span.set_attribute("has_structural", has_structural)

        violations_entry = {
            "round": state["round"],
            "violations": violations,
            "all_pass": all_pass,
            "has_structural": has_structural,
        }

        _event(state, "Constraint Check", "graph_node_exit", {
            "round": state["round"],
            "violations": len(violations),
            "all_pass": all_pass,
            "has_structural": has_structural,
        })

        update = {
            "constraint_result": {
                "violations": violations,
                "all_pass": all_pass,
                "has_structural": has_structural,
            },
            "violations_history": [violations_entry],
            "phase": "constraints",
        }
        _persist_intermediate_state(state, update)
        return update


def _is_non_silicon_validation_block(block: dict) -> bool:
    """Return True for architecture-only validation/reporting blocks.

    These blocks are important ERS/DV collateral, but they are not RTL pipeline
    work items and should not be handed to lint/sim/synthesis/backend.
    """
    name = str(block.get("name", "")).lower()
    desc = str(block.get("description", "")).lower()
    rtl_target = str(block.get("rtl_target", "") or "").strip()
    estimated_gates = block.get("estimated_gates", None)
    text = f"{name} {desc}"
    marker = any(
        token in text
        for token in (
            "flow_smoke",
            "smoke_check",
            "validation harness",
            "validation artifact",
            "non-synthesizable",
            "non-silicon",
        )
    )
    return marker and (not rtl_target or estimated_gates in (0, "0", None))


async def finalize_node(state: ArchGraphState) -> dict:
    """Finalize architecture: persist state and write block_specs.json."""
    from orchestrator.architecture.state import load_state, save_state

    _event(state, "Finalize Architecture", "graph_node_enter", {
        "round": state["round"],
    })

    with _tracer.start_as_current_span("Finalize Architecture") as span:
        span.set_attribute("round", state["round"])

        project_root = state["project_root"]

        # Persist all architecture decisions to state file
        arch_state = load_state(project_root)
        arch_state.requirements = state["requirements"]
        arch_state.target_clock_mhz = state["target_clock_mhz"]
        if state.get("prd_spec"):
            arch_state.prd_spec = state["prd_spec"]
        arch_state.block_diagram = state["block_diagram"]
        arch_state.memory_map = state.get("memory_map", {})
        arch_state.clock_tree = state.get("clock_tree", {})
        arch_state.register_spec = state.get("register_spec", {})
        if state.get("benchmark_data"):
            arch_state.benchmark_results = state["benchmark_data"]
        if state.get("pdk_config"):
            arch_state.pdk_config = state["pdk_config"]

        # Build block specs
        blocks = state["block_diagram"].get("blocks", [])
        block_specs = []
        skipped_non_silicon = []
        for block in blocks:
            if _is_non_silicon_validation_block(block):
                skipped_non_silicon.append(block.get("name", ""))
                continue
            spec = {
                "name": block.get("name", ""),
                "tier": block.get("tier", 1),
                "python_source": block.get("python_source", ""),
                "rtl_target": block.get("rtl_target",
                                        f"rtl/{block.get('name', 'unknown')}.v"),
                "testbench": block.get("testbench",
                                       f"tb/cocotb/test_{block.get('name', 'unknown')}.py"),
                "description": block.get("description", ""),
            }
            block_specs.append(spec)

        arch_state.block_specs = block_specs
        save_state(arch_state, project_root)

        # Write block_specs.json for RTL pipeline handoff (atomic write)
        from orchestrator.utils import atomic_write

        specs_path = Path(project_root) / ".socmate" / "block_specs.json"
        atomic_write(specs_path, json.dumps(block_specs, indent=2))

        span.set_attribute("block_count", len(block_specs))
        span.set_attribute("specs_path", str(specs_path))
        if skipped_non_silicon:
            span.set_attribute("skipped_non_silicon_blocks", ",".join(skipped_non_silicon))

        _event(state, "Finalize Architecture", "graph_node_exit", {
            "round": state["round"],
            "block_count": len(block_specs),
            "skipped_non_silicon_blocks": skipped_non_silicon,
        })

        return {
            "block_specs_path": str(specs_path),
            "success": True,
            "error": "",
            "phase": "finalize",
        }


# ---------------------------------------------------------------------------
# Documentation node
# ---------------------------------------------------------------------------

async def create_documentation_node(state: ArchGraphState) -> dict:
    """Generate block diagram visualization, ERS document, and HTML dashboard.

    1. Block diagram viz: ReactFlow-compatible JSON for the VS Code canvas.
    2. ERS: synthesizes all upstream docs into the final engineering spec.
    3. HTML dashboard: LLM-generated self-contained page with summary,
       block diagram, PRD, SAD, FRD, ERS, memory map, and clock tree.
    """
    from orchestrator.architecture.block_diagram_schema import (
        validate_block_diagram_json,
    )
    from orchestrator.architecture.specialists.block_diagram_doc import (
        generate_block_diagram_doc,
        persist_block_diagram_doc,
    )

    _event(state, "Create Documentation", "graph_node_enter", {
        "round": state["round"],
    })

    with _tracer.start_as_current_span("Create Documentation") as span:
        span.set_attribute("round", state["round"])

        block_diagram = state.get("block_diagram", {})
        if not block_diagram or not block_diagram.get("blocks"):
            span.set_attribute("error", "no_block_diagram")
            _event(state, "Create Documentation", "graph_node_exit", {
                "round": state["round"],
                "error": "No block diagram to document",
            })
            return {
                "phase": "documentation",
                "block_diagram_doc": None,
                "block_diagram_doc_path": "",
            }

        design_name = ""
        prd = state.get("prd_spec", {})
        if prd:
            design_name = prd.get("prd", {}).get("title", "")

        # --- 1. Block diagram visualization ---
        doc = generate_block_diagram_doc(
            block_diagram=block_diagram,
            memory_map=state.get("memory_map"),
            clock_tree=state.get("clock_tree"),
            register_spec=state.get("register_spec"),
            ers_spec=prd,
            design_name=design_name,
        )

        errors = validate_block_diagram_json(doc)
        span.set_attribute("validation_errors", len(errors))

        if errors:
            import logging
            log = logging.getLogger(__name__)
            log.warning(
                "Block diagram doc has %d validation error(s): %s",
                len(errors), "; ".join(errors[:5]),
            )
            span.set_attribute("validation_error_details", "; ".join(errors[:5]))

        arch = doc.get("architecture", {})
        node_count = len(arch.get("systemNodes", []))
        edge_count = len(arch.get("systemEdges", []))
        span.set_attribute("node_count", node_count)
        span.set_attribute("edge_count", edge_count)

        project_root = state["project_root"]
        doc_path = persist_block_diagram_doc(doc, project_root)

        # --- 2. Final ERS (Engineering Requirements Specification) ---
        ers_result = None
        try:
            from orchestrator.architecture.specialists.ers_doc import (
                generate_ers_doc,
            )

            ers_result = await generate_ers_doc(
                prd_spec=state.get("prd_spec"),
                sad_spec=state.get("sad_spec"),
                frd_spec=state.get("frd_spec"),
                block_diagram=block_diagram,
                memory_map=state.get("memory_map"),
                clock_tree=state.get("clock_tree"),
                register_spec=state.get("register_spec"),
            )

            _persist_ers(project_root, ers_result)
            span.set_attribute("ers_generated", True)
        except Exception:
            import logging
            logging.getLogger(__name__).warning(
                "Failed to generate ERS document", exc_info=True,
            )
            span.set_attribute("ers_generated", False)

        # --- 3. HTML Architecture Dashboard (Jinja2 template) ---
        dashboard_path = ""
        try:
            from orchestrator.architecture.specialists.dashboard_doc import (
                generate_dashboard,
            )
            from orchestrator.utils import atomic_write

            html = await generate_dashboard(
                prd_spec=state.get("prd_spec"),
                sad_spec=state.get("sad_spec"),
                frd_spec=state.get("frd_spec"),
                ers_spec=ers_result,
                block_diagram=block_diagram,
                memory_map=state.get("memory_map"),
                clock_tree=state.get("clock_tree"),
                register_spec=state.get("register_spec"),
                project_root=project_root,
            )
            arch_dir = Path(project_root) / ARCH_DOC_DIR
            arch_dir.mkdir(parents=True, exist_ok=True)
            dash_file = arch_dir / "dashboard.html"
            atomic_write(dash_file, html)
            dashboard_path = str(dash_file)
            span.set_attribute("dashboard_generated", True)
        except Exception:
            import logging
            logging.getLogger(__name__).warning(
                "Failed to generate HTML dashboard", exc_info=True,
            )
            span.set_attribute("dashboard_generated", False)

        _event(state, "Create Documentation", "graph_node_exit", {
            "round": state["round"],
            "node_count": node_count,
            "edge_count": edge_count,
            "validation_errors": len(errors),
            "ers_generated": ers_result is not None,
            "dashboard_path": dashboard_path,
            "path": str(doc_path),
        })

        result = {
            "phase": "documentation",
            "block_diagram_doc": doc,
            "block_diagram_doc_path": str(doc_path),
            "block_diagram_doc_validation_errors": errors,
        }
        if ers_result:
            result["ers_spec"] = ers_result
        return result


# ---------------------------------------------------------------------------
# Internal nodes
# ---------------------------------------------------------------------------

async def mark_success_node(state: ArchGraphState) -> dict:
    _event(state, "Architecture Complete", "graph_node_exit", {
        "round": state["round"],
        "blocks": len(state.get("block_diagram", {}).get("blocks", [])),
    })
    with _tracer.start_as_current_span("Architecture Complete") as span:
        span.set_attribute("round", state["round"])
        span.set_attribute("block_count",
                           len(state.get("block_diagram", {}).get("blocks", [])))
    return {"success": True, "error": ""}


# ---------------------------------------------------------------------------
# Final Review escalation (OK2DEV gate)
# ---------------------------------------------------------------------------

async def escalate_final_review_node(state: ArchGraphState) -> dict:
    """Escalate to human for final architecture sign-off before RTL.

    Presents the complete architecture summary and asks the architect
    to approve for development (OK2DEV) or request revisions (REVISE).
    """
    bd = state.get("block_diagram", {})
    blocks = bd.get("blocks", [])
    block_names = [b.get("name", "") for b in blocks]
    total_gates = sum(b.get("estimated_gates", 0) for b in blocks)

    # Read the architecture summary from disk if available
    summary_path = Path(_pr(state)) / ARCH_DOC_DIR / "summary_architecture.md"
    summary_text = ""
    if summary_path.exists():
        try:
            summary_text = summary_path.read_text(encoding="utf-8")
        except OSError:
            pass

    # Build PRD highlights
    prd = state.get("prd_spec", {})
    prd_doc = prd.get("prd", {}) if isinstance(prd, dict) else {}
    ers_title = prd_doc.get("title", "Untitled Design")
    ers_summary = prd_doc.get("summary", "")

    # Build concise block list for the payload
    block_summaries = []
    for b in blocks:
        block_summaries.append({
            "name": b.get("name", ""),
            "tier": b.get("tier", 1),
            "description": b.get("description", ""),
            "estimated_gates": b.get("estimated_gates", 0),
            "interfaces": list(b.get("interfaces", {}).keys())
                          if isinstance(b.get("interfaces"), dict) else [],
        })

    # Block specs path
    specs_path = state.get("block_specs_path", "")
    block_diagram_doc_errors = state.get("block_diagram_doc_validation_errors", [])

    _event(state, "Final Review", "graph_node_enter", {
        "round": state["round"],
        "block_count": len(blocks),
        "total_estimated_gates": total_gates,
    })

    payload = {
        "type": "final_review",
        "phase": "final_review",
        "round": state["round"],
        "title": ers_title,
        "ers_summary": ers_summary[:1000],
        "block_count": len(blocks),
        "block_names": block_names,
        "block_summaries": block_summaries,
        "total_estimated_gates": total_gates,
        "architecture_summary": summary_text[:5000] if summary_text else "(summary not yet generated)",
        "block_specs_path": specs_path,
        "block_diagram_doc_validation_errors": block_diagram_doc_errors,
        "constraint_rounds_used": state["round"],
        "max_rounds": state["max_rounds"],
        "supported_actions": ["accept", "feedback", "abort"],
        "instructions": (
            "Architecture is complete. Review the design summary above.\n\n"
            "Do not approve OK2DEV while block_diagram_doc_validation_errors is non-empty; "
            "request feedback to fix the block diagram references first.\n\n"
            "Actions:\n"
            "  - accept (OK2DEV): Approve the architecture and proceed to RTL generation.\n"
            "  - feedback (REVISE): Provide revision notes; architecture will re-iterate "
            "from Block Diagram with your feedback.\n"
            "  - abort: Cancel the architecture run.\n\n"
            "Use resume_architecture(action='accept') to approve, or "
            "resume_architecture(action='feedback', feedback='...revision notes...') to revise."
        ),
    }

    response = interrupt(payload)

    action = response.get("action", "abort") if isinstance(response, dict) else "abort"
    feedback_text = response.get("feedback", "") if isinstance(response, dict) else ""

    _event(state, "Final Review", "graph_node_exit", {
        "round": state["round"],
        "action": action,
        "has_feedback": bool(feedback_text),
    })

    updated: dict = {"human_response": response}
    updated["human_response_history"] = [{
        "phase": "final_review", "round": state["round"],
        "action": action, "response": response,
    }]
    if feedback_text:
        updated["human_feedback"] = feedback_text

    return updated


async def increment_round_node(state: ArchGraphState) -> dict:
    new_round = state["round"] + 1
    new_total = state.get("total_rounds", 0) + 1

    _event(state, "Constraint Iteration", "graph_node_enter", {
        "round": state["round"],
        "new_round": new_round,
        "total_rounds": new_total,
    })

    with _tracer.start_as_current_span(
        f"Constraint Iteration #{new_round} ({new_round}/{state['max_rounds']})"
    ) as span:
        span.set_attribute("round.previous", state["round"])
        span.set_attribute("round.new", new_round)
        span.set_attribute("total_rounds", new_total)
        span.set_attribute("max_rounds", state["max_rounds"])

    _event(state, "Constraint Iteration", "graph_node_exit", {
        "new_round": new_round,
        "total_rounds": new_total,
        "max_rounds": state["max_rounds"],
    })

    return {"round": new_round, "total_rounds": new_total}


# ---------------------------------------------------------------------------
# Escalation nodes (interrupt-based human-in-the-loop)
# ---------------------------------------------------------------------------

async def escalate_prd_node(state: ArchGraphState) -> dict:
    """Escalate to human to answer PRD sizing questions.

    Always fires after Phase 1 of the PRD specialist.  The interrupt
    payload contains the structured questions so the outer agent can
    present them to the user (e.g. via AskQuestion).

    The user's answers are returned via resume_architecture(action="continue")
    with the answers dict in the human_response.
    """
    questions = state.get("prd_questions", [])

    _event(state, "Escalate PRD", "graph_node_enter", {
        "round": state["round"],
        "question_count": len(questions),
        "questions": [
            {"id": q.get("id", ""), "question": q.get("question", ""),
             "category": q.get("category", ""), "options": q.get("options")}
            for q in questions[:20]
        ],
    })

    by_category: dict[str, list] = {}
    for q in questions:
        cat = q.get("category", "general")
        by_category.setdefault(cat, []).append(q)

    payload = {
        "type": "prd_questions",
        "phase": "prd",
        "round": state["round"],
        "questions": questions,
        "questions_by_category": by_category,
        "category_order": [
            "technology", "speed_and_feeds", "area", "power", "dataflow",
        ],
        "requirements_preview": state["requirements"][:500],
        "supported_actions": [
            "continue",   # user has provided answers, proceed to PRD draft
            "abort",      # stop architecture
        ],
        "instructions": (
            "Please answer the sizing questions below.  Return your answers "
            "as a dict mapping question IDs to answer strings, e.g.:\n"
            '  {"target_technology": "sky130", "input_data_rate": "270 Mbps", ...}\n'
            "Use action='continue' with answers in the response."
        ),
    }

    response = interrupt(payload)

    action = response.get("action", "abort") if isinstance(response, dict) else "abort"
    has_answers = (
        isinstance(response, dict)
        and isinstance(response.get("answers"), dict)
        and len(response.get("answers", {})) > 0
    )
    answer_keys = list(response.get("answers", {}).keys()) if has_answers else []

    _event(state, "Escalate PRD", "graph_node_exit", {
        "round": state["round"],
        "action": action,
        "has_answers": has_answers,
        "answer_count": len(answer_keys),
        "answer_keys": answer_keys[:10],
    })

    if not has_answers and action != "abort":
        _event(state, "Escalate PRD", "escalation_warning", {
            "warning": "Resumed without answers! Gather Requirements will re-generate questions.",
            "human_response_keys": list(response.keys()) if isinstance(response, dict) else [],
            "hint": "Use resume_architecture(action='continue', feedback='{...json answers...}')",
        })

    updated: dict = {"human_response": response}
    updated["human_response_history"] = [{
        "phase": "prd", "round": state["round"],
        "action": action, "response": response,
    }]
    return updated


def _block_diagram_summary(state: ArchGraphState) -> dict:
    """Build a compact summary of the current block diagram for interrupt payloads."""
    bd = state.get("block_diagram", {})
    blocks = bd.get("blocks", [])
    return {
        "block_count": len(blocks),
        "block_names": [b.get("name", "") for b in blocks],
        "total_estimated_gates": sum(b.get("estimated_gates", 0) for b in blocks),
        "tiers": {
            1: len([b for b in blocks if b.get("tier") == 1]),
            2: len([b for b in blocks if b.get("tier") == 2]),
            3: len([b for b in blocks if b.get("tier") == 3]),
        },
        "connection_count": len(bd.get("connections", [])),
        "reasoning": bd.get("reasoning", "")[:500],
    }


async def escalate_diagram_node(state: ArchGraphState) -> dict:
    """Escalate to human when block diagram has questions or ambiguities.

    This node calls interrupt() to pause the graph.
    """
    bd = state.get("block_diagram", {})
    questions = bd.get("questions", [])

    _event(state, "Escalate Diagram", "graph_node_enter", {
        "round": state["round"],
        "question_count": len(questions),
        "questions": [
            q.get("question", str(q)) if isinstance(q, dict) else str(q)
            for q in questions[:10]
        ],
    })

    payload = {
        "type": "architecture_review_needed",
        "phase": "block_diagram",
        "round": state["round"],
        "max_rounds": state["max_rounds"],
        "questions": questions,
        "block_diagram_summary": _block_diagram_summary(state),
        "supported_actions": [
            "continue",   # accept diagram and proceed to memory map
            "feedback",   # provide text feedback, re-run block diagram
            "abort",      # stop architecture
        ],
    }

    response = interrupt(payload)

    action = response.get("action", "abort") if isinstance(response, dict) else "abort"

    feedback_text = response.get("feedback", "") if isinstance(response, dict) else ""
    _event(state, "Escalate Diagram", "graph_node_exit", {
        "round": state["round"],
        "action": action,
        "feedback": feedback_text[:2000] if feedback_text else "",
    })

    updated: dict = {"human_response": response}
    updated["human_response_history"] = [{
        "phase": "block_diagram", "round": state["round"],
        "action": action, "response": response,
    }]
    if action == "feedback" and isinstance(response, dict):
        updated["human_feedback"] = response.get("feedback", "")

    return updated


async def escalate_constraints_node(state: ArchGraphState) -> dict:
    """Escalate to human on structural constraint violations.

    Fires when violations involve peripheral count vs block diagram
    disagreement, memory overlaps, connectivity, or SRAM overflow.
    """
    violations = state.get("constraint_result", {}).get("violations", [])
    structural = [v for v in violations if v.get("category") == "structural"]

    _event(state, "Escalate Constraints", "graph_node_enter", {
        "round": state["round"],
        "structural_count": len(structural),
        "total_violations": len(violations),
        "violations": [
            {"violation": v.get("violation", str(v))[:300],
             "category": v.get("category", "")}
            for v in violations[:15]
        ],
        "structural_violations": [
            {"violation": v.get("violation", str(v))[:300],
             "category": v.get("category", "")}
            for v in structural[:10]
        ],
    })

    payload = {
        "type": "architecture_review_needed",
        "phase": "constraints",
        "round": state["round"],
        "max_rounds": state["max_rounds"],
        "violations": violations,
        "structural_violations": structural,
        "block_diagram_summary": _block_diagram_summary(state),
        "memory_map_summary": {
            "peripheral_count": state.get("memory_map", {}).get(
                "result", {}
            ).get("peripheral_count", 0),
        },
        "supported_actions": [
            "retry",      # re-run block diagram with violations as feedback
            "accept",     # accept despite violations, finalize
            "feedback",   # provide text feedback, re-run block diagram
            "abort",      # stop architecture
        ],
    }

    response = interrupt(payload)

    action = response.get("action", "abort") if isinstance(response, dict) else "abort"

    feedback_text = response.get("feedback", "") if isinstance(response, dict) else ""
    _event(state, "Escalate Constraints", "graph_node_exit", {
        "round": state["round"],
        "action": action,
        "feedback": feedback_text[:2000] if feedback_text else "",
    })

    updated: dict = {"human_response": response}
    updated["human_response_history"] = [{
        "phase": "constraints", "round": state["round"],
        "action": action, "response": response,
    }]
    if action == "feedback" and isinstance(response, dict):
        updated["human_feedback"] = response.get("feedback", "")

    return updated


async def escalate_exhausted_node(state: ArchGraphState) -> dict:
    """Escalate to human when max constraint rounds are exhausted.

    Always fires instead of silently ending the graph.
    """
    violations = state.get("constraint_result", {}).get("violations", [])
    history = state.get("violations_history", [])

    _event(state, "Escalate Exhausted", "graph_node_enter", {
        "round": state["round"],
        "max_rounds": state["max_rounds"],
        "remaining_violations": len(violations),
        "violations": [
            {"violation": v.get("violation", str(v))[:300],
             "category": v.get("category", "")}
            for v in violations[:15]
        ],
    })

    payload = {
        "type": "architecture_review_needed",
        "phase": "max_rounds_exhausted",
        "round": state["round"],
        "max_rounds": state["max_rounds"],
        "violations": violations,
        "violations_history": history[-3:],  # last 3 rounds of history
        "block_diagram_summary": _block_diagram_summary(state),
        "supported_actions": [
            "retry",      # reset round counter, try again with feedback
            "accept",     # accept despite violations, finalize
            "feedback",   # provide text feedback, reset and retry
            "abort",      # stop architecture
        ],
    }

    response = interrupt(payload)

    action = response.get("action", "abort") if isinstance(response, dict) else "abort"

    feedback_text = response.get("feedback", "") if isinstance(response, dict) else ""
    _event(state, "Escalate Exhausted", "graph_node_exit", {
        "round": state["round"],
        "action": action,
        "feedback": feedback_text[:2000] if feedback_text else "",
    })

    updated: dict = {"human_response": response}
    updated["human_response_history"] = [{
        "phase": "max_rounds_exhausted", "round": state["round"],
        "action": action, "response": response,
    }]
    if action in ("retry", "feedback"):
        # Reset round counter on retry/feedback (but NOT total_rounds)
        updated["round"] = 1
    if action == "feedback" and isinstance(response, dict):
        updated["human_feedback"] = response.get("feedback", "")

    return updated


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_after_prd(state: ArchGraphState) -> str:
    """Route after Gather Requirements: need user answers or PRD is complete."""
    prd = state.get("prd_spec")
    phase = "prd_complete" if prd else "questions"

    if phase == "prd_complete":
        target = "System Architecture"
    else:
        target = "Escalate PRD"

    span = trace.get_current_span()
    if span.is_recording():
        span.update_name(f"Route: After PRD - {phase.upper()}")
        span.add_event("route", {"from": "Gather Requirements", "to": target})
    return target

route_after_prd.__edge_labels__ = {
    "Escalate PRD": "QUESTIONS",
    "System Architecture": "PRD_COMPLETE",
}


def route_after_prd_escalation(state: ArchGraphState) -> str:
    """Route after user answers PRD sizing questions."""
    response = state.get("human_response", {}) or {}
    action = response.get("action", "abort")

    if action == "abort":
        target = "Abort"
    else:
        target = "Gather Requirements"

    span = trace.get_current_span()
    if span.is_recording():
        span.update_name(f"Route: After PRD Escalation - {action.upper()}")
        span.add_event("route", {"from": "Escalate PRD", "to": target, "action": action})
    return target

route_after_prd_escalation.__edge_labels__ = {
    "Gather Requirements": "CONTINUE",
    "Abort": "ABORT",
}


def review_diagram(state: ArchGraphState) -> str:
    """Route after block diagram: check if LLM returned questions or empty blocks."""
    bd = state.get("block_diagram", {})
    questions = bd.get("questions", [])
    blocks = bd.get("blocks", [])
    has_questions = len(questions) > 0
    has_no_blocks = len(blocks) == 0

    if has_questions or has_no_blocks:
        target = "Escalate Diagram"
    else:
        target = "Memory Map"

    span = trace.get_current_span()
    if span.is_recording():
        if has_no_blocks:
            label = "Empty Diagram (failure)"
        elif has_questions:
            label = f"{len(questions)} Questions"
        else:
            label = "Clean"
        span.update_name(f"Route: Review Diagram - {label}")
        span.add_event("route", {
            "from": "Block Diagram",
            "to": target,
            "questions": len(questions),
            "blocks": len(blocks),
        })
    return target

review_diagram.__edge_labels__ = {
    "Escalate Diagram": "QUESTIONS",
    "Memory Map": "CLEAN",
}


def route_after_diagram_escalation(state: ArchGraphState) -> str:
    """Route after human reviews diagram questions."""
    response = state.get("human_response", {}) or {}
    action = response.get("action", "abort")

    if action == "abort":
        target = "Abort"
    elif action == "feedback":
        target = "Block Diagram"
    else:
        target = "Memory Map"

    span = trace.get_current_span()
    if span.is_recording():
        span.update_name(f"Route: After Diagram Escalation - {action.upper()}")
        span.add_event("route", {"from": "Escalate Diagram", "to": target, "action": action})
    return target

route_after_diagram_escalation.__edge_labels__ = {
    "Memory Map": "CONTINUE",
    "Block Diagram": "FEEDBACK",
    "Abort": "ABORT",
}


def route_after_constraints(state: ArchGraphState) -> str:
    """Three-way route based on constraint check results.

    - PASS: all constraints satisfied -> finalize
    - STRUCTURAL: violations require human input -> escalate
    - AUTO_FIX: violations are auto-fixable by LLM -> iterate
    """
    cr = state.get("constraint_result", {})
    all_pass = cr.get("all_pass", False)
    has_structural = cr.get("has_structural", False)

    if all_pass:
        target = "Finalize Architecture"
    elif has_structural:
        target = "Escalate Constraints"
    else:
        target = "Constraint Iteration"

    span = trace.get_current_span()
    if span.is_recording():
        if all_pass:
            outcome = "All Passed"
        elif has_structural:
            outcome = "Structural Violations"
        else:
            outcome = "Auto-fixable Violations"
        span.update_name(f"Route: After Constraints - {outcome}")
        span.add_event("route", {
            "from": "Constraint Check",
            "to": target,
            "round": state["round"],
            "all_pass": all_pass,
            "has_structural": has_structural,
        })
    return target

route_after_constraints.__edge_labels__ = {
    "Finalize Architecture": "PASS",
    "Escalate Constraints": "STRUCTURAL",
    "Constraint Iteration": "AUTO_FIX",
}


def route_after_constraint_escalation(state: ArchGraphState) -> str:
    """Route after human reviews structural constraint violations."""
    response = state.get("human_response", {}) or {}
    action = response.get("action", "abort")

    if action == "abort":
        target = "Abort"
    elif action == "accept":
        target = "Finalize Architecture"
    elif action == "feedback":
        target = "Block Diagram"
    else:  # retry
        target = "Block Diagram"

    span = trace.get_current_span()
    if span.is_recording():
        span.update_name(f"Route: After Constraint Escalation - {action.upper()}")
        span.add_event("route", {
            "from": "Escalate Constraints", "to": target, "action": action,
        })
    return target

route_after_constraint_escalation.__edge_labels__ = {
    "Block Diagram": "RETRY",
    "Finalize Architecture": "ACCEPT",
    "Abort": "ABORT",
}


def route_after_final_review(state: ArchGraphState) -> str:
    """Route after human reviews the final architecture: OK2DEV or REVISE."""
    response = state.get("human_response", {}) or {}
    action = response.get("action", "abort")

    if action == "abort":
        target = "Abort"
    elif action in ("accept", "continue"):
        # OK2DEV -- proceed to completion
        target = "Architecture Complete"
    elif action == "feedback":
        target = _feedback_revision_target(response.get("feedback", ""))
    else:
        # Default to accept
        target = "Architecture Complete"

    _event_data = {
        "from": "Final Review",
        "to": target,
        "action": action,
    }
    span = trace.get_current_span()
    if span.is_recording():
        span.update_name(f"Route: After Final Review - {action.upper()}")
        span.add_event("route", _event_data)
    return target

route_after_final_review.__edge_labels__ = {
    "Architecture Complete": "OK2DEV",
    "Gather Requirements": "REVISE_PRD",
    "System Architecture": "REVISE_SAD",
    "Functional Requirements": "REVISE_FRD",
    "Block Diagram": "REVISE",
    "Abort": "ABORT",
}


def route_after_increment(state: ArchGraphState) -> str:
    """Route: continue iterating or escalate on exhaustion.

    Fix #6: also check total_rounds (lifetime counter) against a hard
    ceiling of max_rounds * 3 to prevent infinite retry loops when the
    user keeps choosing 'retry' at Escalate Exhausted.
    """
    max_rounds = state["max_rounds"]
    total_rounds = state.get("total_rounds", 0)
    hard_ceiling = max_rounds * 3

    exhausted = state["round"] > max_rounds or total_rounds > hard_ceiling
    target = "Escalate Exhausted" if exhausted else "Block Diagram"
    span = trace.get_current_span()
    if span.is_recording():
        if exhausted:
            label = f"Max Rounds Exhausted ({max_rounds}, total={total_rounds})"
        else:
            label = f"Round {state['round']}/{max_rounds} (total={total_rounds})"
        span.update_name(f"Route: After Iteration - {label}")
        span.add_event("route", {
            "from": "Constraint Iteration",
            "to": target,
            "round": state["round"],
            "total_rounds": total_rounds,
            "exhausted": exhausted,
        })
    return target

route_after_increment.__edge_labels__ = {
    "Block Diagram": "CONTINUE",
    "Escalate Exhausted": "EXHAUSTED",
}


def route_after_exhausted_escalation(state: ArchGraphState) -> str:
    """Route after human reviews exhausted-rounds escalation."""
    response = state.get("human_response", {}) or {}
    action = response.get("action", "abort")

    if action == "abort":
        target = "Abort"
    elif action == "accept":
        target = "Finalize Architecture"
    else:  # retry or feedback
        target = "Block Diagram"

    span = trace.get_current_span()
    if span.is_recording():
        span.update_name(f"Route: After Exhausted Escalation - {action.upper()}")
        span.add_event("route", {
            "from": "Escalate Exhausted", "to": target, "action": action,
        })
    return target

route_after_exhausted_escalation.__edge_labels__ = {
    "Block Diagram": "RETRY",
    "Finalize Architecture": "ACCEPT",
    "Abort": "ABORT",
}


# ---------------------------------------------------------------------------
# Terminal nodes
# ---------------------------------------------------------------------------

async def abort_node(state: ArchGraphState) -> dict:
    """Human chose to abort the architecture."""
    _event(state, "Abort", "graph_node_exit", {"round": state["round"]})
    return {
        "success": False,
        "error": "Architecture aborted by user.",
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_architecture_graph(checkpointer=None):
    """Build and compile the architecture decision graph.

    Runs as an autonomous background task with interrupt-based escalation.
    The checkpointer provides durability across MCP server restarts.

    Args:
        checkpointer: LangGraph checkpointer for state persistence.
            Use MemorySaver for tests, AsyncSqliteSaver for production.

    Returns:
        Compiled StateGraph ready for invoke/ainvoke.
    """
    graph = StateGraph(ArchGraphState)

    # PRD (requirements gathering) node
    graph.add_node("Gather Requirements", gather_requirements_node)

    # Document hierarchy nodes (SAD, FRD)
    graph.add_node("System Architecture", system_architecture_node)
    graph.add_node("Functional Requirements", functional_requirements_node)

    # Specialist nodes
    graph.add_node("Block Diagram", block_diagram_node)
    graph.add_node("Memory Map", memory_map_node)
    graph.add_node("Clock Tree", clock_tree_node)
    graph.add_node("Register Spec", register_spec_node)
    graph.add_node("Constraint Check", constraint_check_node)
    graph.add_node("Finalize Architecture", finalize_node)
    graph.add_node("Create Documentation", create_documentation_node)

    # Internal nodes
    graph.add_node("Architecture Complete", mark_success_node)
    graph.add_node("Constraint Iteration", increment_round_node)
    graph.add_node("Abort", abort_node)

    # Escalation nodes (interrupt-based)
    graph.add_node("Escalate PRD", escalate_prd_node)
    graph.add_node("Escalate Diagram", escalate_diagram_node)
    graph.add_node("Escalate Constraints", escalate_constraints_node)
    graph.add_node("Escalate Exhausted", escalate_exhausted_node)
    graph.add_node("Final Review", escalate_final_review_node)

    # --- Edges ---

    # START -> Gather Requirements -> route (questions or PRD complete)
    graph.add_edge(START, "Gather Requirements")
    graph.add_conditional_edges("Gather Requirements", route_after_prd)

    # Escalate PRD -> user answers -> back to Gather Requirements (Phase 2)
    graph.add_conditional_edges("Escalate PRD", route_after_prd_escalation)

    # PRD_COMPLETE -> System Architecture -> Functional Requirements -> Block Diagram
    graph.add_edge("System Architecture", "Functional Requirements")
    graph.add_edge("Functional Requirements", "Block Diagram")

    # Block Diagram -> review (conditional)
    graph.add_conditional_edges("Block Diagram", review_diagram)

    # Escalate Diagram -> route after escalation
    graph.add_conditional_edges("Escalate Diagram", route_after_diagram_escalation)

    # Memory Map -> Clock Tree -> Register Spec -> Constraint Check
    graph.add_edge("Memory Map", "Clock Tree")
    graph.add_edge("Clock Tree", "Register Spec")
    graph.add_edge("Register Spec", "Constraint Check")

    # After constraints: 3-way route
    graph.add_conditional_edges("Constraint Check", route_after_constraints)

    # Escalate Constraints -> route after escalation
    graph.add_conditional_edges(
        "Escalate Constraints", route_after_constraint_escalation,
    )

    # Auto-fix path: Constraint Iteration -> check rounds
    graph.add_conditional_edges("Constraint Iteration", route_after_increment)

    # Escalate Exhausted -> route after escalation
    graph.add_conditional_edges(
        "Escalate Exhausted", route_after_exhausted_escalation,
    )

    # Terminal paths: Finalize -> Documentation -> Final Review -> OK2DEV/REVISE
    graph.add_edge("Finalize Architecture", "Create Documentation")
    graph.add_edge("Create Documentation", "Final Review")
    graph.add_conditional_edges("Final Review", route_after_final_review)
    graph.add_edge("Architecture Complete", END)
    graph.add_edge("Abort", END)

    return graph.compile(checkpointer=checkpointer)
