# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Observability LLM -- asynchronous observer that summarises pipeline state.

Runs as a fire-and-forget ``asyncio.Task`` after every ``graph_node_exit``
event via the exit hook registry in :mod:`event_stream`.  Produces
human-readable markdown summaries at ``arch/summary_{stage}.md``.

Three stages are tracked:

* **architecture** -- architecture spec derived from ``architecture_state.json``
* **frontend** -- RTL pipeline execution status from ``pipeline_events.jsonl``
* **backend** -- physical-design execution status from backend-tagged events

A capable model (``sonnet-4.5``) is used by default for higher-quality
observer summaries.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Debounce / cooldown
# ---------------------------------------------------------------------------
_COOLDOWN_S = 120.0  # Increased from 30s -- observer was 22% of LLM calls
_last_call: dict[str, float] = {}  # stage -> epoch
_locks: dict[str, asyncio.Lock] = {}  # stage -> lock

_SUMMARY_DIR = "arch"

# Fix #16: observer enabled flag and significant nodes filter.
# Loaded from config.yaml on first call. Only these nodes trigger the observer.
_observer_enabled: bool | None = None  # None = not yet loaded

_SIGNIFICANT_NODES: dict[str, set[str]] = {
    "architecture": {
        "Gather Requirements", "Block Diagram", "Constraint Check",
        "Finalize Architecture", "Architecture Complete",
    },
    "frontend": {
        "generate_rtl", "Generate RTL",
        "block_done", "Block Done",
        "pipeline_complete", "Pipeline Complete",
    },
    "backend": {
        "init_block", "Init Block", "timing_signoff", "Timing Signoff",
        "backend_complete", "Backend Complete",
    },
}


def _is_observer_enabled() -> bool:
    """Check whether the observer is enabled (from config.yaml).

    Default is now ``False`` -- the observer was responsible for ~22% of
    the run's LLM call budget for a feature most users don't actively
    consume, so it's opt-in. Set ``observer.enabled: true`` in
    ``orchestrator/config.yaml`` or the env var ``CORESMITH_OBSERVER=1``
    to re-enable.
    """
    import os as _os
    global _observer_enabled
    if _observer_enabled is not None:
        return _observer_enabled
    # Env-var override takes precedence over config.yaml so power users
    # can flip it on without editing the file.
    env = _os.environ.get("CORESMITH_OBSERVER", "").strip().lower()
    if env in {"1", "true", "yes", "on"}:
        _observer_enabled = True
        return True
    if env in {"0", "false", "no", "off"}:
        _observer_enabled = False
        return False
    try:
        import yaml
        cfg_path = Path(__file__).resolve().parents[1] / "config.yaml"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f) or {}
            obs_cfg = cfg.get("observer", {})
            _observer_enabled = bool(obs_cfg.get("enabled", False))
            global _COOLDOWN_S
            if "cooldown_s" in obs_cfg:
                _COOLDOWN_S = float(obs_cfg["cooldown_s"])
        else:
            _observer_enabled = False
    except Exception:
        _observer_enabled = False
    return _observer_enabled


def _is_significant_node(node_name: str, stage: str) -> bool:
    """Check if *node_name* is in the significant set for *stage*."""
    sig = _SIGNIFICANT_NODES.get(stage, set())
    return node_name in sig


def _get_lock(stage: str) -> asyncio.Lock:
    if stage not in _locks:
        _locks[stage] = asyncio.Lock()
    return _locks[stage]


# ---------------------------------------------------------------------------
# Stage detection
# ---------------------------------------------------------------------------

def _detect_stage(event_record: dict) -> str:
    """Determine which graph stage an event belongs to."""
    graph = event_record.get("graph", "")
    if graph == "architecture":
        return "architecture"
    if graph == "backend":
        return "backend"
    return "frontend"


# ---------------------------------------------------------------------------
# Context gatherers
# ---------------------------------------------------------------------------

def _gather_architecture_context(project_root: str) -> str:
    """Read architecture document files and format as context for the LLM."""
    state_path = Path(project_root) / _SUMMARY_DIR / "architecture_state.json"
    prd_path = Path(project_root) / _SUMMARY_DIR / "prd_spec.json"

    state: dict = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    # Read PRD from standalone file
    prd_data: dict = {}
    if prd_path.exists():
        try:
            prd_data = json.loads(prd_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    # Fall back to PRD embedded in architecture state
    if not prd_data and state.get("prd_spec"):
        prd_data = state["prd_spec"]

    if not state and not prd_data:
        return "No architecture state or PRD found yet."

    parts: list[str] = []

    # PRD (Product Requirements Document)
    prd_doc = prd_data.get("prd", {})
    if prd_doc:
        parts.append("## Product Requirements Document (PRD)")
        if prd_doc.get("title"):
            parts.append(f"**{prd_doc['title']}** (rev {prd_doc.get('revision', '?')})")
        if prd_doc.get("summary"):
            parts.append(f"\n{prd_doc['summary']}")

        tech = prd_doc.get("target_technology", {})
        if tech:
            parts.append("\n### Target Technology")
            parts.append(f"- PDK: {tech.get('pdk', 'N/A')}, {tech.get('process_nm', '?')} nm")
            parts.append(f"- Rationale: {tech.get('rationale', 'N/A')}")

        sf = prd_doc.get("speed_and_feeds", {})
        if sf:
            parts.append("\n### Speed & Feeds")
            parts.append(f"- Input rate: {sf.get('input_data_rate_mbps', 'N/A')} Mbps")
            parts.append(f"- Output rate: {sf.get('output_data_rate_mbps', 'N/A')} Mbps")
            parts.append(f"- Clock: {sf.get('target_clock_mhz', 'N/A')} MHz")
            parts.append(f"- Latency: {sf.get('latency_budget_us', 'N/A')} us")
            if sf.get("throughput_requirements"):
                parts.append(f"- Throughput: {sf['throughput_requirements']}")

        area = prd_doc.get("area_budget", {})
        if area:
            parts.append("\n### Area Budget")
            parts.append(f"- Max gates: {area.get('max_gate_count', 'N/A')}")
            parts.append(f"- Max die area: {area.get('max_die_area_mm2', 'N/A')} mm²")

        power = prd_doc.get("power_budget", {})
        if power:
            parts.append("\n### Power Budget")
            parts.append(f"- Total: {power.get('total_power_mw', 'N/A')} mW")
            domains = power.get("power_domains", [])
            if domains:
                parts.append(f"- Domains: {', '.join(domains)}")

        df = prd_doc.get("dataflow", {})
        if df:
            parts.append("\n### Dataflow")
            parts.append(f"- Topology: {df.get('topology', 'N/A')}")
            parts.append(f"- Bus: {df.get('bus_protocol', 'N/A')}")
            parts.append(f"- Width: {df.get('data_width_bits', 'N/A')} bits")

        func_reqs = prd_doc.get("functional_requirements", [])
        if func_reqs:
            parts.append("\n### Functional Requirements")
            for r in func_reqs:
                parts.append(f"- {r}")

        constraints = prd_doc.get("constraints", [])
        if constraints:
            parts.append("\n### Constraints")
            for c in constraints:
                parts.append(f"- {c}")

        open_items = prd_doc.get("open_items", [])
        if open_items:
            parts.append("\n### Open Items")
            for item in open_items:
                parts.append(f"- {item}")

        parts.append("")

    # Requirements
    reqs = state.get("requirements", "")
    if reqs:
        parts.append(f"## Requirements\n{reqs}")

    # Block diagram
    bd = state.get("block_diagram", {})
    if bd:
        blocks = bd.get("blocks", [])
        parts.append(f"## Block Diagram\n{len(blocks)} blocks defined")
        for b in blocks:
            name = b.get("name", "?")
            tier = b.get("tier", "?")
            desc = b.get("description", "")
            ifaces = b.get("interfaces", {})
            iface_str = ", ".join(
                f"{k}: {v.get('protocol', '?')}"
                for k, v in ifaces.items()
            ) if isinstance(ifaces, dict) else ""
            parts.append(f"- **{name}** (Tier {tier}): {desc}")
            if iface_str:
                parts.append(f"  Interfaces: {iface_str}")

        conns = bd.get("connections", [])
        if conns:
            parts.append(f"\n### Connections ({len(conns)})")
            for c in conns[:20]:
                parts.append(
                    f"- {c.get('from_block', '?')}.{c.get('from_port', '?')} -> "
                    f"{c.get('to_block', '?')}.{c.get('to_port', '?')}"
                )

    # Memory map (specialists wrap output in a "result" key)
    mm_raw = state.get("memory_map", {})
    mm = mm_raw.get("result", mm_raw) if isinstance(mm_raw, dict) else {}
    if mm and mm.get("peripherals"):
        parts.append("\n## Memory Map")
        for p in mm["peripherals"]:
            parts.append(
                f"- {p.get('name', '?')}: 0x{p.get('base_address', 0):08X} "
                f"(size=0x{p.get('size', 0):X})"
            )

    # Clock tree (specialists wrap output in a "result" key)
    ct_raw = state.get("clock_tree", {})
    ct = ct_raw.get("result", ct_raw) if isinstance(ct_raw, dict) else {}
    if ct and ct.get("domains"):
        parts.append("\n## Clock Tree")
        for d in ct["domains"]:
            parts.append(
                f"- {d.get('name', '?')}: {d.get('frequency_mhz', '?')} MHz"
            )

    # Register spec (specialists wrap output in a "result" key)
    rs_raw = state.get("register_spec", {})
    rs = rs_raw.get("result", rs_raw) if isinstance(rs_raw, dict) else {}
    if rs and rs.get("blocks"):
        parts.append("\n## Register Spec")
        parts.append(f"{len(rs['blocks'])} blocks with register definitions")

    # Pending questions
    pq = state.get("pending_questions", [])
    if pq:
        parts.append(f"\n## Pending Questions ({len(pq)})")
        for q in pq:
            parts.append(f"- [{q.get('priority', '?')}] {q.get('question', '?')}")

    return "\n".join(parts) if parts else "Architecture state is empty."


def _read_uarch_specs(project_root: str) -> dict[str, dict]:
    """Read uArch spec files and extract JSON summaries + overviews.

    Returns a dict keyed by block name with ``summary`` (parsed JSON)
    and ``overview`` (first paragraph from the spec document).
    """
    import re as _re

    specs_dir = Path(project_root) / _SUMMARY_DIR / "uarch_specs"
    if not specs_dir.is_dir():
        return {}

    results: dict[str, dict] = {}
    for spec_file in sorted(specs_dir.glob("*.md")):
        block_name = spec_file.stem
        try:
            content = spec_file.read_text(encoding="utf-8")
        except OSError:
            continue

        # Extract JSON summary block
        summary: dict = {}
        json_match = _re.search(r"```json\s*\n(.*?)```", content, _re.DOTALL)
        if json_match:
            try:
                summary = json.loads(json_match.group(1))
            except (json.JSONDecodeError, ValueError):
                pass

        # Extract Block Overview (first section body)
        overview = ""
        overview_match = _re.search(
            r"##\s*1\.\s*Block\s+Overview\s*\n(.*?)(?=\n##\s|\Z)",
            content,
            _re.DOTALL,
        )
        if overview_match:
            raw = overview_match.group(1).strip()
            # Take just the first paragraph (up to 300 chars)
            first_para = raw.split("\n\n")[0].strip()
            overview = first_para[:300]

        results[block_name] = {"summary": summary, "overview": overview}

    return results


def _gather_frontend_context(project_root: str) -> str:
    """Aggregate frontend pipeline events into a summary context."""
    from orchestrator.langgraph.event_stream import (
        read_events,
        aggregate_failure_categories,
    )

    events = read_events(project_root)
    frontend_events = [
        e for e in events if e.get("graph", "") not in ("architecture", "backend")
    ]

    if not frontend_events:
        return "No frontend pipeline events recorded yet."

    # Use shared helper for failure aggregation
    failure_agg = aggregate_failure_categories(frontend_events)

    # Block progress tracking
    blocks: dict[str, dict] = {}
    gate_counts: dict[str, int] = {}
    total_attempts: dict[str, int] = {}

    for e in frontend_events:
        block = e.get("block", "")
        node = e.get("node", "")
        etype = e.get("event", "")

        if not block:
            continue

        if block not in blocks:
            blocks[block] = {
                "tier": e.get("tier"),
                "phase": "init",
                "attempt": 1,
                "status": "running",
            }

        if etype == "block_start":
            blocks[block]["attempt"] = e.get("attempt", 1)
            total_attempts[block] = e.get("attempt", 1)
            if e.get("tier"):
                blocks[block]["tier"] = e["tier"]

        if etype == "graph_node_enter":
            blocks[block]["phase"] = node
            blocks[block]["status"] = "running"

        if etype == "graph_node_exit":
            blocks[block]["phase"] = node

            if node == "Advance Block":
                if e.get("success"):
                    blocks[block]["status"] = "done"
                else:
                    blocks[block]["status"] = "failed"
            elif e.get("passed") is False or e.get("clean") is False or e.get("success") is False:
                blocks[block]["status"] = "failing"

            if e.get("gate_count") is not None:
                gate_counts[block] = e["gate_count"]

        if etype == "graph_node_exit" and node == "Pipeline Complete":
            pass  # handled by per-block status

    # Detect HITL-waiting blocks: entered a HITL node but not exited
    HITL_NODES = {"Review Uarch Spec", "Ask Human"}
    hitl_open: dict[str, str] = {}  # block -> node
    for e in frontend_events:
        block = e.get("block", "")
        node = e.get("node", "")
        etype = e.get("event", "")
        if not block or not node:
            continue
        if node in HITL_NODES:
            if "enter" in etype:
                hitl_open[block] = node
            elif "exit" in etype:
                hitl_open.pop(block, None)

    # Mark HITL-waiting blocks
    for block_name, node_name in hitl_open.items():
        if block_name in blocks:
            blocks[block_name]["status"] = "waiting"
            blocks[block_name]["phase"] = node_name

    # Classify blocks
    completed = [b for b, s in blocks.items() if s["status"] == "done"]
    failed = [b for b, s in blocks.items() if s["status"] == "failed"]
    running = [b for b, s in blocks.items() if s["status"] == "running"]
    failing = [b for b, s in blocks.items() if s["status"] == "failing"]
    waiting = [b for b, s in blocks.items() if s["status"] == "waiting"]

    parts: list[str] = []
    total = len(blocks)
    parts.append(f"## Progress: {len(completed)}/{total} blocks completed, "
                 f"{len(running)} running, {len(failed)} failed")

    # HITL alert -- prominent section at the top
    if waiting:
        parts.append("")
        parts.append("---")
        parts.append("")
        parts.append("## \u26A0\uFE0F ACTION REQUIRED: Human Review Needed")
        parts.append("")
        parts.append(f"**{len(waiting)} block(s) are waiting for human review** "
                     "and cannot proceed until approved.")
        parts.append("")
        for block_name in sorted(waiting):
            node = hitl_open.get(block_name, "Review Uarch Spec")
            if node == "Review Uarch Spec":
                parts.append(f"- **{block_name}**: uArch spec review pending "
                             f"(approve, revise, or skip)")
            else:
                parts.append(f"- **{block_name}**: human intervention needed "
                             f"(retry, fix_rtl, add_constraint, skip, or abort)")
        parts.append("")
        parts.append("**To resume:** `resume_pipeline(action=\"approve\")` "
                     "to approve all, or `resume_pipeline(action=\"skip\")` to skip.")
        parts.append("")
        parts.append("---")

    # Block table
    parts.append("\n## Block Status")
    parts.append("| Block | Tier | Phase | Attempt | Status |")
    parts.append("|-------|------|-------|---------|--------|")
    for name, info in sorted(blocks.items()):
        tier = info.get("tier") or "?"
        phase = info.get("phase", "?")
        attempt = total_attempts.get(name, info.get("attempt", 1))
        status = info.get("status", "?")
        parts.append(f"| {name} | {tier} | {phase} | {attempt} | {status} |")

    # Microarchitecture specs
    uarch_specs = _read_uarch_specs(project_root)
    if uarch_specs:
        parts.append(f"\n## Microarchitecture Specifications ({len(uarch_specs)} blocks)")
        for block_name, spec_info in sorted(uarch_specs.items()):
            summary = spec_info.get("summary", {})
            overview = spec_info.get("overview", "")

            parts.append(f"\n### {block_name}")
            if overview:
                parts.append(f"{overview}")

            if summary:
                latency = summary.get("latency_cycles")
                throughput = summary.get("throughput_samples_per_cycle")
                pipeline = summary.get("pipeline_stages")
                regs = summary.get("register_count")
                est_gates = summary.get("estimated_gate_count")
                fsm = summary.get("fsm_states", [])
                dw_in = summary.get("data_width_in")
                dw_out = summary.get("data_width_out")
                rom = summary.get("rom_bits")
                fp_fmt = summary.get("fixed_point_format")

                metrics: list[str] = []
                if latency is not None:
                    metrics.append(f"Latency: {latency} cycles")
                if throughput is not None:
                    metrics.append(f"Throughput: {throughput} samples/cycle")
                if pipeline is not None:
                    metrics.append(f"Pipeline stages: {pipeline}")
                if regs is not None:
                    metrics.append(f"Registers: {regs:,}")
                if rom is not None and rom > 0:
                    metrics.append(f"ROM: {rom:,} bits")
                if est_gates is not None:
                    metrics.append(f"Est. gates: {est_gates:,}")
                if dw_in is not None or dw_out is not None:
                    metrics.append(f"Data width: {dw_in or '?'}b in / {dw_out or '?'}b out")
                if fp_fmt and fp_fmt != "N/A":
                    metrics.append(f"Fixed-point: {fp_fmt}")
                if fsm:
                    metrics.append(f"FSM states: {', '.join(fsm)}")

                if metrics:
                    for m in metrics:
                        parts.append(f"- {m}")

    # DV trends (from shared helper)
    failure_categories = failure_agg.get("failure_categories", {})
    if failure_categories:
        parts.append("\n## DV Failure Trends")
        for cat, count in sorted(failure_categories.items(), key=lambda x: -x[1]):
            parts.append(f"- **{cat}**: {count} occurrences")
        systematic = failure_agg.get("systematic_patterns", [])
        if systematic:
            parts.append(f"\nSystematic patterns (3+ blocks): {', '.join(systematic)}")

    avg_retries = failure_agg.get("avg_retries", 0.0)
    if avg_retries > 0:
        parts.append(f"\nAverage retries per block: {avg_retries:.1f}")

    # Gate counts
    if gate_counts:
        parts.append("\n## Synthesis Results")
        for block, gc in sorted(gate_counts.items()):
            parts.append(f"- {block}: {gc:,} gates")

    # Blocks needing attention
    attention = failed + failing
    if attention:
        parts.append("\n## Needs Attention")
        for b in attention:
            info = blocks[b]
            parts.append(f"- **{b}**: {info['status']} at phase {info['phase']}")

    return "\n".join(parts)


def _gather_backend_context(project_root: str) -> str:
    """Aggregate backend-tagged events into a summary context."""
    from orchestrator.langgraph.event_stream import read_events

    events = read_events(project_root)
    backend_events = [e for e in events if e.get("graph") == "backend"]

    if not backend_events:
        return "No backend pipeline events recorded yet."

    blocks: dict[str, dict] = {}
    for e in backend_events:
        block = e.get("block", "")
        node = e.get("node", "")
        etype = e.get("event", "")

        if not block:
            continue

        if block not in blocks:
            blocks[block] = {"phase": "init", "attempt": 1, "status": "running"}

        if etype == "block_start":
            blocks[block]["attempt"] = e.get("attempt", 1)
        if etype == "graph_node_enter":
            blocks[block]["phase"] = node
            blocks[block]["status"] = "running"
        if etype == "graph_node_exit":
            blocks[block]["phase"] = node
            if node == "Advance Block":
                blocks[block]["status"] = "done" if e.get("success") else "failed"
            elif e.get("passed") is False or e.get("clean") is False:
                blocks[block]["status"] = "failing"

    completed = [b for b, s in blocks.items() if s["status"] == "done"]
    parts: list[str] = []
    parts.append(f"## Progress: {len(completed)}/{len(blocks)} blocks through physical design")

    parts.append("\n## Block PnR Status")
    parts.append("| Block | Phase | Attempt | Status |")
    parts.append("|-------|-------|---------|--------|")
    for name, info in sorted(blocks.items()):
        parts.append(
            f"| {name} | {info['phase']} | {info['attempt']} | {info['status']} |"
        )

    attention = [b for b, s in blocks.items() if s["status"] in ("failed", "failing")]
    if attention:
        parts.append("\n## Needs Attention")
        for b in attention:
            info = blocks[b]
            parts.append(f"- **{b}**: {info['status']} at phase {info['phase']}")

    return "\n".join(parts)


_CONTEXT_GATHERERS = {
    "architecture": _gather_architecture_context,
    "frontend": _gather_frontend_context,
    "backend": _gather_backend_context,
}

# ---------------------------------------------------------------------------
# System prompts per stage
# ---------------------------------------------------------------------------

_SYSTEM_PROMPTS = {
    "architecture": """\
You are an ASIC architecture observer. Given the current architecture state,
produce a concise 4-5 sentence status summary suitable for a hardware
architect reviewing the design at a glance. Focus on:

- Current phase and what has been completed so far
- Key architectural decisions made (if any)
- Outstanding items or next steps
- Any blocking issues or questions pending human review

Be brief and status-oriented -- this is a dashboard summary, not a full spec.
The detailed PRD, SAD, FRD, and ERS are shown in separate cards.
Output ONLY the markdown -- no preamble, no code fences.""",

    "frontend": """\
You are an RTL pipeline observer. Given the current pipeline execution state,
produce a concise 4-5 sentence status summary for a verification engineer.
Focus on:

- **CRITICAL**: If there is an "ACTION REQUIRED: Human Review Needed" section
  in the input, mention it prominently first.
- Overall progress (completed / total blocks, running count)
- DV trends: pass rates, common failure categories, average retries
- Active failures or blocks needing human attention
- Brief recommendations or next steps

Be brief and status-oriented -- this is a dashboard summary. The detailed
uArch specs are shown in separate collapsible cards.
Output ONLY the markdown -- no preamble, no code fences.""",

    "backend": """\
You are a physical design observer. Given the current backend execution state,
produce a concise markdown status report for a PnR engineer. Include:

- Overall progress (completed / total blocks)
- Per-block PnR status (current phase, attempt, status)
- DRC/LVS pass rates if available
- Timing closure summary if available
- Human-actionable items

Use markdown tables and headers. Be concise and actionable.
Output ONLY the markdown -- no preamble, no code fences.""",
}

# ---------------------------------------------------------------------------
# Observer LLM
# ---------------------------------------------------------------------------

_llm_instance = None


def _get_llm():
    """Lazy-initialise the observer LLM (claude-sonnet-4-6 via Claude CLI)."""
    global _llm_instance
    if _llm_instance is None:
        from orchestrator.langchain.agents.coresmith_llm import ClaudeLLM
        _llm_instance = ClaudeLLM(
            model="claude-sonnet-4-6",
            timeout=60,
            max_turns=10,
            disable_tools=True,
        )
    return _llm_instance


async def _generate_summary(stage: str, context: str) -> str:
    """Call the observer LLM to produce a markdown summary.

    Runs in a detached OTel context so that observer spans are never
    parented to a graph-node span (which would pollute the webview
    trace view with summarizer invocations).
    """
    from opentelemetry import context as otel_context

    llm = _get_llm()
    system_prompt = _SYSTEM_PROMPTS[stage]
    user_prompt = f"Current {stage} state:\n\n{context}"
    # Detach from the inherited OTel span context so that the LLM
    # span created by ClaudeLLM is a standalone root span,
    # not a child of the graph node that triggered this observer.
    token = otel_context.attach(otel_context.Context())
    try:
        return await llm.call(
            system=system_prompt,
            prompt=user_prompt,
            run_name=f"Observer [{stage}]",
        )
    finally:
        otel_context.detach(token)


def _write_summary(project_root: str, stage: str, content: str) -> None:
    """Write the summary markdown to disk."""
    summary_path = Path(project_root) / _SUMMARY_DIR / f"summary_{stage}.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Hook entry point
# ---------------------------------------------------------------------------

async def observer_hook(
    project_root: str, node_name: str, event_record: dict
) -> None:
    """Async exit hook -- called after every ``graph_node_exit``.

    Determines the stage, gathers context, calls the observer LLM, and
    writes the summary to disk.  Debounced per-stage to avoid flooding
    the LLM on rapid-fire node exits.

    Fix #16: Only triggers for significant nodes and respects the
    ``observer.enabled`` config setting.
    """
    if not _is_observer_enabled():
        return

    stage = _detect_stage(event_record)

    # Fix #16: skip non-significant nodes to reduce LLM calls
    if not _is_significant_node(node_name, stage):
        logger.debug("Observer skipped for non-significant node=%s stage=%s", node_name, stage)
        return

    now = time.time()

    # Debounce: skip if last call for this stage was too recent
    if now - _last_call.get(stage, 0) < _COOLDOWN_S:
        logger.debug("Observer debounced for stage=%s (cooldown)", stage)
        return

    lock = _get_lock(stage)
    if lock.locked():
        logger.debug("Observer skipped for stage=%s (already running)", stage)
        return

    async with lock:
        _last_call[stage] = now

        # Gather context
        gatherer = _CONTEXT_GATHERERS.get(stage)
        if not gatherer:
            return
        context = gatherer(project_root)

        if not context or "No " in context[:10]:
            # Minimal context -- write a placeholder instead of calling the LLM
            _write_summary(project_root, stage, f"# {stage.title()} Summary\n\n_{context}_\n")
            return

        try:
            summary = await _generate_summary(stage, context)
            _write_summary(project_root, stage, summary)
            logger.info("Observer summary updated for stage=%s (%d chars)", stage, len(summary))
        except Exception:
            logger.exception("Observer LLM call failed for stage=%s", stage)
