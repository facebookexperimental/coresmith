# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Interface Definition Specialist -- freezes the bit-level edge contracts
between blocks so per-block uArch specs that come later don't drift.

This specialist runs after the block diagram is finalized and before
SAD/FRD/ERS / per-block uArch spec generation. It expands every directed
edge in the block diagram into a canonical contract: handshake protocol
(AXI-Stream or sRdy/dRdy), total data width, field list with explicit
[MSB:LSB] positions, signedness, encoding, and bootstrap policy for
edges that participate in closed feedback cycles.

Downstream:
  * per-block uArch spec generators read `interface_contracts.json` and
    treat it as authoritative for port bit layouts;
  * the `cross_spec_contract_adherence` constraint subagent validates
    that per-block specs match the contract field-for-field.

Origin: surfaced by the v5 h264 codec_v3 autopilot run, where the
architecture-LLM produced N+1 locally-consistent documents (block
diagram + N per-block specs) that disagreed at the edges — endianness
mismatches, port-partition shape drift, missing bootstrap policy. The
fix moves the bit-level contract from "emergent from per-block specs"
to "frozen up front and enforced".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from orchestrator.langchain.prompts.skills import load_skills as _load_skills

_PROMPT_FILE = (
    Path(__file__).resolve().parents[2]
    / "langchain"
    / "prompts"
    / "interface_definition.md"
)
SYSTEM_PROMPT = _PROMPT_FILE.read_text(encoding="utf-8")

# Inject the handshake skills so the specialist has access to the same
# AXI-Stream and sRdy/dRdy convention notes the uArch spec generator and
# integration_lead use.
_SKILLS_TEXT = _load_skills("axi_stream", "srdy_drdy", "arithmetic_precision")
if _SKILLS_TEXT:
    SYSTEM_PROMPT = (
        SYSTEM_PROMPT
        + "\n\n# Reference Skills (use to choose handshake + packing)\n\n"
        + _SKILLS_TEXT
    )


_DEFAULT_RESULT: dict[str, Any] = {
    "design_summary": "",
    "default_packing_convention": "msb_first_by_field_list",
    "default_endianness_rationale": "",
    "contracts": [],
    "open_questions": [],
}


async def analyze_interface_definition(
    block_diagram: dict,
    requirements: str = "",
    sad_spec: dict | None = None,
    frd_spec: dict | None = None,
    project_root: str = ".",
) -> dict[str, Any]:
    """Freeze canonical bit-level contracts for every block-diagram edge.

    Args:
        block_diagram: Block diagram produced by `analyze_block_diagram`.
        requirements: Top-level system requirements text.
        sad_spec / frd_spec: Optional upstream architecture artifacts.
        project_root: Where to write `interface_contracts.json`.

    Returns:
        Dict with keys `result` (the full contract artifact) and
        `questions` (any open questions the specialist deferred).
    """
    from opentelemetry import trace as _trace

    tracer = _trace.get_tracer("coresmith.architecture.interface_definition")

    with tracer.start_as_current_span("analyze_interface_definition") as span:
        blocks = block_diagram.get("blocks", []) or []
        connections = (
            block_diagram.get("connections", [])
            or block_diagram.get("edges", [])
            or []
        )
        span.set_attribute("input_block_count", len(blocks))
        span.set_attribute("input_edge_count", len(connections))

        # No-op when there are no inter-block edges to contract for.
        if not connections:
            span.set_attribute("no_op", True)
            result = dict(_DEFAULT_RESULT)
            result["design_summary"] = (
                "Single-block or unconnected design: no inter-block "
                "contracts required."
            )
            return {"result": result, "questions": []}

        target_path = Path(project_root) / ".coresmith" / "interface_contracts.json"
        target_path.parent.mkdir(parents=True, exist_ok=True)

        parts = [
            "Freeze the bit-level interface contracts for the following "
            "ASIC block diagram. Produce one contract per directed edge; "
            "no edge may be skipped. Respond with the JSON schema "
            "specified in your system prompt and write the result to disk.",
            f"\n--- BLOCK DIAGRAM ---\n{json.dumps(block_diagram, indent=2)[:20000]}",
        ]
        if requirements:
            parts.append(f"\n--- REQUIREMENTS ---\n{requirements[:6000]}")
        if sad_spec:
            sad_summary = sad_spec.get("sad", sad_spec)
            parts.append(
                f"\n--- SAD (excerpt) ---\n{json.dumps(sad_summary, indent=2)[:6000]}"
            )
        if frd_spec:
            frd_summary = frd_spec.get("frd", frd_spec)
            parts.append(
                f"\n--- FRD (excerpt) ---\n{json.dumps(frd_summary, indent=2)[:6000]}"
            )

        parts.append(
            f"\n\nIMPORTANT: Write the contract JSON to:\n  {target_path}\n"
            "After writing, respond with only the file path confirmation."
        )
        user_message = "\n".join(parts)

        from orchestrator.langchain.agents.coresmith_llm import (
            DEFAULT_MODEL,
            ClaudeLLM,
        )

        llm = ClaudeLLM(model=DEFAULT_MODEL, timeout=1200)

        try:
            content = await llm.call(
                system=SYSTEM_PROMPT,
                prompt=user_message,
                run_name="interface_definition",
            )
        except Exception as exc:  # pragma: no cover -- LLM transport failure
            span.set_attribute("error", str(exc))
            span.set_status(_trace.StatusCode.ERROR, str(exc))
            result = dict(_DEFAULT_RESULT)
            result["design_summary"] = (
                f"Interface Definition specialist failed: {exc}. Per-block "
                "specs will be generated without a frozen contract; expect "
                "downstream drift to be caught by integration_check."
            )
            return {"result": result, "questions": []}

        from orchestrator.utils import read_back_json

        disk_result, ok = read_back_json(
            target_path,
            content,
            dict(_DEFAULT_RESULT),
            context="interface_definition",
        )
        result = disk_result if ok else _parse_response(content)

        validated, validation_notes = _validate_contracts(result, connections)
        result.update(validated)
        if validation_notes:
            existing = result.get("validation_notes", []) or []
            result["validation_notes"] = list(existing) + validation_notes

        span.set_attribute("contract_count", len(result.get("contracts", []) or []))
        span.set_attribute(
            "open_question_count", len(result.get("open_questions", []) or [])
        )
        return {
            "result": result,
            "questions": result.get("open_questions", []) or [],
        }


def _parse_response(content: str) -> dict[str, Any]:
    """Extract structured JSON from LLM response."""
    from orchestrator.utils import parse_llm_json

    parsed, _ok = parse_llm_json(
        content, dict(_DEFAULT_RESULT), context="interface_definition"
    )
    return parsed


def _validate_contracts(
    result: dict[str, Any],
    expected_edges: list[dict],
) -> tuple[dict[str, Any], list[str]]:
    """Light structural sanity checks on the specialist's output.

    These are NOT a substitute for the cross_spec_contract_adherence
    subagent that runs at constraint-check time. They catch only the
    obvious cases: missing edges, fields that don't sum to the declared
    data width, overlapping bit ranges within a contract. Anything else
    falls through to downstream review.
    """
    notes: list[str] = []
    contracts = list(result.get("contracts", []) or [])

    # Build adjacency for cycle detection so we can validate
    # flow_control_policy against the cycle membership of each edge.
    adjacency: dict[str, list[str]] = {}
    for edge in expected_edges:
        src = str(
            edge.get("from") or edge.get("from_block") or edge.get("source") or ""
        )
        dst = str(
            edge.get("to") or edge.get("to_block") or edge.get("target") or ""
        )
        if src and dst:
            adjacency.setdefault(src, []).append(dst)
    cycle_edges = _edges_in_cycles(adjacency)

    # Per-contract structural checks
    for c in contracts:
        cid = c.get("edge_id") or f"{c.get('producer_block', '?')}->{c.get('consumer_block', '?')}"
        fields = list(c.get("fields", []) or [])
        if not fields:
            continue
        try:
            field_sum = sum(int(f.get("width", 0) or 0) for f in fields)
        except (TypeError, ValueError):
            notes.append(f"{cid}: non-numeric field width(s); skipping width sum check.")
            continue
        declared = int(c.get("data_width_bits", 0) or 0)
        if declared and declared != field_sum:
            notes.append(
                f"{cid}: declared data_width_bits={declared} but field "
                f"widths sum to {field_sum}; specialist disagreement."
            )

        # Overlap / gap detection
        ranges: list[tuple[int, int, str]] = []
        for f in fields:
            try:
                msb = int(f.get("msb"))
                lsb = int(f.get("lsb"))
            except (TypeError, ValueError):
                continue
            if msb < lsb:
                notes.append(f"{cid}.{f.get('name')}: msb({msb})<lsb({lsb}).")
                continue
            ranges.append((msb, lsb, str(f.get("name"))))
        ranges.sort()
        for i in range(1, len(ranges)):
            prev_msb, prev_lsb, prev_name = ranges[i - 1]
            cur_msb, cur_lsb, cur_name = ranges[i]
            if cur_lsb <= prev_msb:
                notes.append(
                    f"{cid}: fields {prev_name}[{prev_msb}:{prev_lsb}] and "
                    f"{cur_name}[{cur_msb}:{cur_lsb}] overlap."
                )

        # Flow-control sanity: any edge on a closed cycle must NOT
        # use free_running or skid semantics, because both assume the
        # producer or the consumer can hold a transaction indefinitely.
        # The v7/v8 h264 deadlock landed exactly in this gap.
        producer = str(c.get("producer_block", ""))
        consumer = str(c.get("consumer_block", ""))
        fc = c.get("flow_control_policy") or {}
        semantics = (fc.get("semantics") or "").strip()
        on_cycle = (producer, consumer) in cycle_edges
        if on_cycle and semantics in ("", "free_running", "skid"):
            notes.append(
                f"{cid}: edge participates in a feedback cycle but "
                f"flow_control_policy.semantics={semantics or 'unset'} "
                "— pick elastic_fifo, credit, or request_response."
            )
        if on_cycle and not fc.get("feedback_cycle"):
            notes.append(
                f"{cid}: edge is in a feedback cycle in the block diagram "
                "but flow_control_policy.feedback_cycle is not true."
            )
        if semantics == "elastic_fifo":
            try:
                depth = int(fc.get("min_buffer_depth_beats") or 0)
            except (TypeError, ValueError):
                depth = 0
            if depth < 2:
                notes.append(
                    f"{cid}: elastic_fifo requires min_buffer_depth_beats>=2; "
                    f"got {depth}."
                )

    # Cross-edge coverage check: every directed edge in the block diagram
    # should be represented by exactly one contract entry. Edge identity
    # is matched by (producer_block, consumer_block) pair OR by edge_id
    # if the specialist provided one matching the connection record.
    declared_pairs = {
        (str(c.get("producer_block", "")), str(c.get("consumer_block", "")))
        for c in contracts
    }
    for edge in expected_edges:
        pair = (
            str(edge.get("from") or edge.get("from_block") or edge.get("source") or ""),
            str(edge.get("to") or edge.get("to_block") or edge.get("target") or ""),
        )
        if not all(pair):
            continue
        if pair not in declared_pairs:
            notes.append(
                f"missing contract for edge {pair[0]} -> {pair[1]} (declared "
                "in block_diagram but absent from contracts[])."
            )

    return ({}, notes)


def _edges_in_cycles(adjacency: dict[str, list[str]]) -> set[tuple[str, str]]:
    """Return the set of (src, dst) edges that participate in any
    directed cycle of the adjacency graph.

    Uses Tarjan's SCC algorithm: any edge whose endpoints lie in the
    same strongly-connected component (of size > 1 OR self-loop) is on
    a cycle. The graph is tiny (block-diagram scale, dozens of nodes),
    so the iterative-recursive simplification is fine.
    """
    if not adjacency:
        return set()

    nodes = set(adjacency.keys())
    for dests in adjacency.values():
        nodes.update(dests)

    index_counter = [0]
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    sccs: list[list[str]] = []

    def _strongconnect(v: str) -> None:
        indices[v] = index_counter[0]
        lowlinks[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)
        for w in adjacency.get(v, []):
            if w not in indices:
                _strongconnect(w)
                lowlinks[v] = min(lowlinks[v], lowlinks[w])
            elif w in on_stack:
                lowlinks[v] = min(lowlinks[v], indices[w])
        if lowlinks[v] == indices[v]:
            comp: list[str] = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                comp.append(w)
                if w == v:
                    break
            sccs.append(comp)

    for node in nodes:
        if node not in indices:
            _strongconnect(node)

    edges_in_cycles: set[tuple[str, str]] = set()
    node_to_scc: dict[str, int] = {}
    for i, comp in enumerate(sccs):
        for n in comp:
            node_to_scc[n] = i

    for src, dests in adjacency.items():
        for dst in dests:
            same_scc = (
                node_to_scc.get(src) is not None
                and node_to_scc.get(src) == node_to_scc.get(dst)
                and (len(sccs[node_to_scc[src]]) > 1 or src == dst)
            )
            if same_scc:
                edges_in_cycles.add((src, dst))
    return edges_in_cycles
