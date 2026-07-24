# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Per-block lookup into the canonical `interface_contracts.json` produced
by the Interface Definition architecture stage.

Per-block generators (uArch spec + RTL) call `load_block_contracts(...)`
to receive only the edge contracts where this block participates as the
producer or consumer, plus the design-wide defaults. The result is
formatted as a prompt fragment ready to drop into the user message.

Why this exists: the v7 video_codec autopilot run produced a correct
`interface_contracts.json` (incl. `bootstrap_policy.reset_seed` for the
neighbor edge), but the per-block RTL generator did not honor it because
it never read the file. This helper bridges that gap by giving each
generator the relevant slice of contracts directly in its prompt.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_interface_contracts(project_root: str) -> dict[str, Any]:
    """Load the full interface_contracts.json from disk.

    Returns an empty dict if the file does not exist or fails to parse —
    so callers can treat "no contracts" as a no-op rather than an error.
    """
    if not project_root:
        return {}
    path = Path(project_root) / ".coresmith" / "interface_contracts.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def filter_contracts_for_block(
    contracts: list[dict] | None,
    block_name: str,
) -> list[dict]:
    """Return the subset of contracts where `block_name` is the producer
    or the consumer. Tolerant of missing fields."""
    if not contracts:
        return []
    out: list[dict] = []
    for c in contracts:
        producer = str(c.get("producer_block", "")).strip()
        consumer = str(c.get("consumer_block", "")).strip()
        if block_name == producer or block_name == consumer:
            out.append(c)
    return out


def load_block_contracts(project_root: str, block_name: str) -> dict[str, Any]:
    """Compose a per-block contract view suitable for prompt injection.

    Returns a dict with two keys:
      - `defaults`: top-level design conventions (default packing,
        endianness rationale) from the canonical file. Empty when the
        file is missing.
      - `edges`: list of contract entries where `block_name` participates,
        each tagged with `role` ("producer" or "consumer") so the generator
        knows which side it has to implement.
    """
    full = load_interface_contracts(project_root)
    if not full:
        return {"defaults": {}, "edges": []}

    contracts = full.get("contracts") or []
    edges: list[dict] = []
    for c in filter_contracts_for_block(contracts, block_name):
        # Annotate role so the generator knows which port to expose.
        producer = str(c.get("producer_block", "")).strip()
        role = "producer" if producer == block_name else "consumer"
        edge = dict(c)
        edge["role"] = role
        edges.append(edge)

    defaults = {
        k: full.get(k)
        for k in (
            "default_packing_convention",
            "default_endianness_rationale",
            "design_summary",
        )
        if full.get(k)
    }
    return {"defaults": defaults, "edges": edges}


def format_block_contracts_prompt_slim(block_name: str, view: dict[str, Any]) -> str:
    """Slim contract fragment: keep the load-bearing bootstrap policy + a CLI
    pointer to the full edge list, instead of dumping every edge JSON.

    Used under ``CORESMITH_PROMPT_SLIM`` (default on). Returns '' when there are
    no relevant edges (caller skips injection cleanly).
    """
    edges = view.get("edges") or []
    if not edges:
        return ""
    lines = [
        "",
        f"## Interface contracts for `{block_name}` "
        "(pull the full bit-level edge list: "
        f"`\"${{CORESMITH_CLI:-coresmith}}\" contracts {block_name}`)",
    ]
    defaults = view.get("defaults") or {}
    pack = defaults.get("default_packing_convention")
    if pack:
        lines.append(f"- design-wide packing convention: `{pack}`")
    bootstrap_edges = [
        e for e in edges if (e.get("bootstrap_policy") or {}).get("required")
    ]
    for e in bootstrap_edges:
        bp = e.get("bootstrap_policy") or {}
        lines.append(
            f"- bootstrap: edge `{e.get('edge_id', '?')}` (role={e.get('role')}) "
            f"requires `{bp.get('policy_type', 'unspecified')}` -- "
            f"{(bp.get('rationale') or '').strip()}"
        )
    if bootstrap_edges:
        lines.append(
            "  (If producer on a `reset_seed` edge, drive a valid seeded output "
            "on cycle 1 after reset; the equivalence gate checks the exact "
            "contract with a fresh seed.)"
        )
    return "\n".join(lines)


def format_block_contracts_prompt(block_name: str, view: dict[str, Any]) -> str:
    """Render a `load_block_contracts(...)` result as a prompt fragment.

    Returns the empty string when there are no relevant edges, so the
    caller can `if fragment: parts.append(fragment)` and skip injection
    cleanly when contracts aren't available (e.g., older runs).
    """
    edges = view.get("edges") or []
    if not edges:
        return ""

    defaults = view.get("defaults") or {}
    lines = [
        "",
        "## CANONICAL INTERFACE CONTRACTS for this block",
        f"({len(edges)} edge contract(s) where `{block_name}` participates. "
        "These are AUTHORITATIVE — the bit layouts, field positions, handshake "
        "protocol, sideband signals, and bootstrap policy below MUST match in "
        "your output exactly. Do not invent new fields, change widths, or "
        "skip the bootstrap policy.)",
        "",
    ]

    pack = defaults.get("default_packing_convention")
    if pack:
        rationale = defaults.get("default_endianness_rationale") or ""
        lines.append(
            f"**Design-wide convention:** `{pack}`"
            + (f" ({rationale})" if rationale else "")
        )
        lines.append("")

    lines.append("```json")
    lines.append(json.dumps(edges, indent=2))
    lines.append("```")

    # Highlight bootstrap_policy explicitly because it's silently easy to miss.
    bootstrap_edges = [
        e for e in edges
        if (e.get("bootstrap_policy") or {}).get("required")
    ]
    if bootstrap_edges:
        lines.append("")
        lines.append("**Bootstrap policy notice:**")
        for e in bootstrap_edges:
            bp = e.get("bootstrap_policy") or {}
            lines.append(
                f"- Edge `{e.get('edge_id', '?')}` (role={e.get('role')}) "
                f"requires `{bp.get('policy_type', 'unspecified')}` bootstrap. "
                f"{(bp.get('rationale') or '').strip()}"
            )
        lines.append(
            "If you are the producer on a `reset_seed` edge, your RTL MUST "
            "drive a valid output with the specified seed value on cycle 1 "
            "after reset, holding until the consumer's first ready handshake. "
            "If you are the consumer on a `request_driven` edge, your RTL "
            "MUST emit the request before waiting for a response."
        )

    # Highlight flow_control_policy on every edge — this is the v8 codec
    # deadlock fix. Without an explicit elasticity contract the producer
    # and consumer arrive at incompatible assumptions about who stalls
    # when. The notice surfaces the chosen semantics, the required
    # buffer depth, and the implementation contract per role.
    fc_edges = [e for e in edges if e.get("flow_control_policy")]
    if fc_edges:
        lines.append("")
        lines.append("**Flow control policy notice (skid/elastic/credit/request):**")
        for e in fc_edges:
            fc = e.get("flow_control_policy") or {}
            sem = fc.get("semantics", "unset")
            depth = fc.get("min_buffer_depth_beats")
            credit = fc.get("credit_words")
            cycle = fc.get("feedback_cycle")
            extras = []
            if depth is not None:
                extras.append(f"min_buffer_depth_beats={depth}")
            if credit is not None:
                extras.append(f"credit_words={credit}")
            if cycle:
                extras.append("feedback_cycle=true")
            extras_str = (" [" + ", ".join(extras) + "]") if extras else ""
            lines.append(
                f"- Edge `{e.get('edge_id', '?')}` (role={e.get('role')}) "
                f"uses `{sem}`{extras_str}. "
                f"{(fc.get('rationale') or '').strip()}"
            )
        lines.append(
            "Implementation rules per semantics:\n"
            "  * `free_running`: producer must NEVER assert backpressure on "
            "the upstream side; consumer must accept every beat.\n"
            "  * `skid`: insert a 1-deep skid register; both sides may stall "
            "for one cycle without losing the in-flight beat.\n"
            "  * `elastic_fifo`: instantiate a FIFO of at least the declared "
            "min_buffer_depth_beats between producer and consumer, with full "
            "and empty backpressure plumbed to tready and tvalid.\n"
            "  * `credit`: producer counts down credits on each beat sent; "
            "consumer issues credit_returns on a reverse-channel sideband. "
            "Producer must NOT send when credits == 0.\n"
            "  * `request_response`: consumer issues a request packet on a "
            "reverse channel; producer must respond with exactly one packet "
            "per request. Producer MUST NOT push unsolicited."
        )
        # Hard requirement on FIFO depth -- the v9 codec deadlock was
        # exactly this: contract said depth=N but the RTL author picked
        # a smaller convenient value and the loop wedged. The
        # cross_spec_fifo_depth_adherence audit will reject any FIFO
        # whose declared DEPTH is below the contract minimum, so failing
        # this rule guarantees a downstream pipeline failure.
        elastic_edges = [
            e for e in fc_edges
            if (e.get("flow_control_policy") or {}).get("semantics") == "elastic_fifo"
        ]
        if elastic_edges:
            lines.append("")
            lines.append("**FIFO depth requirement (HARD, enforced by audit):**")
            for e in elastic_edges:
                fc = e.get("flow_control_policy") or {}
                depth = fc.get("min_buffer_depth_beats")
                if depth is None:
                    continue
                lines.append(
                    f"- Edge `{e.get('edge_id', '?')}` (role={e.get('role')}) "
                    f"requires FIFO DEPTH >= {depth} beats. Declare it as "
                    f"`localparam DEPTH = {depth};` (or larger) and size all "
                    "depth-dependent registers + pointer widths accordingly."
                )
            lines.append(
                "Do NOT downsize the FIFO below the contract minimum even "
                "if synthesis or area pressure pushes back. The "
                "min_buffer_depth_beats values were derived from the "
                "worst-case stall window of the feedback graph; choosing a "
                "smaller depth WILL deadlock the design. The "
                "`cross_spec_fifo_depth_adherence` constraint subagent reads "
                "the synthesized RTL after generation and fails the run if "
                "any `localparam DEPTH` (or equivalent) is below the value "
                "above."
            )

    return "\n".join(lines)


__all__ = [
    "filter_contracts_for_block",
    "format_block_contracts_prompt",
    "load_block_contracts",
    "load_interface_contracts",
]
