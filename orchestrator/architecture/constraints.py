# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Constraint Checker -- validates cross-cutting architectural constraints.

Two layers:

1. Deterministic shuttle checks (GPIO pad budget, area fit). These run only
   when explicitly enabled (MPW/OpenFrame/Caravel wrapper runs); they
   operate on structured block_diagram fields, not on prose.

2. LLM subagents, one per registered constraint. Each subagent reads the
   full artifact bundle, evaluates a single named constraint, and returns
   pass/fail with evidence. Subagents fan out in parallel.

The previous implementation had a third layer of regex-driven Python
checks (derived geometry, transaction counts, throughput arithmetic) that
text-scanned the concatenated artifacts + raw requirements.md for
phrases like ``<N> columns ... <M> rows`` and ``<N> blocks per frame``.
That regex pipeline produced two real problems and one structural one:

* False positives: the regex matched anti-pattern warning sentences in
  human-authored requirements ("do not transpose it to 45 columns by 80
  rows") as if they were claims, then disagreed with the (correct)
  structured artifact field and reported a violation.

* No coverage outside macroblock-based image processing: the regex
  whitelisted ``frame``/``image``/``video``/``pixel``/``macroblock``/
  ``tile`` vocabulary. FFTs, audio codecs, packet processors, CPU cores,
  sensor frontends, etc. got *no* coverage at all.

* Architectural mismatch: arithmetic that's derivable from PRD inputs
  was being re-derived inside the verifier and compared against LLM
  output, instead of being written by code at artifact-time and
  consumed unchanged. The shared-derive-then-bind design lives in the
  block-diagram persistence path; the verifier shouldn't try to recover
  derived facts from prose.

Each violation dict has:
  - violation: human-readable description string
  - category: "structural" (requires human input) or "auto_fixable" (LLM can iterate)
  - check: which constraint check flagged it (e.g. "block_connectivity")
  - severity: "error" or "warning"
  - evidence: (optional) subagent citation from the artifact bundle
  - suggested_fix: (optional) subagent's concrete repair proposal
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Deterministic shuttle checks (kept; structured, no regex)
# ---------------------------------------------------------------------------

def _get_shuttle_limits() -> dict:
    """Load shuttle physical limits from config.yaml tapeout section."""
    from orchestrator.langgraph.pipeline_helpers import load_config

    cfg = load_config()
    tapeout = cfg.get("tapeout", {})
    die_w = tapeout.get("die_width_um", 3520.0)
    die_h = tapeout.get("die_height_um", 5188.0)
    margin = tapeout.get("core_margin_um", 100.0)
    io_pads = tapeout.get("io_pads", 44)
    reserved_pads = 2  # GPIO[0]=clk, GPIO[1]=rst

    user_w = die_w - 2 * margin
    user_h = die_h - 2 * margin
    return {
        "target": tapeout.get("target", "openframe"),
        "die_width_um": die_w,
        "die_height_um": die_h,
        "die_area_mm2": (die_w * die_h) / 1e6,
        "user_width_um": user_w,
        "user_height_um": user_h,
        "user_area_mm2": (user_w * user_h) / 1e6,
        "core_margin_um": margin,
        "total_io_pads": io_pads,
        "reserved_pads": reserved_pads,
        "usable_io_pads": io_pads - reserved_pads,
    }


def _count_block_io_pads(block_diagram: dict) -> tuple[int, list[dict]]:
    """Count total GPIO pads needed by all blocks in the block diagram.

    Cross-references the ``connections`` array so that inter-block ports
    (which become internal wires, not chip-boundary pads) are excluded.

    Returns (total_pads_needed, per_block_details).
    """
    blocks = block_diagram.get("blocks", [])
    connections = block_diagram.get("connections", [])

    connected_ports: set[tuple[str, str]] = set()
    for conn in connections:
        from_block = conn.get("from", conn.get("from_block", ""))
        to_block = conn.get("to", conn.get("to_block", ""))
        from_port = conn.get("from_port", conn.get("interface", ""))
        to_port = conn.get("to_port", conn.get("interface", ""))
        if from_block and from_port:
            connected_ports.add((from_block, from_port))
        if to_block and to_port:
            connected_ports.add((to_block, to_port))

    total = 0
    details: list[dict] = []

    for block in blocks:
        name = block.get("name", "unknown")
        interfaces = block.get("interfaces", {})
        block_pads = 0

        for port_name, port_info in interfaces.items():
            if port_name in ("clk", "rst", "rst_n"):
                continue
            if (name, port_name) in connected_ports:
                continue
            if isinstance(port_info, dict):
                width = port_info.get("width", 1)
            elif isinstance(port_info, int):
                width = port_info
            else:
                width = 1
            block_pads += width

        total += block_pads
        details.append({"block": name, "pads_needed": block_pads})

    return total, details


def _check_shuttle_constraints(
    block_diagram: dict,
    ers_spec: dict | None,
) -> list[dict]:
    """Deterministic shuttle constraint checks (GPIO pad budget + area fit).

    Structured arithmetic only; no prose scanning.
    """
    violations: list[dict] = []
    shuttle = _get_shuttle_limits()

    total_pads, pad_details = _count_block_io_pads(block_diagram)
    usable = shuttle["usable_io_pads"]

    if total_pads > usable:
        overflow = total_pads - usable
        detail_str = ", ".join(
            f"{d['block']}={d['pads_needed']}" for d in pad_details if d["pads_needed"] > 0
        )
        violations.append({
            "violation": (
                f"GPIO pad budget exceeded: design needs {total_pads} I/O pads "
                f"but {shuttle['target'].upper()} shuttle provides only {usable} "
                f"usable pads ({shuttle['total_io_pads']} total minus "
                f"{shuttle['reserved_pads']} reserved for clk/rst). "
                f"Overflow: {overflow} pads. "
                f"Per-block: {detail_str}. "
                f"Consider pin-muxing, serialization, or reducing interface widths."
            ),
            "category": "structural",
            "check": "gpio_pad_budget",
            "severity": "error",
        })
    elif total_pads > usable * 0.9:
        violations.append({
            "violation": (
                f"GPIO pad budget tight: design uses {total_pads}/{usable} "
                f"available pads ({total_pads * 100 / usable:.0f}% utilization). "
                f"Less than 10% headroom for debug/test pins."
            ),
            "category": "auto_fixable",
            "check": "gpio_pad_budget",
            "severity": "warning",
        })

    blocks = block_diagram.get("blocks", [])
    total_gates = sum(b.get("estimated_gates", 0) for b in blocks)

    ers = (ers_spec.get("prd", ers_spec.get("ers", {})) if isinstance(ers_spec, dict) else {}) or {}
    area_budget = ers.get("area_budget", {}) or {}
    prd_die_area = area_budget.get("max_die_area_mm2")

    if prd_die_area and prd_die_area > shuttle["user_area_mm2"]:
        violations.append({
            "violation": (
                f"PRD die area budget ({prd_die_area:.3f} mm²) exceeds "
                f"{shuttle['target'].upper()} shuttle user area "
                f"({shuttle['user_area_mm2']:.3f} mm²). "
                f"Die dimensions: {shuttle['user_width_um']:.0f} x "
                f"{shuttle['user_height_um']:.0f} um. "
                f"The design will not fit in the shuttle frame. "
                f"Reduce area budget or choose a larger shuttle."
            ),
            "category": "structural",
            "check": "shuttle_area_fit",
            "severity": "error",
        })

    if total_gates > 0:
        gate_density_per_mm2 = 200_000  # ~200K gates/mm² for Sky130 HD at 60% util
        estimated_area_mm2 = total_gates / gate_density_per_mm2
        if estimated_area_mm2 > shuttle["user_area_mm2"]:
            violations.append({
                "violation": (
                    f"Estimated design area ({estimated_area_mm2:.3f} mm² from "
                    f"{total_gates:,} gates at ~200K gates/mm²) exceeds "
                    f"{shuttle['target'].upper()} shuttle user area "
                    f"({shuttle['user_area_mm2']:.3f} mm²). "
                    f"Reduce block count, gate estimates, or functionality."
                ),
                "category": "structural",
                "check": "shuttle_area_fit",
                "severity": "error",
            })
        elif estimated_area_mm2 > shuttle["user_area_mm2"] * 0.8:
            violations.append({
                "violation": (
                    f"Estimated design area ({estimated_area_mm2:.3f} mm²) uses "
                    f"{estimated_area_mm2 * 100 / shuttle['user_area_mm2']:.0f}% of "
                    f"shuttle user area ({shuttle['user_area_mm2']:.3f} mm²). "
                    f"Limited room for routing, power grid, and fill. "
                    f"Consider reducing gate count."
                ),
                "category": "auto_fixable",
                "check": "shuttle_area_fit",
                "severity": "warning",
            })

    return violations


def _shuttle_constraints_enabled(requirements: str = "", ers_spec: dict | None = None) -> bool:
    """Return whether package/shuttle constraints should be enforced.

    Most CoreSmith frontend runs generate reusable soft IP, where block ports
    are internal module interfaces rather than package GPIO. Shuttle pad/area
    checks are useful for MPW benchmark wrappers, but they must be opt-in to
    avoid forcing every soft IP through an OpenFrame pin budget.
    """
    value = os.environ.get("CORESMITH_ENABLE_SHUTTLE_CONSTRAINTS", "").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False

    text = requirements.lower()
    if any(token in text for token in ("openframe wrapper", "mpw wrapper", "shuttle gpio")):
        return True

    ers = (ers_spec.get("prd", ers_spec.get("ers", {})) if isinstance(ers_spec, dict) else {}) or {}
    target = json.dumps(ers.get("technology", {})).lower()
    return any(token in target for token in ("openframe", "caravel", "mpw"))


# ---------------------------------------------------------------------------
# Constraint catalog -- registered constraints checked by LLM subagents
# ---------------------------------------------------------------------------
#
# Each entry is a single named constraint. The subagent reads the artifact
# bundle and answers PASS/FAIL for that one constraint. New designs add new
# entries here, not new regex patterns. Each constraint declares an
# ``applies`` predicate so subagents only fan out for the checks that this
# particular design needs.

_CONSTRAINT_CATALOG: list[dict] = [
    {
        "id": "block_connectivity",
        "category": "structural",
        "severity": "error",
        "description": (
            "Every non-infrastructure block in `block_diagram.blocks` must "
            "appear in at least one entry of `block_diagram.connections` "
            "(as `from_block`, `to_block`, `from`, or `to`). Infrastructure "
            "blocks (clock generators, reset controllers, debug, "
            "synchronizers, AXI infrastructure) are exempt. Ingress/egress "
            "IO blocks that intentionally have only chip-boundary ports are "
            "also exempt. Flag any block that appears in `blocks` but never "
            "in `connections` and is not in the exempt categories above."
        ),
        "applies": lambda ctx: len(ctx["block_diagram"].get("blocks", [])) >= 2,
    },
    {
        "id": "data_width_compatibility",
        "category": "auto_fixable",
        "severity": "error",
        "description": (
            "For each entry in `block_diagram.connections`, the source port's "
            "data_width must equal the destination port's data_width unless "
            "the connection itself names an adapter block, or the spec "
            "explicitly permits a width change at that interface. Flag any "
            "connection where the two widths differ and no adapter is named."
        ),
        "applies": lambda ctx: bool(ctx["block_diagram"].get("connections")),
    },
    {
        "id": "derived_arithmetic_consistency",
        "category": "auto_fixable",
        "severity": "error",
        "description": (
            "Any quantity that the artifacts derive arithmetically from PRD "
            "inputs (frame_size / block_size -> columns / rows / "
            "total_transactions / coordinate_max; clock_hz / target_fps / "
            "transactions_per_frame -> cycles_per_transaction_budget; "
            "burst_size / burst_count -> total_payload_bytes; etc.) must "
            "be internally consistent.\n\n"
            "Procedure: read the PRD/ERS for the *source* numeric inputs. "
            "For each derived quantity claimed in the architecture artifacts "
            "(block_diagram, SAD, FRD, ERS), compute what it *should* be "
            "from those source inputs, and compare against the claimed "
            "value. Flag any mismatch.\n\n"
            "Important: lines in the requirements doc that are explicit "
            "anti-pattern warnings (\"do not transpose to ...\", "
            "\"incorrect: ...\", \"avoid ...\") are *forbidden patterns* "
            "told to the agent, NOT claims to verify. Skip them."
        ),
        "applies": lambda ctx: True,
    },
    {
        "id": "gate_budget",
        "category": "auto_fixable",
        "severity": "error",
        "description": (
            "The sum of `estimated_gates` across all blocks in "
            "`block_diagram.blocks` must not exceed the PRD's area_budget "
            "max_gate_count (default 2,000,000 if absent). Flag if "
            "exceeded; cite both the sum and the budget."
        ),
        "applies": lambda ctx: any(
            b.get("estimated_gates") for b in ctx["block_diagram"].get("blocks", [])
        ),
    },
    {
        "id": "memory_address_overlap",
        "category": "structural",
        "severity": "error",
        "description": (
            "If a memory_map is present with peripheral address regions, no "
            "two regions may overlap on the same bus. Flag every overlapping "
            "pair, naming both peripherals and the overlapping range."
        ),
        "applies": lambda ctx: bool(
            (ctx.get("memory_map") or {}).get("peripherals")
            or (ctx.get("memory_map") or {}).get("result", {}).get("peripherals")
        ),
    },
    {
        "id": "clock_domain_crossings",
        "category": "auto_fixable",
        "severity": "error",
        "description": (
            "If clock_tree declares more than one clock domain, every "
            "connection between blocks in different domains must reference "
            "an explicit CDC synchronizer (e.g., a 2-FF synchronizer block, "
            "an async FIFO, a handshake module). Flag any cross-domain "
            "connection without CDC."
        ),
        "applies": lambda ctx: len(
            (ctx.get("clock_tree") or {}).get("domains")
            or (ctx.get("clock_tree") or {}).get("result", {}).get("domains")
            or []
        ) > 1,
    },
    {
        "id": "tier_dependency_acyclic",
        "category": "structural",
        "severity": "warning",
        "description": (
            "If blocks have a `tier` field, dependency direction should be "
            "increasing-tier in the canonical lifecycle. A connection from "
            "a higher-tier block to a strictly lower-tier block within the "
            "same pipeline lifecycle is suspect; flag those as a warning "
            "(it may be a legitimate feedback path that should be marked, "
            "or it may be a topology error)."
        ),
        "applies": lambda ctx: any(
            b.get("tier") for b in ctx["block_diagram"].get("blocks", [])
        ),
    },
    {
        "id": "interface_contract_coherence",
        "category": "structural",
        "severity": "error",
        "description": (
            "For each connection, the `from_block` must actually declare an "
            "output port named `from_port` (or the connection's `interface` "
            "name) in its `interfaces`, and the `to_block` must declare a "
            "matching input port. Flag any connection that names a port no "
            "block declares."
        ),
        "applies": lambda ctx: bool(ctx["block_diagram"].get("connections")),
    },
    {
        "id": "inter_block_payload_protocol_coherence",
        "category": "structural",
        "severity": "error",
        "description": (
            "For EACH edge in `block_diagram.connections`, the producer block "
            "and the consumer block must agree on three things. Width-matched "
            "wires alone are not enough. Read each end's uArch spec carefully "
            "(SAD/FRD if present too) and check every connection.\n\n"
            "1. **Payload bit ledger (field-by-field).** The producer's "
            "output payload bit layout and the consumer's input payload bit "
            "layout must define the SAME fields at the SAME bit positions. "
            "Widths matching is not enough; bit order matters. Example "
            "failure: a 21-bit metadata word where producer packs "
            "[6:0]=mb_x, [12:7]=mb_y, ... but consumer decodes "
            "[20:14]=mb_x, [13:8]=mb_y, ... — both sides claim 21 bits and "
            "the AXI wire connects, but the receiver gets bit-reversed "
            "garbage. Flag every connection where producer/consumer bit "
            "ledgers disagree on field name, position, width, or endianness. "
            "Cite the producer-side and consumer-side ledger lines.\n\n"
            "2. **Handshake protocol family + signal naming.** Both ends "
            "must declare the same protocol family (AXI-Stream, AXI-Lite, "
            "AXI4 full, OCP, Avalon-ST, Wishbone, sRDY/DRDY, ready/valid "
            "with optional last, request/grant, …) AND use compatible "
            "signal names. AXI-Stream requires {tvalid, tready, tdata, "
            "optional tlast/tuser/tkeep}. AXI-Lite requires {awvalid, "
            "awaddr, wvalid, wdata, bvalid, arvalid, araddr, rvalid, rdata}. "
            "Avalon-ST requires {valid, ready, data, optional sop/eop}. A "
            "producer emitting `tvalid/tready/tdata` cannot connect to a "
            "consumer expecting `valid/grant/payload` — even if the data "
            "widths match. Flag any mismatch in protocol family OR signal "
            "naming. If one side leaves the protocol implicit, flag it as "
            "ambiguous.\n\n"
            "3. **Flow control: pipeline bubbles, rate matching, buffering.** "
            "For streaming connections, determine whether the producer's "
            "burst behavior and the consumer's consumption rate are "
            "compatible:\n"
            "   - If producer emits 1 word per N cycles and consumer "
            "consumes 1 word per M cycles with M > N, a buffer of depth "
            "≥ (M-N) * burst_length is needed between them; otherwise the "
            "producer will deadlock on backpressure during a burst. Flag "
            "the missing buffer (or undersized FIFO).\n"
            "   - If the consumer asserts ready intermittently (bubbles), "
            "verify the producer keeps `tvalid+tdata` stable across the "
            "bubble (AXI-Stream protocol rule). If the producer's spec "
            "says it deasserts valid mid-transaction, that's an AXI-Stream "
            "compliance violation — flag it.\n"
            "   - If the consumer's spec says it requires N back-to-back "
            "input words with no bubbles (e.g., a transform engine that "
            "blocks on incomplete 8x8 blocks), but the producer can stall, "
            "an elastic FIFO must be declared at that boundary. Flag if "
            "the connection lacks it.\n"
            "   - For sized FIFOs, verify the declared depth is at least "
            "the burst-length × any expected stall window. If the producer "
            "spec says \"emits 64 words per macroblock\" but the inter-"
            "block FIFO is 8 entries deep, flag it.\n\n"
            "Cite the producer-side and consumer-side spec evidence for "
            "each violation. Pass if every connection has consistent ledger "
            "+ matching protocol family + adequate flow control evidence."
        ),
        "applies": lambda ctx: bool(ctx["block_diagram"].get("connections")),
    },
]


# ---------------------------------------------------------------------------
# Subagent dispatcher
# ---------------------------------------------------------------------------

def _build_artifact_bundle(
    block_diagram: dict,
    memory_map: dict,
    clock_tree: dict,
    register_spec: dict,
    benchmark_results: dict | None,
    pdk_config: dict | None,
    requirements: str,
    ers_spec: dict | None,
    project_root: str,
) -> str:
    """Concatenate every artifact a subagent might need to cite, in a stable
    section-headered format. Each subagent receives the same bundle, so the
    bundle is built once and shared.
    """
    parts: list[str] = []

    if requirements:
        parts.append(f"## requirements.md\n{requirements}")

    if ers_spec:
        parts.append(f"## PRD / ERS (structured)\n```json\n{json.dumps(ers_spec, indent=2)}\n```")

    parts.append(f"## block_diagram.json\n```json\n{json.dumps(block_diagram, indent=2)}\n```")

    if memory_map and (memory_map.get("peripherals") or memory_map.get("result")):
        parts.append(f"## memory_map.json\n```json\n{json.dumps(memory_map, indent=2)}\n```")

    if clock_tree and (clock_tree.get("domains") or clock_tree.get("result")):
        parts.append(f"## clock_tree.json\n```json\n{json.dumps(clock_tree, indent=2)}\n```")

    if register_spec and (register_spec.get("register_blocks") or register_spec.get("result")):
        parts.append(f"## register_spec.json\n```json\n{json.dumps(register_spec, indent=2)}\n```")

    if benchmark_results:
        parts.append(f"## benchmark_results\n```json\n{json.dumps(benchmark_results, indent=2)}\n```")

    if pdk_config:
        parts.append(f"## pdk_config\n```json\n{json.dumps(pdk_config, indent=2)}\n```")

    root = Path(project_root)
    for doc_name, doc_label in [
        ("sad_spec.md", "SAD (System Architecture Document)"),
        ("frd_spec.md", "FRD (Functional Requirements Document)"),
        ("ers_spec.md", "ERS (Engineering Requirements Specification)"),
    ]:
        doc_path = root / "arch" / doc_name
        if doc_path.exists():
            try:
                parts.append(f"## {doc_label}\n{doc_path.read_text()}")
            except OSError:
                pass

    return "\n\n".join(parts)


_SUBAGENT_SYSTEM_PROMPT = """\
You verify a single architectural constraint against a design's artifacts.

You will receive:
- The constraint description (one named rule).
- The full artifact bundle (requirements, PRD/ERS, block_diagram, optional
  memory_map / clock_tree / register_spec, optional SAD/FRD).

Your job: read the artifacts carefully, decide PASS or FAIL on the one
constraint, and respond with JSON only.

Rules of engagement:
- Be conservative. Only FAIL when you can cite specific evidence in the
  artifacts that the constraint is violated.
- A line that is itself a *forbidden pattern warning* in the requirements
  doc ("do not transpose it to 45 columns by 80 rows", "incorrect: ...",
  "avoid using X") is NOT a claim to verify. The architecture is being
  warned against that pattern, not asserting it. Skip those.
- If the constraint genuinely does not apply to this design (e.g., the
  constraint is about memory maps but the design has no memory map), PASS
  with evidence "constraint not applicable to this design".
- Do not invent constraints. If you see something else wrong, ignore it;
  another subagent owns it.

Response format (JSON only, no surrounding prose):

  {
    "pass": <true | false>,
    "evidence": "<short citation from the artifacts; quote the relevant
                  field path or snippet that justifies your verdict>",
    "violation_text": "<required only when pass=false; one-sentence
                        human-readable description of what's wrong>",
    "suggested_fix": "<required only when pass=false; concrete change>"
  }
"""


async def _run_constraint_subagent(
    constraint: dict,
    artifact_bundle: str,
    timeout_seconds: int,
) -> list[dict]:
    """Run a focused LLM subagent for ONE constraint. Returns a list of
    violations (empty list means PASS, one or more dicts means FAIL).

    Subagent failures (timeout, parse error) are reported as a warning-
    severity violation so the caller can see that this constraint was
    not actually verified.
    """
    from orchestrator.langchain.agents.coresmith_llm import (
        DEFAULT_MODEL,
        ClaudeLLM,
    )

    llm = ClaudeLLM(model=DEFAULT_MODEL, timeout=timeout_seconds)

    user_prompt = (
        "## Constraint to verify\n"
        f"**id:** `{constraint['id']}`\n\n"
        f"**rule:**\n{constraint['description']}\n\n"
        "## Artifact bundle\n"
        f"{artifact_bundle}\n\n"
        "Verify the constraint. Respond with JSON only."
    )

    try:
        content = await llm.call(
            system=_SUBAGENT_SYSTEM_PROMPT,
            prompt=user_prompt,
            run_name=f"constraint_subagent:{constraint['id']}",
        )
    except Exception as exc:
        return [{
            "violation": (
                f"Constraint subagent '{constraint['id']}' did not complete: "
                f"{type(exc).__name__}: {exc}. This constraint was NOT verified."
            ),
            "category": "structural",
            "check": f"{constraint['id']}_subagent_error",
            "severity": "warning",
        }]

    parsed = _parse_subagent_response(content)
    if parsed.get("pass"):
        return []

    return [{
        "violation": parsed.get(
            "violation_text",
            f"Constraint '{constraint['id']}' failed (no description supplied)."
        ),
        "category": constraint["category"],
        "check": constraint["id"],
        "severity": constraint["severity"],
        "evidence": (parsed.get("evidence") or "")[:1000],
        "suggested_fix": (parsed.get("suggested_fix") or "")[:600],
    }]


def _parse_subagent_response(content: str) -> dict:
    """Extract the subagent's JSON verdict. Tolerant of code fences and
    surrounding prose; falls back to {"pass": false} if unparseable so an
    unparseable subagent is treated as a failure, not a silent pass."""
    from orchestrator.utils import parse_llm_json

    default = {"pass": False, "evidence": "", "violation_text": "", "suggested_fix": ""}
    result, ok = parse_llm_json(content, default, context="constraint_subagent")
    if not ok:
        result["violation_text"] = (
            "Subagent response was not valid JSON. Treating as failure to "
            "avoid silently passing a constraint."
        )
    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def check_constraints(
    block_diagram: dict,
    memory_map: dict,
    clock_tree: dict,
    register_spec: dict,
    benchmark_results: dict | None = None,
    pdk_config: dict | None = None,
    requirements: str = "",
    ers_spec: dict | None = None,
    project_root: str = ".",
) -> list[dict]:
    """Validate cross-cutting architectural constraints.

    1. Run deterministic shuttle checks if enabled (structured arithmetic).
    2. Fan out a focused LLM subagent for each registered constraint that
       applies to this design.

    Args mirror the previous regex-based implementation so the architecture
    graph's call site stays unchanged.

    Returns a list of violation dicts (empty list = all constraints pass).
    Each dict has at least: `violation`, `category`, `check`, `severity`,
    and optionally `evidence` + `suggested_fix` from the subagent.
    """
    from opentelemetry import trace as _trace

    tracer = _trace.get_tracer("coresmith.architecture.constraints")

    with tracer.start_as_current_span("check_constraints") as span:
        shuttle_enabled = _shuttle_constraints_enabled(requirements, ers_spec)
        shuttle_violations = (
            _check_shuttle_constraints(block_diagram, ers_spec)
            if shuttle_enabled
            else []
        )

        artifact_bundle = _build_artifact_bundle(
            block_diagram=block_diagram or {},
            memory_map=memory_map or {},
            clock_tree=clock_tree or {},
            register_spec=register_spec or {},
            benchmark_results=benchmark_results,
            pdk_config=pdk_config,
            requirements=requirements,
            ers_spec=ers_spec,
            project_root=project_root,
        )

        applicability_ctx = {
            "block_diagram": block_diagram or {},
            "memory_map": memory_map or {},
            "clock_tree": clock_tree or {},
            "register_spec": register_spec or {},
            "ers_spec": ers_spec or {},
        }
        applicable = []
        for constraint in _CONSTRAINT_CATALOG:
            try:
                if constraint["applies"](applicability_ctx):
                    applicable.append(constraint)
            except Exception:
                # A buggy applies() predicate shouldn't break the whole
                # checker; treat as "applies" and let the subagent decide.
                applicable.append(constraint)

        span.set_attribute("subagent_count", len(applicable))
        span.set_attribute("shuttle_enabled", shuttle_enabled)
        span.set_attribute("shuttle_violation_count", len(shuttle_violations))

        timeout_seconds = int(
            os.environ.get("CORESMITH_CONSTRAINT_SUBAGENT_TIMEOUT", "600")
        )

        subagent_results = await asyncio.gather(
            *[
                _run_constraint_subagent(c, artifact_bundle, timeout_seconds)
                for c in applicable
            ],
            return_exceptions=False,
        )

        llm_violations: list[dict] = [v for batch in subagent_results for v in batch]
        violations = shuttle_violations + llm_violations

        span.set_attribute("llm_violation_count", len(llm_violations))
        span.set_attribute("violation_count", len(violations))
        span.set_attribute("all_pass", len(violations) == 0)
        if violations:
            span.set_attribute(
                "violations",
                "; ".join(v.get("violation", "") for v in violations[:5])[:2000],
            )

        return violations


# ---------------------------------------------------------------------------
# Small util kept for callers/tests that already import it
# ---------------------------------------------------------------------------

def _safe_int(value: Any, default: int) -> int:
    """Extract an integer from a value that may be None, str, or numeric."""
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default
