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

Origin: surfaced by the v5 video_codec codec_v3 autopilot run, where the
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
_SKILLS_TEXT = _load_skills(
    "axi_stream", "srdy_drdy", "arithmetic_precision", "serialization_contract",
    "buffer_stride_contract", "qspi_slave_frontend_protocol")
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

        # SELECTION: anchor each contract's handshake_protocol to the block-
        # diagram edge's authoritative declared family, overriding invented
        # port spelling, BEFORE structural validation runs (so the feedback-
        # cycle exemption in _validate_contracts is anchored to the diagram
        # edge, not to self-consistent specialist invention). Gated default-ON.
        result, propagation_notes = _propagate_edge_families(result, connections)
        span.set_attribute("family_propagation_notes", len(propagation_notes))

        # PERSIST the post-processed (family-anchored + backpressure-neutralized)
        # contracts back to interface_contracts.json. _propagate_edge_families
        # mutates the in-memory ``result`` ONLY; without this write-back the
        # on-disk artifact keeps the RAW LLM policy (e.g. request_response +
        # feedback_cycle=true on a valid_only / static edge), and BOTH downstream
        # consumers read that stale file:
        #   * the deterministic interface-family coherence gate reloads the file
        #     from disk (``check_constraints`` is called with
        #     ``interface_contracts=None``, so ``run_constraint_check`` re-reads
        #     interface_contracts.json), so the honest gate FAILS on the
        #     un-neutralized policy even though the returned state is correct; and
        #   * per-block uArch spec generators treat interface_contracts.json as
        #     authoritative for the flow-control policy.
        # Only rewrite when propagation actually changed something (notes
        # non-empty) so a design the LLM already got right keeps its original
        # file verbatim. Gated by the same env var (default ON) so
        # CORESMITH_INTERFACE_FAMILY_PROPAGATION=0 leaves the raw LLM file on
        # disk (pre-fix behavior). A write failure is non-fatal (fail-open).
        if _family_propagation_enabled() and propagation_notes:
            try:
                target_path.write_text(
                    json.dumps(result, indent=2), encoding="utf-8"
                )
                span.set_attribute("family_propagation_writeback", True)
            except OSError as exc:  # pragma: no cover -- disk failure is non-fatal
                span.set_attribute("family_propagation_writeback_error", str(exc))

        validated, validation_notes = _validate_contracts(result, connections)
        result.update(validated)
        all_notes = list(propagation_notes) + list(validation_notes)
        if all_notes:
            existing = result.get("validation_notes", []) or []
            result["validation_notes"] = list(existing) + all_notes

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


def _max_interface_width() -> int:
    """A-Fix 3(c): a contract that declares a bus wider than this without a
    ``serialized: true`` flag is a structural violation (the 7000-bit-bus
    class). Default 1024 bits; override with ``CORESMITH_MAX_INTERFACE_WIDTH``.
    """
    import os

    try:
        return max(1, int(os.environ.get("CORESMITH_MAX_INTERFACE_WIDTH", "1024")))
    except (TypeError, ValueError):
        return 1024


# ---------------------------------------------------------------------------
# Interface-family propagation (SELECTION): anchor each contract's
# handshake_protocol to the block-diagram edge's AUTHORITATIVE declared family.
# ---------------------------------------------------------------------------
#
# The block-diagram stage now declares an authoritative ``handshake_protocol``
# on every connection (block_diagram.md rule 4). The Interface Definition
# specialist's job is to freeze the bit layout -- NOT to re-decide the family
# from the port names it happened to invent. A stray ``wr_ready`` on a write-
# only-memory edge must NOT upgrade that edge to a backpressure stream. This
# deterministic post-processing step forces the declared family onto each
# contract and strips ready / response / elastic-FIFO artifacts that a
# no-backpressure family forbids. Env-gated (default ON) so
# ``CORESMITH_INTERFACE_FAMILY_PROPAGATION=0`` restores the pre-fix behavior
# (specialist's self-consistent invention is trusted verbatim).

_STREAMING_FAMILIES = frozenset({"axi_stream", "srdy_drdy"})
_NO_BACKPRESSURE_FAMILIES = frozenset({"mem_write", "valid_only", "static"})
_BACKPRESSURE_SEMANTICS = frozenset(
    {"elastic_fifo", "credit", "request_response", "skid"}
)
_READY_TOKENS = ("tready", "drdy", "ready")
# A field/sideband whose name carries a read-response qualifier is the
# fingerprint of an addressed read / request-response transaction. rdata /
# rvalid / rresp never appear on a pure write/strobe/static edge, so their
# presence means the edge MUST be req_resp -- never a no-backpressure family
# that would silently drop the response. Deliberately excludes the ambiguous
# bare "fault" token (a write edge may report a commit fault) so a real write
# edge is never mis-derived into req_resp.
_RESPONSE_FIELD_TOKENS = ("rdata", "rvalid", "rresp")


def _family_propagation_enabled() -> bool:
    """Default-ON gate for interface-family propagation (SELECTION)."""
    import os

    return (
        os.environ.get("CORESMITH_INTERFACE_FAMILY_PROPAGATION", "1") or "1"
    ) != "0"


def _edge_field(edge: dict, *keys: str) -> str:
    for k in keys:
        val = edge.get(k)
        if val:
            return str(val)
    return ""


class _EdgeFamilyIndex:
    """PORT-AWARE lookup of a block-diagram edge's declared ``handshake_protocol``.

    Keyed by ``(producer, consumer, port)`` for every port spelling the edge
    declares (its ``interface`` and any ``from_port`` / ``to_port`` / ``port``),
    so PARALLEL edges between the same block pair -- e.g. a ``req_resp`` read and
    a ``mem_write`` write BOTH between blocks A and B -- each keep their own
    declared family instead of the last edge written silently overwriting the
    first. That last-wins collision otherwise mislabels a read edge as
    ``mem_write`` (it inherits a sibling write edge's family when both share a
    block pair), corrupting every parallel-edge family the same way.

    A pair-only fallback resolves single-edge pairs and contracts whose ports
    were not spelled -- but ONLY when the pair declares exactly one family, so
    the fallback can never re-introduce the collision it exists to avoid.
    """

    def __init__(self) -> None:
        self._by_port: dict[tuple[str, str, str], str] = {}
        self._by_pair: dict[tuple[str, str], set[str]] = {}

    def add(self, src: str, dst: str, proto: str, ports: list[str]) -> None:
        for p in ports:
            if p:
                self._by_port[(src, dst, p)] = proto
        self._by_pair.setdefault((src, dst), set()).add(proto)

    def resolve(
        self,
        producer: str,
        consumer: str,
        producer_port: str = "",
        consumer_port: str = "",
        edge_id: str = "",
    ) -> str:
        """Return the declared family for this contract's edge, or "" if the
        diagram declares none (or a parallel-edge pair the ports don't
        disambiguate). Port match wins; the pair fallback fires only for an
        unambiguous single-family pair."""
        for port in (producer_port, consumer_port):
            if port:
                got = self._by_port.get((producer, consumer, port))
                if got:
                    return got
        protos = self._by_pair.get((producer, consumer))
        if protos and len(protos) == 1:
            return next(iter(protos))
        return ""

    def unambiguous_pairs(self) -> dict[tuple[str, str], str]:
        return {
            pair: next(iter(protos))
            for pair, protos in self._by_pair.items()
            if len(protos) == 1
        }


def build_edge_family_index(expected_edges: list[dict]) -> _EdgeFamilyIndex:
    """Build the PORT-AWARE index of block-diagram edges -> declared family.
    Edges with no declared ``handshake_protocol`` are skipped."""
    idx = _EdgeFamilyIndex()
    for edge in expected_edges or []:
        src = _edge_field(edge, "from", "from_block", "source")
        dst = _edge_field(edge, "to", "to_block", "target")
        proto = str(edge.get("handshake_protocol") or "").strip()
        if not (src and dst and proto):
            continue
        ports = [
            str(edge.get("interface") or "").strip(),
            str(edge.get("from_port") or "").strip(),
            str(edge.get("to_port") or "").strip(),
            str(edge.get("port") or "").strip(),
        ]
        idx.add(src, dst, proto, ports)
    return idx


def edge_intent_map(expected_edges: list[dict]) -> dict[tuple[str, str], str]:
    """Pair-level view of the port-aware edge index: ``(producer, consumer)``
    -> declared ``handshake_protocol``, for pairs that declare exactly ONE
    family. Pairs with PARALLEL edges of differing families are intentionally
    OMITTED (resolve them port-awarely via ``build_edge_family_index``) so this
    convenience map can never re-introduce the last-wins collision. Edges with
    no declared family are omitted."""
    return build_edge_family_index(expected_edges).unambiguous_pairs()


def _contract_carries_response(contract: dict[str, Any]) -> bool:
    """True when the contract exposes a read/response channel (a field or
    sideband named with an ``rdata`` / ``rvalid`` / ``rresp`` qualifier). This
    is the honest fingerprint of an addressed read / request-response
    transaction that MUST resolve to ``req_resp`` and must never be a
    no-backpressure write/strobe/static edge (which would drop the response).

    Keys off signal NAMES only -- NOT off ``flow_control_policy.semantics`` --
    so a mem_write edge the LLM merely mislabeled ``request_response`` (with no
    actual response field) is still correctly neutralized to a write, not
    upgraded to a read."""
    names: list[str] = []
    for f in contract.get("fields") or []:
        names.append(str((f or {}).get("name", "")).lower())
    for s in contract.get("sideband_signals") or []:
        names.append(str((s or {}).get("name", "")).lower())
    return any(any(tok in n for tok in _RESPONSE_FIELD_TOKENS) for n in names)


def _neutralize_backpressure(contract: dict[str, Any]) -> bool:
    """Strip ready / elastic-FIFO / stall backpressure from a contract whose
    family is always-accepted (mem_write / valid_only / static). Returns True if
    anything changed."""
    changed = False
    fc = contract.get("flow_control_policy")
    if isinstance(fc, dict):
        if str(fc.get("semantics") or "").strip() in _BACKPRESSURE_SEMANTICS:
            fc["semantics"] = "free_running"
            changed = True
        if fc.get("min_buffer_depth_beats"):
            fc["min_buffer_depth_beats"] = 0
            changed = True
        if fc.get("credit_words") is not None:
            fc["credit_words"] = None
            changed = True
        for k in ("consumer_can_stall", "producer_can_stall", "feedback_cycle"):
            if fc.get(k):
                fc[k] = False
                changed = True
    sb = contract.get("sideband_signals")
    if isinstance(sb, list):
        kept = [
            s
            for s in sb
            if not any(
                tok in str((s or {}).get("name", "")).lower()
                for tok in _READY_TOKENS
            )
        ]
        if len(kept) != len(sb):
            contract["sideband_signals"] = kept
            changed = True
    return changed


def _propagate_edge_families(
    result: dict[str, Any],
    expected_edges: list[dict],
) -> tuple[dict[str, Any], list[str]]:
    """Anchor each contract's ``handshake_protocol`` to the block-diagram edge's
    authoritative declared family (SELECTION), overriding invented port spelling,
    and strip ready / response / elastic-FIFO artifacts a no-backpressure family
    forbids. Returns ``(result, notes)``. Deterministic; no LLM. No-op when the
    gate is off or no edge declares a family."""
    notes: list[str] = []
    if not _family_propagation_enabled():
        return result, notes
    index = build_edge_family_index(expected_edges)
    for c in result.get("contracts", []) or []:
        producer = str(c.get("producer_block", ""))
        consumer = str(c.get("consumer_block", ""))
        producer_port = str(c.get("producer_port", ""))
        consumer_port = str(c.get("consumer_port", ""))
        edge_id = str(c.get("edge_id", ""))
        label = (
            f"{producer}.{producer_port}->{consumer}.{consumer_port}"
            if (producer_port or consumer_port)
            else f"{producer}->{consumer}"
        )
        # (1) Anchor handshake_protocol to the block-diagram edge's declared
        # family, resolved PORT-AWARELY so PARALLEL edges between the same block
        # pair (e.g. a req_resp read + a mem_write write) never overwrite each
        # other. A diagram edge that omits the family leaves the specialist's
        # own label in place.
        declared = index.resolve(
            producer, consumer, producer_port, consumer_port, edge_id
        )
        if declared:
            current = str(c.get("handshake_protocol") or "").strip()
            if current != declared:
                c["handshake_protocol"] = declared
                notes.append(
                    f"{label}: handshake_protocol anchored to block-diagram "
                    f"intent '{declared}' (specialist proposed "
                    f"'{current or 'unset'}'); declared family wins over invented "
                    "port spelling (port-aware, parallel-edge safe)."
                )
        # (1.5) RESPONSE PRESERVATION (SELECTION, defense-in-depth): an edge that
        # carries a response channel (rdata / rvalid / rresp) is an addressed
        # read / request-response transaction. It must NEVER resolve to a
        # no-backpressure family (mem_write / valid_only / static) -- those are
        # always-accepted writes/strobes and would silently DROP the response.
        # Derive req_resp BEFORE the monotonic flow-policy neutralization runs so
        # a genuine request-response edge is never free_running-ified. This fires
        # even when the diagram intent is missing/ambiguous (a parallel-edge
        # collision the ports couldn't disambiguate) or the LLM mislabeled the
        # read outright.
        final_family = str(c.get("handshake_protocol") or "").strip()
        if final_family in _NO_BACKPRESSURE_FAMILIES and _contract_carries_response(c):
            c["handshake_protocol"] = "req_resp"
            notes.append(
                f"{label}: family '{final_family}' carries a response channel "
                "(rdata/rvalid/rresp); derived req_resp -- an addressed "
                "read-with-response is never a no-backpressure write/strobe/"
                "static edge (never mem_write/valid_only/static)."
            )
            final_family = "req_resp"
        # req_resp keeps flow_control_policy.semantics = request_response and is
        # NEVER free_running. Coerce an empty/free_running policy so a read edge
        # anchored/derived out of a (possibly already-neutralized) no-backpressure
        # state is honest for downstream uArch spec generators.
        if final_family == "req_resp":
            fc = c.get("flow_control_policy")
            if not isinstance(fc, dict):
                fc = {}
                c["flow_control_policy"] = fc
            if (
                str(fc.get("semantics") or "").strip() in ("", "free_running")
                and fc.get("semantics") != "request_response"
            ):
                fc["semantics"] = "request_response"
                notes.append(
                    f"{label}: req_resp flow_control_policy.semantics set to "
                    "request_response (a request-response edge is not "
                    "free_running)."
                )
        # (2) Neutralize backpressure keyed off the contract's FINAL
        # handshake_protocol (post-anchor + post-response-preservation), NOT only
        # the diagram-edge intent. A no-backpressure family (mem_write /
        # valid_only / static) is always-accepted, so its flow_control_policy
        # must be free_running / no feedback / depth 0 regardless of what the LLM
        # proposed and regardless of whether the family label came from the
        # diagram edge or only from the contract itself. Keying off the final
        # family (a) fixes the valid_only/static edges the diagram-intent-only
        # guard skipped when its edge wasn't in the intent map, and (b) matches
        # the deterministic coherence gate, which authoritatively resolves the
        # family as `declared_intent or contract_family`.
        if final_family in _NO_BACKPRESSURE_FAMILIES:
            if _neutralize_backpressure(c):
                notes.append(
                    f"{label}: dropped ready / elastic-FIFO / backpressure "
                    f"artifacts forbidden by family '{final_family}' "
                    "(always-accepted edge; flow_control_policy forced to "
                    "free_running / no feedback / depth 0)."
                )
    return result, notes


def _validate_contracts(
    result: dict[str, Any],
    expected_edges: list[dict],
) -> tuple[dict[str, Any], list[str]]:
    """Structural validation of the specialist's output.

    A-Fix 3(c): these are now BLOCKING when they fire. The returned dict
    carries ``contract_violations`` (a list of structured violation dicts) in
    addition to the human-readable ``notes``. ``interface_definition_node``
    routes a non-empty violation set to the ``Escalate Constraints`` node.
    They are NOT a substitute for the cross_spec_contract_adherence subagent
    that runs at constraint-check time; they catch the obvious structural
    breaks: missing edges (when a contract set exists), fields that don't sum
    to the declared data width, overlapping bit ranges, feedback-cycle edges
    with unsafe flow-control semantics, under-depth elastic FIFOs, and the
    over-wide-bus sanity bound.
    """
    notes: list[str] = []
    violations: list[dict[str, Any]] = []
    max_width = _max_interface_width()

    def _violation(edge: str, vtype: str, description: str) -> None:
        notes.append(description)
        violations.append({
            "edge": edge,
            "type": vtype,
            "category": "structural",
            "severity": "error",
            "violation": description,
            "source": "interface_definition",
        })

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
        try:
            declared = int(c.get("data_width_bits", 0) or 0)
        except (TypeError, ValueError):
            declared = 0

        # Over-wide-bus sanity bound: a bus wider than max_width without an
        # explicit serialized flag is the parallel-mega-bus antipattern. Checked
        # before the fields guard so a fieldless mega-bus is still caught.
        serialized = bool(c.get("serialized"))
        if declared > max_width and not serialized:
            _violation(
                cid, "over_wide_bus",
                f"{cid}: declared data_width_bits={declared} exceeds the "
                f"{max_width}-bit interface sanity bound and the contract is "
                "not marked serialized: true; a parallel bus this wide is "
                "almost always an un-decomposed record -- serialize it or "
                "split the edge."
            )

        fields = list(c.get("fields", []) or [])
        if not fields:
            continue
        try:
            field_sum = sum(int(f.get("width", 0) or 0) for f in fields)
        except (TypeError, ValueError):
            notes.append(f"{cid}: non-numeric field width(s); skipping width sum check.")
            continue
        if declared and declared != field_sum:
            _violation(
                cid, "width_field_sum_mismatch",
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
                _violation(
                    cid, "field_overlap",
                    f"{cid}: fields {prev_name}[{prev_msb}:{prev_lsb}] and "
                    f"{cur_name}[{cur_msb}:{cur_lsb}] overlap."
                )

        # Flow-control sanity: any edge on a closed cycle must NOT
        # use free_running or skid semantics, because both assume the
        # producer or the consumer can hold a transaction indefinitely.
        # The v7/v8 video_codec deadlock landed exactly in this gap.
        producer = str(c.get("producer_block", ""))
        consumer = str(c.get("consumer_block", ""))
        fc = c.get("flow_control_policy") or {}
        semantics = (fc.get("semantics") or "").strip()
        # The feedback-cycle -> elastic_fifo requirement is a BACKPRESSURE-
        # STREAM guard (the v7/v8 video_codec deadlock class): it applies only to
        # streaming handshakes. A non-streaming edge -- a memory write/read
        # (mem_write/req_resp), an always-accepted pulse/bundle (valid_only),
        # or static wires (static) -- can close a graph cycle (a block both
        # writes and reads a memory looks like a 2-cycle) yet is fixed-latency
        # and always-accepted, so it is NOT a feedback stream and must NOT be
        # forced into an elastic_fifo. Exempt those families.
        #
        # The family is read from the contract's handshake_protocol, which
        # ``_propagate_edge_families`` has already ANCHORED to the block-diagram
        # edge's authoritative declaration (SELECTION) -- so this exemption keys
        # off the diagram-declared intent, NOT off self-consistent specialist
        # invention (a stray wr_ready can no longer flip a mem_write edge into a
        # streaming edge and force a bogus FIFO onto it).
        proto = str(c.get("handshake_protocol") or "").strip()
        _streaming = proto in ("", "axi_stream", "srdy_drdy")
        on_cycle = (producer, consumer) in cycle_edges and _streaming
        if on_cycle and semantics in ("", "free_running", "skid"):
            _violation(
                cid, "cycle_edge_semantics",
                f"{cid}: streaming edge participates in a feedback cycle but "
                f"flow_control_policy.semantics={semantics or 'unset'} "
                "— pick elastic_fifo, credit, or request_response."
            )
        if on_cycle and not fc.get("feedback_cycle"):
            _violation(
                cid, "cycle_edge_semantics",
                f"{cid}: streaming edge is in a feedback cycle in the block "
                "diagram but flow_control_policy.feedback_cycle is not true."
            )
        if semantics == "elastic_fifo":
            try:
                depth = int(fc.get("min_buffer_depth_beats") or 0)
            except (TypeError, ValueError):
                depth = 0
            if depth < 2:
                _violation(
                    cid, "fifo_depth",
                    f"{cid}: elastic_fifo requires min_buffer_depth_beats>=2; "
                    f"got {depth}."
                )

    # Cross-edge coverage check: every directed edge in the block diagram
    # should be represented by exactly one contract entry. Edge identity
    # is matched by (producer_block, consumer_block) pair OR by edge_id
    # if the specialist provided one matching the connection record.
    #
    # A total-absence result (contracts == []) is NOT treated as a per-edge
    # missing-contract violation: that is a specialist/transport failure the
    # design_summary already records and integration_check catches downstream.
    # We only block on missing edges when a contract SET exists but orphans an
    # edge (partial coverage) -- the orphaned-edge decomposition bug.
    if contracts:
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
                _violation(
                    f"{pair[0]}->{pair[1]}", "missing_contract",
                    f"missing contract for edge {pair[0]} -> {pair[1]} (declared "
                    "in block_diagram but absent from contracts[])."
                )
    elif expected_edges:
        notes.append(
            "no interface contracts produced for a design with "
            f"{len(expected_edges)} inter-block edge(s); "
            "downstream integration_check will validate cross-block wiring."
        )

    return ({"contract_violations": violations}, notes)


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
