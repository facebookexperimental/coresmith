"""Skill loader + per-call skill SELECTION for agent system prompts.

Skills are reference markdown documents under
`orchestrator/langchain/prompts/skills/`. Agents that author or consume
hardware interfaces should pull the relevant skills into their system
prompt so they have access to coresmith's interface conventions.

Usage:
    from orchestrator.langchain.prompts.skills import load_skill

    sys_prompt = base_prompt + "\\n\\n" + load_skill("axi_stream")

Each skill is a markdown file named `<id>.md`. The top-level heading
and surrounding text become the prompt fragment as-is.

WHY SELECTION EXISTS
--------------------
The uArch generator used to concatenate ten skills at IMPORT time, so every
block -- a register file included -- carried the full 18 KB bitstream
serialization skill plus 115 KB of other reference material it could not use.
Measured on a live run: a 141 K-char system prompt of which ~133 K was skills.

:func:`select_skills` maps EVIDENCE (the block's spec/description/model source,
its contract edges, its block-diagram slice) to the skills that evidence
implicates, and everything else is listed in a compact MANIFEST -- name, one
sentence, absolute path -- with the instruction to READ THE FILE before
authoring in that domain. The worker has filesystem read access, so nothing
becomes unavailable; only the unconditional token tax goes away.

The classifier is deliberately CONSERVATIVE: a rule that cannot judge (its
evidence channel was not supplied) votes INCLUDE. A false include costs ~18 K
chars; a false exclude costs a defect class.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_SKILLS_DIR = Path(__file__).resolve().parent


class MissingSkillError(RuntimeError):
    """A skill selected for INLINE injection has no file on disk.

    Raised loudly at first use rather than degrading to an empty fragment: an
    agent that silently loses its interface conventions produces RTL that
    passes lint and fails composition.
    """


#: One sentence per skill, used for the "not inlined" manifest. Every skill
#: that can be SELECTED must have an entry -- :func:`skill_manifest` raises
#: otherwise, so adding a skill file without describing it fails loudly.
SKILL_PURPOSES: dict[str, str] = {
    "arithmetic_precision": (
        "bit-width derivation, Q-format, rounding/saturation/wraparound rules "
        "for fixed-point datapaths"
    ),
    "axi_stream": (
        "AXI-Stream tvalid/tready/tlast/tuser framing, drain/EOF handshake and "
        "the ack-and-drop trap"
    ),
    "buffer_stride_contract": (
        "byte-identical buffer read/write address expressions; slot and row "
        "stride agreement between producer and consumer"
    ),
    "control_pulse_handshake": (
        "single-cycle control pulses (start/done/busy/event) and how they are "
        "latched, acknowledged and cleared"
    ),
    "feedback_coupled_decomposition": (
        "how to split blocks that sit in a feedback loop without creating a "
        "combinational cycle"
    ),
    "memory_macro_vs_flops": (
        "when storage must be a cs_sram macro instead of a flop array, and the "
        "available sky130 SRAM geometries"
    ),
    "no_stimulus_keyed_memorization": (
        "why a model/RTL must not key its behaviour off the test stimulus "
        "(anti-cheat)"
    ),
    "output_contract_ownership": (
        "which block owns each output field so no field is emitted twice or "
        "dropped between blocks"
    ),
    "pipeline_contract": (
        "arithmetic-per-stage scheduling: realize each declared stage as a "
        "registered boundary instead of one combinational cloud"
    ),
    "port_naming": (
        "the canonical <channel>_<field> port name and the contract's primacy "
        "over golden-model port identifiers"
    ),
    "qspi_slave_frontend_protocol": (
        "complete QSPI-slave bus protocol for the block that owns the external "
        "chassis boundary"
    ),
    "serialization_contract": (
        "bitstream serialization: emission cadence, field order, container "
        "framing and terminal flush"
    ),
    "srdy_drdy": (
        "sRdy/dRdy handshake semantics: a transfer is valid && ready sampled on "
        "the clock edge, nothing more"
    ),
    "throughput_budget_contract": (
        "declared cycles/op and initiation interval as a hard, measured "
        "contract"
    ),
    "verify_in_context": (
        "run your own check before declaring the artifact done"
    ),
}

#: The reference-skill catalog the per-block AUTHORING agents (uArch spec,
#: block model) choose from. Order is stable so assembled prompts are
#: reproducible (and prompt-cacheable).
UARCH_SKILL_CANDIDATES: tuple[str, ...] = (
    "axi_stream",
    "srdy_drdy",
    "arithmetic_precision",
    "memory_macro_vs_flops",
    "serialization_contract",
    "buffer_stride_contract",
    "pipeline_contract",
    "throughput_budget_contract",
    "control_pulse_handshake",
)

#: Injected INLINE on every call, never manifested. Small (~2 K) and the one
#: rule with no cheap recovery path: a collapsed port name is not caught until
#: the pre-sim conformance gate, and costs a full regeneration.
ALWAYS_INLINE: tuple[str, ...] = ("port_naming",)


def skill_path(skill_id: str) -> Path:
    """Absolute path of a skill's markdown file (whether or not it exists)."""
    return _SKILLS_DIR / f"{skill_id}.md"


def load_skill(skill_id: str) -> str:
    """Return the markdown body of a skill, or empty string if missing.

    `skill_id` matches the filename stem (e.g. `"axi_stream"` for
    `axi_stream.md`). Missing skills return empty rather than raising
    so a prompt builder degrades gracefully.
    """
    path = skill_path(skill_id)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def load_skill_strict(skill_id: str) -> str:
    """Like :func:`load_skill` but RAISES when the file is missing.

    Used by the per-call assembly path: a skill the classifier decided this
    block needs must actually be in the prompt, and an empty string is not a
    detectable failure downstream.
    """
    path = skill_path(skill_id)
    try:
        body = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise MissingSkillError(
            f"skill '{skill_id}' was selected for inline injection but "
            f"{path} could not be read: {exc}"
        ) from exc
    if not body.strip():
        raise MissingSkillError(
            f"skill '{skill_id}' was selected for inline injection but "
            f"{path} is empty"
        )
    return body


def load_skills(*skill_ids: str, separator: str = "\n\n---\n\n") -> str:
    """Concatenate multiple skills for inclusion in a system prompt.

    Empty/missing skills are skipped so adding a new skill id never
    breaks an agent that hasn't shipped it yet.
    """
    parts = [load_skill(sid) for sid in skill_ids]
    return separator.join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Evidence extraction
# ---------------------------------------------------------------------------

def _text_of(obj: Any) -> str:
    """Flatten any JSON-ish object to lowercase searchable text."""
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj.lower()
    try:
        return json.dumps(obj, default=str).lower()
    except (TypeError, ValueError):
        return str(obj).lower()


def _edges(contracts: Any) -> list[dict]:
    """Accept any shape the callers actually hold.

    ``contracts`` may be the full ``interface_contracts.json`` dict, the
    per-block view from ``contract_lookup.load_block_contracts`` (``{"edges":
    [...]}``), a bare list of edges, or a pre-rendered string.
    """
    if isinstance(contracts, dict):
        for key in ("edges", "contracts"):
            v = contracts.get(key)
            if isinstance(v, list):
                return [e for e in v if isinstance(e, dict)]
        return []
    if isinstance(contracts, list):
        return [e for e in contracts if isinstance(e, dict)]
    return []


def _signal_names(edges: list[dict]) -> list[str]:
    names: list[str] = []
    for e in edges:
        for key in ("fields", "sideband_signals"):
            for f in e.get(key) or []:
                n = f.get("name") if isinstance(f, dict) else f
                if n:
                    names.append(str(n).lower())
        for key in ("producer_port", "consumer_port"):
            v = e.get(key)
            if v:
                names.append(str(v).lower())
    return names


#: Contract VALUES that carry design intent. The contract schema's KEYS
#: ("handshake_protocol", "consumer_can_stall", "signed", ...) are identical in
#: every design, so dumping raw JSON would make several rules fire on schema
#: boilerplate rather than on evidence. Only these values are searched.
#:
#: LABELS are machine-chosen identifiers (protocol name, channel name, packing
#: convention). PROSE is free text an LLM wrote about the edge. They are kept
#: apart because prose lies in the direction that matters: a real contract note
#: reads "this is a physical pin bundle, NEVER axi or srdy/drdy", and a
#: substring search over it selects both handshake skills for a block whose
#: labels say `static`.
_CONTRACT_LABEL_KEYS = (
    "edge_id", "producer_block", "consumer_block", "producer_port",
    "consumer_port", "handshake_protocol", "packing_convention", "encoding",
    "policy_type", "semantics",
)
_CONTRACT_PROSE_KEYS = ("rate_description", "notes", "rationale", "description")


def _contract_evidence_text(contracts: Any, edges: list[dict]) -> tuple[str, str]:
    """``(labels, prose)`` from contract VALUES (never the schema's keys)."""
    if isinstance(contracts, str):
        return contracts.lower(), ""
    labels: list[str] = []
    prose: list[str] = []
    if isinstance(contracts, dict):
        v = contracts.get("default_packing_convention")
        if isinstance(v, str):
            labels.append(v)
        for k in ("design_summary", "default_endianness_rationale"):
            v = contracts.get(k)
            if isinstance(v, str):
                prose.append(v)

    def _walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    _walk(v)
                elif v is None:
                    continue
                elif k in _CONTRACT_LABEL_KEYS:
                    labels.append(str(v))
                elif k in _CONTRACT_PROSE_KEYS:
                    prose.append(str(v))
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    for e in edges:
        _walk(e)
    return " ".join(labels).lower(), " ".join(prose).lower()


class _Evidence:
    """What we actually know about the block, per channel.

    Each channel is independently "supplied or not". A rule whose channel was
    not supplied returns ``None`` (= no judgment) and the caller INCLUDES.
    """

    def __init__(self, block_spec: Any, contracts: Any, block_diagram: Any):
        spec = block_spec if isinstance(block_spec, dict) else {}
        self.spec = spec
        self.spec_text = _text_of(block_spec) if block_spec else ""
        self.model_source = str(
            spec.get("model_source")
            or spec.get("python_source")
            or spec.get("golden_source")
            or spec.get("source")
            or ""
        ).lower()
        self.edges = _edges(contracts)
        self.signals = _signal_names(self.edges)
        self.contract_labels, self.contract_prose = (
            _contract_evidence_text(contracts, self.edges) if contracts
            else ("", "")
        )
        self.contract_text = f"{self.contract_labels} {self.contract_prose}"
        self.diagram_text = _text_of(block_diagram) if block_diagram else ""
        self.has_spec = bool(block_spec)
        self.has_contracts = bool(contracts)
        self.has_diagram = bool(block_diagram)
        self.has_model = bool(self.model_source.strip())
        self.any = self.has_spec or self.has_contracts or self.has_diagram
        self.all_text = " ".join(
            (self.spec_text, self.contract_text, self.diagram_text)
        )
        #: Everything EXCEPT the contract's free prose -- what the protocol
        #: rules judge on, so a note that says "never AXI" cannot select the
        #: AXI skill.
        self.label_text = " ".join(
            (self.spec_text, self.contract_labels, self.diagram_text)
        )

    def hit(self, *tokens: str) -> bool:
        """Substring match -- for tokens long enough to be unambiguous."""
        hay = self.all_text
        return any(t in hay for t in tokens)

    def label_hit(self, *tokens: str) -> bool:
        """Substring match over LABELS only (contract prose excluded)."""
        hay = self.label_text
        return any(t in hay for t in tokens)

    def label_word(self, *tokens: str) -> bool:
        hay = self.label_text
        return any(
            re.search(r"(?<![a-z0-9_])" + re.escape(t) + r"(?![a-z0-9_])", hay)
            for t in tokens
        )

    def word(self, *tokens: str) -> bool:
        """Word-boundary match -- for short tokens.

        ``"rom"``/``"lut"``/``"round"`` as substrings match ``from``,
        ``absolute`` and ``background``; that is how a classifier ends up
        including every skill for every block and proving nothing.
        """
        hay = self.all_text
        return any(
            re.search(r"(?<![a-z0-9_])" + re.escape(t) + r"(?![a-z0-9_])", hay)
            for t in tokens
        )

    def signal_hit(self, *tokens: str) -> bool:
        return any(t in s for s in self.signals for t in tokens)


# --- per-skill rules: True = include, False = exclude, None = cannot judge ---

def _rule_memory(ev: _Evidence) -> bool | None:
    if not ev.any:
        return None
    if any(k in ev.spec for k in ("memory_map", "sram", "storage_bits",
                                  "sram_budget", "memories")):
        return True
    if ev.hit("memory_map", "sram", "register file", "register_file",
              "regfile", "scratchpad", "line buffer", "line_buffer",
              "frame buffer", "framebuffer", "lookup table", "buffer",
              "memory", "storage"):
        return True
    if ev.word("rom", "ram", "lut", "fifo", "cache", "mem", "dram", "cs_sram"):
        return True
    if ev.signal_hit("addr", "waddr", "raddr", "wdata", "rdata", "wmask",
                     "write_enable", "read_enable"):
        return True
    return False


def _rule_axi(ev: _Evidence) -> bool | None:
    # Judged on LABELS (protocol/channel names) + signal names, never on the
    # contract's free prose.
    if ev.label_hit("tvalid", "tready", "tdata", "tlast", "tuser",
                    "axi_stream", "axi-stream", "axis_", "_axis"):
        return True
    # "stream" as a WORD only: `encrypt_stream` is a golden-model function
    # name, not an interface label.
    if ev.label_word("axi", "beat", "beats", "stream", "streams", "streaming"):
        return True
    if ev.signal_hit("tvalid", "tready", "tdata", "tlast", "tuser"):
        return True
    # The interface labels live in the CONTRACT. Without it we are guessing.
    return False if ev.has_contracts else None


def _rule_srdy(ev: _Evidence) -> bool | None:
    if ev.label_hit("srdy", "drdy", "backpressure", "back-pressure",
                    "valid/ready", "ready/valid", "valid_ready", "ready_valid",
                    "elastic", "req_ack", "req/ack", "handshak"):
        return True
    if ev.label_word("credit"):
        return True
    if ev.signal_hit("srdy", "drdy", "valid", "ready", "_vld", "_rdy"):
        return True
    return False if ev.has_contracts else None


def _rule_serialization(ev: _Evidence) -> bool | None:
    if not ev.any:
        return None
    if ev.hit("serializ", "serialis", "bitstream", "bit stream", "bit-stream",
              "packing", "bitpack", "bit-pack", "unpack", "entropy",
              "huffman", "varint", "framing", "marshal", "encoder", "decoder",
              "codeword"):
        return True
    if ev.hit("packed bit", "bit-packed", "packed record of", "pack into"):
        return True
    if ev.word("container", "payload", "tlv", "emit", "encode", "decode",
               "escape"):
        return True
    if ev.signal_hit("bitstream", "bit_len", "bitlen", "nbits", "packed",
                     "byte_out", "flush", "eob"):
        return True
    return False


def _rule_buffer_stride(ev: _Evidence) -> bool | None:
    # Same evidence class as serialization (a packed buffer is read back by
    # slot/stride), plus explicit stride/geometry language.
    if not ev.any:
        return None
    if _rule_serialization(ev):
        return True
    if ev.hit("stride", "sub-block", "subblock", "sub_block",
              "row_offset", "slot index", "slot_index", "byte offset",
              "scan order", "scan_order"):
        return True
    if ev.word("tile", "tiles"):
        return True
    return False


def _rule_pipeline(ev: _Evidence) -> bool | None:
    if not ev.any:
        return None
    if any(k in ev.spec for k in ("stages", "pipeline_depth", "stage_budget",
                                  "latency_cycles")):
        return True
    if ev.hit("pipeline", "multi-cycle", "multicycle", "state machine",
              "iterat", "sequential", "latency", "throughput"):
        return True
    if ev.word("stage", "stages", "fsm", "round", "rounds", "cycles"):
        return True
    return False


def _rule_throughput(ev: _Evidence) -> bool | None:
    if not ev.any:
        return None
    if any(k in ev.spec for k in ("perf", "throughput", "cycles_per_op",
                                  "cyc_per_op", "initiation_interval",
                                  "target_throughput")):
        return True
    if ev.hit("throughput", "cycles/op", "cycles per op", "cyc/op",
              "initiation interval", "ii=", "samples per", "bandwidth",
              "per second", "mbps", "gbps", "fps", "frames/s", "throughput_"):
        return True
    return False


def _rule_control_pulse(ev: _Evidence) -> bool | None:
    if not ev.any:
        return None
    if ev.hit("control pulse", "single-cycle pulse", "start/done",
              "interrupt", "trigger"):
        return True
    if ev.word("irq", "event", "events", "pulse", "abort"):
        return True
    if ev.signal_hit("start", "done", "busy", "kick", "go_", "_go", "trigger",
                     "pulse", "irq", "abort", "cmd_valid", "soft_reset",
                     "launch", "complete"):
        return True
    return False


def _rule_arithmetic(ev: _Evidence) -> bool | None:
    if ev.hit("arithmetic", "fixed-point", "fixed point", "q-format",
              "quantiz", "saturat", "truncat", "multiply", "multiplier",
              "accumulat", "dot product", "filter", "transform", "galois",
              "overflow", "underflow", "rounding"):
        return True
    if ev.word("round", "mac", "fft", "dct", "idct", "gf", "add", "sum",
               "scale", "divide", "sqrt"):
        return True
    if not ev.has_model:
        # No model source supplied -> we cannot see the datapath math. Include.
        return None
    if re.search(r"[*/%]|<<|>>|\bmath\.|\bround\b|\bint\(|\bfloat\(",
                 ev.model_source):
        return True
    return False


_RULES = {
    "axi_stream": _rule_axi,
    "srdy_drdy": _rule_srdy,
    "arithmetic_precision": _rule_arithmetic,
    "memory_macro_vs_flops": _rule_memory,
    "serialization_contract": _rule_serialization,
    "buffer_stride_contract": _rule_buffer_stride,
    "pipeline_contract": _rule_pipeline,
    "throughput_budget_contract": _rule_throughput,
    "control_pulse_handshake": _rule_control_pulse,
}


def select_skills(
    block_spec: Any = None,
    contracts: Any = None,
    block_diagram: Any = None,
    candidates: tuple[str, ...] | list[str] = UARCH_SKILL_CANDIDATES,
) -> list[str]:
    """Deterministically pick the skills this block's evidence implicates.

    ``block_spec`` is anything dict-ish describing the block (``name``,
    ``description``, ``model_source``/``python_source``, budgets...);
    ``contracts`` is the block's contract slice in any of the shapes callers
    hold; ``block_diagram`` is the diagram (or this block's slice of it).

    Returns a list ordered by ``candidates`` (stable -> prompt-cacheable).

    CONSERVATIVE: with no evidence at all, every candidate is returned. A rule
    whose evidence channel was not supplied votes include.
    """
    ev = _Evidence(block_spec, contracts, block_diagram)
    if not ev.any:
        return list(candidates)
    out: list[str] = []
    for sid in candidates:
        rule = _RULES.get(sid)
        if rule is None:
            out.append(sid)          # unknown skill -> cannot judge -> include
            continue
        verdict = rule(ev)
        if verdict is None or verdict:
            out.append(sid)
    return out


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

_MANIFEST_HEADER = (
    "# Reference skills NOT inlined (READ THE FILE before authoring in that "
    "domain)\n\n"
    "These reference documents were not inlined because this block's spec, "
    "contract and diagram show no evidence they apply. They are NOT optional "
    "if you turn out to need them: you have filesystem read access on this "
    "machine, so if your design touches one of these domains you MUST read "
    "the file at the path given before writing anything in that domain.\n"
)


def skill_manifest(skill_ids) -> str:
    """One compact line per NOT-inlined skill: name, purpose, absolute path."""
    ids = [s for s in skill_ids]
    if not ids:
        return ""
    lines = [_MANIFEST_HEADER]
    for sid in ids:
        purpose = SKILL_PURPOSES.get(sid)
        if not purpose:
            raise MissingSkillError(
                f"skill '{sid}' has no SKILL_PURPOSES entry; add one so the "
                "manifest can describe it"
            )
        lines.append(f"- `{sid}` -- {purpose}. READ: {skill_path(sid)}")
    return "\n".join(lines)


def build_skill_section(
    selected,
    candidates: tuple[str, ...] | list[str] = UARCH_SKILL_CANDIDATES,
    always: tuple[str, ...] | list[str] = ALWAYS_INLINE,
    heading: str = "# Reference Skills (use when authoring interfaces)",
    separator: str = "\n\n---\n\n",
) -> str:
    """Inline the selected (+ always) skill bodies; manifest the rest.

    Inlined skills are loaded STRICTLY -- a missing file raises
    :class:`MissingSkillError` at first use rather than silently shrinking the
    prompt.
    """
    sel = list(dict.fromkeys(list(always) + [s for s in selected]))
    bodies = [load_skill_strict(sid) for sid in sel]
    excluded = [c for c in candidates if c not in sel]
    parts: list[str] = []
    if bodies:
        parts.append(heading + "\n\n" + separator.join(bodies))
    manifest = skill_manifest(excluded)
    if manifest:
        parts.append(manifest)
    return "\n\n".join(parts)


__all__ = [
    "ALWAYS_INLINE",
    "MissingSkillError",
    "SKILL_PURPOSES",
    "UARCH_SKILL_CANDIDATES",
    "build_skill_section",
    "load_skill",
    "load_skill_strict",
    "load_skills",
    "select_skills",
    "skill_manifest",
    "skill_path",
]
