# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
UarchSpecGenerator -- Agent that produces a microarchitecture spec.

Reads a Python golden model, understands the algorithm, and generates
a detailed microarchitecture specification that an RTL engineer (or the
RTL generator LLM) can implement unambiguously.

The spec covers interfaces, datapath, control FSM, storage elements,
algorithm mapping, timing, and edge cases -- every decision needed
before writing Verilog.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from opentelemetry import trace

from orchestrator._timeouts import scaled
from orchestrator.langchain.prompts.skills import (
    UARCH_SKILL_CANDIDATES,
    build_skill_section,
    select_skills,
)

from .coresmith_llm import ClaudeLLM

_tracer = trace.get_tracer(__name__)

_PROMPT_FILE = (
    Path(__file__).resolve().parent.parent / "prompts" / "uarch_spec_generator.md"
)
if _PROMPT_FILE.exists():
    SYSTEM_PROMPT = _PROMPT_FILE.read_text()
else:
    SYSTEM_PROMPT = (
        "You are an expert digital VLSI micro-architect. "
        "Produce a detailed microarchitecture specification from a Python model."
    )

# Reference skills are NO LONGER concatenated at import time. All ten used to
# be injected unconditionally, so a register-file block carried the full 18 K
# bitstream-serialization skill: measured on a live run, 133 K of a 141 K-char
# system prompt was skills, most of them inapplicable. They are now selected
# PER CALL from the block's own evidence (see build_system_prompt below);
# unselected skills are listed in a compact manifest with their absolute paths,
# which the worker (which has filesystem read access) must read before
# authoring in that domain. Nothing became unavailable; the token tax did.
#
# The missing-file fail-fast property is preserved and strengthened: a SELECTED
# skill is loaded with load_skill_strict, which raises loudly at first use
# instead of silently shrinking the prompt.

# Inject the LIVE set of pre-built SRAM macros discovered in the PDK, so the
# uArch author picks from what is actually available (not a hardcoded list) --
# including any macro generated on demand by the OpenRAM fallback. Best-effort:
# never let macro discovery break prompt construction.
try:
    from orchestrator.langgraph.macro_registry import (
        discover_macros as _discover_macros,
    )
    from orchestrator.langgraph.macro_registry import (
        macro_menu_markdown as _macro_menu_markdown,
    )

    _MACRO_MENU = _macro_menu_markdown(_discover_macros())
    # Surface whether OpenRAM is LIVE, so a non-pre-built-but-generatable geometry
    # is NOT flagged infeasible by conservatism (the model otherwise reasons "no
    # pre-built macro -> risk" even though OpenRAM would generate it).
    try:
        from orchestrator.langgraph.openram_gen import (
            openram_available as _oram_avail,
        )
        _OPENRAM_LIVE = _oram_avail()
    except Exception:  # noqa: BLE001
        _OPENRAM_LIVE = False
    if _OPENRAM_LIVE:
        _MACRO_POLICY = (
            "OpenRAM IS available on this flow: a WIDTH x DEPTH geometry NOT in the "
            "pre-built list below is generated on demand by OpenRAM at backend and "
            "is fully placeable. Treat any reasonable SRAM geometry as buildable -- "
            "do NOT record a non-pre-built geometry as an [area] or macro "
            "feasibility blocker; just declare the cs_sram WIDTH x DEPTH. (An "
            "[area] blocker is valid ONLY if the SRAM's priced area busts the "
            "block's area budget -- a decomposition/budget issue, not a "
            "macro-availability one.)"
        )
    else:
        _MACRO_POLICY = (
            "OpenRAM is NOT available on this flow: ONLY the pre-built macros listed "
            "below are placeable. A required WIDTH x DEPTH geometry NOT in the list "
            "below IS a genuine [area]/macro feasibility blocker -- flag it."
        )
    SYSTEM_PROMPT = (
        SYSTEM_PROMPT
        + "\n\n# On-chip memory (SRAM) policy\n\n"
        + "Any storage >= 16384 bits AND >= 256 words deep is an SRAM, not flops. "
        + "Spec it as a generic `cs_sram_1rw` / `cs_sram_1rw1r` wrapper "
        + "instance (parametrized by WIDTH/DEPTH) -- behavioral in simulation, "
        + "replaced by an SRAM macro at synth & backend. Do NOT spec a raw "
        + "reg-array memory that large (the lint gate rejects it), and do NOT "
        + "pin a specific macro name. "
        + _MACRO_POLICY
        + " The pre-built macros below are what the backend can place today; list "
        + "the WIDTH x DEPTH a block needs so resolution can match.\n\n"
        + _MACRO_MENU
    )
except Exception:  # pragma: no cover - discovery is best-effort
    pass


def _constraint_precedence_line() -> str:
    """Naming precedence for a block's ACCUMULATED constraints ('' on error).

    The accumulated constraints are a debug agent's reading of past failures;
    the frozen contract is design intent. On a port NAME they disagree about,
    the contract wins. Imported lazily so the agent module keeps no import-time
    dependency on the langgraph package.
    """
    try:
        from orchestrator.langgraph.contract_conformance import (
            CONSTRAINT_PRECEDENCE_LINE,
        )
        return CONSTRAINT_PRECEDENCE_LINE
    except Exception:  # noqa: BLE001 - prompt garnish, never blocks a spec
        return ""


def build_system_prompt(
    block_spec: Any = None,
    contracts: Any = None,
    block_diagram: Any = None,
) -> str:
    """The uArch author's system prompt for ONE block.

    ``SYSTEM_PROMPT`` (the authored prompt + the live SRAM macro menu) plus the
    reference-skill section assembled for this block: ``port_naming`` always
    inline, the skills this block's evidence implicates inline, everything else
    named in the manifest with an absolute path to read.
    """
    section = build_skill_section(
        select_skills(block_spec, contracts, block_diagram),
        candidates=UARCH_SKILL_CANDIDATES,
    )
    return SYSTEM_PROMPT + "\n\n" + section if section else SYSTEM_PROMPT


def normalize_feasibility(summary: dict) -> dict:
    """In-place normalise a uArch spec JSON summary's feasibility verdict.

    Guarantees ``summary["blocking_issues"]`` is a ``list[str]`` and
    ``summary["feasible"]`` is a ``bool`` (False whenever any blocking issue is
    present, even if the model forgot to flip ``feasible`` -- issues win). This
    is the contract the uarch->RTL feasibility gate tests by plain truthiness,
    mirroring backend_graph's ``blocking_issues`` pattern.
    """
    raw = summary.get("blocking_issues")
    if isinstance(raw, str):
        issues = [raw.strip()] if raw.strip() else []
    elif isinstance(raw, list):
        issues = [str(x).strip() for x in raw if str(x).strip()]
    else:
        issues = []
    summary["blocking_issues"] = issues
    feasible = summary.get("feasible")
    if not isinstance(feasible, bool):
        feasible = not issues
    summary["feasible"] = feasible and not issues
    return summary


def feasibility_from_spec_text(spec_text: str) -> tuple[bool, list]:
    """Extract ``(feasible, blocking_issues)`` from a spec's embedded ```json.

    Robust to the codex disk-first path: the canonical spec ``.md`` carries the
    JSON summary even when the agent's stdout did not, so the feasibility gate
    reads the on-disk spec rather than a possibly-fallback state value. Returns
    ``(True, [])`` when no parsable summary with a feasibility verdict is present
    (fail-open: absence of an explicit infeasible verdict = feasible, matching
    legacy specs that predate this field).
    """
    matches = re.findall(r"```json\s*\n(.*?)```", spec_text, re.DOTALL)
    for blk in reversed(matches):  # the summary is the LAST json block
        try:
            summary = json.loads(blk)
        except json.JSONDecodeError:
            continue
        if isinstance(summary, dict) and (
            "feasible" in summary or "blocking_issues" in summary
        ):
            normalize_feasibility(summary)
            return summary["feasible"], summary["blocking_issues"]
    return True, []


class UarchSpecGenerator:
    """Agent for generating microarchitecture specifications."""

    def __init__(self, model: str | None = None, temperature: float = 0.2):
        from orchestrator.langchain.agents.coresmith_llm import arch_reasoning_effort, block_model
        model = model or block_model()
        # The uarch spec carries the feasibility verdict + the frozen
        # microarchitecture every RTL attempt lowers -> higher reasoning tier
        # (codex-only; no-op on other providers).
        self.llm = ClaudeLLM(
            model=model,
            timeout=scaled(2700, env="CORESMITH_UARCH_TIMEOUT"),
            reasoning_effort=arch_reasoning_effort(),
        )

    async def generate(
        self,
        block_name: str,
        python_source: str,
        description: str = "",
        feedback: str = "",
        previous_spec: str = "",
        constraints: list[dict] | None = None,
        callbacks: list | None = None,
        project_root: str = "",
        resume_session_id: str | None = None,
    ) -> dict[str, Any]:
        """Generate a microarchitecture specification from Python source.

        Args:
            block_name: Name of the hardware block.
            python_source: Python source code of the golden model.
            description: Human-readable description of the block.
            feedback: Human feedback for revision (if revising a prior spec).
            previous_spec: The prior spec text being revised.
            constraints: Accumulated design constraints.
            callbacks: Optional callbacks (unused, kept for API compat).
            project_root: Path to project root for reading arch docs from disk.
            resume_session_id: When set (codex + CORESMITH_CODEX_RESUME), resume
                the block's prior codex session; ``None`` starts fresh. Dropping
                it on an entrenched respec is the fresh-session escalation.

        Returns:
            Dict with keys: spec_text, spec_summary, block_name
        """
        block_title = block_name.replace("_", " ").title()
        revision_label = " - Revision" if previous_spec else ""
        span_name = f"Uarch Spec [{block_title}]{revision_label}"

        with _tracer.start_as_current_span(span_name) as span:
            span.set_attribute("block_name", block_name)
            span.set_attribute("is_revision", bool(previous_spec))

            parts = [
                "Generate a microarchitecture specification for the following block.",
                f"\nBlock name: {block_name}",
                f"Description: {description}",
            ]

            # Evidence for per-call reference-skill selection (see
            # build_system_prompt). Populated below from disk when a
            # project_root is supplied; empty means "no evidence", and the
            # classifier then conservatively inlines every skill.
            _bd_block: dict = {}
            _contracts_view: dict = {}

            if constraints:
                parts.append("\n--- DESIGN CONSTRAINTS ---")
                for i, c in enumerate(constraints, 1):
                    if isinstance(c, dict):
                        parts.append(f"  {i}. {c.get('rule', str(c))}")
                    else:
                        parts.append(f"  {i}. {c}")
                parts.append(_constraint_precedence_line())
                parts.append("")

            # PDK arithmetic timing budget (gated; best-effort). Arms the author
            # with REAL per-op delays + how many chain per stage at the clock, so
            # it sizes pipeline stages instead of emitting a single-cycle
            # combinational cloud. Characterized once at the PRD stage; consumed
            # here. No-op unless CORESMITH_PDK_CHAR is on and the model is cached.
            #
            # The budget now prices the crypto/codec primitives too -- lut
            # (S-box/ROM), gfmul (Galois-field multiply), xortree (wide XOR) --
            # and carries a HARD PIPELINE-WITHIN-OP rule: for ANY op whose
            # predicted single-instance delay exceeds the target clock period,
            # the uArch MUST insert register(s) INSIDE the op and state the
            # stage count, citing the op-delay estimate. Injected as a MANDATORY
            # block (mirroring the flip_flop_budget / area_budget injections),
            # because an un-priced round op (AES S-box + MixColumns) collapsing
            # into one unregistered ~27 ns cloud is the exact Fmax-miss class
            # this section exists to prevent.
            try:
                from orchestrator.langgraph import arith_characterize as _arith
                from orchestrator.langgraph import pdk_characterize as _pdkc
                from orchestrator.langgraph.pipeline_scheduler import (
                    pdk_budget_section as _budget,
                )
                # P0 (chip-lead 2026-07-08): gate the crypto/timing budget
                # injection on the ARITHMETIC delay model ALONE. The budget uses
                # ZERO memory data, but the old guard called the COMBINED
                # pdk_characterize.is_characterized() (arith AND mem). On a box
                # whose memory table was not warmed for this PDK hash, that
                # returned False and the crypto budget (lut/gfmul/xortree) was
                # SILENTLY dropped -> the scheduler could not price the S-box /
                # MixColumns and emitted an unregistered ~27 ns cloud that misses
                # Fmax. Arith-only makes the Fmax fix robust; a loud warning
                # names the missing cache so a cold arith model is never silent.
                if _pdkc.stage_enabled():
                    if _arith.is_characterized():
                        _b = _budget(50.0)
                        if _b:
                            parts.append(
                                "\n--- PDK TIMING BUDGET (characterized on this PDK; "
                                "MANDATORY -- size pipeline stages from these REAL "
                                "per-op delays, and PIPELINE WITHIN any op whose "
                                "single-instance delay exceeds the clock period) ---\n"
                                + _b
                            )
                            try:
                                import logging as _lg
                                _lg.getLogger(__name__).info(
                                    "[PDK-CHAR] crypto/timing budget INJECTED "
                                    "(arith hash=%s liberty=%s)",
                                    _arith.pdk_hash(), _arith.LIBERTY,
                                )
                            except Exception:  # noqa: BLE001
                                pass
                    else:
                        import logging as _lg2
                        import sys as _sys2
                        try:
                            _h = _arith.pdk_hash()
                            _cp = _arith._cache_path()
                        except Exception:  # noqa: BLE001
                            _h, _cp = "?", "?"
                        _warn = (
                            "[PDK-CHAR] WARNING: CORESMITH_PDK_CHAR ENABLED but the "
                            "ARITHMETIC delay model is NOT characterized for this PDK "
                            f"(hash={_h}, expected cache {_cp}, liberty="
                            f"{getattr(_arith, 'LIBERTY', '?')}). Crypto/timing "
                            "budget (lut/gfmul/xortree) will NOT be injected -> the "
                            "scheduler cannot price the S-box / GF / wide-XOR clouds and "
                            "may emit an UNREGISTERED combinational cloud that misses "
                            "Fmax. Warm it first (ensure_pdk_characterized())."
                        )
                        _lg2.getLogger(__name__).warning(_warn)
                        print(_warn, file=_sys2.stderr, flush=True)
            except Exception:  # noqa: BLE001 - best-effort, never block spec gen
                pass

            if previous_spec and feedback:
                parts.append(
                    "\n--- REVISION REQUESTED ---\n"
                    f"The previous specification was reviewed and needs changes.\n"
                    f"Human feedback:\n{feedback}\n"
                    f"\n--- Previous Specification ---\n{previous_spec}\n"
                    f"\nRevise the specification to address ALL feedback points.\n"
                )
            elif previous_spec:
                parts.append(
                    "\n--- REVISION REQUESTED ---\n"
                    "The previous specification was rejected. Please revise it.\n"
                    f"\n--- Previous Specification ---\n{previous_spec}\n"
                )

            # Read architecture docs from disk so the LLM actually has
            # the ERS, block diagram connections, and FRD it's told to follow
            if project_root:
                import json as _json
                from pathlib import Path as _P

                _root = _P(project_root)

                # ERS -- the authoritative engineering spec
                ers_path = _root / "arch" / "ers_spec.md"
                if ers_path.exists():
                    try:
                        parts.append(
                            "\n--- ENGINEERING REQUIREMENTS SPECIFICATION (ERS) ---\n"
                            f"{ers_path.read_text()}\n"
                            "--- END ERS ---\n"
                        )
                    except OSError:
                        pass

                # Block diagram connections for this block
                bd_path = _root / ".coresmith" / "block_diagram.json"
                if bd_path.exists():
                    try:
                        bd = _json.loads(bd_path.read_text())
                        conns = [
                            c for c in bd.get("connections", [])
                            if c.get("from") == block_name or c.get("to") == block_name
                        ]
                        if conns:
                            parts.append(
                                "\n--- CONNECTION GRAPH (this block's connections) ---\n"
                                f"{_json.dumps(conns, indent=2)}\n"
                                "--- END CONNECTION GRAPH ---\n"
                            )

                        for blk in bd.get("blocks", []):
                            if blk.get("name") == block_name:
                                _bd_block = blk if isinstance(blk, dict) else {}
                                ifaces = blk.get("interfaces", {})
                                if ifaces:
                                    parts.append(
                                        "\n--- BLOCK INTERFACES (from block diagram) ---\n"
                                        f"{_json.dumps(ifaces, indent=2)}\n"
                                    )
                                # HARD per-block flop budget allocated by the
                                # block diagram (PRD chip cap split per block).
                                # Inject as an explicit HARD constraint so the
                                # uArch spec carries it verbatim and the PPA
                                # gate can enforce it -- never let the spec
                                # self-derive a larger budget for a comb cloud.
                                _ffb = blk.get("flip_flop_budget")
                                if _ffb:
                                    parts.append(
                                        "\n--- HARD FLOP BUDGET (from block diagram; "
                                        "MANDATORY) ---\n"
                                        f"flip_flop_budget = {_ffb} FF -- this is a HARD "
                                        "cap on standard-cell sequential elements for this "
                                        "block (bulk memories/FIFOs >=2 Kbit are SRAM "
                                        "macros, excluded). Your uArch spec MUST emit "
                                        f"`flip_flop_budget` <= {_ffb} and design within it; "
                                        "sequentialize iterated search rather than exceeding "
                                        "it.\n"
                                    )
                                # HARD per-block DIE-AREA budget (um^2, SRAM
                                # INCLUDED). cs_sram macros are 0 flops so the
                                # flop budget can't price them -- the area gate
                                # does, charging ~CORESMITH_SRAM_UM2_PER_BIT
                                # (~1.7) um^2 per stored bit. This is what stops
                                # an oversized full-frame / whole-bitstream
                                # buffer (GDS-intractable on sky130).
                                _ab = blk.get("area_budget_um2") or blk.get("die_area_budget_um2")
                                if _ab:
                                    parts.append(
                                        "\n--- HARD AREA BUDGET (from block diagram; "
                                        "MANDATORY, SRAM INCLUDED) ---\n"
                                        f"area_budget_um2 = {_ab} -- HARD cap on total die "
                                        "area = std cells + SRAM macros. Each cs_sram of "
                                        "WIDTH x DEPTH bits costs ~1.7 um^2/bit. Your uArch "
                                        f"spec MUST emit `area_budget_um2` <= {_ab} and keep "
                                        "logic+SRAM within it: prefer a top-row LINE BUFFER "
                                        "over a full-frame store, and STREAM the output "
                                        "rather than buffering the whole bitstream.\n"
                                    )
                                break
                    except (OSError, _json.JSONDecodeError):
                        pass

                # Inject canonical interface_contracts.json slice for this
                # block. The Interface Definition arch stage produces this
                # as the authoritative source for bit-level edge contracts
                # (handshake protocol, field positions, bootstrap policy).
                # Inlining prevents the spec author from drifting from it.
                from .contract_lookup import (
                    format_block_contracts_prompt,
                    load_block_contracts,
                )
                _contracts_view = load_block_contracts(project_root, block_name) or {}
                _contracts_fragment = format_block_contracts_prompt(
                    block_name, _contracts_view
                )
                if _contracts_fragment:
                    parts.append(_contracts_fragment)

                # FRD for testable requirements context
                frd_path = _root / "arch" / "frd_spec.md"
                if frd_path.exists():
                    try:
                        frd_text = frd_path.read_text()
                        if len(frd_text) > 8000:
                            frd_text = frd_text[:8000] + "\n... (truncated)"
                        parts.append(
                            "\n--- FUNCTIONAL REQUIREMENTS (FRD) ---\n"
                            f"{frd_text}\n"
                            "--- END FRD ---\n"
                        )
                    except OSError:
                        pass

            # D1: an EMPTY golden must not be dressed up as a golden. This block
            # used to append the section unconditionally, so a block whose
            # python_source resolved to "" got `--- Python Golden Model ---`
            # followed by an empty code fence -- indistinguishable, to the spec
            # author, from a golden that says nothing. Measured on a
            # validation run: all 8 uArch calls carried an empty golden block,
            # silently. Say what actually happened instead.
            if python_source.strip():
                parts.append(
                    f"\n--- Python Golden Model ---\n```python\n{python_source}\n```"
                )
            else:
                parts.append(
                    "\n--- Python Golden Model: NONE SUPPLIED ---\n"
                    "This block has NO reference golden model. Nothing below is "
                    "a transcription target: derive the microarchitecture from "
                    "the description, the interface contracts and the ERS above, "
                    "and state explicitly in the spec that no golden model "
                    "constrained it. Do NOT invent one and do NOT assume the "
                    "golden was omitted for brevity -- it does not exist."
                )

            user_message = "\n".join(parts)

            # Per-call system prompt: port_naming always inline, the rest
            # selected from THIS block's evidence, unselected skills manifested
            # with absolute paths.
            _block_spec = dict(_bd_block)
            _block_spec.update({
                "name": block_name,
                "description": description,
                "model_source": python_source,
            })
            system_prompt = build_system_prompt(
                block_spec=_block_spec,
                contracts=_contracts_view or None,
                block_diagram=_bd_block or None,
            )
            span.set_attribute("system_prompt_chars", len(system_prompt))

            run_name = f"Generate Uarch Spec [{block_title}]{revision_label}"
            content = await self.llm.call(
                system=system_prompt,
                prompt=user_message,
                run_name=run_name,
                resume_session_id=resume_session_id,
            )
            span.set_attribute("resume", bool(resume_session_id))

            spec_text, spec_summary = self._parse_response(content, block_name)

            return {
                "spec_text": spec_text,
                "spec_summary": spec_summary,
                "block_name": block_name,
            }

    def _parse_response(
        self, content: str, block_name: str
    ) -> tuple[str, dict]:
        """Extract the spec document and JSON summary from the LLM response."""
        spec_text = content.strip()

        # Extract JSON summary block
        spec_summary = {}
        json_match = re.search(r"```json\s*\n(.*?)```", content, re.DOTALL)
        if json_match:
            try:
                spec_summary = json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        if not spec_summary:
            spec_summary = {"block_name": block_name}

        # Normalise the feasibility verdict so the uarch->RTL gate can test it by
        # plain truthiness (mirrors backend_graph's blocking_issues contract).
        normalize_feasibility(spec_summary)

        return spec_text, spec_summary
