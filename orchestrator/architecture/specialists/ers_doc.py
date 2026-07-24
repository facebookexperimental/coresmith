# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
ERS (Engineering Requirements Specification) Document Generator.

Synthesizes the final ERS from all upstream architecture artifacts:
PRD, SAD, FRD, block diagram, memory map, clock tree, register spec.

The ERS answers: "What is needed to enable the functionality?"

Document hierarchy:  PRD -> SAD -> FRD -> Block Diagram -> ... -> ERS
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from orchestrator.langchain.prompts.skills import load_skills as _load_skills

_PROMPT_FILE = Path(__file__).resolve().parents[2] / "langchain" / "prompts" / "ers_doc.md"
SYSTEM_PROMPT = _PROMPT_FILE.read_text()

# The ERS is where memory implementation (SRAM macro vs. flip-flop array) is
# fixed for the whole pipeline — downstream uArch/RTL obey it. Inject the same
# SRAM-vs-flops skill the uArch agent gets so the ERS does not blanket-mandate
# flip-flop scratchpads (which synthesize into huge, slow register files).
_SKILLS_TEXT = _load_skills("memory_macro_vs_flops")


async def generate_ers_doc(
    prd_spec: dict | None,
    sad_spec: dict | None,
    frd_spec: dict | None,
    block_diagram: dict | None,
    memory_map: dict | None,
    clock_tree: dict | None,
    register_spec: dict | None,
    project_root: str = ".",
) -> dict[str, Any]:
    """Generate the final ERS by synthesizing all architecture artifacts.

    Args:
        prd_spec: Full PRD document.
        sad_spec: Full SAD document.
        frd_spec: Full FRD document.
        block_diagram: Block diagram result (blocks, connections).
        memory_map: Memory map result.
        clock_tree: Clock tree result.
        project_root: Directory where ERS collateral should be written.
        register_spec: Register spec result.

    Returns:
        {"ers": {...}, "phase": "ers_complete"}
    """
    from opentelemetry import trace as _trace

    tracer = _trace.get_tracer("coresmith.architecture.ers_doc")

    with tracer.start_as_current_span("generate_ers_doc") as span:
        def _ctx(data, key=None, text_key=None):
            if not data:
                return "Not available."
            if text_key and isinstance(data.get(text_key), str):
                return data[text_key]
            doc = data.get(key, data) if key else data
            return json.dumps(doc, indent=2)

        prd_context = _ctx(prd_spec, "prd")
        sad_context = _ctx(sad_spec, "sad", text_key="sad_text")
        frd_context = _ctx(frd_spec, "frd", text_key="frd_text")
        bd_context = _ctx(block_diagram)
        mm_context = _ctx(memory_map)
        ct_context = _ctx(clock_tree)
        rs_context = _ctx(register_spec)

        golden_lines = []
        if block_diagram:
            for blk in block_diagram.get("blocks", []):
                src = blk.get("python_source", "")
                name = blk.get("name", "unknown")
                if src and src.strip():
                    golden_lines.append(
                        f"  - {name}: golden model at `{src}`"
                    )
                else:
                    golden_lines.append(
                        f"  - {name}: NO golden model (write algorithm_pseudocode)"
                    )
        golden_model_context = "\n".join(golden_lines) if golden_lines else "None available."

        span.set_attribute("has_prd", prd_spec is not None)
        span.set_attribute("has_sad", sad_spec is not None)
        span.set_attribute("has_frd", frd_spec is not None)
        span.set_attribute("has_block_diagram", block_diagram is not None)

        system_prompt = SYSTEM_PROMPT.format(
            prd_context=prd_context,
            sad_context=sad_context,
            frd_context=frd_context,
            block_diagram_context=bd_context,
            memory_map_context=mm_context,
            clock_tree_context=ct_context,
            register_spec_context=rs_context,
            golden_model_context=golden_model_context,
        )
        # Append skills AFTER .format() so skill-body braces are never treated
        # as format placeholders.
        if _SKILLS_TEXT:
            system_prompt = system_prompt + "\n\n" + _SKILLS_TEXT

        target_path = Path(project_root) / ".coresmith" / "ers_spec.json"
        target_path.parent.mkdir(parents=True, exist_ok=True)

        user_message = (
            "Produce the Engineering Requirements Specification (ERS) "
            "by synthesizing all the upstream architecture documents "
            "provided in the system prompt.\n\n"
            f"IMPORTANT: Write the complete ERS JSON to: {target_path}\n"
            "After writing, respond with only the file path confirmation."
        )

        import os as _os

        from orchestrator.langchain.agents.coresmith_llm import DEFAULT_MODEL, ClaudeLLM
        _arch_to = int(_os.environ.get("CORESMITH_ARCH_LLM_TIMEOUT_S", "1200") or 1200)
        llm = ClaudeLLM(model=DEFAULT_MODEL, timeout=_arch_to)

        from orchestrator.utils import read_back_json
        ers_default: dict[str, Any] = {
            "ers": {
                "title": "Engineering Requirements Specification",
                "summary": "",
                "functional_requirements": [],
                "per_block_requirements": [],
                "constraints": [],
                "verification_requirements": [],
                "open_items": [],
            },
            "phase": "ers_complete",
        }

        async def _invoke_generator(prompt: str, run_name: str) -> dict[str, Any]:
            """Run the ERS generator once (the LLM writes the JSON to disk); read
            it back, falling back to parsing the response text."""
            content = await llm.call(
                system=system_prompt, prompt=prompt, run_name=run_name
            )
            disk_result, disk_ok = read_back_json(
                target_path, content, ers_default, context="ers_doc"
            )
            return disk_result if disk_ok else _parse_response(content)

        try:
            result = await _invoke_generator(user_message, "generate_ers_doc")
            # param-schema-1: the ERS doc-validation path. Schema-check + repair
            # the typed `parameters` block (auto-compute boundary_values, drop
            # malformed entries -> ers.open_items) and re-persist the normalized
            # JSON so disk == the returned dict. A doc with no parameters key is
            # left untouched here (legacy) -- the presence BACKSTOP below owns the
            # wholesale-omission case. Never raises -> never blocks generation.
            try:
                from orchestrator.architecture import param_schema as _psch
                from orchestrator.utils import atomic_write as _atomic_write

                result, _perrs = _psch.validate_and_normalize_ers_parameters(result)
                if _perrs:
                    span.set_attribute("ers_param_schema_errors", len(_perrs))
                # [rung3r2-fixes-3] parameters-block presence BACKSTOP. The
                # [param-schema-1] mandate is prompt-only + normalize-if-present;
                # when the LLM omits the `parameters` key ENTIRELY the doc would
                # silently pass as legacy (maxgeo no-ops, manifest strictness
                # never auto-flips). This makes the mandate deterministic: one
                # bounded retry with explicit feedback, then derive-or-record.
                if _params_backstop_enabled():
                    _frd_text = (
                        frd_spec.get("frd_text")
                        if isinstance(frd_spec, dict)
                        and isinstance(frd_spec.get("frd_text"), str)
                        else None
                    )
                    result = await _enforce_parameters_backstop(
                        result,
                        invoke=_invoke_generator,
                        base_user_message=user_message,
                        upstream_sources=[
                            prd_spec, sad_spec, frd_spec, block_diagram,
                            memory_map, register_spec,
                        ],
                        frd_text=_frd_text,
                        span=span,
                    )
                _atomic_write(
                    target_path, json.dumps(result, indent=2, default=str)
                )
            except Exception as _pe:  # noqa: BLE001
                span.set_attribute("ers_param_schema_exception", str(_pe))
            span.set_attribute("phase", "ers_complete")
            return result

        except Exception as e:
            span.set_attribute("error", str(e))
            return {
                "ers": {
                    "title": "ERS (generation failed)",
                    "summary": f"ERS generation failed: {e}",
                    "functional_requirements": [],
                    "per_block_requirements": [],
                    "constraints": [],
                    "verification_requirements": [],
                    "open_items": [f"ERS generation error: {e}"],
                },
                "phase": "ers_complete",
            }


def _parse_response(content: str) -> dict[str, Any]:
    """Extract structured JSON from LLM response."""
    from orchestrator.utils import parse_llm_json

    default: dict[str, Any] = {
        "ers": {
            "title": "Engineering Requirements Specification",
            "summary": "",
            "functional_requirements": [],
            "per_block_requirements": [],
            "constraints": [],
            "verification_requirements": [],
            "open_items": [],
        },
        "phase": "ers_complete",
    }
    result, _ok = parse_llm_json(content, default, context="ers_doc")
    return result


# ---------------------------------------------------------------------------
# [rung3r2-fixes-3] ERS `parameters`-block presence backstop
# ---------------------------------------------------------------------------
#
# Defense-in-depth for the [param-schema-1] mandate. The mandate is enforced
# by-prompt (ers_doc.md) + normalize-if-present (validate_and_normalize_ers_
# parameters). Both are silent when the LLM omits the `parameters` KEY entirely:
# the doc passes as "legacy", maxgeo's deterministic half no-ops, and mem-manifest
# strictness never auto-flips. This backstop makes wholesale omission a
# deterministic, LOUD, recovered event (not yet observed live -- the first live
# schema run emitted a 28-param block -- but an LLM genuinely could omit it).
#
# NOTE (nesting): every read here uses the `ers` sub-key nesting via the engine's
# own dict shape ({"ers": {...}}), never a raw top-level `get`. A present but
# EMPTY `parameters: []` that the LLM affirmed (with its no-dims statement) is a
# valid first-pass outcome -> NO retry. Only wholesale KEY ABSENCE triggers it.

_PARAMS_RETRY_FEEDBACK = (
    "\n\n"
    "=== REVISION REQUIRED (parameters block) ===\n"
    "Your previous ERS JSON OMITTED the `parameters` key entirely. The typed "
    "`parameters` block is MANDATORY. Re-emit the COMPLETE ERS JSON to the same "
    "file path, and this time include the top-level `parameters` array under "
    "`ers`, exactly per the schema in the system prompt:\n"
    "  - emit `parameters: [ {\"name\": ..., \"role\": "
    "\"dimension|mode|range\", \"max\": <n>}, ... ]` enumerating every "
    "design-parameter axis (the extents that drive RTL index/address/counter "
    "widths and the mode selects), OR\n"
    "  - emit `parameters: []` ACCOMPANIED by an explicit "
    "\"no dimensional parameters\" statement in the summary or an open item, "
    "ONLY if the design genuinely has none.\n"
    "Do NOT omit the key again."
)


def _params_backstop_enabled() -> bool:
    """CORESMITH_ERS_PARAMS_BACKSTOP (default ON): enforce the `parameters`-block
    presence backstop. ``=0`` restores today's behavior (prompt-only mandate +
    normalize-if-present; a wholesale-omitted key is left absent)."""
    import os
    return (os.environ.get("CORESMITH_ERS_PARAMS_BACKSTOP", "1") or "1") != "0"


def _params_key_absent(result: Any) -> bool:
    """True when the ERS result dict structurally lacks the `parameters` key
    (read via the `ers` sub-key nesting). An affirmed `parameters: []` is
    PRESENT -> False."""
    ers = result.get("ers") if isinstance(result, dict) else None
    return isinstance(ers, dict) and "parameters" not in ers


async def _enforce_parameters_backstop(
    result: dict[str, Any],
    *,
    invoke,
    base_user_message: str,
    upstream_sources: list,
    frd_text,
    span,
) -> dict[str, Any]:
    """Deterministic backstop for a wholesale-omitted `parameters` key.

    Only fires when the key is ABSENT (an affirmed `[]` returns unchanged). One
    bounded retry with explicit feedback; if the retry supplies the block it is
    normalized + returned. If the retry still omits it: (a) DERIVE dims from the
    machine-readable sources and write a `parameters` block tagged
    ``derived: True`` (maxgeo + manifest strictness go active), else (b) write an
    empty block tagged as backstop-empty + a LOUD open_item (gates stay legacy).
    Never raises."""
    from orchestrator.architecture import param_schema as _psch

    if not _params_key_absent(result):
        return result  # present (incl. affirmed []-with-statement): no backstop
    span.set_attribute("ers_params_backstop_triggered", True)

    # (1) one bounded retry with explicit feedback
    retried = None
    try:
        retried = await invoke(
            base_user_message + _PARAMS_RETRY_FEEDBACK,
            "generate_ers_doc_params_retry",
        )
        retried, _rerrs = _psch.validate_and_normalize_ers_parameters(retried)
        if _rerrs:
            span.set_attribute("ers_params_backstop_retry_schema_errors", len(_rerrs))
    except Exception as _re:  # noqa: BLE001
        span.set_attribute("ers_params_backstop_retry_exception", str(_re))
        retried = None

    if retried is not None and not _params_key_absent(retried):
        span.set_attribute("ers_params_backstop_retry_succeeded", True)
        return retried
    span.set_attribute("ers_params_backstop_retry_failed", True)

    # base to mutate: prefer the (structurally-valid) retried doc, else the first
    base = (
        retried
        if isinstance(retried, dict) and isinstance(retried.get("ers"), dict)
        else result
    )
    ers = base["ers"]
    open_items = ers.get("open_items")
    if not isinstance(open_items, list):
        open_items = []

    # (2a) deterministic derivation fallback (harvest the same sources the legacy
    # maxgeo fallback reads + FRD FUNC-vector geometry)
    derived: list[dict] = []
    try:
        derived = _psch.derive_parameters(list(upstream_sources) + [base], frd_text)
    except Exception as _de:  # noqa: BLE001
        span.set_attribute("ers_params_backstop_derive_exception", str(_de))
        derived = []

    if derived:
        ers["parameters"] = derived
        span.set_attribute("ers_params_backstop_derived", len(derived))
        names = ", ".join(str(p.get("name")) for p in derived)
        open_items.append(
            "ERS parameters backstop [rung3r2-fixes-3]: the generator OMITTED "
            "the mandatory `parameters` block on both the initial attempt and "
            f"the retry; DERIVED {len(derived)} dimensional parameter(s) "
            f"[{names}] from the machine-readable PRD/constraints/block-diagram/"
            "FRD sources and tagged them `derived: true`. VERIFY these extents -- "
            "max-geometry + memory-manifest strictness are now active off the "
            "derived block."
        )
    else:
        # (2b) nothing derivable: empty block + backstop marker + LOUD record.
        ers["parameters"] = []
        ers[_psch.BACKSTOP_MARKER_KEY] = _psch.BACKSTOP_EMPTY_MARKER
        span.set_attribute("ers_params_backstop_empty", True)
        open_items.append(
            "ERS parameters backstop [rung3r2-fixes-3]: generator failed to emit "
            "parameters after retry; no dims derivable -- maxgeo/manifest "
            "strictness running in legacy mode."
        )
    ers["open_items"] = open_items
    return base
