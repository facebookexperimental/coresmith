# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
FRD (Functional Requirements Document) Specialist.

Takes the PRD and SAD to produce detailed, quantitative, measurable
functional requirements with acceptance criteria.

The FRD answers: "How well should the functionality work?"

Document hierarchy:  PRD -> SAD -> FRD -> Block Diagram -> ... -> ERS

The LLM produces Markdown directly (no JSON parsing required).
"""

from __future__ import annotations

import json
from typing import Any

from pathlib import Path


_PROMPT_FILE = Path(__file__).resolve().parents[2] / "langchain" / "prompts" / "frd_spec.md"
SYSTEM_PROMPT = _PROMPT_FILE.read_text()


async def generate_frd(
    prd_spec: dict,
    sad_spec: dict,
    requirements: str,
    project_root: str = ".",
    constraint_feedback: str | None = None,
) -> dict[str, Any]:
    """Generate the Functional Requirements Document from PRD + SAD.

    Args:
        prd_spec: Full PRD document.
        sad_spec: Full SAD document (contains ``sad_text`` markdown key).
        requirements: Original high-level requirements text.
        project_root: Directory where FRD collateral should be written.
        constraint_feedback: Optional constraint-repair feedback. When set,
            the FRD is REGENERATED to fix the cited violation(s) -- used by the
            architecture graph's Doc Fix path for ``auto_fixable`` violations
            whose ``source_doc`` is the FRD. Appended to the generation request.

    Returns:
        {"frd_text": "<markdown>", "phase": "frd_complete"}
    """
    from opentelemetry import trace as _trace

    tracer = _trace.get_tracer("coresmith.architecture.frd_spec")

    with tracer.start_as_current_span("generate_frd") as span:
        prd_doc = prd_spec.get("prd", {}) if prd_spec else {}
        span.set_attribute("has_prd", bool(prd_doc))

        if sad_spec and isinstance(sad_spec.get("sad_text"), str):
            sad_context = sad_spec["sad_text"]
        elif sad_spec and sad_spec.get("sad"):
            sad_context = json.dumps(sad_spec["sad"], indent=2)
        else:
            sad_context = "No SAD available."
        span.set_attribute("has_sad", sad_context != "No SAD available.")

        prd_context = json.dumps(prd_doc, indent=2) if prd_doc else "No PRD available."

        from orchestrator.architecture.specialists.sad_spec import _build_shuttle_context

        system_prompt = SYSTEM_PROMPT.format(
            prd_context=prd_context,
            sad_context=sad_context,
            shuttle_context=_build_shuttle_context(),
        )

        target_path = Path(project_root) / "arch" / "frd_spec.md"
        target_path.parent.mkdir(parents=True, exist_ok=True)

        user_message = (
            f"Produce the Functional Requirements Document for this design.\n\n"
            f"Original requirements:\n{requirements}\n\n"
            f"The PRD and SAD have been provided in the system prompt.\n\n"
            f"IMPORTANT: Write the complete FRD document to: {target_path}\n"
            f"After writing, respond with only the file path confirmation."
        )

        if constraint_feedback:
            user_message += (
                "\n\nCONSTRAINT REPAIR FEEDBACK (a prior FRD failed a "
                "constraint check -- REGENERATE the document with these "
                "corrections applied; keep everything else consistent with "
                "the PRD/SAD):\n"
                f"{constraint_feedback}"
            )

        from orchestrator.langchain.agents.coresmith_llm import (
            DEFAULT_MODEL, ClaudeLLM, arch_reasoning_effort)

        import os as _os
        _arch_to = int(_os.environ.get("CORESMITH_ARCH_LLM_TIMEOUT_S", "1200") or 1200)
        # FRD carries the FUNC vectors + perf budgets the gates enforce ->
        # higher reasoning tier (codex-only; no-op on other providers).
        llm = ClaudeLLM(model=DEFAULT_MODEL, timeout=_arch_to,
                        reasoning_effort=arch_reasoning_effort())

        try:
            content = await llm.call(
                system=system_prompt,
                prompt=user_message,
                run_name="generate_frd",
            )
            from orchestrator.utils import read_back_text
            frd_text = read_back_text(target_path, content.strip())
            # dv-hardening-15 freeze check: the FRD must define a
            # MISSION-SCALE acceptance test (see prompt section). This is the
            # armC lesson made deterministic -- a missing/degenerate
            # acceptance test silently redefines the whole mission (a 30dB
            # floor "passed" on one macroblock while real frames were 21dB).
            # Warning-grade (logged + span attr), never blocking: the human
            # sees it at final review; Full Model DV honest-skips without the
            # artifact.
            _accept_ok = "Mission-Scale Acceptance Test" in (frd_text or "")
            span.set_attribute("acceptance_test_declared", _accept_ok)
            if not _accept_ok:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "FRD has NO 'Mission-Scale Acceptance Test' section -- "
                    "the mission is unverified by construction (Full Model "
                    "DV and RTL Acceptance DV will honest-skip). Escalate "
                    "at PRD/final review."
                )
            span.set_attribute("phase", "frd_complete")
            return {"frd_text": frd_text, "phase": "frd_complete",
                    "acceptance_test_declared": _accept_ok}

        except Exception as e:
            span.set_attribute("error", str(e))
            return {
                "frd_text": f"# Functional Requirements Document (generation failed)\n\nFRD generation failed: {e}\n",
                "phase": "frd_complete",
            }
