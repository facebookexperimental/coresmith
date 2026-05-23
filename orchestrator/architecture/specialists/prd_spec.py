# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
PRD (Product Requirements Document) Specialist.

Generates critical sizing questions for the SoC architect, then drafts
a structured PRD document based on the user's answers.  The PRD captures
"what functionality is needed" and becomes the first input in the document
hierarchy:  PRD -> SAD -> FRD -> Block Diagram -> ... -> ERS.

Flow inside the architecture graph:
    START -> Gather Requirements (LLM) -> Escalate PRD (interrupt)
          -> [user answers questions] -> Gather Requirements (LLM, 2nd pass)
          -> System Architecture -> Functional Requirements -> Block Diagram -> ...

The specialist runs in two modes:
  1. **Question mode** (no prior answers): generates a list of critical
     sizing questions covering technology, speed/feeds, area, power, and
     dataflow.
  2. **Draft mode** (answers provided): consumes the user's answers and
     produces the full PRD document as structured JSON.
"""

from __future__ import annotations

from typing import Any

from pathlib import Path


# ---------------------------------------------------------------------------
# System prompt -- loaded from external .md file
# ---------------------------------------------------------------------------

_PROMPT_FILE = Path(__file__).resolve().parents[2] / "langchain" / "prompts" / "prd_spec.md"
SYSTEM_PROMPT = _PROMPT_FILE.read_text()


# ---------------------------------------------------------------------------
# Prompt context helpers
# ---------------------------------------------------------------------------

def _build_answers_context(
    user_answers: dict[str, str] | None,
    previous_questions: list[dict] | None = None,
) -> str:
    """Format architect answers without dropping unmatched manual context."""
    if not user_answers:
        return ""

    lines = ["USER ANSWERS TO SIZING QUESTIONS:"]
    seen: set[str] = set()

    if previous_questions:
        for q in previous_questions:
            qid = str(q.get("id", "") or "")
            if not qid:
                continue
            seen.add(qid)
            answer = user_answers.get(qid, "(not answered)")
            lines.append(f"  {qid}: {q.get('question', '')}")
            lines.append(f"    Answer: {answer}")

    extras = [(str(qid), answer) for qid, answer in user_answers.items()
              if str(qid) not in seen]
    if extras:
        if previous_questions:
            lines.append("")
            lines.append("ADDITIONAL ARCHITECT ANSWERS AND REVIEW FEEDBACK:")
        for qid, answer in extras:
            lines.append(f"  {qid}: {answer}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def gather_prd(
    requirements: str,
    pdk_summary: str,
    target_clock_mhz: float,
    user_answers: dict[str, str] | None = None,
    previous_questions: list[dict] | None = None,
    project_root: str = ".",
) -> dict[str, Any]:
    """Generate PRD questions or draft the PRD document.

    Args:
        requirements: High-level requirements text from the user.
        pdk_summary: Available PDK technologies summary.
        target_clock_mhz: Initial target clock frequency.
        user_answers: Dict mapping question IDs to user answers.
            If None, generates questions (Phase 1).
            If provided, drafts the PRD (Phase 2).
        previous_questions: The questions that were asked (for Phase 2
            context so the LLM knows what was asked).
        project_root: Directory where PRD collateral should be written.

    Returns:
        Phase 1: {"questions": [...], "phase": "questions"}
        Phase 2: {"prd": {...}, "phase": "prd_complete"}
    """
    from opentelemetry import trace as _trace

    tracer = _trace.get_tracer("coresmith.architecture.prd_spec")

    with tracer.start_as_current_span("gather_prd") as span:
        span.set_attribute("has_answers", user_answers is not None)
        span.set_attribute("target_clock_mhz", target_clock_mhz)

        # Build context sections
        pdk_context = pdk_summary if pdk_summary else "No PDK information available."

        answers_context = _build_answers_context(user_answers, previous_questions)

        # Build the system prompt with template variables filled in
        system_prompt = SYSTEM_PROMPT.format(
            pdk_context=pdk_context,
            answers_context=answers_context,
        )

        target_path = Path(project_root) / ".coresmith" / "prd_spec.json"
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Build user message
        if user_answers:
            user_message = (
                f"The architect has answered the sizing questions.  "
                f"Write the full Product Requirements Document.\n\n"
                f"Original requirements:\n{requirements}\n\n"
                f"Target clock: {target_clock_mhz} MHz\n\n"
                f"IMPORTANT: Write the complete PRD JSON to: {target_path}\n"
                f"After writing, respond with only the file path confirmation."
            )
        else:
            user_message = (
                f"Generate the critical sizing questions for this SoC.\n\n"
                f"Requirements:\n{requirements}\n\n"
                f"Target clock: {target_clock_mhz} MHz\n\n"
                f"Ask every question needed to write the PRD.  "
                f"Cover all five categories: technology, speed_and_feeds, "
                f"area, power, dataflow."
            )

        from orchestrator.langchain.agents.coresmith_llm import DEFAULT_MODEL, ClaudeLLM

        llm = ClaudeLLM(model=DEFAULT_MODEL, timeout=1200)

        try:
            content = await llm.call(
                system=system_prompt,
                prompt=user_message,
                run_name="gather_prd",
            )

            if user_answers:
                from orchestrator.utils import read_back_json
                prd_default: dict[str, Any] = {"questions": [], "phase": "prd_complete"}
                disk_result, disk_ok = read_back_json(
                    target_path, content, prd_default, context="prd_spec"
                )
                result = disk_result if disk_ok else _parse_response(content)
            else:
                result = _parse_response(content)

            if user_answers:
                span.set_attribute("phase", "prd_complete")
                span.set_attribute("has_prd", "prd" in result)
            else:
                span.set_attribute("phase", "questions")
                span.set_attribute("question_count",
                                   len(result.get("questions", [])))

            return result

        except Exception as e:
            span.set_attribute("error", str(e))
            return {
                "questions": [{
                    "id": "error",
                    "category": "technology",
                    "question": f"PRD generation failed: {e}. "
                                "Please review requirements or retry.",
                    "context": str(e),
                    "options": [],
                    "required": True,
                }],
                "phase": "questions",
            }


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _parse_response(content: str) -> dict[str, Any]:
    """Extract structured JSON from LLM response."""
    from orchestrator.utils import parse_llm_json

    default: dict[str, Any] = {"questions": [], "phase": "questions"}
    result, ok = parse_llm_json(content, default, context="prd_spec")
    if not ok:
        result["questions"] = [{
            "id": "parse_error",
            "category": "technology",
            "question": "Could not parse PRD response. Please retry.",
            "context": content[:1000],
            "options": [],
            "required": True,
        }]
    return result
