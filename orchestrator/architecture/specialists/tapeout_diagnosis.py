# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Tapeout Diagnosis Agent.

LLM-based triage of DRC, LVS, and precheck failures in the tapeout
pipeline.  Reads failure artifacts, classifies the root cause, and
decides whether to auto-retry with adjusted PnR parameters, continue
past a benign issue, or escalate to the outer agent.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


_PROMPT_FILE = (
    Path(__file__).resolve().parents[2]
    / "langchain"
    / "prompts"
    / "tapeout_diagnosis.md"
)
SYSTEM_PROMPT = _PROMPT_FILE.read_text()


async def diagnose_tapeout_failure(
    phase: str,
    attempt: int,
    max_attempts: int,
    error_summary: str,
    wrapper_drc_result: dict | None = None,
    wrapper_lvs_result: dict | None = None,
    precheck_result: dict | None = None,
    pnr_params: dict | None = None,
    previous_diagnosis: dict | None = None,
    project_root: str = ".",
) -> dict:
    """Diagnose a tapeout failure via LLM.

    Returns a structured dict with keys:
        category, diagnosis, confidence, action, suggested_fix, pnr_overrides
    """
    from opentelemetry import trace as _trace

    tracer = _trace.get_tracer("coresmith.tapeout.diagnosis")

    with tracer.start_as_current_span("diagnose_tapeout_failure") as span:
        span.set_attribute("phase", phase)
        span.set_attribute("attempt", attempt)

        root = Path(project_root)

        drc_context = _format_drc(wrapper_drc_result, root)
        lvs_context = _format_lvs(wrapper_lvs_result, root)
        precheck_context = _format_precheck(precheck_result)

        prompt = SYSTEM_PROMPT.format(
            phase=phase,
            attempt=attempt,
            max_attempts=max_attempts,
            error_summary=error_summary[:2000],
            drc_context=drc_context,
            lvs_context=lvs_context,
            precheck_context=precheck_context,
            pnr_params=json.dumps(pnr_params or {}, indent=2),
            previous_diagnosis=(
                json.dumps(previous_diagnosis, indent=2)
                if previous_diagnosis else "None (first attempt)"
            ),
        )

        user_message = (
            "Diagnose the tapeout failure described above. "
            "Return ONLY the JSON object."
        )

        from orchestrator.langchain.agents.coresmith_llm import DEFAULT_MODEL, ClaudeLLM

        llm = ClaudeLLM(model=DEFAULT_MODEL, timeout=120)

        try:
            content = await llm.call(
                system=prompt,
                prompt=user_message,
                run_name="diagnose_tapeout_failure",
            )
            result = _parse_diagnosis(content)
            span.set_attribute("category", result.get("category", ""))
            span.set_attribute("action", result.get("action", ""))
            span.set_attribute("confidence", result.get("confidence", 0))
            return result

        except Exception as exc:
            span.set_attribute("error", str(exc))
            return _fallback_diagnosis(phase, error_summary, str(exc))


# ---------------------------------------------------------------------------
# Context formatters
# ---------------------------------------------------------------------------


def _format_drc(result: dict | None, root: Path) -> str:
    if not result:
        return "No DRC result available."

    lines = [
        f"Clean: {result.get('clean', False)}",
        f"Violation count: {result.get('violation_count', '?')}",
    ]

    if result.get("error"):
        lines.append(f"Error: {result['error']}")

    violations = result.get("violations", [])
    if violations:
        lines.append("Violations (first 20):")
        for v in violations[:20]:
            lines.append(f"  - {v}")

    report_path = result.get("report_path") or ""
    if not report_path:
        for candidate in [
            root / "openframe_submission" / "pnr" / "magic_drc.rpt",
            root / "openframe_submission" / "precheck_magic_drc.rpt",
        ]:
            if candidate.exists():
                report_path = str(candidate)
                break

    if report_path and Path(report_path).exists():
        try:
            text = Path(report_path).read_text()[:3000]
            lines.append(f"\nDRC report excerpt ({report_path}):")
            lines.append(text)
        except OSError:
            pass

    return "\n".join(lines)


def _format_lvs(result: dict | None, root: Path) -> str:
    if not result:
        return "No LVS result available."

    lines = [
        f"Match: {result.get('match', False)}",
        f"Device delta: {result.get('device_delta', '?')}",
        f"Net delta: {result.get('net_delta', '?')}",
    ]

    if result.get("error"):
        lines.append(f"Error: {result['error']}")

    report_path = result.get("report_path") or ""
    if not report_path:
        for candidate in root.glob("**/lvs_*.rpt"):
            report_path = str(candidate)
            break

    if report_path and Path(report_path).exists():
        try:
            text = Path(report_path).read_text()
            final_lines = text.strip().split("\n")[-30:]
            lines.append(f"\nLVS report (last 30 lines from {report_path}):")
            lines.extend(final_lines)
        except OSError:
            pass

    return "\n".join(lines)


def _format_precheck(result: dict | None) -> str:
    if not result:
        return "No precheck result available."

    lines = [f"Overall pass: {result.get('pass', False)}"]

    checks = result.get("checks", {})
    for name, check_result in checks.items():
        if isinstance(check_result, dict):
            passed = check_result.get("pass", False)
            lines.append(f"  {name}: {'PASS' if passed else 'FAIL'}")
            if not passed and check_result.get("errors"):
                for err in check_result["errors"][:5]:
                    lines.append(f"    - {err}")
        else:
            lines.append(f"  {name}: {'PASS' if check_result else 'FAIL'}")

    errors = result.get("errors", [])
    if errors:
        lines.append("Errors:")
        for e in errors[:10]:
            lines.append(f"  - {e}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------


def _parse_diagnosis(content: str) -> dict:
    """Extract the JSON diagnosis from the LLM response."""
    content = content.strip()

    m = re.search(r"```(?:json)?\s*\n(.*?)```", content, re.DOTALL)
    if m:
        content = m.group(1).strip()

    if content.startswith("{"):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

    m = re.search(r"\{[^{}]*\"category\"[^{}]*\}", content, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return _fallback_diagnosis("unknown", content, "Failed to parse LLM JSON")


def _fallback_diagnosis(phase: str, error: str, parse_error: str) -> dict:
    """Deterministic fallback when LLM call or parsing fails."""
    return {
        "category": f"{phase.upper()}_FAILURE" if phase else "UNKNOWN",
        "diagnosis": f"LLM diagnosis unavailable ({parse_error}). Raw error: {error[:500]}",
        "confidence": 0.2,
        "action": "escalate",
        "suggested_fix": "Manual inspection required.",
        "pnr_overrides": {},
    }
