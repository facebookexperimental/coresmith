"""ContractAuditAgent -- triage top-level DV failures for contract gaps."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from opentelemetry import trace

from .coresmith_llm import ClaudeLLM

_tracer = trace.get_tracer(__name__)

_PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts" / "contract_audit.md"
CONTRACT_AUDIT_PROMPT = _PROMPT_FILE.read_text(encoding="utf-8")


class ContractAuditAgent:
    """Audits integration/validation failures for cross-block contract issues."""

    def __init__(self, model: str | None = None, temperature: float = 0.1):
        from orchestrator.langchain.agents.coresmith_llm import DEFAULT_MODEL

        model = model or DEFAULT_MODEL
        self.llm = ClaudeLLM(model=model, timeout=900)

    async def analyze(
        self,
        *,
        stage: str,
        project_root: str,
        context_path: str,
        output_path: str,
        callbacks: list | None = None,
    ) -> dict[str, Any]:
        """Run the contract audit and return the structured JSON result."""
        with _tracer.start_as_current_span(f"Contract Audit [{stage}]") as span:
            span.set_attribute("stage", stage)
            span.set_attribute("context_path", context_path)
            span.set_attribute("output_path", output_path)

            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)

            default = self._default_result(stage, context_path)
            try:
                prompt = (
                    f"Stage: {stage}\n"
                    f"Project root: {project_root}\n"
                    f"Failure context JSON: {context_path}\n"
                    f"Output path: {output_path}\n\n"
                    "Read the failure context JSON first. Then inspect the referenced "
                    "RTL, testbench, logs, VCD/WaveKit audit, ERS/PRD, uArch specs, "
                    "and golden model files. Write only the audit JSON to the output "
                    "path. If evidence is insufficient, still write a JSON result with "
                    "category='UNKNOWN' or 'DV_PROCESS_ERROR' and a precise missing "
                    "evidence list."
                )
                content = await self.llm.call(
                    system=CONTRACT_AUDIT_PROMPT,
                    prompt=prompt,
                    run_name=f"Contract Audit [{stage}]",
                )

                if out.exists():
                    result = json.loads(out.read_text(encoding="utf-8"))
                else:
                    result = self._parse_json(content, default)
                    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

                result = self._normalize(stage, result)
                out.write_text(json.dumps(result, indent=2), encoding="utf-8")
                span.set_attribute("category", result.get("category", "UNKNOWN"))
                span.set_attribute("contract_failure", result.get("contract_failure", False))
                return result
            except Exception as exc:
                result = default | {
                    "category": "UNKNOWN",
                    "confidence": 0.0,
                    "recommended_action": "ask_human",
                    "outer_agent_summary": f"Contract audit agent failed: {exc}",
                    "human_question": (
                        "Contract audit failed. Please inspect the validation/integration "
                        "failure context and decide whether this is TB, RTL, or uArch."
                    ),
                }
                out.write_text(json.dumps(result, indent=2), encoding="utf-8")
                return result

    @staticmethod
    def _default_result(stage: str, context_path: str) -> dict[str, Any]:
        return {
            "stage": stage,
            "passed": False,
            "category": "UNKNOWN",
            "contract_failure": False,
            "local_fix_possible": False,
            "confidence": 0.0,
            "first_divergence": {
                "summary": "Contract audit did not produce a concrete first divergence.",
                "golden_observation": "",
                "rtl_observation": "",
                "vcd_signals": [],
                "log_refs": [context_path],
            },
            "missing_or_broken_contract": "",
            "affected_blocks": [],
            "recommended_action": "ask_human",
            "suggested_fix": "Inspect the top-level DV failure manually.",
            "required_uarch_patch": {"rationale": "", "sections_to_replace": []},
            "outer_agent_summary": "Contract audit was inconclusive.",
            "evidence": [],
            "human_question": "",
        }

    @staticmethod
    def _parse_json(content: str, default: dict[str, Any]) -> dict[str, Any]:
        match = re.search(r"```json\s*\n(.*?)```", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        result = dict(default)
        result["outer_agent_summary"] = content[:1000]
        return result

    @staticmethod
    def _normalize(stage: str, result: dict[str, Any]) -> dict[str, Any]:
        normalized = ContractAuditAgent._default_result(stage, "")
        normalized.update(result)
        normalized["stage"] = normalized.get("stage") or stage
        category = str(normalized.get("category", "UNKNOWN"))
        normalized["category"] = category
        if category in {
            "UARCH_INTERFACE_CONTRACT_ERROR",
            "UARCH_SPEC_ERROR",
            "ARCHITECTURE_ERROR",
        }:
            normalized["contract_failure"] = True
            normalized["local_fix_possible"] = False
            if normalized.get("recommended_action") in ("", "fix_rtl", "fix_tb"):
                normalized["recommended_action"] = "revise_uarch"
        normalized["affected_blocks"] = list(normalized.get("affected_blocks") or [])
        normalized["evidence"] = list(normalized.get("evidence") or [])
        if not isinstance(normalized.get("first_divergence"), dict):
            normalized["first_divergence"] = {
                "summary": str(normalized.get("first_divergence", "")),
                "golden_observation": "",
                "rtl_observation": "",
                "vcd_signals": [],
                "log_refs": [],
            }
        if not isinstance(normalized.get("required_uarch_patch"), dict):
            normalized["required_uarch_patch"] = {
                "rationale": str(normalized.get("required_uarch_patch", "")),
                "sections_to_replace": [],
            }
        return normalized
