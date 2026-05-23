# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Backend EDA Agent -- LLM-driven TCL script adaptation for physical design.

Each backend EDA step (synthesis, PnR, DRC, LVS, timing signoff, MPW precheck)
uses an LLM to review and adapt baseline scripts before execution. The LLM
receives the template-generated script plus design context and prior failure
logs, and returns a modified script optimized for the specific design.

This replaces the purely deterministic template approach with an intelligent
agent that can reason about failure patterns and adapt EDA tool parameters.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .coresmith_llm import DEFAULT_MODEL

logger = logging.getLogger(__name__)

_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"

_PROMPT_FILES = {
    "synthesis": _PROMPT_DIR / "backend_synthesis.md",
    "pnr": _PROMPT_DIR / "backend_pnr.md",
    "drc": _PROMPT_DIR / "backend_drc.md",
    "lvs": _PROMPT_DIR / "backend_lvs.md",
    "timing_signoff": _PROMPT_DIR / "backend_timing_signoff.md",
    "mpw_precheck": _PROMPT_DIR / "backend_mpw_precheck.md",
}

_PROMPTS: dict[str, str] = {}
for _step, _path in _PROMPT_FILES.items():
    if _path.exists():
        _PROMPTS[_step] = _path.read_text()
    else:
        _PROMPTS[_step] = ""
        logger.warning("Backend EDA prompt not found: %s", _path)


class BackendEDAAgent:
    """LLM agent for backend EDA script adaptation.

    Wraps ``ClaudeLLM`` with step-specific prompts. The agent receives
    a baseline script and design context, and returns a modified script
    optimized for the design.

    Usage::

        agent = BackendEDAAgent(step="pnr")
        modified_script = await agent.adapt_script(
            baseline_script=tcl_content,
            context={"design_name": "chip_top", ...},
        )
    """

    def __init__(
        self,
        step: str,
        model: str = DEFAULT_MODEL,
        timeout: int = 180,
    ) -> None:
        self.step = step
        self.model = model
        self.timeout = timeout
        self._prompt_template = _PROMPTS.get(step, "")
        if not self._prompt_template:
            raise ValueError(f"No prompt template for backend step: {step}")

    async def adapt_script(
        self,
        baseline_script: str,
        context: dict,
    ) -> str:
        """Call the LLM to adapt a baseline EDA script.

        Args:
            baseline_script: The template-generated script content.
            context: Design context dict with keys matching the prompt
                template placeholders.

        Returns:
            The LLM-modified script content. Falls back to the baseline
            on any LLM failure.
        """
        from orchestrator.langchain.agents.coresmith_llm import ClaudeLLM

        ctx = {**context, "baseline_script": baseline_script}
        try:
            system_prompt = self._prompt_template.format(**ctx)
        except KeyError as exc:
            logger.warning(
                "Missing prompt context key for %s: %s -- using baseline",
                self.step, exc,
            )
            return baseline_script

        llm = ClaudeLLM(model=self.model, timeout=self.timeout, disable_tools=True)

        try:
            result = await llm.call(
                system=system_prompt,
                prompt=(
                    f"Adapt the {self.step} script for this design. "
                    f"Return only the modified script content."
                ),
                run_name=f"Backend EDA [{self.step}]",
            )
            if result and len(result.strip()) > 50:
                return result.strip()
            logger.warning(
                "Backend EDA agent returned empty/short response for %s",
                self.step,
            )
            return baseline_script
        except Exception as exc:
            logger.warning(
                "Backend EDA agent failed for %s: %s -- using baseline",
                self.step, exc,
            )
            return baseline_script

    async def analyze(self, context: dict) -> dict:
        """Call the LLM for analysis steps (timing signoff, LVS, MPW precheck).

        These steps return structured JSON instead of a script.

        Args:
            context: Design context dict with keys matching the prompt
                template placeholders.

        Returns:
            Parsed JSON dict from the LLM. Falls back to a default
            result on failure.
        """
        from orchestrator.langchain.agents.coresmith_llm import ClaudeLLM

        try:
            system_prompt = self._prompt_template.format(**context)
        except KeyError as exc:
            logger.warning(
                "Missing prompt context key for %s: %s", self.step, exc,
            )
            return self._fallback_analysis()

        llm = ClaudeLLM(model=self.model, timeout=self.timeout, disable_tools=True)

        try:
            result = await llm.call(
                system=system_prompt,
                prompt=(
                    f"Analyze the {self.step} results and return your "
                    f"assessment as JSON."
                ),
                run_name=f"Backend Analysis [{self.step}]",
            )
            return self._parse_json(result)
        except Exception as exc:
            logger.warning(
                "Backend analysis agent failed for %s: %s", self.step, exc,
            )
            return self._fallback_analysis()

    def _parse_json(self, text: str) -> dict:
        """Extract JSON from LLM response, handling markdown fences."""
        import re

        cleaned = text.strip()
        m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", cleaned, re.DOTALL)
        if m:
            cleaned = m.group(1).strip()
        # Also strip leading/trailing non-JSON text
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            cleaned = cleaned[start:end]

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from %s agent", self.step)
            return self._fallback_analysis()

    def _fallback_analysis(self) -> dict:
        """Default analysis result when LLM fails."""
        if self.step == "timing_signoff":
            return {
                "timing_met": False,
                "waivable": False,
                "assessment": "LLM analysis unavailable -- manual review required.",
                "critical_paths": "",
                "recommendations": [],
                "power_assessment": "",
                "sign_off": "FAIL",
            }
        elif self.step == "lvs":
            return {
                "preprocess_verilog": False,
                "preprocess_commands": "",
                "netgen_options": "",
                "expected_benign_deltas": {"device_delta_max": 0, "net_delta_max": 0},
                "analysis": "LLM analysis unavailable -- running LVS with defaults.",
            }
        elif self.step == "mpw_precheck":
            return {
                "submission_ready": False,
                "blocking_issues": ["LLM analysis unavailable"],
                "auto_fixable": [],
                "waivable": [],
                "assessment": "LLM analysis unavailable -- manual review required.",
                "recommendations": [],
            }
        return {}
