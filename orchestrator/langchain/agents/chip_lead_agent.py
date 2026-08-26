"""In-graph chip-lead agent: resolves pipeline interrupts autonomously.

The outer-agent decision contract (CLAUDE.md "Outer-agent decision contract")
implemented as a graph-internal LLM call, so the LangGraph itself carries the
long-horizon outer loop instead of a cron-driven external agent. Behind
``CORESMITH_ENABLE_CHIP_LEAD``; see ``pipeline_graph._resolve_interrupt`` for
the fail-safe (any failure here trips back to parked interrupts).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from opentelemetry import trace

from .coresmith_llm import ClaudeLLM

_tracer = trace.get_tracer(__name__)

_PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "chip_lead.md"
CHIP_LEAD_PROMPT = _PROMPT_FILE.read_text(encoding="utf-8")


class ChipLeadAgent:
    """Decides a resume action for a parked pipeline interrupt."""

    def __init__(self, model: str | None = None):
        from .coresmith_llm import DEFAULT_MODEL

        model = (
            model
            or os.environ.get("CORESMITH_CHIP_LEAD_MODEL")
            or DEFAULT_MODEL
        )
        self.llm = ClaudeLLM(model=model, timeout=900)

    async def decide(
        self,
        *,
        payload: dict[str, Any],
        prior_decisions: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return a resume-shaped decision dict (``{"action": ..., ...}``).

        Raises on LLM failure or unparseable output; the caller trips
        chip-lead mode to parked interrupts on any exception.
        """
        itype = payload.get("type", "unknown")
        with _tracer.start_as_current_span(f"Chip Lead [{itype}]") as span:
            span.set_attribute("interrupt_type", itype)
            history = "\n".join(prior_decisions or []) or "(none)"
            prompt = (
                "## Pending interrupt payload\n"
                f"```json\n{json.dumps(payload, indent=2, default=str)[:20000]}\n```\n\n"
                f"## Your prior decisions this run (oldest first)\n{history}\n\n"
                "Decide the resume action now. Investigate the files the "
                "payload references (RTL, step logs, contract audit) before "
                "deciding. Reply with ONLY the JSON decision object."
            )
            content = await self.llm.call(
                system=CHIP_LEAD_PROMPT,
                prompt=prompt,
                run_name=f"Chip Lead [{itype}]",
            )
            decision = _parse_decision(content)
            span.set_attribute("action", decision.get("action", ""))
            return decision


def _parse_decision(content: str) -> dict[str, Any]:
    """Extract the first JSON object carrying an ``action`` key."""
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    candidates = fenced + re.findall(r"\{.*\}", content, re.DOTALL)
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("action"):
            return obj
    raise ValueError(
        f"chip-lead output had no JSON decision object: {content[:400]!r}"
    )
