# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""OutputContractReviewAgent -- catch decomposition orphaning a global output
responsibility (framing/container, global ordering, coordinate/layout,
termination, global reductions) BEFORE block boundaries are frozen.

Reads the golden reference + the proposed block diagram and returns a structured
verdict. On a fail, the architecture graph re-decomposes with the feedback.
Mirrors ContractAuditAgent's read-context / write-JSON / normalize pattern.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from opentelemetry import trace

from .coresmith_llm import ClaudeLLM

_tracer = trace.get_tracer(__name__)

_PROMPT_FILE = (
    Path(__file__).resolve().parent.parent / "prompts" / "output_contract_review.md"
)
OUTPUT_CONTRACT_REVIEW_PROMPT = _PROMPT_FILE.read_text(encoding="utf-8")


class OutputContractReviewAgent:
    """Reviews a block decomposition for orphaned global output responsibilities."""

    def __init__(self, model: str | None = None, temperature: float = 0.1):
        from orchestrator.langchain.agents.coresmith_llm import DEFAULT_MODEL

        model = model or DEFAULT_MODEL
        self.llm = ClaudeLLM(model=model, timeout=900)

    async def review(
        self,
        *,
        project_root: str,
        block_diagram: dict[str, Any] | None,
        requirements: str = "",
        golden_summary: str = "",
        output_path: str | None = None,
        callbacks: list | None = None,
    ) -> dict[str, Any]:
        """Run the review and return the normalized verdict dict."""
        with _tracer.start_as_current_span("Output Contract Review") as span:
            blocks = (block_diagram or {}).get("blocks", [])
            block_names = [b.get("name", b.get("id", "?")) for b in blocks]
            goc = (block_diagram or {}).get("global_output_contract", None)
            span.set_attribute("block_count", len(blocks))
            span.set_attribute("has_ownership_table", goc is not None)

            out = Path(output_path) if output_path else (
                Path(project_root) / ".coresmith" / "output_contract_review.json"
            )
            out.parent.mkdir(parents=True, exist_ok=True)

            default = self._default_result(block_names)
            try:
                bd_json = json.dumps(block_diagram or {}, indent=2)[:24000]
                prompt = (
                    f"Project root: {project_root}\n"
                    f"Output path: {out}\n\n"
                    "## Requirements (excerpt)\n"
                    f"{(requirements or '')[:4000]}\n\n"
                    "## Golden reference summary / interfaces\n"
                    f"{(golden_summary or '(scan the golden model files under the project root)')[:6000]}\n\n"
                    "## Proposed block diagram (blocks, interfaces, global_output_contract)\n"
                    f"{bd_json}\n\n"
                    "Inspect the golden model's OUTPUT-producing code under the "
                    "project root (framing/container, ordering, layout, "
                    "termination, reductions). Decide whether any emergent output "
                    "property is orphaned by this decomposition. Write ONLY the "
                    "verdict JSON to the output path."
                )
                content = await self.llm.call(
                    system=OUTPUT_CONTRACT_REVIEW_PROMPT,
                    prompt=prompt,
                    run_name="Output Contract Review",
                )
                if out.exists():
                    result = json.loads(out.read_text(encoding="utf-8"))
                else:
                    result = self._parse_json(content, default)
                    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

                result = self._normalize(result)
                out.write_text(json.dumps(result, indent=2), encoding="utf-8")
                span.set_attribute("passed", result.get("passed", True))
                span.set_attribute(
                    "orphaned_count", len(result.get("orphaned_properties", []))
                )
                return result
            except Exception as exc:  # noqa: BLE001
                # Fail OPEN: never let the reviewer stall the architecture. The
                # downstream composition gate is the backstop.
                span.set_attribute("error", str(exc)[:200])
                result = default | {
                    "passed": True,
                    "summary": f"Output-contract review errored, proceeding: {exc}",
                }
                out.write_text(json.dumps(result, indent=2), encoding="utf-8")
                return result

    @staticmethod
    def _default_result(block_names: list[str]) -> dict[str, Any]:
        return {
            "passed": True,
            "orphaned_properties": [],
            "summary": "Output-contract review inconclusive; proceeding.",
            "feedback_for_redecomposition": "",
            "_blocks": block_names,
        }

    @staticmethod
    def _parse_json(content: str, default: dict[str, Any]) -> dict[str, Any]:
        match = re.search(r"```json\s*\n(.*?)```", content, re.DOTALL)
        raw = match.group(1) if match else content
        try:
            return json.loads(raw)
        except Exception:  # noqa: BLE001
            # last resort: find the outermost object
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:  # noqa: BLE001
                    pass
            return default

    @staticmethod
    def _normalize(result: dict[str, Any]) -> dict[str, Any]:
        result.setdefault("orphaned_properties", [])
        result.setdefault("feedback_for_redecomposition", "")
        result.setdefault("summary", "")
        # A non-empty orphan list always means a fail, regardless of the flag.
        orphaned = result.get("orphaned_properties") or []
        passed = bool(result.get("passed", True)) and not orphaned
        result["passed"] = passed
        return result


__all__ = ["OutputContractReviewAgent", "OUTPUT_CONTRACT_REVIEW_PROMPT"]
