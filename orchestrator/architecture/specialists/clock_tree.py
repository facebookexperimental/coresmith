# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Clock Tree Specialist -- designs clock domains and reset strategy via LLM.

Uses ClaudeLLM for LLM inference. Analyzes the block diagram to
determine whether multiple clock domains are needed, designs CDC crossings,
and specifies the reset strategy.

The prompt encodes socmate conventions (single-domain baseline, synchronous
active-low reset, 2-FF synchronizer) while allowing the LLM to propose
multi-domain architectures when the block diagram warrants it.
"""

from __future__ import annotations

import json
from typing import Any

from pathlib import Path

_PROMPT_FILE = Path(__file__).resolve().parents[2] / "langchain" / "prompts" / "clock_tree.md"
SYSTEM_PROMPT = _PROMPT_FILE.read_text()


async def analyze_clock_tree(
    block_diagram: dict,
    target_clock_mhz: float = 50.0,
    requirements: str = "",
    project_root: str = ".",
) -> dict[str, Any]:
    """Design the clock domain structure for the ASIC.

    Args:
        block_diagram: Block diagram from analyze_block_diagram().
        target_clock_mhz: Target clock frequency in MHz.
        requirements: High-level system requirements for context.
        project_root: Project root path (unused, kept for interface compat).

    Returns:
        Dict with keys: result (the clock tree), questions.
    """
    from opentelemetry import trace as _trace

    tracer = _trace.get_tracer("socmate.architecture.clock_tree")

    with tracer.start_as_current_span("analyze_clock_tree") as span:
        blocks = block_diagram.get("blocks", [])
        span.set_attribute("target_clock_mhz", target_clock_mhz)
        span.set_attribute("block_count", len(blocks))

        parts = [
            "Design the clock tree for the following ASIC block diagram.",
            f"\nTarget clock frequency: {target_clock_mhz} MHz",
            f"Clock period: {1000.0 / target_clock_mhz:.2f} ns",
        ]
        if requirements:
            parts.append(f"\nSystem requirements: {requirements}")

        parts.append(
            f"\n--- BLOCK DIAGRAM ---\n{json.dumps(block_diagram, indent=2)}"
        )

        from pathlib import Path as _P
        _root = _P(project_root)
        for doc_name, doc_label in [
            ("sad_spec.md", "SYSTEM ARCHITECTURE DOCUMENT (SAD)"),
            ("frd_spec.md", "FUNCTIONAL REQUIREMENTS DOCUMENT (FRD)"),
        ]:
            doc_path = _root / "arch" / doc_name
            if doc_path.exists():
                try:
                    parts.append(f"\n--- {doc_label} ---\n{doc_path.read_text()}")
                except OSError:
                    pass

        target_path = _P(project_root) / ".socmate" / "clock_tree.json" if project_root else _P.cwd() / ".socmate" / "clock_tree.json"
        target_path.parent.mkdir(parents=True, exist_ok=True)

        parts.append(
            f"\nIMPORTANT: Write the clock tree JSON to: {target_path}\n"
            "After writing, respond with only the file path confirmation."
        )

        user_message = "\n".join(parts)

        from orchestrator.langchain.agents.socmate_llm import DEFAULT_MODEL, ClaudeLLM

        llm = ClaudeLLM(model=DEFAULT_MODEL, timeout=1200)

        try:
            content = await llm.call(
                system=SYSTEM_PROMPT,
                prompt=user_message,
                run_name="clock_tree",
            )
            from orchestrator.utils import read_back_json
            ct_default: dict[str, Any] = {
                "domains": [],
                "crossings": [],
                "reset": {"name": "rst_n", "type": "synchronous",
                          "polarity": "active_low"},
                "cdc_required": False,
                "num_domains": 1,
            }
            disk_result, disk_ok = read_back_json(
                target_path, content, ct_default, context="clock_tree"
            )
            result = disk_result if disk_ok else _parse_response(content, target_clock_mhz)

            num_domains = result.get("num_domains", len(result.get("domains", [])))
            span.set_attribute("num_domains", num_domains)
            span.set_attribute("cdc_required", result.get("cdc_required", False))
            span.set_attribute("crossing_count", len(result.get("crossings", [])))

            return {"result": result, "questions": []}

        except Exception as e:
            span.set_attribute("error", str(e))
            span.set_status(_trace.StatusCode.ERROR, str(e))
            block_names = [b.get("name", "") for b in blocks]
            return {
                "result": {
                    "domains": [{
                        "name": "clk",
                        "frequency_mhz": target_clock_mhz,
                        "period_ns": 1000.0 / target_clock_mhz,
                        "source": "external",
                        "blocks": block_names,
                        "description": f"Fallback single domain ({e})",
                    }],
                    "crossings": [],
                    "reset": {
                        "name": "rst_n", "type": "synchronous",
                        "polarity": "active_low", "synchronizer": "rst_sync",
                        "description": "Synchronous active-low reset (fallback)",
                    },
                    "num_domains": 1,
                    "cdc_required": False,
                    "reasoning": f"Clock tree generation failed: {e}",
                },
                "questions": [],
            }


def _parse_response(content: str, target_clock_mhz: float) -> dict[str, Any]:
    """Extract structured JSON from LLM response."""
    from orchestrator.utils import parse_llm_json

    default = {
        "domains": [{
            "name": "clk",
            "frequency_mhz": target_clock_mhz,
            "period_ns": 1000.0 / target_clock_mhz,
            "source": "external",
            "blocks": [],
            "description": "Single system clock domain (parse fallback)",
        }],
        "crossings": [],
        "reset": {
            "name": "rst_n", "type": "synchronous",
            "polarity": "active_low", "synchronizer": "rst_sync",
            "description": "Synchronous active-low reset with 2-FF synchronizer",
        },
        "num_domains": 1,
        "cdc_required": False,
        "reasoning": "",
    }
    result, _ok = parse_llm_json(content, default, context="clock_tree")
    return result
