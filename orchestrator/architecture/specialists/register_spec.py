# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Register Spec Specialist -- designs CSR register layouts via LLM.

Uses ClaudeLLM for LLM inference. Takes the block diagram and
memory map to produce per-block register definitions that are tailored
to each block's actual function rather than generic templates.

The prompt encodes the axil_csr.v convention (config at 0x00, status at
0x40, 32-bit registers, byte-addressed) while letting the LLM design
meaningful field definitions for each block.
"""

from __future__ import annotations

import json
from typing import Any

from pathlib import Path

_PROMPT_FILE = Path(__file__).resolve().parents[2] / "langchain" / "prompts" / "register_spec.md"
SYSTEM_PROMPT = _PROMPT_FILE.read_text()


async def analyze_register_spec(
    block_diagram: dict,
    memory_map: dict | None = None,
    requirements: str = "",
    project_root: str = ".",
) -> dict[str, Any]:
    """Design CSR register layouts for all blocks.

    Args:
        block_diagram: Block diagram from analyze_block_diagram().
        memory_map: Memory map for address context (optional).
        requirements: High-level system requirements for context.
        project_root: Project root path (unused, kept for interface compat).

    Returns:
        Dict with keys: result (register specs), questions.
    """
    from opentelemetry import trace as _trace

    tracer = _trace.get_tracer("socmate.architecture.register_spec")

    with tracer.start_as_current_span("analyze_register_spec") as span:
        blocks = block_diagram.get("blocks", [])
        span.set_attribute("input_block_count", len(blocks))

        parts = [
            "Design the CSR register layouts for each block in the following "
            "ASIC block diagram. Tailor registers to each block's actual "
            "function -- do not use generic placeholder registers.",
        ]
        if requirements:
            parts.append(f"\nSystem requirements: {requirements}")

        parts.append(
            f"\n--- BLOCK DIAGRAM ---\n{json.dumps(block_diagram, indent=2)}"
        )

        if memory_map:
            parts.append(
                f"\n--- MEMORY MAP ---\n{json.dumps(memory_map, indent=2)}"
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

        target_path = _P(project_root) / ".socmate" / "register_spec.json" if project_root else _P.cwd() / ".socmate" / "register_spec.json"
        target_path.parent.mkdir(parents=True, exist_ok=True)

        parts.append(
            f"\nIMPORTANT: Write the register spec JSON to: {target_path}\n"
            "After writing, respond with only the file path confirmation."
        )

        user_message = "\n".join(parts)

        from orchestrator.langchain.agents.socmate_llm import DEFAULT_MODEL, ClaudeLLM

        llm = ClaudeLLM(model=DEFAULT_MODEL, timeout=1200)

        try:
            content = await llm.call(
                system=SYSTEM_PROMPT,
                prompt=user_message,
                run_name="register_spec",
            )
            from orchestrator.utils import read_back_json
            rs_default: dict[str, Any] = {
                "blocks": [],
                "total_blocks": 0,
                "register_convention": "",
                "reasoning": "",
            }
            disk_result, disk_ok = read_back_json(
                target_path, content, rs_default, context="register_spec"
            )
            result = disk_result if disk_ok else _parse_response(content)

            total_blocks = result.get("total_blocks", len(result.get("blocks", [])))
            total_regs = sum(len(b.get("registers", [])) for b in result.get("blocks", []))
            span.set_attribute("register_block_count", total_blocks)
            span.set_attribute("total_registers", total_regs)

            return {"result": result, "questions": []}

        except Exception as e:
            span.set_attribute("error", str(e))
            span.set_status(_trace.StatusCode.ERROR, str(e))
            return {
                "result": {
                    "blocks": [],
                    "total_blocks": 0,
                    "register_convention": "axil_csr: config (0x00-0x3C) + status (0x40-0x7C), 32-bit",
                    "reasoning": f"Register spec generation failed: {e}",
                },
                "questions": [],
            }


def _parse_response(content: str) -> dict[str, Any]:
    """Extract structured JSON from LLM response."""
    from orchestrator.utils import parse_llm_json

    default = {
        "blocks": [],
        "total_blocks": 0,
        "register_convention": "axil_csr: config (0x00-0x3C) + status (0x40-0x7C), 32-bit",
        "reasoning": "",
    }
    result, _ok = parse_llm_json(content, default, context="register_spec")
    return result
