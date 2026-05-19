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
from typing import Any

from pathlib import Path


_PROMPT_FILE = Path(__file__).resolve().parents[2] / "langchain" / "prompts" / "ers_doc.md"
SYSTEM_PROMPT = _PROMPT_FILE.read_text()


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

    tracer = _trace.get_tracer("socmate.architecture.ers_doc")

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

        target_path = Path(project_root) / ".socmate" / "ers_spec.json"
        target_path.parent.mkdir(parents=True, exist_ok=True)

        user_message = (
            "Produce the Engineering Requirements Specification (ERS) "
            "by synthesizing all the upstream architecture documents "
            "provided in the system prompt.\n\n"
            f"IMPORTANT: Write the complete ERS JSON to: {target_path}\n"
            "After writing, respond with only the file path confirmation."
        )

        from orchestrator.langchain.agents.socmate_llm import DEFAULT_MODEL, ClaudeLLM

        llm = ClaudeLLM(model=DEFAULT_MODEL, timeout=1200)

        try:
            content = await llm.call(
                system=system_prompt,
                prompt=user_message,
                run_name="generate_ers_doc",
            )
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
            disk_result, disk_ok = read_back_json(
                target_path, content, ers_default, context="ers_doc"
            )
            result = disk_result if disk_ok else _parse_response(content)
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
