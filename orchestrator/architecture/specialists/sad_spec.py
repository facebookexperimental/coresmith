# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
SAD (System Architecture Document) Specialist.

Analyzes the PRD to produce system-level architecture decisions:
HW/FW/SW partitioning, system flows, technology rationale, and
architecture decisions with rationale.

The SAD answers: "How do we get there and why?"

Document hierarchy:  PRD -> SAD -> FRD -> Block Diagram -> ... -> ERS

The LLM produces Markdown directly (no JSON parsing required).
"""

from __future__ import annotations

import json
from typing import Any

from pathlib import Path


_PROMPT_FILE = Path(__file__).resolve().parents[2] / "langchain" / "prompts" / "sad_spec.md"
SYSTEM_PROMPT = _PROMPT_FILE.read_text()


def _build_shuttle_context() -> str:
    """Build shuttle constraint context from config.yaml tapeout section."""
    from orchestrator.langgraph.pipeline_helpers import load_config

    cfg = load_config()
    tapeout = cfg.get("tapeout", {})
    target = tapeout.get("target", "openframe")
    die_w = tapeout.get("die_width_um", 3520.0)
    die_h = tapeout.get("die_height_um", 5188.0)
    margin = tapeout.get("core_margin_um", 100.0)
    io_pads = tapeout.get("io_pads", 44)

    user_w = die_w - 2 * margin
    user_h = die_h - 2 * margin
    user_area_mm2 = (user_w * user_h) / 1e6

    return (
        f"Target shuttle: {target.upper()}\n"
        f"Die dimensions: {die_w:.0f} x {die_h:.0f} um "
        f"({die_w * die_h / 1e6:.3f} mm²)\n"
        f"User area: {user_w:.0f} x {user_h:.0f} um "
        f"({user_area_mm2:.3f} mm²)\n"
        f"Core margin: {margin:.0f} um on all sides\n"
        f"Total I/O pads: {io_pads}\n"
        f"Reserved pads: GPIO[0]=clk, GPIO[1]=rst (2 pads reserved)\n"
        f"Usable I/O pads: {io_pads - 2}\n"
        f"GPIO pad distribution: 19 left, 19 right, 6 bottom\n"
        f"Power domains: vccd1/vssd1 (digital 1.8V), "
        f"vdda1/vssa1 (analog 3.3V), vddio/vssio (I/O 3.3V)\n"
        f"Metal stack: 5 layers (li1, met1-met5), max routing met4\n"
        f"Wrapper module: openframe_project_wrapper\n"
        f"Wrapper ports: io_in[{io_pads - 1}:0], io_out[{io_pads - 1}:0], "
        f"io_oeb[{io_pads - 1}:0]\n"
        f"All outputs must be driven (unused tied to 0)\n"
        f"All OEB must be assigned (0=output, 1=input)\n"
    )


async def generate_sad(
    prd_spec: dict,
    requirements: str,
    pdk_summary: str,
    project_root: str = ".",
) -> dict[str, Any]:
    """Generate the System Architecture Document from the PRD.

    Args:
        prd_spec: Full PRD document (output of gather_prd Phase 2).
        requirements: Original high-level requirements text.
        pdk_summary: Available PDK technologies summary.
        project_root: Directory where SAD collateral should be written.

    Returns:
        {"sad_text": "<markdown>", "phase": "sad_complete"}
    """
    from opentelemetry import trace as _trace

    tracer = _trace.get_tracer("socmate.architecture.sad_spec")

    with tracer.start_as_current_span("generate_sad") as span:
        prd_doc = prd_spec.get("prd", {}) if prd_spec else {}
        span.set_attribute("has_prd", bool(prd_doc))

        prd_context = json.dumps(prd_doc, indent=2) if prd_doc else "No PRD available."

        system_prompt = SYSTEM_PROMPT.format(
            prd_context=prd_context,
            pdk_context=pdk_summary or "No PDK information available.",
            shuttle_context=_build_shuttle_context(),
        )

        target_path = Path(project_root) / "arch" / "sad_spec.md"
        target_path.parent.mkdir(parents=True, exist_ok=True)

        user_message = (
            f"Produce the System Architecture Document for this design.\n\n"
            f"Original requirements:\n{requirements}\n\n"
            f"The PRD has been provided in the system prompt.\n\n"
            f"IMPORTANT: Write the complete SAD document to: {target_path}\n"
            f"After writing, respond with only the file path confirmation."
        )

        from orchestrator.langchain.agents.socmate_llm import DEFAULT_MODEL, ClaudeLLM

        llm = ClaudeLLM(model=DEFAULT_MODEL, timeout=1200)

        try:
            content = await llm.call(
                system=system_prompt,
                prompt=user_message,
                run_name="generate_sad",
            )
            from orchestrator.utils import read_back_text
            sad_text = read_back_text(target_path, content.strip())
            span.set_attribute("phase", "sad_complete")
            return {"sad_text": sad_text, "phase": "sad_complete"}

        except Exception as e:
            span.set_attribute("error", str(e))
            return {
                "sad_text": f"# System Architecture Document (generation failed)\n\nSAD generation failed: {e}\n",
                "phase": "sad_complete",
            }
