# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
IntegrationTestbenchGenerator -- Generates cocotb testbenches for the
chip-level integration module that wires all blocks together.

Unlike the per-block TestbenchGeneratorAgent, this agent:
- Takes the top-level RTL + all block port summaries as context
- Generates system-level tests (reset, smoke, throughput, backpressure)
- Does NOT rely on golden model imports (self-contained test vectors)
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from opentelemetry import trace

from .socmate_llm import DEFAULT_MODEL, ClaudeLLM

_tracer = trace.get_tracer(__name__)

_PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts" / "integration_testbench.md"
if _PROMPT_FILE.exists():
    SYSTEM_PROMPT = _PROMPT_FILE.read_text()
else:
    SYSTEM_PROMPT = (
        "You are a Lead DV engineer. Generate a cocotb integration testbench "
        "for the chip top-level module. Output a single Python file."
    )


class IntegrationTestbenchGenerator:
    """Agent for chip-level integration cocotb testbench generation."""

    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = 0.1):
        self.llm = ClaudeLLM(
            model=model,
            timeout=int(os.environ.get("SOCMATE_INTEGRATION_TB_TIMEOUT", "2700")),
        )

    async def generate(
        self,
        design_name: str,
        top_rtl_source: str,
        block_summaries: list[dict],
        connections: list[dict],
        prd_summary: str = "",
        block_rtl_paths: dict[str, str] | None = None,
        output_path: str = "",
        prior_failure: str = "",
    ) -> dict[str, Any]:
        """Generate a cocotb integration testbench.

        Args:
            design_name: Top-level module name (e.g. "fft_processor_top").
            top_rtl_source: Full Verilog of the top-level wrapper.
            block_summaries: List of dicts with block name, port count, ports.
            connections: Architecture connection list.
            prd_summary: PRD summary text for requirements context.
            block_rtl_paths: Map of block_name -> RTL file path.
            output_path: File path to write the generated testbench to.

        Returns:
            Dict with keys: tb_path (str), test_count (int).
        """
        with _tracer.start_as_current_span(
            f"Integration Testbench [{design_name}]"
        ) as span:
            span.set_attribute("design_name", design_name)
            span.set_attribute("block_count", len(block_summaries))

            parts = [
                f"Generate a cocotb integration testbench for the top-level "
                f"module '{design_name}'.",
            ]

            if output_path:
                parts.append(
                    f"Write the complete testbench to: {output_path}."
                )

            parts.append(
                f"\n--- TOP-LEVEL VERILOG ---\n```verilog\n{top_rtl_source}\n```"
            )

            parts.append("\n--- BLOCK SUMMARIES ---")
            for bs in block_summaries:
                name = bs.get("name", "unknown")
                ports = bs.get("ports", [])
                port_str = ", ".join(
                    f"{p['name']}({p['direction']} [{p.get('width',1)}-bit])"
                    for p in ports[:20]
                )
                parts.append(f"  {name}: {port_str}")

            if connections:
                parts.append("\n--- ARCHITECTURE CONNECTIONS ---")
                for c in connections[:30]:
                    fb = c.get("from_block", c.get("from", "?"))
                    tb = c.get("to_block", c.get("to", "?"))
                    iface = c.get("interface", c.get("name", ""))
                    dw = c.get("data_width", "?")
                    parts.append(f"  {fb} -> {tb} ({iface}, {dw}-bit)")

            if prd_summary:
                parts.append(f"\n--- PRD SUMMARY ---\n{prd_summary}")

            if block_rtl_paths:
                verilog_sources = " ".join(block_rtl_paths.values())
                parts.append(
                    f"\n--- VERILOG_SOURCES for Makefile ---\n"
                    f"All block RTL files (space-separated):\n{verilog_sources}"
                )

            parts.append(
                "\n--- VCD / WAVEKIT REQUIREMENT ---\n"
                "The pipeline will dump sim_build/integration/dump.vcd and "
                "audit it with WaveKit. Generate tests that advance time and "
                "exercise reset, top-level handshakes, block-boundary flow, "
                "backpressure, sideband metadata, and final outputs so the "
                "waveform audit contains meaningful evidence."
            )

            if prior_failure:
                parts.append(
                    "\n--- PRIOR ATTEMPT FAILURE / CONTRACT AUDIT ---\n"
                    "This is a retry. The previous attempt failed. You MUST "
                    "address the first divergence and suggested fix below; do "
                    "not regenerate the same bug.\n"
                    f"{prior_failure}"
                )

            user_message = "\n".join(p for p in parts if p)

            content = await self.llm.call(
                system=SYSTEM_PROMPT,
                prompt=user_message,
                run_name=f"Integration Testbench [{design_name}]",
            )

            # Don't trust ClaudeLLM.call to raise on CLI failure -- it
            # returns an error string in the output position. Validate
            # the response actually contains a Python code block with
            # @cocotb.test() functions before claiming success; the
            # previous max(test_count, 1) lied to downstream SIM.
            testbench = self._extract_python(content)
            if output_path:
                disk_path = Path(output_path)
                if disk_path.exists():
                    disk_content = disk_path.read_text(encoding="utf-8")
                    disk_test_count = len(
                        re.findall(r"@cocotb\.test\(\)", disk_content)
                    )
                    if disk_test_count > 0:
                        testbench = disk_content
            test_count = len(re.findall(r"@cocotb\.test\(\)", testbench))

            if not testbench or test_count == 0:
                raise RuntimeError(
                    "Integration testbench generation failed: "
                    "claude CLI returned no usable Python code block "
                    "with @cocotb.test() functions"
                )
            if not output_path:
                raise RuntimeError(
                    "Integration testbench generation failed: no "
                    "output_path provided; cannot persist the testbench"
                )

            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(testbench, encoding="utf-8")
            written_path = output_path

            span.set_attribute("test_count", test_count)

            return {
                "tb_path": written_path,
                "test_count": test_count,
            }

    def _extract_python(self, content: str) -> str:
        """Extract Python code block from LLM response."""
        match = re.search(r"```python\s*\n(.*?)```", content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return content.strip()
