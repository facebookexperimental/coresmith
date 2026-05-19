# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Memory Map Specialist -- designs the address space layout via LLM.

Uses ClaudeLLM for LLM inference. Takes the block diagram and
system requirements to produce a structured memory map with SRAM,
peripheral CSR blocks, and top-level CSR.

The prompt encodes the socmate bus conventions (nibble decode, peripheral
stride, SRAM range) so the LLM produces architecturally correct output
while making intelligent decisions about sizing and layout.
"""

from __future__ import annotations

import json
from typing import Any

from pathlib import Path

_PROMPT_FILE = Path(__file__).resolve().parents[2] / "langchain" / "prompts" / "memory_map.md"
SYSTEM_PROMPT = _PROMPT_FILE.read_text()


_DEFAULT_TOPOLOGY = """CONVENTIONS (default Ibex-style SoC -- override via ERS if needed):
1. SRAM occupies the lowest address range.
   Default: base 0x00000000, size 32KB (0x8000 bytes).
2. Peripheral CSR blocks start at 0x10000000 with stride 0x10000000.
   (peripheral 0 at 0x10000000, peripheral 1 at 0x20000000, etc.)
3. The peripheral bus uses addr[31:28] nibble decode, supporting up to
   8 peripherals (values 0x1 through 0x8).
4. Each peripheral CSR block defaults to 256 bytes (0x100) but should be
   sized based on the block's actual register needs.
5. Infrastructure blocks (axis_fifo, axis_adapter, axil_csr, clk_div,
   rst_sync, ibex_wrapper, periph_bus) do NOT get CSR allocations --
   they are controlled through the blocks they serve.
6. A top-level CSR block is always placed at 0x80000000 (256 bytes)."""

_SIMPLE_DESIGN_RESULT = {
    "sram": None,
    "peripherals": [],
    "top_csr": None,
    "address_decode_bits": "N/A",
    "peripheral_count": 0,
    "reasoning": "No bus architecture needed -- simple design with <= 3 blocks.",
}


async def analyze_memory_map(
    block_diagram: dict,
    target_clock_mhz: float = 50.0,
    requirements: str = "",
    project_root: str = ".",
    ers_spec: dict | None = None,
) -> dict[str, Any]:
    """Design the address space layout for the ASIC.

    Fix #3: Now supports ERS-driven topology.  For simple designs
    (<= 3 blocks, no bus infrastructure), returns a no-op memory map.

    Args:
        block_diagram: Block diagram from analyze_block_diagram().
        target_clock_mhz: Target clock frequency.
        requirements: High-level system requirements for context.
        project_root: Project root path (unused, kept for interface compat).
        ers_spec: Product Requirements Document (structured dict).

    Returns:
        Dict with keys: result (the memory map), questions.
    """
    from opentelemetry import trace as _trace

    tracer = _trace.get_tracer("socmate.architecture.memory_map")

    with tracer.start_as_current_span("analyze_memory_map") as span:
        blocks = block_diagram.get("blocks", [])
        span.set_attribute("input_block_count", len(blocks))

        # Simple-design escape hatch
        infra_names = {"axis_fifo", "axis_adapter", "axil_csr", "clk_div",
                       "rst_sync", "ibex_wrapper", "periph_bus"}
        datapath_blocks = [b for b in blocks if b.get("name", "") not in infra_names]
        has_bus_infra = any(b.get("name", "") in infra_names for b in blocks)

        if len(datapath_blocks) <= 3 and not has_bus_infra:
            span.set_attribute("simple_design", True)
            return {"result": dict(_SIMPLE_DESIGN_RESULT), "questions": []}

        # Build topology context from PRD or use defaults
        ers = (ers_spec.get("prd", ers_spec.get("ers", {})) if isinstance(ers_spec, dict) else {}) or {}
        dataflow = ers.get("dataflow", {}) or {}
        bus_protocol = dataflow.get("bus_protocol", "")
        data_width = dataflow.get("data_width_bits", "")
        dma_required = dataflow.get("dma_required", "")

        if bus_protocol and bus_protocol not in ("N/A", "none", "None"):
            topology_context = (
                f"ERS-SPECIFIED BUS ARCHITECTURE:\n"
                f"- Bus protocol: {bus_protocol}\n"
                f"- Data width: {data_width} bits\n"
                f"- DMA required: {dma_required}\n\n"
                f"Design the address map using the conventions of {bus_protocol}.\n"
                f"Choose address ranges, stride, and decode bits appropriate for "
                f"this protocol and the block count ({len(blocks)} blocks)."
            )
        else:
            topology_context = _DEFAULT_TOPOLOGY

        parts = [
            "Design the memory map for the following ASIC block diagram.",
            f"\nTarget clock: {target_clock_mhz} MHz",
        ]
        if requirements:
            parts.append(f"\nSystem requirements: {requirements}")

        parts.append(
            f"\n--- BLOCK DIAGRAM ---\n{json.dumps(block_diagram, indent=2)}"
        )

        # Read arch docs from disk for additional context
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

        target_path = _P(project_root) / ".socmate" / "memory_map.json"
        target_path.parent.mkdir(parents=True, exist_ok=True)

        parts.append(
            f"\nIMPORTANT: Write the memory map JSON to: {target_path}\n"
            "After writing, respond with only the file path confirmation."
        )

        user_message = "\n".join(parts)

        from orchestrator.langchain.agents.socmate_llm import DEFAULT_MODEL, ClaudeLLM

        llm = ClaudeLLM(model=DEFAULT_MODEL, timeout=1200)
        system_prompt = SYSTEM_PROMPT.format(topology_context=topology_context)

        try:
            content = await llm.call(
                system=system_prompt,
                prompt=user_message,
                run_name="memory_map",
            )
            from orchestrator.utils import read_back_json
            default = {
                "sram": {"base_address": "0x00000000", "base_address_int": 0,
                         "size": 32768, "size_kb": 32},
                "peripherals": [],
                "top_csr": {"name": "top_csr", "base_address": "0x80000000",
                            "base_address_int": 0x80000000, "size": 256,
                            "description": "Top-level control and status registers"},
                "address_decode_bits": "[31:28]",
                "peripheral_count": 0,
                "reasoning": "",
            }
            disk_result, disk_ok = read_back_json(
                target_path, content, default, context="memory_map"
            )
            result = disk_result if disk_ok else _parse_response(content)
            span.set_attribute("peripheral_count",
                               result.get("peripheral_count", 0))
            return {"result": result, "questions": []}

        except Exception as e:
            span.set_attribute("error", str(e))
            span.set_status(_trace.StatusCode.ERROR, str(e))
            return {
                "result": {
                    "sram": {"base_address": "0x00000000",
                             "base_address_int": 0,
                             "size": 32768, "size_kb": 32},
                    "peripherals": [],
                    "top_csr": {"name": "top_csr",
                                "base_address": "0x80000000",
                                "base_address_int": 0x80000000,
                                "size": 256,
                                "description": "Top-level control and status registers"},
                    "address_decode_bits": "[31:28]",
                    "peripheral_count": 0,
                    "reasoning": f"Memory map generation failed: {e}",
                },
                "questions": [],
            }


def _parse_response(content: str) -> dict[str, Any]:
    """Extract structured JSON from LLM response."""
    from orchestrator.utils import parse_llm_json

    default = {
        "sram": {"base_address": "0x00000000", "base_address_int": 0,
                 "size": 32768, "size_kb": 32},
        "peripherals": [],
        "top_csr": {"name": "top_csr", "base_address": "0x80000000",
                    "base_address_int": 0x80000000, "size": 256,
                    "description": "Top-level control and status registers"},
        "address_decode_bits": "[31:28]",
        "peripheral_count": 0,
        "reasoning": "",
    }
    result, _ok = parse_llm_json(content, default, context="memory_map")
    return result
