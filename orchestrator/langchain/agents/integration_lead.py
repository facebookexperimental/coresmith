# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
IntegrationLeadAgent -- LLM-driven integration of RTL blocks into a
chip-level top module.

Replaces the previous deterministic compatibility check and top-level
Verilog generation with an agent that can reason about port semantics,
naming conventions, and cross-block wiring.

The agent:
1. Analyzes all block RTL ports against the architecture connection graph
2. Identifies compatibility issues (width mismatches, missing ports, etc.)
3. Generates a synthesizable top-level Verilog module wiring all blocks
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from opentelemetry import trace

from orchestrator.langchain.prompts.skills import load_skills as _load_skills

from .coresmith_llm import DEFAULT_MODEL, ClaudeLLM

_tracer = trace.get_tracer(__name__)

_PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts" / "integration_lead.md"
if _PROMPT_FILE.exists():
    SYSTEM_PROMPT = _PROMPT_FILE.read_text()
else:
    SYSTEM_PROMPT = (
        "You are an Integration Lead engineer. Analyze block RTL ports for "
        "compatibility and generate a top-level Verilog module wiring all "
        "blocks together. Respond with JSON only."
    )

# Pull the handshake-protocol skills into the system prompt so the
# Integration Lead has access to the same packing / bootstrap / sideband
# conventions the uArch spec generator used. Missing skills degrade
# silently.
_SKILLS_TEXT = _load_skills("axi_stream", "srdy_drdy", "arithmetic_precision")
if _SKILLS_TEXT:
    SYSTEM_PROMPT = (
        SYSTEM_PROMPT
        + "\n\n# Reference Skills (use when wiring chip_top)\n\n"
        + _SKILLS_TEXT
    )


class IntegrationLeadAgent:
    """Agent for chip-level integration: compatibility check + top-level RTL."""

    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = 0.1):
        self.llm = ClaudeLLM(
            model=model,
            timeout=int(os.environ.get("CORESMITH_INTEGRATION_LEAD_TIMEOUT", "2700")),
        )

    async def integrate(
        self,
        design_name: str,
        block_rtl_sources: dict[str, str],
        block_port_summaries: list[dict],
        connections: list[dict],
        prd_summary: str = "",
        output_path: str = "",
    ) -> dict[str, Any]:
        """Analyze compatibility and generate top-level integration module.

        Args:
            design_name: Name for the top-level module.
            block_rtl_sources: Map of block_name -> full Verilog source text.
            block_port_summaries: List of dicts with block name, ports, etc.
            connections: Architecture connection list from block diagram.
            prd_summary: PRD summary for requirements context.
            output_path: File path to write the generated Verilog to.

        Returns:
            Dict with keys: rtl_path, mismatches, module_name, wire_count,
            skipped_connections, notes.
        """
        with _tracer.start_as_current_span(
            f"Integration Lead [{design_name}]"
        ) as span:
            span.set_attribute("design_name", design_name)
            span.set_attribute("block_count", len(block_rtl_sources))

            user_message = self._build_prompt(
                design_name, block_rtl_sources, block_port_summaries,
                connections, prd_summary, output_path,
            )

            content = await self.llm.call(
                system=SYSTEM_PROMPT,
                prompt=user_message,
                run_name=f"Integration Lead [{design_name}]",
            )

            result = self._parse_response(content, design_name)

            verilog = result.pop("verilog", "")

            # The LLM may have used a file-edit tool to write the top RTL
            # directly to disk and then returned a JSON whose `verilog`
            # field is a self-`include` or other placeholder (seen with
            # Codex/gpt-5.5: the response wrote `\`include "<output_path>"\``
            # which trivially compiles to a recursive include).  Refuse to
            # overwrite a working on-disk module with a broken one-liner,
            # and refuse to write a broken one-liner from scratch -- raise
            # so the integration_check node escalates with a retry-able
            # error.
            def _has_real_module(text: str, design_name: str) -> bool:
                if not text:
                    return False
                if "module" not in text:
                    return False
                if f"module {design_name}" not in text and "module " not in text:
                    return False
                if len(text.strip().splitlines()) < 5:
                    return False
                # Reject obvious self-include placeholders.
                stripped = text.strip()
                if (
                    stripped.startswith("`include")
                    and output_path
                    and output_path in stripped
                ):
                    return False
                return True

            disk_content = ""
            if output_path:
                disk_path = Path(output_path)
                if disk_path.exists():
                    try:
                        disk_content = disk_path.read_text(encoding="utf-8")
                    except OSError:
                        disk_content = ""

            chosen_verilog = ""
            if _has_real_module(verilog, design_name):
                chosen_verilog = verilog
            elif _has_real_module(disk_content, design_name):
                chosen_verilog = disk_content
                result.setdefault("notes", "")
                result["notes"] += (
                    " (used existing on-disk integration top because the LLM "
                    "response did not contain a valid module declaration)"
                )

            if chosen_verilog and output_path:
                out = Path(output_path)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(chosen_verilog, encoding="utf-8")
                result["rtl_path"] = output_path
            elif output_path:
                # Neither the response nor disk has a valid module.
                # Raise so the pipeline retries instead of writing junk
                # that produces a recursive-include lint error.
                result["rtl_path"] = ""
                raise RuntimeError(
                    "Integration Lead returned no valid top-level Verilog "
                    f"module for '{design_name}'. The response `verilog` "
                    f"field had {len(verilog)} chars but did not contain a "
                    f"proper module declaration, and the on-disk file at "
                    f"{output_path} did not contain a valid module either. "
                    "Retry."
                )
            else:
                result["rtl_path"] = ""

            span.set_attribute("mismatch_count", len(result.get("mismatches", [])))
            span.set_attribute("wire_count", result.get("wire_count", 0))
            span.set_attribute("module_name", result.get("module_name", ""))

            return result

    def _build_prompt(
        self,
        design_name: str,
        block_rtl_sources: dict[str, str],
        block_port_summaries: list[dict],
        connections: list[dict],
        prd_summary: str,
        output_path: str = "",
    ) -> str:
        parts = [
            f"Design name: {design_name}",
            f"Total blocks: {len(block_rtl_sources)}",
        ]

        parts.append("\n--- BLOCK RTL SOURCES ---")
        for name, source in sorted(block_rtl_sources.items()):
            truncated = source[:8000]
            parts.append(f"\n### {name}\n```verilog\n{truncated}\n```")

        parts.append("\n--- PARSED PORT SUMMARIES ---")
        for bs in block_port_summaries:
            name = bs.get("name", "unknown")
            ports = bs.get("ports", [])
            port_lines = []
            for p in ports[:30]:
                w = p.get("width", 1)
                width_str = f"[{w}-bit]" if w > 1 else "[1-bit]"
                port_lines.append(
                    f"    {p['direction']} {width_str} {p['name']}"
                )
            parts.append(f"  {name}:")
            parts.extend(port_lines)

        if connections:
            parts.append("\n--- ARCHITECTURE CONNECTIONS ---")
            for c in connections[:50]:
                fb = c.get("from_block", c.get("from", "?"))
                fp = c.get("from_port", "")
                tb = c.get("to_block", c.get("to", "?"))
                tp = c.get("to_port", "")
                iface = c.get("interface", c.get("name", ""))
                dw = c.get("data_width", "?")
                parts.append(
                    f"  {fb}.{fp} -> {tb}.{tp} "
                    f"(interface: {iface}, width: {dw})"
                )

        if prd_summary:
            parts.append(f"\n--- PRD SUMMARY ---\n{prd_summary}")

        out_instr = ""
        if output_path:
            out_instr = (
                f" Write the complete top-level Verilog module to: {output_path}."
            )

        parts.append(
            f"\nGenerate the integration analysis and top-level Verilog "
            f"for module '{design_name}'.{out_instr} Respond with JSON only."
        )

        return "\n".join(parts)

    def _parse_response(self, content: str, design_name: str) -> dict[str, Any]:
        """Parse the LLM JSON response, with fallback extraction."""
        content = content.strip()

        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return self._validate_result(data, design_name)
            except json.JSONDecodeError:
                pass

        return {
            "verilog": "",
            "mismatches": [],
            "module_name": design_name,
            "wire_count": 0,
            "skipped_connections": [],
            "notes": f"Failed to parse LLM response. Raw: {content[:500]}",
            "parse_error": True,
        }

    def _validate_result(self, data: dict, design_name: str) -> dict[str, Any]:
        """Validate and normalize the parsed result."""
        result = {
            "verilog": data.get("verilog", ""),
            "mismatches": data.get("mismatches", []),
            "module_name": data.get("module_name", design_name),
            "wire_count": data.get("wire_count", 0),
            "skipped_connections": data.get("skipped_connections", []),
            "notes": data.get("notes", ""),
        }

        for m in result["mismatches"]:
            if "severity" not in m:
                m["severity"] = "warning"
            if "issue_type" not in m:
                m["issue_type"] = "unknown"

        if result["verilog"] and "module" not in result["verilog"]:
            result["notes"] += " WARNING: verilog field does not contain a module declaration."

        return result


def assert_blocks_instantiated(
    chip_top_verilog: str, expected_block_names: set[str]
) -> str | None:
    """Postcondition: every expected block must appear as an instantiation
    inside the Integration Lead's chip_top Verilog. Returns None on success
    or a descriptive error string listing the missing blocks.

    The Integration Lead has historically been observed to silently drop
    blocks from block_diagram.json and substitute glue stubs (e.g.,
    core_block -> rle_to_packer_token_bridge). Lint passes because the
    substitute compiles, but the chip is structurally wrong.
    """
    if not chip_top_verilog or not expected_block_names:
        return None

    # Strip line and block comments to avoid matching block names that
    # appear only in commentary.
    code = re.sub(r"//[^\n]*", "", chip_top_verilog)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)

    missing: list[str] = []
    for block_name in expected_block_names:
        # An instantiation is either
        #   <module> <inst_name> ( ... );           non-parameterized
        # or
        #   <module> #( <params> ) <inst_name> (    parameterized
        # We accept either by allowing the next token after the module name
        # to be `#` (parameter override) or an identifier followed by `(`.
        pattern = (
            rf"\b{re.escape(block_name)}\s+"
            rf"(?:#|[a-zA-Z_]\w*\s*\()"
        )
        if not re.search(pattern, code):
            missing.append(block_name)

    if missing:
        return (
            f"Integration Lead postcondition failed: chip_top RTL does NOT "
            f"instantiate {len(missing)} expected block(s): "
            f"{sorted(missing)}. The Integration Lead may have silently "
            f"dropped blocks or substituted glue stubs (the core_block -> "
            f"rle_to_packer_token_bridge failure mode). Refusing to proceed."
        )
    return None
