# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
RTLGeneratorAgent -- converts Python DSP models to Verilog RTL.

Reads a Python source file implementing a signal processing block (e.g., LFSR
scrambler, Reed-Solomon encoder, FFT butterfly), understands the algorithm,
and generates synthesizable Verilog-2005 with AXI-Stream interfaces.

All invocations are traced via OpenTelemetry for observability and evaluation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from opentelemetry import trace

from .socmate_llm import ClaudeLLM

_tracer = trace.get_tracer(__name__)

# Load system prompt from template file, fall back to inline
_PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts" / "rtl_generator.md"
if _PROMPT_FILE.exists():
    SYSTEM_PROMPT = _PROMPT_FILE.read_text()
else:
    SYSTEM_PROMPT = """\
You are an expert digital design engineer specializing in converting Python
signal processing models into synthesizable Verilog-2005 RTL for ASIC
implementation on the SkyWater Sky130 130nm process.

RULES:
1. Output ONLY valid Verilog-2005 (no SystemVerilog constructs).
2. Use AXI-Stream (tdata/tvalid/tready/tlast) for data interfaces.
3. Use synchronous active-low reset (rst_n).
4. Use a single clock domain (clk).
5. All arithmetic must be fixed-point -- no floating point.
6. Use explicit bit widths on all signals. No implicit widths.
7. Include a module header comment with: block name, description, I/O ports.
8. Registers must have reset values.
9. FSMs must use localparam for state encoding.
10. No latches -- every conditional must have an else clause.
11. Combinational logic in always @(*) blocks, sequential in always @(posedge clk).
12. Target: fully synthesizable by Yosys for Sky130.

AXI-STREAM OUTPUT FSM -- CRITICAL:
When producing output on an AXI-Stream master port, you MUST follow this
two-phase pattern to avoid the "valid self-cancellation" bug:

  WRONG (valid is set and cleared in the same combinational pass):
    ST_OUTPUT: begin
        m_tvalid_next = 1'b1;          // set valid...
        if (m_tready)                   // ...but tready is already 1...
            m_tvalid_next = 1'b0;      // ...so valid is immediately cleared!
    end
    // Result: m_tvalid_reg NEVER becomes 1. Deadlock.

  CORRECT (set valid, wait one cycle for handshake):
    ST_OUTPUT: begin
        m_tvalid_next = 1'b1;          // assert valid
        if (m_tvalid_reg && m_tready)   // handshake on REGISTERED valid
            m_tvalid_next = 1'b0;      // clear after transfer
            state_next = ST_IDLE;
        end
    end
    // Result: valid rises for at least 1 cycle, handshake completes.

  SIMPLEST (registered output, always correct):
    always @(posedge clk)
        if (!rst_n) m_tvalid <= 0;
        else if (produce_data) m_tvalid <= 1;
        else if (m_tvalid && m_tready) m_tvalid <= 0;

When converting Python to Verilog:
- Map numpy arrays to register files or SRAM.
- Map Python loops to FSMs with counters or combinational unrolling.
- Map dictionary lookups to ROM/LUT.
- Map floating-point math to fixed-point (specify Q format in comments).
- Handle variable-length data with valid/ready handshaking.

If the previous attempt failed, the error will be provided. Fix the specific
issue while maintaining correctness.

Output format:
1. The complete Verilog module (one module per response).
2. After the module, a JSON block with port information:
   ```json
   {{"module_name": "...", "ports": {{"clk": "input", ...}}}}
   ```
"""


class RTLGeneratorAgent:
    """Agent for Python-to-Verilog RTL generation.

    DISK-FIRST: Tools are enabled so the agent reads all input files
    (uArch spec, ERS, constraints, golden model, previous error) directly
    from disk and writes the Verilog output to disk.  On retry (attempt > 1),
    the agent can use Edit to incrementally fix existing RTL instead of
    regenerating from scratch.
    """

    _DEFAULT_PROCESS = "SkyWater Sky130 130nm"
    _DEFAULT_RTL_LANG = "Verilog-2005"
    _DEFAULT_SYNTH_TOOL = "Yosys"
    _DEFAULT_CONSTRAINTS = "No tri-state buffers, no async resets, no latches (Sky130 limitations)."

    def __init__(self, model: str | None = None, temperature: float = 0.1):
        # Block agents default to Sonnet (cheaper than Opus); override via
        # SOCMATE_BLOCK_MODEL or the model= kwarg.
        from orchestrator.langchain.agents.socmate_llm import block_model
        model = model or block_model()
        # 1800s default; bump via SOCMATE_RTL_TIMEOUT env var for complex blocks
        # like CPUs / multi-stage pipelines that need more agent turns to write.
        self.llm = ClaudeLLM(
            model=model,
            timeout=int(os.environ.get("SOCMATE_RTL_TIMEOUT", "1800")),
        )

    async def generate(
        self,
        block_name: str,
        description: str = "",
        attempt: int = 1,
        rtl_target: str = "",
        python_source_path: str = "",
        project_root: str = "",
        target_process: str = "",
        rtl_language: str = "",
        synthesis_tool: str = "",
        process_constraints: str = "",
        callbacks: list = None,
    ) -> dict[str, Any]:
        """Generate Verilog RTL -- agent reads all context from disk.

        Args:
            block_name: Name of the hardware block
            description: Human-readable description
            attempt: Current attempt number
            rtl_target: Relative path to write Verilog (e.g. rtl/foo/bar.v)
            python_source_path: Relative path to Python golden model
            project_root: Path to project root

        Returns:
            Dict with keys: rtl_path (or error)
        """
        block_title = block_name.replace("_", " ").title()
        retry_label = f" - Retry #{attempt - 1}" if attempt > 1 else ""
        span_name = f"RTL Generator [{block_title}]{retry_label}"

        with _tracer.start_as_current_span(span_name) as span:
            span.set_attribute("block_name", block_name)
            span.set_attribute("attempt", attempt)

            _proc = target_process or self._DEFAULT_PROCESS
            _lang = rtl_language or self._DEFAULT_RTL_LANG
            _tool = synthesis_tool or self._DEFAULT_SYNTH_TOOL
            _pcon = process_constraints or self._DEFAULT_CONSTRAINTS

            parts = [
                f"Block name: {block_name}",
                f"Description: {description}",
                f"Attempt: {attempt}",
                "",
                "## Working Files",
                "Read these files to understand the design:",
                f"- uArch Spec: arch/uarch_specs/{block_name}.md",
                "- ERS: arch/ers_spec.md",
                f"- Constraints: .socmate/blocks/{block_name}/constraints.json",
                f"- Golden Model: {python_source_path}",
                "- Block Diagram: .socmate/block_diagram.json (for interface context)",
            ]

            if attempt > 1:
                parts.extend([
                    f"- Previous Error: .socmate/blocks/{block_name}/previous_error.txt",
                    f"- Existing RTL: {rtl_target} (use Edit to fix incrementally if possible)",
                ])

            parts.extend([
                "",
                "## Output",
                f"Write the complete synthesizable {_lang} module to: {rtl_target}",
                "Keep reset, handshakes, state, sideband metadata, error flags, "
                "and pipeline boundary signals named and observable so the "
                "downstream Verilator VCD can be audited with WaveKit.",
                "",
            ])

            if attempt > 1:
                parts.append(
                    "This is a RETRY. Read the previous error and the existing RTL. "
                    "If the fix is surgical, use the Edit tool to modify the existing "
                    "RTL in-place. Only regenerate from scratch if the design is "
                    "fundamentally wrong."
                )
            else:
                parts.append(
                    "Read the uArch spec and golden model, then generate the "
                    "complete Verilog module and write it to the output path."
                )

            user_message = "\n".join(parts)

            system_prompt = SYSTEM_PROMPT.format(
                target_process=_proc,
                rtl_language=_lang,
                synthesis_tool=_tool,
                process_constraints=_pcon,
            )

            run_name = f"Generate Verilog [{block_title}]{retry_label}"
            await self.llm.call(
                system=system_prompt,
                prompt=user_message,
                run_name=run_name,
            )

            rtl_path = Path(project_root) / rtl_target if project_root else Path(rtl_target)
            if rtl_path.exists():
                return {"rtl_path": str(rtl_path)}
            else:
                return {"error": f"Agent did not write RTL to {rtl_target}"}

    def _parse_response(
        self, content: str, block_name: str
    ) -> tuple[str, dict]:
        """Extract Verilog code and port info from LLM response."""
        import json
        import re

        # Extract Verilog code block
        verilog_match = re.search(
            r"```(?:verilog|v)?\s*\n(.*?)```",
            content,
            re.DOTALL,
        )
        if verilog_match:
            verilog = verilog_match.group(1).strip()
        else:
            # Assume the entire response is Verilog if no code block found
            verilog = content.strip()

        # Extract JSON port info
        ports = {}
        json_match = re.search(r"```json\s*\n(.*?)```", content, re.DOTALL)
        if json_match:
            try:
                port_info = json.loads(json_match.group(1))
                ports = port_info.get("ports", port_info)
            except json.JSONDecodeError:
                pass

        # Remove JSON block from verilog if it got included
        if "```json" in verilog:
            verilog = verilog[:verilog.index("```json")].strip()

        # Validate: reject error messages and prose written as Verilog
        if not verilog or "[ClaudeLLM error:" in verilog:
            raise ValueError(f"RTL generation returned error, not Verilog: {verilog[:200]}")
        first_nonblank = verilog.lstrip()[:10]
        if first_nonblank.startswith(("##", "# ", "---", "The ", "I ", "Right")):
            raise ValueError(f"RTL response contains prose, not Verilog: {verilog[:200]}")
        if not re.search(r"^\s*module\s+\w+", verilog, re.MULTILINE):
            raise ValueError(f"RTL response does not contain a Verilog module declaration: {verilog[:200]}")

        return verilog, ports
