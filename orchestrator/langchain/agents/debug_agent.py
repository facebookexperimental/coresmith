# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
DebugAgent -- Analyzes simulation and synthesis failures to diagnose
root causes and propose fixes.

Operates in two modes:
1. "debug": Diagnose a simulation mismatch and suggest RTL patches
2. "architecture_review": Assess whether the failure requires interface
   changes that affect other blocks

Core tenet: every subagent may fail. The debug agent itself must be
resilient -- it produces a structured diagnosis even if the LLM response
is partial or unexpected.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from opentelemetry import trace

from .socmate_llm import ClaudeLLM

_tracer = trace.get_tracer(__name__)

# Load system prompt from template file, fall back to inline
_PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts" / "debug_agent.md"
if _PROMPT_FILE.exists():
    DEBUG_SYSTEM_PROMPT = _PROMPT_FILE.read_text()
else:
    DEBUG_SYSTEM_PROMPT = """\
You are an expert digital design debug engineer. You analyze simulation
failures in RTL (Verilog) designs and diagnose root causes.

Given:
- Block name and description
- Simulation error log (cocotb output)
- Previous attempt history (what was tried before)

Your job:
1. Identify which signal diverged first (if waveform info available).
2. Determine the root cause category:
   - LOGIC_ERROR: incorrect combinational/sequential logic
   - TIMING_ISSUE: race condition, setup/hold violation
   - INTERFACE_MISMATCH: wrong handshaking protocol or data width
   - RESET_BUG: incorrect reset behavior
   - ARITHMETIC_ERROR: overflow, truncation, wrong fixed-point format
   - STATE_MACHINE_BUG: wrong state transitions or missing states
3. Propose a specific fix (code change, not vague advice).
4. Assess whether this is fixable locally or requires architecture change.
5. Extract concrete constraints (MUST / MUST NOT rules) that the RTL generator
   should follow on the next attempt to avoid repeating this mistake.
6. If you have seen the same failure category twice in the attempt history, or
   your confidence is below 0.5, set needs_human to true and provide a clear
   question for the human engineer.

Output a JSON object with these fields:
- diagnosis: string describing the root cause
- category: one of the categories above
- suggested_fix: specific code change description
- affected_blocks: list of other blocks that would need changes (empty if local fix)
- escalate: boolean -- true if this needs architecture revision
- confidence: float 0-1
- constraints: list of strings, each a MUST or MUST NOT rule for the RTL generator
- needs_human: boolean -- true if same category appeared 2+ times or confidence < 0.5
- human_question: string -- a clear question for the human engineer (empty if needs_human is false)
"""

ARCH_REVIEW_PROMPT = """\
You are a chip architect reviewing a block that has failed repeatedly.
After {attempts} attempts, the block '{block_name}' cannot converge.

Error history:
{error_history}

Analyze whether:
1. The issue is local (fixable by changing this block's RTL only)
2. The issue requires interface changes (affecting neighboring blocks)
3. The issue requires architectural rethinking (e.g., different algorithm,
   pipelining, memory architecture)

For each affected block, specify what interface change is needed.

Output a JSON object with:
- diagnosis: root cause analysis
- local_fix_possible: boolean
- interface_changes: dict mapping block_name -> description of change
- affected_blocks: list of block names that need re-verification
- architectural_changes: list of high-level changes needed
"""


class DebugAgent:
    """Agent for failure analysis and debugging.

    DISK-FIRST: Tools are enabled so the agent reads all files directly
    from disk (RTL, testbench, error logs, uArch spec, constraints,
    block diagram connections) with NO truncation.  It writes diagnosis
    and updated constraints back to disk.
    """

    def __init__(self, model: str | None = None, temperature: float = 0.1):
        from orchestrator.langchain.agents.socmate_llm import block_model
        model = model or block_model()
        self.llm = ClaudeLLM(model=model, timeout=900)

    async def analyze(
        self,
        block_name: str,
        phase: str = "sim",
        project_root: str = "",
        mode: str = "debug",
        callbacks: list = None,
    ) -> dict[str, Any]:
        """Analyze a failure and produce a diagnosis.

        Disk-first: the agent reads all context from disk using file
        paths.  No content is passed through function arguments.

        Args:
            block_name: Name of the failed block
            phase: Failed phase ("lint", "sim", "synth")
            project_root: Path to project root
            mode: "debug" or "architecture_review"

        Returns:
            Dict with diagnosis, suggested_fix, affected_blocks, escalate
        """
        block_title = block_name.replace("_", " ").title()
        mode_label = "Architecture Review" if mode == "architecture_review" else "Debug Analysis"
        span_name = f"{mode_label} [{block_title}]"

        with _tracer.start_as_current_span(span_name) as span:
            span.set_attribute("block_name", block_name)
            span.set_attribute("mode", mode)

            if mode == "architecture_review":
                return await self._architecture_review(
                    block_name, project_root, callbacks=callbacks,
                )
            else:
                return await self._debug_analysis(
                    block_name, phase, project_root, callbacks=callbacks,
                )

    async def _debug_analysis(
        self,
        block_name: str,
        phase: str,
        project_root: str,
        callbacks: list = None,
    ) -> dict[str, Any]:
        """Standard debug analysis -- agent reads all files from disk."""
        block_title = block_name.replace("_", " ").title()

        try:
            user_message = (
                f"Block: {block_name}\n"
                f"Failed phase: {phase}\n\n"
                f"## Working Files\n"
                f"Read these files to understand the failure:\n"
                f"- Error log: .socmate/blocks/{block_name}/previous_error.txt\n"
                f"- Step logs: .socmate/step_logs/{block_name}/ (read the latest {phase}_attempt*.log)\n"
                f"- VCD waveform: sim_build/{block_name}/dump.vcd\n"
                f"- WaveKit audit: sim_build/{block_name}/wavekit_audit.json\n"
                f"- Integration VCD, if this is a chip-level failure: sim_build/integration/dump.vcd\n"
                f"- Integration WaveKit audit, if relevant: sim_build/integration/wavekit_audit.json\n"
                f"- RTL source: find the .v file for this block under rtl/\n"
                f"- Testbench: tb/cocotb/test_{block_name}.py\n"
                f"- uArch spec: arch/uarch_specs/{block_name}.md\n"
                f"- Constraints: .socmate/blocks/{block_name}/constraints.json\n"
                f"- Attempt history: .socmate/blocks/{block_name}/attempt_history.json\n"
                f"- Block diagram: .socmate/block_diagram.json (for interface context)\n"
                f"- ERS: arch/ers_spec.md\n\n"
                f"## Instructions\n"
                f"1. Read the error log, step logs, VCD, and WaveKit audit to understand what failed\n"
                f"2. Read the full RTL source and testbench to understand the design\n"
                f"3. Read the uArch spec for design intent\n"
                f"4. Diagnose the root cause\n"
                f"5. Write your diagnosis to .socmate/blocks/{block_name}/diagnosis.json\n"
                f"6. If you identify new constraints, append them to "
                f".socmate/blocks/{block_name}/constraints.json\n\n"
                f"The diagnosis.json must contain these fields:\n"
                f'  diagnosis, category, suggested_fix, affected_blocks, '
                f'escalate, confidence, constraints, needs_human, '
                f'human_question, is_testbench_bug\n\n'
                f"Categories: LOGIC_ERROR, TIMING_ISSUE, INTERFACE_MISMATCH, "
                f"RESET_BUG, ARITHMETIC_ERROR, STATE_MACHINE_BUG, "
                f"TESTBENCH_BUG, INFRASTRUCTURE_ERROR"
            )

            run_name = f"Analyze Failure [{block_title}]"
            await self.llm.call(
                system=DEBUG_SYSTEM_PROMPT,
                prompt=user_message,
                run_name=run_name,
            )

            diag_path = Path(project_root) / ".socmate" / "blocks" / block_name / "diagnosis.json"
            if diag_path.exists():
                return json.loads(diag_path.read_text())

            return {
                "diagnosis": "Debug agent did not write diagnosis file",
                "category": "AGENT_ERROR",
                "suggested_fix": "Review manually",
                "affected_blocks": [],
                "escalate": False,
                "confidence": 0.3,
                "constraints": [],
                "needs_human": False,
                "human_question": "",
                "is_testbench_bug": False,
            }

        except Exception as e:
            return {
                "diagnosis": f"Debug agent error: {e}",
                "category": "AGENT_ERROR",
                "suggested_fix": "Review RTL and testbench manually",
                "affected_blocks": [],
                "escalate": True,
                "confidence": 0.0,
                "constraints": [],
                "needs_human": True,
                "human_question": f"Debug agent failed ({e}). Please review the error log manually.",
                "is_testbench_bug": False,
            }

    async def _architecture_review(
        self,
        block_name: str,
        project_root: str,
        callbacks: list = None,
    ) -> dict[str, Any]:
        """Architecture-level review -- agent reads all context from disk."""
        block_title = block_name.replace("_", " ").title()

        try:
            user_message = (
                f"Block: {block_name}\n\n"
                f"## Working Files\n"
                f"- Attempt history: .socmate/blocks/{block_name}/attempt_history.json\n"
                f"- Error log: .socmate/blocks/{block_name}/previous_error.txt\n"
                f"- RTL: find the .v file for this block under rtl/\n"
                f"- uArch spec: arch/uarch_specs/{block_name}.md\n"
                f"- Block diagram: .socmate/block_diagram.json\n"
                f"- ERS: arch/ers_spec.md\n\n"
                f"Review whether the failure requires architectural changes."
            )

            system_prompt = ARCH_REVIEW_PROMPT.format(
                block_name=block_name,
                attempts="(read from attempt_history.json)",
                error_history="(read from attempt_history.json and previous_error.txt)",
            )

            run_name = f"Review Architecture [{block_title}]"
            content = await self.llm.call(
                system=system_prompt,
                prompt=user_message,
                run_name=run_name,
            )

            result = self._parse_json_response(
                content,
                default={
                    "diagnosis": f"Architecture review for {block_name}",
                    "local_fix_possible": False,
                    "interface_changes": {},
                    "affected_blocks": [],
                    "architectural_changes": [],
                },
            )

            return {
                "diagnosis": result.get("diagnosis", ""),
                "suggested_fix": json.dumps(result.get("architectural_changes", [])),
                "affected_blocks": result.get("affected_blocks", []),
                "escalate": not result.get("local_fix_possible", True),
            }

        except Exception as e:
            return {
                "diagnosis": f"Architecture review agent error: {e}",
                "suggested_fix": "Manual architecture review required",
                "affected_blocks": [],
                "escalate": True,
            }

    def _parse_json_response(
        self, content: str, default: dict
    ) -> dict[str, Any]:
        """Extract JSON from LLM response, falling back to defaults."""
        # Try to find JSON block
        json_match = re.search(r"```json\s*\n(.*?)```", content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try parsing entire response as JSON
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Fall back to defaults with whatever text we got
        result = dict(default)
        result["diagnosis"] = content[:1000]
        return result
