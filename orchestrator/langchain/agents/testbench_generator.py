# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
TestbenchGeneratorAgent -- Generates cocotb testbenches that co-simulate
the RTL against the Python golden model.

Strategy:
- Reads the Python golden model source
- Creates a cocotb test that instantiates the DUT
- Feeds identical stimuli to both Python model and RTL
- Compares outputs bit-exactly
- Extracts test vectors from the existing pytest suite where possible
"""

from __future__ import annotations

from orchestrator._timeouts import scaled
import re
from pathlib import Path
from typing import Any

from opentelemetry import trace

from .socmate_llm import ClaudeLLM

_tracer = trace.get_tracer(__name__)

_PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts" / "testbench_generator.md"
if not _PROMPT_FILE.exists():
    raise FileNotFoundError(
        f"Testbench generator prompt not found at {_PROMPT_FILE}. "
        f"This file is the single source of truth for testbench generation rules."
    )
SYSTEM_PROMPT = _PROMPT_FILE.read_text()


def _recover_codex_artifact(abs_path: str) -> str:
    """Recover a testbench written inside a Codex per-call workdir."""
    path = Path(abs_path)
    try:
        root = path.parents[2]
        rel = path.relative_to(root)
    except (IndexError, ValueError):
        return ""
    try:
        candidates = sorted(
            root.glob(f"codex-call-*/{rel.as_posix()}"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return ""
    for candidate in candidates:
        try:
            text = candidate.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if "@cocotb.test()" in text:
            return text + "\n"
    return ""


class TestbenchGeneratorAgent:
    """Agent for cocotb testbench generation.

    DISK-FIRST: Tools are enabled so the agent reads RTL, golden model,
    uArch spec, and constraints from disk.  Writes testbench directly.
    Now has access to uArch spec and constraints (previously invisible).
    """

    def __init__(self, model: str | None = None, temperature: float = 0.1):
        from orchestrator.langchain.agents.socmate_llm import block_model
        model = model or block_model()
        # 1800s default; bump via SOCMATE_TB_TIMEOUT env var for complex blocks
        # whose testbenches need many turns of tool use.
        self.llm = ClaudeLLM(
            model=model,
            timeout=scaled(1800, env="SOCMATE_TB_TIMEOUT"),
        )

    async def generate(
        self,
        block_name: str,
        rtl_path: str = "",
        python_source_path: str = "",
        testbench_path: str = "",
        project_root: str = "",
        callbacks: list = None,
    ) -> dict[str, Any]:
        """Generate a cocotb testbench -- agent reads all files from disk.

        Args:
            block_name: Name of the block under test
            rtl_path: Path to the RTL Verilog file
            python_source_path: Relative path to Python golden model
            testbench_path: Path to write the testbench
            project_root: Project root path

        Returns:
            Dict with keys: testbench_path (str), test_count (int)
        """
        block_title = block_name.replace("_", " ").title()
        span_name = f"Testbench Generator [{block_title}]"

        with _tracer.start_as_current_span(span_name) as span:
            span.set_attribute("block_name", block_name)

            user_message = (
                f"Generate a cocotb testbench for the '{block_name}' Verilog module.\n\n"
                f"## Working Files\n"
                f"Read these files:\n"
                f"- RTL Verilog: {rtl_path} (use EXACT port names from this!)\n"
                f"- Python Golden Model: {python_source_path}\n"
                f"- uArch Spec: arch/uarch_specs/{block_name}.md\n"
                f"- Constraints: .socmate/blocks/{block_name}/constraints.json\n"
                f"- DV Rules: arch/DV_RULES.md (if it exists, read and follow ALL rules)\n\n"
                f"## Output\n"
                f"Write the complete cocotb testbench to: {testbench_path}\n\n"
                f"## Instructions\n"
                f"1. Read the RTL to get EXACT port names and widths\n"
                f"2. Read the golden model to understand the algorithm\n"
                f"3. Read the uArch spec for timing and protocol details\n"
                f"4. Read constraints for any rules learned from prior failures\n"
                f"5. Generate a testbench that imports and uses the Python model "
                f"to generate expected outputs for comparison against the RTL DUT\n"
                f"6. CRITICAL: Use the EXACT signal names from the Verilog module ports\n"
                f"7. Use RisingEdge(dut.clk) for all output sampling (never Timer(0))\n"
                f"8. The pipeline will dump sim_build/{block_name}/dump.vcd "
                f"and audit it with WaveKit; exercise reset, handshakes, "
                f"datapath, metadata, and outputs with meaningful transitions\n"
            )

            run_name = f"Generate Testbench [{block_title}]"
            await self.llm.call(
                system=SYSTEM_PROMPT,
                prompt=user_message,
                run_name=run_name,
            )

            # ClaudeLLM.call swallows non-zero exits (returns an error
            # string instead of raising), so post-validate that the CLI
            # actually wrote a real testbench. Without this check the
            # downstream SIM step sees no file and logs the misleading
            # "Skipped -- testbench file not found", and the previous
            # max(test_count, 1) pretended the step generated 1 test.
            tb_file = Path(testbench_path) if testbench_path else None
            if not tb_file or not tb_file.exists():
                recovered = _recover_codex_artifact(testbench_path)
                if recovered and tb_file:
                    tb_file.parent.mkdir(parents=True, exist_ok=True)
                    tb_file.write_text(recovered, encoding="utf-8")
                else:
                    raise RuntimeError(
                        f"Testbench generation failed: claude CLI did not "
                        f"write {testbench_path}"
                    )
            tb_text = tb_file.read_text()
            test_count = len(re.findall(r"@cocotb\.test\(\)", tb_text))
            if test_count == 0:
                raise RuntimeError(
                    f"Testbench at {testbench_path} contains no "
                    f"@cocotb.test() functions"
                )

            return {
                "testbench_path": testbench_path,
                "test_count": test_count,
            }
