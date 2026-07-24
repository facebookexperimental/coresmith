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

import re
from pathlib import Path
from typing import Any

from opentelemetry import trace

from orchestrator._timeouts import scaled
from orchestrator.langchain.prompts.skills import load_skills as _load_skills

from .coresmith_llm import ClaudeLLM

_tracer = trace.get_tracer(__name__)

_PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts" / "testbench_generator.md"
if not _PROMPT_FILE.exists():
    raise FileNotFoundError(
        f"Testbench generator prompt not found at {_PROMPT_FILE}. "
        f"This file is the single source of truth for testbench generation rules."
    )
SYSTEM_PROMPT = _PROMPT_FILE.read_text()

# Hand the testbench author the in-context verification skill: how to CHECK its
# work + PULL state via the `coresmith` CLI, and the fresh-seed gate discipline
# (a TB that only passes on a pinned seed FAILS the gate). Wired UNCONDITIONALLY
# so it is present whether or not CORESMITH_PROMPT_SLIM is set.
_VERIFY_SKILL = _load_skills("verify_in_context")
if _VERIFY_SKILL:
    SYSTEM_PROMPT = (
        SYSTEM_PROMPT
        + "\n\n# Reference Skill (verify-in-context -- self-check + pull state)"
        + "\n\n"
        + _VERIFY_SKILL
    )

# Section 5d: a block armed by a START/GO with a DONE/status must be DV'd
# back-to-back (new START soon after prior DONE, DIFFERENT config) so a
# stale-config re-trigger or an early DONE leak is caught.
_CTRL_PULSE_SKILL = _load_skills("control_pulse_handshake")
if _CTRL_PULSE_SKILL:
    SYSTEM_PROMPT = (
        SYSTEM_PROMPT
        + "\n\n# Reference Skill (control-pulse handshake -- back-to-back DV case)"
        + "\n\n"
        + _CTRL_PULSE_SKILL
    )


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
        if re.search(r"@cocotb\.test\s*\(", text):
            return text + "\n"
    return ""


class TestbenchGeneratorAgent:
    """Agent for cocotb testbench generation.

    DISK-FIRST: Tools are enabled so the agent reads RTL, golden model,
    uArch spec, and constraints from disk.  Writes testbench directly.
    Now has access to uArch spec and constraints (previously invisible).
    """

    def __init__(self, model: str | None = None, temperature: float = 0.1):
        from orchestrator.langchain.agents.coresmith_llm import block_model
        model = model or block_model()
        # 1800s default; bump via CORESMITH_TB_TIMEOUT env var for complex blocks
        # whose testbenches need many turns of tool use.
        self.llm = ClaudeLLM(
            model=model,
            timeout=scaled(1800, env="CORESMITH_TB_TIMEOUT"),
        )

    async def generate(
        self,
        block_name: str,
        rtl_path: str = "",
        python_source_path: str = "",
        testbench_path: str = "",
        project_root: str = "",
        callbacks: list = None,
        block_golden_path: str = "",
    ) -> dict[str, Any]:
        """Generate a cocotb testbench -- agent reads all files from disk.

        Args:
            block_name: Name of the block under test
            rtl_path: Path to the RTL Verilog file
            python_source_path: Relative path to Python golden model
            testbench_path: Path to write the testbench
            project_root: Project root path
            block_golden_path: (env-gated) path to this block's block-level
                golden model (arch/block_goldens/<block>.py). When set, the
                testbench should prefer it as the per-block expected-value
                oracle over the chip-wide reference implementation.

        Returns:
            Dict with keys: testbench_path (str), test_count (int)
        """
        block_title = block_name.replace("_", " ").title()
        span_name = f"Testbench Generator [{block_title}]"

        # FUNCTIONAL-QUALITY block (e.g. a rate-distortion encoder): its per-block
        # DV must NOT byte-exact-compare its output stream to one reference. Such a
        # block makes valid-but-different RD choices, and a faithful sequentialised
        # encoder can have a per-MB latency so large that a byte-exact stress test
        # over hundreds of macroblocks is computationally intractable in sim.
        # Instead we instruct the generator to build a FUNCTIONAL testbench:
        # decode the DUT's emitted output through the block model's own
        # inverse/reconstruction reference and assert a real reconstruction-quality
        # (PSNR) bound + structural validity + a sane rate bound, on a SMALL
        # stimulus. This still genuinely fails a garbage encoder.
        try:
            from orchestrator.architecture import composition as _composition
            functional = _composition.is_functional_block(block_name)
        except Exception:  # noqa: BLE001
            functional = False

        with _tracer.start_as_current_span(span_name) as span:
            span.set_attribute("block_name", block_name)
            span.set_attribute("functional_acceptance", functional)

            # When an Amaranth block model exists for this block (feature flag on),
            # it is the AUTHORITATIVE per-block oracle: it transcribes this
            # block's exact responsibility at its I/O boundary with real
            # clock/handshake/latency, validated by the model-integration gate.
            # Prefer it over the chip-wide reference implementation, which may
            # need slicing/replumbing to isolate one block's contribution.
            if block_golden_path:
                span.set_attribute("block_golden_path", block_golden_path)
                golden_line = (
                    f"- Amaranth block model (PREFERRED ORACLE): "
                    f"{block_golden_path}\n"
                    f"  It defines an Amaranth `Elaboratable` class named after this "
                    f"block (constructor (clk, rst, <handshake + data ports>)). "
                    f"Use it as the reference: simulate it (or transcribe its "
                    f"exact math) to derive the EXACT expected outputs for this "
                    f"block, INSTEAD OF the chip-wide Python golden model; the "
                    f"chip-wide model below is context only.\n"
                )
                golden_instruction = (
                    f"2. Read the Amaranth block model at {block_golden_path} and "
                    f"use its transcribed math / simulated behaviour to compute "
                    f"expected outputs (its constructor names this block's "
                    f"clock, reset, handshake, and data ports)\n"
                )
            else:
                golden_line = ""
                golden_instruction = (
                    "2. Read the golden model to understand the algorithm\n"
                )

            user_message = (
                f"Generate a cocotb testbench for the '{block_name}' Verilog module.\n\n"
                f"## Working Files\n"
                f"Read these files:\n"
                f"- RTL Verilog: {rtl_path} (use EXACT port names from this!)\n"
                f"{golden_line}"
                f"- Python Golden Model (reference implementation, chip-wide): "
                f"{python_source_path}\n"
                f"- uArch Spec: arch/uarch_specs/{block_name}.md\n"
                f"- Constraints: .coresmith/blocks/{block_name}/constraints.json\n"
                f"- DV Rules: arch/DV_RULES.md (if it exists, read and follow ALL rules)\n\n"
                f"## Output\n"
                f"Write the complete cocotb testbench to: {testbench_path}\n\n"
                f"## Instructions\n"
                f"1. Read the RTL to get EXACT port names and widths\n"
                f"{golden_instruction}"
                f"3. Read the uArch spec for timing and protocol details\n"
                f"4. Read constraints for any rules learned from prior failures\n"
                f"5. Generate a testbench that imports and uses the "
                f"{'block-level golden model' if block_golden_path else 'Python model'} "
                f"to generate expected outputs for comparison against the RTL DUT\n"
                f"6. CRITICAL: Use the EXACT signal names from the Verilog module ports\n"
                f"7. Use RisingEdge(dut.clk) for all output sampling (never Timer(0))\n"
                f"8. The pipeline will dump sim_build/{block_name}/dump.vcd "
                f"and audit it with WaveKit; exercise reset, handshakes, "
                f"datapath, metadata, and outputs with meaningful transitions\n"
            )

            if functional:
                user_message += (
                    "\n## FUNCTIONAL-QUALITY ACCEPTANCE (this block ONLY)\n"
                    "This block is a RATE-DISTORTION encoder. Do NOT byte/bit-exact "
                    "compare its emitted output stream (syntax words) against the "
                    "block model word-for-word -- a correct RD encoder may make "
                    "valid-but-different mode/quant/trellis choices, and the "
                    "faithful sequentialised RTL has a very large per-macroblock "
                    "latency so a byte-exact stress test over many macroblocks does "
                    "NOT finish in sim. Instead build an HONEST FUNCTIONAL gate:\n"
                    "  A. Use a SMALL stimulus: the reset test PLUS one functional "
                    "     test that drives ONE (or at most a FEW) macroblock(s) only "
                    "     -- enough for the DUT to emit a complete syntax record, NOT "
                    "     hundreds. Keep the total simulated time well under the sim "
                    "     timeout (the per-MB latency may be hundreds of thousands of "
                    "     cycles, so ONE MB is the right size).\n"
                    "  B. RECONSTRUCT from the DUT output: decode the DUT's emitted "
                    "     syntax word(s) back to a reconstructed macroblock using the "
                    "     block model's OWN inverse-transform / dequant / prediction "
                    "     reference helpers (import them from the block model -- e.g. "
                    "     the inverse zigzag, dequantize, idct4x4, dc_dequantize, "
                    "     pred_4x4 / pred_16x16 functions). Do the SAME reconstruction "
                    "     from the block model's reference syntax output for the same "
                    "     stimulus.\n"
                    "  C. ASSERT QUALITY, not bytes:\n"
                    "     - the DUT output must be STRUCTURALLY VALID (mb_type / "
                    "       modes in legal range; coefficient fields decodable; no "
                    "       X/Z bits);\n"
                    "     - the DUT reconstruction PSNR vs the SOURCE macroblock must "
                    "       be >= (block-model reconstruction PSNR vs the same SOURCE "
                    "       macroblock) - 1.0 dB (a small defensible RD margin); "
                    "       compute PSNR as 10*log10(255**2 / mean_squared_error) and "
                    "       treat a zero-MSE (identical) case as a pass;\n"
                    "     - the DUT's implied coded rate (e.g. total nonzero "
                    "       coefficient count, or coded bit count if available) must "
                    "       be within 1.5x of the block model's for the same MB.\n"
                    "  D. This gate MUST genuinely FAIL a broken encoder: an all-zero "
                    "     / flat / wrong-residual output reconstructs poorly and must "
                    "     trip the PSNR assertion. Do NOT weaken or delete asserts to "
                    "     force a pass, do NOT hardcode/replay expected bytes, do NOT "
                    "     build a shadow datapath. Keep the reset test's strict "
                    "     post-reset-idle assertions exactly as for any block.\n"
                    "  E. If decoding the syntax word back to a reconstruction is not "
                    "     tractable from the emitted fields alone, instead assert a "
                    "     RELATIONAL quality proxy that still rejects garbage: the "
                    "     DUT's quantised residual energy / mode decision must yield "
                    "     a distortion (SSD vs source) no worse than the block "
                    "     model's distortion + a small slack -- never an "
                    "     always-true comparison.\n"
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
            test_count = len(re.findall(r"@cocotb\.test\s*\(", tb_text))
            if test_count == 0:
                raise RuntimeError(
                    f"Testbench at {testbench_path} contains no "
                    f"@cocotb.test() functions"
                )

            return {
                "testbench_path": testbench_path,
                "test_count": test_count,
            }
