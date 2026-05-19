# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Local-only live architecture stress tests for derived contracts.

These tests intentionally exercise only the SocMate architecture stage. They
start the architecture graph, answer sizing questions, allow architecture
constraint iteration, accept final review, and then assert that the
deterministic derived-constraint audit is clean.

They call the configured LLM provider and are skipped unless explicitly enabled:

    SOCMATE_RUN_LOCAL_ARCH_STRESS=1 pytest \
      orchestrator/tests/test_local_architecture_contract_stress.py -v --tb=short

To run one or a subset:

    SOCMATE_RUN_LOCAL_ARCH_STRESS=1 SOCMATE_ARCH_STRESS_CASES=codec_640x360,gemm_tiles pytest ...
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path

import pytest

from orchestrator.tests.conftest import wait_for_status
from orchestrator.tests.test_live_architecture import _auto_answer_ers


RUN_STRESS = os.environ.get("SOCMATE_RUN_LOCAL_ARCH_STRESS", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


ARCH_STRESS_CASES = [
    pytest.param(
        "codec_640x360",
        """
Design a Sky130 soft IP grayscale intra-frame video encoder. Input frames are
640x360 pixels, 8 bits per pixel, row-major. The transform unit consumes 4x4
blocks exactly as implemented by examples/multiframe_codec/codec_golden.py.
The architecture must preserve a measurable contract for 160 block columns,
90 block rows, block_x/block_y ranges, total blocks per frame, byte-stream
output, and PSNR/bpp validation against that Python golden model. Row-major
pixel input may use enough line/reorder buffering to form 4x4 blocks without
dropping sustained input. Use AXI-Stream internally and no memory-mapped
registers. Freeze codec configuration as synthesis-time constants for this
run: qp=36, use_matrix=false, use_intra_pred=false, frame_type=intra, and
do_deblock=false.
""",
        id="codec_640x360",
    ),
    pytest.param(
        "padded_video_1080p",
        """
Design a streaming video preprocessor for 1920x1080 10-bit luma frames using
16x16 processing tiles. Because height is not divisible by 16, the architecture
must explicitly choose and document crop/pad policy, tile columns/rows, terminal
tile coordinate, valid-pixel mask semantics, and output transaction count.
""",
        id="padded_video_1080p",
    ),
    pytest.param(
        "gemm_tiles",
        """
Design a matrix-multiply accelerator for C[M,N] = A[M,K] * B[K,N] with
M=96, N=64, K=128. Use 16x8 output tiles and an 8-wide K step. Preserve
contracts for tile counts, tile coordinates, partial-sum ordering, SRAM banking,
and final writeback transaction count.
""",
        id="gemm_tiles",
    ),
    pytest.param(
        "transformer_block",
        """
Design one fixed-point transformer block accelerator for a TinyStories-class
model with d_model=64, n_heads=4, head_dim=16, sequence length=64, MLP hidden
size=172, INT8 activations, INT4 weights streamed from QSPI flash, and a 48 KB
on-chip SRAM budget. Preserve tensor shapes, KV-cache capacity, QSPI bandwidth
budget, and checkpoint-vector validation contracts.
""",
        id="transformer_block",
    ),
    pytest.param(
        "packet_parser",
        """
Design an Ethernet-like streaming packet parser for frames up to 1518 bytes.
The parser consumes 64-bit AXI-Stream beats with byte-valid strobes, extracts a
14-byte header, variable payload, and 4-byte CRC. Preserve contracts for header
beat crossing, payload length, byte lanes, tkeep/tlast alignment, checksum span,
and malformed packet handling.
""",
        id="packet_parser",
    ),
    pytest.param(
        "stft_audio",
        """
Design a streaming audio STFT feature extractor: 16-bit mono PCM at 48 kHz,
1024-sample Hann windows, 256-sample hop, 512-bin real FFT magnitude output.
Preserve contracts for overlap buffering, window index, frame cadence, FFT bin
count, fixed-point scaling, and output frame ordering.
""",
        id="stft_audio",
    ),
    pytest.param(
        "rs255_decoder",
        """
Design a Reed-Solomon RS(255,223) GF(256) decoder soft IP. Preserve contracts
for 255 received symbols, 32 parity symbols, syndrome count, erasure locator
capacity, Chien search order, corrected-symbol output ordering, and failure
flag behavior when errors exceed correction capacity.
""",
        id="rs255_decoder",
    ),
    pytest.param(
        "qspi_dma",
        """
Design a QSPI flash-to-stream DMA engine with 24-bit byte addresses, 256-byte
bursts, 4 data lanes, 8 dummy cycles, and backpressured AXI-Stream output.
Preserve contracts for command/address/dummy/data phase lengths, burst byte
count, alignment, underrun reporting, and transaction completion.
""",
        id="qspi_dma",
    ),
]


def _case_selected(case_id: str) -> bool:
    selected = os.environ.get("SOCMATE_ARCH_STRESS_CASES", "").strip()
    if not selected:
        return True
    return case_id in {item.strip() for item in selected.split(",") if item.strip()}


def _export_case_artifacts(case_id: str, project_root: Path, state: dict) -> None:
    artifact_root = os.environ.get("SOCMATE_STRESS_ARTIFACT_ROOT", "").strip()
    if not artifact_root:
        return

    dest = Path(artifact_root) / case_id
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "final_architecture_state.json").write_text(json.dumps(state, indent=2, default=str))

    for relpath in [
        ".socmate/derived_constraints_audit.json",
        ".socmate/block_specs.json",
        ".socmate/block_diagram.json",
        ".socmate/prd_spec.json",
        ".socmate/ers_spec.json",
        ".socmate/architecture_events.jsonl",
        ".socmate/pipeline_events.jsonl",
        "arch/sad_spec.md",
        "arch/frd_spec.md",
    ]:
        src = project_root / relpath
        if src.exists() and src.is_file():
            dst = dest / relpath
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


@pytest.fixture
async def live_arch(reset_mcp_state):
    import orchestrator.mcp_server as mcp

    yield mcp

    if mcp._architecture.task and not mcp._architecture.task.done():
        mcp._architecture.task.cancel()
        try:
            await mcp._architecture.task
        except (asyncio.CancelledError, Exception):
            pass
    await mcp._architecture.cleanup()


async def _run_architecture_only(mcp, requirements: str) -> dict:
    result = json.loads(await mcp.start_architecture(
        requirements=requirements,
        target_clock_mhz=50.0,
    ))
    assert "error" not in result, f"start_architecture failed: {result}"

    seen_constraint_violations = 0
    for _ in range(30):
        status = await wait_for_status(
            mcp._architecture,
            {"interrupted", "done", "error"},
            timeout=900,
        )
        assert status != "error", mcp._architecture.error_message

        state = json.loads(await mcp.get_architecture_state())
        if state.get("status") == "done":
            return state

        payload = state.get("interrupt_payload", {}) or {}
        itype = payload.get("type", "")
        phase = payload.get("phase", "")

        if itype in {"prd_questions", "ers_questions"}:
            answers = _auto_answer_ers(payload.get("questions", []))
            await mcp.resume_architecture("continue", json.dumps(answers))
        elif itype == "architecture_review_needed" and phase == "constraints":
            violations = (
                state.get("constraint_result", {}).get("violations", [])
                or payload.get("violations", [])
            )
            seen_constraint_violations += len(violations)
            await mcp.resume_architecture("feedback", _repair_feedback(violations))
        elif itype == "architecture_review_needed" and phase == "max_rounds_exhausted":
            violations = payload.get("violations", []) or []
            raise AssertionError(
                "architecture did not converge after constraint repair rounds: "
                + json.dumps(violations, indent=2)
            )
        elif itype == "architecture_final_review":
            await mcp.resume_architecture("accept")
        else:
            await mcp.resume_architecture("continue")

    raise AssertionError(
        f"architecture did not complete within 30 interrupt cycles; "
        f"saw {seen_constraint_violations} constraint violations"
    )


def _repair_feedback(violations: list[dict]) -> str:
    lines = [
        "Repair every constraint violation below and regenerate the block diagram.",
        "This stress test intentionally allows first-pass mistakes; success requires convergence.",
        "Do not weaken the user requirements. Fix the architecture contract or ask a blocking question.",
        "For derived geometry/count errors, recompute from the source dimensions and block/tile size.",
        "For payload-width errors, update the payload ledger so field widths sum exactly to interface and connection widths.",
        "For throughput/buffering errors, state concrete buffer depth, byte burst bound, cycle budget, and acceptance/output cadence.",
        "For variable-output blocks, justify byte/packet bounds from the golden/reference model, a conservative raw/escape rule, or a validation-DV proof obligation.",
        "If current block-diagram invariants resolve an older SAD/FRD open item, make the current handoff contract explicit and do not weaken the user requirement.",
        "For golden-model reproducibility, name the golden path and required deterministic vectors/artifacts.",
        "",
        "Violations:",
    ]
    for idx, violation in enumerate(violations, start=1):
        check = violation.get("check", "unknown")
        category = violation.get("category", "unknown")
        text = violation.get("violation", str(violation))
        lines.append(f"{idx}. [{category}/{check}] {text}")
    return "\n".join(lines)


@pytest.mark.local_arch_stress
@pytest.mark.slow
@pytest.mark.skipif(not RUN_STRESS, reason="set SOCMATE_RUN_LOCAL_ARCH_STRESS=1")
@pytest.mark.parametrize("case_id,requirements", ARCH_STRESS_CASES)
@pytest.mark.asyncio
async def test_architecture_stage_preserves_derived_contracts(case_id, requirements, live_arch):
    if not _case_selected(case_id):
        pytest.skip(f"{case_id} not selected by SOCMATE_ARCH_STRESS_CASES")

    state = await _run_architecture_only(live_arch, requirements)
    project_root = Path(state.get("project_root") or ".")
    constraint_result = state.get("constraint_result", {}) or {}
    assert constraint_result.get("all_pass", False), json.dumps(
        constraint_result.get("violations", []),
        indent=2,
    )

    audit_path = project_root / ".socmate" / "derived_constraints_audit.json"
    assert audit_path.exists(), "derived constraint audit was not written"

    audit = json.loads(audit_path.read_text())
    assert audit["violations"] == [], json.dumps(audit["violations"], indent=2)

    block_specs = project_root / ".socmate" / "block_specs.json"
    assert block_specs.exists(), "architecture did not finalize block_specs.json"

    _export_case_artifacts(case_id, project_root, state)
