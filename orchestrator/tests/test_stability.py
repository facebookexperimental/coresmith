# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
End-to-end stability tests for the coresmith architecture pipeline.

Verifies that the full flow doesn't crash, transitions between stages
work correctly, and the FFT16 reference design passes through the
entire architecture graph and pipeline graph without errors.

Tests:
- Architecture stage-to-stage transitions (PRD -> SAD -> FRD -> BD -> MM -> CT -> RS -> CC)
- Cross-graph handoff (block_specs.json from architecture to pipeline)
- Multi-round constraint iteration loop
- Pipeline stage stability with FFT16 blocks
"""

from __future__ import annotations

import asyncio
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# These node-level stability tests construct REAL LLM agents whose patch set
# does not cover every call site (integration_dv triage, the block-diagram
# output-contract ownership review, ...) -- unmarked, they shell out to the
# live claude CLI from the "fast" suite (observed: a 50-turn Opus
# ownership-review call mid-suite; the run sat for 72+ minutes and burned
# real quota). Mark the whole module live_llm until their patches are
# extended to cover the newer agent call sites.
pytestmark = pytest.mark.live_llm

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from orchestrator.langgraph.architecture_graph import build_architecture_graph

from orchestrator.tests.fft16_fixtures import (
    FFT16_BLOCK_DIAGRAM,
    FFT16_CLOCK_TREE,
    FFT16_FRD_MARKDOWN,
    FFT16_MEMORY_MAP,
    FFT16_PRD_ANSWERS,
    FFT16_PRD_DOCUMENT,
    FFT16_PRD_QUESTIONS,
    FFT16_REGISTER_SPEC,
    FFT16_SAD_MARKDOWN,
)
from orchestrator.tests.conftest import assert_doc_files


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_all_arch_specialists(
    prd_result=None,
    sad_result=None,
    frd_result=None,
    block_diagram_result=None,
    memory_map_result=None,
    clock_tree_result=None,
    register_spec_result=None,
    constraint_result=None,
):
    """Patch all architecture specialists at the node level.

    Patches the lazy imports inside architecture_graph.py node functions
    so the underlying specialist modules are never imported (avoids
    import-time side effects like reading prompt files from disk).
    """
    stack = ExitStack()

    stack.enter_context(patch(
        "orchestrator.architecture.specialists.prd_spec.gather_prd",
        new_callable=AsyncMock,
        return_value=prd_result or FFT16_PRD_QUESTIONS,
    ))
    stack.enter_context(patch(
        "orchestrator.architecture.specialists.sad_spec.generate_sad",
        new_callable=AsyncMock,
        return_value=sad_result or FFT16_SAD_MARKDOWN,
    ))
    stack.enter_context(patch(
        "orchestrator.architecture.specialists.frd_spec.generate_frd",
        new_callable=AsyncMock,
        return_value=frd_result or FFT16_FRD_MARKDOWN,
    ))
    stack.enter_context(patch(
        "orchestrator.architecture.specialists.block_diagram.analyze_block_diagram",
        new_callable=AsyncMock,
        return_value=block_diagram_result or FFT16_BLOCK_DIAGRAM,
    ))
    stack.enter_context(patch(
        "orchestrator.architecture.specialists.memory_map.analyze_memory_map",
        new_callable=AsyncMock,
        return_value=memory_map_result or FFT16_MEMORY_MAP,
    ))
    stack.enter_context(patch(
        "orchestrator.architecture.specialists.clock_tree.analyze_clock_tree",
        new_callable=AsyncMock,
        return_value=clock_tree_result or FFT16_CLOCK_TREE,
    ))
    stack.enter_context(patch(
        "orchestrator.architecture.specialists.register_spec.analyze_register_spec",
        new_callable=AsyncMock,
        return_value=register_spec_result or FFT16_REGISTER_SPEC,
    ))
    stack.enter_context(patch(
        "orchestrator.architecture.constraints.check_constraints",
        new_callable=AsyncMock,
        return_value=constraint_result if constraint_result is not None else [],
    ))
    stack.enter_context(patch(
        "orchestrator.architecture.specialists.ers_doc.generate_ers_doc",
        new_callable=AsyncMock,
        return_value={"ers": {"title": "FFT16 ERS", "summary": "test"}, "phase": "ers_complete"},
    ))
    return stack


async def _get_interrupt(graph, config) -> dict | None:
    """Get the interrupt payload from the graph state, or None."""
    state = await graph.aget_state(config)
    if state.tasks:
        for task in state.tasks:
            if task.interrupts:
                return task.interrupts[0].value
    return None


async def _accept_final_review(graph, config) -> dict:
    """Auto-accept the Final Review interrupt and return the final state."""
    payload = await _get_interrupt(graph, config)
    if payload and payload.get("type") == "final_review":
        return await graph.ainvoke(
            Command(resume={"action": "accept"}),
            config,
        )
    state = await graph.aget_state(config)
    return state.values if state else {}


# ═══════════════════════════════════════════════════════════════════════════
# 8.1 Architecture Stage Stability (FFT16 design)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.stability
class TestArchitectureStageTransitions:
    """Verify each stage transition completes without error."""

    @pytest.mark.asyncio
    async def test_prd_to_block_diagram_transition(self, arch_graph, fft16_initial_state):
        """PRD Phase 2 -> SAD -> FRD -> Block Diagram (no crash)."""
        config = {"configurable": {"thread_id": "stab-prd-bd-1"}}

        with _patch_all_arch_specialists(prd_result=FFT16_PRD_QUESTIONS):
            await arch_graph.ainvoke(fft16_initial_state, config)

        with _patch_all_arch_specialists(
            prd_result=FFT16_PRD_DOCUMENT,
            constraint_result=[],
        ):
            await asyncio.wait_for(
                arch_graph.ainvoke(
                    Command(resume={"action": "continue", "answers": FFT16_PRD_ANSWERS}),
                    config,
                ),
                timeout=180,
            )
            result = await _accept_final_review(arch_graph, config)

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_full_architecture_flow_no_crash(self, arch_graph, fft16_initial_state):
        """Full flow: PRD -> SAD -> FRD -> BD -> MM -> CT -> RS -> CC -> Finalize -> Final Review.

        Verifies all specialist stages run without error on the FFT16 design.
        """
        config = {"configurable": {"thread_id": "stab-full-flow-1"}}

        with _patch_all_arch_specialists(prd_result=FFT16_PRD_QUESTIONS):
            await arch_graph.ainvoke(fft16_initial_state, config)

        with _patch_all_arch_specialists(
            prd_result=FFT16_PRD_DOCUMENT,
            constraint_result=[],
        ):
            await asyncio.wait_for(
                arch_graph.ainvoke(
                    Command(resume={"action": "continue", "answers": FFT16_PRD_ANSWERS}),
                    config,
                ),
                timeout=180,
            )
            result = await _accept_final_review(arch_graph, config)

        assert result["success"] is True
        assert result["error"] == ""
        assert result["block_specs_path"] != ""

    @pytest.mark.asyncio
    async def test_finalize_writes_block_specs(self, arch_graph, fft16_initial_state):
        """Verify finalize node writes block_specs.json to the isolated project."""
        config = {"configurable": {"thread_id": "stab-finalize-1"}}

        with _patch_all_arch_specialists(prd_result=FFT16_PRD_QUESTIONS):
            await arch_graph.ainvoke(fft16_initial_state, config)

        with _patch_all_arch_specialists(
            prd_result=FFT16_PRD_DOCUMENT,
            constraint_result=[],
        ):
            await arch_graph.ainvoke(
                Command(resume={"action": "continue", "answers": FFT16_PRD_ANSWERS}),
                config,
            )
            result = await _accept_final_review(arch_graph, config)

        specs_path = Path(result["block_specs_path"])
        assert specs_path.exists()
        specs = json.loads(specs_path.read_text())
        assert len(specs) == 3
        assert {s["name"] for s in specs} == {"fft_butterfly", "twiddle_rom", "fft_controller"}


# ═══════════════════════════════════════════════════════════════════════════
# 8.2 Cross-Graph Handoff
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.stability
class TestCrossGraphHandoff:
    def test_block_spec_schema_compatibility(self):
        """Verify FFT16 block specs have fields the pipeline graph expects."""
        for block in FFT16_BLOCK_DIAGRAM["blocks"]:
            assert "name" in block
            assert "tier" in block
            assert "python_source" in block
            assert "rtl_target" in block
            assert "testbench" in block
            assert "description" in block

    @pytest.mark.asyncio
    async def test_pipeline_init_block_accepts_fft16_spec(self):
        """Verify pipeline init_block_node can consume an FFT16 block spec."""
        from orchestrator.langgraph.pipeline_graph import init_block_node

        block_specs = [
            {
                "name": b["name"],
                "tier": b["tier"],
                "python_source": b["python_source"],
                "rtl_target": b["rtl_target"],
                "testbench": b["testbench"],
                "description": b["description"],
            }
            for b in FFT16_BLOCK_DIAGRAM["blocks"]
        ]

        state = {
            "project_root": "/tmp/test",
            "target_clock_mhz": 50.0,
            "max_attempts": 3,
            "current_block": block_specs[0],
            "attempt": 1,
            "phase": "init",
            "constraints": [],
            "attempt_history": [],
            "previous_error": "",
            "uarch_spec": None,
            "uarch_approved": False,
            "uarch_feedback": "",
            "rtl_result": None,
            "lint_result": None,
            "tb_result": None,
            "sim_result": None,
            "synth_result": None,
            "debug_result": None,
            "completed_blocks": [],
            "human_response": None,
        }

        result = await init_block_node(state)
        assert result["attempt"] == 1


# ═══════════════════════════════════════════════════════════════════════════
# 8.3 Multi-Round Stability
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.stability
class TestMultiRoundStability:
    @pytest.mark.asyncio
    async def test_constraint_auto_fix_loop_completes(self, fft16_initial_state):
        """max_rounds=3: fail rounds 1-2 (auto-fixable), pass round 3."""
        graph = build_architecture_graph(checkpointer=MemorySaver())
        fft16_initial_state["max_rounds"] = 3
        config = {"configurable": {"thread_id": "stab-multiround-1"}}

        call_count = {"n": 0}

        async def constraint_side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                return [{"violation": "Gate budget exceeded", "severity": "warning", "category": "auto_fixable"}]
            return []

        with _patch_all_arch_specialists(prd_result=FFT16_PRD_QUESTIONS):
            await graph.ainvoke(fft16_initial_state, config)

        with _patch_all_arch_specialists(prd_result=FFT16_PRD_DOCUMENT) as stack:
            stack.enter_context(patch(
                "orchestrator.architecture.constraints.check_constraints",
                new_callable=AsyncMock,
                side_effect=constraint_side_effect,
            ))
            await asyncio.wait_for(
                graph.ainvoke(
                    Command(resume={"action": "continue", "answers": FFT16_PRD_ANSWERS}),
                    config,
                ),
                timeout=180,
            )
            result = await _accept_final_review(graph, config)

        assert result["success"] is True
        assert call_count["n"] == 3

    @pytest.mark.asyncio
    async def test_constraint_loop_increments_round(self, fft16_initial_state):
        """Verify round counter increments through iterations."""
        graph = build_architecture_graph(checkpointer=MemorySaver())
        fft16_initial_state["max_rounds"] = 3
        config = {"configurable": {"thread_id": "stab-round-inc-1"}}



        async def bd_side_effect(*args, **kwargs):
            return FFT16_BLOCK_DIAGRAM

        call_count = {"n": 0}

        async def constraint_side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                return [{"violation": "Gate exceeded", "severity": "warning", "category": "auto_fixable"}]
            return []

        with _patch_all_arch_specialists(prd_result=FFT16_PRD_QUESTIONS):
            await graph.ainvoke(fft16_initial_state, config)

        with _patch_all_arch_specialists(prd_result=FFT16_PRD_DOCUMENT) as stack:
            stack.enter_context(patch(
                "orchestrator.architecture.constraints.check_constraints",
                new_callable=AsyncMock,
                side_effect=constraint_side_effect,
            ))
            await graph.ainvoke(
                Command(resume={"action": "continue", "answers": FFT16_PRD_ANSWERS}),
                config,
            )
            result = await _accept_final_review(graph, config)

        assert result["success"] is True
        assert result["round"] >= 3


# ═══════════════════════════════════════════════════════════════════════════
# 8.5 Pipeline Stage Stability (FFT16 blocks)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.stability
class TestPipelineStability:
    def _make_pipeline_state(self, blocks):
        return {
            "project_root": "/tmp/test",
            "target_clock_mhz": 50.0,
            "max_attempts": 3,
            "block_queue": blocks,
            "tier_list": [],
            "current_tier_index": 0,
            "completed_blocks": [],
            "pipeline_done": False,
        }

    def _fft16_pipeline_blocks(self):
        return [
            {
                "name": b["name"],
                "tier": b["tier"],
                "python_source": b["python_source"],
                "rtl_target": b["rtl_target"],
                "testbench": b["testbench"],
                "description": b["description"],
            }
            for b in FFT16_BLOCK_DIAGRAM["blocks"]
        ]

    def _pipeline_patches(self, tmp_dir, block_name="fft_butterfly"):
        """Return patch context managers for all pipeline helpers.

        Creates real RTL/TB files under *tmp_dir* so node-level file
        existence checks pass without a global Path.exists patch.
        """
        stack = ExitStack()
        _mod = "orchestrator.langgraph.pipeline_graph"

        (tmp_dir / ".coresmith").mkdir(exist_ok=True)

        def _make_rtl(block, *a, **kw):
            name = block["name"]
            rtl_file = tmp_dir / "rtl" / "dvbt" / f"{name}.v"
            rtl_file.parent.mkdir(parents=True, exist_ok=True)
            rtl_file.write_text(f"module {name}(); endmodule\n")
            return {"rtl_path": str(rtl_file), "ports": {"clk": "input"}}

        def _make_tb(block, *a, **kw):
            name = block["name"]
            tb_file = tmp_dir / "tb" / "cocotb" / f"test_{name}.py"
            tb_file.parent.mkdir(parents=True, exist_ok=True)
            tb_file.write_text("# test\n")
            return {"testbench": "# test", "testbench_path": str(tb_file)}

        uarch = {
            "spec_text": f"## Spec for {block_name}",
            "spec_summary": {"block_name": block_name, "latency_cycles": 1},
            "spec_path": str(tmp_dir / "arch" / "uarch_specs" / f"{block_name}.md"),
            "block_name": block_name,
        }
        lint_ok = {"clean": True, "warnings": ""}
        sim_pass = {"passed": True, "log": "all tests passed", "returncode": 0}
        synth_ok = {"success": True, "gate_count": 2000, "netlist_path": "/tmp/net.v",
                    "sdc_path": "/tmp/f.sdc", "log": ""}

        stack.enter_context(patch(f"{_mod}.generate_uarch_spec", new_callable=AsyncMock,
                                  side_effect=lambda block, **kw: {**uarch, "block_name": block["name"]}))
        stack.enter_context(patch(f"{_mod}.generate_rtl", new_callable=AsyncMock,
                                  side_effect=_make_rtl))
        stack.enter_context(patch(f"{_mod}.lint_rtl", return_value=lint_ok))
        stack.enter_context(patch(f"{_mod}.generate_testbench", new_callable=AsyncMock,
                                  side_effect=_make_tb))
        stack.enter_context(patch(f"{_mod}.run_simulation", return_value=sim_pass))
        stack.enter_context(patch(f"{_mod}.synthesize_block", return_value=synth_ok))
        stack.enter_context(patch(f"{_mod}.create_golden_model_wrapper"))

        mock_review = AsyncMock(return_value={
            "summary": "No issues found.",
            "issues_found": 0,
            "issues_fixed": 0,
        })
        mock_agent_cls = MagicMock()
        mock_agent_cls.return_value.review = mock_review
        stack.enter_context(patch(
            "orchestrator.langchain.agents.integration_review_agent.IntegrationReviewAgent",
            mock_agent_cls,
        ))

        return stack

    @pytest.mark.asyncio
    async def test_pipeline_single_fft_block_no_crash(self, pipeline_graph, tmp_path):
        """Run fft_butterfly through the full pipeline with mocked helpers.

        review_uarch_spec auto-approves. integration_review fires an
        interrupt with the mocked agent result -- we auto-approve it.
        """
        config = {"configurable": {"thread_id": "stab-pipeline-1"}}
        blocks = self._fft16_pipeline_blocks()[:1]
        state = self._make_pipeline_state(blocks)
        state["project_root"] = str(tmp_path)

        with self._pipeline_patches(tmp_path, "fft_butterfly"):
            current_input = state
            for _ in range(10):
                result = await pipeline_graph.ainvoke(current_input, config)
                if result.get("pipeline_done"):
                    break
                snap = await pipeline_graph.aget_state(config)
                interrupts = []
                if snap and snap.tasks:
                    for task in snap.tasks:
                        for intr in task.interrupts:
                            interrupts.append(intr.id)
                if not interrupts:
                    break
                resume_value = {"action": "approve"}
                if len(interrupts) > 1:
                    current_input = Command(
                        resume={iid: resume_value for iid in interrupts}
                    )
                else:
                    current_input = Command(resume=resume_value)

        assert result["pipeline_done"] is True
        assert len(result["completed_blocks"]) == 1
        assert result["completed_blocks"][0]["success"] is True

    @pytest.mark.asyncio
    async def test_pipeline_three_fft_blocks_no_crash(self, pipeline_graph, tmp_path):
        """Run all 3 FFT16 blocks through the pipeline with mocked helpers.

        review_uarch_spec auto-approves. integration_review fires after
        each tier -- we auto-approve all interrupts.
        """
        config = {"configurable": {"thread_id": "stab-pipeline-3"}}
        blocks = self._fft16_pipeline_blocks()
        state = self._make_pipeline_state(blocks)
        state["project_root"] = str(tmp_path)

        with self._pipeline_patches(tmp_path):
            current_input = state
            for _ in range(20):
                result = await pipeline_graph.ainvoke(current_input, config)
                if result.get("pipeline_done"):
                    break
                snap = await pipeline_graph.aget_state(config)
                interrupts = []
                if snap and snap.tasks:
                    for task in snap.tasks:
                        for intr in task.interrupts:
                            interrupts.append(intr.id)
                if not interrupts:
                    break
                resume_value = {"action": "approve"}
                if len(interrupts) > 1:
                    current_input = Command(
                        resume={iid: resume_value for iid in interrupts}
                    )
                else:
                    current_input = Command(resume=resume_value)

        assert result["pipeline_done"] is True
        assert len(result["completed_blocks"]) == 3
        assert all(b["success"] for b in result["completed_blocks"])


# ═══════════════════════════════════════════════════════════════════════════
# 8.6 Document Set Completeness (post-refactor)
#
# These tests verify that after the full architecture flow, all per-document
# .md/.json files are emitted and architecture_state.json is NOT written.
# They will pass only after the PRD->SAD->FRD refactor is complete.
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.stability
class TestDocumentSetCompleteness:
    """Verify that the full architecture flow emits all per-document files."""

    def _patch_all_with_sad_frd(
        self,
        prd_result=None,
        sad_result=None,
        frd_result=None,
        block_diagram_result=None,
        memory_map_result=None,
        clock_tree_result=None,
        register_spec_result=None,
        constraint_result=None,
    ):
        """Patch all specialists including the new SAD and FRD nodes."""
        stack = ExitStack()

        stack.enter_context(patch(
            "orchestrator.architecture.specialists.prd_spec.gather_prd",
            new_callable=AsyncMock,
            return_value=prd_result or FFT16_PRD_QUESTIONS,
        ))
        stack.enter_context(patch(
            "orchestrator.architecture.specialists.sad_spec.generate_sad",
            new_callable=AsyncMock,
            return_value=sad_result or FFT16_SAD_MARKDOWN,
        ))
        stack.enter_context(patch(
            "orchestrator.architecture.specialists.frd_spec.generate_frd",
            new_callable=AsyncMock,
            return_value=frd_result or FFT16_FRD_MARKDOWN,
        ))
        stack.enter_context(patch(
            "orchestrator.architecture.specialists.block_diagram.analyze_block_diagram",
            new_callable=AsyncMock,
            return_value=block_diagram_result or FFT16_BLOCK_DIAGRAM,
        ))
        stack.enter_context(patch(
            "orchestrator.architecture.specialists.memory_map.analyze_memory_map",
            new_callable=AsyncMock,
            return_value=memory_map_result or FFT16_MEMORY_MAP,
        ))
        stack.enter_context(patch(
            "orchestrator.architecture.specialists.clock_tree.analyze_clock_tree",
            new_callable=AsyncMock,
            return_value=clock_tree_result or FFT16_CLOCK_TREE,
        ))
        stack.enter_context(patch(
            "orchestrator.architecture.specialists.register_spec.analyze_register_spec",
            new_callable=AsyncMock,
            return_value=register_spec_result or FFT16_REGISTER_SPEC,
        ))
        stack.enter_context(patch(
            "orchestrator.architecture.constraints.check_constraints",
            new_callable=AsyncMock,
            return_value=constraint_result if constraint_result is not None else [],
        ))
        stack.enter_context(patch(
            "orchestrator.architecture.specialists.ers_doc.generate_ers_doc",
            new_callable=AsyncMock,
            return_value={"ers": {"title": "FFT16 ERS", "summary": "test"}, "phase": "ers_complete"},
        ))
        return stack

    @pytest.mark.asyncio
    async def test_full_flow_emits_all_document_files(self, fft16_initial_state):
        """After OK2DEV approval, all 8 document .md/.json pairs should exist.

        Flow: PRD -> SAD -> FRD -> Block Diagram -> Memory Map -> Clock Tree
              -> Register Spec -> Constraints (pass) -> Finalize -> Documentation
        """
        graph = build_architecture_graph(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "stab-doc-completeness-1"}}

        with self._patch_all_with_sad_frd(prd_result=FFT16_PRD_QUESTIONS):
            await graph.ainvoke(fft16_initial_state, config)

        with self._patch_all_with_sad_frd(
            prd_result=FFT16_PRD_DOCUMENT,
            constraint_result=[],
        ):
            await asyncio.wait_for(
                graph.ainvoke(
                    Command(resume={"action": "continue", "answers": FFT16_PRD_ANSWERS}),
                    config,
                ),
                timeout=180,
            )
            result = await _accept_final_review(graph, config)

        assert result["success"] is True

        project_root = fft16_initial_state["project_root"]
        assert_doc_files(project_root, [
            "prd_spec", "sad_spec", "frd_spec",
            "block_diagram", "memory_map", "clock_tree",
            "register_spec", "ers_spec",
        ])

        assert (Path(project_root) / ".coresmith" / "block_specs.json").exists()

    @pytest.mark.asyncio
    async def test_no_architecture_state_json_after_full_flow(self, fft16_initial_state):
        """architecture_state.json should NOT exist after the full flow."""
        graph = build_architecture_graph(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "stab-no-arch-state-1"}}

        with self._patch_all_with_sad_frd(prd_result=FFT16_PRD_QUESTIONS):
            await graph.ainvoke(fft16_initial_state, config)

        with self._patch_all_with_sad_frd(
            prd_result=FFT16_PRD_DOCUMENT,
            constraint_result=[],
        ):
            await asyncio.wait_for(
                graph.ainvoke(
                    Command(resume={"action": "continue", "answers": FFT16_PRD_ANSWERS}),
                    config,
                ),
                timeout=180,
            )
            result = await _accept_final_review(graph, config)

        assert result["success"] is True

        project_root = fft16_initial_state["project_root"]
        arch_state = Path(project_root) / ".coresmith" / "architecture_state.json"
        if arch_state.exists():
            pytest.xfail(
                "architecture_state.json still written "
                "(expected until per-doc migration is complete)"
            )
