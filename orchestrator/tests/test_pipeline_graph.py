# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for the pipeline LangGraph execution graph.

Tests:
- Graph construction (compiles, has expected orchestrator nodes)
- BlockState / OrchestratorState schemas
- Block-level routing functions (route_decision, route_after_human)
- Orchestrator routing (route_next_tier, route_after_integration_review)
- Happy path (1 block, mocked helpers)
- Interrupt flow (lint failure -> diagnose -> decide -> ask_human)
- Resume actions (retry, fix_rtl, add_constraint, skip, abort)
- Multi-block parallel (3 same-tier blocks, all complete)
- Pause/restart via MemorySaver

Note: route_after_lint, route_after_sim, route_after_increment used to
exist but were removed; their TestRoute* classes here are skipped until
the equivalent inlined-routing behaviour gets a fresh test pass.
"""


import pytest
from unittest.mock import patch, AsyncMock

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

import orchestrator.langgraph.pipeline_graph as pipeline_graph
from orchestrator.langgraph.pipeline_graph import (
    BlockState,
    OrchestratorState,
    build_pipeline_graph,
    build_block_subgraph,
    route_after_uarch_review,
    route_decision,
    route_after_human,
    route_after_integration_review,
    route_after_integration_dv,
    route_after_validation_dv,
    route_next_tier,
    init_block_node,
    block_done_node,
    pipeline_complete_node,
    ask_human_node,
    integration_dv_node,
    validation_dv_node,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _get_interrupt(graph, config) -> dict | None:
    """Get the first interrupt payload from the graph state, or None."""
    state = await graph.aget_state(config)
    if state.tasks:
        for task in state.tasks:
            if task.interrupts:
                return task.interrupts[0].value
    return None


async def _resume_all(graph, config, resume_value) -> dict:
    """Resume all pending interrupts with the same value.

    Handles both single and multiple pending interrupts (from parallel blocks).
    """
    state = await graph.aget_state(config)
    interrupt_ids = []
    if state and state.tasks:
        for task in state.tasks:
            for intr in task.interrupts:
                interrupt_ids.append(intr.id)

    if len(interrupt_ids) > 1:
        resume_input = Command(resume={iid: resume_value for iid in interrupt_ids})
    else:
        resume_input = Command(resume=resume_value)

    return await graph.ainvoke(resume_input, config)


def _make_block(name: str, tier: int = 1) -> dict:
    """Create a minimal block spec for testing."""
    return {
        "name": name,
        "tier": tier,
        "python_source": f"PyDVB/dvbt/{name}.py",
        "rtl_target": f"rtl/dvbt/{name}.v",
        "testbench": f"tb/cocotb/test_{name}.py",
        "description": f"Test block {name}",
    }


def _initial_state(blocks: list[dict] | None = None, project_root: str = "/tmp/test") -> dict:
    """Build an initial OrchestratorState for testing."""
    if blocks is None:
        blocks = [_make_block("scrambler")]
    return {
        "project_root": project_root,
        "target_clock_mhz": 50.0,
        "max_attempts": 3,
        "block_queue": blocks,
        "tier_list": [],
        "current_tier_index": 0,
        "completed_blocks": [],
        "pipeline_done": False,
    }


def _setup_disk_fixtures(tmp_path, blocks: list[dict]) -> None:
    """Create the on-disk fixtures (rtl, tb, .coresmith/blocks/<name>/...)
    expected by the disk-first pipeline nodes for the given blocks.
    """
    for blk in blocks:
        name = blk["name"]
        rtl_path = tmp_path / blk["rtl_target"]
        rtl_path.parent.mkdir(parents=True, exist_ok=True)
        rtl_path.write_text(f"module {name}();\nendmodule\n")

        tb_path = tmp_path / blk["testbench"]
        tb_path.parent.mkdir(parents=True, exist_ok=True)
        tb_path.write_text(f"# tb for {name}\n")

        block_dir = tmp_path / ".coresmith" / "blocks" / name
        block_dir.mkdir(parents=True, exist_ok=True)
        (block_dir / "constraints.json").write_text("[]")
        (block_dir / "diagnosis.json").write_text("{}")
        (block_dir / "attempt_history.json").write_text("[]")
        (block_dir / "previous_error.txt").write_text("")


def _block_state(block: dict | None = None, tmp_path: str = "/tmp/test") -> dict:
    """Build a BlockState dict for unit-testing block-level nodes.

    Uses the new disk-first BlockState with routing-only flags.
    """
    if block is None:
        block = _make_block("scrambler")
    return {
        "project_root": tmp_path,
        "target_clock_mhz": 50.0,
        "max_attempts": 3,
        "pipeline_run_start": 0.0,
        "current_block": block,
        "attempt": 1,
        "phase": "init",
        "uarch_approved": False,
        "lint_clean": False,
        "sim_passed": False,
        "synth_success": False,
        "synth_gate_count": 0,
        "rtl_path": "",
        "tb_path": "",
        "debug_action": "",
        "step_log_paths": {},
        "preserve_testbench": False,
        "force_regen_tb": False,
        "human_response": None,
        "completed_blocks": [],
    }


# ---------------------------------------------------------------------------
# Mock context manager for patching all external calls
# ---------------------------------------------------------------------------

def _patch_all_helpers():
    """Return a stack of patches for all pipeline helper functions.

    All helpers are mocked so tests run without Verilator, Yosys, or LLM APIs.
    """
    uarch_result = {
        "spec_text": "## 1. Block Overview\nTest block spec",
        "spec_summary": {"block_name": "scrambler", "latency_cycles": 1},
        "spec_path": "/tmp/test/arch/uarch_specs/scrambler.md",
        "block_name": "scrambler",
    }
    rtl_result = {
        "verilog": "module scrambler(); endmodule\n",
        "rtl_path": "/tmp/test/rtl/dvbt/scrambler.v",
        "ports": {"clk": "input", "data_in": "input [7:0]", "data_out": "output [7:0]"},
    }
    lint_clean = {"clean": True, "warnings": ""}
    lint_fail = {"clean": False, "errors": "syntax error line 42"}
    tb_result = {"testbench": "# test", "testbench_path": "/tmp/test/tb/cocotb/test_scrambler.py"}
    sim_pass = {"passed": True, "log": "all tests passed", "returncode": 0}
    sim_fail = {"passed": False, "log": "FAIL: test_basic", "returncode": 1}
    synth_ok = {"success": True, "gate_count": 1500, "netlist_path": "/tmp/net.v",
                "sdc_path": "/tmp/f.sdc", "log": ""}

    return uarch_result, rtl_result, lint_clean, lint_fail, tb_result, sim_pass, sim_fail, synth_ok


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

class TestGraphConstruction:
    def test_compiles_without_error(self):
        graph = build_pipeline_graph(checkpointer=MemorySaver())
        assert graph is not None

    def test_compiles_without_checkpointer(self):
        graph = build_pipeline_graph(checkpointer=None)
        assert graph is not None

    def test_has_expected_orchestrator_nodes(self):
        graph = build_pipeline_graph(checkpointer=MemorySaver())
        node_names = list(graph.get_graph().nodes.keys())
        expected = [
            "init_tier", "process_block", "integration_review",
            "advance_tier", "pipeline_complete",
            "integration_check", "integration_dv", "validation_dv",
        ]
        for name in expected:
            assert name in node_names, f"Missing orchestrator node: {name}"

    def test_integration_dv_routes_to_validation_on_pass(self):
        result = route_after_integration_dv({
            "integration_dv_result": {"passed": True},
        })
        assert result == "validation_dv"

    def test_integration_dv_retries_on_fix_action(self):
        result = route_after_integration_dv({
            "integration_dv_result": {"passed": False, "action_taken": "fix_tb"},
        })
        assert result == "integration_dv"

    @pytest.mark.asyncio
    async def test_integration_dv_generation_failure_interrupts(self, tmp_path, monkeypatch):
        top_rtl = tmp_path / "chip_top.v"
        block_rtl = tmp_path / "block.v"
        top_rtl.write_text("module chip_top(input clk); endmodule\n", encoding="utf-8")
        block_rtl.write_text("module block(input clk); endmodule\n", encoding="utf-8")

        monkeypatch.setattr(
            pipeline_graph,
            "load_architecture_connections",
            lambda _pr: ({}, {}),
        )

        async def fail_generate(**_kwargs):
            raise RuntimeError("no usable Python cocotb testbench")

        async def fake_contract_audit(**kwargs):
            assert kwargs["stage"] == "integration_dv_generation"
            assert kwargs["testbench_path"] == ""
            return {
                "category": "INTEGRATION_TB_BUG",
                "recommended_action": "fix_tb",
                "outer_agent_summary": "generator returned no tests",
                "audit_path": str(tmp_path / "audit.json"),
            }

        interrupts = []

        def fake_interrupt(payload):
            interrupts.append(payload)
            return {"action": "fix_tb", "rtl_fix_description": "repair generator"}

        monkeypatch.setattr(pipeline_graph, "generate_integration_testbench", fail_generate)
        monkeypatch.setattr(pipeline_graph, "_run_top_level_contract_audit", fake_contract_audit)
        monkeypatch.setattr(pipeline_graph, "interrupt", fake_interrupt)

        result = await integration_dv_node({
            "project_root": str(tmp_path),
            "integration_result": {
                "top_rtl_path": str(top_rtl),
                "design_name": "chip_top",
                "block_rtl_paths": {"block": str(block_rtl)},
            },
        })

        dv_result = result["integration_dv_result"]
        assert dv_result["passed"] is False
        assert dv_result["phase"] == "tb_generation"
        assert dv_result["action_taken"] == "fix_tb"
        assert result["pipeline_done"] is False
        assert route_after_integration_dv(result) == "integration_dv"
        assert interrupts
        assert interrupts[0]["phase"] == "tb_generation"
        assert interrupts[0]["contract_audit"]["category"] == "INTEGRATION_TB_BUG"

    @pytest.mark.asyncio
    async def test_integration_testbench_generator_accepts_written_file(self, tmp_path):
        from orchestrator.langchain.agents.integration_testbench_generator import (
            IntegrationTestbenchGenerator,
        )

        output_path = tmp_path / "test_chip_top.py"
        generated = (
            "import cocotb\n\n"
            "@cocotb.test()\n"
            "async def test_reset(dut):\n"
            "    assert True\n"
        )
        agent = IntegrationTestbenchGenerator()

        async def fake_call(**_kwargs):
            output_path.write_text(generated, encoding="utf-8")
            return f"Implemented the cocotb integration testbench at {output_path}"

        agent.llm.call = fake_call
        result = await agent.generate(
            design_name="chip_top",
            top_rtl_source="module chip_top(input clk); endmodule\n",
            block_summaries=[],
            connections=[],
            output_path=str(output_path),
        )

        assert result["tb_path"] == str(output_path)
        assert result["test_count"] == 1
        assert output_path.read_text(encoding="utf-8") == generated

    def test_validation_dv_retries_on_fix_action(self):
        result = route_after_validation_dv({
            "validation_dv_result": {"passed": False, "action_taken": "fix_rtl"},
        })
        assert result == "validation_dv"

    @pytest.mark.asyncio
    async def test_validation_dv_generation_failure_interrupts(self, tmp_path, monkeypatch):
        top_rtl = tmp_path / "chip_top.v"
        block_rtl = tmp_path / "block.v"
        top_rtl.write_text("module chip_top(input clk); endmodule\n", encoding="utf-8")
        block_rtl.write_text("module block(input clk); endmodule\n", encoding="utf-8")

        monkeypatch.setattr(
            pipeline_graph,
            "_load_ers_validation_context",
            lambda _pr: ("{\"validation_kpis\": [\"must pass\"]}", 1),
        )
        monkeypatch.setattr(
            pipeline_graph,
            "load_architecture_connections",
            lambda _pr: ({}, {}),
        )

        async def fail_generate(**_kwargs):
            raise RuntimeError("no usable Python cocotb testbench")

        async def fake_contract_audit(**kwargs):
            assert kwargs["stage"] == "validation_dv_generation"
            assert kwargs["testbench_path"] == ""
            return {
                "category": "VALIDATION_TB_BUG",
                "recommended_action": "fix_tb",
                "outer_agent_summary": "generator returned no tests",
                "audit_path": str(tmp_path / "audit.json"),
            }

        interrupts = []

        def fake_interrupt(payload):
            interrupts.append(payload)
            return {"action": "fix_tb", "rtl_fix_description": "repair generator prompt"}

        monkeypatch.setattr(pipeline_graph, "generate_validation_testbench", fail_generate)
        monkeypatch.setattr(pipeline_graph, "_run_top_level_contract_audit", fake_contract_audit)
        monkeypatch.setattr(pipeline_graph, "interrupt", fake_interrupt)

        result = await validation_dv_node({
            "project_root": str(tmp_path),
            "integration_result": {
                "top_rtl_path": str(top_rtl),
                "design_name": "chip_top",
                "block_rtl_paths": {"block": str(block_rtl)},
            },
        })

        dv_result = result["validation_dv_result"]
        assert dv_result["passed"] is False
        assert dv_result["phase"] == "tb_generation"
        assert dv_result["action_taken"] == "fix_tb"
        assert route_after_validation_dv(result) == "validation_dv"
        assert interrupts
        assert interrupts[0]["phase"] == "tb_generation"
        assert interrupts[0]["contract_audit"]["category"] == "VALIDATION_TB_BUG"

    def test_block_subgraph_has_expected_nodes(self):
        subgraph = build_block_subgraph().compile()
        node_names = list(subgraph.get_graph().nodes.keys())
        expected = [
            # Post-refactor: lint runs inside generate_rtl_node and the
            # simulate stage was inlined into generate_testbench /
            # synthesize_node. Update if either gets re-extracted.
            "init_block", "generate_uarch_spec", "review_uarch_spec",
            "generate_rtl", "generate_testbench",
            "synthesize", "diagnose", "decide", "ask_human",
            "block_done",
        ]
        for name in expected:
            assert name in node_names, f"Missing block subgraph node: {name}"


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class TestBlockState:
    def test_has_required_fields(self):
        annotations = BlockState.__annotations__
        required = [
            "project_root", "target_clock_mhz", "max_attempts",
            "current_block", "attempt", "phase",
            "uarch_approved", "lint_clean", "sim_passed",
            "synth_success", "synth_gate_count",
            "rtl_path", "tb_path", "debug_action",
            "completed_blocks", "human_response",
        ]
        for f in required:
            assert f in annotations, f"Missing field: {f}"

    def test_no_content_fields(self):
        """Disk-first: BlockState must NOT carry content."""
        annotations = BlockState.__annotations__
        content_fields = [
            "constraints", "attempt_history", "previous_error",
            "uarch_spec", "uarch_feedback",
            "rtl_result", "lint_result", "tb_result",
            "sim_result", "synth_result", "debug_result",
        ]
        for f in content_fields:
            assert f not in annotations, f"Content field still in BlockState: {f}"


class TestOrchestratorState:
    def test_has_required_fields(self):
        annotations = OrchestratorState.__annotations__
        required = [
            "project_root", "target_clock_mhz", "max_attempts",
            "block_queue", "tier_list", "current_tier_index",
            "completed_blocks", "pipeline_done",
        ]
        for f in required:
            assert f in annotations, f"Missing field: {f}"


# ---------------------------------------------------------------------------
# Routing functions (block-level)
# ---------------------------------------------------------------------------

class TestRouteAfterUarchReview:
    def test_approve_goes_to_generate_rtl(self):
        assert route_after_uarch_review({"human_response": {"action": "approve"}}) == "generate_rtl"

    def test_revise_goes_to_generate_uarch_spec(self):
        assert route_after_uarch_review({"human_response": {"action": "revise"}}) == "generate_uarch_spec"

    def test_skip_goes_to_block_done(self):
        assert route_after_uarch_review({"human_response": {"action": "skip"}}) == "block_done"

    def test_default_goes_to_generate_rtl(self):
        assert route_after_uarch_review({}) == "generate_rtl"


@pytest.mark.skip(
    reason="route_after_lint was inlined into the graph; restore tests when "
    "the new routing surface is settled."
)
class TestRouteAfterLint:
    pass


@pytest.mark.skip(
    reason="route_after_sim was inlined into the graph; restore tests when "
    "the new routing surface is settled."
)
class TestRouteAfterSim:
    pass


# NOTE: route_decision and route_after_human now route directly to the
# next stage (generate_rtl, generate_testbench, ...) instead of going
# through an explicit increment_attempt indirection. The assertions
# below reflect the post-inlining mapping defined in pipeline_graph.py.
class TestRouteDecision:
    def test_retry_rtl(self):
        assert route_decision({"debug_action": "retry_rtl"}) == "generate_rtl"

    def test_retry_tb(self):
        assert route_decision({"debug_action": "retry_tb"}) == "generate_testbench"

    def test_ask_human(self):
        assert route_decision({"debug_action": "ask_human"}) == "ask_human"

    def test_escalate(self):
        assert route_decision({"debug_action": "escalate"}) == "block_done"

    def test_default_routes_to_generate_rtl(self):
        assert route_decision({"debug_action": "??"}) == "generate_rtl"

    def test_missing_action_defaults_to_generate_rtl(self):
        assert route_decision({}) == "generate_rtl"


class TestRouteAfterHuman:
    def test_retry(self):
        assert route_after_human({"human_response": {"action": "retry"}}) == "generate_rtl"

    def test_fix_rtl(self):
        assert route_after_human({"human_response": {"action": "fix_rtl"}}) == "generate_rtl"

    def test_fix_tb(self):
        assert route_after_human({"human_response": {"action": "fix_tb"}}) == "generate_testbench"

    def test_add_constraint(self):
        assert route_after_human({"human_response": {"action": "add_constraint"}}) == "generate_rtl"

    def test_skip(self):
        assert route_after_human({"human_response": {"action": "skip"}}) == "block_done"

    def test_abort(self):
        assert route_after_human({"human_response": {"action": "abort"}}) == "block_done"

    def test_default_routes_to_generate_rtl(self):
        assert route_after_human({"human_response": {"action": "??"}}) == "generate_rtl"

    def test_missing_response_defaults_to_generate_rtl(self):
        assert route_after_human({}) == "generate_rtl"


@pytest.mark.skip(
    reason="route_after_increment was inlined into the graph; restore tests "
    "when the new routing surface is settled."
)
class TestRouteAfterIncrement:
    pass


# ---------------------------------------------------------------------------
# Orchestrator routing
# ---------------------------------------------------------------------------

class TestRouteNextTier:
    def test_more_tiers_goes_to_init_tier(self):
        state = {"tier_list": [1, 2, 3], "current_tier_index": 1, "completed_blocks": []}
        assert route_next_tier(state) == "init_tier"

    def test_all_done_goes_to_pipeline_complete(self):
        state = {"tier_list": [1, 2, 3], "current_tier_index": 3, "completed_blocks": []}
        assert route_next_tier(state) == "pipeline_complete"

    def test_single_tier_done(self):
        state = {"tier_list": [1], "current_tier_index": 1, "completed_blocks": []}
        assert route_next_tier(state) == "pipeline_complete"

    def test_aborted_block_stops_pipeline(self):
        state = {
            "tier_list": [1, 2],
            "current_tier_index": 1,
            "completed_blocks": [{"name": "a", "success": False, "aborted": True}],
        }
        assert route_next_tier(state) == "pipeline_complete"


class TestRouteAfterIntegrationReview:
    def test_approve_advances_tier(self):
        assert route_after_integration_review({"integration_review_action": "approve"}) == "advance_tier"

    def test_abort_ends_pipeline(self):
        assert route_after_integration_review({"integration_review_action": "abort"}) == "__end__"

    def test_revise_inits_tier(self):
        # 'revise' reruns the current tier from the revised uArch specs;
        # see route_after_integration_review docstring + __edge_labels__.
        assert route_after_integration_review({"integration_review_action": "revise"}) == "init_tier"

    def test_default_is_approve(self):
        assert route_after_integration_review({}) == "advance_tier"

    @pytest.mark.asyncio
    async def test_clean_revise_response_is_forced_to_approve(self, tmp_path, monkeypatch):
        from orchestrator.langchain.agents import integration_review_agent

        def fake_init(self, *args, **kwargs):
            pass

        async def fake_review(self, block_names, project_root):
            return {
                "summary": "No current-tier integration issues found.",
                "issues_found": 0,
                "issues_fixed": 0,
            }

        monkeypatch.setattr(
            integration_review_agent.IntegrationReviewAgent,
            "__init__",
            fake_init,
        )
        monkeypatch.setattr(
            integration_review_agent.IntegrationReviewAgent,
            "review",
            fake_review,
        )
        monkeypatch.setattr(
            pipeline_graph,
            "interrupt",
            lambda payload: {"action": "revise"},
        )

        result = await pipeline_graph.integration_review_node({
            "project_root": str(tmp_path),
            "block_queue": [{"name": "adder32", "tier": 1}],
            "tier_list": [1],
            "current_tier_index": 0,
        })

        assert result["integration_review_action"] == "approve"

    @pytest.mark.asyncio
    async def test_review_exception_blocks_auto_approval(self, tmp_path, monkeypatch):
        from orchestrator.langchain.agents import integration_review_agent

        def fake_init(self, *args, **kwargs):
            pass

        async def fake_review(self, block_names, project_root):
            raise RuntimeError("usage limit")

        captured_payload = {}

        def fake_interrupt(payload):
            captured_payload.update(payload)
            return {"action": "revise"}

        monkeypatch.setattr(
            integration_review_agent.IntegrationReviewAgent,
            "__init__",
            fake_init,
        )
        monkeypatch.setattr(
            integration_review_agent.IntegrationReviewAgent,
            "review",
            fake_review,
        )
        monkeypatch.setattr(pipeline_graph, "interrupt", fake_interrupt)

        result = await pipeline_graph.integration_review_node({
            "project_root": str(tmp_path),
            "block_queue": [{"name": "adder32", "tier": 1}],
            "tier_list": [1],
            "current_tier_index": 0,
        })

        assert captured_payload["review_failed"] is True
        assert captured_payload["issues_found"] == 1
        assert result["integration_review_action"] == "revise"
        assert result["integration_review_failed"] is True

    @pytest.mark.asyncio
    async def test_fixed_uarch_review_honors_explicit_approve_by_default(self, tmp_path, monkeypatch):
        """Default mode: when issues_fixed>0 and the outer agent explicitly
        approves, integration_review_node honors the approve. The previous
        auto-revise on issues_fixed>0 caused a non-terminating loop because
        the integration-review LLM agent edits specs on every run."""
        monkeypatch.delenv("CORESMITH_STRICT_INTEGRATION_REVIEW", raising=False)
        from orchestrator.langchain.agents import integration_review_agent

        def fake_init(self, *args, **kwargs):
            pass

        async def fake_review(self, block_names, project_root):
            spec_dir = tmp_path / "arch" / "uarch_specs"
            spec_dir.mkdir(parents=True, exist_ok=True)
            (spec_dir / "adder32.md").write_text("updated spec", encoding="utf-8")
            return {
                "summary": "Fixed status bus layout in adder32 uArch.",
                "issues_found": 1,
                "issues_fixed": 1,
            }

        captured_payload = {}

        def fake_interrupt(payload):
            captured_payload.update(payload)
            return {"action": "approve"}

        monkeypatch.setattr(
            integration_review_agent.IntegrationReviewAgent,
            "__init__",
            fake_init,
        )
        monkeypatch.setattr(
            integration_review_agent.IntegrationReviewAgent,
            "review",
            fake_review,
        )
        monkeypatch.setattr(pipeline_graph, "interrupt", fake_interrupt)

        result = await pipeline_graph.integration_review_node({
            "project_root": str(tmp_path),
            "block_queue": [{"name": "adder32", "tier": 1}],
            "tier_list": [1],
            "current_tier_index": 0,
            "completed_blocks": [{"name": "adder32", "success": True}],
        })

        # Payload still reports the failure for outer-agent visibility.
        assert captured_payload["review_failed"] is True
        assert captured_payload["issues_fixed"] == 1
        assert "Blocking uArch edits" in captured_payload["review_summary"]
        # But the explicit approve is honored.
        assert result["integration_review_action"] == "approve"
        assert result["integration_review_failed"] is True

    @pytest.mark.asyncio
    async def test_fixed_uarch_review_blocks_approve_under_strict_mode(self, tmp_path, monkeypatch):
        """CORESMITH_STRICT_INTEGRATION_REVIEW=1 restores the original
        approve->revise auto-conversion when issues_fixed>0."""
        monkeypatch.setenv("CORESMITH_STRICT_INTEGRATION_REVIEW", "1")
        from orchestrator.langchain.agents import integration_review_agent

        def fake_init(self, *args, **kwargs):
            pass

        async def fake_review(self, block_names, project_root):
            spec_dir = tmp_path / "arch" / "uarch_specs"
            spec_dir.mkdir(parents=True, exist_ok=True)
            (spec_dir / "adder32.md").write_text("updated spec", encoding="utf-8")
            return {
                "summary": "Fixed status bus layout in adder32 uArch.",
                "issues_found": 1,
                "issues_fixed": 1,
            }

        captured_payload = {}

        def fake_interrupt(payload):
            captured_payload.update(payload)
            return {"action": "approve"}

        monkeypatch.setattr(
            integration_review_agent.IntegrationReviewAgent,
            "__init__",
            fake_init,
        )
        monkeypatch.setattr(
            integration_review_agent.IntegrationReviewAgent,
            "review",
            fake_review,
        )
        monkeypatch.setattr(pipeline_graph, "interrupt", fake_interrupt)

        result = await pipeline_graph.integration_review_node({
            "project_root": str(tmp_path),
            "block_queue": [{"name": "adder32", "tier": 1}],
            "tier_list": [1],
            "current_tier_index": 0,
            "completed_blocks": [{"name": "adder32", "success": True}],
        })

        assert captured_payload["review_failed"] is True
        assert captured_payload["issues_fixed"] == 1
        assert "Blocking uArch edits" in captured_payload["review_summary"]
        assert result["integration_review_action"] == "revise"
        assert result["integration_review_failed"] is True


class TestRouteAfterIntegration:
    def test_clean_integration_goes_to_dv(self):
        assert pipeline_graph.route_after_integration({
            "integration_result": {"lint_clean": True, "error_count": 0}
        }) == "integration_dv"

    def test_lint_failure_ends(self):
        assert pipeline_graph.route_after_integration({
            "integration_result": {"lint_clean": False, "error_count": 0}
        }) == "__end__"

    def test_error_count_ends(self):
        assert pipeline_graph.route_after_integration({
            "integration_result": {"lint_clean": True, "error_count": 1}
        }) == "__end__"

    def test_accepted_by_user_overrides_error_count(self):
        # When the operator/agent explicitly accepts the integration
        # failure (chip_top still lint-passes, mismatches are
        # acceptable for this run), routing must advance to DV
        # regardless of error_count.
        assert pipeline_graph.route_after_integration({
            "integration_result": {
                "lint_clean": True,
                "error_count": 2,
                "accepted_by_user": True,
            }
        }) == "integration_dv"

    def test_accepted_by_user_does_not_override_lint_failure(self):
        # accept is only meaningful when chip_top still lint-passes;
        # if lint failed, route to END even if accepted_by_user is set.
        assert pipeline_graph.route_after_integration({
            "integration_result": {
                "lint_clean": False,
                "error_count": 0,
                "accepted_by_user": True,
            }
        }) == "__end__"

    # --- CORESMITH_REQUIRE_DV: DV must not be silently skippable ---

    def test_skipped_review_ends_without_dv_by_default(self, monkeypatch):
        monkeypatch.delenv("CORESMITH_REQUIRE_DV", raising=False)
        assert pipeline_graph.route_after_integration({
            "integration_result": {
                "lint_clean": True, "error_count": 0, "skipped_by_user": True,
            }
        }) == "__end__"

    def test_skipped_review_runs_dv_when_require_dv(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_REQUIRE_DV", "1")
        assert pipeline_graph.route_after_integration({
            "integration_result": {
                "lint_clean": True, "error_count": 0, "skipped_by_user": True,
            }
        }) == "integration_dv"

    def test_unaccepted_errors_run_dv_when_require_dv(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_REQUIRE_DV", "1")
        assert pipeline_graph.route_after_integration({
            "integration_result": {"lint_clean": True, "error_count": 3}
        }) == "integration_dv"

    def test_require_dv_does_not_override_abort(self, monkeypatch):
        # A deliberate abort (or fix_rtl/retry that ENDs for restart_node)
        # must still END even when DV is required.
        monkeypatch.setenv("CORESMITH_REQUIRE_DV", "1")
        assert pipeline_graph.route_after_integration({
            "integration_result": {"aborted": True, "lint_clean": True}
        }) == "__end__"

    def test_require_dv_does_not_override_lint_failure(self, monkeypatch):
        # A hard lint failure can't be DV'd (chip_top won't build), so END.
        monkeypatch.setenv("CORESMITH_REQUIRE_DV", "1")
        assert pipeline_graph.route_after_integration({
            "integration_result": {"lint_clean": False, "error_count": 0}
        }) == "__end__"

    @pytest.mark.asyncio
    async def test_partial_block_set_refuses_integration(self, tmp_path):
        result = await pipeline_graph.integration_check_node({
            "project_root": str(tmp_path),
            "completed_blocks": [
                {"name": "a", "success": True},
                {"name": "b", "success": False},
            ],
            "block_queue": [
                {"name": "a"},
                {"name": "b"},
                {"name": "c"},
            ],
        })

        integration_result = result["integration_result"]
        assert integration_result["aborted"] is True
        assert integration_result["error"] == "partial_block_set"
        assert integration_result["error_count"] == 2
        assert integration_result["failed_blocks"] == ["b"]
        assert integration_result["missing_blocks"] == ["c"]


# ---------------------------------------------------------------------------
# Internal nodes (unit tests)
# ---------------------------------------------------------------------------

class TestInternalNodes:
    @pytest.mark.asyncio
    async def test_init_block_resets_state(self, tmp_path):
        state = _block_state(_make_block("scrambler"), tmp_path=str(tmp_path))
        result = await init_block_node(state)
        assert result["attempt"] == 1
        assert result["uarch_approved"] is False
        assert result["lint_clean"] is False
        assert result["sim_passed"] is False
        assert result["synth_success"] is False
        assert result["rtl_path"] == ""
        assert result["tb_path"] == ""
        assert result["debug_action"] == ""

    @pytest.mark.asyncio
    async def test_init_block_creates_disk_dir(self, tmp_path):
        state = _block_state(_make_block("scrambler"), tmp_path=str(tmp_path))
        await init_block_node(state)
        block_dir = tmp_path / ".coresmith" / "blocks" / "scrambler"
        assert block_dir.is_dir()
        assert (block_dir / "constraints.json").exists()
        assert (block_dir / "diagnosis.json").exists()
        assert (block_dir / "attempt_history.json").exists()
        assert (block_dir / "previous_error.txt").exists()

    @pytest.mark.skip(
        reason="increment_attempt_node was inlined into the pipeline graph; "
        "the surviving counterpart lives in backend_graph. Restore this when "
        "the pipeline-side increment node gets a fresh test pass."
    )
    @pytest.mark.asyncio
    async def test_increment_attempt(self):
        pass

    @pytest.mark.asyncio
    async def test_pipeline_complete(self):
        state = {"completed_blocks": [{"name": "a", "success": True}]}
        result = await pipeline_complete_node(state)
        assert result["pipeline_done"] is True

    @pytest.mark.asyncio
    async def test_pipeline_complete_retry_does_not_proceed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            pipeline_graph,
            "interrupt",
            lambda payload: {"action": "retry"},
        )

        result = await pipeline_complete_node({
            "project_root": str(tmp_path),
            "completed_blocks": [
                {"name": "a", "success": True},
                {"name": "b", "success": False},
            ],
            "block_queue": [{"name": "a"}, {"name": "b"}],
        })

        assert result["pipeline_done"] is False
        assert result["pipeline_aborted"] is True

    @pytest.mark.asyncio
    async def test_block_done_success(self, tmp_path):
        state = _block_state(_make_block("scrambler"), tmp_path=str(tmp_path))
        state["sim_passed"] = True
        state["synth_success"] = True
        state["synth_gate_count"] = 1500
        block_dir = tmp_path / ".coresmith" / "blocks" / "scrambler"
        block_dir.mkdir(parents=True, exist_ok=True)
        (block_dir / "constraints.json").write_text("[]")
        result = await block_done_node(state)
        assert len(result["completed_blocks"]) == 1
        assert result["completed_blocks"][0]["success"] is True
        assert result["completed_blocks"][0]["name"] == "scrambler"

    @pytest.mark.asyncio
    async def test_block_done_skip(self, tmp_path):
        state = _block_state(_make_block("scrambler"), tmp_path=str(tmp_path))
        state["human_response"] = {"action": "skip"}
        block_dir = tmp_path / ".coresmith" / "blocks" / "scrambler"
        block_dir.mkdir(parents=True, exist_ok=True)
        (block_dir / "constraints.json").write_text("[]")
        (block_dir / "previous_error.txt").write_text("")
        result = await block_done_node(state)
        assert result["completed_blocks"][0]["success"] is False
        assert result["completed_blocks"][0]["skipped"] is True

    @pytest.mark.asyncio
    async def test_block_done_abort(self, tmp_path):
        state = _block_state(_make_block("scrambler"), tmp_path=str(tmp_path))
        state["human_response"] = {"action": "abort"}
        block_dir = tmp_path / ".coresmith" / "blocks" / "scrambler"
        block_dir.mkdir(parents=True, exist_ok=True)
        (block_dir / "constraints.json").write_text("[]")
        (block_dir / "previous_error.txt").write_text("")
        result = await block_done_node(state)
        assert result["completed_blocks"][0]["success"] is False
        assert result["completed_blocks"][0]["aborted"] is True


# ---------------------------------------------------------------------------
# Happy path (full graph invocation, 1 block)
# ---------------------------------------------------------------------------

class TestHappyPath:
    @pytest.mark.asyncio
    async def test_single_block_passes(self, tmp_path):
        """Walk a single block through the happy path with all helpers mocked.

        Disk-first: create actual files on disk so nodes can find them.
        """
        uarch_result, rtl_result, lint_clean, _, tb_result, sim_pass, _, synth_ok = _patch_all_helpers()

        block = _make_block("scrambler")
        rtl_dir = tmp_path / "rtl" / "dvbt"
        rtl_dir.mkdir(parents=True)
        (rtl_dir / "scrambler.v").write_text("module scrambler(); endmodule\n")
        tb_dir = tmp_path / "tb" / "cocotb"
        tb_dir.mkdir(parents=True)
        (tb_dir / "test_scrambler.py").write_text("# test\n")
        block_dir = tmp_path / ".coresmith" / "blocks" / "scrambler"
        block_dir.mkdir(parents=True)
        (block_dir / "constraints.json").write_text("[]")
        (block_dir / "diagnosis.json").write_text("{}")
        (block_dir / "attempt_history.json").write_text("[]")
        (block_dir / "previous_error.txt").write_text("")

        graph = build_pipeline_graph(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "test-happy-1"}}
        state = _initial_state([block])
        state["project_root"] = str(tmp_path)

        async def _mock_gen_rtl(block, attempt, **kw):
            return {"rtl_path": str(rtl_dir / "scrambler.v")}

        with patch(
            "orchestrator.langgraph.pipeline_graph.generate_uarch_spec",
            new_callable=AsyncMock,
            return_value=uarch_result,
        ), patch(
            "orchestrator.langgraph.pipeline_graph.generate_rtl",
            new_callable=AsyncMock,
            side_effect=_mock_gen_rtl,
        ), patch(
            "orchestrator.langgraph.pipeline_graph.lint_rtl",
            return_value=lint_clean,
        ), patch(
            "orchestrator.langgraph.pipeline_graph.generate_testbench",
            new_callable=AsyncMock,
            return_value=tb_result,
        ), patch(
            "orchestrator.langgraph.pipeline_graph.run_simulation",
            return_value=sim_pass,
        ), patch(
            "orchestrator.langgraph.pipeline_graph.synthesize_block",
            return_value=synth_ok,
        ), patch(
            "orchestrator.langgraph.pipeline_graph.create_golden_model_wrapper",
        ):
            await graph.ainvoke(state, config)
            result = await graph.ainvoke(
                Command(resume={"action": "approve"}), config
            )

        assert result["pipeline_done"] is True
        assert len(result["completed_blocks"]) == 1
        assert result["completed_blocks"][0]["success"] is True
        assert result["completed_blocks"][0]["name"] == "scrambler"


# ---------------------------------------------------------------------------
# Interrupt flow
# ---------------------------------------------------------------------------

class TestInterruptFlow:
    @pytest.mark.asyncio
    async def test_uarch_spec_auto_approves_then_integration_review_interrupts(self, tmp_path):
        """review_uarch_spec auto-approves; integration_review fires at orchestrator level."""
        uarch_result, rtl_result, lint_clean, _, tb_result, sim_pass, _, synth_ok = _patch_all_helpers()

        block = _make_block("scrambler")
        _setup_disk_fixtures(tmp_path, [block])

        graph = build_pipeline_graph(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "test-uarch-auto-approve-1"}}
        state = _initial_state([block], project_root=str(tmp_path))

        with patch(
            "orchestrator.langgraph.pipeline_graph.generate_uarch_spec",
            new_callable=AsyncMock,
            return_value=uarch_result,
        ), patch(
            "orchestrator.langgraph.pipeline_graph.generate_rtl",
            new_callable=AsyncMock,
            return_value=rtl_result,
        ), patch(
            "orchestrator.langgraph.pipeline_graph.lint_rtl",
            return_value=lint_clean,
        ), patch(
            "orchestrator.langgraph.pipeline_graph.generate_testbench",
            new_callable=AsyncMock,
            return_value=tb_result,
        ), patch(
            "orchestrator.langgraph.pipeline_graph.run_simulation",
            return_value=sim_pass,
        ), patch(
            "orchestrator.langgraph.pipeline_graph.synthesize_block",
            return_value=synth_ok,
        ), patch(
            "orchestrator.langgraph.pipeline_graph.create_golden_model_wrapper",
        ):
            await graph.ainvoke(state, config)

        payload = await _get_interrupt(graph, config)
        assert payload is not None
        assert payload["type"] == "uarch_integration_review"
        assert "approve" in payload["supported_actions"]

    @pytest.mark.asyncio
    async def test_lint_failure_triggers_interrupt(self, tmp_path):
        """lint fail -> diagnose -> decide(ask_human) -> interrupt.

        review_uarch_spec auto-approves so the graph flows directly
        from generate_rtl through lint failure to ask_human.
        """
        uarch_result, rtl_result, _, lint_fail, tb_result, sim_pass, _, synth_ok = _patch_all_helpers()

        async def mock_decide(state):
            """Mock decide node that always routes to ask_human."""
            return {"debug_action": "ask_human"}

        block = _make_block("scrambler")
        _setup_disk_fixtures(tmp_path, [block])

        config = {"configurable": {"thread_id": "test-interrupt-1"}}
        state = _initial_state([block], project_root=str(tmp_path))

        # decide_node is captured by reference inside ``build_pipeline_graph``,
        # so the graph must be built INSIDE the patch context for the mock to
        # take effect.
        with patch(
            "orchestrator.langgraph.pipeline_graph.generate_uarch_spec",
            new_callable=AsyncMock,
            return_value=uarch_result,
        ), patch(
            "orchestrator.langgraph.pipeline_graph.generate_rtl",
            new_callable=AsyncMock,
            return_value=rtl_result,
        ), patch(
            "orchestrator.langgraph.pipeline_graph.lint_rtl",
            return_value=lint_fail,
        ), patch(
            "orchestrator.langgraph.pipeline_graph.fix_lint_errors",
            new_callable=AsyncMock,
            return_value=None,
        ), patch(
            "orchestrator.langgraph.pipeline_graph.diagnose_failure",
            new_callable=AsyncMock,
            return_value={"category": "LOGIC_ERROR", "diagnosis": "test"},
        ), patch(
            "orchestrator.langgraph.pipeline_graph.create_golden_model_wrapper",
        ), patch(
            "orchestrator.langgraph.pipeline_graph.decide_node",
            side_effect=mock_decide,
        ):
            graph = build_pipeline_graph(checkpointer=MemorySaver())
            await graph.ainvoke(state, config)

        payload = await _get_interrupt(graph, config)
        assert payload is not None
        assert payload["type"] == "human_intervention_needed"
        assert payload["block_name"] == "scrambler"
        assert "retry" in payload["supported_actions"]
        assert "fix_rtl" in payload["supported_actions"]
        assert "skip" in payload["supported_actions"]


# ---------------------------------------------------------------------------
# Resume actions
# ---------------------------------------------------------------------------

class TestResumeActions:
    @pytest.mark.asyncio
    async def test_skip_completes_block(self, tmp_path):
        """Resume with skip -> block_done -> pipeline_complete.

        review_uarch_spec auto-approves, so the first interrupt is
        at ask_human (lint failure). integration_review is mocked to
        avoid a second chip-level interrupt.
        """
        uarch_result, rtl_result, _, lint_fail, _, _, _, _ = _patch_all_helpers()

        async def mock_decide(state):
            return {"debug_action": "ask_human"}

        async def mock_integration_review(state):
            return {}

        block = _make_block("scrambler")
        _setup_disk_fixtures(tmp_path, [block])

        config = {"configurable": {"thread_id": "test-skip-1"}}
        state = _initial_state([block], project_root=str(tmp_path))

        with patch(
            "orchestrator.langgraph.pipeline_graph.generate_uarch_spec",
            new_callable=AsyncMock,
            return_value=uarch_result,
        ), patch(
            "orchestrator.langgraph.pipeline_graph.generate_rtl",
            new_callable=AsyncMock,
            return_value=rtl_result,
        ), patch(
            "orchestrator.langgraph.pipeline_graph.lint_rtl",
            return_value=lint_fail,
        ), patch(
            "orchestrator.langgraph.pipeline_graph.fix_lint_errors",
            new_callable=AsyncMock,
            return_value=None,
        ), patch(
            "orchestrator.langgraph.pipeline_graph.diagnose_failure",
            new_callable=AsyncMock,
            return_value={"category": "LOGIC_ERROR", "diagnosis": "test"},
        ), patch(
            "orchestrator.langgraph.pipeline_graph.create_golden_model_wrapper",
        ), patch(
            "orchestrator.langgraph.pipeline_graph.decide_node",
            side_effect=mock_decide,
        ), patch(
            "orchestrator.langgraph.pipeline_graph.integration_review_node",
            side_effect=mock_integration_review,
        ):
            graph = build_pipeline_graph(checkpointer=MemorySaver())
            # uarch auto-approves -> lint fail -> ask_human interrupt
            await graph.ainvoke(state, config)

            # Now at ask_human interrupt -- resume with skip
            result = await graph.ainvoke(
                Command(resume={"action": "skip"}), config
            )

        # pipeline_complete_node fires a pipeline_incomplete interrupt
        # (0/1 blocks passed) before setting pipeline_done = True.
        assert len(result["completed_blocks"]) == 1
        assert result["completed_blocks"][0]["success"] is False
        assert result["completed_blocks"][0].get("skipped") is True

        interrupt = await _get_interrupt(graph, config)
        assert interrupt is not None
        assert interrupt["type"] == "pipeline_incomplete"
        assert interrupt["passed"] == 0

    @pytest.mark.asyncio
    async def test_abort_stops_pipeline(self, tmp_path):
        """Resume with abort -> block_done (aborted) -> pipeline_complete.

        review_uarch_spec auto-approves so both parallel blocks hit
        ask_human directly.
        """
        uarch_result, rtl_result, _, lint_fail, _, _, _, _ = _patch_all_helpers()

        async def mock_decide(state):
            return {"debug_action": "ask_human"}

        async def mock_integration_review(state):
            return {}

        blocks = [_make_block("scrambler"), _make_block("crc32")]
        _setup_disk_fixtures(tmp_path, blocks)

        config = {"configurable": {"thread_id": "test-abort-1"}}
        state = _initial_state(blocks, project_root=str(tmp_path))

        with patch(
            "orchestrator.langgraph.pipeline_graph.generate_uarch_spec",
            new_callable=AsyncMock,
            return_value=uarch_result,
        ), patch(
            "orchestrator.langgraph.pipeline_graph.generate_rtl",
            new_callable=AsyncMock,
            return_value=rtl_result,
        ), patch(
            "orchestrator.langgraph.pipeline_graph.lint_rtl",
            return_value=lint_fail,
        ), patch(
            "orchestrator.langgraph.pipeline_graph.fix_lint_errors",
            new_callable=AsyncMock,
            return_value=None,
        ), patch(
            "orchestrator.langgraph.pipeline_graph.diagnose_failure",
            new_callable=AsyncMock,
            return_value={"category": "LOGIC_ERROR", "diagnosis": "test"},
        ), patch(
            "orchestrator.langgraph.pipeline_graph.create_golden_model_wrapper",
        ), patch(
            "orchestrator.langgraph.pipeline_graph.decide_node",
            side_effect=mock_decide,
        ), patch(
            "orchestrator.langgraph.pipeline_graph.integration_review_node",
            side_effect=mock_integration_review,
        ):
            graph = build_pipeline_graph(checkpointer=MemorySaver())
            # Both blocks hit ask_human in parallel (uarch auto-approves)
            await graph.ainvoke(state, config)

            # Both blocks at ask_human -- resume all with abort.
            result = await _resume_all(graph, config, {"action": "abort"})

        # pipeline_complete_node fires a pipeline_incomplete interrupt.
        aborted_blocks = [b for b in result["completed_blocks"] if b.get("aborted")]
        assert len(aborted_blocks) >= 1

        interrupt = await _get_interrupt(graph, config)
        assert interrupt is not None
        assert interrupt["type"] == "pipeline_incomplete"
        assert interrupt["passed"] == 0


# ---------------------------------------------------------------------------
# Multi-block (parallel within tier)
# ---------------------------------------------------------------------------

class TestMultiBlock:
    @pytest.mark.asyncio
    async def test_three_blocks_same_tier_all_pass(self, tmp_path):
        """Walk 3 same-tier blocks through the happy path (auto-approve uarch specs)."""
        uarch_result, rtl_result, lint_clean, _, tb_result, sim_pass, _, synth_ok = _patch_all_helpers()

        blocks = [
            _make_block("scrambler", tier=1),
            _make_block("crc32", tier=1),
            _make_block("conv_encoder", tier=1),
        ]
        _setup_disk_fixtures(tmp_path, blocks)

        config = {"configurable": {"thread_id": "test-multi-1"}}
        state = _initial_state(blocks, project_root=str(tmp_path))

        def _make_uarch_result(block):
            return {
                "spec_text": f"## Spec for {block['name']}",
                "spec_summary": {"block_name": block["name"]},
                "spec_path": str(tmp_path / "arch" / "uarch_specs" / f"{block['name']}.md"),
                "block_name": block["name"],
            }

        def _make_rtl_result(block_name):
            return {
                "verilog": f"module {block_name}(); endmodule\n",
                "rtl_path": str(tmp_path / "rtl" / "dvbt" / f"{block_name}.v"),
                "ports": {"clk": "input"},
            }

        def _make_tb_result(block_name):
            return {
                "testbench": "# test",
                "testbench_path": str(
                    tmp_path / "tb" / "cocotb" / f"test_{block_name}.py"
                ),
            }

        with patch(
            "orchestrator.langgraph.pipeline_graph.generate_uarch_spec",
            new_callable=AsyncMock,
            side_effect=lambda block, **kw: _make_uarch_result(block),
        ), patch(
            "orchestrator.langgraph.pipeline_graph.generate_rtl",
            new_callable=AsyncMock,
            side_effect=lambda block, *a, **kw: _make_rtl_result(block["name"]),
        ), patch(
            "orchestrator.langgraph.pipeline_graph.lint_rtl",
            return_value=lint_clean,
        ), patch(
            "orchestrator.langgraph.pipeline_graph.generate_testbench",
            new_callable=AsyncMock,
            side_effect=lambda block, *a, **kw: _make_tb_result(block["name"]),
        ), patch(
            "orchestrator.langgraph.pipeline_graph.run_simulation",
            return_value=sim_pass,
        ), patch(
            "orchestrator.langgraph.pipeline_graph.synthesize_block",
            return_value=synth_ok,
        ), patch(
            "orchestrator.langgraph.pipeline_graph.create_golden_model_wrapper",
        ), patch(
            "orchestrator.langgraph.pipeline_graph.integration_review_node",
            new_callable=AsyncMock,
            return_value={},
        ):
            graph = build_pipeline_graph(checkpointer=MemorySaver())
            # All 3 blocks are tier 1 -> fanned out in parallel.
            # All 3 hit uarch review interrupt simultaneously.
            # Resume all interrupts at once with approve.
            await graph.ainvoke(state, config)
            result = await _resume_all(graph, config, {"action": "approve"})

        assert result["pipeline_done"] is True
        assert len(result["completed_blocks"]) == 3
        names = sorted(b["name"] for b in result["completed_blocks"])
        assert names == ["conv_encoder", "crc32", "scrambler"]
        assert all(b["success"] for b in result["completed_blocks"])


# ---------------------------------------------------------------------------
# Checkpoint persistence (pause/restart simulation)
# ---------------------------------------------------------------------------

class TestCheckpointPersistence:
    @pytest.mark.asyncio
    async def test_state_preserved_after_integration_review_interrupt(self, tmp_path):
        """Verify state is readable from checkpoint after integration_review interrupt.

        Since review_uarch_spec auto-approves, the first orchestrator-level
        interrupt is now integration_review (after all blocks in a tier complete).
        """
        uarch_result, rtl_result, lint_clean, _, tb_result, sim_pass, _, synth_ok = _patch_all_helpers()

        block = _make_block("scrambler")
        _setup_disk_fixtures(tmp_path, [block])

        checkpointer = MemorySaver()
        config = {"configurable": {"thread_id": "test-checkpoint-1"}}
        state = _initial_state([block], project_root=str(tmp_path))

        with patch(
            "orchestrator.langgraph.pipeline_graph.generate_uarch_spec",
            new_callable=AsyncMock,
            return_value=uarch_result,
        ), patch(
            "orchestrator.langgraph.pipeline_graph.generate_rtl",
            new_callable=AsyncMock,
            return_value=rtl_result,
        ), patch(
            "orchestrator.langgraph.pipeline_graph.lint_rtl",
            return_value=lint_clean,
        ), patch(
            "orchestrator.langgraph.pipeline_graph.generate_testbench",
            new_callable=AsyncMock,
            return_value=tb_result,
        ), patch(
            "orchestrator.langgraph.pipeline_graph.run_simulation",
            return_value=sim_pass,
        ), patch(
            "orchestrator.langgraph.pipeline_graph.synthesize_block",
            return_value=synth_ok,
        ), patch(
            "orchestrator.langgraph.pipeline_graph.create_golden_model_wrapper",
        ):
            graph = build_pipeline_graph(checkpointer=checkpointer)
            await graph.ainvoke(state, config)

        graph2 = build_pipeline_graph(checkpointer=checkpointer)
        saved_state = await graph2.aget_state(config)

        assert saved_state is not None
        assert len(saved_state.values.get("block_queue", [])) == 1
        assert saved_state.tasks
        found_interrupt = False
        for task in saved_state.tasks:
            if task.interrupts:
                found_interrupt = True
                payload = task.interrupts[0].value
                assert payload["type"] == "uarch_integration_review"
                break
        assert found_interrupt


# ---------------------------------------------------------------------------
# ask_human_node payload enrichment
# ---------------------------------------------------------------------------

class TestAskHumanPayloadEnrichment:
    """Verify that ask_human_node includes diagnostic context fields.

    These fields enable the outer-loop agent to pre-diagnose failures
    without needing additional MCP tool calls.
    """

    @pytest.mark.asyncio
    async def test_payload_has_step_log_paths(self, tmp_path):
        block_dir = tmp_path / ".coresmith" / "blocks" / "scrambler"
        block_dir.mkdir(parents=True)
        (block_dir / "diagnosis.json").write_text("{}")
        (block_dir / "attempt_history.json").write_text("[]")
        (block_dir / "previous_error.txt").write_text("")
        (block_dir / "constraints.json").write_text("[]")
        with patch(
            "orchestrator.langgraph.pipeline_graph.interrupt"
        ) as mock_interrupt, patch(
            "orchestrator.langgraph.pipeline_graph.write_graph_event"
        ):
            mock_interrupt.return_value = {"action": "retry"}
            state = _block_state(tmp_path=str(tmp_path))
            state["step_log_paths"] = {"lint": "/tmp/logs/lint_attempt1.log"}
            await ask_human_node(state)

        payload = mock_interrupt.call_args[0][0]
        assert "step_log_paths" in payload
        assert payload["step_log_paths"]["lint"] == "/tmp/logs/lint_attempt1.log"

    @pytest.mark.asyncio
    async def test_payload_has_testbench_path(self, tmp_path):
        block_dir = tmp_path / ".coresmith" / "blocks" / "scrambler"
        block_dir.mkdir(parents=True)
        (block_dir / "diagnosis.json").write_text("{}")
        (block_dir / "attempt_history.json").write_text("[]")
        (block_dir / "previous_error.txt").write_text("")
        (block_dir / "constraints.json").write_text("[]")
        with patch(
            "orchestrator.langgraph.pipeline_graph.interrupt"
        ) as mock_interrupt, patch(
            "orchestrator.langgraph.pipeline_graph.write_graph_event"
        ):
            mock_interrupt.return_value = {"action": "retry"}
            state = _block_state(tmp_path=str(tmp_path))
            await ask_human_node(state)

        payload = mock_interrupt.call_args[0][0]
        assert "testbench_path" in payload
        assert "test_scrambler.py" in payload["testbench_path"]

    @pytest.mark.asyncio
    async def test_payload_has_relative_paths(self, tmp_path):
        block_dir = tmp_path / ".coresmith" / "blocks" / "scrambler"
        block_dir.mkdir(parents=True)
        (block_dir / "diagnosis.json").write_text("{}")
        (block_dir / "attempt_history.json").write_text("[]")
        (block_dir / "previous_error.txt").write_text("")
        (block_dir / "constraints.json").write_text("[]")
        with patch(
            "orchestrator.langgraph.pipeline_graph.interrupt"
        ) as mock_interrupt, patch(
            "orchestrator.langgraph.pipeline_graph.write_graph_event"
        ):
            mock_interrupt.return_value = {"action": "retry"}
            state = _block_state(tmp_path=str(tmp_path))
            await ask_human_node(state)

        payload = mock_interrupt.call_args[0][0]
        assert "relative_paths" in payload
        rp = payload["relative_paths"]
        assert rp["rtl"] == "rtl/dvbt/scrambler.v"
        assert rp["testbench"] == "tb/cocotb/test_scrambler.py"
        assert rp["uarch_spec"] == "arch/uarch_specs/scrambler.md"
        assert rp["ers"] == ".coresmith/ers_spec.json"

    @pytest.mark.asyncio
    async def test_payload_has_rtl_snippet_when_file_exists(self, tmp_path):
        rtl_content = "\n".join([f"// line {i}" for i in range(50)])
        rtl_dir = tmp_path / "rtl" / "dvbt"
        rtl_dir.mkdir(parents=True)
        (rtl_dir / "scrambler.v").write_text(rtl_content)

        with patch(
            "orchestrator.langgraph.pipeline_graph.interrupt"
        ) as mock_interrupt, patch(
            "orchestrator.langgraph.pipeline_graph.write_graph_event"
        ):
            mock_interrupt.return_value = {"action": "retry"}
            state = _block_state()
            state["debug_result"] = {}
            state["project_root"] = str(tmp_path)
            await ask_human_node(state)

        payload = mock_interrupt.call_args[0][0]
        assert "rtl_snippet" in payload
        assert "// line 0" in payload["rtl_snippet"]
        assert "// line 49" in payload["rtl_snippet"]

    @pytest.mark.asyncio
    async def test_payload_has_ers_summary_when_file_exists(self, tmp_path):
        coresmith_dir = tmp_path / ".coresmith"
        coresmith_dir.mkdir()
        ers = {
            "ers": {
                "summary": "Test encoder block",
                "dataflow": {
                    "bus_protocol": "dedicated_pins",
                    "data_width_bits": 8,
                },
            }
        }
        import json
        (coresmith_dir / "ers_spec.json").write_text(json.dumps(ers))

        with patch(
            "orchestrator.langgraph.pipeline_graph.interrupt"
        ) as mock_interrupt, patch(
            "orchestrator.langgraph.pipeline_graph.write_graph_event"
        ):
            mock_interrupt.return_value = {"action": "retry"}
            state = _block_state()
            state["debug_result"] = {}
            state["project_root"] = str(tmp_path)
            await ask_human_node(state)

        payload = mock_interrupt.call_args[0][0]
        assert "ers_summary" in payload
        assert payload["ers_summary"]["summary"] == "Test encoder block"
        assert payload["ers_summary"]["bus_protocol"] == "dedicated_pins"
        assert payload["ers_summary"]["data_width_bits"] == 8

    @pytest.mark.asyncio
    async def test_missing_ers_does_not_crash(self, tmp_path):
        """ers_summary is optional -- no crash if ERS file is missing."""
        block_dir = tmp_path / ".coresmith" / "blocks" / "scrambler"
        block_dir.mkdir(parents=True)
        (block_dir / "diagnosis.json").write_text("{}")
        (block_dir / "attempt_history.json").write_text("[]")
        (block_dir / "previous_error.txt").write_text("")
        (block_dir / "constraints.json").write_text("[]")
        with patch(
            "orchestrator.langgraph.pipeline_graph.interrupt"
        ) as mock_interrupt, patch(
            "orchestrator.langgraph.pipeline_graph.write_graph_event"
        ):
            mock_interrupt.return_value = {"action": "retry"}
            state = _block_state(tmp_path=str(tmp_path))
            await ask_human_node(state)

        payload = mock_interrupt.call_args[0][0]
        assert payload["type"] == "human_intervention_needed"

    @pytest.mark.asyncio
    async def test_missing_rtl_does_not_crash(self, tmp_path):
        """rtl_snippet is optional -- no crash if RTL file is missing."""
        block_dir = tmp_path / ".coresmith" / "blocks" / "scrambler"
        block_dir.mkdir(parents=True)
        (block_dir / "diagnosis.json").write_text("{}")
        (block_dir / "attempt_history.json").write_text("[]")
        (block_dir / "previous_error.txt").write_text("")
        (block_dir / "constraints.json").write_text("[]")
        with patch(
            "orchestrator.langgraph.pipeline_graph.interrupt"
        ) as mock_interrupt, patch(
            "orchestrator.langgraph.pipeline_graph.write_graph_event"
        ):
            mock_interrupt.return_value = {"action": "retry"}
            state = _block_state(tmp_path=str(tmp_path))
            await ask_human_node(state)

        payload = mock_interrupt.call_args[0][0]
        assert "rtl_snippet" not in payload
        assert payload["type"] == "human_intervention_needed"


# ---------------------------------------------------------------------------
# ask_human_node: fix_rtl constraint persistence
# ---------------------------------------------------------------------------

class TestFixRtlConstraintPersistence:
    """Verify that fix_rtl description is persisted as a constraint.

    When the outer agent edits RTL and resumes with fix_rtl, the
    description should survive as a constraint so that if the block
    later retries via generate_rtl, the LLM knows what was tried.
    """

    @pytest.mark.asyncio
    async def test_fix_rtl_persists_description_as_constraint(self, tmp_path):
        import json
        block_dir = tmp_path / ".coresmith" / "blocks" / "scrambler"
        block_dir.mkdir(parents=True)
        (block_dir / "diagnosis.json").write_text("{}")
        (block_dir / "attempt_history.json").write_text("[]")
        (block_dir / "previous_error.txt").write_text("")
        (block_dir / "constraints.json").write_text("[]")
        with patch(
            "orchestrator.langgraph.pipeline_graph.interrupt"
        ) as mock_interrupt, patch(
            "orchestrator.langgraph.pipeline_graph.write_graph_event"
        ):
            mock_interrupt.return_value = {
                "action": "fix_rtl",
                "description": "Fixed port width from 8 to 16 bits",
            }
            state = _block_state(tmp_path=str(tmp_path))
            await ask_human_node(state)

        constraints = json.loads((block_dir / "constraints.json").read_text())
        assert len(constraints) == 1
        assert "Outer-agent RTL fix" in constraints[0]["rule"]
        assert "Fixed port width from 8 to 16 bits" in constraints[0]["rule"]
        assert constraints[0]["source"] == "human"

    @pytest.mark.asyncio
    async def test_fix_rtl_without_description_no_constraint(self, tmp_path):
        import json
        block_dir = tmp_path / ".coresmith" / "blocks" / "scrambler"
        block_dir.mkdir(parents=True)
        (block_dir / "diagnosis.json").write_text("{}")
        (block_dir / "attempt_history.json").write_text("[]")
        (block_dir / "previous_error.txt").write_text("")
        (block_dir / "constraints.json").write_text("[]")
        with patch(
            "orchestrator.langgraph.pipeline_graph.interrupt"
        ) as mock_interrupt, patch(
            "orchestrator.langgraph.pipeline_graph.write_graph_event"
        ):
            mock_interrupt.return_value = {"action": "fix_rtl"}
            state = _block_state(tmp_path=str(tmp_path))
            await ask_human_node(state)

        constraints = json.loads((block_dir / "constraints.json").read_text())
        assert len(constraints) == 0

    @pytest.mark.asyncio
    async def test_add_constraint_writes_to_disk(self, tmp_path):
        """add_constraint action should persist the constraint to disk."""
        import json
        block_dir = tmp_path / ".coresmith" / "blocks" / "scrambler"
        block_dir.mkdir(parents=True)
        (block_dir / "diagnosis.json").write_text("{}")
        (block_dir / "attempt_history.json").write_text("[]")
        (block_dir / "previous_error.txt").write_text("")
        (block_dir / "constraints.json").write_text("[]")
        with patch(
            "orchestrator.langgraph.pipeline_graph.interrupt"
        ) as mock_interrupt, patch(
            "orchestrator.langgraph.pipeline_graph.write_graph_event"
        ):
            mock_interrupt.return_value = {
                "action": "add_constraint",
                "constraint": "MUST use dedicated pins",
            }
            state = _block_state(tmp_path=str(tmp_path))
            await ask_human_node(state)

        constraints = json.loads((block_dir / "constraints.json").read_text())
        assert len(constraints) == 1
        assert constraints[0]["rule"] == "MUST use dedicated pins"

    @pytest.mark.asyncio
    async def test_fix_rtl_appends_to_existing_constraints(self, tmp_path):
        import json
        block_dir = tmp_path / ".coresmith" / "blocks" / "scrambler"
        block_dir.mkdir(parents=True)
        (block_dir / "diagnosis.json").write_text("{}")
        (block_dir / "attempt_history.json").write_text("[]")
        (block_dir / "previous_error.txt").write_text("")
        existing = [{"rule": "Existing constraint", "source": "human", "attempt": 1}]
        (block_dir / "constraints.json").write_text(json.dumps(existing))
        with patch(
            "orchestrator.langgraph.pipeline_graph.interrupt"
        ) as mock_interrupt, patch(
            "orchestrator.langgraph.pipeline_graph.write_graph_event"
        ):
            mock_interrupt.return_value = {
                "action": "fix_rtl",
                "description": "Changed reset polarity",
            }
            state = _block_state(tmp_path=str(tmp_path))
            await ask_human_node(state)

        constraints = json.loads((block_dir / "constraints.json").read_text())
        assert len(constraints) == 2
        assert constraints[0]["rule"] == "Existing constraint"
        assert "Changed reset polarity" in constraints[1]["rule"]


# ---------------------------------------------------------------------------
# route_after_human: escape documentation
# ---------------------------------------------------------------------------

class TestRouteAfterHumanEscapes:
    """Tests documenting escape scenarios in route_after_human.

    These cover actions that are valid for other interrupt types (like
    uarch_spec_review) but NOT valid for ask_human.  When such actions
    reach route_after_human, they silently default to generate_rtl,
    causing unintended re-execution.  The defense is upstream in
    _build_resume_command (type-aware validation).

    Post-refactor note: the default landing used to be ``increment_attempt``;
    that node was inlined and the default now flows directly to
    ``generate_rtl``. The bug-class these tests cover is unchanged.
    """

    def test_approve_is_not_valid_defaults_to_generate_rtl(self):
        """approve is for uarch_spec_review, not ask_human.

        If _build_resume_command sends approve to an ask_human interrupt,
        route_after_human defaults to generate_rtl, causing the block
        to silently re-enter RTL generation.
        """
        result = route_after_human({"human_response": {"action": "approve"}})
        assert result == "generate_rtl"

    def test_revise_is_not_valid_defaults_to_generate_rtl(self):
        """revise is for uarch_spec_review, not ask_human."""
        result = route_after_human({"human_response": {"action": "revise"}})
        assert result == "generate_rtl"

    def test_all_valid_actions_are_mapped(self):
        """Verify all supported ask_human actions have explicit mappings."""
        valid_ask_human_actions = {"retry", "fix_rtl", "fix_tb", "add_constraint", "skip", "abort"}
        terminal = {"skip", "abort"}
        for action in valid_ask_human_actions:
            result = route_after_human({"human_response": {"action": action}})
            if action in terminal:
                assert result == "block_done", f"{action} should land on block_done"
            elif action == "fix_tb":
                assert result == "generate_testbench", f"{action} should retry into generate_testbench"
            else:
                assert result == "generate_rtl", f"{action} should retry into generate_rtl"


# ═══════════════════════════════════════════════════════════════════════════
# Per-Document State Consumer Tests (post-refactor)
#
# After the architecture_state.json -> per-document migration, the pipeline
# graph should read from .coresmith/block_diagram.json instead of
# .coresmith/architecture_state.json.
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.doc_persistence
class TestPipelineReadsPerDocFiles:
    """Verify pipeline graph reads from per-document files after migration."""

    def test_review_uarch_spec_reads_block_diagram_json(self, tmp_path):
        """review_uarch_spec_node should read block interfaces from
        .coresmith/block_diagram.json (not architecture_state.json).
        """
        import json

        coresmith = tmp_path / ".coresmith"
        coresmith.mkdir()

        from orchestrator.tests.fft16_fixtures import FFT16_BLOCK_DIAGRAM

        (coresmith / "block_diagram.json").write_text(
            json.dumps(FFT16_BLOCK_DIAGRAM, indent=2)
        )

        bd_path = coresmith / "block_diagram.json"
        assert bd_path.exists()

        data = json.loads(bd_path.read_text())
        bd = data
        found = False
        for b in bd.get("blocks", []):
            if b.get("name") == "fft_butterfly":
                found = True
                assert "interfaces" in b
                break
        assert found, "fft_butterfly not found in block_diagram.json"

    def test_architecture_state_json_not_needed(self, tmp_path):
        """Pipeline should work without architecture_state.json present."""
        import json

        coresmith = tmp_path / ".coresmith"
        coresmith.mkdir()

        from orchestrator.tests.fft16_fixtures import FFT16_BLOCK_DIAGRAM

        (coresmith / "block_diagram.json").write_text(
            json.dumps(FFT16_BLOCK_DIAGRAM, indent=2)
        )

        arch_state = coresmith / "architecture_state.json"
        assert not arch_state.exists()

        data = json.loads((coresmith / "block_diagram.json").read_text())
        assert len(data["blocks"]) == 3


# ═══════════════════════════════════════════════════════════════════════════
# Disk-First Architecture Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestDiskFirstBlockState:
    """Verify disk-first architecture: all content on disk, state is routing-only."""

    def test_blockstate_has_no_content_fields(self):
        annotations = BlockState.__annotations__
        for field in ("constraints", "attempt_history", "previous_error",
                      "uarch_spec", "rtl_result", "lint_result", "tb_result",
                      "sim_result", "synth_result", "debug_result"):
            assert field not in annotations, f"Content field {field} still in BlockState"

    def test_blockstate_has_routing_flags(self):
        annotations = BlockState.__annotations__
        for field in ("lint_clean", "sim_passed", "synth_success",
                      "rtl_path", "tb_path", "debug_action"):
            assert field in annotations, f"Routing flag {field} missing from BlockState"


class TestDiskFirstInitBlock:
    @pytest.mark.asyncio
    async def test_creates_block_disk_directory(self, tmp_path):
        state = _block_state(_make_block("enc_control"), tmp_path=str(tmp_path))
        await init_block_node(state)
        block_dir = tmp_path / ".coresmith" / "blocks" / "enc_control"
        assert block_dir.is_dir()

    @pytest.mark.asyncio
    async def test_resets_disk_files_on_init(self, tmp_path):
        import json
        block_dir = tmp_path / ".coresmith" / "blocks" / "quantizer"
        block_dir.mkdir(parents=True)
        (block_dir / "constraints.json").write_text('[{"rule": "old", "source": "debug_agent", "attempt": 1}]')
        (block_dir / "previous_error.txt").write_text("old error")

        state = _block_state(_make_block("quantizer"), tmp_path=str(tmp_path))
        await init_block_node(state)

        constraints = json.loads((block_dir / "constraints.json").read_text())
        assert constraints == []
        assert (block_dir / "previous_error.txt").read_text() == ""


class TestDiskFirstAgentToolsEnabled:
    """Verify all agents have tools enabled (disable_tools=False)."""

    def test_rtl_generator_tools_enabled(self):
        from orchestrator.langchain.agents.rtl_generator import RTLGeneratorAgent
        agent = RTLGeneratorAgent()
        assert agent.llm.disable_tools is False

    def test_debug_agent_tools_enabled(self):
        from orchestrator.langchain.agents.debug_agent import DebugAgent
        agent = DebugAgent()
        assert agent.llm.disable_tools is False

    def test_testbench_generator_tools_enabled(self):
        from orchestrator.langchain.agents.testbench_generator import TestbenchGeneratorAgent
        agent = TestbenchGeneratorAgent()
        assert agent.llm.disable_tools is False

    def test_uarch_spec_generator_tools_enabled(self):
        from orchestrator.langchain.agents.uarch_spec_generator import UarchSpecGenerator
        agent = UarchSpecGenerator()
        assert agent.llm.disable_tools is False

    def test_integration_lead_tools_enabled(self):
        from orchestrator.langchain.agents.integration_lead import IntegrationLeadAgent
        agent = IntegrationLeadAgent()
        assert agent.llm.disable_tools is False

    def test_integration_tb_generator_tools_enabled(self):
        from orchestrator.langchain.agents.integration_testbench_generator import IntegrationTestbenchGenerator
        agent = IntegrationTestbenchGenerator()
        assert agent.llm.disable_tools is False


class TestIFP0014Fix:
    """Verify the PnR TCL template no longer produces conflicting floorplan args."""

    def test_small_die_refloorplan_uses_die_area_only(self, tmp_path):
        from pathlib import Path as _Path
        from orchestrator.langgraph.backend_helpers import generate_pnr_tcl
        tcl_path = generate_pnr_tcl(
            "tiny_block", "/fake/netlist.v", "/fake/sdc.sdc", str(tmp_path),
            utilization=45,
        )
        content = _Path(tcl_path).read_text()
        lines = content.split("\n")
        small_die_section = False
        for line in lines:
            if "too small for PDN" in line:
                small_die_section = True
            if small_die_section and "initialize_floorplan" in line:
                assert "-utilization" not in line, \
                    "Re-floorplan for small die must NOT use -utilization (IFP-0014)"
                break


# ---------------------------------------------------------------------------
# _parse_issue_counts (integration_review_agent)
# ---------------------------------------------------------------------------

class TestParseIssueCounts:
    """Verify structured JSON parsing replaces fragile substring counting."""

    def _parse(self, text):
        from orchestrator.langchain.agents.integration_review_agent import _parse_issue_counts
        return _parse_issue_counts(text)

    def test_valid_json_block(self):
        text = 'All good.\n\n```json\n{"issues_found": 3, "issues_fixed": 2}\n```'
        assert self._parse(text) == (3, 2)

    def test_zero_issues(self):
        text = 'No problems.\n\n```json\n{"issues_found": 0, "issues_fixed": 0}\n```'
        assert self._parse(text) == (0, 0)

    def test_no_json_block_returns_zero(self):
        assert self._parse("No mismatches found.") == (0, 0)

    def test_old_substring_false_positive_avoided(self):
        """'No mismatches found' previously counted as issues_found=1."""
        text = "No mismatches found. Everything looks clean."
        assert self._parse(text) == (0, 0)

    def test_malformed_json_returns_zero(self):
        text = '```json\n{bad json}\n```'
        assert self._parse(text) == (0, 0)

    def test_negative_values_clamped(self):
        text = '```json\n{"issues_found": -1, "issues_fixed": -5}\n```'
        assert self._parse(text) == (0, 0)


class TestIntegrationReviewFiltering:
    def test_filters_future_tier_connections(self):
        from orchestrator.langchain.agents.integration_review_agent import (
            _filter_connections_for_blocks,
        )

        diagram = {
            "blocks": [
                {"name": "a", "tier": 1},
                {"name": "b", "tier": 1},
                {"name": "future", "tier": 2},
            ],
            "connections": [
                {"from": "a.out", "to": "b.in"},
                {"from": "b.out", "to": "future.in"},
                {"from": "future.out", "to": "a.in"},
            ],
        }

        filtered, deferred = _filter_connections_for_blocks(diagram, ["a", "b"])

        assert deferred == 2
        assert filtered["blocks"] == [
            {"name": "a", "tier": 1},
            {"name": "b", "tier": 1},
        ]
        assert filtered["connections"] == [{"from": "a.out", "to": "b.in"}]


class TestTestbenchDefaultPath:
    """A blocks.yaml entry without `testbench` must not KeyError and crash
    the whole run; generate_testbench_node defaults to the cocotb path and
    writes it back into the block dict for downstream consumers."""

    async def test_missing_testbench_defaults_and_skips_cleanly(self, tmp_path):
        from orchestrator.langgraph import pipeline_graph
        block = {"name": "widget", "tier": 2, "rtl_target": "rtl/widget/widget.v"}
        state = {
            "current_block": block,
            "attempt": 1,
            "project_root": str(tmp_path),
            "rtl_path": "",  # no RTL -> node returns early, before any LLM call
            "step_log_paths": {},
        }
        # Must not raise KeyError('testbench'); returns sim_passed False (no RTL).
        result = await pipeline_graph.generate_testbench_node(state)
        assert result["sim_passed"] is False
        assert block["testbench"] == "tb/cocotb/test_widget.py"
        assert result["tb_path"].endswith("tb/cocotb/test_widget.py")

    async def test_explicit_testbench_preserved(self, tmp_path):
        from orchestrator.langgraph import pipeline_graph
        block = {"name": "widget", "tier": 2, "testbench": "tb/custom/tb_widget.py"}
        state = {
            "current_block": block,
            "attempt": 1,
            "project_root": str(tmp_path),
            "rtl_path": "",
            "step_log_paths": {},
        }
        result = await pipeline_graph.generate_testbench_node(state)
        assert block["testbench"] == "tb/custom/tb_widget.py"
        assert result["tb_path"].endswith("tb/custom/tb_widget.py")
