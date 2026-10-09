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


import json
from unittest.mock import AsyncMock, patch

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from orchestrator.langgraph import pipeline_graph
from orchestrator.langgraph.pipeline_graph import (
    BlockState,
    OrchestratorState,
    ask_human_node,
    block_done_node,
    build_block_subgraph,
    build_pipeline_graph,
    init_block_node,
    integration_dv_decision_node,
    integration_dv_node,
    pipeline_complete_node,
    route_after_human,
    route_after_integration_dv,
    route_after_integration_dv_decision,
    route_after_integration_review,
    route_after_uarch_review,
    route_after_validation_dv,
    route_after_validation_dv_decision,
    route_decision,
    route_next_tier,
    validation_dv_decision_node,
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


@pytest.fixture(autouse=True)
def _legacy_uarch_gate_off(monkeypatch):
    # These end-to-end fakes never build a SystemC model; the B2 phase gate
    # (default on) would park them before the interrupts they exercise.
    monkeypatch.setenv("CORESMITH_UARCH_PHASE_GATE", "0")


def _setup_disk_fixtures(tmp_path, blocks: list[dict]) -> None:
    """Create the on-disk fixtures (rtl, tb, .coresmith/blocks/<name>/...)
    expected by the disk-first pipeline nodes for the given blocks.
    """
    for blk in blocks:
        name = blk["name"]
        spec_path = tmp_path / "arch" / "uarch_specs" / f"{name}.md"
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(f"# {name}\n\n## Interfaces\nNo external ports.\n")
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


    from orchestrator.state_store.trust import capture_run_baseline
    capture_run_baseline(tmp_path)


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
            "integration_dv_decision", "validation_dv_decision",
        ]
        for name in expected:
            assert name in node_names, f"Missing orchestrator node: {name}"

    def test_final_report_node_present_and_before_end(self):
        # The signoff scorecard is a terminal funnel: final_report -> END, and
        # the genuine terminals (validation_dv / integration_dv /
        # pipeline_complete) route their END sentinel into final_report rather
        # than straight to END. Assert on the builder (the drawable get_graph()
        # collapses conditional edges); builder.branches[src].ends is the map.
        graph = build_pipeline_graph(checkpointer=MemorySaver())
        b = graph.builder
        END = pipeline_graph.END
        assert "final_report" in b.nodes
        assert (("final_report", END) in b.edges), b.edges

        def _ends(src):
            merged = {}
            for br in b.branches.get(src, {}).values():
                merged.update(br.ends or {})
            return merged

        for src in ("validation_dv", "integration_dv", "pipeline_complete"):
            assert _ends(src).get(END) == "final_report", (src, _ends(src))

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

        # run3-followups two-phase flow: the DV node itself NEVER calls
        # interrupt() (the one-cycle-late defect) -- it parks with the
        # payload in state; the dedicated decision node consumes the
        # response.
        dv_result = result["integration_dv_result"]
        assert dv_result["passed"] is False
        assert dv_result["phase"] == "tb_generation"
        assert dv_result["pending_decision"] is True
        assert result["pipeline_done"] is False
        assert not interrupts, "DV node must not call interrupt() directly"
        payload = dv_result["interrupt_payload"]
        assert payload["phase"] == "tb_generation"
        assert payload["contract_audit"]["category"] == "INTEGRATION_TB_BUG"
        assert route_after_integration_dv(result) == "integration_dv_decision"

        decision = await integration_dv_decision_node({
            "project_root": str(tmp_path),
            "integration_dv_result": dv_result,
        })
        assert interrupts and interrupts[0] is payload
        d_result = decision["integration_dv_result"]
        assert d_result["action_taken"] == "fix_tb"
        assert d_result["pending_decision"] is False
        assert d_result["fix_applied"] == "repair generator"
        assert route_after_integration_dv_decision(decision) == "integration_dv"
        assert route_after_integration_dv(decision) == "integration_dv"

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
        # WP-76: an RTL fix invalidates the adopted candidate, so it re-enters
        # the integration check (re-assembly + re-adoption) before any sim;
        # a testbench fix re-runs validation DV directly.
        result = route_after_validation_dv({
            "validation_dv_result": {"passed": False, "action_taken": "fix_rtl"},
        })
        assert result == "integration_check"
        result = route_after_validation_dv({
            "validation_dv_result": {"passed": False, "action_taken": "fix_tb"},
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

        # run3-followups two-phase flow (see the integration variant).
        dv_result = result["validation_dv_result"]
        assert dv_result["passed"] is False
        assert dv_result["phase"] == "tb_generation"
        assert dv_result["pending_decision"] is True
        assert not interrupts, "DV node must not call interrupt() directly"
        payload = dv_result["interrupt_payload"]
        assert payload["phase"] == "tb_generation"
        assert payload["contract_audit"]["category"] == "VALIDATION_TB_BUG"
        assert route_after_validation_dv(result) == "validation_dv_decision"

        decision = await validation_dv_decision_node({
            "project_root": str(tmp_path),
            "validation_dv_result": dv_result,
        })
        assert interrupts and interrupts[0] is payload
        d_result = decision["validation_dv_result"]
        assert d_result["action_taken"] == "fix_tb"
        assert d_result["pending_decision"] is False
        assert route_after_validation_dv_decision(decision) == "validation_dv"
        assert route_after_validation_dv(decision) == "validation_dv"

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
    @pytest.mark.asyncio
    async def test_resume_does_not_rerun_model_backed_review(self, tmp_path, monkeypatch):
        """A real LangGraph interrupt/resume approves the checkpointed review."""
        from langgraph.graph import END, START, StateGraph

        from orchestrator.langchain.agents import integration_review_agent

        spec = tmp_path / "arch" / "uarch_specs" / "leaf.md"
        spec.parent.mkdir(parents=True)
        spec.write_text("REVISION A")
        calls = []

        def fake_init(self, *args, **kwargs):
            pass

        async def fake_review(self, block_names, project_root):
            calls.append(len(calls) + 1)
            # A replay would expose a different revision and finding.
            if len(calls) > 1:
                spec.write_text("REVISION B")
                return {"summary": "new finding", "issues_found": 1,
                        "issues_fixed": 1, "edited_blocks": ["leaf"],
                        "reviewed_specs": {"leaf": str(spec)}}
            return {"summary": "clean A", "issues_found": 0,
                    "issues_fixed": 0, "edited_blocks": [],
                    "reviewed_specs": {"leaf": str(spec)}}

        monkeypatch.delenv("CORESMITH_ENABLE_CHIP_LEAD", raising=False)
        monkeypatch.setattr(integration_review_agent.IntegrationReviewAgent,
                            "__init__", fake_init)
        monkeypatch.setattr(integration_review_agent.IntegrationReviewAgent,
                            "review", fake_review)
        builder = StateGraph(OrchestratorState)
        builder.add_node("prepare", pipeline_graph.integration_review_prepare_node)
        builder.add_node("decision", pipeline_graph.integration_review_node)
        builder.add_edge(START, "prepare")
        builder.add_edge("prepare", "decision")
        builder.add_edge("decision", END)
        graph = builder.compile(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "review-replay"}}
        state = {"project_root": str(tmp_path), "block_queue": [{"name": "leaf", "tier": 1}],
                 "tier_list": [1], "current_tier_index": 0, "completed_blocks": []}

        await graph.ainvoke(state, config)
        result = await graph.ainvoke(Command(resume={"action": "approve"}), config)

        assert calls == [1]
        assert spec.read_text() == "REVISION A"
        assert result["integration_review_action"] == "approve"

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
    @pytest.mark.parametrize("strict", ["0", "1"])
    async def test_explicit_approve_adopts_then_reverifies_without_review_loop(self, tmp_path, monkeypatch, strict):
        from orchestrator.tests.test_wp62_spec_adoption import setup
        state, canonical, source, db, review = setup(tmp_path, monkeypatch, "success")
        monkeypatch.setenv("CORESMITH_STRICT_INTEGRATION_REVIEW", strict)
        result = await pipeline_graph.integration_review_node(state)
        assert result["integration_review_action"] == "approve"
        assert not result["integration_review_failed"]
        assert canonical.read_text() == "NEW SPEC"
        assert db.result("leaf", "best") is None
        assert pipeline_graph.route_after_integration_review(result) == "init_tier"


class TestRouteAfterIntegration:

    def test_lint_failure_ends(self):
        assert pipeline_graph.route_after_integration({
            "integration_result": {"lint_clean": False, "error_count": 0}
        }) == "__end__"

    def test_error_count_ends(self):
        assert pipeline_graph.route_after_integration({
            "integration_result": {"lint_clean": True, "error_count": 1}
        }) == "__end__"


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
    async def test_init_block_keeps_regeneration_proof_constraints(self, tmp_path):
        # The chip-level revise/fix pins and operator rules are the whole point
        # of constraints.json surviving regeneration; only the per-lifecycle
        # (debug-agent) entries are dropped.
        block_dir = tmp_path / ".coresmith" / "blocks" / "scrambler"
        block_dir.mkdir(parents=True)
        (block_dir / "constraints.json").write_text(json.dumps([
            {"rule": "INTEGRATION_DV REVISION", "source": "chip_dv_revise"},
            {"rule": "keep the fix", "source": "chip_dv_fix"},
            {"rule": "operator rule", "source": "human"},
            {"rule": "per-attempt hint", "source": "debug_agent"},
        ]))
        state = _block_state(_make_block("scrambler"), tmp_path=str(tmp_path))
        await init_block_node(state)
        kept = json.loads((block_dir / "constraints.json").read_text())
        assert [c["source"] for c in kept] == [
            "chip_dv_revise", "chip_dv_fix", "human"]

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
    async def test_pipeline_complete(self, tmp_path):
        # Per-block frontend completion sets frontend_complete, NOT pipeline_done
        # (fix #5): the deliverable is the verified chip_top, so pipeline_done
        # stays False until integration_dv + validation_dv (+ chip-top synth).
        # Setting pipeline_done here is the leak that let a parked-at-integration
        # run report as "done". (A project root without a database: the
        # recorded-build receipts are checked only where a ledger exists.)
        state = {"project_root": str(tmp_path), "completed_blocks": [{"name": "a", "success": True}]}
        result = await pipeline_complete_node(state)
        assert result["frontend_complete"] is True
        assert result.get("pipeline_done") is not True

    @pytest.mark.asyncio
    async def test_pipeline_complete_retry_does_not_proceed(self, tmp_path, monkeypatch):
        # retry at the incomplete gate must NEVER advance to integration with a
        # failed block. New (default) behavior: retry re-validates the blocks
        # (revalidate_pending -> init_tier), not pipeline_done/integration_check.
        monkeypatch.delenv("CORESMITH_REVALIDATE_INCOMPLETE", raising=False)
        monkeypatch.setattr(pipeline_graph, "interrupt", lambda payload: {"action": "retry"})
        monkeypatch.setattr(pipeline_graph, "write_graph_event", lambda *a, **k: None)
        incomplete = {
            "project_root": str(tmp_path),
            "completed_blocks": [
                {"name": "a", "success": True},
                {"name": "b", "success": False},
            ],
            "block_queue": [{"name": "a"}, {"name": "b"}],
        }
        result = await pipeline_complete_node(dict(incomplete))
        assert result["pipeline_done"] is False
        assert result.get("revalidate_pending") is True
        assert pipeline_graph._pipeline_complete_route(result) != "integration_check"

        # With re-validation disabled, retry restores the old abort behavior.
        monkeypatch.setenv("CORESMITH_REVALIDATE_INCOMPLETE", "0")
        result_off = await pipeline_complete_node(dict(incomplete))
        assert result_off["pipeline_done"] is False
        assert result_off["pipeline_aborted"] is True

    @pytest.mark.asyncio
    async def test_block_done_without_a_recorded_build_is_not_a_success(self, tmp_path):
        """sim + synth passed in memory, but the lifecycle carries no recorded
        build and runs outside a checkpointed graph thread: nothing is
        published and the block is not done (the staged-build contract)."""
        state = _block_state(_make_block("scrambler"), tmp_path=str(tmp_path))
        state["sim_passed"] = True
        state["synth_success"] = True
        state["synth_gate_count"] = 1500
        block_dir = tmp_path / ".coresmith" / "blocks" / "scrambler"
        block_dir.mkdir(parents=True, exist_ok=True)
        (block_dir / "constraints.json").write_text("[]")
        result = await block_done_node(state)
        assert len(result["completed_blocks"]) == 1
        done = result["completed_blocks"][0]
        assert done["success"] is False and done["name"] == "scrambler"
        assert done["persistence_failed"] is True and "BUILD_IDENTITY_MISSING" in done["error"]

    @pytest.mark.asyncio
    async def test_block_done_cannot_record_failed_timing_as_success(self, tmp_path):
        state = _block_state(_make_block("scrambler"), tmp_path=str(tmp_path))
        state.update(sim_passed=True, synth_success=True, timing_ok=False)
        block_dir = tmp_path / ".coresmith" / "blocks" / "scrambler"
        block_dir.mkdir(parents=True, exist_ok=True)
        (block_dir / "constraints.json").write_text("[]")
        (block_dir / "previous_error.txt").write_text(
            "PPA gate: STA ran but produced no parseable timing"
        )

        result = await block_done_node(state)

        completed = result["completed_blocks"][0]
        assert completed["success"] is False
        assert completed["synth_success"] is True

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
        from orchestrator.langchain.agents.integration_testbench_generator import (
            IntegrationTestbenchGenerator,
        )
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


# ---------------------------------------------------------------------------
# Two-pass µarch-model restructure (CORESMITH_BLOCK_GOLDENS)
# ---------------------------------------------------------------------------

# Today's (single-pass) topology -- the flag-off invariant. If a legitimate
# graph change adds/removes a node or top-level edge with the flag OFF, update
# this baseline AND justify why the flag-off topology changed.
_SINGLE_PASS_NODES = {
    "__start__", "__end__",
    "init_tier", "process_block", "integration_review", "advance_tier",
    "pipeline_complete", "integration_check", "model_integration",
    "integration_dv", "validation_dv",
    # run3-followups: dedicated interrupt-decision nodes (resume responses are
    # consumed immediately, not one regeneration cycle late).
    "integration_dv_decision", "validation_dv_decision",
    # Deterministic signoff-scorecard node: the pre-END funnel for the genuine
    # terminals (validation_dv done / integration_dv terminal-fail /
    # pipeline_complete abort). Shared by both topologies.
    "final_report",
}
_SINGLE_PASS_BLOCK_NODES = {
    "__start__", "__end__",
    "init_block", "generate_uarch_spec", "review_uarch_spec",
    "generate_rtl", "generate_testbench", "synthesize",
    "diagnose", "decide", "ask_human", "block_done",
}


class TestTwoPassTopology:
    """Flag ON => the two-pass graph: µarch gate between fan-outs, no
    post-integration_check model_integration node."""


    def test_route_after_integration_flag_on_goes_to_dv(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_BLOCK_GOLDENS", "1")
        assert pipeline_graph.route_after_integration({
            "integration_result": {"lint_clean": True, "error_count": 0}
        }) == "integration_dv"


class TestRouteAfterUarchReviewPhase:

    def test_rtl_phase_approve_goes_to_generate_rtl(self):
        assert route_after_uarch_review(
            {"pipeline_phase": "rtl", "human_response": {"action": "approve"}}
        ) == "generate_rtl"

    def test_flag_off_default_approve_goes_to_generate_rtl(self):
        assert route_after_uarch_review(
            {"human_response": {"action": "approve"}}
        ) == "generate_rtl"

    def test_revise_always_respecs(self):
        assert route_after_uarch_review(
            {"pipeline_phase": "uarch", "human_response": {"action": "revise"}}
        ) == "generate_uarch_spec"


class TestRouteNextTierPhase:

    def test_rtl_phase_exhausted_goes_to_complete(self):
        assert route_next_tier({
            "tier_list": [1], "current_tier_index": 1,
            "completed_blocks": [], "pipeline_phase": "rtl",
        }) == "pipeline_complete"

    def test_flag_off_default_goes_to_complete(self):
        assert route_next_tier({
            "tier_list": [1], "current_tier_index": 1, "completed_blocks": [],
        }) == "pipeline_complete"

    def test_more_tiers_goes_to_init_tier_in_uarch_phase(self):
        assert route_next_tier({
            "tier_list": [1, 2], "current_tier_index": 1,
            "completed_blocks": [], "pipeline_phase": "uarch",
        }) == "init_tier"


class TestReducerNoDoubleCount:
    """completed_blocks accumulates BOTH passes; consumers filter to the
    current phase + dedup by name."""

    def test_current_phase_completed_filters_by_phase(self):
        state = {
            "pipeline_phase": "rtl",
            "completed_blocks": [
                {"name": "a", "success": True, "phase": "uarch"},
                {"name": "b", "success": True, "phase": "uarch"},
                {"name": "a", "success": True, "phase": "rtl"},
                {"name": "b", "success": True, "phase": "rtl"},
            ],
        }
        cur = pipeline_graph._current_phase_completed(state)
        assert {b["name"] for b in cur} == {"a", "b"}
        assert all(b["phase"] == "rtl" for b in cur)
        assert len(cur) == 2  # no double-count


    def test_phaseless_entries_always_kept(self):
        # Flag-off / legacy entries with no phase key are kept regardless.
        state = {
            "pipeline_phase": "rtl",
            "completed_blocks": [
                {"name": "a", "success": True},
                {"name": "b", "success": False},
            ],
        }
        cur = pipeline_graph._current_phase_completed(state)
        assert len(cur) == 2

    def test_dedup_keeps_last(self):
        state = {
            "pipeline_phase": "rtl",
            "completed_blocks": [
                {"name": "a", "success": False, "phase": "rtl"},
                {"name": "a", "success": True, "phase": "rtl"},
            ],
        }
        cur = pipeline_graph._current_phase_completed(state)
        assert len(cur) == 1
        assert cur[0]["success"] is True


class TestGateFeedbackThreading:
    """Engine Fix #5: a gate-triggered re-spec (route_after_uarch_gate ->
    init_tier with a FAILED model_integration_result in scope) must thread the
    gate's divergence diagnosis to disk (.coresmith/blocks/<b>/gate_feedback.txt)
    for the implicated block(s), which generate_uarch_spec_node then folds into
    the re-spec. Disk-first (no state content field). Without this the Fix #4
    bounded re-spec loop just redraws identical blocks and exhausts."""

    def _state(self, tmp_path, mir):
        return {
            "project_root": str(tmp_path),
            "target_clock_mhz": 50.0,
            "max_attempts": 3,
            "block_queue": [
                {"name": "frame_ctrl", "tier": 1},
                {"name": "other_blk", "tier": 1},
            ],
            "tier_list": [1],
            "current_tier_index": 0,
            "pipeline_phase": "uarch",
            "uarch_pass_done": False,
            "model_integration_result": mir,
        }

    def _fb_path(self, tmp_path, name):
        return tmp_path / ".coresmith" / "blocks" / name / "gate_feedback.txt"


    @pytest.mark.asyncio
    async def test_clearing_preserves_other_writers_feedback(self, tmp_path):
        # A sibling node (uarch_patch-on-retry) delivers its prescription
        # through the SAME file; init_tier must not delete it before
        # generate_uarch_spec reads it.
        fb = self._fb_path(tmp_path, "frame_ctrl")
        fb.parent.mkdir(parents=True)
        fb.write_text("MICROARCH REVISION (auto-applied from a high-confidence "
                      "diagnose uarch_patch, confidence 0.95): ...")
        await pipeline_graph.init_tier_node(self._state(tmp_path, None))
        assert fb.exists()
        assert "MICROARCH REVISION" in fb.read_text()

    @pytest.mark.asyncio
    async def test_advisory_bypass_is_not_a_gate_failure(self, tmp_path):
        # The advisory bypass reports passed=False on purpose -- it must NOT
        # broadcast "gate FAILED" re-spec feedback to every tier block.
        mir = {"passed": False, "advisory_bypass": True,
               "gap_class": "block_math"}
        await pipeline_graph.init_tier_node(self._state(tmp_path, mir))
        for name in ("frame_ctrl", "other_blk"):
            assert not self._fb_path(tmp_path, name).exists()

    @pytest.mark.asyncio
    async def test_passed_gate_means_no_feedback(self, tmp_path):
        mir = {"passed": True, "first_divergence_block": "frame_ctrl"}
        await pipeline_graph.init_tier_node(self._state(tmp_path, mir))
        assert not self._fb_path(tmp_path, "frame_ctrl").exists()

    @pytest.mark.asyncio
    async def test_generate_uarch_spec_node_consumes_disk_feedback(
        self, tmp_path, monkeypatch
    ):
        # The node must fold disk gate-feedback into the generator call (and read
        # previous_spec so it's a revision, not a fresh gen).
        captured = {}

        async def _fake_gen(block, **kw):
            captured["feedback"] = kw.get("feedback", "")
            captured["previous_spec"] = kw.get("previous_spec", "")
            return {"spec_text": "## revised", "spec_summary": {}}
        monkeypatch.setattr(pipeline_graph, "generate_uarch_spec", _fake_gen)

        spec_dir = tmp_path / "arch" / "uarch_specs"
        spec_dir.mkdir(parents=True)
        (spec_dir / "frame_ctrl.md").write_text("## old spec")
        fb = self._fb_path(tmp_path, "frame_ctrl")
        fb.parent.mkdir(parents=True)
        fb.write_text("gate says: derive geometry from sideband")

        await pipeline_graph.generate_uarch_spec_node({
            "project_root": str(tmp_path),
            "current_block": {"name": "frame_ctrl"},
            "human_response": None,
        })
        assert "derive geometry from sideband" in captured["feedback"]
        assert captured["previous_spec"] == "## old spec"


class TestInitTierInitializesPhase:
    """init_tier_node is the orchestrator entry (START -> init_tier).  On the
    FIRST entry of a block-goldens run it must seed pipeline_phase="uarch" so
    pass 1 (spec+model only) is actually entered; otherwise fan_out_tier's
    ``state.get("pipeline_phase", "rtl")`` default sends every block down the
    single-pass RTL path and the two-fan-out restructure never engages.  Flag
    off must stay byte-identical (no pipeline_phase key added).  A pass-2
    re-entry (begin_rtl_pass already wrote "rtl") must NOT be clobbered back."""

    def _state(self):
        return {
            "block_queue": [{"name": "a", "tier": 1}],
            "current_tier_index": 0,
            "project_root": "/tmp/x",
        }


    @pytest.mark.asyncio
    async def test_first_entry_no_phase_key_when_flag_off(self, monkeypatch):
        monkeypatch.delenv("CORESMITH_BLOCK_GOLDENS", raising=False)
        out = await pipeline_graph.init_tier_node(self._state())
        # Byte-identical single-pass: do not introduce a phase; downstream
        # fan_out_tier defaults to "rtl".
        assert out.get("pipeline_phase", "rtl") == "rtl"

    @pytest.mark.asyncio
    async def test_does_not_override_existing_rtl_phase(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_BLOCK_GOLDENS", "1")
        st = self._state()
        st["pipeline_phase"] = "rtl"  # pass-2 re-entry written by begin_rtl_pass
        out = await pipeline_graph.init_tier_node(st)
        assert out.get("pipeline_phase", "rtl") == "rtl"


class TestIntegrationReviewSkipsPass1:
    """In two-pass pass 1 the per-tier LLM integration review is redundant
    (the uarch_integration_gate validates cross-block coherence after all
    tiers) and costly (LangGraph re-runs the reviewer LLM on every resume,
    which can hang). integration_review_node must skip -> auto-approve in pass
    1 without constructing the reviewer agent, and run normally in pass 2
    (phase "rtl") and when the flag is off."""

    def _state(self, phase):
        st = {
            "project_root": "/tmp/x",
            "block_queue": [{"name": "a", "tier": 1}],
            "tier_list": [1],
            "current_tier_index": 0,
        }
        if phase is not None:
            st["pipeline_phase"] = phase
        return st

    def _ban_agent(self, monkeypatch):
        """Make constructing the reviewer agent an error -- proves we skipped."""
        import orchestrator.langchain.agents.integration_review_agent as ira

        class _Boom:
            def __init__(self, *a, **k):
                raise AssertionError("reviewer agent must NOT run in pass 1")
        monkeypatch.setattr(ira, "IntegrationReviewAgent", _Boom)


    @pytest.mark.asyncio
    async def test_strict_mode_runs_reviewer_in_rtl_phase(self, monkeypatch):
        # CORESMITH_STRICT_INTEGRATION_REVIEW=1 restores the old per-tier review
        # even under block-goldens.
        monkeypatch.setenv("CORESMITH_BLOCK_GOLDENS", "1")
        monkeypatch.setenv("CORESMITH_STRICT_INTEGRATION_REVIEW", "1")
        constructed = {"n": 0}
        import orchestrator.langchain.agents.integration_review_agent as ira

        class _Fake:
            def __init__(self, *a, **k):
                constructed["n"] += 1

            async def review(self, **k):
                return {"summary": "ok", "issues_found": 0, "issues_fixed": 0}
        monkeypatch.setattr(ira, "IntegrationReviewAgent", _Fake)
        monkeypatch.setattr(pipeline_graph, "interrupt",
                            lambda payload: {"action": "approve"})
        out = await pipeline_graph.integration_review_node(self._state("rtl"))
        assert constructed["n"] == 1
        assert out["integration_review_action"] == "approve"

    @pytest.mark.asyncio
    async def test_runs_reviewer_when_flag_off(self, monkeypatch):
        monkeypatch.delenv("CORESMITH_BLOCK_GOLDENS", raising=False)
        constructed = {"n": 0}
        import orchestrator.langchain.agents.integration_review_agent as ira

        class _Fake:
            def __init__(self, *a, **k):
                constructed["n"] += 1

            async def review(self, **k):
                return {"summary": "ok", "issues_found": 0, "issues_fixed": 0}
        monkeypatch.setattr(ira, "IntegrationReviewAgent", _Fake)
        monkeypatch.setattr(pipeline_graph, "interrupt",
                            lambda payload: {"action": "approve"})
        await pipeline_graph.integration_review_node(self._state("uarch"))
        assert constructed["n"] == 1


# ---------------------------------------------------------------------------
# Two-pass end-to-end (toy, MemorySaver) -- verification items 2 & 7
# ---------------------------------------------------------------------------

class TestTwoPassEndToEnd:
    """A toy 2-block run with CORESMITH_BLOCK_GOLDENS=1 exercises the full
    two-pass topology on MemorySaver: pass 1 (spec only) -> µarch gate ->
    begin_rtl_pass (phase flip + index reset, once) -> pass 2 (RTL+synth) ->
    integration_check -> DV. RTL nodes fail loud if hit in phase uarch."""

    def _mocks(self, monkeypatch, tmp_path, gate_violations=None,
               gate_calls=None, integration_done=None):
        uarch_result, rtl_result, lint_clean, _, tb_result, sim_pass, _, synth_ok = (
            _patch_all_helpers()
        )

        def _uarch(block, **kw):
            return {
                "spec_text": f"## {block['name']}",
                "spec_summary": {"block_name": block["name"]},
                "spec_path": str(
                    tmp_path / "arch" / "uarch_specs" / f"{block['name']}.md"
                ),
                "block_name": block["name"],
            }

        def _rtl(block, *a, **kw):
            return {
                "verilog": f"module {block['name']}(); endmodule\n",
                "rtl_path": str(tmp_path / "rtl" / "dvbt" / f"{block['name']}.v"),
                "ports": {"clk": "input"},
            }

        def _tb(block, *a, **kw):
            return {
                "testbench": "# t",
                "testbench_path": str(
                    tmp_path / "tb" / "cocotb" / f"test_{block['name']}.py"
                ),
            }

        monkeypatch.setattr(pipeline_graph, "generate_uarch_spec",
                            AsyncMock(side_effect=_uarch))
        monkeypatch.setattr(pipeline_graph, "generate_rtl",
                            AsyncMock(side_effect=_rtl))
        monkeypatch.setattr(pipeline_graph, "lint_rtl", lambda *a, **k: lint_clean)
        monkeypatch.setattr(pipeline_graph, "generate_testbench",
                            AsyncMock(side_effect=_tb))
        monkeypatch.setattr(pipeline_graph, "run_simulation",
                            lambda *a, **k: sim_pass)
        monkeypatch.setattr(pipeline_graph, "synthesize_block",
                            lambda *a, **k: synth_ok)
        monkeypatch.setattr(pipeline_graph, "create_golden_model_wrapper",
                            lambda *a, **k: None)

        # Integration review: auto-approve (no chip-level interrupt).
        async def _review(_state):
            return {"integration_review_action": "approve",
                    "integration_review_failed": False}
        monkeypatch.setattr(pipeline_graph, "integration_review_node", _review)

        # µarch gate: count calls, return controlled violations (default clean).
        # run_model_integration_gate is SYNC (the node calls it via to_thread)
        # and is imported inside the node from the source module, so patch there.
        def _gate(_pr):
            if gate_calls is not None:
                gate_calls.append(1)
            return list(gate_violations or [])
        from orchestrator.architecture import model_integration as _mi
        monkeypatch.setattr(_mi, "run_model_integration_gate", _gate)

        async def _maybe_gen(_pr):
            return None
        monkeypatch.setattr(pipeline_graph, "_maybe_generate_chip_model",
                            _maybe_gen)

        # Terminal nodes: stop cleanly, no real EDA. Record which phase the
        # rtl-pass consumers see (must be "rtl").
        async def _icheck(state):
            if integration_done is not None:
                integration_done["check_phase"] = state.get("pipeline_phase")
                integration_done["check_completed"] = (
                    pipeline_graph._current_phase_completed(state)
                )
            return {"integration_result": {
                "lint_clean": True, "error_count": 0,
                "top_rtl_path": "", "design_name": "chip_top",
                "block_rtl_paths": {},
            }}
        monkeypatch.setattr(pipeline_graph, "integration_check_node", _icheck)

        async def _idv(_state):
            return {"integration_dv_result": {"passed": True, "skipped": True}}
        monkeypatch.setattr(pipeline_graph, "integration_dv_node", _idv)

        async def _vdv(_state):
            return {"validation_dv_result": {"passed": True, "skipped": True},
                    "pipeline_done": True}
        monkeypatch.setattr(pipeline_graph, "validation_dv_node", _vdv)


class TestRevalidateIncompleteGate:
    """Recoverable incomplete-gate / completion bookkeeping (CORESMITH_REVALIDATE_*).

    On a `retry` resume at the pipeline_incomplete gate, the graph should
    re-validate failed/missing blocks against the outer controller's on-disk RTL
    fixes (re-run rtl-phase tiers, recount) instead of dead-ending — bounded so a
    truly-failing block aborts. Both env branches covered per AGENTS.md.
    """

    # ---- pure routing decision ----
    def test_route_clean_to_integration(self):
        assert pipeline_graph._pipeline_complete_route(
            {"pipeline_done": True}) == "integration_check"

    def test_route_aborted_to_end(self):
        assert pipeline_graph._pipeline_complete_route(
            {"pipeline_aborted": True}) == "end"

    def test_route_revalidate_to_init_tier(self):
        assert pipeline_graph._pipeline_complete_route(
            {"revalidate_pending": True}) == "init_tier"

    def test_route_aborted_takes_precedence_over_revalidate(self):
        # defensive: abort wins even if a stale revalidate flag is present
        assert pipeline_graph._pipeline_complete_route(
            {"pipeline_aborted": True, "revalidate_pending": True}) == "end"

    # ---- env helpers ----
    def test_revalidate_enabled_default_on(self, monkeypatch):
        monkeypatch.delenv("CORESMITH_REVALIDATE_INCOMPLETE", raising=False)
        assert pipeline_graph._revalidate_enabled() is True

    def test_revalidate_disabled_when_zero(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_REVALIDATE_INCOMPLETE", "0")
        assert pipeline_graph._revalidate_enabled() is False

    def test_revalidate_max_default(self, monkeypatch):
        monkeypatch.delenv("CORESMITH_REVALIDATE_MAX", raising=False)
        assert pipeline_graph._revalidate_max() == 2

    def test_revalidate_max_custom_and_invalid(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_REVALIDATE_MAX", "5")
        assert pipeline_graph._revalidate_max() == 5
        monkeypatch.setenv("CORESMITH_REVALIDATE_MAX", "notanint")
        assert pipeline_graph._revalidate_max() == 2

    # ---- node behavior at the incomplete gate (mock interrupt) ----
    def _incomplete_state(self, tmp_path, attempts=0):
        return {
            "project_root": str(tmp_path),
            "pipeline_phase": "rtl",
            "block_queue": [
                {"name": "a", "tier": 1}, {"name": "b", "tier": 1},
                {"name": "c", "tier": 1},
            ],
            "completed_blocks": [
                {"name": "a", "success": True, "phase": "rtl"},
                {"name": "b", "success": True, "phase": "rtl"},
                {"name": "c", "success": False, "phase": "rtl"},
            ],
            "revalidate_attempts": attempts,
        }

    async def test_retry_revalidates_when_enabled(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CORESMITH_REVALIDATE_INCOMPLETE", raising=False)
        monkeypatch.setattr(pipeline_graph, "interrupt", lambda payload: {"action": "retry"})
        monkeypatch.setattr(pipeline_graph, "write_graph_event", lambda *a, **k: None)
        result = await pipeline_graph.pipeline_complete_node(
            self._incomplete_state(tmp_path, attempts=0))
        assert result.get("revalidate_pending") is True
        assert result.get("pipeline_aborted") is False
        assert result.get("revalidate_attempts") == 1
        assert result.get("current_tier_index") == 0
        # routes back into the block-rerun loop
        assert pipeline_graph._pipeline_complete_route(result) == "init_tier"

    async def test_retry_aborts_when_disabled(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_REVALIDATE_INCOMPLETE", "0")
        monkeypatch.setattr(pipeline_graph, "interrupt", lambda payload: {"action": "retry"})
        monkeypatch.setattr(pipeline_graph, "write_graph_event", lambda *a, **k: None)
        result = await pipeline_graph.pipeline_complete_node(
            self._incomplete_state(tmp_path, attempts=0))
        assert result.get("pipeline_aborted") is True
        assert result.get("revalidate_pending") is False
        assert pipeline_graph._pipeline_complete_route(result) == "end"

    async def test_retry_aborts_when_cap_reached(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CORESMITH_REVALIDATE_INCOMPLETE", raising=False)
        monkeypatch.setenv("CORESMITH_REVALIDATE_MAX", "2")
        monkeypatch.setattr(pipeline_graph, "interrupt", lambda payload: {"action": "retry"})
        monkeypatch.setattr(pipeline_graph, "write_graph_event", lambda *a, **k: None)
        result = await pipeline_graph.pipeline_complete_node(
            self._incomplete_state(tmp_path, attempts=2))  # already at cap
        assert result.get("pipeline_aborted") is True
        assert result.get("revalidate_pending") is False

    async def test_abort_action_aborts(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pipeline_graph, "interrupt", lambda payload: {"action": "abort"})
        monkeypatch.setattr(pipeline_graph, "write_graph_event", lambda *a, **k: None)
        result = await pipeline_graph.pipeline_complete_node(
            self._incomplete_state(tmp_path, attempts=0))
        assert result.get("pipeline_aborted") is True
        assert result.get("revalidate_pending") is False

    async def test_all_pass_completes_clean(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pipeline_graph, "write_graph_event", lambda *a, **k: None)
        # interrupt must NOT be called on the clean path
        def _boom(payload):
            raise AssertionError("interrupt() called on all-pass path")
        monkeypatch.setattr(pipeline_graph, "interrupt", _boom)
        state = self._incomplete_state(tmp_path)
        state["completed_blocks"][2]["success"] = True  # all 3 pass
        result = await pipeline_graph.pipeline_complete_node(state)
        # Per-block frontend completion is NOT pipeline_done (fix #5: the
        # deliverable is a verified chip_top set at the end of validation_dv;
        # setting pipeline_done here let a parked-at-integration run report
        # as "done"). The clean path still routes to integration_check.
        assert result.get("frontend_complete") is True
        assert result.get("pipeline_done") is False
        assert result.get("revalidate_pending") is False
        assert pipeline_graph._pipeline_complete_route(result) == "integration_check"


class TestChipModelStaleRegen:
    """Engine fix: revise_uarch must re-compose, not reuse a stale _chip_model.py
    (CORESMITH_REGEN_STALE_CHIP_MODEL). Both env branches per AGENTS.md."""
    import os as _os

    def _setup(self, tmp_path):
        md = tmp_path / "arch" / "block_models"
        md.mkdir(parents=True)
        (md / "a.py").write_text("# block a\n")
        (md / "b.py").write_text("# block b\n")
        chip = md / "_chip_model.py"
        chip.write_text("# composed\n")
        return md, chip

    def _set_mtime(self, p, t):
        import os
        os.utime(p, (t, t))


# --------------------------------------------------------------------------- #
# engine-v31 step 3: apply a high-confidence uarch_patch on the retry path
# --------------------------------------------------------------------------- #
def test_apply_uarch_patch_to_spec_replaces_sections():
    spec = "Intro.\nAAA original one BBB.\nMiddle.\noriginal two.\nEnd."
    patch = {"sections_to_replace": [
        {"original": "original one", "replacement": "REVISED one"},
        {"original": "original two", "replacement": "REVISED two"},
        {"original": "not present", "replacement": "ignored"},
    ]}
    out, n = pipeline_graph.apply_uarch_patch_to_spec(spec, patch)
    assert n == 2
    assert "REVISED one" in out and "REVISED two" in out
    assert "original one" not in out and "original two" not in out
    # idempotent: re-applying finds nothing to change
    out2, n2 = pipeline_graph.apply_uarch_patch_to_spec(out, patch)
    assert n2 == 0 and out2 == out


def _seed_uarch_patch_block(root, name, confidence):
    import json as _j
    from pathlib import Path as _P
    bd = _P(root) / ".coresmith" / "blocks" / name
    bd.mkdir(parents=True, exist_ok=True)
    specs = _P(root) / "arch" / "uarch_specs"
    specs.mkdir(parents=True, exist_ok=True)
    (specs / f"{name}.md").write_text(
        "# uArch\nThe round uses one registered phase per round.\nLatency = 11 clocks.")
    diag = {
        "category": "UARCH_SPEC_ERROR", "confidence": confidence, "escalate": True,
        "suggested_fix": "split the round into two registered phases",
        "uarch_patch": {
            "rationale": "mapped timing needs a 2-phase round split",
            "sections_to_replace": [
                {"original": "one registered phase per round",
                 "replacement": "two registered phases per round"},
                {"original": "Latency = 11 clocks",
                 "replacement": "Latency = 21 clocks"},
            ],
        },
    }
    (bd / "diagnosis.json").write_text(_j.dumps(diag))
    return bd, specs / f"{name}.md"


def test_route_uarch_patch_on_retry_applies_and_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setenv("CORESMITH_UARCH_PATCH_ON_RETRY", "1")
    monkeypatch.setenv("CORESMITH_UARCH_PATCH_CONFIDENCE", "0.9")
    bd, spec_path = _seed_uarch_patch_block(tmp_path, "aes_engine", 0.99)
    patched = pipeline_graph._route_uarch_patch_on_retry(str(tmp_path), ["aes_engine"])
    assert patched == ["aes_engine"]
    txt = spec_path.read_text()
    assert "two registered phases per round" in txt
    assert "Latency = 21 clocks" in txt
    assert (bd / "gate_feedback.txt").exists()
    assert (bd / "uarch_patch_applied").exists()
    # bounded: a second retry does NOT re-apply (escalates as before)
    patched2 = pipeline_graph._route_uarch_patch_on_retry(str(tmp_path), ["aes_engine"])
    assert patched2 == []


def test_route_uarch_patch_low_confidence_not_applied(tmp_path, monkeypatch):
    monkeypatch.setenv("CORESMITH_UARCH_PATCH_ON_RETRY", "1")
    monkeypatch.setenv("CORESMITH_UARCH_PATCH_CONFIDENCE", "0.9")
    bd, spec_path = _seed_uarch_patch_block(tmp_path, "blk_lo", 0.5)
    patched = pipeline_graph._route_uarch_patch_on_retry(str(tmp_path), ["blk_lo"])
    assert patched == []
    assert "one registered phase per round" in spec_path.read_text()
    assert not (bd / "uarch_patch_applied").exists()


def test_route_uarch_patch_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("CORESMITH_UARCH_PATCH_ON_RETRY", "0")
    _seed_uarch_patch_block(tmp_path, "blk_off", 0.99)
    assert pipeline_graph._route_uarch_patch_on_retry(str(tmp_path), ["blk_off"]) == []


# ---------------------------------------------------------------------------
# Post-synthesis gate-level simulation gate (CORESMITH_GATE_SIM)
# ---------------------------------------------------------------------------

class TestRouteAfterSynthGateSim:
    """``route_after_synth`` must send a block whose SYNTHESIZED NETLIST does
    not reproduce the verified RTL back to diagnose -- clean synthesis is not
    'done' when DV and PPA were measured on different hardware.

    Both env-var branches are covered: with the gate ON the harness produces
    ``gate_sim_ok=False`` and routing diverts; with it OFF the harness never
    runs, ``gate_sim_ok`` stays None, and routing is byte-identical to before.
    """

    _BASE = {"synth_success": True, "ppa_ok": None, "ppa_reasons": []}

    def test_gate_sim_pass_reaches_block_done(self):
        state = dict(self._BASE, gate_sim_ok=True, gate_sim_status="pass")
        assert pipeline_graph.route_after_synth(state) == "evaluate_targets"

    def test_gate_sim_fail_routes_to_diagnose(self):
        state = dict(self._BASE, gate_sim_ok=False, gate_sim_status="fail")
        assert pipeline_graph.route_after_synth(state) == "diagnose"

    def test_gate_sim_not_run_never_blocks(self):
        """None = the gate did not apply (disabled / no netlist / no toolchain).
        It must never block, exactly like ``ppa_ok`` of None."""
        state = dict(self._BASE, gate_sim_ok=None, gate_sim_status="not_run")
        assert pipeline_graph.route_after_synth(state) == "evaluate_targets"

    def test_absent_key_is_legacy_behaviour(self):
        assert pipeline_graph.route_after_synth(dict(self._BASE)) == "evaluate_targets"

    def test_synth_failure_still_wins(self):
        state = {"synth_success": False, "gate_sim_ok": True}
        assert pipeline_graph.route_after_synth(state) == "diagnose"


class TestGateSimNodeWiring:
    """``_run_gate_sim_gate`` is the synth node's adapter onto the harness."""

    _BLOCK = {"name": "blk"}

    def _state(self, tmp_path, **over):
        state = {"project_root": str(tmp_path), "attempt": 1,
                 "tb_path": str(tmp_path / "tb.py"), "current_block": self._BLOCK}
        state.update(over)
        return state

    def test_env_off_returns_disabled_and_never_calls_harness(self, tmp_path, monkeypatch):
        """OFF branch: the harness is not invoked and the verdict is an explicit
        'disabled', never a silent pass."""
        from orchestrator.harness import gate_sim as gs

        monkeypatch.setenv("CORESMITH_GATE_SIM", "0")
        called = []
        monkeypatch.setattr(gs, "check_gate_sim",
                            lambda **kw: called.append(kw))
        ok, status, reason = pipeline_graph._run_gate_sim_gate(
            self._state(tmp_path), self._BLOCK, "blk",
            {"netlist_path": str(tmp_path / "n.v")}, "rtl.v",
        )
        assert called == []
        assert ok is None and status == "disabled"
        assert "CORESMITH_GATE_SIM" in reason

    def test_env_on_runs_harness_and_propagates_failure(self, tmp_path, monkeypatch):
        """ON branch: the harness runs, a FAIL propagates, and the actionable
        report is written where the RTL fixer reads it."""
        from orchestrator.harness import gate_sim as gs

        monkeypatch.setenv("CORESMITH_GATE_SIM", "1")
        res = gs.GateSimResult(
            ran=True, ok=False, status=gs.STATUS_FAIL,
            reason="the gate netlist diverges from the verified RTL",
            cycles_compared=42,
            first_divergence={"cycle": 7, "port": "q", "expected": "1",
                              "actual": "0"},
        )
        monkeypatch.setattr(gs, "check_gate_sim", lambda **kw: res)
        ok, status, reason = pipeline_graph._run_gate_sim_gate(
            self._state(tmp_path), self._BLOCK, "blk",
            {"netlist_path": str(tmp_path / "n.v")}, "rtl.v",
        )
        assert ok is False and status == "fail"
        bd = tmp_path / ".coresmith" / "blocks" / "blk"
        assert json.loads((bd / "gate_sim_report.json").read_text())["status"] == "fail"
        prev = (bd / "previous_error.txt").read_text()
        assert "GATE-LEVEL SIMULATION FAILURE" in prev
        assert "FIRST DIVERGENCE at cycle 7" in prev

    def test_env_on_pass_returns_true(self, tmp_path, monkeypatch):
        from orchestrator.harness import gate_sim as gs

        monkeypatch.setenv("CORESMITH_GATE_SIM", "1")
        monkeypatch.setattr(
            gs, "check_gate_sim",
            lambda **kw: gs.GateSimResult(ran=True, ok=True,
                                          status=gs.STATUS_PASS,
                                          reason="matched", cycles_compared=9),
        )
        ok, status, _ = pipeline_graph._run_gate_sim_gate(
            self._state(tmp_path), self._BLOCK, "blk",
            {"netlist_path": str(tmp_path / "n.v")}, "rtl.v",
        )
        assert ok is True and status == "pass"

    def test_no_netlist_is_not_run_not_pass(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_GATE_SIM", "1")
        ok, status, reason = pipeline_graph._run_gate_sim_gate(
            self._state(tmp_path), self._BLOCK, "blk", None, "rtl.v",
        )
        assert ok is None and status == "not_run"
        assert "netlist" in reason

    def test_harness_crash_is_non_blocking_by_default(self, tmp_path, monkeypatch):
        from orchestrator.harness import gate_sim as gs

        monkeypatch.setenv("CORESMITH_GATE_SIM", "1")
        monkeypatch.delenv("CORESMITH_GATE_SIM_STRICT", raising=False)

        def boom(**_kw):
            raise RuntimeError("plumbing")

        monkeypatch.setattr(gs, "check_gate_sim", boom)
        ok, status, _ = pipeline_graph._run_gate_sim_gate(
            self._state(tmp_path), self._BLOCK, "blk",
            {"netlist_path": str(tmp_path / "n.v")}, "rtl.v",
        )
        assert ok is None and status == "not_run"

    def test_harness_crash_under_strict_blocks(self, tmp_path, monkeypatch):
        from orchestrator.harness import gate_sim as gs

        monkeypatch.setenv("CORESMITH_GATE_SIM", "1")
        monkeypatch.setenv("CORESMITH_GATE_SIM_STRICT", "1")

        def boom(**_kw):
            raise RuntimeError("plumbing")

        monkeypatch.setattr(gs, "check_gate_sim", boom)
        ok, _status, _ = pipeline_graph._run_gate_sim_gate(
            self._state(tmp_path), self._BLOCK, "blk",
            {"netlist_path": str(tmp_path / "n.v")}, "rtl.v",
        )
        assert ok is False


@pytest.mark.asyncio
async def test_gate_sim_skipped_when_synthesis_failed(tmp_path, monkeypatch):
    """A failed synth already wrote its own actionable ``previous_error.txt``.
    The gate-level sim must NOT run (there is no current netlist -- only a stale
    one) and must NOT overwrite that error."""
    from orchestrator.harness import gate_sim as gs

    monkeypatch.setenv("CORESMITH_GATE_SIM", "1")
    monkeypatch.setenv("CORESMITH_SKIP_SYNTH", "0")
    called = []
    monkeypatch.setattr(gs, "check_gate_sim", lambda **kw: called.append(kw))

    rtl = tmp_path / "blk.v"
    rtl.write_text("module blk(input clk); endmodule\n")
    tb = tmp_path / "tb.py"
    tb.write_text("# tb\n")

    def fake_synth(block, rtl_path, **kwargs):
        return {"success": False, "gate_count": 0, "log": "SYNTH BOOM",
                "netlist_path": str(tmp_path / "stale_netlist.v")}

    monkeypatch.setattr(pipeline_graph, "synthesize_block", fake_synth)
    monkeypatch.setattr(pipeline_graph, "fix_synth_errors",
                        AsyncMock(return_value=None))

    out = await pipeline_graph.synthesize_node({
        "project_root": str(tmp_path), "current_block": {"name": "blk"},
        "attempt": 1, "rtl_path": str(rtl), "tb_path": str(tb),
        "target_clock_mhz": 50.0,
    })

    assert called == []                      # harness never invoked
    assert out["synth_success"] is False
    assert out["gate_sim_ok"] is None
    assert out["gate_sim_status"] == "not_run"
    prev = (tmp_path / ".coresmith" / "blocks" / "blk" / "previous_error.txt")
    assert "SYNTH BOOM" in prev.read_text()  # synth error preserved
    assert pipeline_graph.route_after_synth(out) == "diagnose"


class TestContainerSubsumeGroups:
    def _item(self, name, area):
        from orchestrator.langgraph.mem_price import RollupItem
        return RollupItem(name=name, area_um2=area, source="ppa_history")

    def test_disjoint_containers_each_counted(self, monkeypatch):
        # Two independent container tops: each is de-duplicated against ITS OWN
        # leaves, so the die rollup is not reduced to the biggest container.
        monkeypatch.delenv("CORESMITH_DIE_ROLLUP_CONTAINER_DEDUP", raising=False)
        items = [self._item("enc_top", 300000.0), self._item("dec_top", 250000.0),
                 self._item("a", 60000.0), self._item("b", 40000.0),
                 self._item("c", 60000.0), self._item("d", 40000.0)]
        out, note = pipeline_graph._subsume_container_items(
            items, {"enc_top", "dec_top"},
            {"enc_top": {"a", "b"}, "dec_top": {"c", "d"}})
        assert sorted(i.name for i in out) == ["dec_top", "enc_top"]
        assert sum(i.area_um2 for i in out) == 550000.0
        assert note

    def test_nested_container_absorbed_once(self, monkeypatch):
        monkeypatch.delenv("CORESMITH_DIE_ROLLUP_CONTAINER_DEDUP", raising=False)
        items = [self._item("chip_top", 300000.0), self._item("mid", 200000.0),
                 self._item("a", 50000.0)]
        out, _note = pipeline_graph._subsume_container_items(
            items, {"chip_top", "mid"},
            {"chip_top": {"mid"}, "mid": {"a"}})
        assert [i.name for i in out] == ["chip_top"]


# ---------------------------------------------------------------------------
# Audit 20260901 regressions: gate re-run, override scoping, TB-gen park revise
# ---------------------------------------------------------------------------

_GATE_VIOLATIONS = [{
    "type": "model_integration_failure",
    "gap_class": "block_math",
    "first_divergence_block": "recon",
    "expected": [1, 2, 3],
    "observed": [1, 2, 4],
    "suggested_fix": "fix the transcribed rounding",
}]


def _park_failing_uarch_gate(monkeypatch, action):
    """Make the two-pass µarch gate fail and resolve its park with ``action``."""
    import orchestrator.architecture.model_integration as _mi

    async def _noop_gen(_pr):
        return None

    monkeypatch.setenv("CORESMITH_BLOCK_GOLDENS", "1")
    monkeypatch.delenv("CORESMITH_DETERMINISTIC_BFM", raising=False)
    monkeypatch.setattr(pipeline_graph, "_maybe_generate_chip_model", _noop_gen)
    monkeypatch.setattr(_mi, "run_model_integration_gate",
                        lambda pr, *a, **k: list(_GATE_VIOLATIONS))
    monkeypatch.setattr(pipeline_graph, "interrupt",
                        lambda payload: {"action": action})


class TestFeasibilityOverrideScope:
    """The ``uarch_feasibility_override`` marker is a SCOPE, not a global waiver.

    A bare-"1" marker waived every budget gate for the block, so overrides
    granted for stale ORACLE-DISCOVERY findings (category [interface] / tooling)
    forced blocks past the deterministic mem_price gate and mislabelled
    within-budget blocks "over-budget accepted".
    """

    def _block(self, root, name="blk"):
        (root / "arch" / "uarch_specs").mkdir(parents=True, exist_ok=True)
        (root / "arch" / "uarch_specs" / f"{name}.md").write_text(
            "# MEM big: 32x4096 ports=1rw impl=fpmem justification=x\n"
            "area_budget_um2: 20000\n")
        bdir = root / ".coresmith" / "blocks" / name
        bdir.mkdir(parents=True, exist_ok=True)
        return bdir

    def _write_contract(self, root, name="blk"):
        """A real contract edge so the block has a contract version."""
        from orchestrator.state_store.project_db import open_project
        db = open_project(root)
        db.import_contracts({"contracts": [
            {"edge_id": "e0", "producer_block": name, "consumer_block": "sink",
             "width": 32, "name": "e0"}]})
        return db.block_contract_version(name)

    def _marker(self, bdir, **fields):
        (bdir / "uarch_feasibility_override").write_text(json.dumps(fields))

    def _deferred(self, root, name="blk"):
        """True when the mem_price gate took the override (defer) branch."""
        verdict = pipeline_graph._mem_price_gate_verdict(str(root), name)
        led = root / ".coresmith" / "blocks" / name / "mem_price.json"
        if not led.exists():
            return False
        return verdict is None and json.loads(led.read_text()).get("deferred") is True

    # -- category parsing --------------------------------------------------
    def test_parses_leading_bracket_tags(self):
        assert pipeline_graph._feas_issue_categories([
            "[interface] frozen port too narrow",
            "[area] storage exceeds budget",
            "[area] again",              # de-duplicated
            "no tag at all",
        ]) == ["interface", "area"]

    def test_no_marker_is_no_scope(self, tmp_path):
        self._block(tmp_path)
        assert pipeline_graph._feas_override_scope(str(tmp_path), "blk") is None
        assert pipeline_graph._feas_override_covers(
            str(tmp_path), "blk", "area") is False

    # -- the bleed itself --------------------------------------------------
    def test_interface_only_override_does_not_defer_mem_price(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_MEM_PRICE_MAX_REVISE", "3")
        bdir = self._block(tmp_path)
        self._marker(bdir, gate="uarch_feasibility", categories=["interface"])
        assert pipeline_graph._feas_override_covers(
            str(tmp_path), "blk", "area") is False
        assert self._deferred(tmp_path) is False

    def test_area_override_defers_mem_price(self, tmp_path):
        bdir = self._block(tmp_path)
        self._marker(bdir, gate="uarch_feasibility",
                     categories=["interface", "area"])
        assert self._deferred(tmp_path) is True
        led = json.loads((bdir / "mem_price.json").read_text())
        assert "[area]" in led.get("deferred_reason", "")

    def test_legacy_one_marker_still_defers(self, tmp_path):
        # Compat: pre-JSON runs only ever overrode [area] blockers.
        bdir = self._block(tmp_path)
        (bdir / "uarch_feasibility_override").write_text("1")
        scope = pipeline_graph._feas_override_scope(str(tmp_path), "blk")
        assert scope == {"categories": ["area"], "legacy": True}
        assert self._deferred(tmp_path) is True

    # -- expiry ------------------------------------------------------------
    def test_contract_sha1_mismatch_expires_the_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_MEM_PRICE_MAX_REVISE", "3")
        bdir = self._block(tmp_path)
        self._write_contract(tmp_path)
        self._marker(bdir, gate="uarch_feasibility", categories=["area"],
                     contract_version=999)
        assert pipeline_graph._feas_override_scope(str(tmp_path), "blk") is None
        assert self._deferred(tmp_path) is False

    def test_matching_contract_sha1_keeps_the_override(self, tmp_path):
        bdir = self._block(tmp_path)
        ver = self._write_contract(tmp_path)
        assert ver, "fixture must produce a contract version"
        self._marker(bdir, gate="uarch_feasibility", categories=["area"],
                     contract_version=ver)
        assert self._deferred(tmp_path) is True

    def test_unhashable_contract_does_not_revoke_the_override(self, tmp_path):
        # No contracts -> the block's contract version is 0; a missing
        # provenance axis must not silently revoke a granted override.
        bdir = self._block(tmp_path)
        self._marker(bdir, gate="uarch_feasibility", categories=["area"],
                     contract_version=999)
        assert self._deferred(tmp_path) is True

    # -- the post-synth budget gate is scoped the same way ------------------
    def test_post_synth_budget_gate_is_area_scoped(self, tmp_path):
        bdir = self._block(tmp_path)
        self._marker(bdir, gate="uarch_feasibility", categories=["interface"])
        assert pipeline_graph._feas_override_covers(
            str(tmp_path), "blk", "area") is False
        self._marker(bdir, gate="uarch_feasibility", categories=["area"])
        assert pipeline_graph._feas_override_covers(
            str(tmp_path), "blk", "area") is True

    # -- what the override site writes -------------------------------------
    def test_override_marker_is_written_as_a_scope(self, tmp_path, monkeypatch):
        bdir = self._block(tmp_path)
        ver = self._write_contract(tmp_path)
        blocking = ["[interface] port too narrow", "[area] over budget"]
        monkeypatch.setattr(pipeline_graph, "interrupt",
                            lambda payload: {"action": "override"})
        # Drive the write site the same way the node does.
        (bdir / "uarch_feasibility_override").write_text(json.dumps({
            "gate": "uarch_feasibility",
            "categories": pipeline_graph._feas_issue_categories(blocking),
            "contract_version": pipeline_graph._db(str(tmp_path)).block_contract_version("blk"),
            "ts": 0.0,
        }))
        scope = pipeline_graph._feas_override_scope(str(tmp_path), "blk")
        assert scope["gate"] == "uarch_feasibility"
        assert scope["categories"] == ["interface", "area"]
        assert scope["contract_version"] == ver


class TestTbGenerationParkOffersRevise:
    """A TB that cannot be GENERATED is often a spec-level defect.

    Both "could not generate a usable cocotb testbench" parks offered only
    retry/fix_rtl/fix_tb/abort, so the chip lead had no lawful way to ask for
    tier regeneration and had to abort. The daemon validates the resume action
    against ``supported_actions``, so the payload is the load-bearing half.
    """

    def _rtl(self, tmp_path):
        top_rtl = tmp_path / "chip_top.v"
        block_rtl = tmp_path / "block.v"
        top_rtl.write_text("module chip_top(input clk); endmodule\n")
        block_rtl.write_text("module block(input clk); endmodule\n")
        (tmp_path / "arch" / "uarch_specs").mkdir(parents=True)
        (tmp_path / "arch" / "uarch_specs" / "recon.md").write_text("# recon\n")
        return {"top_rtl_path": str(top_rtl), "design_name": "chip_top",
                "block_rtl_paths": {"block": str(block_rtl)}}

    def _stub(self, monkeypatch, tmp_path, generator, stage):
        async def fail_generate(**_kwargs):
            raise RuntimeError("no usable Python cocotb testbench")

        async def fake_contract_audit(**kwargs):
            assert kwargs["stage"] == stage
            # The audit is told what the park can offer, so it can point the
            # operator at an offerable action.
            assert "revise" in kwargs["supported_actions"]
            return {"category": "SPEC_BUG", "recommended_action": "revise_uarch",
                    "outer_agent_summary": "the uArch specs are un-testable",
                    "affected_blocks": ["recon"],
                    "audit_path": str(tmp_path / "audit.json")}

        monkeypatch.setattr(pipeline_graph, "load_architecture_connections",
                            lambda _pr: ({}, {}))
        monkeypatch.setattr(pipeline_graph, generator, fail_generate)
        monkeypatch.setattr(pipeline_graph, "_run_top_level_contract_audit",
                            fake_contract_audit)

    @pytest.mark.asyncio
    async def test_integration_tb_generation_park_dispatches_revise(
        self, tmp_path, monkeypatch
    ):
        integ = self._rtl(tmp_path)
        self._stub(monkeypatch, tmp_path, "generate_integration_testbench",
                   "integration_dv_generation")
        monkeypatch.setattr(pipeline_graph, "interrupt", lambda p: {
            "action": "revise", "feedback": "widen the recon output field",
            "affected_blocks": ["recon"]})

        result = await integration_dv_node({
            "project_root": str(tmp_path), "integration_result": integ})
        dv = result["integration_dv_result"]
        payload = dv["interrupt_payload"]
        assert payload["phase"] == "tb_generation"
        assert "revise" in payload["supported_actions"]
        assert "revise" in payload["outer_agent_guidance"]

        decision = await integration_dv_decision_node({
            "project_root": str(tmp_path), "integration_dv_result": dv})
        d = decision["integration_dv_result"]
        assert d["action_taken"] == "revise"
        assert d["revised_blocks"] == ["recon"]
        # Same dispatch as the main DV-failed park: feedback on disk, tiers rerun.
        spec = (tmp_path / "arch" / "uarch_specs" / "recon.md").read_text()
        assert "widen the recon output field" in spec
        assert decision["current_tier_index"] == 0
        assert route_after_integration_dv_decision(decision) == "init_tier"

    @pytest.mark.asyncio
    async def test_validation_tb_generation_park_dispatches_revise(
        self, tmp_path, monkeypatch
    ):
        integ = self._rtl(tmp_path)
        self._stub(monkeypatch, tmp_path, "generate_validation_testbench",
                   "validation_dv_generation")
        monkeypatch.setattr(pipeline_graph, "_load_ers_validation_context",
                            lambda _pr: ('{"validation_kpis": ["must pass"]}', 1))
        monkeypatch.setattr(pipeline_graph, "interrupt", lambda p: {
            "action": "revise", "feedback": "declare the KPI counter",
            "affected_blocks": ["recon"]})

        result = await validation_dv_node({
            "project_root": str(tmp_path), "integration_result": integ})
        dv = result["validation_dv_result"]
        payload = dv["interrupt_payload"]
        assert payload["phase"] == "tb_generation"
        assert "revise" in payload["supported_actions"]
        assert "revise" in payload["outer_agent_guidance"]

        decision = await validation_dv_decision_node({
            "project_root": str(tmp_path), "validation_dv_result": dv})
        d = decision["validation_dv_result"]
        assert d["action_taken"] == "revise"
        assert d["revised_blocks"] == ["recon"]
        spec = (tmp_path / "arch" / "uarch_specs" / "recon.md").read_text()
        assert "declare the KPI counter" in spec
        assert decision["current_tier_index"] == 0
        assert route_after_validation_dv_decision(decision) == "init_tier"
