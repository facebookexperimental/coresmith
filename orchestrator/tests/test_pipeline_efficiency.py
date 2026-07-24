# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for pipeline efficiency fixes (R1-R10).

Covers:
- R1a: validate_rtl_ports removed from block subgraph
- R1b: IntegrationReviewAgent creation
- R1c: review_uarch_spec_node auto-approves, integration_review in orchestrator
- R2: DV_RULES.md existence, debug agent prompt, testbench prompt
- R3: RTL regression guard (best_result.json)
- R4: Testbench reuse prompt
- R7a: RTL generator model is opus
- R7b: RTL generator prompt includes lint instruction
- R9: Backend single-block design name resolution
- R10: PnR die-size awareness with gate_count parameter
- Dashboard: Jinja2 template rendering (no LLM)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# ═══════════════════════════════════════════════════════════════════════════
# R1a: validate_rtl_ports removed
# ═══════════════════════════════════════════════════════════════════════════

class TestValidateRtlPortsRemoved:
    def test_no_validate_rtl_ports_node(self):
        from orchestrator.langgraph.pipeline_graph import build_block_subgraph
        graph = build_block_subgraph()
        node_names = set(graph.nodes.keys())
        assert "validate_rtl_ports" not in node_names

    def test_generate_rtl_connects_to_lint(self):
        # Lint is no longer a separate graph node -- it has been folded into
        # generate_rtl_node, which sets ``lint_clean`` and is then dispatched
        # by ``route_after_rtl``. Verify the conditional branch is wired up
        # on ``generate_rtl`` and that the old explicit ``lint`` node is gone.
        from orchestrator.langgraph.pipeline_graph import build_block_subgraph
        graph = build_block_subgraph()
        assert "generate_rtl" in graph.branches, (
            "generate_rtl should have a conditional edge "
            "(route_after_rtl folds in lint dispatch)"
        )
        assert "route_after_rtl" in graph.branches["generate_rtl"]
        assert "lint" not in graph.nodes


# ═══════════════════════════════════════════════════════════════════════════
# R1b: IntegrationReviewAgent
# ═══════════════════════════════════════════════════════════════════════════

class TestIntegrationReviewAgent:
    def test_agent_importable(self):
        from orchestrator.langchain.agents.integration_review_agent import (
            IntegrationReviewAgent,
        )
        agent = IntegrationReviewAgent()
        assert hasattr(agent, "review")

    def test_prompt_file_exists(self):
        prompt_path = (
            Path(__file__).resolve().parents[1]
            / "langchain" / "prompts" / "integration_review.md"
        )
        assert prompt_path.exists()
        content = prompt_path.read_text()
        assert "Section 9" in content
        assert "block_diagram.json" in content


# ═══════════════════════════════════════════════════════════════════════════
# R1c: review_uarch_spec_node auto-approves
# ═══════════════════════════════════════════════════════════════════════════

class TestUarchAutoApprove:
    @pytest.mark.asyncio
    async def test_review_node_auto_approves(self):
        from orchestrator.langgraph.pipeline_graph import review_uarch_spec_node

        state = {
            "current_block": {"name": "fir_filter", "estimated_gates": 5000},
            "project_root": "/tmp/test",
            "pipeline_run_start": 0,
        }

        result = await review_uarch_spec_node(state)
        assert result["uarch_approved"] is True
        assert result["human_response"]["action"] == "approve"

    def test_integration_review_in_orchestrator(self):
        from langgraph.checkpoint.memory import MemorySaver

        from orchestrator.langgraph.pipeline_graph import build_pipeline_graph

        graph = build_pipeline_graph(checkpointer=MemorySaver())
        node_names = set(graph.get_graph().nodes.keys())
        assert "integration_review" in node_names


# ═══════════════════════════════════════════════════════════════════════════
# R2: DV_RULES.md
# ═══════════════════════════════════════════════════════════════════════════

class TestDvRules:
    def test_dv_rules_file_exists(self):
        # arch/DV_RULES.md is a per-project artefact populated by the
        # debug agent (see debug_agent.md prompt). It does not ship with
        # the repo, so only assert presence when the user has created one.
        rules_path = Path(__file__).resolve().parents[2] / "arch" / "DV_RULES.md"
        if not rules_path.exists():
            pytest.skip("arch/DV_RULES.md is project-local and not seeded in repo")
        content = rules_path.read_text()
        assert "FIFO" in content
        assert "Backpressure" in content
        assert "AXI-Stream" in content

    def test_debug_agent_prompt_mentions_dv_rules(self):
        prompt_path = (
            Path(__file__).resolve().parents[1]
            / "langchain" / "prompts" / "debug_agent.md"
        )
        content = prompt_path.read_text()
        assert "DV_RULES.md" in content
        assert "is_testbench_bug" in content

    def test_testbench_prompt_mentions_dv_rules(self):
        prompt_path = (
            Path(__file__).resolve().parents[1]
            / "langchain" / "prompts" / "testbench_generator.md"
        )
        content = prompt_path.read_text()
        assert "DV_RULES.md" in content
        assert "DV RULES" in content

    def test_testbench_agent_user_message_includes_dv_rules(self):
        import inspect

        from orchestrator.langchain.agents.testbench_generator import (
            TestbenchGeneratorAgent,
        )
        source = inspect.getsource(TestbenchGeneratorAgent.generate)
        assert "DV_RULES.md" in source


# ═══════════════════════════════════════════════════════════════════════════
# R3: RTL regression guard
# ═══════════════════════════════════════════════════════════════════════════

class TestRtlRegressionGuard:
    @pytest.mark.asyncio
    async def test_skips_regen_when_prev_sim_passed(self, tmp_path):
        from orchestrator.langgraph.pipeline_graph import generate_rtl_node

        block_name = "test_block"
        rtl_dir = tmp_path / "rtl" / "datapath"
        rtl_dir.mkdir(parents=True)
        rtl_file = rtl_dir / f"{block_name}.v"
        rtl_file.write_text("module test_block(); endmodule\n")

        block_dir = tmp_path / ".coresmith" / "blocks" / block_name
        block_dir.mkdir(parents=True)
        (block_dir / "best_result.json").write_text(json.dumps({
            "sim_passed": True,
            "attempt": 1,
            "tests_passed": 10,
            "tests_total": 10,
        }))

        state = {
            "current_block": {
                "name": block_name,
                "rtl_target": f"rtl/datapath/{block_name}.v",
            },
            "attempt": 2,
            "project_root": str(tmp_path),
            "pipeline_run_start": 0,
        }

        result = await generate_rtl_node(state)
        assert result["force_regen_tb"] is True
        assert result["rtl_path"] == str(rtl_file)

    @pytest.mark.asyncio
    async def test_no_skip_on_attempt_1(self, tmp_path):
        from orchestrator.langgraph.pipeline_graph import generate_rtl_node

        block_name = "test_block"
        rtl_dir = tmp_path / "rtl" / "datapath"
        rtl_dir.mkdir(parents=True)
        rtl_file = rtl_dir / f"{block_name}.v"
        rtl_file.write_text("module test_block(); endmodule\n")

        state = {
            "current_block": {
                "name": block_name,
                "rtl_target": f"rtl/datapath/{block_name}.v",
            },
            "attempt": 1,
            "project_root": str(tmp_path),
            "pipeline_run_start": 0.0,
        }

        result = await generate_rtl_node(state)
        assert result.get("force_regen_tb") is not True

    @pytest.mark.asyncio
    async def test_sim_pass_writes_best_result(self, tmp_path):
        # simulate_node was merged into generate_testbench_node; the
        # best_result.json side-effect is now produced inside that node.
        from unittest.mock import patch

        from orchestrator.langgraph.pipeline_graph import generate_testbench_node

        block_name = "test_block"
        block_dir = tmp_path / ".coresmith" / "blocks" / block_name
        block_dir.mkdir(parents=True)

        rtl_file = tmp_path / "test.v"
        rtl_file.write_text("module test_block(); endmodule\n")
        tb_file = tmp_path / "test_tb.py"
        tb_file.write_text("import cocotb\n")

        state = {
            "current_block": {
                "name": block_name,
                # generate_testbench_node uses block["testbench"] to derive
                # tb_path_obj; point it at the pre-existing fixture file.
                "testbench": "test_tb.py",
            },
            "rtl_path": str(rtl_file),
            "tb_path": str(tb_file),
            "attempt": 1,
            "project_root": str(tmp_path),
            "pipeline_run_start": 0,
            "step_log_paths": {},
            # Reuse the existing tb file -- skip the (mocked-out) LLM call.
            "preserve_testbench": True,
            "force_regen_tb": False,
        }

        mock_result = {
            "passed": True,
            "log": "PASS",
            "tests_passed": 6,
            "tests_total": 6,
            "log_path": "/tmp/sim.log",
        }

        with patch(
            "orchestrator.langgraph.pipeline_graph.run_simulation",
            return_value=mock_result,
        ):
            await generate_testbench_node(state)

        best_path = block_dir / "best_result.json"
        assert best_path.exists()
        best = json.loads(best_path.read_text())
        assert best["sim_passed"] is True
        assert best["attempt"] == 1


# ═══════════════════════════════════════════════════════════════════════════
# R4: Testbench reuse prompt
# ═══════════════════════════════════════════════════════════════════════════

class TestTestbenchReusePrompt:
    def test_prompt_contains_reuse_instruction(self):
        prompt_path = (
            Path(__file__).resolve().parents[1]
            / "langchain" / "prompts" / "testbench_generator.md"
        )
        content = prompt_path.read_text()
        assert "TESTBENCH REUSE" in content
        assert "targeted edits" in content
        assert "module interface" in content


# ═══════════════════════════════════════════════════════════════════════════
# R7a: RTL generator model is opus
# ═══════════════════════════════════════════════════════════════════════════

class TestRtlGeneratorModel:
    def test_uses_default_model(self):
        """RTL generation should construct the agent with the project default
        model. We assert via the symbolic constant rather than a string literal,
        so model bumps in coresmith_llm don't require touching this test.
        """
        from orchestrator.langchain.agents.coresmith_llm import DEFAULT_MODEL
        from orchestrator.langchain.agents.rtl_generator import RTLGeneratorAgent
        agent = RTLGeneratorAgent()
        assert agent.llm.model == DEFAULT_MODEL


# ═══════════════════════════════════════════════════════════════════════════
# R7b: RTL generator prompt includes lint
# ═══════════════════════════════════════════════════════════════════════════

class TestRtlLintInPrompt:
    def test_prompt_contains_lint_instruction(self):
        prompt_path = (
            Path(__file__).resolve().parents[1]
            / "langchain" / "prompts" / "rtl_generator.md"
        )
        content = prompt_path.read_text()
        assert "verilator --lint-only" in content
        assert "LINT-CLEAN OUTPUT" in content


# ═══════════════════════════════════════════════════════════════════════════
# R9: Backend single-block design name
# ═══════════════════════════════════════════════════════════════════════════

class TestBackendSingleBlock:
    @pytest.mark.asyncio
    async def test_single_block_uses_own_netlist(self, tmp_path):
        # Single-block designs now go through ``integration_check_node`` which
        # writes a passthrough wrapper to ``rtl/integration/<top>.v``; the
        # backend's ``init_design_node`` picks up that integration top-level,
        # not the block's individual synthesis netlist.
        from orchestrator.langgraph.backend_graph import init_design_node

        block_name = "adder_8bit"

        rtl_dir = tmp_path / "rtl" / block_name
        rtl_dir.mkdir(parents=True)
        (rtl_dir / f"{block_name}.v").write_text("module adder_8bit(); endmodule\n")

        # Integration wrapper produced by the frontend pipeline.
        integ_dir = tmp_path / "rtl" / "integration"
        integ_dir.mkdir(parents=True)
        integ_top = integ_dir / "adder_8bit_top.v"
        integ_top.write_text("module adder_8bit_top();\n  adder_8bit u (); endmodule\n")

        state = {
            "project_root": str(tmp_path),
            "design_name": "prd___8_bit_adder_top",
            "frontend_blocks": [
                {"name": block_name, "rtl_target": f"rtl/{block_name}/{block_name}.v"},
            ],
            "block_queue": [
                {"name": block_name, "rtl_target": f"rtl/{block_name}/{block_name}.v"},
            ],
            "current_tier_index": 0,
            "tier_list": [1],
        }

        result = await init_design_node(state)

        assert result["integration_top_path"] == str(integ_top)
        # Module name comes from the integration top-level Verilog, not from
        # the (mangled) PRD-derived design_name.
        assert result["current_block"]["name"] == "adder_8bit_top"


# ═══════════════════════════════════════════════════════════════════════════
# R10: PnR die-size with gate_count
# ═══════════════════════════════════════════════════════════════════════════

class TestPnrDieSizeGateCount:
    def test_gate_count_param_accepted(self, tmp_path):
        from orchestrator.langgraph.backend_helpers import generate_pnr_tcl

        tcl = generate_pnr_tcl(
            "b", "/fake/n.v", "/fake/s.sdc", str(tmp_path), gate_count=100,
        )
        assert Path(tcl).exists()

    def test_zero_gate_count_uses_utilization(self, tmp_path):
        from orchestrator.langgraph.backend_helpers import generate_pnr_tcl

        tcl = generate_pnr_tcl(
            "b", "/fake/n.v", "/fake/s.sdc", str(tmp_path), gate_count=0,
        )
        content = Path(tcl).read_text()
        floorplan = content.split("Floorplan")[1].split("Power")[0]
        assert "-utilization" in floorplan

    def test_run_pnr_flow_accepts_gate_count(self):
        import inspect

        from orchestrator.langgraph.backend_helpers import run_pnr_flow

        sig = inspect.signature(run_pnr_flow)
        assert "gate_count" in sig.parameters


# ═══════════════════════════════════════════════════════════════════════════
# Dashboard: Jinja2 template rendering
# ═══════════════════════════════════════════════════════════════════════════

class TestChipFinishDashboard:
    def test_template_file_exists(self):
        template_path = (
            Path(__file__).resolve().parents[1]
            / "langchain" / "prompts" / "chip_finish_template.html"
        )
        assert template_path.exists()
        content = template_path.read_text()
        assert "{{ design_name }}" in content
        assert "metrics." in content
        assert "id=\"rtl\"" in content
        assert "id=\"testbenches\"" in content
        assert "id=\"layout\"" in content
        assert "gds_viewer" in content

    def test_no_llm_in_dashboard_generator(self):
        import inspect

        from orchestrator.architecture.specialists.chip_finish_dashboard import (
            generate_chip_finish_dashboard,
        )
        source = inspect.getsource(generate_chip_finish_dashboard)
        assert "ClaudeLLM" not in source
        assert "llm.call" not in source
        assert "jinja2" in source.lower() or "Jinja2" in source or "Environment" in source

    @pytest.mark.asyncio
    async def test_renders_dashboard_from_data(self, tmp_path):
        from orchestrator.architecture.specialists.chip_finish_dashboard import (
            generate_chip_finish_dashboard,
        )

        arch_dir = tmp_path / "arch"
        arch_dir.mkdir()
        (arch_dir / "prd_spec.md").write_text("# PRD\nTest design PRD")
        (arch_dir / "sad_spec.md").write_text("# SAD\nTest SAD")
        (arch_dir / "frd_spec.md").write_text("# FRD\nTest FRD")
        (arch_dir / "ers_spec.md").write_text("# ERS\nTest ERS")

        uarch_dir = arch_dir / "uarch_specs"
        uarch_dir.mkdir()
        (uarch_dir / "test_block.md").write_text("# uArch\nTest uArch")

        coresmith_dir = tmp_path / ".coresmith"
        coresmith_dir.mkdir()
        (coresmith_dir / "prd_spec.json").write_text(json.dumps({
            "prd": {
                "title": "Test Design",
                "target_technology": {"pdk": "sky130", "process_nm": 130},
                "area_budget": {"max_gate_count": 5000},
            }
        }))
        (coresmith_dir / "block_diagram.json").write_text(json.dumps({
            "blocks": [
                {"name": "test_block", "tier": 1, "estimated_gates": 100,
                 "description": "A test block"},
            ],
            "connections": [],
        }))
        (coresmith_dir / "pipeline_events.jsonl").write_text("")

        rtl_dir = tmp_path / "rtl" / "test_block"
        rtl_dir.mkdir(parents=True)
        (rtl_dir / "test_block.v").write_text(
            "module test_block (input clk, input rst_n, output [7:0] sum);\n"
            "endmodule\n"
        )

        tb_dir = tmp_path / "tb" / "cocotb"
        tb_dir.mkdir(parents=True)
        (tb_dir / "test_test_block.py").write_text(
            "import cocotb\n@cocotb.test()\nasync def test_reset(dut):\n    pass\n"
        )

        syn_dir = tmp_path / "syn" / "output" / "test_block"
        syn_dir.mkdir(parents=True)

        sim_dir = tmp_path / "sim_build"
        sim_dir.mkdir()

        completed_blocks = [
            {"name": "test_block", "success": True, "synth_gate_count": 52,
             "rtl_target": "rtl/test_block/test_block.v"},
        ]

        html = await generate_chip_finish_dashboard(
            completed_blocks=completed_blocks,
            project_root=str(tmp_path),
            target_clock_mhz=50.0,
        )

        assert "test_block" in html
        assert "Test Design" in html
        assert "sky130" in html.lower() or "SKY130" in html
        assert "module test_block" in html
        assert "import cocotb" in html
        assert "<nav>" in html
        assert "</html>" in html
        assert len(html) > 1000

    @pytest.mark.asyncio
    async def test_gds_3d_viewer_for_small_files(self, tmp_path):
        from orchestrator.architecture.specialists.chip_finish_dashboard import (
            generate_chip_finish_dashboard,
        )

        self._setup_minimal_project(tmp_path)

        cf_dir = tmp_path / "chip_finish"
        cf_dir.mkdir()
        (cf_dir / "3d.html").write_text("<html>3D viewer</html>")

        html = await generate_chip_finish_dashboard(
            completed_blocks=[{"name": "test_block", "success": True,
                               "rtl_target": "rtl/test_block/test_block.v"}],
            project_root=str(tmp_path),
            target_clock_mhz=50.0,
            viewer_3d_available=True,
        )

        assert "3d.html" in html
        assert "iframe" in html or "View 3D" in html

    def _setup_minimal_project(self, tmp_path):
        (tmp_path / "arch").mkdir(exist_ok=True)
        (tmp_path / "arch" / "uarch_specs").mkdir(exist_ok=True)
        coresmith = tmp_path / ".coresmith"
        coresmith.mkdir(exist_ok=True)
        (coresmith / "prd_spec.json").write_text(json.dumps({"prd": {
            "title": "T", "target_technology": {"pdk": "sky130", "process_nm": 130},
            "area_budget": {"max_gate_count": 1000},
        }}))
        (coresmith / "block_diagram.json").write_text(json.dumps({
            "blocks": [{"name": "test_block", "tier": 1}], "connections": [],
        }))
        (coresmith / "pipeline_events.jsonl").write_text("")
        (tmp_path / "rtl" / "test_block").mkdir(parents=True, exist_ok=True)
        (tmp_path / "rtl" / "test_block" / "test_block.v").write_text(
            "module test_block(); endmodule\n"
        )
        (tmp_path / "tb" / "cocotb").mkdir(parents=True, exist_ok=True)
        (tmp_path / "syn" / "output" / "test_block").mkdir(parents=True, exist_ok=True)
        (tmp_path / "sim_build").mkdir(exist_ok=True)
