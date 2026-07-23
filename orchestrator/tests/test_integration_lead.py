# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for the IntegrationLeadAgent and the updated integration_check_node.

Coverage:
- IntegrationLeadAgent._build_prompt: structured prompt construction
- IntegrationLeadAgent._parse_response: JSON extraction, fallback, validation
- IntegrationLeadAgent._validate_result: default severity, missing fields
- IntegrationLeadAgent.integrate: end-to-end with mocked LLM
- integration_check_node: skip conditions, agent call, lint, interrupt flow
- Model name updates: all agent instantiations use claude-sonnet-4-6
- Prompt file loading: SYSTEM_PROMPT loads from disk
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from orchestrator.langchain.agents.integration_lead import (
    IntegrationLeadAgent,
    SYSTEM_PROMPT,
    _PROMPT_FILE,
    assert_blocks_instantiated,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_VERILOG_A = """\
module block_a (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [7:0]  data_in,
    output wire [7:0]  data_out
);
    assign data_out = data_in;
endmodule
"""

SAMPLE_VERILOG_B = """\
module block_b (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [7:0]  data_in,
    output wire [7:0]  data_out,
    output wire        valid_out
);
    assign data_out = data_in;
    assign valid_out = 1'b1;
endmodule
"""

SAMPLE_CONNECTIONS = [
    {
        "from_block": "block_a",
        "from_port": "data_out",
        "to_block": "block_b",
        "to_port": "data_in",
        "interface": "data_pipe",
        "data_width": 8,
    },
]

# Chip-top stubs used by integration_check_node tests. The
# assert_blocks_instantiated postcondition (introduced after the codec
# stub-substitution incident) requires every expected block to appear as
# an instantiation in the generated chip_top, so the mocked chip_top text
# must contain those instances or the postcondition will (rightly) reject
# the agent output mid-test.
CHIP_TOP_AB = """\
module chip_top (input clk, input rst_n);
    a u_a (.clk(clk), .rst_n(rst_n));
    b u_b (.clk(clk), .rst_n(rst_n));
endmodule
"""

CHIP_TOP_BLOCK_A_B = """\
module test_top (input clk, input rst_n);
    block_a u_a (.clk(clk), .rst_n(rst_n));
    block_b u_b (.clk(clk), .rst_n(rst_n));
endmodule
"""

SAMPLE_PORT_SUMMARIES = [
    {
        "name": "block_a",
        "port_count": 4,
        "ports": [
            {"name": "clk", "direction": "input", "width": 1},
            {"name": "rst_n", "direction": "input", "width": 1},
            {"name": "data_in", "direction": "input", "width": 8},
            {"name": "data_out", "direction": "output", "width": 8},
        ],
    },
    {
        "name": "block_b",
        "port_count": 5,
        "ports": [
            {"name": "clk", "direction": "input", "width": 1},
            {"name": "rst_n", "direction": "input", "width": 1},
            {"name": "data_in", "direction": "input", "width": 8},
            {"name": "data_out", "direction": "output", "width": 8},
            {"name": "valid_out", "direction": "output", "width": 1},
        ],
    },
]

SAMPLE_AGENT_RESPONSE = json.dumps({
    "mismatches": [],
    "verilog": (
        "module test_top (\n"
        "  input  wire clk,\n"
        "  input  wire rst_n,\n"
        "  input  wire [7:0] block_a_data_in,\n"
        "  output wire [7:0] block_b_data_out,\n"
        "  output wire block_b_valid_out\n"
        ");\n"
        "  wire [7:0] w_block_a_data_out_to_block_b_data_in;\n"
        "  block_a u_block_a (\n"
        "    .clk(clk),\n"
        "    .rst_n(rst_n),\n"
        "    .data_in(block_a_data_in),\n"
        "    .data_out(w_block_a_data_out_to_block_b_data_in)\n"
        "  );\n"
        "  block_b u_block_b (\n"
        "    .clk(clk),\n"
        "    .rst_n(rst_n),\n"
        "    .data_in(w_block_a_data_out_to_block_b_data_in),\n"
        "    .data_out(block_b_data_out),\n"
        "    .valid_out(block_b_valid_out)\n"
        "  );\n"
        "endmodule\n"
    ),
    "module_name": "test_top",
    "wire_count": 1,
    "skipped_connections": [],
    "notes": "All connections clean",
})

SAMPLE_AGENT_RESPONSE_WITH_MISMATCHES = json.dumps({
    "mismatches": [
        {
            "from_block": "block_a",
            "to_block": "block_b",
            "issue_type": "width_mismatch",
            "severity": "error",
            "description": "block_a.data_out is 16-bit but block_b.data_in is 8-bit",
            "suggested_fix": "Widen block_b.data_in to 16 bits",
        },
    ],
    "verilog": "module test_top();\nendmodule\n",
    "module_name": "test_top",
    "wire_count": 0,
    "skipped_connections": ["block_a->block_b (data_pipe): has errors"],
    "notes": "Width mismatch found",
})


# ---------------------------------------------------------------------------
# IntegrationLeadAgent._build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_includes_design_name(self):
        agent = IntegrationLeadAgent()
        prompt = agent._build_prompt(
            "my_chip_top",
            {"block_a": SAMPLE_VERILOG_A},
            SAMPLE_PORT_SUMMARIES[:1],
            SAMPLE_CONNECTIONS,
            "Test PRD summary",
        )
        assert "my_chip_top" in prompt

    def test_includes_block_rtl_sources(self):
        agent = IntegrationLeadAgent()
        prompt = agent._build_prompt(
            "top", {"block_a": SAMPLE_VERILOG_A}, [], [], "",
        )
        assert "block_a" in prompt
        assert "module block_a" in prompt

    def test_includes_port_summaries(self):
        agent = IntegrationLeadAgent()
        prompt = agent._build_prompt(
            "top", {}, SAMPLE_PORT_SUMMARIES, [], "",
        )
        assert "data_in" in prompt
        assert "data_out" in prompt
        assert "[8-bit]" in prompt

    def test_includes_connections(self):
        agent = IntegrationLeadAgent()
        prompt = agent._build_prompt(
            "top", {}, [], SAMPLE_CONNECTIONS, "",
        )
        assert "block_a.data_out" in prompt
        assert "block_b.data_in" in prompt
        assert "data_pipe" in prompt

    def test_includes_prd_summary(self):
        agent = IntegrationLeadAgent()
        prompt = agent._build_prompt(
            "top", {}, [], [], "50 MHz AXI-Stream pipeline",
        )
        assert "50 MHz AXI-Stream pipeline" in prompt

    def test_omits_prd_section_when_empty(self):
        agent = IntegrationLeadAgent()
        prompt = agent._build_prompt("top", {}, [], [], "")
        assert "PRD SUMMARY" not in prompt

    def test_truncates_large_rtl(self):
        large_rtl = "// " + "x" * 10000
        agent = IntegrationLeadAgent()
        prompt = agent._build_prompt(
            "top", {"big_block": large_rtl}, [], [], "",
        )
        assert len(prompt) < len(large_rtl)

    def test_handles_many_connections(self):
        conns = [
            {
                "from_block": f"blk_{i}",
                "from_port": "out",
                "to_block": f"blk_{i+1}",
                "to_port": "in",
                "interface": f"conn_{i}",
                "data_width": 8,
            }
            for i in range(60)
        ]
        agent = IntegrationLeadAgent()
        prompt = agent._build_prompt("top", {}, [], conns, "")
        conn_lines = [line for line in prompt.split("\n") if "blk_" in line]
        assert len(conn_lines) <= 50


# ---------------------------------------------------------------------------
# IntegrationLeadAgent._parse_response
# ---------------------------------------------------------------------------

class TestParseResponse:
    def test_parses_valid_json(self):
        agent = IntegrationLeadAgent()
        result = agent._parse_response(SAMPLE_AGENT_RESPONSE, "test_top")
        assert result["module_name"] == "test_top"
        assert "module test_top" in result["verilog"]
        assert result["wire_count"] == 1
        assert result["mismatches"] == []

    def test_parses_json_with_surrounding_text(self):
        agent = IntegrationLeadAgent()
        content = f"Here is the result:\n{SAMPLE_AGENT_RESPONSE}\nDone."
        result = agent._parse_response(content, "test_top")
        assert result["module_name"] == "test_top"

    def test_handles_invalid_json(self):
        agent = IntegrationLeadAgent()
        result = agent._parse_response("not json at all", "fallback")
        assert result["parse_error"] is True
        assert result["module_name"] == "fallback"
        assert result["verilog"] == ""

    def test_handles_empty_response(self):
        agent = IntegrationLeadAgent()
        result = agent._parse_response("", "fallback")
        assert result["parse_error"] is True

    def test_handles_partial_json(self):
        agent = IntegrationLeadAgent()
        result = agent._parse_response('{"verilog": "module t();', "fallback")
        assert result["parse_error"] is True

    def test_preserves_mismatches(self):
        agent = IntegrationLeadAgent()
        result = agent._parse_response(
            SAMPLE_AGENT_RESPONSE_WITH_MISMATCHES, "test_top"
        )
        assert len(result["mismatches"]) == 1
        assert result["mismatches"][0]["issue_type"] == "width_mismatch"
        assert result["mismatches"][0]["severity"] == "error"


# ---------------------------------------------------------------------------
# IntegrationLeadAgent._validate_result
# ---------------------------------------------------------------------------

class TestValidateResult:
    def test_adds_default_severity(self):
        agent = IntegrationLeadAgent()
        data = {
            "verilog": "module t(); endmodule",
            "mismatches": [
                {"from_block": "a", "to_block": "b", "description": "test"},
            ],
        }
        result = agent._validate_result(data, "t")
        assert result["mismatches"][0]["severity"] == "warning"
        assert result["mismatches"][0]["issue_type"] == "unknown"

    def test_adds_default_fields(self):
        agent = IntegrationLeadAgent()
        result = agent._validate_result({}, "default_name")
        assert result["module_name"] == "default_name"
        assert result["verilog"] == ""
        assert result["mismatches"] == []
        assert result["wire_count"] == 0

    def test_warns_on_missing_module_declaration(self):
        agent = IntegrationLeadAgent()
        data = {"verilog": "assign x = y;"}
        result = agent._validate_result(data, "test")
        assert "WARNING" in result["notes"]


# ---------------------------------------------------------------------------
# IntegrationLeadAgent.integrate (end-to-end with mocked LLM)
# ---------------------------------------------------------------------------

class TestIntegrate:
    @pytest.mark.asyncio
    async def test_successful_integration(self, tmp_path):
        agent = IntegrationLeadAgent()
        out_path = tmp_path / "test_top.v"
        with patch.object(
            agent.llm, "call", new_callable=AsyncMock,
            return_value=SAMPLE_AGENT_RESPONSE,
        ):
            result = await agent.integrate(
                design_name="test_top",
                block_rtl_sources={
                    "block_a": SAMPLE_VERILOG_A,
                    "block_b": SAMPLE_VERILOG_B,
                },
                block_port_summaries=SAMPLE_PORT_SUMMARIES,
                connections=SAMPLE_CONNECTIONS,
                prd_summary="Test chip",
                output_path=str(out_path),
            )

        # The integrate method now pops verilog out of the returned dict
        # and writes it to ``output_path``; rtl_path points to the file.
        assert result["module_name"] == "test_top"
        assert result["rtl_path"] == str(out_path)
        assert "module test_top" in out_path.read_text()
        assert result["wire_count"] == 1
        assert result["mismatches"] == []

    @pytest.mark.asyncio
    async def test_integration_with_mismatches(self):
        agent = IntegrationLeadAgent()
        with patch.object(
            agent.llm, "call", new_callable=AsyncMock,
            return_value=SAMPLE_AGENT_RESPONSE_WITH_MISMATCHES,
        ):
            result = await agent.integrate(
                design_name="test_top",
                block_rtl_sources={
                    "block_a": SAMPLE_VERILOG_A,
                    "block_b": SAMPLE_VERILOG_B,
                },
                block_port_summaries=SAMPLE_PORT_SUMMARIES,
                connections=SAMPLE_CONNECTIONS,
            )

        assert len(result["mismatches"]) == 1
        assert result["mismatches"][0]["severity"] == "error"

    @pytest.mark.asyncio
    async def test_llm_called_with_correct_params(self):
        agent = IntegrationLeadAgent()
        mock_call = AsyncMock(return_value=SAMPLE_AGENT_RESPONSE)
        with patch.object(agent.llm, "call", mock_call):
            await agent.integrate(
                design_name="my_design",
                block_rtl_sources={"block_a": SAMPLE_VERILOG_A},
                block_port_summaries=SAMPLE_PORT_SUMMARIES[:1],
                connections=[],
            )

        mock_call.assert_called_once()
        call_kwargs = mock_call.call_args
        assert call_kwargs.kwargs["system"] == SYSTEM_PROMPT
        assert "my_design" in call_kwargs.kwargs["prompt"]
        assert "Integration Lead" in call_kwargs.kwargs["run_name"]

    @pytest.mark.asyncio
    async def test_handles_llm_returning_garbage(self):
        agent = IntegrationLeadAgent()
        with patch.object(
            agent.llm, "call", new_callable=AsyncMock,
            return_value="I don't know what to generate",
        ):
            result = await agent.integrate(
                design_name="test",
                block_rtl_sources={"a": "module a(); endmodule"},
                block_port_summaries=[],
                connections=[],
            )

        assert result.get("parse_error") is True


# ---------------------------------------------------------------------------
# Model name verification
# ---------------------------------------------------------------------------

class TestModelNameUpdates:
    def test_integration_lead_default_model(self):
        from orchestrator.langchain.agents.coresmith_llm import DEFAULT_MODEL
        agent = IntegrationLeadAgent()
        assert agent.llm.model == DEFAULT_MODEL

    def test_integration_testbench_default_model(self):
        from orchestrator.langchain.agents.coresmith_llm import DEFAULT_MODEL
        from orchestrator.langchain.agents.integration_testbench_generator import (
            IntegrationTestbenchGenerator,
        )
        agent = IntegrationTestbenchGenerator()
        assert agent.llm.model == DEFAULT_MODEL

    def test_cli_model_map_has_sonnet_46(self):
        from orchestrator.langchain.agents.coresmith_llm import _CLI_MODEL_MAP
        assert "sonnet-4.6" in _CLI_MODEL_MAP

    def test_sonnet_46_resolves(self):
        from orchestrator.langchain.agents.coresmith_llm import _resolve_model
        resolved = _resolve_model("claude-sonnet-4-6")
        assert "claude-sonnet-4-6" in resolved


# ---------------------------------------------------------------------------
# Prompt file loading
# ---------------------------------------------------------------------------

class TestPromptLoading:
    def test_system_prompt_loaded_from_file(self):
        assert _PROMPT_FILE.exists(), f"Prompt file missing: {_PROMPT_FILE}"
        assert len(SYSTEM_PROMPT) > 100
        assert "Integration Lead" in SYSTEM_PROMPT

    def test_prompt_requires_json_output(self):
        assert "JSON" in SYSTEM_PROMPT

    def test_prompt_covers_compatibility_analysis(self):
        assert "missing_port" in SYSTEM_PROMPT
        assert "width_mismatch" in SYSTEM_PROMPT
        assert "direction_error" in SYSTEM_PROMPT

    def test_prompt_covers_top_level_generation(self):
        assert "Verilog" in SYSTEM_PROMPT
        assert "module" in SYSTEM_PROMPT.lower()
        assert "instantiate" in SYSTEM_PROMPT.lower() or "instantiat" in SYSTEM_PROMPT.lower()


# ---------------------------------------------------------------------------
# integration_check_node (pipeline graph node)
# ---------------------------------------------------------------------------

class TestIntegrationCheckNode:
    """Test the updated integration_check_node that uses IntegrationLeadAgent."""

    @pytest.fixture(autouse=True)
    def _disable_deterministic_compat(self, monkeypatch):
        # These tests exercise the Integration Lead AGENT's mismatch flow with
        # degenerate port fixtures (a single clk-only module reused for every
        # block, block names that don't match SAMPLE_CONNECTIONS). The
        # deterministic compatibility checker (A-Fix 3a, default ON) would
        # rightly flag those as missing_block/missing_port and pollute the
        # asserted error/warning counts. Its own behaviour is covered in
        # test_integration_compat_wiring.py, so disable it here.
        monkeypatch.setenv("CORESMITH_DETERMINISTIC_INTEGRATION_CHECK", "0")

    def _make_completed_blocks(self, names: list[str]) -> list[dict]:
        return [
            {
                "name": n,
                "success": True,
                "rtl_path": f"/tmp/test/rtl/{n}/{n}.v",
            }
            for n in names
        ]

    def _make_state(self, blocks: list[dict], project_root: str = "/tmp/test") -> dict:
        return {
            "project_root": project_root,
            "completed_blocks": blocks,
            "pipeline_done": False,
        }

    @pytest.mark.asyncio
    async def test_skips_with_fewer_than_2_blocks(self):
        # The integration_check_node no longer short-circuits on block count
        # alone; it now tries to parse the available block RTL and skips when
        # nothing parses (the test fixture references nonexistent files at
        # /tmp/test/rtl/...). The "skipped" semantics still hold but the
        # reason string changed.
        from orchestrator.langgraph.pipeline_graph import integration_check_node

        blocks = self._make_completed_blocks(["only_one"])
        state = self._make_state(blocks)

        with patch(
            "orchestrator.langgraph.pipeline_graph.write_graph_event"
        ):
            result = await integration_check_node(state)

        ir = result["integration_result"]
        assert ir["skipped"] is True
        assert "rtl" in ir["reason"].lower() or "fewer than 2" in ir["reason"].lower()

    @pytest.mark.asyncio
    async def test_skips_with_no_connections(self):
        from orchestrator.langgraph.pipeline_graph import integration_check_node

        blocks = self._make_completed_blocks(["a", "b"])
        state = self._make_state(blocks)

        with patch(
            "orchestrator.langgraph.pipeline_graph.load_architecture_connections",
            return_value=([], "chip_top"),
        ), patch(
            "orchestrator.langgraph.pipeline_graph.write_graph_event"
        ):
            result = await integration_check_node(state)

        ir = result["integration_result"]
        assert ir["skipped"] is True
        # With no connections AND parsed blocks > 0 the node now falls
        # through to the RTL discovery path; the temp paths don't exist so
        # we end up with "no block rtl could be parsed".
        reason = ir["reason"].lower()
        assert "connection" in reason or "rtl" in reason

    @pytest.mark.asyncio
    async def test_skips_when_no_rtl_parsed(self):
        from orchestrator.langgraph.pipeline_graph import integration_check_node
        from orchestrator.langgraph.integration_helpers import VerilogModule

        blocks = self._make_completed_blocks(["a", "b"])
        state = self._make_state(blocks)

        empty_module = VerilogModule(name="", filepath="")

        with patch(
            "orchestrator.langgraph.pipeline_graph.load_architecture_connections",
            return_value=(SAMPLE_CONNECTIONS, "chip_top"),
        ), patch(
            "orchestrator.langgraph.pipeline_graph.discover_block_rtl",
            return_value={"a": "/tmp/a.v", "b": "/tmp/b.v"},
        ), patch(
            "orchestrator.langgraph.pipeline_graph.parse_verilog_ports",
            return_value=empty_module,
        ), patch(
            "orchestrator.langgraph.pipeline_graph.write_graph_event"
        ):
            result = await integration_check_node(state)

        ir = result["integration_result"]
        assert ir["skipped"] is True
        assert "parsed" in ir["reason"].lower()

    @pytest.mark.asyncio
    async def test_calls_integration_lead_agent(self):
        from orchestrator.langgraph.pipeline_graph import integration_check_node
        from orchestrator.langgraph.integration_helpers import (
            VerilogModule, VerilogPort,
        )

        blocks = self._make_completed_blocks(["block_a", "block_b"])
        state = self._make_state(blocks)

        mod_a = VerilogModule(
            name="block_a",
            ports=[
                VerilogPort("clk", "input"),
                VerilogPort("rst_n", "input"),
                VerilogPort("data_out", "output", width=8, msb=7, lsb=0),
            ],
        )
        mod_b = VerilogModule(
            name="block_b",
            ports=[
                VerilogPort("clk", "input"),
                VerilogPort("rst_n", "input"),
                VerilogPort("data_in", "input", width=8, msb=7, lsb=0),
            ],
        )

        def _mock_parse(path):
            if "block_a" in path:
                return mod_a
            return mod_b

        mock_integrate = AsyncMock(return_value={
            "verilog": CHIP_TOP_BLOCK_A_B,
            "mismatches": [],
            "module_name": "test_top",
            "wire_count": 1,
            "skipped_connections": [],
            "notes": "",
        })

        with patch(
            "orchestrator.langgraph.pipeline_graph.load_architecture_connections",
            return_value=(SAMPLE_CONNECTIONS, "test_top"),
        ), patch(
            "orchestrator.langgraph.pipeline_graph.discover_block_rtl",
            return_value={
                "block_a": "/tmp/block_a.v",
                "block_b": "/tmp/block_b.v",
            },
        ), patch(
            "orchestrator.langgraph.pipeline_graph.parse_verilog_ports",
            side_effect=_mock_parse,
        ), patch(
            "orchestrator.langgraph.pipeline_graph.lint_top_level",
            return_value={"clean": True, "warnings": ""},
        ), patch(
            "orchestrator.langchain.agents.integration_lead.IntegrationLeadAgent.integrate",
            mock_integrate,
        ), patch(
            "orchestrator.langgraph.pipeline_graph.write_graph_event"
        ), patch(
            "pathlib.Path.read_text",
            return_value=CHIP_TOP_BLOCK_A_B,
        ), patch(
            "pathlib.Path.mkdir",
        ), patch(
            "pathlib.Path.write_text",
        ):
            result = await integration_check_node(state)

        mock_integrate.assert_called_once()
        call_kwargs = mock_integrate.call_args.kwargs
        assert call_kwargs["design_name"] == "test_top"
        assert "block_a" in call_kwargs["block_rtl_sources"]
        assert "block_b" in call_kwargs["block_rtl_sources"]

        ir = result["integration_result"]
        assert ir["top_module"] == "test_top"
        assert ir["lint_clean"] is True

    @pytest.mark.asyncio
    async def test_agent_failure_skips_gracefully(self):
        from orchestrator.langgraph.pipeline_graph import integration_check_node
        from orchestrator.langgraph.integration_helpers import (
            VerilogModule, VerilogPort,
        )

        blocks = self._make_completed_blocks(["a", "b"])
        state = self._make_state(blocks)

        mod = VerilogModule(
            name="a", ports=[VerilogPort("clk", "input")],
        )

        with patch(
            "orchestrator.langgraph.pipeline_graph.load_architecture_connections",
            return_value=(SAMPLE_CONNECTIONS, "chip_top"),
        ), patch(
            "orchestrator.langgraph.pipeline_graph.discover_block_rtl",
            return_value={"a": "/tmp/a.v", "b": "/tmp/b.v"},
        ), patch(
            "orchestrator.langgraph.pipeline_graph.parse_verilog_ports",
            return_value=mod,
        ), patch(
            "orchestrator.langchain.agents.integration_lead.IntegrationLeadAgent.integrate",
            new_callable=AsyncMock,
            side_effect=RuntimeError("LLM timeout"),
        ), patch(
            "orchestrator.langgraph.pipeline_graph.write_graph_event"
        ), patch(
            "pathlib.Path.read_text",
            return_value=CHIP_TOP_AB,
        ):
            result = await integration_check_node(state)

        ir = result["integration_result"]
        assert ir["skipped"] is True
        assert "failed" in ir["reason"].lower()

    @pytest.mark.asyncio
    async def test_agent_parse_error_skips(self):
        from orchestrator.langgraph.pipeline_graph import integration_check_node
        from orchestrator.langgraph.integration_helpers import (
            VerilogModule, VerilogPort,
        )

        blocks = self._make_completed_blocks(["a", "b"])
        state = self._make_state(blocks)

        mod = VerilogModule(
            name="a", ports=[VerilogPort("clk", "input")],
        )

        with patch(
            "orchestrator.langgraph.pipeline_graph.load_architecture_connections",
            return_value=(SAMPLE_CONNECTIONS, "chip_top"),
        ), patch(
            "orchestrator.langgraph.pipeline_graph.discover_block_rtl",
            return_value={"a": "/tmp/a.v", "b": "/tmp/b.v"},
        ), patch(
            "orchestrator.langgraph.pipeline_graph.parse_verilog_ports",
            return_value=mod,
        ), patch(
            "orchestrator.langchain.agents.integration_lead.IntegrationLeadAgent.integrate",
            new_callable=AsyncMock,
            return_value={
                "verilog": "",
                "mismatches": [],
                "module_name": "chip_top",
                "wire_count": 0,
                "skipped_connections": [],
                "notes": "parse failed",
                "parse_error": True,
            },
        ), patch(
            "orchestrator.langgraph.pipeline_graph.write_graph_event"
        ), patch(
            "pathlib.Path.read_text",
            return_value=CHIP_TOP_AB,
        ):
            result = await integration_check_node(state)

        ir = result["integration_result"]
        assert ir["skipped"] is True
        assert "unparseable" in ir["reason"].lower()

    @pytest.mark.asyncio
    async def test_lint_failure_triggers_interrupt(self):
        from orchestrator.langgraph.pipeline_graph import integration_check_node
        from orchestrator.langgraph.integration_helpers import (
            VerilogModule, VerilogPort,
        )

        blocks = self._make_completed_blocks(["a", "b"])
        state = self._make_state(blocks)

        mod = VerilogModule(
            name="a", ports=[VerilogPort("clk", "input")],
        )

        with patch(
            "orchestrator.langgraph.pipeline_graph.load_architecture_connections",
            return_value=(SAMPLE_CONNECTIONS, "chip_top"),
        ), patch(
            "orchestrator.langgraph.pipeline_graph.discover_block_rtl",
            return_value={"a": "/tmp/a.v", "b": "/tmp/b.v"},
        ), patch(
            "orchestrator.langgraph.pipeline_graph.parse_verilog_ports",
            return_value=mod,
        ), patch(
            "orchestrator.langchain.agents.integration_lead.IntegrationLeadAgent.integrate",
            new_callable=AsyncMock,
            return_value={
                "verilog": CHIP_TOP_AB,
                "mismatches": [],
                "module_name": "chip_top",
                "wire_count": 0,
                "skipped_connections": [],
                "notes": "",
            },
        ), patch(
            "orchestrator.langgraph.pipeline_graph.lint_top_level",
            return_value={"clean": False, "errors": "syntax error line 5"},
        ), patch(
            "orchestrator.langgraph.pipeline_graph.interrupt",
            return_value={"action": "skip"},
        ) as mock_interrupt, patch(
            "orchestrator.langgraph.pipeline_graph.write_graph_event"
        ), patch(
            "pathlib.Path.read_text",
            return_value=CHIP_TOP_AB,
        ), patch(
            "pathlib.Path.mkdir",
        ), patch(
            "pathlib.Path.write_text",
        ):
            result = await integration_check_node(state)

        mock_interrupt.assert_called_once()
        payload = mock_interrupt.call_args[0][0]
        assert payload["type"] == "integration_failure"
        assert payload["lint_clean"] is False
        assert "skip" in payload["supported_actions"]

        ir = result["integration_result"]
        assert ir.get("skipped_by_user") is True

    @pytest.mark.asyncio
    async def test_mismatch_errors_trigger_interrupt(self):
        from orchestrator.langgraph.pipeline_graph import integration_check_node
        from orchestrator.langgraph.integration_helpers import (
            VerilogModule, VerilogPort,
        )

        blocks = self._make_completed_blocks(["a", "b"])
        state = self._make_state(blocks)

        mod = VerilogModule(
            name="a", ports=[VerilogPort("clk", "input")],
        )

        with patch(
            "orchestrator.langgraph.pipeline_graph.load_architecture_connections",
            return_value=(SAMPLE_CONNECTIONS, "chip_top"),
        ), patch(
            "orchestrator.langgraph.pipeline_graph.discover_block_rtl",
            return_value={"a": "/tmp/a.v", "b": "/tmp/b.v"},
        ), patch(
            "orchestrator.langgraph.pipeline_graph.parse_verilog_ports",
            return_value=mod,
        ), patch(
            "orchestrator.langchain.agents.integration_lead.IntegrationLeadAgent.integrate",
            new_callable=AsyncMock,
            return_value={
                "verilog": CHIP_TOP_AB,
                "mismatches": [
                    {
                        "from_block": "a",
                        "to_block": "b",
                        "issue_type": "missing_port",
                        "severity": "error",
                        "description": "Port not found",
                        "suggested_fix": "Add port",
                    },
                ],
                "module_name": "chip_top",
                "wire_count": 0,
                "skipped_connections": [],
                "notes": "",
            },
        ), patch(
            "orchestrator.langgraph.pipeline_graph.lint_top_level",
            return_value={"clean": True, "warnings": ""},
        ), patch(
            "orchestrator.langgraph.pipeline_graph.interrupt",
            return_value={"action": "abort"},
        ) as mock_interrupt, patch(
            "orchestrator.langgraph.pipeline_graph.write_graph_event"
        ), patch(
            "pathlib.Path.read_text",
            return_value=CHIP_TOP_AB,
        ), patch(
            "pathlib.Path.mkdir",
        ), patch(
            "pathlib.Path.write_text",
        ):
            result = await integration_check_node(state)

        mock_interrupt.assert_called_once()
        payload = mock_interrupt.call_args[0][0]
        assert payload["error_count"] == 1

        ir = result["integration_result"]
        assert ir.get("aborted") is True

    @pytest.mark.asyncio
    async def test_clean_integration_passes(self):
        from orchestrator.langgraph.pipeline_graph import integration_check_node
        from orchestrator.langgraph.integration_helpers import (
            VerilogModule, VerilogPort,
        )

        blocks = self._make_completed_blocks(["a", "b"])
        state = self._make_state(blocks)

        mod = VerilogModule(
            name="a", ports=[VerilogPort("clk", "input")],
        )

        with patch(
            "orchestrator.langgraph.pipeline_graph.load_architecture_connections",
            return_value=(SAMPLE_CONNECTIONS, "test_top"),
        ), patch(
            "orchestrator.langgraph.pipeline_graph.discover_block_rtl",
            return_value={"a": "/tmp/a.v", "b": "/tmp/b.v"},
        ), patch(
            "orchestrator.langgraph.pipeline_graph.parse_verilog_ports",
            return_value=mod,
        ), patch(
            "orchestrator.langchain.agents.integration_lead.IntegrationLeadAgent.integrate",
            new_callable=AsyncMock,
            return_value={
                "verilog": CHIP_TOP_AB,
                "mismatches": [],
                "module_name": "test_top",
                "wire_count": 3,
                "skipped_connections": [],
                "notes": "All clean",
            },
        ), patch(
            "orchestrator.langgraph.pipeline_graph.lint_top_level",
            return_value={"clean": True, "warnings": ""},
        ), patch(
            "orchestrator.langgraph.pipeline_graph.write_graph_event"
        ), patch(
            "pathlib.Path.read_text",
            return_value=CHIP_TOP_AB,
        ), patch(
            "pathlib.Path.mkdir",
        ), patch(
            "pathlib.Path.write_text",
        ):
            result = await integration_check_node(state)

        ir = result["integration_result"]
        assert ir["lint_clean"] is True
        assert ir["error_count"] == 0
        assert ir["wire_count"] == 3
        assert ir["top_module"] == "test_top"
        assert ir.get("skipped") is None
        assert ir.get("aborted") is None

    @pytest.mark.asyncio
    async def test_fix_rtl_resume_action(self):
        from orchestrator.langgraph.pipeline_graph import integration_check_node
        from orchestrator.langgraph.integration_helpers import (
            VerilogModule, VerilogPort,
        )

        blocks = self._make_completed_blocks(["a", "b"])
        state = self._make_state(blocks)

        mod = VerilogModule(
            name="a", ports=[VerilogPort("clk", "input")],
        )

        with patch(
            "orchestrator.langgraph.pipeline_graph.load_architecture_connections",
            return_value=(SAMPLE_CONNECTIONS, "chip_top"),
        ), patch(
            "orchestrator.langgraph.pipeline_graph.discover_block_rtl",
            return_value={"a": "/tmp/a.v", "b": "/tmp/b.v"},
        ), patch(
            "orchestrator.langgraph.pipeline_graph.parse_verilog_ports",
            return_value=mod,
        ), patch(
            "orchestrator.langchain.agents.integration_lead.IntegrationLeadAgent.integrate",
            new_callable=AsyncMock,
            return_value={
                "verilog": CHIP_TOP_AB,
                "mismatches": [],
                "module_name": "chip_top",
                "wire_count": 0,
                "skipped_connections": [],
                "notes": "",
            },
        ), patch(
            "orchestrator.langgraph.pipeline_graph.lint_top_level",
            return_value={"clean": False, "errors": "undeclared wire"},
        ), patch(
            "orchestrator.langgraph.pipeline_graph.interrupt",
            # rung3-fixes-1 (defect 2): fix_rtl records the on-disk edit but,
            # because lint is still not clean, the node RE-PARKS (a final
            # interrupt) instead of silently END'ing. The operator then makes
            # an explicit terminal choice -- here, abort.
            side_effect=[
                {"action": "fix_rtl",
                 "rtl_fix_description": "Added wire declaration"},
                {"action": "abort"},
            ],
        ) as mock_interrupt, patch(
            "orchestrator.langgraph.pipeline_graph.write_graph_event"
        ), patch(
            "pathlib.Path.read_text",
            return_value=CHIP_TOP_AB,
        ), patch(
            "pathlib.Path.mkdir",
        ), patch(
            "pathlib.Path.write_text",
        ):
            result = await integration_check_node(state)

        ir = result["integration_result"]
        # The fix is recorded AND the node re-parked (2 interrupts) rather than
        # returning to a silent END; the explicit abort then terminates.
        assert ir["fix_applied"] == "Added wire declaration"
        assert mock_interrupt.call_count == 2
        assert ir.get("aborted") is True

    @pytest.mark.asyncio
    async def test_only_successful_blocks_used(self):
        from orchestrator.langgraph.pipeline_graph import integration_check_node

        blocks = [
            {"name": "a", "success": True, "rtl_path": "/tmp/a.v"},
            {"name": "b", "success": False, "rtl_path": "/tmp/b.v"},
            {"name": "c", "success": True, "rtl_path": "/tmp/c.v"},
        ]
        state = self._make_state(blocks)

        with patch(
            "orchestrator.langgraph.pipeline_graph.load_architecture_connections",
            return_value=([], "chip_top"),
        ), patch(
            "orchestrator.langgraph.pipeline_graph.write_graph_event"
        ):
            result = await integration_check_node(state)

        ir = result["integration_result"]
        assert ir["skipped"] is True


# ---------------------------------------------------------------------------
# integration_check_node: warning-only triage (matches arch constraint flow)
# ---------------------------------------------------------------------------

class TestIntegrationCheckWarningTriage:
    """integration_check_node must triage warning-only mismatches via an
    outer-agent interrupt before letting the run advance to DV. The
    closed-feedback warning from the video_codec v4 run predicted the exact DV
    deadlock that followed, so warnings are no longer silently dropped."""

    @pytest.fixture(autouse=True)
    def _disable_deterministic_compat(self, monkeypatch):
        # Warning-triage tests assert on an agent WARNING-only scenario; the
        # deterministic compatibility checker (A-Fix 3a, default ON) would add
        # error-severity findings on the degenerate fixtures and change the
        # interrupt path. Covered separately in test_integration_compat_wiring.py.
        monkeypatch.setenv("CORESMITH_DETERMINISTIC_INTEGRATION_CHECK", "0")

    def _state(self):
        return {
            "project_root": "/tmp/test",
            "completed_blocks": [
                {"name": "a", "success": True, "rtl_path": "/tmp/a.v"},
                {"name": "b", "success": True, "rtl_path": "/tmp/b.v"},
            ],
            "pipeline_done": False,
        }

    def _common_patches(self, mismatches, interrupt_response):
        from orchestrator.langgraph.integration_helpers import (
            VerilogModule, VerilogPort,
        )
        from contextlib import ExitStack

        stack = ExitStack()
        mod = VerilogModule(name="a", ports=[VerilogPort("clk", "input")])

        patches = [
            patch(
                "orchestrator.langgraph.pipeline_graph.load_architecture_connections",
                return_value=(SAMPLE_CONNECTIONS, "chip_top"),
            ),
            patch(
                "orchestrator.langgraph.pipeline_graph.discover_block_rtl",
                return_value={"a": "/tmp/a.v", "b": "/tmp/b.v"},
            ),
            patch(
                "orchestrator.langgraph.pipeline_graph.parse_verilog_ports",
                return_value=mod,
            ),
            patch(
                "orchestrator.langchain.agents.integration_lead."
                "IntegrationLeadAgent.integrate",
                new_callable=AsyncMock,
                return_value={
                    "verilog": CHIP_TOP_AB,
                    "mismatches": mismatches,
                    "module_name": "chip_top",
                    "wire_count": 0,
                    "skipped_connections": [],
                    "notes": "",
                },
            ),
            patch(
                "orchestrator.langgraph.pipeline_graph.lint_top_level",
                return_value={"clean": True, "warnings": ""},
            ),
            patch(
                "orchestrator.langgraph.pipeline_graph.interrupt",
                return_value=interrupt_response,
            ),
            patch(
                "orchestrator.langgraph.pipeline_graph.write_graph_event"
            ),
            patch("pathlib.Path.read_text", return_value=CHIP_TOP_AB),
            patch("pathlib.Path.mkdir"),
            patch("pathlib.Path.write_text"),
        ]
        for p in patches:
            stack.enter_context(p)
        return stack

    _WARNING = {
        "from_block": "a",
        "to_block": "b",
        "issue_type": "prd_violation",
        "severity": "warning",
        "description": "Closed AXI-Stream feedback loop without bootstrap",
        "suggested_fix": "Add request-driven neighbor lookup",
    }

    @pytest.mark.asyncio
    async def test_warning_only_fires_triage_interrupt(self, monkeypatch):
        monkeypatch.delenv(
            "CORESMITH_NONBLOCKING_INTEGRATION_WARNINGS", raising=False
        )
        from orchestrator.langgraph.pipeline_graph import integration_check_node

        captured = {}

        def fake_interrupt(payload):
            captured.update(payload)
            return {"action": "accept"}

        with self._common_patches([self._WARNING], {"action": "accept"}), \
                patch(
                    "orchestrator.langgraph.pipeline_graph.interrupt",
                    side_effect=fake_interrupt,
                ):
            await integration_check_node(self._state())

        assert captured["type"] == "integration_warning_review"
        assert captured["warning_count"] == 1
        assert captured["error_count"] == 0
        assert captured["lint_clean"] is True
        assert "accept" in captured["supported_actions"]
        assert "retry" in captured["supported_actions"]
        assert "fix_rtl" in captured["supported_actions"]
        assert "abort" in captured["supported_actions"]
        # Outer agent guidance must explicitly mention the bootstrap class of
        # failure so the reviewer understands what they're triaging.
        assert "bootstrap" in captured["outer_agent_guidance"].lower()

    @pytest.mark.asyncio
    async def test_warning_triage_accept_proceeds(self, monkeypatch):
        monkeypatch.delenv(
            "CORESMITH_NONBLOCKING_INTEGRATION_WARNINGS", raising=False
        )
        from orchestrator.langgraph.pipeline_graph import integration_check_node

        with self._common_patches([self._WARNING], {"action": "accept"}):
            result = await integration_check_node(self._state())

        ir = result["integration_result"]
        assert ir["accepted_warnings"] is True
        assert ir["warning_triage_action"] == "accept"
        assert ir.get("aborted") is None
        assert ir["lint_clean"] is True
        # route_after_integration will see lint_clean=True, error_count=0,
        # no aborted -> routes to integration_dv.
        from orchestrator.langgraph.pipeline_graph import route_after_integration
        assert route_after_integration(result) == "integration_dv"

    @pytest.mark.asyncio
    async def test_warning_triage_abort_ends_pipeline(self, monkeypatch):
        monkeypatch.delenv(
            "CORESMITH_NONBLOCKING_INTEGRATION_WARNINGS", raising=False
        )
        from orchestrator.langgraph.pipeline_graph import integration_check_node

        with self._common_patches([self._WARNING], {"action": "abort"}):
            result = await integration_check_node(self._state())

        ir = result["integration_result"]
        assert ir["aborted"] is True
        assert ir["warning_triage_action"] == "abort"
        assert ir.get("accepted_warnings") is None

    @pytest.mark.asyncio
    async def test_warning_triage_retry_ends_with_fix_recorded(
        self, monkeypatch
    ):
        monkeypatch.delenv(
            "CORESMITH_NONBLOCKING_INTEGRATION_WARNINGS", raising=False
        )
        from orchestrator.langgraph.pipeline_graph import integration_check_node

        with self._common_patches(
            [self._WARNING],
            {
                "action": "fix_rtl",
                "rtl_fix_description": "added neighbor bootstrap port",
            },
        ):
            result = await integration_check_node(self._state())

        ir = result["integration_result"]
        # retry / fix_rtl both route to END so the outer agent can issue
        # restart_node via MCP for the right stage.
        assert ir["aborted"] is True
        assert ir["fix_applied"] == "added neighbor bootstrap port"
        assert ir["warning_triage_action"] == "fix_rtl"

    @pytest.mark.asyncio
    async def test_nonblocking_env_var_restores_old_behaviour(self, monkeypatch):
        """CORESMITH_NONBLOCKING_INTEGRATION_WARNINGS=1 restores the
        pre-change behaviour where warning-only integration results
        silently proceed to integration_dv with no interrupt."""
        monkeypatch.setenv("CORESMITH_NONBLOCKING_INTEGRATION_WARNINGS", "1")
        from orchestrator.langgraph.pipeline_graph import integration_check_node

        # If the env var really suppresses the triage, this fake_interrupt
        # must NOT be called by the warning-only path. (The error/lint path
        # would still call interrupt, but we have no errors and lint is clean
        # in this fixture, so any call here is a regression.)
        fake_interrupt = patch(
            "orchestrator.langgraph.pipeline_graph.interrupt",
            side_effect=AssertionError(
                "interrupt() should not fire when "
                "CORESMITH_NONBLOCKING_INTEGRATION_WARNINGS=1"
            ),
        )

        with self._common_patches([self._WARNING], {"action": "accept"}), \
                fake_interrupt:
            result = await integration_check_node(self._state())

        ir = result["integration_result"]
        assert ir["warning_count"] == 1
        assert ir["lint_clean"] is True
        assert ir.get("accepted_warnings") is None
        assert ir.get("warning_triage_action") is None
        assert ir.get("aborted") is None


# ---------------------------------------------------------------------------
# Graph construction still works
# ---------------------------------------------------------------------------

class TestGraphConstruction:
    def test_pipeline_graph_compiles_with_agent_node(self):
        from langgraph.checkpoint.memory import MemorySaver
        from orchestrator.langgraph.pipeline_graph import build_pipeline_graph

        graph = build_pipeline_graph(checkpointer=MemorySaver())
        assert graph is not None

    def test_integration_check_node_in_graph(self):
        from langgraph.checkpoint.memory import MemorySaver
        from orchestrator.langgraph.pipeline_graph import build_pipeline_graph

        graph = build_pipeline_graph(checkpointer=MemorySaver())
        node_names = list(graph.get_graph().nodes.keys())
        assert "integration_check" in node_names
        assert "integration_dv" in node_names
        assert "validation_dv" in node_names


# ---------------------------------------------------------------------------
# Integration helpers still work (parse_verilog_ports, lint, etc.)
# ---------------------------------------------------------------------------

class TestIntegrationHelpers:
    def test_parse_verilog_ports_ansi(self, tmp_path):
        from orchestrator.langgraph.integration_helpers import parse_verilog_ports

        rtl_file = tmp_path / "test.v"
        rtl_file.write_text(SAMPLE_VERILOG_A)

        mod = parse_verilog_ports(str(rtl_file))
        assert mod.name == "block_a"
        assert len(mod.ports) == 4
        assert mod.port_by_name("data_in").width == 8
        assert mod.port_by_name("data_out").direction == "output"

    def test_parse_verilog_ports_nonexistent(self):
        from orchestrator.langgraph.integration_helpers import parse_verilog_ports

        mod = parse_verilog_ports("/nonexistent/file.v")
        assert mod.name == ""

    def test_discover_block_rtl(self, tmp_path):
        from orchestrator.langgraph.integration_helpers import discover_block_rtl

        rtl_dir = tmp_path / "rtl" / "block_a"
        rtl_dir.mkdir(parents=True)
        (rtl_dir / "block_a.v").write_text(SAMPLE_VERILOG_A)

        completed = [
            {"name": "block_a", "success": True},
        ]
        paths = discover_block_rtl(str(tmp_path), completed)
        assert "block_a" in paths

    def test_load_architecture_connections_empty(self, tmp_path):
        from orchestrator.langgraph.integration_helpers import (
            load_architecture_connections,
        )

        (tmp_path / ".coresmith").mkdir()
        connections, name = load_architecture_connections(str(tmp_path))
        assert connections == []
        assert name == "chip_top"


# ---------------------------------------------------------------------------
# assert_blocks_instantiated postcondition
# ---------------------------------------------------------------------------

class TestAssertBlocksInstantiated:
    """Postcondition for Integration Lead's chip_top output.

    Catches the silent-block-drop / glue-stub-substitution failure mode
    that produced the codec stub (entropy_enc -> rle_to_packer_token_bridge).
    """

    def test_all_blocks_present_passes(self):
        chip_top = """\
module chip_top (input clk, input rst_n);
    block_a u_a (.clk(clk), .rst_n(rst_n));
    block_b u_b (.clk(clk), .rst_n(rst_n));
endmodule
"""
        assert assert_blocks_instantiated(
            chip_top, {"block_a", "block_b"}
        ) is None

    def test_missing_block_fails(self):
        chip_top = """\
module chip_top (input clk);
    block_a u_a (.clk(clk));
endmodule
"""
        err = assert_blocks_instantiated(chip_top, {"block_a", "block_b"})
        assert err is not None
        assert "block_b" in err
        assert "postcondition failed" in err.lower()

    def test_codec_stub_substitution_caught(self):
        chip_top = """\
module chip_top (input clk);
    // entropy_enc replaced with a glue stub
    rle_to_packer_token_bridge u_bridge (.clk(clk));
    other_block u_other (.clk(clk));
endmodule
"""
        err = assert_blocks_instantiated(
            chip_top, {"entropy_enc", "other_block"}
        )
        assert err is not None
        assert "entropy_enc" in err

    def test_parameterized_instantiation_recognized(self):
        chip_top = """\
module chip_top;
    block_a #(.WIDTH(8)) u_a (.clk(clk));
endmodule
"""
        assert assert_blocks_instantiated(chip_top, {"block_a"}) is None

    def test_empty_chip_top_with_no_blocks_passes(self):
        assert assert_blocks_instantiated("", set()) is None

    def test_accepts_exact_openframe_pad_adapter_instance(self):
        chip_top = """\
module chip_top;
    reference_codec_openframe_pad_adapter u_openframe_project_wrapper ();
endmodule
"""
        assert assert_blocks_instantiated(
            chip_top, {"openframe_project_wrapper"}
        ) is None

    def test_rejects_misnamed_openframe_pad_adapter_instance(self):
        chip_top = """\
module chip_top;
    reference_codec_openframe_pad_adapter u_unrelated_adapter ();
endmodule
"""
        err = assert_blocks_instantiated(
            chip_top, {"openframe_project_wrapper"}
        )
        assert err is not None
        assert "openframe_project_wrapper" in err

    def test_block_name_appearing_only_as_substring_fails(self):
        chip_top = """\
module chip_top;
    my_block_alpha u_alpha (.clk(clk));
    block_a_alt   u_alt   (.clk(clk));
endmodule
"""
        err = assert_blocks_instantiated(chip_top, {"block_a"})
        assert err is not None
        assert "block_a" in err

    def test_block_name_in_comment_does_not_count(self):
        chip_top = """\
module chip_top;
    // The block_a module would go here but was dropped
    other_block u (.clk(clk));
endmodule
"""
        err = assert_blocks_instantiated(chip_top, {"block_a"})
        assert err is not None
        assert "block_a" in err
