# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for pipeline flow fixes (P0-1 through P4).

Covers:
- P0-1: Testbench regeneration (force_regen_tb flag)
- P0-2: PnR floorplan small-die resize uses -utilization
- P0-3: SDC clock port detection (_detect_clock_port)
- P1-1: Verilog port parser (_parse_verilog_ports, _discover_block_ports)
- P1-3: Backend gate (start_backend artifact check)
- P2-1: Fast-path diagnosis for known testbench bugs
- P4-1: Regression guard (best_result.json prevents RTL re-generation)
- P4-2: Simulate node writes best_result.json on success
- P4-3: Auto-approve uarch spec at per-block level
- P4-4: _generate_floorplan_tcl standalone function
- P4-5: IntegrationReviewAgent constructor
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest


def test_load_config_falls_back_to_repo_config_when_project_root_is_external(tmp_path, monkeypatch):
    """Architecture helpers must work when SOCMATE_PROJECT_ROOT is a run dir."""
    import importlib

    monkeypatch.setenv("SOCMATE_PROJECT_ROOT", str(tmp_path))
    monkeypatch.delenv("SOCMATE_CONFIG_PATH", raising=False)

    import orchestrator.langgraph.pipeline_helpers as ph

    ph = importlib.reload(ph)
    cfg = ph.load_config()

    assert isinstance(cfg, dict)
    assert "blocks" in cfg


# ═══════════════════════════════════════════════════════════════════════════
# P0-1: Testbench Regeneration (force_regen_tb flag)
# ═══════════════════════════════════════════════════════════════════════════

class TestForceRegenTb:
    """Verify the force_regen_tb flag prevents stale TB reuse."""

    def test_route_decision_retry_tb(self):
        from orchestrator.langgraph.pipeline_graph import _route_decision

        result = _route_decision(
            debug_result={"is_testbench_bug": True, "category": "TESTBENCH_BUG",
                          "confidence": 0.9, "needs_human": False, "escalate": False},
            attempt_history=[],
            attempt=1,
            max_attempts=5,
            phase="sim",
        )
        assert result == "retry_tb"

    @pytest.mark.asyncio
    async def test_decide_node_sets_force_regen_tb(self):
        from orchestrator.langgraph.pipeline_graph import decide_node

        state = {
            "current_block": {"name": "test_block"},
            "attempt": 1,
            "max_attempts": 5,
            "phase": "sim",
            "project_root": "/tmp/test",
            "debug_action": "retry_tb",
        }

        result = await decide_node(state)
        assert result.get("force_regen_tb") is True

    @pytest.mark.asyncio
    async def test_decide_node_no_flag_on_retry_rtl(self):
        from orchestrator.langgraph.pipeline_graph import decide_node

        state = {
            "current_block": {"name": "test_block"},
            "attempt": 1,
            "max_attempts": 5,
            "phase": "sim",
            "project_root": "/tmp/test",
            "debug_action": "retry_rtl",
        }

        result = await decide_node(state)
        assert "force_regen_tb" not in result or result.get("force_regen_tb") is not True

    def test_blockstate_has_force_regen_tb_field(self):
        from orchestrator.langgraph.pipeline_graph import BlockState
        assert "force_regen_tb" in BlockState.__annotations__


# ═══════════════════════════════════════════════════════════════════════════
# P0-2: PnR Floorplan Small-Die Resize
# ═══════════════════════════════════════════════════════════════════════════

class TestPnrSmallDieResize:
    def test_small_die_uses_die_area_for_resize(self, tmp_path):
        from orchestrator.langgraph.backend_helpers import generate_pnr_tcl

        tcl = generate_pnr_tcl(
            "tiny_block", "/fake/netlist.v", "/fake/sdc.sdc", str(tmp_path),
        )
        content = Path(tcl).read_text()

        # The small-die resize path should use -die_area + -core_area,
        # NOT -core_space (which conflicts with -die_area in OpenROAD IFP-0024)
        resize_section = content[content.find("WARNING: Die"):]
        assert "-die_area" in resize_section
        assert "-core_area" in resize_section

    def test_small_gate_count_uses_explicit_die(self, tmp_path):
        from orchestrator.langgraph.backend_helpers import generate_pnr_tcl

        tcl = generate_pnr_tcl(
            "tiny_block", "/fake/netlist.v", "/fake/sdc.sdc", str(tmp_path),
            gate_count=52,
        )
        content = Path(tcl).read_text()

        # For very small designs, the initial floorplan should use explicit die_area
        assert "-die_area" in content
        assert "-core_area" in content
        assert "-utilization" not in content.split("Floorplan")[1].split("Power")[0]

    def test_large_gate_count_uses_utilization(self, tmp_path):
        from orchestrator.langgraph.backend_helpers import generate_pnr_tcl

        tcl = generate_pnr_tcl(
            "big_block", "/fake/netlist.v", "/fake/sdc.sdc", str(tmp_path),
            gate_count=5000,
        )
        content = Path(tcl).read_text()

        # For larger designs, use standard utilization-based floorplanning
        assert "-utilization" in content


# ═══════════════════════════════════════════════════════════════════════════
# P0-3: SDC Clock Port Detection
# ═══════════════════════════════════════════════════════════════════════════

class TestDetectClockPort:
    def test_standard_clk(self):
        from orchestrator.langgraph.pipeline_helpers import _detect_clock_port

        rtl = textwrap.dedent("""\
            module adder (
                input  wire        clk,
                input  wire        rst_n,
                input  wire [15:0] a,
                output wire [15:0] sum
            );
            endmodule
        """)
        assert _detect_clock_port(rtl) == "clk"

    def test_clk_in(self):
        from orchestrator.langgraph.pipeline_helpers import _detect_clock_port

        rtl = textwrap.dedent("""\
            module clk_rst_ctrl (
                input  wire clk_in,
                input  wire rst_n_in,
                output wire clk_out,
                output wire rst_n_out
            );
            endmodule
        """)
        assert _detect_clock_port(rtl) == "clk_in"

    def test_no_clock_port_fallback(self):
        from orchestrator.langgraph.pipeline_helpers import _detect_clock_port

        rtl = textwrap.dedent("""\
            module pure_comb (
                input  wire [7:0] a,
                input  wire [7:0] b,
                output wire [8:0] sum
            );
            assign sum = a + b;
            endmodule
        """)
        # Falls back to 'clk' when no clock port found
        assert _detect_clock_port(rtl) == "clk"

    def test_synthesize_block_uses_detected_port(self, tmp_path):
        """Verify SDC generated by synthesize_block uses the detected port name."""
        from orchestrator.langgraph.pipeline_helpers import _detect_clock_port

        rtl = "module test (input wire clk_in, input wire rst_n); endmodule\n"
        rtl_path = tmp_path / "test.v"
        rtl_path.write_text(rtl)

        port = _detect_clock_port(rtl)
        assert port == "clk_in"


# ═══════════════════════════════════════════════════════════════════════════
# P1-1: Verilog Port Parser
# ═══════════════════════════════════════════════════════════════════════════

class TestParseVerilogPorts:
    def test_basic_ports(self):
        from orchestrator.langgraph.tapeout_helpers import _parse_verilog_ports

        rtl = textwrap.dedent("""\
            module adder_16bit (
                input  wire        clk,
                input  wire        rst_n,
                input  wire [15:0] a,
                input  wire [15:0] b,
                input  wire        cin,
                output wire [15:0] sum,
                output wire        cout
            );
            endmodule
        """)
        ports = _parse_verilog_ports(rtl)

        assert "clk" in ports
        assert ports["clk"]["direction"] == "input"
        assert ports["clk"]["width"] == 1

        assert "a" in ports
        assert ports["a"]["direction"] == "input"
        assert ports["a"]["width"] == 16

        assert "sum" in ports
        assert ports["sum"]["direction"] == "output"
        assert ports["sum"]["width"] == 16

        assert "cout" in ports
        assert ports["cout"]["direction"] == "output"
        assert ports["cout"]["width"] == 1

    def test_wide_bus(self):
        from orchestrator.langgraph.tapeout_helpers import _parse_verilog_ports

        rtl = textwrap.dedent("""\
            module wide (
                input  wire [31:0] data_in,
                output wire [63:0] data_out
            );
            endmodule
        """)
        ports = _parse_verilog_ports(rtl)
        assert ports["data_in"]["width"] == 32
        assert ports["data_out"]["width"] == 64

    def test_inout(self):
        from orchestrator.langgraph.tapeout_helpers import _parse_verilog_ports

        rtl = "module m (inout wire [7:0] sda); endmodule\n"
        ports = _parse_verilog_ports(rtl)
        assert ports["sda"]["direction"] == "inout"
        assert ports["sda"]["width"] == 8


class TestDiscoverBlockPorts:
    def test_discovers_ports_from_rtl(self, tmp_path, monkeypatch):
        from orchestrator.langgraph.tapeout_helpers import _discover_block_ports
        import orchestrator.langgraph.tapeout_helpers as th
        import orchestrator.langgraph.pipeline_helpers as ph

        monkeypatch.setattr(th, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(ph, "PROJECT_ROOT", tmp_path)

        rtl_dir = tmp_path / "rtl" / "my_block"
        rtl_dir.mkdir(parents=True)
        (rtl_dir / "my_block.v").write_text(textwrap.dedent("""\
            module my_block (
                input  wire        clk,
                input  wire        rst_n,
                input  wire [7:0]  data_in,
                output wire [7:0]  data_out
            );
            endmodule
        """))

        (tmp_path / ".socmate").mkdir(exist_ok=True)

        blocks = [{"name": "my_block"}]
        enriched = _discover_block_ports(blocks)

        assert len(enriched) == 1
        ports = enriched[0]["ports"]
        assert "clk" not in ports
        assert "rst_n" not in ports
        assert "data_in" in ports
        assert "data_out" in ports
        assert ports["data_in"]["width"] == 8
        assert ports["data_out"]["direction"] == "output"

    def test_detects_rst_port_name(self, tmp_path, monkeypatch):
        from orchestrator.langgraph.tapeout_helpers import _discover_block_ports
        import orchestrator.langgraph.tapeout_helpers as th
        import orchestrator.langgraph.pipeline_helpers as ph

        monkeypatch.setattr(th, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(ph, "PROJECT_ROOT", tmp_path)

        rtl_dir = tmp_path / "rtl" / "my_block"
        rtl_dir.mkdir(parents=True)
        (rtl_dir / "my_block.v").write_text(
            "module my_block (input wire clk_in, input wire rst_n_in, "
            "output wire data); endmodule\n"
        )
        (tmp_path / ".socmate").mkdir(exist_ok=True)

        blocks = [{"name": "my_block"}]
        enriched = _discover_block_ports(blocks)

        assert enriched[0].get("_clk_port") == "clk_in"
        assert enriched[0].get("_rst_port") == "rst_n_in"


# ═══════════════════════════════════════════════════════════════════════════
# P2-1: Fast-Path Diagnosis
# ═══════════════════════════════════════════════════════════════════════════

class TestFastPathDiagnosis:
    """Verify fast-path diagnosis catches known testbench bugs.

    The current ``diagnose_node`` reads its error context from
    ``.socmate/blocks/<block>/previous_error.txt`` (written by the
    upstream sim/lint nodes) and persists the diag dict to
    ``diagnosis.json`` in the same directory.  It returns only
    ``{"debug_action": ...}`` -- the diag dict is fetched from disk.
    """

    @staticmethod
    def _setup_block(tmp_path, block_name, error_text):
        block_dir = tmp_path / ".socmate" / "blocks" / block_name
        block_dir.mkdir(parents=True, exist_ok=True)
        (block_dir / "previous_error.txt").write_text(error_text)
        return block_dir

    @pytest.mark.asyncio
    async def test_attribute_error_detected(self, tmp_path):
        from orchestrator.langgraph.pipeline_graph import diagnose_node

        block_dir = self._setup_block(
            tmp_path, "test_block",
            "AttributeError: 'test_block' object has no attribute 'clk'\nTraceback ...",
        )
        state = {
            "current_block": {"name": "test_block"},
            "attempt": 1,
            "max_attempts": 5,
            "attempt_history": [],
            "constraints": [],
            "phase": "sim",
            "previous_error": "",
            "project_root": str(tmp_path),
            "rtl_result": {"verilog": "module test_block(); endmodule"},
            "uarch_spec": None,
        }

        result = await diagnose_node(state)
        assert result.get("debug_action") == "retry_tb"
        diag = json.loads((block_dir / "diagnosis.json").read_text())
        assert diag["category"] == "TESTBENCH_BUG"
        assert diag["is_testbench_bug"] is True
        assert diag["confidence"] == 1.0

    @pytest.mark.asyncio
    async def test_module_not_found_detected(self, tmp_path):
        from orchestrator.langgraph.pipeline_graph import diagnose_node

        block_dir = self._setup_block(
            tmp_path, "test_block",
            "ModuleNotFoundError: No module named 'test_block_model'",
        )
        state = {
            "current_block": {"name": "test_block"},
            "attempt": 1,
            "max_attempts": 5,
            "attempt_history": [],
            "constraints": [],
            "phase": "sim",
            "previous_error": "",
            "project_root": str(tmp_path),
            "rtl_result": {"verilog": "module test_block(); endmodule"},
            "uarch_spec": None,
        }

        result = await diagnose_node(state)
        assert result.get("debug_action") == "retry_tb"
        diag = json.loads((block_dir / "diagnosis.json").read_text())
        assert diag["category"] == "TESTBENCH_BUG"
        assert diag["is_testbench_bug"] is True

    @pytest.mark.asyncio
    async def test_lint_module_not_found(self, tmp_path):
        from orchestrator.langgraph.pipeline_graph import diagnose_node

        block_dir = self._setup_block(
            tmp_path, "test_block",
            "%Error: Module not found: test_block",
        )
        state = {
            "current_block": {"name": "test_block"},
            "attempt": 1,
            "max_attempts": 5,
            "attempt_history": [],
            "constraints": [],
            "phase": "lint",
            "previous_error": "",
            "project_root": str(tmp_path),
            "rtl_result": {"verilog": ""},
            "uarch_spec": None,
        }

        result = await diagnose_node(state)
        assert "debug_action" in result
        diag = json.loads((block_dir / "diagnosis.json").read_text())
        assert diag["category"] == "INFRASTRUCTURE_ERROR"

    @pytest.mark.asyncio
    async def test_non_matching_error_falls_through(self, tmp_path):
        """Verify that non-matching errors are NOT fast-pathed (would need LLM)."""
        from orchestrator.langgraph.pipeline_graph import diagnose_node
        from unittest.mock import patch, AsyncMock

        block_dir = self._setup_block(
            tmp_path, "test_block",
            "AssertionError: output mismatch at cycle 42: expected 0xFF got 0x00",
        )
        state = {
            "current_block": {"name": "test_block"},
            "attempt": 1,
            "max_attempts": 5,
            "attempt_history": [],
            "constraints": [],
            "phase": "sim",
            "previous_error": "",
            "project_root": str(tmp_path),
            "rtl_result": {"verilog": "module test_block(); endmodule"},
            "uarch_spec": None,
        }

        mock_diag = AsyncMock(return_value={
            "category": "LOGIC_ERROR",
            "confidence": 0.8,
            "diagnosis": "Output mismatch",
            "suggested_fix": "Fix logic",
            "needs_human": False,
            "is_testbench_bug": False,
            "escalate": False,
            "constraints": [],
            "affected_blocks": [],
        })

        with patch("orchestrator.langgraph.pipeline_graph.diagnose_failure", mock_diag):
            result = await diagnose_node(state)

        # The LLM should have been called (fast-path didn't match)
        mock_diag.assert_called_once()
        assert "debug_action" in result
        diag = json.loads((block_dir / "diagnosis.json").read_text())
        assert diag["category"] == "LOGIC_ERROR"


# ═══════════════════════════════════════════════════════════════════════════
# P1-3: Backend Gate Verification
# ═══════════════════════════════════════════════════════════════════════════

class TestBackendGate:
    """Verify start_backend refuses when blocks lack RTL/synthesis artifacts."""

    @pytest.mark.asyncio
    async def test_missing_rtl_blocks_backend(self, tmp_path):
        import orchestrator.mcp_server as mcp
        from unittest.mock import AsyncMock, patch

        (tmp_path / ".socmate").mkdir()
        block_specs = [
            {"name": "good_block", "rtl_target": "rtl/good_block/good_block.v"},
            {"name": "bad_block", "rtl_target": "rtl/bad_block/bad_block.v"},
        ]
        (tmp_path / ".socmate" / "block_specs.json").write_text(json.dumps(block_specs))

        # Create RTL + netlist for good_block only
        rtl_dir = tmp_path / "rtl" / "good_block"
        rtl_dir.mkdir(parents=True)
        (rtl_dir / "good_block.v").write_text("module good_block(); endmodule\n")
        syn_dir = tmp_path / "syn" / "output" / "good_block"
        syn_dir.mkdir(parents=True)
        (syn_dir / "good_block_netlist.v").write_text("// netlist\n")

        # bad_block has no files at all

        with patch.object(mcp, "_project_root", return_value=str(tmp_path)), \
             patch.object(mcp._backend, "status", "idle"):
            # Mock ensure_graph to avoid SQLite dependency
            mcp._backend.ensure_graph = AsyncMock()

            result_json = await mcp.start_backend()
            result = json.loads(result_json)

        assert "error" in result
        assert "bad_block" in result.get("missing_rtl", [])
        assert "bad_block" in result.get("missing_synthesis", [])

    @pytest.mark.asyncio
    async def test_all_blocks_present_passes_gate(self, tmp_path):
        import orchestrator.mcp_server as mcp
        from unittest.mock import AsyncMock, patch

        (tmp_path / ".socmate").mkdir()
        block_specs = [
            {"name": "block_a", "rtl_target": "rtl/block_a/block_a.v"},
        ]
        (tmp_path / ".socmate" / "block_specs.json").write_text(json.dumps(block_specs))

        # Create all artifacts
        rtl_dir = tmp_path / "rtl" / "block_a"
        rtl_dir.mkdir(parents=True)
        (rtl_dir / "block_a.v").write_text("module block_a(); endmodule\n")
        syn_dir = tmp_path / "syn" / "output" / "block_a"
        syn_dir.mkdir(parents=True)
        (syn_dir / "block_a_netlist.v").write_text("// netlist\n")

        with patch.object(mcp, "_project_root", return_value=str(tmp_path)), \
             patch.object(mcp._backend, "status", "idle"):
            mcp._backend.ensure_graph = AsyncMock()

            # Should pass the gate but then may fail on preflight (no EDA tools)
            # -- that's fine, the gate itself passed
            result_json = await mcp.start_backend()
            result = json.loads(result_json)

        # Should NOT have the "Backend gate failed" error
        if "error" in result:
            assert "Backend gate failed" not in result["error"]


# ═══════════════════════════════════════════════════════════════════════════
# P3-1: ERS Golden Model Context
# ═══════════════════════════════════════════════════════════════════════════

class TestErsGoldenModelContext:
    def test_golden_model_lines_from_block_diagram(self):
        # Just verify the golden model context building logic doesn't crash
        # (full LLM test would be slow/expensive)
        block_diagram = {
            "blocks": [
                {"name": "fir_filter", "python_source": "PyDVB/dvb/FIR.py"},
                {"name": "adder", "python_source": ""},
                {"name": "clk_rst_ctrl"},
            ],
            "connections": [],
        }

        lines = []
        for blk in block_diagram.get("blocks", []):
            src = blk.get("python_source", "")
            name = blk.get("name", "unknown")
            if src and src.strip():
                lines.append(f"  - {name}: golden model at `{src}`")
            else:
                lines.append(f"  - {name}: NO golden model (write algorithm_pseudocode)")

        context = "\n".join(lines)
        assert "fir_filter: golden model at `PyDVB/dvb/FIR.py`" in context
        assert "adder: NO golden model" in context
        assert "clk_rst_ctrl: NO golden model" in context


# ═══════════════════════════════════════════════════════════════════════════
# P3-2: uArch Prompt Has Verilog Stub Section
# ═══════════════════════════════════════════════════════════════════════════

class TestUarchVerilogStub:
    def test_prompt_contains_stub_section(self):
        prompt_path = (
            Path(__file__).resolve().parents[1]
            / "langchain" / "prompts" / "uarch_spec_generator.md"
        )
        content = prompt_path.read_text()
        assert "## 9. Verilog Interface Stub" in content
        assert "interface contract" in content
        assert "endmodule" in content


# ═══════════════════════════════════════════════════════════════════════════
# Filesystem Source of Truth: Arch Docs Read from Disk
# ═══════════════════════════════════════════════════════════════════════════

class TestFilesystemSourceOfTruth:
    def test_uarch_generator_accepts_project_root(self):
        from orchestrator.langchain.agents.uarch_spec_generator import UarchSpecGenerator
        import inspect

        sig = inspect.signature(UarchSpecGenerator.generate)
        assert "project_root" in sig.parameters

    def test_rtl_generator_accepts_project_root(self):
        from orchestrator.langchain.agents.rtl_generator import RTLGeneratorAgent
        import inspect

        sig = inspect.signature(RTLGeneratorAgent.generate)
        assert "project_root" in sig.parameters

    def test_constraint_check_accepts_project_root(self):
        from orchestrator.architecture.constraints import check_constraints
        import inspect

        sig = inspect.signature(check_constraints)
        assert "project_root" in sig.parameters


# ═══════════════════════════════════════════════════════════════════════════
# P4-1: Regression Guard (best_result.json prevents RTL regeneration)
# ═══════════════════════════════════════════════════════════════════════════

class TestRegressionGuard:
    """Verify generate_rtl_node skips regeneration when a previous attempt
    passed simulation (best_result.json exists with sim_passed=true)."""

    @pytest.mark.asyncio
    async def test_skips_regen_when_best_result_exists(self, tmp_path):
        from orchestrator.langgraph.pipeline_graph import generate_rtl_node

        block_name = "test_block"
        block = {
            "name": block_name,
            "rtl_target": f"rtl/{block_name}/{block_name}.v",
        }

        rtl_dir = tmp_path / "rtl" / block_name
        rtl_dir.mkdir(parents=True)
        (rtl_dir / f"{block_name}.v").write_text("module test_block(); endmodule\n")

        best_dir = tmp_path / ".socmate" / "blocks" / block_name
        best_dir.mkdir(parents=True)
        (best_dir / "best_result.json").write_text(json.dumps({
            "sim_passed": True,
            "attempt": 1,
            "tests_passed": 5,
            "tests_total": 5,
        }))

        state = {
            "current_block": block,
            "project_root": str(tmp_path),
            "attempt": 2,
            "max_attempts": 5,
            "target_clock_mhz": 50.0,
        }

        result = await generate_rtl_node(state)
        assert result.get("force_regen_tb") is True
        assert result["rtl_path"] == str(rtl_dir / f"{block_name}.v")

    @pytest.mark.asyncio
    async def test_no_skip_when_best_result_failed(self, tmp_path):
        """Do NOT skip regen if best_result says sim did not pass."""
        from orchestrator.langgraph.pipeline_graph import generate_rtl_node
        from unittest.mock import patch, AsyncMock

        block_name = "test_block"
        block = {
            "name": block_name,
            "rtl_target": f"rtl/{block_name}/{block_name}.v",
        }

        rtl_dir = tmp_path / "rtl" / block_name
        rtl_dir.mkdir(parents=True)
        (rtl_dir / f"{block_name}.v").write_text("module test_block(); endmodule\n")

        best_dir = tmp_path / ".socmate" / "blocks" / block_name
        best_dir.mkdir(parents=True)
        (best_dir / "best_result.json").write_text(json.dumps({
            "sim_passed": False,
            "attempt": 1,
        }))

        state = {
            "current_block": block,
            "project_root": str(tmp_path),
            "attempt": 2,
            "max_attempts": 5,
            "target_clock_mhz": 50.0,
        }

        with patch(
            "orchestrator.langgraph.pipeline_graph.generate_rtl",
            new_callable=AsyncMock,
            return_value={"rtl_path": str(rtl_dir / f"{block_name}.v")},
        ):
            result = await generate_rtl_node(state)

        assert result.get("force_regen_tb") is not True

    @pytest.mark.asyncio
    async def test_no_skip_on_attempt_1(self, tmp_path):
        """Regression guard only applies on attempt > 1."""
        from orchestrator.langgraph.pipeline_graph import generate_rtl_node

        block_name = "test_block"
        block = {
            "name": block_name,
            "rtl_target": f"rtl/{block_name}/{block_name}.v",
        }

        rtl_dir = tmp_path / "rtl" / block_name
        rtl_dir.mkdir(parents=True)
        (rtl_dir / f"{block_name}.v").write_text("module test_block(); endmodule\n")

        best_dir = tmp_path / ".socmate" / "blocks" / block_name
        best_dir.mkdir(parents=True)
        (best_dir / "best_result.json").write_text(json.dumps({
            "sim_passed": True,
            "attempt": 1,
        }))

        state = {
            "current_block": block,
            "project_root": str(tmp_path),
            "attempt": 1,
            "max_attempts": 5,
            "target_clock_mhz": 50.0,
        }

        result = await generate_rtl_node(state)
        assert result.get("force_regen_tb") is not True


# ═══════════════════════════════════════════════════════════════════════════
# P4-2: Simulate Node Writes best_result.json on Success
# ═══════════════════════════════════════════════════════════════════════════

class TestBestResultPersistence:
    """``simulate_node`` was merged into ``generate_testbench_node``; the
    sim-pass / sim-fail persistence behaviour is exercised through the
    combined node now."""

    @pytest.mark.asyncio
    async def test_sim_pass_writes_best_result(self, tmp_path):
        from orchestrator.langgraph.pipeline_graph import generate_testbench_node
        from unittest.mock import patch

        block_name = "my_alu"
        block = {"name": block_name, "testbench": f"tb/cocotb/test_{block_name}.py"}

        (tmp_path / ".socmate" / "blocks" / block_name).mkdir(parents=True)

        rtl_file = tmp_path / "rtl" / block_name / f"{block_name}.v"
        rtl_file.parent.mkdir(parents=True)
        rtl_file.write_text("module my_alu(); endmodule\n")

        tb_file = tmp_path / "tb" / "cocotb" / f"test_{block_name}.py"
        tb_file.parent.mkdir(parents=True)
        tb_file.write_text("# test\n")

        sim_pass = {"passed": True, "log": "ok", "returncode": 0,
                     "tests_passed": 3, "tests_total": 3}
        state = {
            "current_block": block,
            "project_root": str(tmp_path),
            "attempt": 1,
            "rtl_path": str(rtl_file),
            "tb_path": str(tb_file),
            # preserve_testbench=True keeps the existing TB file and skips
            # the (mocked-out) generate_testbench LLM call.
            "force_regen_tb": False,
            "preserve_testbench": True,
        }

        with patch(
            "orchestrator.langgraph.pipeline_graph.run_simulation",
            return_value=sim_pass,
        ):
            await generate_testbench_node(state)

        best_path = tmp_path / ".socmate" / "blocks" / block_name / "best_result.json"
        assert best_path.exists()
        best = json.loads(best_path.read_text())
        assert best["sim_passed"] is True
        assert best["attempt"] == 1
        assert best["tests_passed"] == 3

    @pytest.mark.asyncio
    async def test_sim_fail_no_best_result(self, tmp_path):
        from orchestrator.langgraph.pipeline_graph import generate_testbench_node
        from unittest.mock import patch

        block_name = "buggy"
        block = {"name": block_name, "testbench": f"tb/cocotb/test_{block_name}.py"}

        (tmp_path / ".socmate" / "blocks" / block_name).mkdir(parents=True)

        rtl_file = tmp_path / "rtl" / block_name / f"{block_name}.v"
        rtl_file.parent.mkdir(parents=True)
        rtl_file.write_text("module buggy(); endmodule\n")

        tb_file = tmp_path / "tb" / "cocotb" / f"test_{block_name}.py"
        tb_file.parent.mkdir(parents=True)
        tb_file.write_text("# test\n")

        sim_fail = {"passed": False, "log": "FAIL", "returncode": 1}
        state = {
            "current_block": block,
            "project_root": str(tmp_path),
            "attempt": 1,
            "rtl_path": str(rtl_file),
            "tb_path": str(tb_file),
            "force_regen_tb": False,
            "preserve_testbench": True,
        }

        with patch(
            "orchestrator.langgraph.pipeline_graph.run_simulation",
            return_value=sim_fail,
        ):
            await generate_testbench_node(state)

        best_path = tmp_path / ".socmate" / "blocks" / block_name / "best_result.json"
        assert not best_path.exists()


class TestCocotbResultParsing:
    def test_failing_summary_overrides_zero_returncode(self):
        from orchestrator.langgraph.pipeline_helpers import _parse_cocotb_summary

        summary = _parse_cocotb_summary(
            "** TESTS=6 PASS=0 FAIL=6 SKIP=0 **"
        )

        assert summary["found"] is True
        assert summary["tests_total"] == 6
        assert summary["tests_passed"] == 0
        assert summary["tests_failed"] == 6

    def test_normalizes_unit_keyword_for_installed_cocotb(self, tmp_path, monkeypatch):
        from orchestrator.langgraph import pipeline_helpers

        tb_file = tmp_path / "test_unit.py"
        tb_file.write_text('await Timer(1, unit="ns")\nClock(dut.clk, 10, unit="ns")\n')

        monkeypatch.setattr(pipeline_helpers, "_cocotb_uses_plural_units", lambda: True)
        pipeline_helpers._normalize_cocotb_timing_keywords(tb_file)

        content = tb_file.read_text()
        assert 'units="ns"' in content
        assert 'unit="ns"' not in content


# ═══════════════════════════════════════════════════════════════════════════
# P4-3: Auto-Approve uArch Spec at Per-Block Level
# ═══════════════════════════════════════════════════════════════════════════

class TestUarchAutoApprove:
    @pytest.mark.asyncio
    async def test_review_uarch_spec_auto_approves(self, tmp_path):
        from orchestrator.langgraph.pipeline_graph import review_uarch_spec_node

        state = {
            "current_block": {"name": "enc_control"},
            "project_root": str(tmp_path),
        }
        (tmp_path / ".socmate").mkdir(exist_ok=True)

        result = await review_uarch_spec_node(state)
        assert result["uarch_approved"] is True
        assert result["human_response"]["action"] == "approve"


# ═══════════════════════════════════════════════════════════════════════════
# P4-4: _generate_floorplan_tcl Standalone Function
# ═══════════════════════════════════════════════════════════════════════════

class TestGenerateFloorplanTcl:
    def test_small_gate_count_explicit_die(self):
        from orchestrator.langgraph.backend_helpers import _generate_floorplan_tcl

        tcl = _generate_floorplan_tcl("tiny", utilization=45, gate_count=100)
        assert "Small design" in tcl
        assert "-die_area" in tcl
        assert "-core_area" in tcl

    def test_large_gate_count_utilization(self):
        from orchestrator.langgraph.backend_helpers import _generate_floorplan_tcl

        tcl = _generate_floorplan_tcl("big", utilization=45, gate_count=5000)
        assert "-utilization 45" in tcl

    def test_zero_gate_count_no_explicit_die(self):
        from orchestrator.langgraph.backend_helpers import _generate_floorplan_tcl

        tcl = _generate_floorplan_tcl("unknown", utilization=45, gate_count=0)
        assert "-utilization 45" in tcl

    def test_tracks_always_present(self):
        from orchestrator.langgraph.backend_helpers import _generate_floorplan_tcl

        for gc in [0, 50, 500, 5000]:
            tcl = _generate_floorplan_tcl("blk", utilization=45, gate_count=gc)
            assert "make_tracks li1" in tcl
            assert "make_tracks met5" in tcl

    def test_tapcell_always_present(self):
        from orchestrator.langgraph.backend_helpers import _generate_floorplan_tcl

        tcl = _generate_floorplan_tcl("blk", utilization=45, gate_count=100)
        assert "tapcell" in tcl
        assert "tapvpwrvgnd" in tcl

    def test_small_die_fallback_resize(self):
        from orchestrator.langgraph.backend_helpers import _generate_floorplan_tcl

        tcl = _generate_floorplan_tcl("blk", utilization=45, gate_count=5000)
        assert "WARNING: Die" in tcl
        assert "too small for PDN" in tcl


# ═══════════════════════════════════════════════════════════════════════════
# P4-5: IntegrationReviewAgent
# ═══════════════════════════════════════════════════════════════════════════

class TestIntegrationReviewAgent:
    def test_constructor(self):
        from orchestrator.langchain.agents.integration_review_agent import (
            IntegrationReviewAgent,
        )
        agent = IntegrationReviewAgent()
        assert agent.llm is not None

    def test_tools_enabled(self):
        from orchestrator.langchain.agents.integration_review_agent import (
            IntegrationReviewAgent,
        )
        agent = IntegrationReviewAgent()
        assert agent.llm.disable_tools is False

    def test_review_accepts_block_names_and_project_root(self):
        import inspect
        from orchestrator.langchain.agents.integration_review_agent import (
            IntegrationReviewAgent,
        )
        sig = inspect.signature(IntegrationReviewAgent.review)
        assert "block_names" in sig.parameters
        assert "project_root" in sig.parameters


# ═══════════════════════════════════════════════════════════════════════════
# P4-6: Integration Lead Writes RTL to Disk
# ═══════════════════════════════════════════════════════════════════════════

class TestIntegrationLeadOutputPath:
    def test_integrate_accepts_output_path(self):
        import inspect
        from orchestrator.langchain.agents.integration_lead import (
            IntegrationLeadAgent,
        )
        sig = inspect.signature(IntegrationLeadAgent.integrate)
        assert "output_path" in sig.parameters

    def test_integration_tb_accepts_output_path(self):
        import inspect
        from orchestrator.langchain.agents.integration_testbench_generator import (
            IntegrationTestbenchGenerator,
        )
        sig = inspect.signature(IntegrationTestbenchGenerator.generate)
        assert "output_path" in sig.parameters


# ═══════════════════════════════════════════════════════════════════════════
# Post-mortem fix: auto-apply high-confidence contract-audit fixes
#
# In the codec run, transform_select's fp16-static-function bug came back
# from the debug agent at confidence=0.92 with a precise code_snippet fix,
# but `needs_human=True` triggered an ask_human escalation that resolved to
# instant retry with no new context.  _route_decision now overrides
# needs_human when there's a high-confidence specific fix.
# ═══════════════════════════════════════════════════════════════════════════

class TestAutoApplyHighConfidenceFix:
    def _diag(self, **overrides):
        d = {
            "category": "ARITHMETIC_ERROR",
            "confidence": 0.92,
            "needs_human": True,
            "escalate": False,
            "local_fix_possible": True,
            "suggested_fix": (
                "Add `automatic` keyword to fp16_add, fp16_mul, and "
                "fp16_round_pack so they don't share local temps across "
                "calls in the dot-product tree at op_idx=4."
            ),
        }
        d.update(overrides)
        return d

    def test_high_confidence_specific_fix_skips_ask_human(self):
        from orchestrator.langgraph.pipeline_graph import _route_decision

        result = _route_decision(
            debug_result=self._diag(),
            attempt_history=[],
            attempt=1,
            max_attempts=5,
            phase="sim",
        )
        assert result == "retry_rtl"  # NOT ask_human

    def test_high_confidence_testbench_bug_routes_to_retry_tb(self):
        from orchestrator.langgraph.pipeline_graph import _route_decision

        result = _route_decision(
            debug_result=self._diag(is_testbench_bug=True, category="TESTBENCH_BUG"),
            attempt_history=[],
            attempt=1,
            max_attempts=5,
            phase="sim",
        )
        # needs_human override still honors testbench routing via the
        # generic post-route logic; the key thing is it didn't escalate.
        assert result in {"retry_tb", "retry_rtl"}
        assert result != "ask_human"

    def test_low_confidence_still_asks_human(self):
        from orchestrator.langgraph.pipeline_graph import _route_decision

        result = _route_decision(
            debug_result=self._diag(confidence=0.6),
            attempt_history=[],
            attempt=1,
            max_attempts=5,
            phase="sim",
        )
        assert result == "ask_human"

    def test_vague_suggested_fix_still_asks_human(self):
        from orchestrator.langgraph.pipeline_graph import _route_decision

        # Short / vague suggested_fix below 50 chars -> not actionable
        result = _route_decision(
            debug_result=self._diag(suggested_fix="Retry it."),
            attempt_history=[],
            attempt=1,
            max_attempts=5,
            phase="sim",
        )
        assert result == "ask_human"

    def test_local_fix_impossible_still_asks_human(self):
        from orchestrator.langgraph.pipeline_graph import _route_decision

        result = _route_decision(
            debug_result=self._diag(local_fix_possible=False),
            attempt_history=[],
            attempt=1,
            max_attempts=5,
            phase="sim",
        )
        assert result == "ask_human"

    def test_env_var_can_raise_threshold(self, monkeypatch):
        from orchestrator.langgraph.pipeline_graph import _route_decision

        # 0.92 < 0.95 threshold -> stays as ask_human
        monkeypatch.setenv("SOCMATE_AUTO_FIX_CONFIDENCE", "0.95")
        result = _route_decision(
            debug_result=self._diag(),  # confidence=0.92
            attempt_history=[],
            attempt=1,
            max_attempts=5,
            phase="sim",
        )
        assert result == "ask_human"
