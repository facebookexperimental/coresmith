# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
End-to-end test for the backend (physical design) LangGraph pipeline.

Uses the adder_16bit RTL fixture (orchestrator/tests/fixtures/adder_16bit.v)
as the golden reference design.  The test synthesizes the design with Yosys,
then drives the full backend graph chain:

    init_block -> run_pnr -> drc -> lvs -> timing_signoff -> advance_block -> backend_complete

All EDA tools (Yosys, OpenROAD, Magic, Netgen) run via Nix wrappers.

Marks:
    @pytest.mark.slow         -- takes ~60s
    @pytest.mark.requires_nix -- needs `nix` on PATH
    @pytest.mark.e2e          -- end-to-end integration test
    @pytest.mark.live_llm     -- the backend drives every EDA step (flat-top
                                 synthesis, PnR, ...) through an LLM agent
                                 (``backend_graph._run_eda_driver``), so the
                                 chain needs a real provider CLI

The candidate gate: ``init_design_node`` builds only the design
``integration_check`` adopted (``.coresmith/candidate.json``, read through
``top_module.validated_candidate()``); the fixture records the adder as that
candidate (``_adopt_adder``). ``test_fixture_satisfies_the_candidate_gate``
checks the gate without Nix or an LLM.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from langgraph.checkpoint.memory import MemorySaver

from orchestrator.langgraph.backend_graph import build_backend_graph
from orchestrator.langgraph.backend_helpers import LIBERTY

_DASHBOARD_PATCH = (
    "orchestrator.architecture.specialists.chip_finish_dashboard"
    ".generate_chip_finish_dashboard"
)
_MOCK_DASHBOARD = AsyncMock(return_value="<html><body>stub</body></html>")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_ADDER_RTL = _FIXTURES / "adder_16bit.v"

_HAS_NIX = shutil.which("nix") is not None
_HAS_YOSYS = shutil.which("yosys") is not None

requires_nix = pytest.mark.skipif(not _HAS_NIX, reason="Nix not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthesize_adder(project_root: Path) -> tuple[Path, Path]:
    """Synthesize adder_16bit with Yosys into project_root/syn/output/adder_16bit/.

    Returns (netlist_path, sdc_path).
    """
    block_name = "adder_16bit"
    output_dir = project_root / "syn" / "output" / block_name
    output_dir.mkdir(parents=True, exist_ok=True)

    netlist_path = output_dir / f"{block_name}_netlist.v"
    sdc_path = output_dir / f"{block_name}.sdc"

    liberty = str(LIBERTY)
    rtl = str(_ADDER_RTL)

    script = f"""# Yosys synthesis for {block_name} (Sky130 HD)
read_verilog {rtl}
hierarchy -top {block_name}
proc
flatten
opt
synth -run begin:fine
memory_bram
memory_map
synth -run fine:
dfflibmap -liberty {liberty}
abc -liberty {liberty}
opt_clean
stat -liberty {liberty}
write_verilog -noattr {netlist_path}
"""
    script_path = output_dir / f"synth_{block_name}.ys"
    script_path.write_text(script)

    yosys_bin = str(Path(__file__).resolve().parents[2] / "scripts" / "yosys-nix.sh")

    result = subprocess.run(
        [yosys_bin, "-s", str(script_path)],
        capture_output=True, text=True, timeout=300,
    )

    if result.returncode != 0:
        pytest.fail(
            f"Yosys synthesis failed (rc={result.returncode}):\n"
            f"{result.stderr[-2000:]}"
        )

    assert netlist_path.exists(), f"Netlist not written: {netlist_path}"

    # Generate SDC for 50 MHz
    target_clock_mhz = 50.0
    period_ns = 1000.0 / target_clock_mhz
    sdc_path.write_text(
        f"create_clock -name clk -period {period_ns} [get_ports clk]\n"
        f"set_input_delay {period_ns * 0.2:.1f} -clock clk [all_inputs]\n"
        f"set_output_delay {period_ns * 0.2:.1f} -clock clk [all_outputs]\n"
    )

    return netlist_path, sdc_path


_TOP_RTL_REL = "rtl/adder/adder_16bit.v"


def _adopt_adder(project_root: Path) -> dict:
    """Record the adder as the validated integration candidate.

    ``backend_graph.init_design_node`` only builds the design that
    ``integration_check`` adopted: it reads ``.coresmith/candidate.json``
    through ``top_module.validated_candidate()`` and parks with "Validated
    candidate manifest is missing" otherwise. A single-block design is its own
    top, so the receipt is the adder itself with no expected sub-blocks (the
    same adoption the single-block integration path performs).
    """
    from orchestrator.harness.top_module import write_candidate_receipt

    rtl = project_root / _TOP_RTL_REL
    rtl.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(_ADDER_RTL, rtl)
    return write_candidate_receipt(project_root, "adder_16bit", str(rtl), {}, expected_blocks=[],
                                   note="test_backend_e2e fixture")


def _make_initial_state(project_root: str) -> dict:
    """Build the initial BackendState for a single adder_16bit block."""
    block = {
        "name": "adder_16bit",
        "tier": 1,
        "rtl_path": _TOP_RTL_REL,
        "description": "16-bit pipelined unsigned adder",
    }
    # The per-block synthesis output (flat_top_synthesis_node re-synthesizes
    # the adopted candidate through its EDA driver; kept for the old shape).
    netlist = str(Path(project_root) / "syn" / "output" / "adder_16bit" / "adder_16bit_netlist.v")
    sdc = str(Path(project_root) / "syn" / "output" / "adder_16bit" / "adder_16bit.sdc")
    return {
        "project_root": project_root,
        "target_clock_mhz": 50.0,
        "max_attempts": 3,
        "block_queue": [block],
        # Backend Lead fields
        "frontend_blocks": [block],
        "architecture_connections": [],
        "design_name": "adder_16bit",
        "block_rtl_paths": {},
        "glue_blocks": [],
        "integration_top_path": "",
        "flat_netlist_path": netlist,
        "flat_sdc_path": sdc,
        "synth_gate_count": 0,
        "synth_area_um2": 0.0,
        # Legacy compat
        "current_block_index": 0,
        "current_block": {},
        "attempt": 1,
        "phase": "init",
        "constraints": [],
        "attempt_history": [],
        "previous_error": "",
        "floorplan_result": None,
        "place_result": None,
        "cts_result": None,
        "route_result": None,
        "drc_result": None,
        "lvs_result": None,
        "timing_result": None,
        "power_result": None,
        "debug_result": None,
        "completed_blocks": [],
        "human_response": None,
        "backend_done": False,
        "routed_def_path": "",
        "pnr_verilog_path": "",
        "pwr_verilog_path": "",
        "spef_path": "",
        "gds_path": "",
        "spice_path": "",
        "step_log_paths": {},
        "final_report_path": "",
    }


@pytest.mark.skipif(not _HAS_YOSYS, reason="yosys (the hierarchy elaborator) not installed")
@pytest.mark.asyncio
async def test_fixture_satisfies_the_candidate_gate(tmp_path):
    """The e2e state passes ``init_design_node``'s ``validated_candidate()``
    gate (no Nix needed): the backend gets the adder, not an ask_human park."""
    from orchestrator.langgraph.backend_graph import init_design_node, route_after_init_design

    rec = _adopt_adder(tmp_path)
    assert (tmp_path / ".coresmith" / "candidate.json").is_file() and rec["top_module"] == "adder_16bit"
    out = await init_design_node(_make_initial_state(str(tmp_path)))
    assert out["previous_error"] == "" and out["design_name"] == "adder_16bit"
    assert out["integration_top_path"] == str((tmp_path / _TOP_RTL_REL).resolve())
    assert route_after_init_design(out) == "flat_top_synthesis"


# ═══════════════════════════════════════════════════════════════════════════
# End-to-end: Yosys -> Backend Graph (PnR -> DRC -> LVS -> Timing Sign-off)
# ═══════════════════════════════════════════════════════════════════════════

@requires_nix
@pytest.mark.slow
@pytest.mark.e2e
@pytest.mark.live_llm
class TestBackendE2E:
    """Full backend pipeline on adder_16bit: synth -> PnR -> DRC -> LVS -> sign-off."""

    @pytest.fixture(scope="class")
    def synth_project(self, tmp_path_factory):
        """Synthesize adder_16bit once for all tests in this class.

        Creates the directory structure expected by the backend graph:
            tmp/syn/output/adder_16bit/adder_16bit_netlist.v
            tmp/syn/output/adder_16bit/adder_16bit.sdc
        """
        project_root = tmp_path_factory.mktemp("adder_e2e")
        _adopt_adder(project_root)
        netlist, sdc = _synthesize_adder(project_root)
        return {
            "project_root": project_root,
            "netlist": netlist,
            "sdc": sdc,
        }

    @pytest.mark.asyncio
    @patch(_DASHBOARD_PATCH, _MOCK_DASHBOARD)
    async def test_full_chain_passes(self, synth_project):
        """Run the complete backend graph and verify all stages pass.

        Graph path exercised:
          init_block -> run_pnr -> drc -> lvs -> timing_signoff
          -> advance_block -> backend_complete -> final_report
        """
        project_root = synth_project["project_root"]
        graph = build_backend_graph(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "e2e-adder-16bit"}}
        state = _make_initial_state(str(project_root))

        result = await graph.ainvoke(state, config)

        # -- Graph completed --
        assert result["backend_done"] is True, (
            f"Backend did not complete. Phase: {result.get('phase')}, "
            f"error: {result.get('previous_error', '')[:500]}"
        )

        # -- Single block processed --
        completed = result["completed_blocks"]
        assert len(completed) == 1, f"Expected 1 block, got {len(completed)}"

        block_result = completed[0]
        assert block_result["name"] == "adder_16bit"
        assert block_result["success"] is True, (
            f"Block failed: {block_result.get('error', 'unknown')}"
        )

        # -- PnR produced artifacts --
        assert result.get("routed_def_path"), "No routed DEF produced"
        assert Path(result["routed_def_path"]).exists(), "Routed DEF missing from disk"

        # -- DRC clean --
        drc = result.get("drc_result") or {}
        assert drc.get("clean") is True, (
            f"DRC not clean: {drc.get('violation_count', '?')} violations"
        )

        # -- LVS match --
        lvs = result.get("lvs_result") or {}
        assert lvs.get("match") is True, (
            f"LVS mismatch: device_delta={lvs.get('device_delta')}, "
            f"net_delta={lvs.get('net_delta')}"
        )

        # -- Timing met --
        timing = result.get("timing_result") or {}
        assert timing.get("met") is True, (
            f"Timing violated: WNS={timing.get('wns_ns')} ns"
        )

        # -- GDS produced --
        assert result.get("gds_path"), "No GDS path"
        assert Path(result["gds_path"]).exists(), "GDS missing from disk"

        # -- SPICE produced (for LVS) --
        assert result.get("spice_path"), "No SPICE path"
        assert Path(result["spice_path"]).exists(), "SPICE missing from disk"

        # -- Power metrics non-zero --
        power = result.get("power_result") or {}
        assert power.get("total_power_mw", 0) > 0, "Total power should be > 0"

    @pytest.mark.asyncio
    @patch(_DASHBOARD_PATCH, _MOCK_DASHBOARD)
    async def test_artifacts_are_valid(self, synth_project):
        """Verify the output artifacts contain expected content."""
        project_root = synth_project["project_root"]
        graph = build_backend_graph(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "e2e-adder-artifacts"}}
        state = _make_initial_state(str(project_root))

        result = await graph.ainvoke(state, config)

        if not result.get("backend_done") or not result["completed_blocks"][0].get("success"):
            pytest.skip("Backend did not pass -- artifact validation not applicable")

        # Routed DEF should reference the design
        routed_def = Path(result["routed_def_path"])
        def_text = routed_def.read_text()
        assert "adder_16bit" in def_text, "DEF does not reference adder_16bit"

        # PnR Verilog should exist and reference the design
        pnr_v = Path(result.get("pnr_verilog_path", ""))
        if pnr_v.exists():
            v_text = pnr_v.read_text()
            assert "adder_16bit" in v_text

        # GDS should be non-empty binary
        gds = Path(result["gds_path"])
        assert gds.stat().st_size > 1000, f"GDS too small: {gds.stat().st_size} bytes"

    @pytest.mark.asyncio
    @patch(_DASHBOARD_PATCH, _MOCK_DASHBOARD)
    async def test_step_logs_recorded(self, synth_project):
        """Verify that step log paths are captured in the result."""
        project_root = synth_project["project_root"]
        graph = build_backend_graph(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "e2e-adder-logs"}}
        state = _make_initial_state(str(project_root))

        result = await graph.ainvoke(state, config)

        if not result.get("backend_done"):
            pytest.skip("Backend did not complete")

        logs = result.get("step_log_paths") or {}
        assert "pnr" in logs, f"Missing pnr log. Keys: {list(logs.keys())}"
        assert Path(logs["pnr"]).exists(), f"PnR log file missing: {logs['pnr']}"

        block_result = result["completed_blocks"][0]
        block_logs = block_result.get("step_log_paths", {})
        assert "pnr" in block_logs, "PnR log not recorded in completed_blocks"
