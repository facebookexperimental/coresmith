# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
End-to-end test for the full tapeout flow: synthesis -> backend -> tapeout.

Exercises the complete chain on adder_16bit:
  1. Yosys synthesis (gate-level netlist + SDC)
  2. Backend graph (PnR -> DRC -> LVS -> timing sign-off)
  3. Tapeout graph (wrapper generation -> wrapper PnR -> wrapper DRC
     -> wrapper LVS -> native MPW precheck)

All EDA tools run via Nix wrappers (no Docker required).

Marks:
    @pytest.mark.slow         -- takes ~120s (full physical design flow)
    @pytest.mark.requires_nix -- needs `nix` on PATH
    @pytest.mark.e2e          -- end-to-end integration test
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
from orchestrator.langgraph.tapeout_graph import build_tapeout_graph

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

requires_nix = pytest.mark.skipif(not _HAS_NIX, reason="Nix not installed")


# ---------------------------------------------------------------------------
# Helpers (reused from test_backend_e2e)
# ---------------------------------------------------------------------------

def _synthesize_adder(project_root: Path) -> tuple[Path, Path]:
    """Synthesize adder_16bit with Yosys."""
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

    target_clock_mhz = 50.0
    period_ns = 1000.0 / target_clock_mhz
    sdc_path.write_text(
        f"create_clock -name clk -period {period_ns} [get_ports clk]\n"
        f"set_input_delay {period_ns * 0.2:.1f} -clock clk [all_inputs]\n"
        f"set_output_delay {period_ns * 0.2:.1f} -clock clk [all_outputs]\n"
    )

    return netlist_path, sdc_path


def _make_backend_state(project_root: str) -> dict:
    """Build the initial BackendState for adder_16bit."""
    block = {
        "name": "adder_16bit",
        "tier": 1,
        "rtl_path": "rtl/adder/adder_16bit.v",
        "description": "16-bit pipelined unsigned adder",
        "ports": {
            "clk": {"width": 1, "direction": "input"},
            "rst": {"width": 1, "direction": "input"},
            "a": {"width": 16, "direction": "input"},
            "b": {"width": 16, "direction": "input"},
            "sum": {"width": 16, "direction": "output"},
            "cout": {"width": 1, "direction": "output"},
        },
    }
    return {
        "project_root": project_root,
        "target_clock_mhz": 50.0,
        "max_attempts": 3,
        "block_queue": [block],
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


def _make_tapeout_state(
    project_root: str,
    blocks: list[dict],
    completed_backend_blocks: list[dict],
) -> dict:
    """Build the initial TapeoutState."""
    return {
        "project_root": project_root,
        "target_clock_mhz": 50.0,
        "blocks": blocks,
        "completed_backend_blocks": completed_backend_blocks,
        "gpio_mapping": None,
        "phase": "init",
        "attempt": 1,
        "max_attempts": 2,
        "previous_error": "",
        "wrapper_result": None,
        "wrapper_pnr_result": None,
        "wrapper_drc_result": None,
        "wrapper_lvs_result": None,
        "precheck_result": None,
        "submission_result": None,
        "wrapper_rtl_path": "",
        "wrapper_routed_def": "",
        "wrapper_gds_path": "",
        "wrapper_spice_path": "",
        "submission_dir": "",
        "step_log_paths": {},
        "human_response": None,
        "tapeout_done": False,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Full E2E: Synthesis -> Backend -> Tapeout
# ═══════════════════════════════════════════════════════════════════════════

@requires_nix
@pytest.mark.slow
@pytest.mark.e2e
class TestTapeoutE2E:
    """Full tapeout pipeline: synth -> backend -> tapeout on adder_16bit."""

    @pytest.fixture(scope="class")
    def synth_project(self, tmp_path_factory):
        """Synthesize adder_16bit once for all tests."""
        project_root = tmp_path_factory.mktemp("tapeout_e2e")
        netlist, sdc = _synthesize_adder(project_root)
        return {
            "project_root": project_root,
            "netlist": netlist,
            "sdc": sdc,
        }

    @pytest.mark.asyncio
    @patch(_DASHBOARD_PATCH, _MOCK_DASHBOARD)
    async def test_full_tapeout_chain(self, synth_project):
        """Run backend + tapeout and verify all stages complete.

        This is the primary acceptance test for the OpenFrame tapeout flow.
        """
        project_root = synth_project["project_root"]

        # ---- Phase 1: Backend (PnR -> DRC -> LVS -> timing) ----
        backend_graph = build_backend_graph(checkpointer=MemorySaver())
        backend_config = {"configurable": {"thread_id": "tapeout-e2e-backend"}}
        backend_state = _make_backend_state(str(project_root))

        backend_result = await backend_graph.ainvoke(backend_state, backend_config)

        assert backend_result["backend_done"] is True, (
            f"Backend failed: {backend_result.get('previous_error', '')[:500]}"
        )

        completed = backend_result["completed_blocks"]
        assert len(completed) == 1
        assert completed[0]["success"] is True, (
            f"Block failed: {completed[0].get('error', '')}"
        )

        # ---- Phase 2: Tapeout (wrapper -> PnR -> DRC -> precheck) ----
        tapeout_gph = build_tapeout_graph(checkpointer=MemorySaver())
        tapeout_config = {"configurable": {"thread_id": "tapeout-e2e-tapeout"}}

        block_with_ports = {
            "name": "adder_16bit",
            "tier": 1,
            "ports": {
                "clk": {"width": 1, "direction": "input"},
                "rst": {"width": 1, "direction": "input"},
                "a": {"width": 16, "direction": "input"},
                "b": {"width": 16, "direction": "input"},
                "sum": {"width": 16, "direction": "output"},
                "cout": {"width": 1, "direction": "output"},
            },
        }

        tapeout_state = _make_tapeout_state(
            str(project_root),
            [block_with_ports],
            completed,
        )

        tapeout_result = await tapeout_gph.ainvoke(
            tapeout_state, tapeout_config,
        )

        # -- Tapeout graph completed --
        assert tapeout_result["tapeout_done"] is True, (
            f"Tapeout did not complete. Phase: {tapeout_result.get('phase')}, "
            f"error: {tapeout_result.get('previous_error', '')[:500]}"
        )

        # -- Submission directory created --
        sub_dir = tapeout_result.get("submission_dir", "")
        assert sub_dir, "No submission directory"
        assert Path(sub_dir).is_dir(), f"Submission dir missing: {sub_dir}"

        # -- GDS copied to submission --
        gds_files = list((Path(sub_dir) / "gds").glob("*.gds"))
        assert len(gds_files) >= 1, "No GDS files in submission"

        # -- Precheck ran --
        precheck = tapeout_result.get("precheck_result") or {}
        assert "checks" in precheck, "Precheck did not run"

        # -- Structure check passed --
        structure = precheck.get("checks", {}).get("structure", {})
        assert structure.get("pass") is True, (
            f"Structure check failed: {structure.get('errors', [])}"
        )

        print(f"\n{'='*60}")
        print("  TAPEOUT E2E RESULTS")
        print(f"{'='*60}")
        print(f"  Backend: PASS ({completed[0].get('attempts', '?')} attempts)")
        print(f"  Wrapper DRC: "
              f"{'CLEAN' if (tapeout_result.get('wrapper_drc_result') or {}).get('clean') else 'N/A'}")
        print(f"  Precheck: {'PASS' if precheck.get('pass') else 'FAIL'}")
        for check_name, check_result in precheck.get("checks", {}).items():
            status = "PASS" if check_result.get("pass") else "FAIL"
            print(f"    {check_name}: {status}")
        print(f"  Submission: {sub_dir}")
        print(f"{'='*60}")
