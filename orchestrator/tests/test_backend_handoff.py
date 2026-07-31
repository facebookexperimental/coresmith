# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The frontend -> backend handoff, and the testbench the gate-sim grades.

The chip_top gate-sim is the only step that ever simulates the artifact that
becomes silicon. It lives in the backend graph, and until now the backend graph
was reachable only from an MCP client or a hand-written driver: a daemon run
could reach ``pipeline_done`` and stop, with nobody to press the next button.

These tests cover the two things that made the handoff untrustworthy:

  * the launch path -- does the state the PRODUCTION launcher builds carry what
    the gate-sim needs, and does ``stop_after_gate_sim`` actually stop the graph;
  * the testbench lookup -- the gate looked up ``test_<design_name>.py``, but the
    testbench is named after the FRONTEND design while ``design_name`` is the top
    MODULE (``user_project_wrapper`` on a Caravel chip). The exact-name lookup
    then found nothing and reported ``not_run``.

The lookup tests build a real directory and call the real function; the launcher
test captures the state the production ``launch_backend`` constructed rather than
constructing one itself. This file deliberately does not assert on values it
supplied to the code under test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.langgraph.backend_graph import (
    find_integration_tb,
    route_after_flat_synth,
)


# ---------------------------------------------------------------------------
# Integration-testbench lookup
# ---------------------------------------------------------------------------
def _tb_dir(root: Path) -> Path:
    d = root / "tb" / "integration"
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_exact_design_name_wins(tmp_path):
    d = _tb_dir(tmp_path)
    (d / "test_user_project_wrapper.py").write_text("# chip tb\n")
    (d / "test_raster_top.py").write_text("# frontend-named tb\n")
    tb, note = find_integration_tb(tmp_path, "user_project_wrapper")
    assert tb.endswith("test_user_project_wrapper.py")
    assert note == ""


def test_sim_build_copy_is_preferred_over_tb_dir(tmp_path):
    d = _tb_dir(tmp_path)
    (d / "test_chip.py").write_text("# tb dir\n")
    sb = tmp_path / "sim_build" / "integration"
    sb.mkdir(parents=True)
    (sb / "test_chip.py").write_text("# sim_build copy\n")
    tb, _ = find_integration_tb(tmp_path, "chip")
    assert "sim_build" in tb


def test_falls_back_to_the_only_testbench_present(tmp_path):
    """THE BUG: the TB is named after the frontend design, design_name is the
    top module. One candidate is not a guess -- it is the chip's testbench."""
    d = _tb_dir(tmp_path)
    (d / "test_raster2d_accelerator_top.py").write_text("# frontend-named\n")
    tb, note = find_integration_tb(tmp_path, "user_project_wrapper")
    assert tb.endswith("test_raster2d_accelerator_top.py")
    assert "no test_user_project_wrapper.py" in note
    assert "user_project_wrapper" in note


def test_refuses_to_guess_between_several(tmp_path):
    """Picking by sort order is how a gate grades the wrong stimulus and calls
    the result a pass."""
    d = _tb_dir(tmp_path)
    (d / "test_a_top.py").write_text("# a\n")
    (d / "test_b_top.py").write_text("# b\n")
    tb, note = find_integration_tb(tmp_path, "user_project_wrapper")
    assert tb == ""
    assert "AMBIGUOUS" in note
    assert "test_a_top.py" in note and "test_b_top.py" in note


def test_no_testbench_at_all_is_reported_not_silently_passed(tmp_path):
    tb, note = find_integration_tb(tmp_path, "chip_top")
    assert tb == ""
    assert "no integration-DV testbench found" in note


# ---------------------------------------------------------------------------
# stop_after_gate_sim routing
# ---------------------------------------------------------------------------
def _synth_state(tmp_path, **over):
    netlist = tmp_path / "netlist.v"
    netlist.write_text("module chip_top(); endmodule\n")
    state = {"flat_netlist_path": str(netlist), "chip_gate_sim_ok": True}
    state.update(over)
    return state


def test_default_routing_is_unchanged(tmp_path):
    assert route_after_flat_synth(_synth_state(tmp_path)) == "run_pnr"
    assert route_after_flat_synth(
        _synth_state(tmp_path, chip_gate_sim_ok=False)) == "diagnose"
    assert route_after_flat_synth(
        _synth_state(tmp_path, chip_gate_sim_ok=None)) == "run_pnr"
    assert route_after_flat_synth({"flat_netlist_path": ""}) == "diagnose"


@pytest.mark.parametrize("gate_ok", [True, False, None])
def test_stop_after_gate_sim_ends_the_graph_in_every_outcome(tmp_path, gate_ok):
    """PASS, FAIL and did-not-apply all end here. A FAIL must not pull a caller
    who asked for a verdict into hours of LLM-driven diagnose/retry EDA -- the
    verdict is in state either way."""
    from langgraph.graph import END
    state = _synth_state(tmp_path, chip_gate_sim_ok=gate_ok,
                         stop_after_gate_sim=True)
    assert route_after_flat_synth(state) == END


def test_stop_after_gate_sim_ends_even_without_a_netlist(tmp_path):
    from langgraph.graph import END
    assert route_after_flat_synth(
        {"flat_netlist_path": "", "stop_after_gate_sim": True}) == END


# ---------------------------------------------------------------------------
# The production launcher builds the state the gate-sim needs
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_launch_backend_state_carries_what_init_design_needs(
    tmp_path, monkeypatch
):
    """Captures the state the PRODUCTION launcher constructed.

    This is deliberately not a test that hands ``integration_top_path`` to the
    gate and then asserts the gate saw it -- that shape is exactly how the
    chip_top gate-sim shipped reporting ``not_run`` on every real run while its
    unit test passed. What must be true is that the launcher supplies the inputs
    ``init_design_node`` needs to DISCOVER the integration top and block RTL,
    and that ``stop_after_gate_sim`` reaches the graph.
    """
    monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))
    # one block, with the RTL + synthesis artifacts the launcher gates on
    (tmp_path / ".coresmith").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".coresmith" / "block_specs.json").write_text(
        '[{"name": "blk", "rtl_target": "rtl/blk/blk.v"}]')
    (tmp_path / "rtl" / "blk").mkdir(parents=True)
    (tmp_path / "rtl" / "blk" / "blk.v").write_text("module blk(); endmodule\n")
    (tmp_path / "syn" / "output" / "blk").mkdir(parents=True)
    (tmp_path / "syn" / "output" / "blk" / "blk_netlist.v").write_text("//\n")

    from orchestrator import mcp_server as mcp

    captured: dict = {}

    async def _capture(initial_state, config):
        captured["state"] = initial_state
        captured["config"] = config

    monkeypatch.setattr(mcp._backend, "run_task", _capture)
    monkeypatch.setattr(mcp._backend, "status", "idle")
    monkeypatch.setattr(
        mcp, "_project_root", lambda: str(tmp_path), raising=False)

    async def _ok(_names):
        return {"ok": True, "errors": [], "warnings": []}

    import orchestrator.langgraph.pipeline_helpers as ph
    monkeypatch.setattr(ph, "preflight_check", lambda names: {
        "ok": True, "errors": [], "warnings": []})

    res = await mcp.launch_backend(stop_after_gate_sim=True)
    assert not res.get("error"), res
    st = captured["state"]
    # what init_design_node consumes to discover the gate-sim's reference RTL
    assert st["project_root"] == str(tmp_path)
    assert st["frontend_blocks"], "no blocks -> discover_block_rtl finds nothing"
    assert "block_rtl_paths" in st and "integration_top_path" in st
    # ...and the stop flag actually reaches the graph
    assert st["stop_after_gate_sim"] is True


@pytest.mark.asyncio
async def test_launch_backend_defaults_to_the_full_physical_flow(
    tmp_path, monkeypatch
):
    """Every pre-existing caller (the MCP tool) keeps P&R/DRC/LVS."""
    from orchestrator import mcp_server as mcp
    import inspect
    sig = inspect.signature(mcp.launch_backend)
    assert sig.parameters["stop_after_gate_sim"].default is False
    sig_tool = inspect.signature(mcp.start_backend)
    assert sig_tool.parameters["stop_after_gate_sim"].default is False


# ---------------------------------------------------------------------------
# Daemon wiring
# ---------------------------------------------------------------------------
def test_daemon_exposes_the_backend_endpoints():
    from orchestrator.daemon import server as ds

    paths = {r.path for r in ds.app.routes if hasattr(r, "path")}
    assert {"/backend/start", "/backend/state", "/backend/pause"} <= paths


def test_auto_backend_is_opt_in(monkeypatch):
    from orchestrator.daemon import server as ds

    monkeypatch.delenv("CORESMITH_AUTO_BACKEND", raising=False)
    assert ds._auto_backend_enabled() is False
    for off in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("CORESMITH_AUTO_BACKEND", off)
        assert ds._auto_backend_enabled() is False, off
    for on in ("1", "true", "yes", "on"):
        monkeypatch.setenv("CORESMITH_AUTO_BACKEND", on)
        assert ds._auto_backend_enabled() is True, on


def test_backend_start_defaults_to_stopping_at_the_gate_sim_verdict():
    """P&R/DRC/LVS is hours of EDA; the handoff must not spend it by default."""
    from orchestrator.daemon import server as ds

    assert ds.BackendStartRequest().full is False
