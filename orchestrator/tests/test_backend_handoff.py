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


def test_internal_force_selects_pin_driven_validation_tb(tmp_path):
    d = _tb_dir(tmp_path)
    (d / "test_chip_top.py").write_text(
        "from cocotb.handle import Force\ndut.internal.value = Force(0)\n")
    vd = tmp_path / "tb" / "validation"
    vd.mkdir(parents=True)
    expected = vd / "test_chip_top_validation.py"
    expected.write_text("# pin-only validation stimulus\n")

    tb, note = find_integration_tb(tmp_path, "chip_top")

    assert tb == str(expected)
    assert "Force()" in note and "pin-driven" in note


def test_explicit_gate_sim_tb_is_pin_only_and_fail_closed(tmp_path, monkeypatch):
    safe = tmp_path / "tb" / "validation" / "safe.py"
    safe.parent.mkdir(parents=True)
    safe.write_text("# pin-only\n")
    monkeypatch.setenv("CORESMITH_GATE_SIM_TB", "tb/validation/safe.py")
    assert find_integration_tb(tmp_path, "chip_top")[0] == str(safe)

    safe.write_text("import cocotb\ndut.hidden.value = cocotb.handle.Force(0)\n")
    tb, note = find_integration_tb(tmp_path, "chip_top")
    assert tb == ""
    assert "cannot reproduce internal forcing" in note


# ---------------------------------------------------------------------------
# stop_after_gate_sim routing
# ---------------------------------------------------------------------------
def _synth_state(tmp_path, **over):
    netlist = tmp_path / "netlist.v"
    netlist.write_text("module chip_top(); endmodule\n")
    state = {"flat_netlist_path": str(netlist), "chip_gate_sim_ok": True}
    state.update(over)
    return state


def test_full_flow_requires_a_gate_verdict(tmp_path):
    assert route_after_flat_synth(_synth_state(tmp_path)) == "run_pnr"
    assert route_after_flat_synth(
        _synth_state(tmp_path, chip_gate_sim_ok=False)) == "diagnose"
    assert route_after_flat_synth(
        _synth_state(tmp_path, chip_gate_sim_ok=None)) == "diagnose"
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
    import inspect

    from orchestrator import mcp_server as mcp
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


@pytest.mark.asyncio
async def test_backend_http_pause_delegates_to_authoritative_reaping_path(
    monkeypatch,
):
    import json

    from orchestrator.daemon import server as ds

    calls = []

    class _Task:
        @staticmethod
        def done():
            return False

    class _Handle:
        task = _Task()

    class _Mcp:
        _backend = _Handle()

        @staticmethod
        async def pause_backend():
            calls.append("pause_backend")
            return json.dumps({"status": "paused", "thread_id": "backend"})

    monkeypatch.setattr(ds, "_backend_handle", lambda: _Mcp)

    result = await ds.backend_pause()

    assert calls == ["pause_backend"]
    assert result == {
        "paused": True,
        "status": "paused",
        "thread_id": "backend",
    }


@pytest.mark.asyncio
async def test_backend_http_pause_preserves_idle_response(monkeypatch):
    from orchestrator.daemon import server as ds

    class _Task:
        @staticmethod
        def done():
            return True

    class _Mcp:
        class _Handle:
            task = _Task()

        _backend = _Handle()

        @staticmethod
        async def pause_backend():
            raise AssertionError("idle HTTP pause must not invoke MCP pause")

    monkeypatch.setattr(ds, "_backend_handle", lambda: _Mcp)

    assert await ds.backend_pause() == {
        "paused": False,
        "reason": "no running task",
    }


@pytest.mark.asyncio
async def test_authoritative_backend_pause_fences_and_cancels_the_owned_process_scope(
    monkeypatch,
):
    """The authoritative pause owns the backend's process tree through
    ``orchestrator.processes``: the job owner is FENCED before the graph task
    is cancelled (no new child can start under it), every child of that owner
    is cancelled, and the task is joined before the status reads ``paused``.
    (The old test patched a removed per-CLI kill hook and never set an owner,
    so it exercised nothing the lifecycle does today.)"""
    import asyncio
    import json

    from orchestrator import mcp_server as mcp
    from orchestrator import processes

    order = []
    owner = "backend:test-owner"

    async def running_worker():
        try:
            await asyncio.Event().wait()
        finally:
            order.append("task_cancelled")

    task = asyncio.create_task(running_worker())
    await asyncio.sleep(0)

    class _Snapshot:
        values = {}

    class _Graph:
        @staticmethod
        async def aget_state(_config):
            return _Snapshot()

    async def ensure_graph():
        return None

    monkeypatch.setattr(mcp._backend, "status", "running")
    monkeypatch.setattr(mcp._backend, "task", task)
    monkeypatch.setattr(mcp._backend, "graph", _Graph())
    monkeypatch.setattr(mcp._backend, "ensure_graph", ensure_graph)
    monkeypatch.setattr(mcp._backend, "_job_owner", owner)
    monkeypatch.setattr(processes, "fence", lambda o: order.append(("fenced", o, task.cancelled())))
    monkeypatch.setattr(processes, "cancel", lambda o=None, grace_s=1.0: order.append(("reaped", o)) or 0)

    result = json.loads(await mcp.pause_backend())

    assert result["status"] == "paused"
    assert mcp._backend.status == "paused"
    assert task.cancelled() and "task_cancelled" in order           # the task was joined, not abandoned
    assert order[0] == ("fenced", owner, False)                     # fenced BEFORE the graph task is cancelled
    assert ("reaped", owner) in order                                # every child of the owner is cancelled


@pytest.mark.asyncio
async def test_pause_without_an_owner_touches_no_process_scope(tmp_path, monkeypatch):
    """A lifecycle that never began a job owns no processes: pause cancels the
    task and must not fence or cancel some other owner's children."""
    import asyncio

    from orchestrator import processes
    from orchestrator.graph_lifecycle import GraphLifecycle

    async def running_worker():
        await asyncio.Event().wait()

    lc = GraphLifecycle("probe", str(tmp_path / "probe.db"), "orchestrator.langgraph.pipeline_graph",
                        "build_pipeline_graph", str(tmp_path))
    lc.task = asyncio.create_task(running_worker())
    await asyncio.sleep(0)
    monkeypatch.setattr(lc, "_job_owner", "")
    monkeypatch.setattr(processes, "fence", lambda o: pytest.fail("fenced without an owner"))
    monkeypatch.setattr(processes, "cancel", lambda *a, **k: pytest.fail("cancelled without an owner"))
    assert await lc.safe_pause() is True
    assert lc.task.cancelled() and lc.status == "paused"


@pytest.mark.asyncio
async def test_paused_backend_retry_persists_constraint_before_plain_tick(
    monkeypatch,
):
    import json

    from orchestrator import mcp_server as mcp

    calls = []

    class _Snapshot:
        values = {"constraints": [{"rule": "keep prior"}]}
        tasks = []

    class _Graph:
        @staticmethod
        async def aget_state(_config):
            return _Snapshot()

        @staticmethod
        async def aupdate_state(_config, update):
            calls.append(("update", update))

    async def ensure_graph():
        return None

    async def safe_resume(value, _config):
        calls.append(("resume", value))

    monkeypatch.setattr(mcp._backend, "status", "paused")
    monkeypatch.setattr(mcp._backend, "graph", _Graph())
    monkeypatch.setattr(mcp._backend, "ensure_graph", ensure_graph)
    monkeypatch.setattr(mcp._backend, "safe_resume", safe_resume)

    result = json.loads(await mcp.resume_backend(
        action="retry", constraint="  use the authoritative new template  ",
    ))

    assert result["status"] == "running"
    assert calls == [
        ("update", {"constraints": [
            {"rule": "keep prior"},
            {
                "rule": "use the authoritative new template",
                "source": "paused_backend_resume",
            },
        ]}),
        ("resume", None),
    ]


@pytest.mark.asyncio
async def test_paused_backend_nonretry_does_not_persist_constraint(monkeypatch):
    import json

    from orchestrator import mcp_server as mcp

    calls = []

    class _Snapshot:
        values = {"constraints": []}
        tasks = []

    class _Graph:
        @staticmethod
        async def aget_state(_config):
            return _Snapshot()

        @staticmethod
        async def aupdate_state(*_args, **_kwargs):
            raise AssertionError("abort must not inject a retry constraint")

    async def ensure_graph():
        return None

    async def safe_resume(value, _config):
        calls.append(value)

    monkeypatch.setattr(mcp._backend, "status", "paused")
    monkeypatch.setattr(mcp._backend, "graph", _Graph())
    monkeypatch.setattr(mcp._backend, "ensure_graph", ensure_graph)
    monkeypatch.setattr(mcp._backend, "safe_resume", safe_resume)

    result = json.loads(await mcp.resume_backend(
        action="abort", constraint="must not become retry guidance",
    ))

    assert result["status"] == "running"
    assert calls == [None]


@pytest.mark.asyncio
async def test_frontend_is_done_is_conservative(monkeypatch):
    """All four conditions must hold. A parked interrupt is NOT 'done' even with
    pipeline_done set -- the run is waiting on a decision that could still change
    the RTL the backend would synthesize."""
    from orchestrator.daemon import server as ds

    class _Snap:
        def __init__(self, values, tasks=()):
            self.values = values
            self.tasks = tasks

    class _Graph:
        def __init__(self, snap):
            self._snap = snap

        async def aget_state(self, _cfg):
            return self._snap

    class _Intr:
        id = "i1"
        value = {}

    class _Task:
        interrupts = [_Intr()]

    async def _noop():
        return None

    monkeypatch.setattr(ds._pipeline, "ensure_graph", _noop)
    monkeypatch.setattr(ds._pipeline, "task", None)

    monkeypatch.setattr(ds._pipeline, "graph", _Graph(_Snap({"pipeline_done": True})))
    assert await ds._frontend_is_done() is True

    monkeypatch.setattr(ds._pipeline, "graph", _Graph(_Snap({"pipeline_done": False})))
    assert await ds._frontend_is_done() is False

    monkeypatch.setattr(
        ds._pipeline, "graph",
        _Graph(_Snap({"pipeline_done": True}, tasks=[_Task()])))
    assert await ds._frontend_is_done() is False, "parked interrupt is not done"

    monkeypatch.setattr(ds._pipeline, "graph", _Graph(_Snap({"pipeline_done": True})))

    class _Running:
        @staticmethod
        def done():
            return False

    monkeypatch.setattr(ds._pipeline, "task", _Running())
    assert await ds._frontend_is_done() is False, "still running is not done"


def test_auto_backend_announcements_reach_daemon_log(monkeypatch):
    """The `coresmithd` logger has NO handler under uvicorn -- the same trap the
    profile-seed line hit. An autonomy step that starts real EDA by itself must
    not announce into a black hole."""
    import logging

    from orchestrator.daemon import server as ds

    seen: list[str] = []

    class _Sink(logging.Handler):
        def emit(self, record):
            seen.append(record.getMessage())

    uv = logging.getLogger("uvicorn.error")
    h = _Sink()
    uv.addHandler(h)
    try:
        ds._daemon_log("warning", "AUTO-BACKEND: %s", "hello")
    finally:
        uv.removeHandler(h)
    assert seen == ["AUTO-BACKEND: hello"]
