# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The QSPI PIN-BOUNDARY gate: integration DV must drive the GRADED boundary.

The fail-open this pins closed (observed on ``exp-raster-macro-20260727``): the
assembled integration top declared ``qspi_csn/qspi_sck/qspi_io_in[3:0]/
qspi_io_out[3:0]/qspi_drive_en/irq_level`` -- a design-prefixed pin group, not
the locked Caravel ``io_in``/``io_out``/``io_oeb`` the external grader drives.
The classifier concluded "not a QSPI-slave bus", the caller logged a RED
advisory ("THIS RUN'S INTEGRATION DV IS NOT CONTRACT-ENFORCING") and PROCEEDED
on the LLM-authored, DUT-co-tuned BFM. The run then reported "INTEGRATION DV
PASSED" -- while ``rtl/user_project_wrapper.v`` (which DID declare all three pad
busses and mapped ``qspi_io_out`` -> ``io_out[5:2]``) was never driven.

Three outcomes are pinned here:

* the graded boundary is on the module the sim elaborates -> contract-enforcing;
* the boundary exists but on another module (the wrapper) / nowhere at all while
  the spec says QSPI -> FAIL CLOSED through the run's normal tb_generation
  interrupt, and the co-tuned LLM generator is never reached;
* a genuinely non-QSPI design -> still classified not-QSPI, no error, LLM BFM
  kept with the historical advisory.

Hermetic: no EDA, no LLM (the testbench generator is monkeypatched and asserted
NOT to run on the fail-closed path).
"""

from __future__ import annotations

import json

import pytest

from orchestrator.langgraph import bfm_lib, pipeline_graph
from orchestrator.langgraph.bfm_lib import (
    STATUS_BOUNDARY_OFF_TOP,
    STATUS_CONTRADICTION,
    STATUS_NOT_QSPI,
    STATUS_QSPI_TOP,
    boundary_gate_enabled,
    classify_bus_verdict,
    classify_chip_bus,
    find_pin_boundary,
)
from orchestrator.langgraph.bfm_lib.classifier import (
    _top_has_gpio_pin_boundary,
    declared_pad_ports,
)
from orchestrator.langgraph.integration_helpers import VerilogModule, VerilogPort

# ---------------------------------------------------------------------------
# RTL fixtures (shapes taken from the real raster run)
# ---------------------------------------------------------------------------

# The assembled integration top of exp-raster-macro-20260727: a QSPI-slave
# chassis whose OWN pins are design-prefixed. `qspi_io_in` is deliberately NOT
# `io_in` -- it is a different pin on a different module.
_PREFIXED_PIN_TOP = """
module raster_top (
    input  wire       wb_clk_i,
    input  wire       wb_rst_i,
    input  wire       qspi_csn,
    input  wire       qspi_sck,
    input  wire [3:0] qspi_io_in,
    output wire [3:0] qspi_io_out,
    output wire       qspi_drive_en,
    output wire       irq_level
);
    wire [3:0] w_user_project_wrapper_io_qspi_io_in_to_frontend_qspi_io_in;
    assign w_user_project_wrapper_io_qspi_io_in_to_frontend_qspi_io_in = qspi_io_in;
endmodule
"""

# The graded boundary as the raster run really wrote it: a doc comment naming
# the pads (which must NOT count as evidence) plus real port declarations.
_CARAVEL_WRAPPER = """
/*
 * Block: user_project_wrapper
 * I/O ports:
 *   io_in, io_out, io_oeb    - 38-bit Caravel digital GPIO interface
 */
(* top = 1 *)
module user_project_wrapper (
    input  wire         wb_clk_i,
    input  wire         wb_rst_i,
    input  wire [127:0] la_data_in,
    input  wire [37:0]  io_in,
    output wire [37:0]  io_out,
    output wire [37:0]  io_oeb
);
    wire [3:0] qspi_io_out;
    assign io_out = {{31{1'b0}}, qspi_io_out, 2'b00};
endmodule
"""

# Same file, but the pads exist ONLY in prose -- not evidence of a boundary.
_COMMENT_ONLY_WRAPPER = """
/*
 * Maps io_in, io_out and io_oeb onto the QSPI pads.
 */
module pad_notes (
    input  wire       wb_clk_i,
    input  wire [3:0] qspi_io_in,
    output wire [3:0] qspi_io_out
);
    // io_oeb is driven by the parent
    wire [37:0] io_oeb;
endmodule
"""

_GPIO_TOP = """
module chip_top (
    input  wire        wb_clk_i,
    input  wire        wb_rst_i,
    input  wire [37:0] io_in,
    output wire [37:0] io_out,
    output wire [37:0] io_oeb
);
endmodule
"""

_NON_QSPI_TOP = """
module chip_top (
    input  wire        clk,
    input  wire        rst,
    input  wire [31:0] s_axis_tdata,
    input  wire        s_axis_tvalid,
    output wire        s_axis_tready
);
endmodule
"""


def _make_run(tmp_path, *, qspi: bool = True, wrapper: str = "",
              top_src: str = _PREFIXED_PIN_TOP, top_name: str = "raster_top"):
    """A minimal run dir: spec + assembled top (+ optional wrapper file)."""
    root = tmp_path / "run"
    (root / ".coresmith").mkdir(parents=True, exist_ok=True)
    # The raster run's real shape: prd bus_protocol="custom", ERS naming QSPI.
    proto = (
        "asynchronous dedicated-pin quad QSPI mode 0 externally"
        if qspi
        else "AXI-Stream"
    )
    (root / ".coresmith" / "prd_spec.json").write_text(
        json.dumps({"prd": {"dataflow": {"bus_protocol": "custom"}}})
    )
    (root / ".coresmith" / "ers_spec.json").write_text(
        json.dumps({"ers": {"dataflow": {"bus_protocol": proto}}})
    )
    top = root / "rtl" / "integration" / f"{top_name}.v"
    top.parent.mkdir(parents=True, exist_ok=True)
    top.write_text(top_src)
    if wrapper:
        (root / "rtl").mkdir(parents=True, exist_ok=True)
        (root / "rtl" / "user_project_wrapper.v").write_text(wrapper)
    return root, top


# ---------------------------------------------------------------------------
# evidence: what counts as the graded boundary
# ---------------------------------------------------------------------------

def test_prefixed_pin_group_is_not_the_graded_boundary():
    """`qspi_io_in` must NOT be read as `io_in` (loosening = drive wrong pins)."""
    assert _top_has_gpio_pin_boundary(_PREFIXED_PIN_TOP) is False
    assert declared_pad_ports(_PREFIXED_PIN_TOP) == {}


def test_real_port_declarations_are_evidence():
    pads = declared_pad_ports(_CARAVEL_WRAPPER)
    assert pads == {"io_in": "input", "io_out": "output", "io_oeb": "output"}
    assert _top_has_gpio_pin_boundary(_CARAVEL_WRAPPER) is True
    assert _top_has_gpio_pin_boundary(_GPIO_TOP) is True


def test_prose_and_internal_nets_are_not_evidence():
    """A doc comment naming the pads + an internal `wire io_oeb` is not a boundary."""
    assert _top_has_gpio_pin_boundary(_COMMENT_ONLY_WRAPPER) is False


# ---------------------------------------------------------------------------
# case 1: the wrapper IS found when the assembled top is not the boundary
# ---------------------------------------------------------------------------

def test_wrapper_is_found_when_top_is_not_the_boundary(tmp_path):
    run, top = _make_run(tmp_path, qspi=True, wrapper=_CARAVEL_WRAPPER)
    b = find_pin_boundary(
        str(run), _PREFIXED_PIN_TOP,
        top_module="raster_top", top_rtl_path=str(top),
    )
    assert b is not None, "the graded wrapper must be found, not ignored"
    assert b.module == "user_project_wrapper"
    assert b.path.endswith("rtl/user_project_wrapper.v")
    assert b.ports == ("io_in", "io_oeb", "io_out")
    assert b.in_top_source is False
    # ... but it is NOT the module the integration sim elaborates
    assert b.is_simulated_top is False

    v = classify_bus_verdict(
        str(run), _PREFIXED_PIN_TOP, None, "raster_top", str(top)
    )
    assert v.status == STATUS_BOUNDARY_OFF_TOP
    assert v.contract is not None, "this IS a QSPI-slave chassis"
    assert v.contract_enforcing is False
    assert v.fails_closed is True
    assert v.simulated_top == "raster_top"
    assert "user_project_wrapper.v" in v.reason
    assert "qspi_io_in" in v.reason        # names the pins DV would have driven
    assert "CORESMITH_QSPI_BOUNDARY_GATE=0" in v.reason
    # the same discovery through the contract-or-None API
    assert classify_chip_bus(
        str(run), _PREFIXED_PIN_TOP, None, "raster_top", str(top)
    ) is not None


def test_wrapper_that_only_mentions_pads_in_prose_is_not_counted(tmp_path):
    """Evidence-based, not guessing: a wrapper must really declare the pads."""
    run, top = _make_run(tmp_path, qspi=True, wrapper=_COMMENT_ONLY_WRAPPER)
    assert find_pin_boundary(
        str(run), _PREFIXED_PIN_TOP,
        top_module="raster_top", top_rtl_path=str(top),
    ) is None
    v = classify_bus_verdict(
        str(run), _PREFIXED_PIN_TOP, None, "raster_top", str(top)
    )
    assert v.status == STATUS_CONTRADICTION


def test_boundary_on_the_simulated_top_is_contract_enforcing(tmp_path):
    run, top = _make_run(
        tmp_path, qspi=True, top_src=_GPIO_TOP, top_name="chip_top"
    )
    v = classify_bus_verdict(str(run), _GPIO_TOP, None, "chip_top", str(top))
    assert v.status == STATUS_QSPI_TOP
    assert v.contract_enforcing is True
    assert v.fails_closed is False
    assert v.boundary is not None and v.boundary.is_simulated_top is True


def test_wrapper_found_and_it_is_the_simulated_top(tmp_path):
    """When the graded wrapper IS the integration top, DV can enforce the contract."""
    run, top = _make_run(
        tmp_path, qspi=True, top_src=_CARAVEL_WRAPPER,
        top_name="user_project_wrapper",
    )
    v = classify_bus_verdict(
        str(run), _CARAVEL_WRAPPER, None, "user_project_wrapper", str(top)
    )
    assert v.status == STATUS_QSPI_TOP
    assert v.contract_enforcing is True


def test_plan_deterministic_dv_gives_no_plan_for_an_off_top_boundary(tmp_path):
    """The deterministic host-flow must not be built against the wrong toplevel."""
    run, top = _make_run(tmp_path, qspi=True, wrapper=_CARAVEL_WRAPPER)
    contract, plan = bfm_lib.plan_deterministic_dv(
        str(run), _PREFIXED_PIN_TOP, None, "raster_top", str(top)
    )
    assert (contract, plan) == (None, None)


# ---------------------------------------------------------------------------
# case 2: a genuinely non-QSPI design still classifies as not-QSPI
# ---------------------------------------------------------------------------

def test_genuine_non_qspi_design_is_not_qspi_and_does_not_error(tmp_path):
    run, top = _make_run(
        tmp_path, qspi=False, top_src=_NON_QSPI_TOP, top_name="chip_top"
    )
    v = classify_bus_verdict(str(run), _NON_QSPI_TOP, None, "chip_top", str(top))
    assert v.status == STATUS_NOT_QSPI
    assert v.contract is None
    assert v.fails_closed is False, "an AXI-Stream design must not fail the run"
    assert classify_chip_bus(str(run), _NON_QSPI_TOP, None, "chip_top", str(top)) is None


def test_pad_boundary_without_qspi_corroboration_is_not_qspi(tmp_path):
    """Pads present but nothing claims QSPI -> unchanged: not classified."""
    run, top = _make_run(
        tmp_path, qspi=False, top_src=_GPIO_TOP, top_name="chip_top"
    )
    v = classify_bus_verdict(str(run), _GPIO_TOP, None, "chip_top", str(top))
    assert v.status == STATUS_NOT_QSPI
    assert v.contract is None
    assert v.fails_closed is False


# ---------------------------------------------------------------------------
# case 3: spec says QSPI, no boundary anywhere -> contradiction, fail closed
# ---------------------------------------------------------------------------

def test_spec_says_qspi_but_no_boundary_anywhere_is_a_contradiction(tmp_path):
    run, top = _make_run(tmp_path, qspi=True)          # no wrapper file at all
    v = classify_bus_verdict(
        str(run), _PREFIXED_PIN_TOP, None, "raster_top", str(top)
    )
    assert v.status == STATUS_CONTRADICTION
    assert v.contract is None
    assert v.fails_closed is True
    assert v.spec_says_qspi is True
    assert "CONTRADICTION" in v.reason
    assert "raster_top" in v.reason


def test_connections_alone_can_corroborate_qspi(tmp_path):
    run, top = _make_run(tmp_path, qspi=False)
    conns = [{"interface": "qspi_pin_sample", "from_block": "wrapper"}]
    v = classify_bus_verdict(
        str(run), _PREFIXED_PIN_TOP, conns, "raster_top", str(top)
    )
    assert v.status == STATUS_CONTRADICTION
    assert v.connections_say_qspi is True
    assert v.fails_closed is True


# ---------------------------------------------------------------------------
# the gate flag: both branches
# ---------------------------------------------------------------------------

def test_boundary_gate_default_on(monkeypatch):
    monkeypatch.delenv("CORESMITH_QSPI_BOUNDARY_GATE", raising=False)
    monkeypatch.delenv("CORESMITH_GATE_FAIL_OPEN", raising=False)
    assert boundary_gate_enabled() is True


def test_boundary_gate_env_off(monkeypatch):
    monkeypatch.delenv("CORESMITH_GATE_FAIL_OPEN", raising=False)
    monkeypatch.setenv("CORESMITH_QSPI_BOUNDARY_GATE", "0")
    assert boundary_gate_enabled() is False


def test_boundary_gate_honors_global_fail_open(monkeypatch):
    monkeypatch.delenv("CORESMITH_QSPI_BOUNDARY_GATE", raising=False)
    monkeypatch.setenv("CORESMITH_GATE_FAIL_OPEN", "1")
    assert boundary_gate_enabled() is False


# ---------------------------------------------------------------------------
# integration_dv_node wiring: the co-tuned BFM must be unreachable
# ---------------------------------------------------------------------------

def _setup_node(monkeypatch, tmp_path, *, wrapper: str = "", qspi: bool = True):
    """Wire integration_dv_node for a FRESH (LLM) testbench-generation run."""
    run, top = _make_run(tmp_path, qspi=qspi, wrapper=wrapper)
    blk = run / "rtl" / "a.v"
    blk.write_text("module a(input clk); endmodule\n")

    monkeypatch.setenv("CORESMITH_QSPI_CONFORMANCE", "1")
    monkeypatch.setattr(
        pipeline_graph, "load_architecture_connections", lambda pr: ([], "chip"))
    monkeypatch.setattr(
        pipeline_graph, "parse_verilog_ports",
        lambda p, module=None: VerilogModule(name="a", ports=[VerilogPort("clk", "input")]))
    monkeypatch.setattr(
        pipeline_graph, "run_integration_simulation",
        lambda *a, **k: {"passed": True, "log": "sim ran", "log_path": ""})
    monkeypatch.setattr(pipeline_graph, "_maybe_run_chip_equiv", lambda *a, **k: None)

    async def _fake_audit(**kw):
        return {"category": "TESTBENCH", "outer_agent_summary": "x", "audit_path": ""}
    monkeypatch.setattr(pipeline_graph, "_run_top_level_contract_audit", _fake_audit)
    monkeypatch.setattr(pipeline_graph, "write_graph_event", lambda *a, **k: None)

    state = {
        "project_root": str(run),
        "integration_result": {
            "top_rtl_path": str(top),
            "design_name": "raster_top",
            "block_rtl_paths": {"a": str(blk)},
        },
    }
    return run, state


@pytest.mark.asyncio
async def test_off_top_boundary_fails_closed_and_never_reaches_the_llm_bfm(
    monkeypatch, tmp_path
):
    """The load-bearing test: keeping the co-tuned BFM must be impossible."""
    monkeypatch.delenv("CORESMITH_QSPI_BOUNDARY_GATE", raising=False)
    monkeypatch.delenv("CORESMITH_GATE_FAIL_OPEN", raising=False)
    run, state = _setup_node(monkeypatch, tmp_path, wrapper=_CARAVEL_WRAPPER)

    async def _must_not_run(**kw):
        raise AssertionError(
            "the LLM-authored (DUT-co-tuned) BFM must not be generated once the "
            "pin-boundary gate has failed closed"
        )
    monkeypatch.setattr(
        pipeline_graph, "generate_integration_testbench", _must_not_run)

    captured = {}

    def fake_interrupt(payload):
        captured.update(payload)
        return {"action": "abort"}
    monkeypatch.setattr(pipeline_graph, "interrupt", fake_interrupt)

    result = await pipeline_graph.integration_dv_node(state)

    assert captured.get("type") == "integration_dv_failure"
    assert captured.get("phase") == "tb_generation"
    assert "QSPI pin-boundary gate" in captured.get("sim_log", "")
    assert "user_project_wrapper.v" in captured.get("sim_log", "")
    dv = result["integration_dv_result"]
    assert dv["passed"] is False
    assert result["pipeline_done"] is False


@pytest.mark.asyncio
async def test_contradiction_fails_closed(monkeypatch, tmp_path):
    """Spec says QSPI, no pin boundary anywhere -> same fail-closed interrupt."""
    monkeypatch.delenv("CORESMITH_QSPI_BOUNDARY_GATE", raising=False)
    monkeypatch.delenv("CORESMITH_GATE_FAIL_OPEN", raising=False)
    run, state = _setup_node(monkeypatch, tmp_path, wrapper="")

    async def _must_not_run(**kw):
        raise AssertionError("LLM BFM must not be generated")
    monkeypatch.setattr(
        pipeline_graph, "generate_integration_testbench", _must_not_run)
    captured = {}
    monkeypatch.setattr(
        pipeline_graph, "interrupt",
        lambda p: (captured.update(p), {"action": "retry"})[1])

    await pipeline_graph.integration_dv_node(state)
    assert captured.get("type") == "integration_dv_failure"
    assert "CONTRADICTION" in captured.get("sim_log", "")


@pytest.mark.asyncio
async def test_gate_off_restores_the_llm_bfm_but_records_a_defect(
    monkeypatch, tmp_path
):
    """Flag-off branch: old behavior, and the bypass is NOT silent."""
    monkeypatch.setenv("CORESMITH_QSPI_BOUNDARY_GATE", "0")
    run, state = _setup_node(monkeypatch, tmp_path, wrapper=_CARAVEL_WRAPPER)
    tb = run / "tb" / "integration" / "test_raster_top.py"
    tb.parent.mkdir(parents=True, exist_ok=True)
    tb.write_text("# llm tb\n")

    async def _llm_tb(**kw):
        return {"testbench_path": str(tb), "tb_path": str(tb), "test_count": 3}
    monkeypatch.setattr(pipeline_graph, "generate_integration_testbench", _llm_tb)
    monkeypatch.setattr(
        pipeline_graph, "interrupt",
        lambda p: (_ for _ in ()).throw(AssertionError("no interrupt expected")))

    result = await pipeline_graph.integration_dv_node(state)
    assert result["integration_dv_result"]["passed"] is True
    defects = pipeline_graph.read_carried_forward_defects(str(run))
    assert any(d.get("gate") == "qspi_pin_boundary" for d in defects), defects
    assert any("io_oeb" in str(d.get("unmodeled", "")) for d in defects)


@pytest.mark.asyncio
async def test_non_qspi_design_still_uses_the_llm_bfm(monkeypatch, tmp_path):
    """A genuinely non-QSPI design is untouched by the gate."""
    monkeypatch.delenv("CORESMITH_QSPI_BOUNDARY_GATE", raising=False)
    run, state = _setup_node(monkeypatch, tmp_path, wrapper="", qspi=False)
    tb = run / "tb" / "integration" / "test_raster_top.py"
    tb.parent.mkdir(parents=True, exist_ok=True)
    tb.write_text("# llm tb\n")

    async def _llm_tb(**kw):
        return {"testbench_path": str(tb), "tb_path": str(tb), "test_count": 2}
    monkeypatch.setattr(pipeline_graph, "generate_integration_testbench", _llm_tb)
    monkeypatch.setattr(
        pipeline_graph, "interrupt",
        lambda p: (_ for _ in ()).throw(AssertionError("no interrupt expected")))

    result = await pipeline_graph.integration_dv_node(state)
    assert result["integration_dv_result"]["passed"] is True
    defects = pipeline_graph.read_carried_forward_defects(str(run))
    assert not any(d.get("gate") == "qspi_pin_boundary" for d in defects)
