# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Defect 4: deterministic Caravel ``user_project_wrapper`` top assembly.

The per-block generator emits ``user_project_wrapper`` as a bare pad adapter
(io_in/io_out/io_oeb <-> a few bundle edges), but the harness + grader require
``user_project_wrapper`` to BE the wired chip_top that instantiates and wires
every block. Before this fix the daemon never produced a gradeable wired top and
every chip-lead hand-assembled one. ``generate_caravel_wrapper_top`` now emits it
deterministically: locked Caravel port list, all blocks instantiated, block<->
block bundles wired, pad adapter routed to the top GPIO.

Hermetic -- no LLM. The Verilator-elaboration test skips when verilator is
absent; the structural tests always run.
"""
from __future__ import annotations

import re
import shutil

import pytest

from orchestrator.langgraph import pipeline_graph
from orchestrator.langgraph.integration_helpers import (
    VerilogModule,
    VerilogPort,
    detect_wrapper_block,
    generate_caravel_wrapper_top,
    lint_top_level,
    parse_verilog_ports,
)


# ---------------------------------------------------------------------------
# Env-var gate: both branches (CLAUDE.md convention)
# ---------------------------------------------------------------------------

def test_caravel_gate_default_on(monkeypatch):
    monkeypatch.delenv("CORESMITH_DETERMINISTIC_CARAVEL_TOP", raising=False)
    assert pipeline_graph._deterministic_caravel_top_enabled() is True


def test_caravel_gate_can_be_disabled(monkeypatch):
    monkeypatch.setenv("CORESMITH_DETERMINISTIC_CARAVEL_TOP", "0")
    assert pipeline_graph._deterministic_caravel_top_enabled() is False


# ---------------------------------------------------------------------------
# Synthetic 3-block Caravel design fixture
# ---------------------------------------------------------------------------

def _scalar(name, direction):
    return VerilogPort(name=name, direction=direction, width=1)


def _vec(name, direction, width):
    return VerilogPort(name=name, direction=direction, width=width,
                       msb=width - 1, lsb=0)


def _wrapper_src(tmp_path):
    """A pad-adapter whose MODULE name collides with the top name."""
    src = (
        "module user_project_wrapper (\n"
        "  input wire wb_clk_i, input wire wb_rst_i,\n"
        "  input wire [37:0] io_in, output wire [37:0] io_out,\n"
        "  output wire [37:0] io_oeb,\n"
        "  output wire [5:0] m_axis_pads_tdata,\n"
        "  output wire m_axis_pads_tvalid, input wire m_axis_pads_tready\n"
        ");\n"
        "  assign io_out = 38'b0; assign io_oeb = 38'b0;\n"
        "  assign m_axis_pads_tdata = io_in[5:0];\n"
        "  assign m_axis_pads_tvalid = 1'b1;\n"
        "endmodule\n"
    )
    p = tmp_path / "user_project_wrapper.v"
    p.write_text(src)
    return str(p)


def _src_block_src(tmp_path):
    src = (
        "module stream_src (\n"
        "  input wire wb_clk_i, input wire wb_rst_i,\n"
        "  input wire [5:0] s_axis_pads_tdata,\n"
        "  input wire s_axis_pads_tvalid, output wire s_axis_pads_tready,\n"
        "  output wire [7:0] m_axis_data_tdata,\n"
        "  output wire m_axis_data_tvalid, input wire m_axis_data_tready,\n"
        "  output wire m_axis_data_tlast\n"
        ");\n"
        "  assign s_axis_pads_tready = 1'b1;\n"
        "  assign m_axis_data_tdata = s_axis_pads_tdata + 8'd1;\n"
        "  assign m_axis_data_tvalid = s_axis_pads_tvalid;\n"
        "  assign m_axis_data_tlast = 1'b0;\n"
        "endmodule\n"
    )
    p = tmp_path / "stream_src.v"
    p.write_text(src)
    return str(p)


def _sink_block_src(tmp_path):
    src = (
        "module stream_sink (\n"
        "  input wire wb_clk_i, input wire wb_rst_i,\n"
        "  input wire [7:0] s_axis_data_tdata,\n"
        "  input wire s_axis_data_tvalid, output wire s_axis_data_tready,\n"
        "  input wire s_axis_data_tlast\n"
        ");\n"
        "  assign s_axis_data_tready = 1'b1;\n"
        "endmodule\n"
    )
    p = tmp_path / "stream_sink.v"
    p.write_text(src)
    return str(p)


def _build_design(tmp_path):
    rtl_paths = {
        "user_project_wrapper": _wrapper_src(tmp_path),
        "stream_src": _src_block_src(tmp_path),
        "stream_sink": _sink_block_src(tmp_path),
    }
    modules = {bn: parse_verilog_ports(p) for bn, p in rtl_paths.items()}
    edges = [
        {"producer_block": "user_project_wrapper", "consumer_block": "stream_src",
         "data_width": 6, "edge_id": "wrap_to_src"},
        {"producer_block": "stream_src", "consumer_block": "stream_sink",
         "data_width": 8, "edge_id": "src_to_sink"},
    ]
    return modules, edges, rtl_paths


# ---------------------------------------------------------------------------
# detect_wrapper_block
# ---------------------------------------------------------------------------

def test_detect_wrapper_by_name(tmp_path):
    modules, _e, _p = _build_design(tmp_path)
    assert detect_wrapper_block(modules) == "user_project_wrapper"


def test_detect_wrapper_by_io_pads(tmp_path):
    # Rename the block key so name-match fails; detection falls back to the
    # io_in/io_out/io_oeb pad vector.
    modules, _e, _p = _build_design(tmp_path)
    modules["pad_ring"] = modules.pop("user_project_wrapper")
    assert detect_wrapper_block(modules) == "pad_ring"


def test_detect_none_for_non_caravel(tmp_path):
    modules, _e, _p = _build_design(tmp_path)
    del modules["user_project_wrapper"]  # no io pads left
    assert detect_wrapper_block(modules) is None


# ---------------------------------------------------------------------------
# generate_caravel_wrapper_top -- structure
# ---------------------------------------------------------------------------

def test_top_is_named_user_project_wrapper(tmp_path):
    modules, edges, rtl_paths = _build_design(tmp_path)
    asm = generate_caravel_wrapper_top(
        modules, edges, rtl_paths, str(tmp_path / "out"))
    assert asm["module_name"] == "user_project_wrapper"
    assert re.search(r"module\s+user_project_wrapper\s*\(", asm["verilog"])


def test_all_blocks_instantiated(tmp_path):
    modules, edges, rtl_paths = _build_design(tmp_path)
    asm = generate_caravel_wrapper_top(
        modules, edges, rtl_paths, str(tmp_path / "out"))
    for bn in modules:
        assert f"u_{bn} (" in asm["verilog"], f"{bn} not instantiated"
    assert sorted(asm["instantiated"]) == sorted(modules)


def test_pad_adapter_renamed_to_dodge_collision(tmp_path):
    modules, edges, rtl_paths = _build_design(tmp_path)
    asm = generate_caravel_wrapper_top(
        modules, edges, rtl_paths, str(tmp_path / "out"))
    # The top is `module user_project_wrapper`; the pad block is instantiated
    # under the renamed module so there is exactly one top-name definition.
    assert asm["renamed_pad_path"].endswith("user_project_wrapper_pads.v")
    assert "user_project_wrapper_pads u_user_project_wrapper" in asm["verilog"]
    assert asm["verilog"].count("module user_project_wrapper (") == 1
    renamed = open(asm["renamed_pad_path"]).read()
    assert "module user_project_wrapper_pads" in renamed
    assert "module user_project_wrapper (" not in renamed


def test_clk_rst_route_to_top(tmp_path):
    modules, edges, rtl_paths = _build_design(tmp_path)
    asm = generate_caravel_wrapper_top(
        modules, edges, rtl_paths, str(tmp_path / "out"))
    assert ".wb_clk_i(wb_clk_i)" in asm["verilog"]
    assert ".wb_rst_i(wb_rst_i)" in asm["verilog"]
    # No clk/rst got bundled onto an internal wire.
    assert "w_" not in "".join(
        l for l in asm["verilog"].splitlines() if "wb_clk_i" in l and "wire w_" in l
    )


def test_wrapper_io_routes_to_top_gpio(tmp_path):
    modules, edges, rtl_paths = _build_design(tmp_path)
    asm = generate_caravel_wrapper_top(
        modules, edges, rtl_paths, str(tmp_path / "out"))
    for io in ("io_in", "io_out", "io_oeb"):
        assert f".{io}({io})" in asm["verilog"]


def test_core_bundle_is_wired(tmp_path):
    modules, edges, rtl_paths = _build_design(tmp_path)
    asm = generate_caravel_wrapper_top(
        modules, edges, rtl_paths, str(tmp_path / "out"))
    # The src->sink data bundle (tdata/tvalid/tready/tlast) must share wires:
    # the producer m_axis_data_tdata and consumer s_axis_data_tdata reference
    # the SAME internal wire.
    m_prod = re.search(r"\.m_axis_data_tdata\((\w+)\)", asm["verilog"])
    m_cons = re.search(r"\.s_axis_data_tdata\((\w+)\)", asm["verilog"])
    assert m_prod and m_cons
    assert m_prod.group(1) == m_cons.group(1)
    assert m_prod.group(1).startswith("w_")
    assert asm["wire_count"] >= 4  # tdata/tvalid/tready/tlast


def test_unused_caravel_ifaces_tied_off(tmp_path):
    modules, edges, rtl_paths = _build_design(tmp_path)
    asm = generate_caravel_wrapper_top(
        modules, edges, rtl_paths, str(tmp_path / "out"))
    for tie in ("assign wbs_ack_o = 1'b0;", "assign user_irq = 3'b0;",
                "assign la_data_out = 128'b0;"):
        assert tie in asm["verilog"]


def test_power_pins_guarded_by_ifdef(tmp_path):
    modules, edges, rtl_paths = _build_design(tmp_path)
    asm = generate_caravel_wrapper_top(
        modules, edges, rtl_paths, str(tmp_path / "out"))
    assert "`ifdef USE_POWER_PINS" in asm["verilog"]
    assert "inout  wire vccd1" in asm["verilog"]


# ---------------------------------------------------------------------------
# Verilator elaboration (skips when verilator is absent)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(shutil.which("verilator") is None,
                    reason="verilator not installed")
def test_assembled_top_elaborates(tmp_path):
    modules, edges, rtl_paths = _build_design(tmp_path)
    asm = generate_caravel_wrapper_top(
        modules, edges, rtl_paths, str(tmp_path / "out"))
    lint = lint_top_level(
        asm["rtl_path"], list(asm["lint_block_paths"].values()),
        "user_project_wrapper")
    assert lint.get("clean"), lint.get("errors", "")


# ---------------------------------------------------------------------------
# Section 2: normalized-key COLLISION must not short distinct nets (fft case)
#
# A block that exposes BOTH `rd_addr` (24b, out to a memory) and `in_rd_addr`
# (8b, in from a partner) normalizes both to the signal key "rd_addr". The old
# assembler [0]-picked one port per key and max-width-shorted the group ->
# MULTIDRIVEN + WIDTHTRUNC. The structured contract names the exact port each
# edge instantiates, so the two must resolve to TWO distinct, correctly-sized
# wires.
# ---------------------------------------------------------------------------

def _collision_modules():
    core = VerilogModule(name="fft_core", ports=[
        VerilogPort("wb_clk_i", "input", 1),
        VerilogPort("wb_rst_i", "input", 1),
        VerilogPort("rd_addr", "output", 24, msb=23, lsb=0),      # -> memory
        VerilogPort("in_rd_addr", "input", 8, msb=7, lsb=0),      # <- ctrl
    ])
    mem = VerilogModule(name="fft_mem", ports=[
        VerilogPort("wb_clk_i", "input", 1),
        VerilogPort("wb_rst_i", "input", 1),
        VerilogPort("rd_addr", "input", 24, msb=23, lsb=0),
    ])
    ctrl = VerilogModule(name="fft_ctrl", ports=[
        VerilogPort("wb_clk_i", "input", 1),
        VerilogPort("wb_rst_i", "input", 1),
        VerilogPort("addr_out", "output", 8, msb=7, lsb=0),       # -> core.in_rd_addr
    ])
    return {"fft_core": core, "fft_mem": mem, "fft_ctrl": ctrl}


def _conn_wire(verilog, inst, port):
    """Return the net a given .port(...) connects to inside `u_<inst> (`."""
    m = re.search(rf"u_{inst} \((.*?)\n  \);", verilog, re.DOTALL)
    assert m, f"instance u_{inst} not found"
    body = m.group(1)
    pm = re.search(rf"\.{port}\(([^)]*)\)", body)
    assert pm, f".{port} not connected in u_{inst}"
    return pm.group(1).strip()


def test_collision_resolved_by_contract_two_distinct_wires(tmp_path):
    modules = _collision_modules()
    edges = [
        {"producer_block": "fft_core", "producer_port": "rd_addr",
         "consumer_block": "fft_mem", "consumer_port": "rd_addr",
         "data_width": 24, "edge_id": "core_addr_to_mem"},
        {"producer_block": "fft_ctrl", "producer_port": "addr_out",
         "consumer_block": "fft_core", "consumer_port": "in_rd_addr",
         "data_width": 8, "edge_id": "ctrl_to_core_inaddr"},
    ]
    rtl_paths = {bn: str(tmp_path / f"{bn}.v") for bn in modules}
    asm = generate_caravel_wrapper_top(
        modules, edges, rtl_paths, str(tmp_path / "out"), wrapper_block=None)

    # No hazards: the contract disambiguated the collision.
    assert asm["wiring_errors"] == [], asm["wiring_errors"]

    v = asm["verilog"]
    w_rdaddr = _conn_wire(v, "fft_core", "rd_addr")
    w_inaddr = _conn_wire(v, "fft_core", "in_rd_addr")
    # Two DISTINCT internal wires (not shorted).
    assert w_rdaddr.startswith("w_") and w_inaddr.startswith("w_")
    assert w_rdaddr != w_inaddr
    # Partner ports land on the SAME wire as their contracted endpoint.
    assert _conn_wire(v, "fft_mem", "rd_addr") == w_rdaddr
    assert _conn_wire(v, "fft_ctrl", "addr_out") == w_inaddr
    # Correct widths: 24b and 8b wires both declared, no truncation.
    assert re.search(rf"wire \[23:0\] {re.escape(w_rdaddr)};", v)
    assert re.search(rf"wire \[7:0\] {re.escape(w_inaddr)};", v)


def test_ambiguous_key_without_contract_ports_fails_loud(tmp_path):
    # Same collision, but the edges do NOT name the exact ports -- the assembler
    # must refuse to [0]-pick and emit a wiring hazard so the caller falls back.
    modules = _collision_modules()
    edges = [
        {"producer_block": "fft_core", "consumer_block": "fft_mem",
         "data_width": 24, "edge_id": "core_to_mem"},
    ]
    rtl_paths = {bn: str(tmp_path / f"{bn}.v") for bn in modules}
    asm = generate_caravel_wrapper_top(
        modules, edges, rtl_paths, str(tmp_path / "out"), wrapper_block=None)
    assert asm["wiring_errors"], "ambiguous key should have raised a hazard"
    assert any("ambiguous" in e for e in asm["wiring_errors"])


def test_width_mismatch_named_edge_fails_loud(tmp_path):
    # A contract edge naming two ports of DIFFERENT widths must refuse to short.
    modules = _collision_modules()
    # make mem.rd_addr only 16b so it mismatches core.rd_addr (24b)
    modules["fft_mem"].ports[-1] = VerilogPort("rd_addr", "input", 16, msb=15, lsb=0)
    edges = [
        {"producer_block": "fft_core", "producer_port": "rd_addr",
         "consumer_block": "fft_mem", "consumer_port": "rd_addr",
         "data_width": 24, "edge_id": "core_addr_to_mem"},
    ]
    rtl_paths = {bn: str(tmp_path / f"{bn}.v") for bn in modules}
    asm = generate_caravel_wrapper_top(
        modules, edges, rtl_paths, str(tmp_path / "out"), wrapper_block=None)
    assert asm["wiring_errors"], "width mismatch should have raised a hazard"
    assert any("width" in e for e in asm["wiring_errors"])
