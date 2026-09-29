# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""A4: the chip top is assembled deterministically from the contracts, with
stubs for blocks that have no RTL yet, and elaborated as blocks land."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from orchestrator.langgraph import shell_integration as SI

_FX = Path(__file__).parent / "fixtures" / "vip2"

_EDGE = {"edge_id": "req__m_q__to__rsp__s_q", "producer_block": "req", "producer_port": "m_q",
         "consumer_block": "rsp", "consumer_port": "s_q", "handshake_protocol": "req_resp",
         "data_width_bits": 8, "fields": [{"name": "addr", "width": 8, "msb": 7, "lsb": 0}],
         "sideband_signals": [{"name": "req_valid", "width": 1, "direction": "producer->consumer"},
                              {"name": "req_gnt", "width": 1, "direction": "consumer->producer"},
                              {"name": "rsp_valid", "width": 1, "direction": "consumer->producer"},
                              {"name": "rdata", "width": 8, "direction": "consumer->producer"}],
         "timing": {"req_to_rsp_cycles": {"exact": 1}}}
_PIN_EDGE = {"edge_id": "rsp__m_led__to__pads__s_led", "producer_block": "rsp", "producer_port": "m_led",
             "consumer_block": "pads", "consumer_port": "s_led", "handshake_protocol": "static",
             "data_width_bits": 1, "fields": [{"name": "on", "width": 1, "msb": 0, "lsb": 0}]}


def _rsp_rtl(tmp_path, name="rsp"):
    t = (_FX / "responder.v").read_text().replace("module responder", f"module {name}")
    p = tmp_path / f"{name}.v"
    p.write_text(t)
    return p


class TestContractPortsAndStubs:
    def test_ports_follow_the_conformance_naming_and_roles(self):
        rsp = SI.contract_ports([_EDGE], "rsp")
        assert rsp.ports["s_q_addr"].dir == "input" and rsp.ports["s_q_addr"].width == 8
        assert rsp.ports["s_q_rsp_valid"].dir == "output" and rsp.ports["s_q_req_gnt"].dir == "output"
        req = SI.contract_ports([_EDGE], "req")
        assert req.ports["m_q_addr"].dir == "output" and req.ports["m_q_rdata"].dir == "input"

    def test_stub_declares_every_contract_port_and_ties_outputs(self):
        sv = SI.stub_module("req", SI.contract_ports([_EDGE], "req"))
        assert "module req (" in sv and "output wire [7:0] m_q_addr" in sv
        assert "assign m_q_addr = 8'd0;" in sv and "input  wire [7:0] m_q_rdata" in sv
        assert "input  wire clk" in sv and "input  wire rst_n" in sv


class TestAssemble:
    def test_stubs_plus_real_rtl_wire_every_edge(self, tmp_path):
        rsp = _rsp_rtl(tmp_path)
        asm = SI.assemble_top(tmp_path, top_name="chip_top", blocks=["req", "rsp"], edges=[_EDGE],
                              rtl_paths={"rsp": str(rsp)}, out_dir=tmp_path / "shell")
        assert asm.wiring_errors == [] and asm.stubs == ["req"] and asm.wires == 5
        assert "req u_req (" in asm.verilog and "rsp u_rsp (" in asm.verilog
        assert ".s_q_addr(w_req__m_q__to__rsp__s_q__addr)" in asm.verilog
        assert ".m_q_addr(w_req__m_q__to__rsp__s_q__addr)" in asm.verilog
        assert (tmp_path / "shell" / "stubs" / "req.v").exists()

    def test_unconnected_ports_become_boundary_ports(self, tmp_path):
        rsp = _rsp_rtl(tmp_path)
        # a real block with an extra port the contracts do not know about
        rsp.write_text(rsp.read_text().replace("output reg  [7:0] s_q_rdata",
                                               "output reg  [7:0] s_q_rdata,\n  output wire       irq"))
        rsp.write_text(rsp.read_text().replace("endmodule", "  assign irq = 1'b0;\nendmodule"))
        asm = SI.assemble_top(tmp_path, top_name="chip_top", blocks=["req", "rsp"], edges=[_EDGE],
                              rtl_paths={"rsp": str(rsp)}, out_dir=tmp_path / "shell")
        assert [b["name"] for b in asm.boundary_ports] == ["rsp_irq"]
        assert "output wire rsp_irq" in asm.verilog and ".irq(rsp_irq)" in asm.verilog

    def test_boundary_block_named_like_the_top_keeps_its_pins(self, tmp_path):
        rsp = _rsp_rtl(tmp_path)
        rsp.write_text(rsp.read_text().replace("output reg  [7:0] s_q_rdata",
                                               "output reg  [7:0] s_q_rdata,\n  output wire       m_led_on")
                       .replace("endmodule", "  assign m_led_on = s_q_rsp_valid;\nendmodule"))
        pads = tmp_path / "soc_top.v"
        pads.write_text("module soc_top(input wire clk, input wire rst_n, input wire s_led_on, "
                        "output wire led_pin);\n  assign led_pin = s_led_on;\nendmodule\n")
        asm = SI.assemble_top(tmp_path, top_name="soc_top", blocks=["req", "rsp", "pads"],
                              edges=[_EDGE, _PIN_EDGE], rtl_paths={"rsp": str(rsp), "pads": str(pads)},
                              out_dir=tmp_path / "shell")
        assert asm.wiring_errors == []
        assert "soc_top_core u_pads (" in asm.verilog and "output wire led_pin" in asm.verilog
        assert ".s_led_on(w_rsp__m_led__to__pads__s_led__on)" in asm.verilog
        assert (tmp_path / "shell" / "soc_top_core.v").exists()

    def test_missing_contract_port_and_width_mismatch_are_reported(self, tmp_path):
        rsp = _rsp_rtl(tmp_path)
        rsp.write_text(rsp.read_text().replace("[7:0] s_q_addr", "[3:0] s_q_addr"))
        asm = SI.assemble_top(tmp_path, top_name="chip_top", blocks=["req", "rsp"], edges=[_EDGE],
                              rtl_paths={"rsp": str(rsp)}, out_dir=tmp_path / "shell")
        assert any("width mismatch" in e for e in asm.wiring_errors)
        rsp.write_text(rsp.read_text().replace("s_q_rsp_valid", "s_q_resp_valid"))
        asm = SI.assemble_top(tmp_path, top_name="chip_top", blocks=["req", "rsp"], edges=[_EDGE],
                              rtl_paths={"rsp": str(rsp)}, out_dir=tmp_path / "shell")
        assert any("missing" in e and "s_q_rsp_valid" in e for e in asm.wiring_errors)


@pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not installed")
def test_assembly_elaborates_with_verilator(tmp_path):
    rsp = _rsp_rtl(tmp_path)
    asm = SI.assemble_top(tmp_path, top_name="chip_top", blocks=["req", "rsp"], edges=[_EDGE],
                          rtl_paths={"rsp": str(rsp)}, out_dir=tmp_path / "shell")
    res = SI.elaborate(asm)
    assert res["ran"] and res["ok"], res
    asm2 = SI.assemble_top(tmp_path, top_name="chip_top", blocks=["req", "rsp"], edges=[_EDGE],
                           rtl_paths={}, out_dir=tmp_path / "shell2")   # all stubs
    assert SI.elaborate(asm2)["ok"] and asm2.stubs == ["req", "rsp"]


def test_fan_out_from_one_output_is_not_a_hazard(tmp_path):
    fan = {**_PIN_EDGE, "edge_id": "rsp__m_led__to__other__s_led", "consumer_block": "other"}
    asm = SI.assemble_top(tmp_path, top_name="chip_top", blocks=["req", "rsp", "pads", "other"],
                          edges=[_EDGE, _PIN_EDGE, fan], rtl_paths={}, out_dir=tmp_path / "shell")
    assert asm.wiring_errors == []
    assert asm.verilog.count("w_rsp__m_led__to__pads__s_led__on") == 4   # decl + producer + 2 consumers, one net
    two_drivers = {**_PIN_EDGE, "edge_id": "x__m_led__to__pads__s_led", "producer_block": "x", "producer_port": "m_led"}
    asm = SI.assemble_top(tmp_path, top_name="chip_top", blocks=["req", "rsp", "pads", "x"],
                          edges=[_EDGE, _PIN_EDGE, two_drivers], rtl_paths={}, out_dir=tmp_path / "shell2")
    assert any("driven from two edges" in e for e in asm.wiring_errors)


def _sram_rsp_rtl(tmp_path):
    """The responder with its rdata held in the engine's cs_sram_1rw1r."""
    rsp = _rsp_rtl(tmp_path)
    rsp.write_text(rsp.read_text().replace("endmodule", """  wire [7:0] mem_q;
  cs_sram_1rw1r #(.WIDTH(8), .DEPTH(16)) u_mem (
    .clk(clk), .ce0(s_q_req_valid), .we0(s_q_req_valid), .addr0(s_q_addr[3:0]),
    .wdata0(s_q_addr), .wmask0(1'b1), .rdata0(), .ce1(1'b1), .addr1(s_q_addr[3:0]), .rdata1(mem_q));
endmodule"""))
    return rsp


def test_engine_primitive_lib_joins_the_shell_sources(tmp_path):
    from orchestrator.langgraph.sram_wrapper import engine_lib_sources, wrapper_lib_path
    rsp = _sram_rsp_rtl(tmp_path)
    plain = _rsp_rtl(tmp_path, name="plain")
    assert engine_lib_sources([str(rsp)]) == [wrapper_lib_path()]
    assert engine_lib_sources([str(plain)]) == []                       # no primitive, no lib
    assert engine_lib_sources([str(rsp), wrapper_lib_path()]) == []     # never twice
    if shutil.which("verilator") is None:
        pytest.skip("verilator not installed")
    # live-run defect: every block using cs_sram_1rw1r failed shell elaboration
    # with MODMISSING although it passed its own DV/synth against rtl_lib
    asm = SI.assemble_top(tmp_path, top_name="chip_top", blocks=["req", "rsp"], edges=[_EDGE],
                          rtl_paths={"rsp": str(rsp)}, out_dir=tmp_path / "shell")
    res = SI.elaborate(asm)
    assert res["ran"] and res["ok"], res


def test_missing_engine_module_is_not_blamed_on_the_block():
    rsp_err = "%Error-MODMISSING: /p/rtl/rsp.v:177:3: Cannot find file containing module: '{}'"
    errs = [rsp_err.format(m) for m in ("cs_sram_1rw1r", "cs_rom_1r", "cs_fabric_soc", "cs_mem_macro_shell")]
    blamed, engine = SI.attribute_errors(errs, ["req", "rsp"])
    assert blamed == [] and engine == errs
    # a module the block itself should have provided is still the block's
    own = rsp_err.format("rsp_helper")
    blamed, engine = SI.attribute_errors([own, "rsp: contract port 's_q_addr' missing from rsp.v"], ["rsp"])
    assert [b for b, _ in blamed] == ["rsp", "rsp"] and engine == []
