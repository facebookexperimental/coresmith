# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Pre-layout STA binds cs_sram instances to concrete macros (no more black boxes)."""
import subprocess
from types import SimpleNamespace

from orchestrator.langgraph import macro_sta as ms
from orchestrator.langgraph import ppa_check

_NETLIST = """module rv_l1i0(input clk, input [8:0] m_addr, output [31:0] q);
  wire _00004_;
  cs_sram_1rw1r #(
    .DEPTH(32'sd512),
    .WIDTH(32'sd32)
  ) u_data_w0_hi (
    .addr0(m_addr),
    .addr1(9'h000),
    .ce0(_00004_),
    .clk(clk),
    .rdata0(q)
  );
  cs_sram_1rw1r #(.DEPTH(512), .WIDTH(32)) u_data_w0_lo (.addr0(m_addr), .clk(clk));
  cs_sram_1rw #(.WIDTH(8), .DEPTH(1024)) u_tag (.clk(clk), .addr(m_addr));
  sky130_fd_sc_hd__dfxtp_1 ff (.CLK(clk));
endmodule
"""


def _fake_registry(tmp_path):
    mv = tmp_path / "sky130_sram_2kbyte_1rw1r_32x512_8.v"
    mv.write_text("module sky130_sram_2kbyte_1rw1r_32x512_8(clk0,csb0,web0,wmask0,addr0,din0,dout0,clk1,csb1,addr1,dout1);\n"
                  "input clk0; input csb0; input web0; input [3:0] wmask0; input [8:0] addr0; input [31:0] din0; output [31:0] dout0;\n"
                  "input clk1; input csb1; input [8:0] addr1; output [31:0] dout1;\nendmodule\n")
    lib = tmp_path / "sky130_sram_2kbyte_1rw1r_32x512_8_TT_1p8V_25C.lib"
    lib.write_text("library(x){}\n")
    lef = tmp_path / "sky130_sram_2kbyte_1rw1r_32x512_8.lef"
    lef.write_text("MACRO x\n")
    m32x512 = SimpleNamespace(name="sky130_sram_2kbyte_1rw1r_32x512_8", verilog=str(mv), lib=str(lib), lef=str(lef), ports="1rw1r")
    mv8 = tmp_path / "sky130_sram_1kbyte_1rw1r_8x1024_8.v"
    mv8.write_text("module sky130_sram_1kbyte_1rw1r_8x1024_8(clk0,csb0,web0,addr0,din0,dout0,clk1,csb1,addr1,dout1);\n"
                   "input clk0; input csb0; input web0; input [9:0] addr0; input [7:0] din0; output [7:0] dout0;\n"
                   "input clk1; input csb1; input [9:0] addr1; output [7:0] dout1;\nendmodule\n")
    lib8 = tmp_path / "m8.lib"
    lib8.write_text("library(y){}\n")
    m8x1024 = SimpleNamespace(name="sky130_sram_1kbyte_1rw1r_8x1024_8", verilog=str(mv8), lib=str(lib8), ports="1rw1r")
    return {(32, 512): m32x512, (8, 1024): m8x1024}


def _patch_registry(monkeypatch, reg):
    from orchestrator.langgraph import macro_registry as mr
    monkeypatch.setattr(mr, "discover_macros", lambda: reg)
    monkeypatch.setattr(mr, "resolve_shell", lambda spec, registry=None, allow_generate=False: reg.get((spec.width, spec.depth)))


def test_find_instances_parses_yosys_and_plain_parameters():
    inst = ms.find_instances(_NETLIST)
    assert [(k, w, d) for _s, _e, k, w, d in inst] == [("cs_sram_1rw1r", 32, 512), ("cs_sram_1rw1r", 32, 512), ("cs_sram_1rw", 8, 1024)]
    assert ms.find_instances("module t; sky130_fd_sc_hd__inv_1 i(.A(a)); endmodule") == []


def test_bind_rewrites_instances_and_emits_structural_wrappers(tmp_path, monkeypatch):
    _patch_registry(monkeypatch, _fake_registry(tmp_path))
    b = ms.bind_netlist_macros(_NETLIST)
    assert b.ok and b.instances == 3 and len(b.libs) == 2 and len(b.lefs) == 1
    assert "cs_sram_1rw1r__w32_d512 u_data_w0_hi (" in b.netlist and "cs_sram_1rw1r__w32_d512 u_data_w0_lo (" in b.netlist
    assert "cs_sram_1rw__w8_d1024 u_tag (" in b.netlist and "#(" not in b.netlist.split("u_data_w0_hi")[0].split("wire _00004_;")[1]
    assert "module cs_sram_1rw1r__w32_d512 (" in b.wrappers and "sky130_sram_2kbyte_1rw1r_32x512_8 u_macro (" in b.wrappers
    assert ".csb0(ce0)" in b.wrappers and ".dout1(rdata1)" in b.wrappers and "~" not in b.wrappers   # structural only
    assert "module cs_sram_1rw__w8_d1024 (" in b.wrappers and ".csb1" not in b.wrappers.split("cs_sram_1rw__w8_d1024")[1]
    assert sorted(n for _k, _w, _d, n in b.bound) == ["sky130_sram_1kbyte_1rw1r_8x1024_8", "sky130_sram_2kbyte_1rw1r_32x512_8"]


def test_unresolved_geometry_is_reported_not_black_boxed(tmp_path, monkeypatch):
    reg = _fake_registry(tmp_path)
    del reg[(8, 1024)]
    _patch_registry(monkeypatch, reg)
    b = ms.bind_netlist_macros(_NETLIST)
    assert not b.ok and b.unresolved == [("cs_sram_1rw", 8, 1024)] and "cs_sram_1rw 8x1024" in ms.describe_unresolved(b)


def _sta_env(tmp_path, monkeypatch, netlist_text):
    netlist = tmp_path / "n.v"
    netlist.write_text(netlist_text)
    sdc = tmp_path / "t.sdc"
    sdc.write_text("create_clock -name clk -period 15.625 [get_ports clk]\n")
    lib = tmp_path / "std.lib"
    lib.write_text("library(std){}\n")
    monkeypatch.setattr(ppa_check.shutil, "which", lambda n: "/usr/bin/sta" if n == "sta" else None)
    captured = {}

    def fake_run(cmd, **kw):
        captured["tcl"] = open(cmd[-1]).read()
        for line in captured["tcl"].splitlines():
            if line.startswith("read_verilog "):
                captured["verilog"] = open(line.split(" ", 1)[1]).read()
        return subprocess.CompletedProcess(cmd, 0, stdout="Startpoint: a\nEndpoint: b\n  slack (MET)  0.77\nwns max 0.77\ntns max 0.00\n", stderr="")
    monkeypatch.setattr(ppa_check, "run_process", fake_run)
    return netlist, sdc, lib, captured


def test_pre_layout_sta_links_macro_liberty_and_wrappers(tmp_path, monkeypatch):
    _patch_registry(monkeypatch, _fake_registry(tmp_path))
    netlist, sdc, lib, cap = _sta_env(tmp_path, monkeypatch, _NETLIST)
    out = ppa_check.run_pre_layout_sta(str(netlist), str(sdc), str(lib), "rv_l1i0", report_path=str(tmp_path / "r.rpt"))
    assert out["wns_ns"] == 0.77
    tcl = cap["tcl"]
    assert tcl.count("read_liberty ") == 3 and "sky130_sram_2kbyte_1rw1r_32x512_8_TT_1p8V_25C.lib" in tcl
    assert "module cs_sram_1rw1r__w32_d512 (" in cap["verilog"] and "cs_sram_1rw1r__w32_d512 u_data_w0_hi (" in cap["verilog"]
    assert "cs_sram_1rw1r #(" not in cap["verilog"]
    assert "# macros: " in (tmp_path / "r.rpt").read_text()


def test_pre_layout_sta_refuses_to_measure_an_unbound_geometry(tmp_path, monkeypatch):
    reg = _fake_registry(tmp_path)
    del reg[(8, 1024)]
    _patch_registry(monkeypatch, reg)
    netlist, sdc, lib, cap = _sta_env(tmp_path, monkeypatch, _NETLIST)
    out = ppa_check.run_pre_layout_sta(str(netlist), str(sdc), str(lib), "rv_l1i0")
    assert out["wns_ns"] is None and "unresolved" in out["sta_error"] and "tcl" not in cap


def test_binding_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("CORESMITH_STA_BIND_MACROS", "0")
    netlist, sdc, lib, cap = _sta_env(tmp_path, monkeypatch, _NETLIST)
    out = ppa_check.run_pre_layout_sta(str(netlist), str(sdc), str(lib), "rv_l1i0")
    import re
    assert out["wns_ns"] == 0.77 and cap["tcl"].count("read_liberty ") == 1 and re.search(r"cs_sram_1rw1r\s+u_data_w0_hi \(", cap["verilog"])


def test_parse_sta_report_keeps_the_real_margin_when_report_wns_clamps_to_zero():
    rpt = ("Startpoint: a\nEndpoint: b\n          15.48   data required time\n         -13.49   data arrival time\n"
           "           1.99   slack (MET)\n\nStartpoint: c\nEndpoint: d\n           2.10   slack (MET)\n"
           "wns max 0.00\ntns max 0.00\n")
    out = ppa_check.parse_sta_report(rpt)
    assert out["wns_ns"] == 1.99 and out["worst_slack_ns"] == 1.99 and out["tns_ns"] == 0.0
    viol = ppa_check.parse_sta_report("          -3.42   slack (VIOLATED)\nwns max -3.42\ntns max -40.10\n")
    assert viol["wns_ns"] == -3.42 and viol["worst_slack_ns"] == -3.42
    assert ppa_check.parse_sta_report("wns max 0.00\ntns max 0.00\n")["wns_ns"] == 0.0     # nothing better known
