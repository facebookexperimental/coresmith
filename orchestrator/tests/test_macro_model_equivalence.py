"""Memory-model equivalence for the gate-level sim.

Integration DV simulates wrapped memories through ``rtl_lib/cs_sram.v`` (BEHAV);
the gate sim simulates the sky130 macro they become through the generated
cycle-accurate stand-in (``gate_sim.macro_model_source``). If the two disagree,
every gate-sim verdict on a memory-bearing chip is noise. The slow test drives
both -- and the PDK's own behavioural model (``--timing``) as the arbiter where
its semantics are defined -- with write-during-reset, masked byte writes and a
read-first collision, and requires identical read data.

The fast tests cover the gate sim's handling of a netlist whose memories are
UNBOUND ``cs_mem_macro_shell`` placeholders (read data tied to zero): the MCU+FFT
evaluation run's "cycle 15 mcu_halted_o expected 1 got 0" divergence came from
exactly that, not from either memory model.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from orchestrator.harness import gate_sim as g

REPO = Path(__file__).resolve().parents[2]
CS_SRAM = REPO / "orchestrator" / "langgraph" / "rtl_lib" / "cs_sram.v"
MACRO = "sky130_sram_2kbyte_1rw1r_32x512_8"
VERILATOR_536 = Path("/home/ubuntu/.local/verilator-5.036/bin/verilator")


def _pdk_root() -> Path | None:
    for cand in (os.environ.get("PDK_ROOT"), str(REPO / ".pdk")):
        if cand and (Path(cand) / "sky130A").is_dir():
            return Path(cand)
    return None


def _macro_verilog() -> Path | None:
    root = _pdk_root()
    if root is None:
        return None
    p = root / "sky130A" / "libs.ref" / "sky130_sram_macros" / "verilog" / f"{MACRO}.v"
    return p if p.is_file() else None


def _verilator() -> str | None:
    if VERILATOR_536.is_file():
        return str(VERILATOR_536)
    exe = shutil.which("verilator")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True).stdout
        major, minor = (int(x) for x in out.split()[1].split(".")[:2])
    except Exception:  # noqa: BLE001
        return None
    return exe if (major, minor) >= (5, 36) else None


# ---------------------------------------------------------------------------
# slow: the three models on the same stimulus
# ---------------------------------------------------------------------------

_TB = r"""
`timescale 1ns/1ps
module tb;
  reg clk = 0; always #5 clk = ~clk;
  reg ce0 = 0, we0 = 0, ce1 = 0; reg [3:0] wm = 0;
  reg [8:0] a0 = 0, a1 = 0; reg [31:0] d0 = 0;
  wire [31:0] rtl0, rtl1, gen0, gen1, pdk0, pdk1;
  cs_sram_1rw1r #(.WIDTH(32), .DEPTH(512), .USE_WMASK(1), .READ_FIRST(1)) u_rtl (
    .clk(clk), .ce0(ce0), .we0(we0), .addr0(a0), .wdata0(d0), .wmask0(wm), .rdata0(rtl0),
    .ce1(ce1), .addr1(a1), .rdata1(rtl1));
  gen_macro u_gen (.clk0(clk), .csb0(~ce0), .web0(~we0), .wmask0(wm), .addr0(a0),
    .din0(d0), .dout0(gen0), .clk1(clk), .csb1(~ce1), .addr1(a1), .dout1(gen1));
  pdk_macro u_pdk (.clk0(clk), .csb0(~ce0), .web0(~we0), .wmask0(wm), .addr0(a0),
    .din0(d0), .dout0(pdk0), .clk1(clk), .csb1(~ce1), .addr1(a1), .dout1(pdk1));
  // Sample one period after the access edge (before the PDK model's T_HOLD X).
  task step(input [8*8-1:0] tag, input check_pdk);
    begin
      @(posedge clk); #9;
      $display("R %0s %h %h %h %h %h %h %0d", tag, rtl0, gen0, pdk0, rtl1, gen1, pdk1, check_pdk);
    end
  endtask
  integer i;
  initial begin
    // (1) the evaluation run's prefix: reset low, one program byte 0xF0 written
    //     to word 0 lane 0 while in reset, then port 1 fetches word 0.
    for (i = 0; i < 10; i = i + 1) step("idle", 1);
    ce0 = 1; we0 = 1; a0 = 0; d0 = {4{8'hF0}}; wm = 4'b0001; step("wr_rst", 1);
    ce0 = 0; we0 = 0; step("idle", 1); step("idle", 1);
    ce1 = 1; a1 = 0; for (i = 0; i < 6; i = i + 1) step("fetch0", 1);
    // (2) masked byte writes: lane 1 then lane 3 of word 5, then read both ports.
    ce1 = 0; ce0 = 1; we0 = 1; a0 = 5; d0 = 32'hAABBCCDD; wm = 4'b0010; step("wm1", 0);
    wm = 4'b1000; d0 = 32'h11223344; step("wm3", 0);
    wm = 4'b0101; d0 = 32'h55667788; step("wm02", 0);
    we0 = 0; ce1 = 1; a1 = 5; step("rd5", 1);
    // (3) read-first collision: full write to word 5 while port 1 reads it. The
    //     PDK model's same-edge order is undefined (it warns), so only the two
    //     cycle models are compared here -- both must return the OLD word.
    //     (While port 1 is deselected -- the wm* steps -- the PDK model drives
    //     X after every edge; both cycle models hold, so those rows are not
    //     compared against it either.)
    we0 = 1; wm = 4'hF; d0 = 32'hDEADBEEF; step("collide", 0);
    we0 = 0; step("after", 1);
    $finish;
  end
endmodule
"""


@pytest.mark.slow
def test_cs_sram_matches_generated_macro_model(tmp_path):
    verilator = _verilator()
    macro_v = _macro_verilog()
    if verilator is None:
        pytest.skip("Verilator >= 5.036 not available")
    if macro_v is None:
        pytest.skip("sky130 SRAM macro Verilog not found (PDK_ROOT / .pdk)")

    src = g.macro_model_source(MACRO, "1rw1r", 32, 512, 8,
                               iface=g.macro_interface(MACRO, macro_v))
    assert src, "generated macro model unresolved"
    (tmp_path / "gen.v").write_text(src.replace(f"module {MACRO}", "module gen_macro"))
    pdk = macro_v.read_text().replace(f"module {MACRO}", "module pdk_macro") \
        .replace("VERBOSE = 1", "VERBOSE = 0")
    (tmp_path / "pdk.v").write_text(pdk)
    (tmp_path / "tb.v").write_text(_TB)
    build = subprocess.run(
        [verilator, "--binary", "--timing", "-Wno-fatal", "-Wno-lint", "-Wno-style",
         "--top-module", "tb", "-o", "simx", "tb.v", str(CS_SRAM), "gen.v", "pdk.v"],
        cwd=tmp_path, capture_output=True, text=True, timeout=600)
    assert build.returncode == 0, build.stdout[-2000:] + build.stderr[-2000:]
    run = subprocess.run([str(tmp_path / "obj_dir" / "simx")], cwd=tmp_path,
                         capture_output=True, text=True, timeout=120)
    rows = [ln.split()[1:] for ln in run.stdout.splitlines() if ln.startswith("R ")]
    assert len(rows) == 25, run.stdout

    for tag, rtl0, gen0, pdk0, rtl1, gen1, pdk1, check_pdk in rows:
        assert rtl1 == gen1, f"{tag}: port-1 read cs_sram={rtl1} generated={gen1}"
        if check_pdk == "1":
            assert gen1 == pdk1, f"{tag}: port-1 read generated={gen1} pdk={pdk1}"
        if tag in ("rd5", "after"):
            assert rtl0 == gen0 == pdk0, f"{tag}: port-0 {rtl0} {gen0} {pdk0}"

    by_tag = {}
    for r in rows:
        by_tag.setdefault(r[0], []).append(r)
    # write during reset is visible to the first fetch
    assert by_tag["fetch0"][-1][4] == "000000f0"
    # masked writes touched only their lanes
    assert by_tag["rd5"][0][4] == "1166cc88"
    # read-first: the colliding read returns the old word, the next one the new
    assert by_tag["collide"][0][4] == "1166cc88"
    assert by_tag["after"][0][4] == "deadbeef"


# ---------------------------------------------------------------------------
# fast: unbound macro shells in the netlist
# ---------------------------------------------------------------------------

_SHELL = "\\$paramod$de03\\cs_mem_macro_shell"

_NETLIST = f"""
module {_SHELL} (clk, ce0, we0, wmask0, addr0, wdata0, rdata0, ce1, addr1, rdata1);
  input clk;
  wire clk;
  input ce0;
  input we0;
  input [3:0] wmask0;
  input [8:0] addr0;
  input [31:0] wdata0;
  output [31:0] rdata0;
  input ce1;
  input [8:0] addr1;
  output [31:0] rdata1;
  sky130_fd_sc_hd__conb_1 _00_ (
    .LO(rdata1[0])
  );
endmodule

module top(clk, q);
  input clk;
  output [31:0] q;
  {_SHELL}  u_mem (
    .clk(clk)
  );
endmodule
"""


def test_unbound_shell_is_detected_with_geometry():
    shells = g.unbound_macro_shells(_NETLIST, ("sky130_fd_sc_hd",))
    assert [(s.module, s.kind, s.width, s.depth, s.nmask) for s in shells] == [
        (_SHELL, "sram", 32, 512, 4)]


def test_bound_shell_is_not_detected():
    bound = _NETLIST.replace("sky130_fd_sc_hd__conb_1 _00_ (\n    .LO(rdata1[0])\n  );",
                             f"{MACRO} u_macro (.clk0(clk));")
    assert g.unbound_macro_shells(bound, ("sky130_fd_sc_hd",)) == []


def test_bind_shells_flag_both_branches(monkeypatch):
    monkeypatch.delenv(g.GATE_SIM_BIND_SHELLS_ENV, raising=False)
    assert g.gate_sim_bind_shells() is True
    monkeypatch.setenv(g.GATE_SIM_BIND_SHELLS_ENV, "0")
    assert g.gate_sim_bind_shells() is False


def test_unresolvable_shell_reports_error_not_zero():
    shells = g.unbound_macro_shells(_NETLIST, ("sky130_fd_sc_hd",))
    g.bind_macro_shells_for_sim(shells)
    assert shells[0].replacement == ""
    assert "verified prebind manifest" in shells[0].error


def test_shell_binds_to_pdk_macro_and_replaces_placeholder(monkeypatch, tmp_path):
    root = _pdk_root()
    if _macro_verilog() is None:
        pytest.skip("sky130 SRAM macros not available")
    monkeypatch.setenv("PDK_ROOT", str(root))
    from orchestrator.langgraph.macro_registry import discover_macros
    macros = discover_macros(str(root))
    macro = next(m for m in macros.values() if m.name == MACRO)
    shells = g.unbound_macro_shells(_NETLIST, ("sky130_fd_sc_hd",))
    manifest = {"bindings": [{
        "kind": "sram", "width": 32, "depth": 512, "nport": 2,
        "mask_lanes": 4,
        "macro": macro.name, "ports": macro.ports,
        "data_bits": macro.data_bits, "words": macro.words,
        "mask_bits": macro.mask_bits,
        "model": {"path": macro.verilog},
    }]}
    g.bind_macro_shells_for_sim(shells, manifest)
    sh = shells[0]
    assert sh.macro == MACRO, sh.error
    # active-low selects, the shell's own 4-lane mask, both read ports
    for conn in (".csb0(~ce0)", ".web0(~we0)", ".wmask0(wmask0)", ".dout1(rdata1)",
                 ".csb1(~ce1)"):
        assert conn in sh.replacement
    patched = g.netlist_with_bound_shells(_NETLIST, shells)
    assert "conb_1" not in patched and f"{MACRO} u_macro" in patched
    assert patched.count("endmodule") == _NETLIST.count("endmodule")
    assert "module top(" in patched
    files, unresolved = g.bound_shell_model_files(shells, tmp_path, [])
    assert unresolved == [] and len(files) == 1
    assert f"module {MACRO}" in Path(files[0]).read_text()
