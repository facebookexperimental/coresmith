# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""cs_sram macro-binding chain (MEM_IMPL swap + backend MACRO synth + shell
binding + memory-as-flops gate).

Every observable behavior change is env-gated default-ON and tested on BOTH
branches. FAIRNESS: only generic synthetic geometries -- no benchmark/exercise/
golden names anywhere.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from orchestrator.langgraph import macro_registry as mr
from orchestrator.langgraph import ppa_check as pc
from orchestrator.langgraph import sram_wrapper as sw

_HAVE_YOSYS = shutil.which("yosys") is not None
_LIB = sw.wrapper_lib_path()


# ---------------------------------------------------------------------------
# Part B: backend selects MACRO -- the chparam directive (env-gated both ways)
# ---------------------------------------------------------------------------

class TestMacroDirective:
    def test_directive_default_on(self, monkeypatch):
        monkeypatch.delenv("CORESMITH_SRAM_MACRO", raising=False)
        d = sw.backend_sram_macro_directive()
        assert d.startswith('chparam -set MEM_IMPL "MACRO"')
        # names the blessed wrappers + the underlying primitives ...
        for m in ("cs_sram_1rw", "cs_sram_1rw1r", "cs_rom_1r",
                  "cs_mem_1rw", "cs_mem_1rw1r"):
            assert m in d
        # ... but NEVER the flop tier (an explicit FLOP override survives anyway,
        # but we must not even name it).
        assert "cs_fpmem" not in d

    def test_directive_off(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_SRAM_MACRO", "0")
        assert sw.backend_sram_macro_directive() == ""
        # the probe still forces it (models the MACRO netlist regardless of flag)
        assert sw.backend_sram_macro_directive(force=True).startswith("chparam")

    def test_enabled_flag_both_branches(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_SRAM_MACRO", "1")
        assert sw.backend_sram_macro_enabled() is True
        monkeypatch.setenv("CORESMITH_SRAM_MACRO", "0")
        assert sw.backend_sram_macro_enabled() is False


@pytest.mark.skipif(not _HAVE_YOSYS, reason="yosys absent")
class TestBackendMacroSynth:
    """Real-yosys proof that the directive turns a wrapped memory into
    cs_mem_macro_shell leaves with ZERO storage flops (vs a flop array)."""

    def _dut(self, tmp_path) -> Path:
        dut = tmp_path / "memblk.v"
        dut.write_text(
            "module memblk(input clk, input ce0, input we0, input [9:0] a0,\n"
            "  input [31:0] d0, output [31:0] q0, input ce1, input [9:0] a1,\n"
            "  output [31:0] q1);\n"
            "  cs_sram_1rw1r #(.WIDTH(32), .DEPTH(1024)) u (.clk(clk),\n"
            "    .ce0(ce0), .we0(we0), .addr0(a0), .wdata0(d0), .wmask0(4'b0),\n"
            "    .rdata0(q0), .ce1(ce1), .addr1(a1), .rdata1(q1));\n"
            "endmodule\n"
        )
        return dut

    def test_macro_selected_has_shell_zero_flops(self, tmp_path):
        dut = self._dut(tmp_path)
        p = pc.probe_memory_flops([_LIB, str(dut)], "memblk", apply_macro=True)
        assert p is not None and p["elaborated"]
        assert p["macro_shells"] >= 1          # bound to the shell leaf
        assert p["memories"] == []             # no inferred flop-array memory
        assert p["total_mem_bits"] == 0

    def test_default_behav_is_a_flop_array(self, tmp_path):
        # Without the MACRO directive the same wrapper is a flop-backed $mem --
        # the exact pre-fix defect. (apply_macro=False)
        dut = self._dut(tmp_path)
        p = pc.probe_memory_flops([_LIB, str(dut)], "memblk", apply_macro=False)
        assert p is not None and p["elaborated"]
        assert p["macro_shells"] == 0
        assert (32, 1024) in p["memories"]     # 32Kbit flop array

    def test_fpmem_stays_flops_under_macro(self, tmp_path):
        # cs_fpmem passes an explicit .MEM_IMPL("FLOP"); the chparam-changed
        # module default must NOT override it -> the flop tier stays flops.
        dut = tmp_path / "fpblk.v"
        dut.write_text(
            "module fpblk(input clk, input ce, input we, input [9:0] a,\n"
            "  input [31:0] wd, output [31:0] rd);\n"
            "  cs_fpmem_1rw #(.WIDTH(32), .DEPTH(1024)) u (.clk(clk), .ce(ce),\n"
            "    .we(we), .addr(a), .wdata(wd), .rdata(rd));\n"
            "endmodule\n"
        )
        p = pc.probe_memory_flops([_LIB, str(dut)], "fpblk", apply_macro=True)
        assert p is not None and p["elaborated"]
        assert p["macro_shells"] == 0
        assert (32, 1024) in p["memories"]     # still a flop array (FLOP tier)


# ---------------------------------------------------------------------------
# Part C: bind each macro shell to a concrete on-disk macro
# ---------------------------------------------------------------------------

def _synthetic_macro(dirpath: Path, name, width, depth, ports, mask, kind="sram"):
    files = {}
    for ext in ("lef", "gds", "spice", "verilog", "lib"):
        p = dirpath / f"{name}.{ext}"
        p.write_text("stub")
        files[ext] = str(p)
    return mr.MacroInfo(
        name=name, lef=files["lef"], gds=files["gds"], spice=files["spice"],
        verilog=files["verilog"], lib="" if kind == "rom" else files["lib"],
        words=depth, data_bits=width, bits=width * depth, ports=ports,
        mask_bits=mask, kind=kind,
    )


class TestShellDetection:
    def test_explicit_param_form(self):
        src = (
            "cs_mem_macro_shell #(.WIDTH(32), .DEPTH(1024), .NPORT(2)) u0 (.clk(c));\n"
            "cs_rom_macro_shell #(.WIDTH(16), .DEPTH(512)) u1 (.clk(c));\n"
        )
        specs = {s.key() for s in mr.detect_macro_shells(src)}
        assert ("sram", 32, 1024, 2) in specs
        assert ("rom", 16, 512, 1) in specs

    @pytest.mark.skipif(not _HAVE_YOSYS, reason="yosys absent")
    def test_derived_netlist_form(self, tmp_path):
        # A yosys-derived netlist bakes params into $paramod..\cs_mem_macro_shell
        # (no #()); the geometry is recovered from the derived port widths.
        dut = tmp_path / "d.v"
        dut.write_text(
            "module d(input clk, input ce0, input we0, input [9:0] a0,\n"
            "  input [31:0] wd, output [31:0] q0, input ce1, input [9:0] a1,\n"
            "  output [31:0] q1);\n"
            "  cs_sram_1rw1r #(.WIDTH(32), .DEPTH(1024)) u (.clk(clk), .ce0(ce0),\n"
            "    .we0(we0), .addr0(a0), .wdata0(wd), .wmask0(4'b0), .rdata0(q0),\n"
            "    .ce1(ce1), .addr1(a1), .rdata1(q1));\nendmodule\n"
        )
        net = tmp_path / "net.v"
        ys = tmp_path / "s.ys"
        ys.write_text(
            f"read_verilog -sv {_LIB} {dut}\n"
            f'{sw.backend_sram_macro_directive(force=True)}\n'
            f"hierarchy -check -top d\nproc; opt; memory_collect; opt_clean\n"
            f"write_verilog -noattr {net}\n"
        )
        import subprocess
        r = subprocess.run(["yosys", "-s", str(ys)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-500:]
        specs = mr.detect_macro_shells(net.read_text())
        assert any(s.kind == "sram" and s.width == 32 and s.depth == 1024
                   for s in specs)

    def test_no_shell_no_spec(self):
        assert mr.detect_macro_shells("module a; endmodule") == []


class TestShellBinding:
    def test_exact_prebuilt_match(self):
        d = Path(tempfile.mkdtemp())
        reg = {"m": _synthetic_macro(
            d, "sky130_sram_2kbyte_1rw1r_32x1024_8", 32, 1024, "1rw1r", 8)}
        net = "cs_mem_macro_shell #(.WIDTH(32), .DEPTH(1024), .NPORT(2)) u (.clk(c));"
        res = mr.bind_macro_shells(net, registry=reg, allow_generate=False,
                                   is_text=True)
        assert res.ok and not res.errors
        assert [mi.name for _, mi in res.resolved] == [
            "sky130_sram_2kbyte_1rw1r_32x1024_8"]

    def test_unresolvable_is_hard_error_not_flops(self, monkeypatch):
        # empty registry + OpenRAM declines -> a HARD, reported error (never a
        # silent flop fallback). Mock OpenRAM generation to "declined".
        from orchestrator.langgraph import openram_gen as og
        monkeypatch.setattr(og, "generate_openram_macro",
                            lambda *a, **k: None)
        net = "cs_mem_macro_shell #(.WIDTH(29), .DEPTH(777), .NPORT(1)) u (.clk(c));"
        res = mr.bind_macro_shells(net, registry={}, allow_generate=True,
                                   is_text=True)
        assert not res.ok
        assert len(res.errors) == 1
        assert "could NOT be bound" in res.errors[0]
        assert "flop" in res.errors[0].lower()      # explicitly refuses flops
        assert res.resolved == []

    def test_rom_shell_resolves_to_prebuilt(self):
        d = Path(tempfile.mkdtemp())
        reg = {"r": _synthetic_macro(
            d, "rom_1r_32_1024_sky130", 32, 1024, "1r", 0, kind="rom")}
        net = "cs_rom_macro_shell #(.WIDTH(32), .DEPTH(1024)) u (.clk(c));"
        res = mr.bind_macro_shells(net, registry=reg, allow_generate=False,
                                   is_text=True)
        assert res.ok
        assert [mi.name for _, mi in res.resolved] == ["rom_1r_32_1024_sky130"]

    def test_ensure_macro_registry_injection(self):
        # ensure_macro now honors an injected registry (the seam Part C relies
        # on) without touching discover_macros / the PDK.
        from orchestrator.langgraph import openram_gen as og
        d = Path(tempfile.mkdtemp())
        reg = {"m": _synthetic_macro(
            d, "sram_1rw_16_256_8_sky130", 16, 256, "1rw", 8)}
        got = og.ensure_macro(256, 16, allow_generate=False, registry=reg)
        assert got is not None and got.name == "sram_1rw_16_256_8_sky130"
        # empty injected registry, no generation -> None (caller surfaces it)
        assert og.ensure_macro(256, 16, allow_generate=False, registry={}) is None


# ---------------------------------------------------------------------------
# Part D: memory-as-flops hard-block gate (env-gated both ways)
# ---------------------------------------------------------------------------

class TestMemFlopGate:
    def test_flag_both_branches(self, monkeypatch):
        monkeypatch.delenv("CORESMITH_MEM_FLOP_GATE", raising=False)
        assert sw.mem_flop_gate_enabled() is True          # default ON
        monkeypatch.setenv("CORESMITH_MEM_FLOP_GATE", "0")
        assert sw.mem_flop_gate_enabled() is False          # restored old behavior

    def test_large_flop_array_blocked(self):
        # 8b x 2048 = 16384 bits, depth 2048 (>= macro_depth) -> BLOCK
        ok, reasons = sw.gate_memory_as_flops([(8, 2048)])
        assert not ok and len(reasons) == 1
        assert "flip-flop array" in reasons[0]
        assert "cs_sram" in reasons[0]

    def test_macro_bound_passes(self):
        # a MACRO-selected wrapped memory contributes NO inferred memory here.
        ok, reasons = sw.gate_memory_as_flops([])
        assert ok and reasons == []

    def test_small_subthreshold_passes(self):
        # 8b x 64 (512 bits, depth 64) -- well below threshold, legit flops
        ok, _ = sw.gate_memory_as_flops([(8, 64)])
        assert ok

    def test_many_small_arrays_do_not_sum(self):
        # per-memory (never summed): four 8x128 arrays (each sub-threshold)
        ok, _ = sw.gate_memory_as_flops([(8, 128)] * 4)
        assert ok

    def test_fpmem_geometry_exempt_even_when_large(self):
        # a LARGE cs_fpmem (64b x 1024 = 65536 bits, deep) is the blessed flop
        # tier -> exempt because its geometry matches a declared cs_fpmem.
        big = [(64, 1024)]
        assert not sw.gate_memory_as_flops(big)[0]                       # would block
        assert sw.gate_memory_as_flops(big, fpmem_geoms=[(64, 1024)])[0]  # exempt

    def test_deep_narrow_flagged_by_depth(self):
        # 8b x 1024 = 8192 bits (< min_bits) but depth 1024 (>= macro_depth):
        # a 1024:1 read mux -> BLOCK (depth trigger).
        ok, _ = sw.gate_memory_as_flops([(8, 1024)])
        assert not ok

    def test_fpmem_instances_parser(self):
        rtl = ("module b; cs_fpmem_1rw1r #(.WIDTH(16),.DEPTH(64)) u(.clk(c)); "
               "cs_fpmem_1rw #(.DEPTH(32),.WIDTH(8)) v(.clk(c)); endmodule")
        insts = set(sw.fpmem_instances(rtl))
        assert (16, 64) in insts and (8, 32) in insts

    @pytest.mark.skipif(not _HAVE_YOSYS, reason="yosys absent")
    def test_probe_gate_end_to_end_raw_array(self, tmp_path):
        dut = tmp_path / "rawblk.v"
        dut.write_text(
            "module rawblk(input clk, input ce, input we, input [10:0] a,\n"
            "  input [7:0] wd, output reg [7:0] rd);\n"
            "  reg [7:0] mem [0:2047];\n"
            "  always @(posedge clk) if (ce) begin\n"
            "    if (we) mem[a] <= wd; rd <= mem[a]; end\nendmodule\n"
        )
        p = pc.probe_memory_flops([str(dut)], "rawblk", apply_macro=True)
        assert p is not None and p["elaborated"]
        ok, reasons = sw.gate_memory_as_flops(p["memories"])
        assert not ok and reasons
