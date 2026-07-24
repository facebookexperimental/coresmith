# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the unified ``cs_mem`` memory primitive (fix #3).

Covers three contracts:

  (a) cs_mem_1rw1r MEM_IMPL="BEHAV" actually stores+reads (non-zero), and
      MEM_IMPL="MACRO" is an empty shell (reads zero) -- requires Verilator,
      so it is requires_nix-gated like the other EDA tests.
  (b) the source deduper keeps the rtl_lib behavioral body of a cs_sram_* cell
      even when chip_top defines an empty blackbox of the same name (the
      regression that made SRAM-backed blocks read all-zero in DV).
  (c) the Integration Lead postcondition raises on a chip_top that DEFINES a
      cs_* memory primitive.

(b) and (c) are pure (no LLM, no EDA) and run under
``-m "not live_llm and not requires_nix and not e2e"``.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from orchestrator.langchain.agents.integration_lead import (
    assert_no_memory_primitive_defined,
)
from orchestrator.langgraph.integration_helpers import _dedup_module_sources
from orchestrator.langgraph.sram_wrapper import wrapper_lib_path

_LIB = wrapper_lib_path()

requires_verilator = pytest.mark.skipif(
    shutil.which("verilator") is None, reason="Verilator not installed"
)


def _defs(paths, name):
    return sum(
        len(re.findall(rf"^\s*module\s+{name}\b", Path(p).read_text(), re.MULTILINE))
        for p in paths
    )


# ---------------------------------------------------------------------------
# (a) elaboration: BEHAV stores+reads non-zero, MACRO is an empty shell
# ---------------------------------------------------------------------------

_TB_TEMPLATE = """
module tb;
  localparam WIDTH = 8;
  localparam DEPTH = 16;
  reg clk = 0;
  reg ce0 = 0, we0 = 0, ce1 = 0;
  reg  [3:0] addr0 = 0, addr1 = 0;
  reg  [7:0] wdata0 = 0;
  wire [7:0] rdata0, rdata1;
  integer errors = 0;

  cs_mem_1rw1r #(.MEM_IMPL("{IMPL}"), .WIDTH(WIDTH), .DEPTH(DEPTH)) dut (
    .clk(clk), .ce0(ce0), .we0(we0), .addr0(addr0), .wdata0(wdata0),
    .rdata0(rdata0), .ce1(ce1), .addr1(addr1), .rdata1(rdata1)
  );

  always #5 clk = ~clk;

  task step; begin @(posedge clk); #1; end endtask

  initial begin
    // write 0xAB to addr 3 via port0
    ce0 = 1; we0 = 1; addr0 = 3; wdata0 = 8'hAB; step;
    we0 = 0;
    // read addr 3 on the read-only port1 (1-cycle registered read)
    ce1 = 1; addr1 = 3; step;
    // rdata1 now reflects mem[3]
    if (rdata1 !== {EXPECT}) begin
      $display("FAIL impl={IMPL}: rdata1=%h expected={EXPECT}", rdata1);
      errors = errors + 1;
    end
    if (errors == 0) $display("TBPASS");
    else             $display("TBFAIL");
    $finish;
  end
endmodule
"""


def _run_cs_mem_tb(tmp_path: Path, impl: str, expect: str) -> str:
    tb = tmp_path / f"tb_{impl.lower()}.v"
    tb.write_text(
        _TB_TEMPLATE.replace("{IMPL}", impl).replace("{EXPECT}", expect)
    )
    obj = tmp_path / f"obj_{impl.lower()}"
    build = subprocess.run(
        [
            "verilator", "--binary", "-j", "0",
            "-Wno-fatal", "-Wno-DECLFILENAME", "-Wno-PINCONNECTEMPTY",
            "--top-module", "tb",
            "--Mdir", str(obj),
            str(_LIB), str(tb),
        ],
        capture_output=True, text=True, timeout=180,
    )
    assert build.returncode == 0, f"verilator build failed:\n{build.stderr}"
    binp = obj / "Vtb"
    run = subprocess.run([str(binp)], capture_output=True, text=True, timeout=60)
    return run.stdout + run.stderr


@requires_verilator
@pytest.mark.requires_nix
def test_cs_mem_behav_stores_and_reads(tmp_path):
    # BEHAV: the write must be visible on the read port -> non-zero (0xAB).
    out = _run_cs_mem_tb(tmp_path, "BEHAV", "8'hAB")
    assert "TBPASS" in out, out
    assert "TBFAIL" not in out


@requires_verilator
@pytest.mark.requires_nix
def test_cs_mem_macro_is_empty_shell(tmp_path):
    # MACRO: the empty cs_mem_macro_shell drives 0, so the same write reads 0.
    out = _run_cs_mem_tb(tmp_path, "MACRO", "8'h00")
    assert "TBPASS" in out, out
    assert "TBFAIL" not in out


@requires_verilator
@pytest.mark.requires_nix
def test_cs_mem_lib_lints_clean(tmp_path):
    # The whole lib must elaborate for each public cell without %Error.
    for top in ("cs_mem_1rw", "cs_mem_1rw1r", "cs_sram_1rw1r", "cs_fpmem_1rw1r"):
        r = subprocess.run(
            ["verilator", "--lint-only", "-Wall", "-Wno-fatal",
             "-Wno-DECLFILENAME", "-Wno-PINCONNECTEMPTY",
             "--top-module", top, str(_LIB)],
            capture_output=True, text=True, timeout=120,
        )
        assert "%Error" not in r.stderr, f"{top}:\n{r.stderr}"


# ---------------------------------------------------------------------------
# (b) dedup: rtl_lib body of a cs_sram_* cell survives an empty chip_top stub
# ---------------------------------------------------------------------------

# An empty (* blackbox *) cs_sram_1rw1r the Integration Lead authored into
# chip_top. The deduper must STRIP this and keep the real rtl_lib body.
_CHIP_TOP_WITH_STUB = """
module chip_top(input clk);
  // ... instantiates blocks ...
endmodule

(* blackbox *)
module cs_sram_1rw1r #(parameter WIDTH=32, parameter DEPTH=512)(
  input clk
);
endmodule
"""


def test_dedup_keeps_rtl_lib_body_over_chiptop_stub(tmp_path):
    # chip_top (non-lib) is ordered FIRST -- the old first-wins rule would have
    # kept its empty stub and stripped the real lib body. The lib-authoritative
    # rule must invert that: lib body survives, chip_top stub stripped.
    top = tmp_path / "chip_top.v"
    top.write_text(_CHIP_TOP_WITH_STUB)
    out = _dedup_module_sources([str(top), _LIB], tmp_path)

    # Exactly one cs_sram_1rw1r definition survives, and it is the rtl_lib body.
    assert _defs(out, "cs_sram_1rw1r") == 1
    # The surviving copy comes from the lib source (the lib file is passed
    # through unchanged because nothing in it is stripped).
    assert _LIB in out
    # The chip_top copy is the one that got stripped.
    top_out = [p for p in out if p != _LIB][0]
    assert _defs([top_out], "cs_sram_1rw1r") == 0
    assert "[coresmith dedup] removed library cell" in Path(top_out).read_text()
    # The real behavioral body (via cs_mem) is still reachable in the kept lib.
    lib_txt = Path(_LIB).read_text()
    assert "module cs_mem_1rw1r" in lib_txt


def test_dedup_keeps_rtl_lib_body_regardless_of_order(tmp_path):
    # Same as above but lib FIRST -- result is identical: one lib body kept.
    top = tmp_path / "chip_top.v"
    top.write_text(_CHIP_TOP_WITH_STUB)
    out = _dedup_module_sources([_LIB, str(top)], tmp_path)
    assert _defs(out, "cs_sram_1rw1r") == 1
    top_out = [p for p in out if p != _LIB][0]
    assert _defs([top_out], "cs_sram_1rw1r") == 0


# ---------------------------------------------------------------------------
# (c) Integration Lead postcondition: defining a cs_* primitive raises
# ---------------------------------------------------------------------------

def test_postcondition_raises_on_defined_memory_primitive():
    bad = _CHIP_TOP_WITH_STUB
    err = assert_no_memory_primitive_defined(bad)
    assert err is not None
    assert "cs_sram_1rw1r" in err
    assert "library cell" in err


def test_postcondition_passes_when_only_instantiated():
    # Instantiating (not defining) a cs_* cell is allowed.
    good = (
        "module chip_top(input clk);\n"
        "  cs_sram_1rw1r #(.WIDTH(8),.DEPTH(512)) u_mem (.clk(clk));\n"
        "  cs_fpmem_1rw #(.WIDTH(8),.DEPTH(16)) u_fp (.clk(clk));\n"
        "endmodule\n"
    )
    assert assert_no_memory_primitive_defined(good) is None


def test_postcondition_ignores_comment_mentions():
    # A cs_mem_* name appearing only in a comment must not trip the check.
    good = (
        "// this top uses cs_sram_1rw1r from rtl_lib\n"
        "/* module cs_mem_1rw1r would be wrong to define here */\n"
        "module chip_top(input clk);\n"
        "  cs_sram_1rw1r #(.WIDTH(8),.DEPTH(512)) u (.clk(clk));\n"
        "endmodule\n"
    )
    assert assert_no_memory_primitive_defined(good) is None


def test_postcondition_none_on_empty():
    assert assert_no_memory_primitive_defined("") is None
