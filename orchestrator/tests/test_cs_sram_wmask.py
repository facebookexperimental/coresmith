# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""cs_sram_1rw1r optional byte-mask (USE_WMASK): behavioral proof via
iverilog (skipped when the tool is absent) + static contract checks."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_RTL_DIR = Path(__file__).resolve().parent.parent / "langgraph" / "rtl_lib"
_TB = Path(__file__).resolve().parent / "rtl" / "tb_cs_sram_wmask.v"


def test_wmask_port_is_optional_in_source():
    src = (_RTL_DIR / "cs_sram.v").read_text()
    # both the pass-through and the impl mux expose the mask behind USE_WMASK=0
    assert src.count("USE_WMASK = 0") >= 2
    assert "wmask0" in src


@pytest.mark.skipif(shutil.which("iverilog") is None, reason="iverilog absent")
def test_wmask_behavior_sim(tmp_path):
    out = tmp_path / "tb.vvp"
    subprocess.run(
        ["iverilog", "-g2012", "-o", str(out),
         str(_TB), str(_RTL_DIR / "cs_sram.v")],
        check=True, capture_output=True, text=True,
    )
    run = subprocess.run(["vvp", str(out)], capture_output=True, text=True,
                         timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "WMASK_TB_PASS" in run.stdout, run.stdout + run.stderr
    assert "FPMEM_WMASK_TB_PASS" in run.stdout, run.stdout + run.stderr


# --- MEM_IMPL swap on the cs_sram wrappers (Part A) ---------------------------

def test_wrappers_expose_mem_impl_default_behav():
    """cs_sram_1rw / cs_sram_1rw1r now carry a MEM_IMPL parameter that DEFAULTS
    to "BEHAV" (so every existing instantiation is byte-for-byte unchanged) and
    pass it through to the underlying cs_mem_* -- no more hardwired
    .MEM_IMPL("BEHAV")."""
    src = (_RTL_DIR / "cs_sram.v").read_text()
    # both blessed wrappers declare the swappable parameter, default BEHAV
    assert src.count('parameter         MEM_IMPL = "BEHAV"') >= 1
    assert src.count('MEM_IMPL = "BEHAV"') >= 2
    # the pass-through no longer hardwires BEHAV into cs_mem_*
    assert ".MEM_IMPL(MEM_IMPL)" in src
    # the wmask default is preserved (existing contract)
    assert src.count("USE_WMASK = 0") >= 2


@pytest.mark.skipif(shutil.which("iverilog") is None, reason="iverilog absent")
def test_macro_impl_elaborates(tmp_path):
    """MEM_IMPL="MACRO" on the wrapper elaborates (selects the empty
    cs_mem_macro_shell leaf) without error -- the backend-swap path."""
    dut = tmp_path / "dut.v"
    dut.write_text(
        "module dut(input clk, input ce0, input we0, input [9:0] a0,\n"
        "  input [31:0] d0, output [31:0] q0, input ce1, input [9:0] a1,\n"
        "  output [31:0] q1);\n"
        "  cs_sram_1rw1r #(.MEM_IMPL(\"MACRO\"), .WIDTH(32), .DEPTH(1024)) u (\n"
        "    .clk(clk), .ce0(ce0), .we0(we0), .addr0(a0), .wdata0(d0),\n"
        "    .wmask0(4'b0), .rdata0(q0), .ce1(ce1), .addr1(a1), .rdata1(q1));\n"
        "endmodule\n"
    )
    r = subprocess.run(
        ["iverilog", "-g2012", "-o", str(tmp_path / "d.vvp"),
         str(_RTL_DIR / "cs_sram.v"), str(dut)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
