# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the generic SRAM-wrapper gate (sram_wrapper.py)."""
import os

import pytest

from orchestrator.langgraph import sram_wrapper as sw


BIG_RAW = """
module myblock (input clk);
    reg [7:0] recon [0:235519];   // 1.88 Mbit -- should be a macro
    reg [31:0] small [0:7];       // tiny, fine as flops
endmodule
"""

WRAPPED = """
module myblock (input clk);
    cs_sram_1rw #(.WIDTH(8), .DEPTH(235520)) u_recon (
        .clk(clk), .ce(ce), .we(we), .addr(a), .wdata(d), .rdata(q));
endmodule
"""

MACRO_MODEL = """
module sky130_sram_2kbyte_1rw1r_32x512_8 (input clk);
    reg [31:0] mem [0:511];   // the macro's OWN behavioral model -> allowed
endmodule
module openram_sky130_64kbyte_1rw1r_32x16384_8 (input clk);
    reg [31:0] mem [0:16383];
endmodule
"""

WIDE_SHALLOW = """
module myblock (input clk);
    reg [3554:0] fifo [0:1];   // 3555b wide x 2 deep -> now FLAGGED by `wide` (3555 > 128)
endmodule
"""


def test_flags_large_raw_array():
    v = sw.detect_unwrapped_memories(BIG_RAW)
    assert len(v) == 1
    assert v[0].name == "recon"
    assert v[0].bits == 8 * 235520
    assert v[0].depth == 235520


def test_small_array_allowed():
    # the 8-deep reg in BIG_RAW must not be flagged
    names = [u.name for u in sw.detect_unwrapped_memories(BIG_RAW)]
    assert "small" not in names


def test_wrapped_passes():
    assert sw.detect_unwrapped_memories(WRAPPED) == []
    ok, reasons = sw.gate_memory_wrapping(WRAPPED)
    assert ok and reasons == []
    assert sw.uses_wrapper(WRAPPED)


def test_macro_models_excluded():
    # a reg array inside an SRAM macro's own sim-model is NOT a violation
    assert sw.detect_unwrapped_memories(MACRO_MODEL) == []


def test_wide_but_shallow_now_flagged():
    # REVERSAL: 3555b wide x 2 deep was previously EXEMPT (depth-2 -> "just
    # registers"). Under the aggressive OR policy it is FLAGGED by `wide` (3555 >
    # 128): a 3555-bit word is a huge unregistered read/write bus. The fixer
    # resolves it as a registered cs_fpmem or a restructure -- NOT necessarily an
    # SRAM macro -- so this is a guardrail, not a deadlock.
    v = sw.detect_unwrapped_memories(WIDE_SHALLOW)
    assert len(v) == 1 and v[0].width == 3555 and v[0].depth == 2


def test_borderline_accumulator_now_flagged():
    # REVERSAL: quant_scan-style 16b x 256 = 4096 bits was previously EXEMPT under
    # the 16384-bit threshold. Under the aggressive policy it is FLAGGED by `big`
    # (4096 > 2000). The fixer registers it as cs_fpmem (a flop array is fine, it
    # just has to capture the read) or restructures it -- NOT forced into SRAM.
    rtl = "module enc(input clk); reg [15:0] quant_scan [0:255]; endmodule"
    v = sw.detect_unwrapped_memories(rtl)
    assert len(v) == 1 and v[0].name == "quant_scan"
    # a genuinely large memory still trips it too
    big = "module enc(input clk); reg [31:0] buf [0:1023]; endmodule"  # 32768b, 1024 deep
    assert len(sw.detect_unwrapped_memories(big)) == 1


def test_default_thresholds():
    assert sw.min_bits() == 2000 and sw.min_width() == 128 and sw.macro_depth() == 256


def test_deep_narrow_array_flagged_by_depth():
    # 8b x 640 = 5120 bits, depth 640 (> 256): a 640:1 read mux -- flagged by
    # `deep` (and by `big`, 5120 > 2000). The exact class the old bits-AND-depth
    # gate let slip through.
    rtl = "module blk(input clk); reg [7:0] line [0:639]; assign q=line[a]; endmodule"
    v = sw.detect_unwrapped_memories(rtl)
    assert len(v) == 1 and v[0].depth == 640
    # a 1024-deep narrow array likewise
    assert len(sw.detect_unwrapped_memories("module b; reg [7:0] m [0:1023]; endmodule")) == 1


def test_parallel_accumulator_now_flagged():
    # REVERSAL: a fully parallel-read 16b x 256 = 4096-bit scan accumulator is now
    # FLAGGED by `big` (4096 > 2000). It is not a deadlock: the fixer registers it
    # as cs_fpmem (a flop array is fine -- just capture the read), never SRAM-only.
    v = sw.detect_unwrapped_memories("module e; reg [15:0] scan [0:255]; endmodule")
    assert len(v) == 1 and v[0].name == "scan"


def test_cs_fpmem_recognized_as_wrapper():
    blk = ("module blk(input clk); cs_fpmem_1rw #(.WIDTH(8),.DEPTH(64)) "
           "u(.clk(clk),.ce(ce),.we(we),.addr(a),.wdata(d),.rdata(q)); endmodule")
    assert sw.uses_wrapper(blk) and sw.detect_unwrapped_memories(blk) == []
    # the shared lib defining cs_fpmem/cs_sram must not flag its own internal mem[]
    lib = open(sw.wrapper_lib_path()).read()
    assert sw.detect_unwrapped_memories(lib) == []
    assert "module cs_fpmem_1rw" in lib and "module cs_fpmem_1rw1r" in lib


# --- SRAM area accounting (prices RAM into the PPA/area gate) -----------------

def test_sram_instances_and_bits():
    rtl = """
    module enc(input clk);
      cs_sram_1rw1r #(.WIDTH(8), .DEPTH(235520)) u_recon (.clk(clk));
      cs_sram_1rw  #(.DEPTH(512), .WIDTH(32)) u_scratch (.clk(clk));
    endmodule
    """
    insts = sw.sram_instances(rtl)
    assert (8, 235520) in insts and (32, 512) in insts
    assert sw.sram_bits(rtl) == 8 * 235520 + 32 * 512


def test_estimate_sram_area():
    rtl = "module e(input clk); cs_sram_1rw1r #(.WIDTH(8),.DEPTH(1048576)) u(.clk(clk)); endmodule"
    # 8.4 Mbit * 1.7 um^2/bit ~= 14.3 mm^2 -- the GDS-intractable spool
    area = sw.estimate_sram_area_um2(rtl)
    assert area == 8 * 1048576 * sw.um2_per_bit()
    assert area > 1.0e7  # > 10 mm^2


def test_estimate_sram_area_no_macro():
    assert sw.estimate_sram_area_um2("module e; reg [7:0] x; endmodule") == 0.0


def test_gate_blocks_raw_with_reason():
    ok, reasons = sw.gate_memory_wrapping(BIG_RAW)
    assert not ok
    assert len(reasons) == 1
    assert "cs_sram_1rw" in reasons[0]
    assert "235520" in reasons[0]


# Three single-trigger fixtures: each trips exactly ONE leg of the OR so the
# corresponding env knob controls it in isolation.
#   BITS_ONLY : 32b x 100 = 3200 bits, width 32 (<=128), depth 100 (<=256) -> big
#   WIDTH_ONLY: 256b x 4   = 1024 bits (<=2000), depth 4 (<=256)           -> wide
#   DEPTH_ONLY: 4b x 300   = 1200 bits (<=2000), width 4 (<=128)           -> deep
BITS_ONLY = "module myblock(input clk); reg [31:0] buf [0:99]; endmodule"
WIDTH_ONLY = "module myblock(input clk); reg [255:0] w [0:3]; endmodule"
DEPTH_ONLY = "module myblock(input clk); reg [3:0] col [0:299]; endmodule"


def test_bits_env(monkeypatch):
    assert len(sw.detect_unwrapped_memories(BITS_ONLY)) == 1   # flagged by `big`
    monkeypatch.setenv("CORESMITH_SRAM_MIN_BITS", "4000")      # raise above 3200
    assert sw.detect_unwrapped_memories(BITS_ONLY) == []


def test_min_width_env(monkeypatch):
    assert len(sw.detect_unwrapped_memories(WIDTH_ONLY)) == 1  # flagged by `wide`
    monkeypatch.setenv("CORESMITH_SRAM_MIN_WIDTH", "256")      # raise to width 256 (strict >)
    assert sw.detect_unwrapped_memories(WIDTH_ONLY) == []


def test_macro_depth_env(monkeypatch):
    assert len(sw.detect_unwrapped_memories(DEPTH_ONLY)) == 1  # flagged by `deep`
    monkeypatch.setenv("CORESMITH_SRAM_MACRO_DEPTH", "300")    # raise to depth 300 (strict >)
    assert sw.detect_unwrapped_memories(DEPTH_ONLY) == []


def test_each_or_condition_fires_independently():
    # each leg of the OR flags its array on its own (defaults: 2000/128/256)
    assert len(sw.detect_unwrapped_memories(BITS_ONLY)) == 1   # big only
    assert len(sw.detect_unwrapped_memories(WIDTH_ONLY)) == 1  # wide only
    assert len(sw.detect_unwrapped_memories(DEPTH_ONLY)) == 1  # deep only


def test_tiny_buffer_not_flagged():
    # 8 wide, 8 deep, 64 bits -- below ALL three thresholds -> not flagged.
    # (this change does NOT force small buffers into SRAM.)
    tiny = "module tiny(input clk); reg [7:0] buf [0:7]; endmodule"
    assert sw.detect_unwrapped_memories(tiny) == []


def test_resolve_macro_no_pdk_is_graceful():
    # find_exact with an empty registry -> None, never raises
    assert sw.resolve_macro(32, 512, registry={}) is None


def test_wrapper_lib_exists():
    p = sw.wrapper_lib_path()
    assert p.endswith("cs_sram.v") and os.path.exists(p)
    txt = open(p).read()
    assert "module cs_sram_1rw" in txt and "module cs_sram_1rw1r" in txt
    assert "CORESMITH_SRAM_SYNTH" in txt
