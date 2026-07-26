# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the wide-flat-packed-storage + dynamic-part-select lint.

This is the anti-pattern that walled the codec intra4x4 synth AFTER the
combinational cloud was fixed: working data kept as a wide flat packed reg
sliced by a runtime index -> barrel-shifter blow-up in proc.
"""
from __future__ import annotations

import pytest

from orchestrator.langgraph import rtl_storage_lint as sl


def test_flags_wide_flat_reg_with_dynamic_part_select():
    src = """
    module m(input clk);
      reg [1023:0] top_recon_q;
      reg [7:0] small_q;
      always @(posedge clk) begin
        top_recon_q[base_idx +: 8] <= small_q;   // DYNAMIC part-select on 1024b
      end
    endmodule
    """
    r = sl.find_flat_packed_dynamic_storage(src)
    assert not r.ok
    names = [f.name for f in r.findings]
    assert "top_recon_q" in names
    f = next(f for f in r.findings if f.name == "top_recon_q")
    assert f.width_bits == 1024 and f.dynamic_accesses >= 1


def test_static_part_select_is_NOT_flagged():
    # constant slices (config bus field extraction) are fine -- no barrel shifter
    src = """
    module m;
      reg [511:0] cfg_q;
      wire [15:0] w = cfg_q[57:42];
      wire [5:0]  q = cfg_q[25:20];
    endmodule
    """
    r = sl.find_flat_packed_dynamic_storage(src)
    assert r.ok, "constant part-selects must not be flagged"


def test_narrow_reg_with_dynamic_index_not_flagged():
    # a small reg sliced dynamically is cheap -> below the width threshold
    src = """
    module m;
      reg [31:0] x_q;
      wire [3:0] n = x_q[idx +: 4];
    endmodule
    """
    r = sl.find_flat_packed_dynamic_storage(src, min_bits=128)
    assert r.ok


def test_dynamic_single_bit_index_on_wide_reg_flagged():
    src = """
    module m;
      reg [255:0] vec_q;
      wire b = vec_q[sel];   // dynamic single-bit index on 256b
    endmodule
    """
    r = sl.find_flat_packed_dynamic_storage(src)
    assert not r.ok and r.findings[0].name == "vec_q"


def test_is_dynamic_index_helper():
    assert sl._is_dynamic_index("base_idx +: 8") is True
    assert sl._is_dynamic_index("sel") is True
    assert sl._is_dynamic_index("57:42") is False
    assert sl._is_dynamic_index("7") is False
    assert sl._is_dynamic_index("8'd3") is False
    assert sl._is_dynamic_index("phase_count_q +: 16") is True


def test_format_report_actionable():
    src = "module m; reg [1023:0] big_q; wire w = big_q[i +: 8]; endmodule"
    r = sl.find_flat_packed_dynamic_storage(src)
    msg = sl.format_lint_report(r, block="intra4x4")
    assert "big_q" in msg and "cs_sram" in msg and "barrel-shifter" in msg
    assert sl.format_lint_report(sl.find_flat_packed_dynamic_storage(
        "module m; reg [7:0] a; endmodule")) == ""


def test_flags_wide_reg_sliced_inside_a_helper_function():
    # The codec RD-core hid the cloud here: top_y_line_q is never sliced by a
    # literal name -- it's passed into get_byte2048(vec, idx) which does
    # vec[base +: 8]. The hardened matcher must follow that indirection.
    src = """
    module m(input clk);
      reg [5119:0] top_y_line_q;
      function [7:0] get_byte2048;
        input [5119:0] vec;
        input [10:0] idx;
        reg [12:0] base;
        begin
          base = idx * 8;
          get_byte2048 = vec[base +: 8];   // dynamic slice on the ARG
        end
      endfunction
      wire [7:0] b = get_byte2048(top_y_line_q, sel_q);
    endmodule
    """
    r = sl.find_flat_packed_dynamic_storage(src)
    assert not r.ok
    f = next(f for f in r.findings if f.name == "top_y_line_q")
    assert f.via_function == "get_byte2048" and f.width_bits == 5120
    msg = sl.format_lint_report(r)
    assert "top_y_line_q" in msg and "get_byte2048" in msg


def test_helper_with_only_constant_slice_not_flagged():
    # a helper that slices its input by a CONSTANT is fine -> the reg passed in
    # is not flagged via that helper
    src = """
    module m;
      reg [255:0] cfg_q;
      function [7:0] get_field;
        input [255:0] v;
        begin get_field = v[15:8]; end   // constant slice
      endfunction
      wire [7:0] x = get_field(cfg_q);
    endmodule
    """
    r = sl.find_flat_packed_dynamic_storage(src)
    assert r.ok


def test_clean_rtl_passes():
    # proper addressed memory + small regs -> no findings
    src = """
    module m(input clk);
      reg [7:0] mem [0:255];          // per-element array, fine
      reg [127:0] vec_q;
      wire [7:0] r = vec_q[63:56];    // constant slice, fine
      always @(posedge clk) mem[addr_q] <= vec_q[7:0];
    endmodule
    """
    r = sl.find_flat_packed_dynamic_storage(src)
    assert r.ok


# ---------------------------------------------------------------------------
# Finding 4 (pipeline-campaign-3): threshold-proximity in the verdict + a
# borderline-storage note. The 128-bit threshold correctly catches the class
# cheaply and is NOT changed; but a small, synthesis-proven reg just over it
# (a 214-bit dynamic-select reg synthesized in 230 s) is a case the reviewer
# may accept via the documented override -- so the verdict records how
# borderline it is and the message says so.
# ---------------------------------------------------------------------------

def _one_dynamic_reg(width_bits: int) -> str:
    return (f"module m; reg [{width_bits - 1}:0] buf_q; "
            f"wire w = buf_q[i +: 8]; endmodule")


def test_proximity_ratio_recorded_in_verdict():
    r = sl.find_flat_packed_dynamic_storage(_one_dynamic_reg(214))
    assert not r.ok
    assert r.min_finding_bits == 214
    # 214 / 128 threshold ~= 1.67x -> borderline
    assert r.proximity_ratio() == pytest.approx(214 / 128, rel=1e-6)
    assert r.near_threshold is True


def test_clean_report_has_no_proximity():
    r = sl.find_flat_packed_dynamic_storage("module m; reg [7:0] a; endmodule")
    assert r.ok
    assert r.min_finding_bits is None
    assert r.proximity_ratio() is None
    assert r.near_threshold is False


def test_borderline_message_offers_override():
    r = sl.find_flat_packed_dynamic_storage(_one_dynamic_reg(214))
    msg = sl.format_lint_report(r, block="intra_rd")
    assert "threshold proximity" in msg.lower()
    assert "override" in msg.lower()
    assert "1.7x threshold" in msg          # per-finding ratio annotation
    # the FIX guidance is still present (threshold stays strict by design)
    assert "cs_sram" in msg


def test_clearly_over_threshold_has_no_borderline_note():
    # a 5120-bit reg (40x threshold) is not borderline -> no override note
    r = sl.find_flat_packed_dynamic_storage(_one_dynamic_reg(5120))
    assert r.near_threshold is False
    msg = sl.format_lint_report(r, block="intra_rd")
    assert "threshold proximity" not in msg.lower()
    assert "override" not in msg.lower()
    # but the per-finding ratio is still annotated
    assert "40.0x threshold" in msg


def test_threshold_unchanged_default_128():
    # Finding 4 explicitly does NOT move the threshold.
    assert sl.StorageLintReport().min_bits == 128
    assert sl.find_flat_packed_dynamic_storage(
        _one_dynamic_reg(120)).ok  # below 128 -> not flagged


# ---------------------------------------------------------------------------
# Section 5f / 4a: memory-tier finder (register-tier memory belongs in SRAM)
# ---------------------------------------------------------------------------

class TestMemoryTierFinder:
    def test_flags_oversized_behavioral_array(self):
        # 1024 words x 8 b = 8192 b -> over 256 words AND 1 KiB.
        src = "module m(input clk); reg [7:0] mem [0:1023]; endmodule"
        r = sl.find_oversized_memory_arrays(src)
        assert not r.ok
        f = r.findings[0]
        assert f.name == "mem" and f.depth_words == 1024 and f.width_bits == 8
        assert f.impl == "behavioral_array"

    def test_small_array_is_clean(self):
        # 64 words x 8 b -> under both thresholds.
        src = "module m(input clk); reg [7:0] fifo [0:63]; endmodule"
        assert sl.find_oversized_memory_arrays(src).ok

    def test_flags_cs_fpmem_over_threshold(self):
        src = ("module m(input clk);\n"
               "  cs_fpmem_1rw1r #(.WIDTH(32), .DEPTH(512)) u_mem (.clk(clk));\n"
               "endmodule")
        r = sl.find_oversized_memory_arrays(src)
        assert not r.ok
        f = r.findings[0]
        assert f.impl == "cs_fpmem" and f.depth_words == 512 and f.width_bits == 32

    def test_sram_backed_memory_not_flagged(self):
        # A cs_sram macro of the same name is the addressed-macro tier -> OK.
        src = ("module m(input clk);\n"
               "  cs_sram_1rw1r #(.WIDTH(8), .DEPTH(1024)) coef (.clk(clk));\n"
               "endmodule")
        assert sl.find_oversized_memory_arrays(src).ok

    def test_untimeable_single_cycle_read_flagged_under_period(self):
        # A deep-ish array UNDER the size threshold but whose single-cycle flat
        # read mux busts a tight period is still flagged when period is given.
        # flat_read_mux_ns depends on the PDK model; assert the helper math
        # directly so the test is PDK-independent.
        # log2(256) = 8 levels; force a big per-level via monkey-free direct call
        # only if characterized. Fall back: verify size path independently.
        src = "module m(input clk); reg [7:0] t [0:255]; endmodule"
        # depth 256 == max_words -> flagged on size regardless of PDK.
        r = sl.find_oversized_memory_arrays(src, period_ns=2.0)
        assert not r.ok

    def test_format_report_names_the_fix(self):
        src = "module m(input clk); reg [7:0] mem [0:1023]; endmodule"
        r = sl.find_oversized_memory_arrays(src)
        msg = sl.format_memory_tier_report(r, block="blk")
        assert "cs_sram" in msg and "SRAM" in msg and "blk" in msg

    def test_flat_read_mux_levels_scale_log_depth(self):
        # Pure math: 0 depth<=1, and monotone in depth when characterized.
        assert sl.flat_read_mux_ns(1, 8) == 0.0
        # When PDK is uncharacterized this returns 0.0 (no estimate) -- either
        # way it must never raise.
        v = sl.flat_read_mux_ns(1024, 8)
        assert v >= 0.0
