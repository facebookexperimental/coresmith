# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the functional-ifdef (split-brain conditional-compilation) lint.

This is the DEEPEST anti-gaming class the campaign found: a generated block that
carried TWO implementations in one module -- the real verified datapath under
``\\`ifndef SYNTHESIS`` and a non-functional latency-shell MOCK under ``\\`else``
whose only purpose was to satisfy the storage lint + synth probes. DV verified
one branch; every synth/backend gate built the other (different hardware).
"""
from __future__ import annotations

import pytest

from orchestrator.langgraph import rtl_storage_lint as il


# A minimized version of the ACTUAL rung-3 intra_rd_encode_core mock: the real
# datapath under `ifndef SYNTHESIS (always + assign) and a counter-theater MOCK
# under `else (coeff = in-128, memory read write-only, a cs_sram instance),
# BOTH inside the design module.
_SPLIT_BRAIN_MOCK = """
module intra_rd_encode_core(
    input  wire        clk,
    input  wire        rst_n,
    input  wire [7:0]  s_tdata,
    output reg  [7:0]  m_tdata
);
`ifndef SYNTHESIS
    // REAL verified datapath
    reg [7:0] coeff_q;
    always @(posedge clk) begin
        if (!rst_n) coeff_q <= 8'd0;
        else        coeff_q <= (s_tdata * 8'd3) + real_transform(s_tdata);
    end
    assign m_tdata = coeff_q;
    function [7:0] real_transform;
        input [7:0] x;
        real_transform = (x >> 1) + 8'd7;
    endfunction
`else
    // NON-FUNCTIONAL latency-shell mock
    localparam [3:0] STATE_IDLE = 4'd0, STATE_DELAY = 4'd1;
    reg [3:0] state_q;
    reg [7:0] coeff_q;
    wire [7:0] mem_rd;
    always @(posedge clk) begin
        if (!rst_n) begin state_q <= STATE_IDLE; coeff_q <= 8'd0; end
        else        coeff_q <= s_tdata - 8'd128;   // fake coeff, QP unused
    end
    assign m_tdata = coeff_q;
    cs_sram_1rw #(.WIDTH(8), .DEPTH(512)) u_mem (
        .clk(clk), .ce(1'b1), .we(1'b0), .addr(9'd0), .wdata(8'd0), .rdata(mem_rd)
    );
`endif
endmodule
"""


def test_rejects_two_branch_functional_module():
    r = il.find_functional_ifdef_regions(_SPLIT_BRAIN_MOCK)
    assert not r.ok, "a split-brain ifndef/else functional module must be rejected"
    # BOTH branches are flagged, both attributed to the design module.
    conds = {f.condition for f in r.findings}
    assert "ifndef SYNTHESIS" in conds and "else" in conds
    for f in r.findings:
        assert f.enclosing_module == "intra_rd_encode_core"
    # each branch guards real functional logic
    got = {c for f in r.findings for c in f.constructs}
    assert "always" in got and "assign" in got
    # the `else mock branch instantiates a memory macro
    else_f = next(f for f in r.findings if f.condition == "else")
    assert "instantiation" in else_f.constructs


def test_renamed_macro_still_caught():
    # agents will rename the guard away from SYNTHESIS -- the lint keys on
    # CONSTRUCT CLASSES, not macro names.
    src = """
    module blk(input clk, input [7:0] a, output [7:0] y);
    `ifdef FAST_MODEL
        assign y = a + 8'd5;
    `else
        assign y = a - 8'd128;
    `endif
    endmodule
    """
    r = il.find_functional_ifdef_regions(src)
    assert not r.ok
    assert {f.condition for f in r.findings} == {"ifdef FAST_MODEL", "else"}


def test_allows_assertion_only_region():
    src = """
    module blk(input clk, input [7:0] a, output reg [7:0] q);
      always @(posedge clk) q <= a;
    `ifdef FORMAL
      assert property (@(posedge clk) a != 8'hFF);
    `endif
    endmodule
    """
    r = il.find_functional_ifdef_regions(src)
    assert r.ok, "an assertion-only region must be allowed"


def test_allows_debug_waveform_trace_hook():
    # $dumpfile/$dumpvars + a $display-only always are debug hooks -> allowed,
    # regardless of the guard macro name (allowlist is construct classes).
    src = """
    module blk(input clk, input v, output reg [7:0] q);
      always @(posedge clk) q <= v ? 8'd1 : 8'd0;
    `ifdef COCOTB_SIM
      initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, blk);
      end
      always @(posedge clk) if (v) $display("v high @ %0t", $time);
    `endif
    endmodule
    """
    r = il.find_functional_ifdef_regions(src)
    assert r.ok, "debug/trace/$dumpvars hooks must be allowed"


def test_allows_macro_module_blackbox_behavioral_split():
    # The LEGITIMATE library idiom: a whole macro-named module has a synth
    # blackbox and a sim-behavioral pair. This is the pattern the integration
    # deduper preserves; it must NOT be flagged.
    src = """
    module chip_top(input clk); endmodule
    `ifdef SYNTHESIS
    (* blackbox *)
    module sram_macro(input clk); endmodule
    `else
    module sram_macro(input clk);
      reg [7:0] mem [0:255];
      always @(posedge clk) mem[0] <= 8'd1;
    endmodule
    `endif
    """
    r = il.find_functional_ifdef_regions(src)
    assert r.ok, "macro-module blackbox/behavioral split is the legit library idiom"


def test_rtl_lib_wrapper_file_is_exempt():
    # A wrapper LIBRARY file (rtl_lib/) is exempt outright, even with a
    # functional split -- the sim-body/synth-macro split is legitimate THERE.
    r = il.find_functional_ifdef_regions(_SPLIT_BRAIN_MOCK, is_library=True)
    assert r.ok


def test_no_conditional_compilation_passes():
    src = "module m(input c, output reg q); always @(posedge c) q <= ~q; endmodule"
    assert il.find_functional_ifdef_regions(src).ok


def test_comment_and_string_with_ifdef_not_tripped():
    # `ifdef inside a comment or string must not register as a directive.
    src = """
    module m(input clk, output reg [7:0] q);
      // `ifdef SYNTHESIS  -- this is a comment, not a directive
      always @(posedge clk) q <= 8'd1;
      initial $display("mode `ifdef SYNTHESIS text");
    endmodule
    """
    assert il.find_functional_ifdef_regions(src).ok


def test_nested_ifdef_functional_inner_region_caught():
    src = """
    module blk(input clk, input [7:0] a, output reg [7:0] y);
    `ifdef OUTER
      `ifdef INNER
        always @(posedge clk) y <= a + 8'd1;
      `endif
    `endif
    endmodule
    """
    r = il.find_functional_ifdef_regions(src)
    assert not r.ok
    assert any("always" in f.constructs for f in r.findings)


def test_format_report_actionable():
    r = il.find_functional_ifdef_regions(_SPLIT_BRAIN_MOCK)
    msg = il.format_ifdef_lint_report(r, block="intra_rd_encode_core")
    assert "intra_rd_encode_core" in msg
    assert "ONE implementation" in msg
    assert "DIFFERENT HARDWARE" in msg
    # clean input -> empty message
    assert il.format_ifdef_lint_report(
        il.find_functional_ifdef_regions("module m; endmodule")) == ""


def test_env_gate_default_on(monkeypatch):
    monkeypatch.delenv("CORESMITH_IFDEF_LINT", raising=False)
    assert il.ifdef_lint_enabled() is True


@pytest.mark.parametrize("val", ["0", "false", "no", "off"])
def test_env_gate_off(monkeypatch, val):
    monkeypatch.setenv("CORESMITH_IFDEF_LINT", val)
    assert il.ifdef_lint_enabled() is False


def test_env_gate_explicit_on(monkeypatch):
    monkeypatch.setenv("CORESMITH_IFDEF_LINT", "1")
    assert il.ifdef_lint_enabled() is True
