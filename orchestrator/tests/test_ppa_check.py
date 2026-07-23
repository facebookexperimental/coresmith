# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the deterministic post-synthesis PPA gate (ppa_check).

The gate is the *judge* (deterministic thresholds); the synth_fixer LLM
stays the *fixer*. It compares the synthesized result against the uArch
storage budget (flip_flop_budget) + an optional area budget + pre-layout
WNS, and only flags DIVERGENCE from the spec's intent -- so a memory the
uArch deliberately kept as flops (no available sky130 macro fits) is not
flagged.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from orchestrator.langgraph.ppa_check import (
    PpaVerdict,
    count_ff_bits_from_stat,
    count_flops_from_stat,
    evaluate_ppa,
    parse_ff_budget,
    parse_mem_from_stat,
    parse_sta_report,
    run_pre_layout_sta,
    strip_instance_parameters,
    strip_signed_declaration_qualifiers,
)


class TestParseFfBudget:
    def test_extracts_approx_form(self):
        assert parse_ff_budget("flip_flop_budget ≈ 1200 FF") == 1200

    def test_extracts_colon_form(self):
        assert parse_ff_budget("- `flip_flop_budget`: 3,400 flip-flops") == 3400

    def test_returns_none_when_absent(self):
        assert parse_ff_budget("no budget mentioned here") is None


class TestCountFlopsFromStat:
    SKY130_STAT = """
=== block ===
     1234   sky130_fd_sc_hd__dfxtp_1
       56   sky130_fd_sc_hd__dfrtp_2
        8   sky130_fd_sc_hd__dlxtp_1
      900   sky130_fd_sc_hd__nand2_1
"""
    GENERIC_STAT = """
     320   $sdffe
      16   $dff
       4   $dffsr
     128   $mux
"""

    def test_counts_sky130_flops_excludes_latches_and_logic(self):
        # 1234 dfxtp + 56 dfrtp = 1290; dlxtp is a LATCH (dl, not df), nand2 is logic
        assert count_flops_from_stat(self.SKY130_STAT) == 1290

    def test_counts_generic_dff_family(self):
        # 320 + 16 + 4 = 340; mux excluded
        assert count_flops_from_stat(self.GENERIC_STAT) == 340

    # yosys 0.33 prints "<cellname>  <count>" (count LAST), unlike 0.65's
    # "<count>  <cellname>". The parser must handle both, and never read the
    # digits inside a name like $mem_v2.
    YOSYS_033_STAT = """
   Number of cells:                346
     $mux                          290
     $mem_v2                         2
     $sdff                           3
     $sdffe                          9
"""

    def test_counts_name_first_yosys033_format(self):
        # 3 + 9 = 12 flops; $mux and $mem_v2 excluded (and the "2" inside
        # $mem_v2 must not be mistaken for a count).
        assert count_flops_from_stat(self.YOSYS_033_STAT) == 12

    # Real Yosys output interleaves the `stat` cell table with LOG lines from
    # passes like opt_dff -- those mention "$dff" AND carry stray integers (bit
    # positions), and the old parser summed them as phantom flops (the codec
    # run saw a true ~1,178-flop block reported as ~279k, falsely tripping the
    # PPA gate). Only the 2-token cell-table lines are real flop counts.
    NOISY_REAL_OUTPUT = """
2.49. Printing statistics.

=== stripe_macroblock_buffer ===

   Number of wires:               4096
   Number of cells:               1378
     sky130_fd_sc_hd__dfxtp_1       1178
     sky130_sram_1kbyte_1rw1r_8x1024_8           5
     sky130_fd_sc_hd__nand2_1        200
"""
    OPT_DFF_LOG_NOISE = """
Setting constant 1-bit at position 47 on $procdff$1234 ($dff) from module top.
Setting constant 1-bit at position 511 on $procmux$5678 ($dff) from module top.
Removing init bit 256 of FF cell pixel_block_q ($dff) in module codec.
"""

    def test_ignores_opt_dff_log_noise(self):
        # Only the stat table's 1178 dfxtp flops are real; the bit-position
        # integers (47 + 511 + 256) in the opt_dff LOG lines must be ignored.
        # The SRAM macro line (5) is not a flop cell, so it is excluded too.
        assert count_flops_from_stat(
            self.NOISY_REAL_OUTPUT + self.OPT_DFF_LOG_NOISE
        ) == 1178

    def test_mem_count_name_first_excludes_name_digits(self):
        out = parse_mem_from_stat(self.YOSYS_033_STAT)
        assert out["mem_count"] == 2  # the trailing count, not the "2" in _v2


class TestParseMemFromStat:
    SUMMARY = """
   Number of memory bits:         32768
       3   $mem_v2
     136   $sdffe
"""
    NO_SUMMARY = """
       2   $mem_v2
     900   $nand
"""

    def test_counts_mem_cells_and_bits(self):
        out = parse_mem_from_stat(self.SUMMARY)
        assert out["mem_count"] == 3       # from the $mem_v2 cell line
        assert out["mem_bits"] == 32768    # from the bits summary line

    def test_counts_mem_cells_without_bits_line(self):
        out = parse_mem_from_stat(self.NO_SUMMARY)
        assert out["mem_count"] == 2
        assert out["mem_bits"] == 0

    def test_memory_preserving_ff_excludes_memory(self):
        # The whole point: a memory-preserving stat reports the big array as
        # $mem (bits), and count_flops_from_stat counts ONLY logic flops --
        # so a correctly-inferred SRAM scratchpad does NOT inflate the FF count.
        assert count_flops_from_stat(self.SUMMARY) == 136
        assert parse_mem_from_stat(self.SUMMARY)["mem_bits"] == 32768


class TestCountFfBitsFromStat:
    # `stat -width` breaks generic FFs out by port width: a 32-bit register is
    # ONE $dff cell but 32 flip-flop BITS (rung3-fixes-1, minor 5).
    WIDTH_STAT = """
        1   $dff_32
        1   $dff_8
"""

    def test_sums_bit_widths_not_cell_counts(self):
        # 32 + 8 = 40 BITS -- count_flops_from_stat would return 2 (word-level).
        assert count_ff_bits_from_stat(self.WIDTH_STAT) == 40
        assert count_flops_from_stat(self.WIDTH_STAT) == 2

    def test_one_bit_flop(self):
        assert count_ff_bits_from_stat("        1   $dff_1\n") == 1

    def test_mixed_flop_family_widths(self):
        stat = "        2   $sdffe_16\n        1   $dffsr_4\n"
        assert count_ff_bits_from_stat(stat) == 2 * 16 + 4

    def test_excludes_memory_and_logic(self):
        stat = "        1   $mem_v2\n        1   $mux_8\n        1   $dff_16\n"
        assert count_ff_bits_from_stat(stat) == 16

    def test_degrades_to_word_count_without_width_suffix(self):
        # A stat WITHOUT -width (plain $dff family) -> 1 bit each == word count.
        generic = "     320   $sdffe\n      16   $dff\n       4   $dffsr\n"
        assert count_ff_bits_from_stat(generic) == 340
        assert count_flops_from_stat(generic) == 340

    def test_sky130_drive_strength_not_mistaken_for_width(self):
        # The trailing `_2` is a DRIVE STRENGTH on a 1-bit Sky130 flop cell,
        # NOT a width -- 3 cells == 3 bits (only $-prefixed generic cells carry
        # a width suffix).
        stat = "        3   sky130_fd_sc_hd__dfxtp_2\n"
        assert count_ff_bits_from_stat(stat) == 3


class TestProbeSynthGeneric:
    GOOD = (
        "module tiny(input clk, input d, output reg q);\n"
        "  always @(posedge clk) q <= d;\n"
        "endmodule\n"
    )
    WIDE = (
        "module wide(input clk, input [31:0] d, output reg [31:0] q);\n"
        "  always @(posedge clk) q <= d;\n"
        "endmodule\n"
    )

    @pytest.mark.skipif(not shutil.which("yosys"), reason="yosys not installed")
    def test_good_rtl_elaborates_with_logic_ff(self, tmp_path):
        from orchestrator.langgraph.ppa_check import probe_synth_generic
        f = tmp_path / "tiny.v"
        f.write_text(self.GOOD)
        p = probe_synth_generic(str(f), "tiny", timeout_s=60)
        assert p is not None and p["elaborated"] is True
        assert p["logic_ff"] == 1

    @pytest.mark.skipif(not shutil.which("yosys"), reason="yosys not installed")
    def test_logic_ff_is_bit_level(self, tmp_path):
        # A 32-bit register is 32 flip-flop BITS (minor 5) -- matching the
        # bit-level flip_flop_budget, not a word-level count of 1.
        from orchestrator.langgraph.ppa_check import probe_synth_generic
        f = tmp_path / "wide.v"
        f.write_text(self.WIDE)
        p = probe_synth_generic(str(f), "wide", timeout_s=60)
        assert p is not None and p["elaborated"] is True
        assert p["logic_ff"] == 32

    @pytest.mark.skipif(not shutil.which("yosys"), reason="yosys not installed")
    def test_missing_top_is_not_elaborated_not_none(self, tmp_path):
        # An un-elaboratable design is a synthesizability signal (fail), not
        # "cannot judge" (None).
        from orchestrator.langgraph.ppa_check import probe_synth_generic
        f = tmp_path / "tiny.v"
        f.write_text(self.GOOD)
        p = probe_synth_generic(str(f), "does_not_exist", timeout_s=60)
        assert p is not None and p["elaborated"] is False

    def test_absent_rtl_returns_none(self):
        from orchestrator.langgraph.ppa_check import probe_synth_generic
        assert probe_synth_generic("/no/such/file.v", "x") is None


class TestEvaluatePpa:
    def test_within_budget_passes(self):
        v = evaluate_ppa(actual_ff=1100, ff_budget=1200)
        assert isinstance(v, PpaVerdict)
        assert v.ok is True

    def test_ff_far_over_budget_fails_with_reason(self):
        # 33,000 flops vs a 1,200 budget = the memory-as-flops signature
        v = evaluate_ppa(actual_ff=33000, ff_budget=1200)
        assert v.ok is False
        assert any("flip-flop" in r.lower() for r in v.reasons)

    def test_ff_modestly_over_within_tolerance_passes(self):
        # 1320 vs 1200 with default 25% tolerance (limit 1500) passes
        v = evaluate_ppa(actual_ff=1320, ff_budget=1200)
        assert v.ok is True

    def test_small_block_over_budget_but_under_floor_passes(self):
        # The real case: an 8-deep FIFO budgeted at 96 FF synthesized to 507.
        # 5x over budget, but 507 < 2000-FF floor -> negligible area, a code
        # nit, NOT a should-be-SRAM disaster. Must NOT stall the pipeline.
        v = evaluate_ppa(actual_ff=507, ff_budget=96)
        assert v.ok is True

    def test_big_memory_as_flops_still_fails_above_floor(self):
        # The disaster the gate exists for: a 4 KiB scratchpad as ~33k flops.
        v = evaluate_ppa(actual_ff=33000, ff_budget=1200)
        assert v.ok is False

    def test_negative_wns_fails(self):
        v = evaluate_ppa(actual_ff=100, ff_budget=1200, wns_ns=-4.5)
        assert v.ok is False
        assert any("slack" in r.lower() or "timing" in r.lower() for r in v.reasons)

    def test_positive_wns_passes(self):
        v = evaluate_ppa(actual_ff=100, ff_budget=1200, wns_ns=0.8)
        assert v.ok is True

    def test_area_over_budget_fails(self):
        v = evaluate_ppa(actual_ff=100, ff_budget=1200,
                         actual_area_um2=2_000_000.0, area_budget_um2=300_000.0)
        assert v.ok is False

    def test_no_budget_cannot_judge_passes(self):
        # No budget + a modest FF count -> the gate can't judge -> must not
        # block. (A no-budget block OVER the absolute hard ceiling IS blocked --
        # see test_hard_ceiling_fires_with_no_budget.)
        v = evaluate_ppa(actual_ff=1500, ff_budget=None)
        assert v.ok is True
        # A-Fix 2e: "cannot judge" is now recorded, not silently swallowed.
        assert any(u["metric"] == "flip_flop_count" for u in v.unmeasured)

    def test_full_budget_and_measurement_records_no_unmeasured_ff(self):
        # Both budget and measurement present -> FF is judged, not "unmeasured".
        v = evaluate_ppa(actual_ff=1100, ff_budget=1200)
        assert not any(u["metric"] == "flip_flop_count" for u in v.unmeasured)

    def test_storage_ff_separated_from_logic_ff(self):
        # A legit 144-byte (1152-bit) line buffer flopped, on top of ~1100 logic
        # flops = 2252 total, over a 1200 budget's 1500 limit. Separating the
        # STORAGE flops out leaves logic_ff=1100 <= 1500 -> PASS. This is the
        # false-flag on a legitimate buffer that got the gate disabled.
        v = evaluate_ppa(actual_ff=2252, ff_budget=1200, storage_ff=1152)
        assert v.ok is True
        ff = [c for c in v.checks if c["metric"] == "flip_flop_count"][0]
        assert ff["actual"] == 1100 and ff["storage_ff"] == 1152
        assert ff["total_ff"] == 2252

    def test_logic_ff_check_stays_strict_after_separation(self):
        # Storage separated, but the LOGIC flops still blow the budget -> FAIL.
        # The separation must not weaken the logic-FF check.
        v = evaluate_ppa(actual_ff=35000, ff_budget=1200, storage_ff=1000)
        assert v.ok is False
        assert any("logic flip-flop" in r.lower() for r in v.reasons)

    def test_storage_ff_does_not_defeat_hard_ceiling(self):
        # Total flops over the absolute hard ceiling still fail even if a chunk
        # is classified as storage (a should-be-SRAM explosion is caught).
        v = evaluate_ppa(actual_ff=136103, ff_budget=None, storage_ff=2000)
        assert v.ok is False
        assert any(c["metric"] == "flip_flop_hard_ceiling" for c in v.checks)


class TestTimingGateActionable:
    """item 3: a negative-slack verdict must be ACTIONABLE re-pipeline feedback
    (which path is over + by how much + stages to add) and a GROSS violation
    must not be silently ignorable even in advisory mode."""

    def test_wns_reason_quantifies_repipeline_with_period(self):
        v = evaluate_ppa(actual_ff=100, ff_budget=1200,
                         wns_ns=-7.31, period_ns=20.0)
        assert v.ok is False
        r = " ".join(v.reasons).lower()
        assert "re-pipeline" in r
        assert "27.31" in r          # path delay = 20 + 7.31
        assert "stage" in r
        wns = [c for c in v.checks if c["metric"] == "wns_ns"][0]
        assert wns["period_ns"] == 20.0

    def test_gross_violation_not_silently_ignorable_under_advisory(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_PPA_TIMING_ADVISORY", "1")
        # -7.31 ns on a 20 ns period = 37% over -> GROSS -> must still FAIL even
        # though the timing sub-check is set to advisory.
        v = evaluate_ppa(actual_ff=100, ff_budget=1200,
                         wns_ns=-7.31, period_ns=20.0)
        assert v.ok is False
        wns = [c for c in v.checks if c["metric"] == "wns_ns"][0]
        assert wns["gross"] is True and wns["passed"] is False

    def test_marginal_violation_is_advisory_under_advisory(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_PPA_TIMING_ADVISORY", "1")
        # -0.5 ns on a 20 ns period = 2.5% over (< 25% hard frac) -> marginal ->
        # advisory downgrades it (record, do NOT block).
        v = evaluate_ppa(actual_ff=100, ff_budget=1200,
                         wns_ns=-0.5, period_ns=20.0)
        assert v.ok is True
        wns = [c for c in v.checks if c["metric"] == "wns_ns"][0]
        assert wns["advisory"] is True and wns["gross"] is False


def test_parse_sdc_period_ns():
    from orchestrator.langgraph.ppa_check import parse_sdc_period_ns
    assert parse_sdc_period_ns(
        "create_clock -name clk -period 20 [get_ports clk]\n") == 20.0
    assert parse_sdc_period_ns(
        "create_clock -period 12.5 [get_ports c1]\n"
        "create_clock -period 8.0 [get_ports c2]\n") == 8.0  # tightest wins
    assert parse_sdc_period_ns("no clock here") is None


class TestParseStaReport:
    REPORT = """
Startpoint: ...
wns -3.42
tns -57.10
"""

    def test_parses_wns_tns(self):
        out = parse_sta_report(self.REPORT)
        assert out["wns_ns"] == pytest.approx(-3.42)
        assert out["tns_ns"] == pytest.approx(-57.10)

    def test_clean_timing(self):
        out = parse_sta_report("wns 0.00\ntns 0.00\n")
        assert out["wns_ns"] == 0.0

    # pdk-fixes-1 Gap 2: OpenSTA 3.x prints the corner token ("wns max N").
    def test_parses_opensta_3x_max_form(self):
        out = parse_sta_report("wns max -3.42\ntns max -57.10\n")
        assert out["wns_ns"] == pytest.approx(-3.42)
        assert out["tns_ns"] == pytest.approx(-57.10)

    def test_parses_opensta_3x_clean(self):
        out = parse_sta_report("wns max 0.00\ntns max 0.00\n")
        assert out["wns_ns"] == 0.0
        assert out["tns_ns"] == 0.0

    def test_report_command_echo_is_not_matched(self):
        # A bare `report_wns` / `report_tns` echo (no number) must not match:
        # the \b word-boundary keeps `wns` inside `report_wns` from matching.
        out = parse_sta_report("report_wns\nreport_tns\n")
        assert out["wns_ns"] is None
        assert out["tns_ns"] is None


class TestStripInstanceParameters:
    """pdk-fixes-1 Gap 1(a): strip parameter-override lists from blackbox
    instantiations so OpenSTA's read_verilog can parse an SRAM-bearing netlist.
    """

    def test_multiline_yosys_param_block(self):
        # The exact shape yosys `write_verilog -noattr` emits for a
        # parameterized blackbox instance.
        src = (
            "  cs_sram_1rw #(\n"
            "    .ADDRW(32'sd8),\n"
            "    .DEPTH(32'sd256),\n"
            "    .WIDTH(32'sd32)\n"
            "  ) u_mem (\n"
            "    .addr(addr),\n"
            "    .clk(clk)\n"
            "  );\n"
        )
        out = strip_instance_parameters(src)
        assert "#(" not in out
        assert "cs_sram_1rw" in out
        assert "u_mem" in out
        # Ports (the actual connectivity STA needs) survive untouched.
        assert ".addr(addr)" in out
        assert ".clk(clk)" in out

    def test_single_line(self):
        out = strip_instance_parameters("foo #(.W(8)) inst (.a(a));")
        assert "#(" not in out
        assert "foo" in out and "inst" in out and ".a(a)" in out

    def test_nested_parens_in_param_value(self):
        out = strip_instance_parameters(
            "foo #(.W((8+1)*2), .X({1,2})) inst (.a(a));"
        )
        assert "#(" not in out
        assert ".a(a)" in out

    def test_string_with_parens_in_param(self):
        # A ')' or '(' inside a string literal must not unbalance the scan.
        out = strip_instance_parameters('m #(.INIT("a)b(c")) inst (.a(a));')
        assert "#(" not in out
        assert "inst" in out and ".a(a)" in out

    def test_module_declaration_not_stripped(self):
        # `module m #( ... ) ( ... )` -- ')' is followed by '(' (the port
        # list), not an instance name, so the declaration is preserved.
        src = "module m #(parameter W=8) (input a); endmodule"
        assert strip_instance_parameters(src) == src

    def test_assign_delay_not_stripped(self):
        # `assign #(2,3) y = a;` -- '#(...)' is not followed by `ident (`.
        src = "assign #(2,3) y = a;"
        assert strip_instance_parameters(src) == src

    def test_plain_delay_not_touched(self):
        src = "bar #5 inst2 (.a(a));"
        assert strip_instance_parameters(src) == src

    def test_no_params_unchanged(self):
        src = "foo inst (.a(a), .b(b));"
        assert strip_instance_parameters(src) == src

    def test_idempotent(self):
        src = "cs_sram_1rw #(.DEPTH(256), .WIDTH(32)) u_mem (.clk(clk));"
        once = strip_instance_parameters(src)
        assert strip_instance_parameters(once) == once
        assert "#(" not in once

    def test_multiple_instances(self):
        src = (
            "a #(.P(1)) i0 (.x(x));\n"
            "b #(.Q(2)) i1 (.y(y));\n"
        )
        out = strip_instance_parameters(src)
        assert "#(" not in out
        assert "i0" in out and "i1" in out
        assert ".x(x)" in out and ".y(y)" in out


class TestStripSignedDeclarationQualifiers:
    def test_structural_declarations_are_normalized(self):
        src = (
            "wire signed [8:0] difference;\n"
            "reg signed [7:0] saved;\n"
            "module top(input wire signed [3:0] a, "
            "output reg signed [3:0] y);\nendmodule\n"
        )
        out = strip_signed_declaration_qualifiers(src)
        assert "wire signed" not in out
        assert "reg signed" not in out
        assert "input wire signed" not in out
        assert "output reg signed" not in out
        assert "[8:0] difference" in out
        assert "[3:0] y" in out

    def test_comments_strings_calls_and_identifiers_are_preserved(self):
        src = (
            "// wire signed [7:0] comment_only;\n"
            "wire [7:0] assigned_value = $signed(raw);\n"
            "wire [7:0] signed_count;\n"
            "initial $display(\"wire signed remains in a string\");\n"
            "wire [7:0] \\signed ;\n"
            "/* output signed [3:0] also_a_comment; */\n"
        )
        assert strip_signed_declaration_qualifiers(src) == src

    def test_idempotent(self):
        src = "wire signed [17:0] current_dc_ext_w;\n"
        once = strip_signed_declaration_qualifiers(src)
        assert strip_signed_declaration_qualifiers(once) == once
        assert once == "wire  [17:0] current_dc_ext_w;\n"


class TestRunPreLayoutStaLoud:
    """pdk-fixes-1 Gap 1(b): STA measurement absence must be LOUD (dict with
    ``sta_error`` + WARNING), not a silent ``None``, when STA was actually
    attempted for a block that has a netlist. A genuinely absent ``sta`` binary
    or a missing input still returns ``None`` (STA could not be attempted).
    """

    def _inputs(self, tmp_path):
        nl = tmp_path / "n.v"
        nl.write_text(
            "module top(input clk);\n"
            "  wire signed [8:0] endpoint_difference;\n"
            "  cs_sram_1rw #(.DEPTH(256), .WIDTH(32)) u (.clk(clk));\n"
            "endmodule\n"
        )
        sdc = tmp_path / "n.sdc"
        sdc.write_text("create_clock -name clk -period 20 [get_ports clk]\n")
        lib = tmp_path / "l.lib"
        lib.write_text("library(x){}\n")
        return str(nl), str(sdc), str(lib)

    class _R:
        def __init__(self, stdout="", stderr="", rc=0):
            self.stdout, self.stderr, self.returncode = stdout, stderr, rc

    def test_success_and_params_stripped(self, tmp_path, monkeypatch):
        import orchestrator.langgraph.ppa_check as pc
        nl, sdc, lib = self._inputs(tmp_path)
        monkeypatch.setattr(pc.shutil, "which", lambda _x: "/usr/bin/true")
        seen = {}

        def fake_run(cmd, capture_output, text, timeout):
            tcl = Path(cmd[-1]).read_text()
            rv = [l.split(None, 1)[1].strip()
                  for l in tcl.splitlines() if l.startswith("read_verilog")][0]
            seen["netlist"] = Path(rv).read_text()
            return self._R(stdout="wns max -2.50\ntns max -10.00\n", rc=0)

        monkeypatch.setattr(pc.subprocess, "run", fake_run)
        res = run_pre_layout_sta(nl, sdc, lib, "top")
        assert res == {"wns_ns": -2.50, "tns_ns": -10.00}
        # The netlist STA actually read had its instance params stripped.
        assert "#(" not in seen["netlist"]
        assert "wire signed" not in seen["netlist"]

    def test_unparseable_is_loud(self, tmp_path, monkeypatch, caplog):
        import logging
        import orchestrator.langgraph.ppa_check as pc
        nl, sdc, lib = self._inputs(tmp_path)
        monkeypatch.setattr(pc.shutil, "which", lambda _x: "/usr/bin/true")
        monkeypatch.setattr(
            pc.subprocess, "run",
            lambda *a, **k: self._R(stdout="Error: link failed",
                                    stderr="cannot find cs_sram_1rw", rc=1),
        )
        with caplog.at_level(logging.WARNING):
            res = run_pre_layout_sta(nl, sdc, lib, "top")
        assert res["wns_ns"] is None and res["tns_ns"] is None
        assert res["sta_error"]
        assert any("no timing" in r.message for r in caplog.records)

    def test_timeout_is_loud(self, tmp_path, monkeypatch):
        import orchestrator.langgraph.ppa_check as pc
        nl, sdc, lib = self._inputs(tmp_path)
        monkeypatch.setattr(pc.shutil, "which", lambda _x: "/usr/bin/true")

        def boom(*a, **k):
            raise pc.subprocess.TimeoutExpired(cmd="sta", timeout=300)

        monkeypatch.setattr(pc.subprocess, "run", boom)
        res = run_pre_layout_sta(nl, sdc, lib, "top")
        assert res["wns_ns"] is None
        assert "timed out" in res["sta_error"]

    def test_absent_binary_returns_none(self, tmp_path, monkeypatch):
        import orchestrator.langgraph.ppa_check as pc
        nl, sdc, lib = self._inputs(tmp_path)
        monkeypatch.setattr(pc.shutil, "which", lambda _x: None)
        assert run_pre_layout_sta(nl, sdc, lib, "top") is None

    def test_missing_input_returns_none(self, tmp_path, monkeypatch):
        import orchestrator.langgraph.ppa_check as pc
        _nl, sdc, lib = self._inputs(tmp_path)
        monkeypatch.setattr(pc.shutil, "which", lambda _x: "/usr/bin/true")
        assert run_pre_layout_sta(
            str(tmp_path / "nope.v"), sdc, lib, "top"
        ) is None


class TestRouteAfterSynthPpaGate:
    """route_after_synth must preserve current behavior unless the gate is
    enabled (CLAUDE.md both-branch env-gating convention)."""

    def _state(self, **kw):
        base = {"synth_success": True, "ppa_ok": False}
        base.update(kw)
        return base

    def test_gate_off_compiles_is_done_even_if_over_budget(self, monkeypatch):
        monkeypatch.delenv("CORESMITH_PPA_GATE", raising=False)
        from orchestrator.langgraph.pipeline_graph import route_after_synth
        assert route_after_synth(self._state()) == "block_done"

    def test_gate_on_over_budget_routes_to_diagnose(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_PPA_GATE", "1")
        from orchestrator.langgraph.pipeline_graph import route_after_synth
        assert route_after_synth(self._state(ppa_ok=False)) == "diagnose"

    def test_gate_on_within_budget_is_done(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_PPA_GATE", "1")
        from orchestrator.langgraph.pipeline_graph import route_after_synth
        assert route_after_synth(self._state(ppa_ok=True)) == "block_done"

    def test_synth_failure_always_diagnoses(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_PPA_GATE", "1")
        from orchestrator.langgraph.pipeline_graph import route_after_synth
        assert route_after_synth(self._state(synth_success=False)) == "diagnose"


# --- Hardened gate: real-FF judgment + absolute hard ceiling (codec mb_emitter) ---

def test_hard_ceiling_fires_with_no_budget():
    # 136k FF (mb_emitter) with NO flip_flop_budget -> hard ceiling blocks it
    from orchestrator.langgraph.ppa_check import evaluate_ppa
    v = evaluate_ppa(actual_ff=136103, ff_budget=None)
    assert not v.ok
    assert any(c["metric"] == "flip_flop_hard_ceiling" for c in v.checks)


def test_hard_ceiling_fires_over_small_budget():
    from orchestrator.langgraph.ppa_check import evaluate_ppa
    v = evaluate_ppa(actual_ff=136103, ff_budget=3000)
    assert not v.ok


def test_macro_backed_block_passes():
    # recon_store: real synth FF low because the macro is a blackbox instance
    from orchestrator.langgraph.ppa_check import evaluate_ppa
    v = evaluate_ppa(actual_ff=1546, ff_budget=2000)
    assert v.ok


def test_hard_ceiling_does_not_override_deliberate_large_budget():
    # a block whose uArch deliberately budgets large flops (no macro fits) and
    # is within that budget must NOT be hard-failed
    from orchestrator.langgraph.ppa_check import evaluate_ppa
    v = evaluate_ppa(actual_ff=58000, ff_budget=60000)
    assert v.ok
    assert not any(c["metric"] == "flip_flop_hard_ceiling" for c in v.checks)


class TestYosys066LibertyStat:
    """armD defect #7: yosys 0.66 liberty-aware stat prints THREE columns
    (count area cellname) and step-log wrappers append === STDERR === headers
    AFTER the tables -- both defeated the FF counters (read 0 against a
    4,595-flop netlist, leaving the FF-budget gate silently blind)."""

    _REAL_FORMAT = """=== SYNTHESIZE LOG ===
=== STDOUT ===
=== intra_rd_encode_core ===

    22375 2.14E+05 cells
     2618 5.24E+04   sky130_fd_sc_hd__dfxtp_1
     1977 5.94E+04   sky130_fd_sc_hd__edfxtp_1
      107  803.27    sky130_fd_sc_hd__a211oi_1
=== STDERR ===
some stderr noise
"""

    def test_three_column_stat_counts_bits(self):
        from orchestrator.langgraph.ppa_check import count_ff_bits_from_stat
        assert count_ff_bits_from_stat(self._REAL_FORMAT) == 4595

    def test_three_column_stat_counts_cells(self):
        from orchestrator.langgraph.ppa_check import count_flops_from_stat
        assert count_flops_from_stat(self._REAL_FORMAT) == 4595

    def test_wrapper_headers_after_table_do_not_blind(self):
        from orchestrator.langgraph.ppa_check import count_ff_bits_from_stat
        # the last === marker is STDERR with no cells; candidates must fall
        # back to the real table section
        assert count_ff_bits_from_stat(self._REAL_FORMAT) > 0
