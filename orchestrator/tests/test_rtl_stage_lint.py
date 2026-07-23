# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the deterministic stage-realization lint (pipeline-campaign).

Covers all three deliverables:
  D1 -- the arithmetic census (cloud fixture FAILS w/ 171-amplification evidence;
        passing fixture PASSES; documented edge cases).
  D2 -- the module-per-stage structural check + protocol-not-applicable path.
  D3 -- directive-rich, trajectory-aware retry feedback + prompt pinning.

Hermetic: no LLM, no EDA toolchain, no network. Uses the checked-in RTL fixtures
(the operator-fixed rung-3 run2 negative gold + a clean run2 positive block).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from orchestrator.langgraph.rtl_stage_lint import (
    BlockCensus,
    StageMap,
    census_rtl,
    census_signature,
    check_stage_modules,
    format_stage_lint_report,
    load_stage_map,
    stage_lint_enabled,
    stage_map_from_budget,
    stage_modules_enabled,
)

_FIX = Path(__file__).resolve().parent / "fixtures" / "rtl"
_CLOUD = _FIX / "intra_rd_encode_core.v"      # md5 a822f2c8... (the walled RTL)
_CLEAN = _FIX / "input_stream_adapter.v"      # a run2 block that synthesized clean
_STAGED = _FIX / "staged_fsm_encode_core.v"   # attempt-4 staged near-miss (recon)


def _cloud_src() -> str:
    return _CLOUD.read_text()


def _clean_src() -> str:
    return _CLEAN.read_text()


def _staged_src() -> str:
    return _STAGED.read_text()


# A stage map for the staged near-miss fixture: 6 named stages, a small op
# budget (floor 256 dominates), a deep declared latency -> the module-per-stage
# protocol APPLIES. The fixture spreads its datapath across an FSM's arms.
_STAGED_STAGE_BUDGET = [
    {"name": "predict", "ops": ["add16", "add16", "sub16"]},
    {"name": "residual", "ops": ["sub16"]},
    {"name": "transform", "ops": ["add16", "mul16"]},
    {"name": "quant", "ops": ["mul16", "shift16"]},
    {"name": "recon", "ops": ["add16", "cmp16"]},
    {"name": "cost", "ops": ["add32", "mul32"]},
]


def _staged_stage_map() -> StageMap:
    return stage_map_from_budget(_STAGED_STAGE_BUDGET, declared_latency=60)


# A stage map standing in for the RD core's declared datapath (the spec declares
# registered boundaries for predict/residual/fwd/quant/dequant/idct/recon/ssd/
# bits/cost-mul/cost-add/best-cmp, a 50-cycle candidate slot, 472-cycle latency).
_RD_STAGE_BUDGET = [
    {"name": "predict", "latency_cycles": 1, "iters": 9, "ops": ["add16", "add16", "shift16"]},
    {"name": "residual", "latency_cycles": 1, "iters": 9, "ops": ["sub16"]},
    {"name": "fwd_h", "latency_cycles": 1, "iters": 9, "ops": ["add16", "add16"]},
    {"name": "fwd_v", "latency_cycles": 1, "iters": 9, "ops": ["add16", "add16"]},
    {"name": "quant", "latency_cycles": 16, "iters": 9, "ops": ["mul16", "add16", "shift16"]},
    {"name": "dequant", "latency_cycles": 16, "iters": 9, "ops": ["mul16", "shift16"]},
    {"name": "idct", "latency_cycles": 1, "iters": 9, "ops": ["add16", "shift16"]},
    {"name": "recon", "latency_cycles": 1, "iters": 9, "ops": ["add16", "cmp16"]},
    {"name": "ssd", "latency_cycles": 16, "iters": 9, "ops": ["sub16", "mul16", "add32"]},
    {"name": "coeff_bits", "latency_cycles": 16, "iters": 9, "ops": ["cmp16", "add16"]},
    {"name": "cost_mul", "latency_cycles": 1, "iters": 9, "ops": ["mul32"]},
    {"name": "cost_add", "latency_cycles": 1, "iters": 9, "ops": ["add32"]},
    {"name": "best_cmp", "latency_cycles": 1, "iters": 9, "ops": ["cmp32"]},
]


def _rd_stage_map() -> StageMap:
    return stage_map_from_budget(_RD_STAGE_BUDGET, declared_latency=472)


# ---------------------------------------------------------------------------
# Deliverable 1 -- the census
# ---------------------------------------------------------------------------
class TestCloudFixtureFails:
    def test_fixture_present_and_pinned(self):
        assert _CLOUD.exists(), "negative gold fixture missing"
        import hashlib
        md5 = hashlib.md5(_CLOUD.read_bytes()).hexdigest()
        assert md5.startswith("a822f2c8"), f"cloud fixture drifted: {md5}"

    def test_cloud_fails_multiplier_cap(self):
        rep = census_rtl(_cloud_src(), stage_map=_rd_stage_map())
        assert not rep.ok
        assert rep.mul_violations, "expected a multiplier-cap violation"
        # the collapsed S_COMPUTE search is thousands of effective multipliers
        assert rep.worst_mul > 1000, rep.worst_mul
        assert rep.worst_mul > rep.mul_cap * 10

    def test_cloud_names_the_171_amplifier(self):
        rep = census_rtl(_cloud_src(), stage_map=_rd_stage_map())
        # the single always block is `seq_logic`
        blk = max(rep.blocks, key=lambda b: b.eff_mul)
        assert blk.name == "seq_logic"
        # its amplification comes from the inlined task tree, not direct writes
        assert blk.self_mul < 50
        assert blk.eff_mul > 1000
        # the top amplifier is the encode stage task
        callees = {c for c, _w, _m in blk.top_calls}
        assert "encode4_stage_task" in callees
        # 9 modes x 19 RDOQ cuts = 171 invocations of the candidate evaluator:
        # eval_scan_candidate must be reachable with a large effective weight.
        assert "eval_scan_candidate" in rep.defs
        assert rep.defs["encode4_stage_task"].eff_mul > 1000

    def test_cloud_factor_violation(self):
        rep = census_rtl(_cloud_src(), stage_map=_rd_stage_map())
        # total effective ops dwarf the declared stage-map op budget * factor
        assert rep.factor_violations
        assert rep.total_eff_ops > rep.stage_map.total_op_slots * rep.factor

    def test_cloud_fails_even_without_stage_map(self):
        # the multiplier cap alone (no stage map) still nails the cloud
        rep = census_rtl(_cloud_src())
        assert not rep.ok
        assert rep.mul_violations


class TestCleanFixturePasses:
    def test_clean_passes(self):
        rep = census_rtl(_clean_src(), stage_map=None)
        assert rep.ok, [(b.name, b.eff_mul) for b in rep.blocks]
        assert rep.worst_mul <= rep.mul_cap

    def test_clean_passes_with_a_stage_map(self):
        # even handed a (small) stage map, the adapter is well under budget
        sm = stage_map_from_budget(
            [{"name": "s0", "ops": ["mul16"]}], declared_latency=3
        )
        rep = census_rtl(_clean_src(), stage_map=sm)
        assert rep.ok
        assert not rep.factor_violations


class TestCensusEdgeCases:
    def test_constant_operand_ops_excluded(self):
        # pure constant folds do not count; a runtime * does.
        src = """
        module m;
          always @(posedge clk) begin
            a <= 4 >> 2;          // constant fold -> not counted
            b <= 2 + 2;           // constant fold -> not counted
            c <= x * y;           // runtime multiply -> 1 mul
          end
        endmodule
        """
        rep = census_rtl(src)
        blk = rep.blocks[0]
        assert blk.self_mul == 1, blk.self_mul

    def test_multiply_by_power_of_two_is_a_shift_not_a_multiplier(self):
        src = """
        module m;
          always @(posedge clk) begin
            a <= x * 2;           // shift, not a multiplier
            b <= x * 4;           // shift, not a multiplier
            c <= x * 3;           // genuine constant-coeff multiplier
          end
        endmodule
        """
        rep = census_rtl(src)
        blk = rep.blocks[0]
        assert blk.self_mul == 1, blk.self_mul  # only x*3 counts as a multiplier

    def test_task_with_no_callers_is_not_amplified(self):
        # a task defined but never called contributes 0 to any block.
        src = """
        module m;
          task dead_multiplier;
            input integer a; input integer b; output integer c;
            begin c = a * b * a * b; end
          endtask
          always @(posedge clk) begin
            q <= r + s;           // no calls; no multipliers
          end
        endmodule
        """
        rep = census_rtl(src)
        assert rep.worst_mul == 0
        assert rep.ok
        # the def is censused (has multipliers) but never expanded into a block
        assert rep.defs["dead_multiplier"].self_mul >= 3

    def test_loop_weighting_amplifies_a_called_task(self):
        # a task with one multiply, called inside a 16-trip loop = 16 multipliers.
        src = """
        module m;
          function integer mac;
            input integer a; input integer b;
            begin mac = a * b; end
          endfunction
          always @(posedge clk) begin : dp
            integer i;
            for (i = 0; i < 16; i = i + 1) begin
              acc[i] <= mac(x[i], w[i]);
            end
          end
        endmodule
        """
        rep = census_rtl(src)
        blk = rep.blocks[0]
        assert blk.eff_mul == 16, blk.eff_mul

    def test_nonconstant_loop_bound_is_weight_one(self):
        # a dynamic loop bound cannot be statically unrolled -> weight 1 (lenient).
        src = """
        module m;
          always @(posedge clk) begin : dp
            integer i;
            for (i = 0; i < n; i = i + 1) begin
              acc[i] <= x[i] * w[i];
            end
          end
        endmodule
        """
        rep = census_rtl(src)
        assert rep.blocks[0].eff_mul == 1, rep.blocks[0].eff_mul

    def test_nested_loops_multiply(self):
        src = """
        module m;
          always @(posedge clk) begin : dp
            integer i; integer j;
            for (i = 0; i < 9; i = i + 1) begin
              for (j = 0; j < 19; j = j + 1) begin
                acc <= x[i] * w[j];
              end
            end
          end
        endmodule
        """
        rep = census_rtl(src)
        assert rep.blocks[0].eff_mul == 9 * 19, rep.blocks[0].eff_mul

    def test_continuous_assign_cloud_is_caught(self):
        # a cloud hidden in continuous assigns + a function chain is still counted
        # in the synthetic <continuous> pseudo-block.
        src = """
        module m;
          function integer big;
            input integer a; input integer b;
            integer i; integer acc;
            begin
              acc = 0;
              for (i = 0; i < 200; i = i + 1) acc = acc + (a * b);
              big = acc;
            end
          endfunction
          assign z = big(p, q);
        endmodule
        """
        rep = census_rtl(src, mul_cap=64)
        cont = [b for b in rep.blocks if b.name == "<continuous>"]
        assert cont, "expected a <continuous> pseudo-block"
        assert cont[0].eff_mul == 200
        assert not rep.ok  # 200 > 64


# ---------------------------------------------------------------------------
# pipeline-campaign-2 -- FSM-aware census (MAX-over-arms, not SUM)
#
# An iterative FSM keeps its whole datapath textually inside one always block's
# case arms / else-if branches, but exactly ONE arm executes per cycle. The
# census must reflect the worst SINGLE path per cycle (MAX over arms), not the
# sum of every arm -- otherwise it false-positives the protocol's own
# recommended controller shape as a combinational cloud.
# ---------------------------------------------------------------------------
def _eff_mul(src: str, cap: int = 64) -> int:
    return census_rtl(src, mul_cap=cap).blocks[0].eff_mul


class TestFsmAwareCensus:
    def test_case_arms_are_maxed_not_summed(self):
        # two arms each with one runtime multiply; only one runs per cycle -> 1.
        src = """
        module m; always @(posedge clk) begin
          case (state)
            2'd0: y <= a * b;   // 1 mul
            2'd1: y <= c * d;   // 1 mul (mutually exclusive)
            default: y <= 0;
          endcase
        end endmodule
        """
        assert _eff_mul(src) == 1

    def test_ops_outside_case_still_sum(self):
        # a multiply outside the case executes every cycle -> summed with the
        # maxed arm: 1 (outside) + max(1, 1) (arms) = 2.
        src = """
        module m; always @(posedge clk) begin
          z <= p * q;           // 1 mul, outside the case -> summed
          case (state)
            2'd0: y <= a * b;   // 1 mul
            2'd1: y <= c * d;   // 1 mul
          endcase
        end endmodule
        """
        assert _eff_mul(src) == 2

    def test_else_if_chain_is_maxed(self):
        # parallel else-if arms are mutually exclusive -> max, not sum.
        src = """
        module m; always @(posedge clk) begin
          if (s == 0) y <= a * b;
          else if (s == 1) y <= c * d;
          else y <= e * f;
        end endmodule
        """
        assert _eff_mul(src) == 1

    def test_separate_if_statements_still_sum(self):
        # two DISTINCT if-statements both execute every cycle -> summed.
        src = """
        module m; always @(posedge clk) begin
          if (s == 0) y <= a * b;   // 1
          if (t == 0) z <= c * d;   // 1 (separate statement -> sums)
        end endmodule
        """
        assert _eff_mul(src) == 2

    def test_nested_case_recurses_max(self):
        # a bare (non-begin) nested case as an arm body -> recursive max.
        src = """
        module m; always @(posedge clk) begin
          case (outer)
            2'd0: case (inner)
                    2'd0: y <= a * b;   // 1
                    2'd1: y <= c * d;   // 1
                  endcase
            2'd1: y <= e * f;           // 1
          endcase
        end endmodule
        """
        assert _eff_mul(src) == 1

    def test_loop_inside_one_arm_still_sums_then_arms_max(self):
        # a for-loop inside ONE arm executes fully in that state's cycle
        # (trip-weighted, summed); across arms it maxes: max(8, 1) = 8.
        src = """
        module m; always @(posedge clk) begin : dp
          integer i;
          case (state)
            2'd0: for (i = 0; i < 8; i = i + 1) acc <= acc + x[i] * w[i];
            2'd1: y <= a * b;
          endcase
        end endmodule
        """
        assert _eff_mul(src) == 8

    def test_task_called_from_three_arms_counts_once(self):
        # a task called once from each of three arms counts once per arm, and
        # the arms are maxed -> effective 1, not 3.
        src = """
        module m;
          function integer mul1;
            input integer a; input integer b;
            begin mul1 = a * b; end
          endfunction
          always @(posedge clk) begin
            case (state)
              2'd0: y <= mul1(a, b);
              2'd1: y <= mul1(c, d);
              2'd2: y <= mul1(e, f);
            endcase
          end
        endmodule
        """
        assert _eff_mul(src) == 1

    def test_spread_over_arms_passes_but_all_in_one_arm_fails(self):
        # SAME total multiplier work: spread one 20-mul loop per FSM state
        # (worst single path 20 -> under cap) vs. all four loops crammed into
        # one cycle (80 -> over cap). This is the exact cloud-vs-controller
        # distinction the FSM-aware census draws.
        spread = """
        module m; always @(posedge clk) begin : dp
          integer i;
          case (state)
            3'd0: for (i=0;i<20;i=i+1) a <= a + x[i]*w[i];
            3'd1: for (i=0;i<20;i=i+1) b <= b + x[i]*w[i];
            3'd2: for (i=0;i<20;i=i+1) c <= c + x[i]*w[i];
            3'd3: for (i=0;i<20;i=i+1) d <= d + x[i]*w[i];
          endcase
        end endmodule
        """
        crammed = """
        module m; always @(posedge clk) begin : dp
          integer i;
          for (i=0;i<20;i=i+1) a <= a + x[i]*w[i];
          for (i=0;i<20;i=i+1) b <= b + x[i]*w[i];
          for (i=0;i<20;i=i+1) c <= c + x[i]*w[i];
          for (i=0;i<20;i=i+1) d <= d + x[i]*w[i];
        end endmodule
        """
        rs = census_rtl(spread, mul_cap=64)
        rc = census_rtl(crammed, mul_cap=64)
        assert rs.worst_mul == 20 and rs.ok
        assert rc.worst_mul == 80 and not rc.ok


class TestStagedFsmNearMissFixture:
    """The reconstructed attempt-4 near-miss: a genuinely staged, multi-module
    design whose per-state datapath lives inside one always block's FSM arms.
    Under the old SUM census it false-positived the FACTOR gate; the FSM-aware
    MAX-over-arms census must PASS it."""

    def test_fixture_present(self):
        assert _STAGED.exists(), "staged near-miss fixture missing"

    def test_staged_fsm_passes_after_fix(self):
        rep = census_rtl(_staged_src(), stage_map=_staged_stage_map(),
                         enforce_stage_modules=True)
        assert rep.ok, [(b.name, b.eff_mul, b.eff_ops)
                        for b in rep.blocks if b.eff_ops or b.eff_mul]
        assert not rep.mul_violations
        assert not rep.factor_violations

    def test_staged_fsm_is_the_multi_module_near_miss(self):
        # matches the reported attempt-4 census shape:
        rep = census_rtl(_staged_src(), stage_map=_staged_stage_map(),
                         enforce_stage_modules=True)
        # multi-module (protocol satisfied) -> never structurally deficient
        assert rep.module_instances >= rep.stage_map.stage_count
        assert not rep.stage_module_deficient
        # worst effective multipliers 12 (< cap 64): mul was never the problem
        assert rep.worst_mul == 12
        # the two FSM blocks each collapse to their heaviest arm, well under the
        # floor-256 factor budget (they SUMMED to 306 / 288 under the old census)
        by = {b.name: b for b in rep.blocks}
        assert by["seq_logic"].eff_ops < 256
        assert by["predict_calc_seq"].eff_ops < 256

    def test_staged_fsm_worst_arm_is_realistic_single_stage(self):
        # the collapsed per-cycle op counts are one stage's worth, not the whole
        # search: seq_logic's heaviest arm == 42 ops, predict's == 48.
        rep = census_rtl(_staged_src(), stage_map=_staged_stage_map())
        by = {b.name: b for b in rep.blocks}
        assert by["seq_logic"].eff_ops == 42
        assert by["predict_calc_seq"].eff_ops == 48


# ---------------------------------------------------------------------------
# Deliverable 2 -- module-per-stage protocol
# ---------------------------------------------------------------------------
_THREE_STAGE_MODULAR = """
module top_pipe(input clk, input rst_n, output done);
  wire v0, v1, v2;
  stage_predict #(.W(16)) u_s0 (.clk(clk), .rst_n(rst_n), .valid_out(v0));
  stage_transform u_s1 (.clk(clk), .in_valid(v0), .valid_out(v1));
  stage_quant u_s2 (.clk(clk), .in_valid(v1), .valid_out(v2));
  ctrl_fsm u_ctrl (.clk(clk), .rst_n(rst_n), .done(done));
endmodule
module stage_predict(input clk, input rst_n, output valid_out);
  reg [15:0] acc; always @(posedge clk) acc <= acc + 1; assign valid_out = 1'b1;
endmodule
module stage_transform(input clk, input in_valid, output valid_out);
  reg [15:0] q; always @(posedge clk) q <= q + 2; assign valid_out = in_valid;
endmodule
module stage_quant(input clk, input in_valid, output valid_out);
  reg [15:0] r; always @(posedge clk) r <= r + 3; assign valid_out = in_valid;
endmodule
module ctrl_fsm(input clk, input rst_n, output done); assign done = 1'b1; endmodule
"""


class TestStageModuleProtocol:
    def test_applies_predicate(self):
        assert _rd_stage_map().applies                       # 472-cycle datapath
        assert stage_map_from_budget(
            [{"name": "a"}, {"name": "b"}, {"name": "c"}], declared_latency=None
        ).applies                                            # >=3 stages
        assert not stage_map_from_budget(
            [{"name": "a"}], declared_latency=4
        ).applies                                            # single shallow stage
        assert not StageMap().applies                        # no stage map

    def test_modular_three_stage_passes(self):
        sm = stage_map_from_budget(
            [{"name": "predict", "ops": ["add16"]},
             {"name": "transform", "ops": ["mul16"]},
             {"name": "quant", "ops": ["mul16"]}],
            declared_latency=30,
        )
        rep = check_stage_modules(_THREE_STAGE_MODULAR, sm)
        assert rep.applies
        assert rep.instantiated_submodules >= rep.declared_stage_count
        assert rep.ok

    def test_monolith_flagged_deficient(self):
        sm = _rd_stage_map()
        rep = check_stage_modules(_cloud_src(), sm)
        assert rep.applies
        assert rep.instantiated_submodules == 0      # one module, only tasks
        assert not rep.ok

    def test_deficiency_rides_on_census_violation(self):
        # structural deficiency only fires in census_rtl when the arithmetic
        # census already flags the block AND enforcement is on.
        rep = census_rtl(_cloud_src(), stage_map=_rd_stage_map(),
                         enforce_stage_modules=True)
        assert rep.stage_module_deficient
        # with enforcement OFF, no structural finding (census verdict unchanged)
        rep2 = census_rtl(_cloud_src(), stage_map=_rd_stage_map(),
                          enforce_stage_modules=False)
        assert not rep2.stage_module_deficient
        assert not rep2.ok  # still fails on arithmetic

    def test_protocol_not_applicable_leaves_single_stage_untouched(self):
        # a single-stage block with a tiny/no stage map: protocol does not apply,
        # so the structural check is a no-op pass.
        rep = check_stage_modules(_clean_src(), StageMap())
        assert not rep.applies
        assert rep.ok
        # and a genuinely single-module clean block never trips the census
        assert census_rtl(_clean_src()).ok


# ---------------------------------------------------------------------------
# Deliverable 3 -- retry feedback + escalation
# ---------------------------------------------------------------------------
class TestRetryFeedback:
    def test_clean_report_is_empty(self):
        rep = census_rtl(_clean_src())
        assert format_stage_lint_report(rep, block="input_stream_adapter") == ""

    def test_feedback_has_census_table_stage_map_and_numbered_remedy(self):
        rep = census_rtl(_cloud_src(), stage_map=_rd_stage_map(),
                         enforce_stage_modules=True)
        msg = format_stage_lint_report(rep, block="intra_rd_encode_core")
        # census table
        assert "ARITHMETIC CENSUS" in msg
        assert "seq_logic" in msg
        assert "effective multipliers" in msg.lower() or "eff. multipliers" in msg
        # stage map verbatim
        assert "DECLARED STAGE MAP" in msg
        assert "predict" in msg and "cost_mul" in msg
        assert "472 cycles" in msg
        # numbered mechanical remedy (module-per-stage)
        assert "MANDATORY REMEDY" in msg
        assert "  1." in msg and "  2." in msg and "  3." in msg
        assert "submodule per" in msg
        assert "controller FSM" in msg
        # anti-pattern guard names the exact failure
        assert "FORBIDDEN" in msg
        assert "task" in msg and "combinational cone" in msg.lower()
        # structural deficiency remedy present
        assert "STRUCTURAL" in msg

    def test_signature_stable_and_sensitive(self):
        r1 = census_rtl(_cloud_src(), stage_map=_rd_stage_map())
        r2 = census_rtl(_cloud_src(), stage_map=_rd_stage_map())
        assert census_signature(r1) == census_signature(r2)
        assert census_signature(r1) != census_signature(census_rtl(_clean_src()))

    def test_trajectory_and_fresh_session_escalation_text(self):
        rep = census_rtl(_cloud_src(), stage_map=_rd_stage_map())
        base = format_stage_lint_report(rep, block="b")
        assert "TRAJECTORY" not in base and "FRESH-SESSION" not in base
        esc = format_stage_lint_report(rep, block="b", trajectory="identical")
        assert "STRUCTURALLY IDENTICAL" in esc
        fresh = format_stage_lint_report(rep, block="b", fresh_session=True)
        assert "FRESH-SESSION ESCALATION" in fresh


# ---------------------------------------------------------------------------
# Env gates (both branches)
# ---------------------------------------------------------------------------
class TestEnvGates:
    def test_stage_lint_default_on(self, monkeypatch):
        monkeypatch.delenv("CORESMITH_STAGE_LINT", raising=False)
        assert stage_lint_enabled()

    def test_stage_lint_can_be_disabled(self, monkeypatch):
        for v in ("0", "false", "no", "off"):
            monkeypatch.setenv("CORESMITH_STAGE_LINT", v)
            assert not stage_lint_enabled()

    def test_stage_modules_default_on_and_toggle(self, monkeypatch):
        monkeypatch.delenv("CORESMITH_STAGE_MODULES", raising=False)
        assert stage_modules_enabled()
        monkeypatch.setenv("CORESMITH_STAGE_MODULES", "0")
        assert not stage_modules_enabled()

    def test_mul_cap_env_tunable(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_STAGE_LINT_MUL_CAP", "100000")
        # with an absurd cap, even the cloud's multiplier count passes the cap
        rep = census_rtl(_cloud_src())  # no stage map -> only the cap applies
        assert rep.ok
        monkeypatch.setenv("CORESMITH_STAGE_LINT_MUL_CAP", "64")
        assert not census_rtl(_cloud_src()).ok

    def test_factor_env_tunable(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_STAGE_LINT_MUL_CAP", "10000000")
        monkeypatch.setenv("CORESMITH_STAGE_LINT_FACTOR", "1000000")
        rep = census_rtl(_cloud_src(), stage_map=_rd_stage_map())
        assert not rep.factor_violations  # factor so large nothing trips it


# ---------------------------------------------------------------------------
# Prompt pinning (Deliverable 2 contract lives in the prompts)
# ---------------------------------------------------------------------------
class TestPromptPinning:
    def test_pipeline_contract_skill_has_module_per_stage(self):
        from orchestrator.langchain.prompts.skills import load_skills
        t = load_skills("pipeline_contract")
        assert "Module-per-stage protocol" in t
        assert "one submodule per named stage" in t.lower()
        assert "controller FSM" in t
        assert "CORESMITH_STAGE_LINT" in t

    def test_rtl_generator_prompt_has_stage_module_rule(self):
        p = (Path(__file__).resolve().parents[1] / "langchain" / "prompts"
             / "rtl_generator.md")
        t = p.read_text()
        assert "N+1 MODULES" in t
        assert "iterative" in t.lower() and "controller" in t.lower()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
