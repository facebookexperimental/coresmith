// Copyright (c) Meta Platforms, Inc. and affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
//
// FIXTURE PROVENANCE -- reconstructed FSM near-miss (pipeline-campaign-2).
// A faithful reconstruction of the reported census shape (not original
// bytes) of a fat encode-core regen: a GENUINELY staged,
// multi-module design (stage_module_deficient=False; worst effective
// multipliers 12 < cap 64) whose iterative controller keeps its per-state
// datapath textually inside ONE always block's FSM arms -- case-on-state in
// `seq_logic`, else-if-on-state in `predict_calc_seq`. Exactly one state runs
// per cycle, so the per-cycle arithmetic is the WORST SINGLE arm, not the SUM.
// Under the old SUM census both blocks tripped the stage-realization FACTOR
// gate (seq_logic 306 ops, predict_calc_seq 288 ops vs floor-256 budget) -- a
// false positive; under the FSM-aware MAX-over-arms census each block collapses
// to its heaviest arm (seq_logic 42, predict_calc_seq 48) and PASSES.

module stage_predict   (input clk, input rst_n, input en, output reg vo); always @(posedge clk) vo <= en; endmodule
module stage_residual  (input clk, input rst_n, input en, output reg vo); always @(posedge clk) vo <= en; endmodule
module stage_transform (input clk, input rst_n, input en, output reg vo); always @(posedge clk) vo <= en; endmodule
module stage_quant     (input clk, input rst_n, input en, output reg vo); always @(posedge clk) vo <= en; endmodule
module stage_recon     (input clk, input rst_n, input en, output reg vo); always @(posedge clk) vo <= en; endmodule
module stage_cost      (input clk, input rst_n, input en, output reg vo); always @(posedge clk) vo <= en; endmodule

// ---------------------------------------------------------------------------
// predict_engine: a small iterative stage whose controller carries the
// candidate math across an else-if-on-state chain (predict_calc_seq).
// ---------------------------------------------------------------------------
module predict_engine (
    input              clk,
    input              rst_n,
    input      [2:0]   pstate,
    input      [15:0]  x   [0:23],
    input      [15:0]  y   [0:23],
    output reg [31:0]  pacc
);
    integer i;
    reg [31:0] p;
    always @(posedge clk) begin : predict_calc_seq
        if (!rst_n) begin
            p <= 32'd0;
        end else if (pstate == 3'd0) begin
            for (i = 0; i < 24; i = i + 1) begin
                p = p + x[i] - y[i];        // 24*2 = 48 runtime ops (this cycle)
            end
        end else if (pstate == 3'd1) begin
            for (i = 0; i < 24; i = i + 1) begin
                p = p + x[i] - y[i];        // 48
            end
        end else if (pstate == 3'd2) begin
            for (i = 0; i < 24; i = i + 1) begin
                p = p + x[i] - y[i];        // 48
            end
        end else if (pstate == 3'd3) begin
            for (i = 0; i < 24; i = i + 1) begin
                p = p + x[i] - y[i];        // 48
            end
        end else if (pstate == 3'd4) begin
            for (i = 0; i < 24; i = i + 1) begin
                p = p + x[i] - y[i];        // 48
            end
        end else begin
            for (i = 0; i < 24; i = i + 1) begin
                p = p + x[i] - y[i];        // 48
            end
        end
        pacc <= p;
    end
endmodule

// ---------------------------------------------------------------------------
// Top: instantiates one registered submodule per stage + drives the iterative
// controller (seq_logic) that walks the search over CYCLES on one datapath.
// ---------------------------------------------------------------------------
module intra_rd_stage_top (
    input             clk,
    input             rst_n,
    input     [2:0]   state,
    input     [15:0]  a    [0:19],
    input     [15:0]  b    [0:19],
    input     [15:0]  coef [0:11],
    input     [15:0]  src  [0:11],
    output reg [31:0] acc,
    output            done
);
    wire v0, v1, v2, v3, v4, v5;
    stage_predict   u_s0 (.clk(clk), .rst_n(rst_n), .en(1'b1), .vo(v0));
    stage_residual  u_s1 (.clk(clk), .rst_n(rst_n), .en(v0),   .vo(v1));
    stage_transform u_s2 (.clk(clk), .rst_n(rst_n), .en(v1),   .vo(v2));
    stage_quant     u_s3 (.clk(clk), .rst_n(rst_n), .en(v2),   .vo(v3));
    stage_recon     u_s4 (.clk(clk), .rst_n(rst_n), .en(v3),   .vo(v4));
    stage_cost      u_s5 (.clk(clk), .rst_n(rst_n), .en(v4),   .vo(v5));
    assign done = v5;

    integer i;
    reg [31:0] acc0, acc1, prod;
    always @(posedge clk) begin : seq_logic
        if (!rst_n) begin
            acc  <= 32'd0;
            acc0 <= 32'd0;
            acc1 <= 32'd0;
            prod <= 32'd0;
        end else begin
            case (state)
                3'd0: begin
                    for (i = 0; i < 20; i = i + 1) acc0 = acc0 + a[i] - b[i];  // 40
                end
                3'd1: begin
                    for (i = 0; i < 20; i = i + 1) acc0 = acc0 + a[i] - b[i];  // 40
                end
                3'd2: begin
                    for (i = 0; i < 20; i = i + 1) acc1 = acc1 + a[i] - b[i];  // 40
                end
                3'd3: begin
                    // the one transform state with genuine multipliers
                    for (i = 0; i < 12; i = i + 1) prod = prod + coef[i] * src[i];  // 12 mul + 12 add = 24
                end
                3'd4: begin
                    for (i = 0; i < 20; i = i + 1) acc1 = acc1 + a[i] - b[i];  // 40
                end
                3'd5: begin
                    for (i = 0; i < 20; i = i + 1) acc0 = acc0 + a[i] - b[i];  // 40
                end
                3'd6: begin
                    for (i = 0; i < 20; i = i + 1) acc1 = acc1 + a[i] - b[i];  // 40
                end
                default: begin
                    for (i = 0; i < 20; i = i + 1) acc0 = acc0 + a[i] - b[i];  // 40
                    acc <= acc0 + acc1 + prod;
                end
            endcase
        end
    end
endmodule
