// Copyright (c) Meta Platforms, Inc. and affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
//
// tiny_matmul -- 2x2 signed int8 matrix multiply, registered outputs.
//
// The minimal end-to-end fixture for the byo-pdk branch: small enough to lint
// and synthesize in well under a second, but real logic (multipliers + adders +
// flops) so a generic synth reports a non-trivial cell count.
//
//   C = A * B    where A = [[a00 a01],[a10 a11]], B = [[b00 b01],[b10 b11]]
//
// Each element is signed 8-bit; each output is the sum of two 16-bit products,
// held in 18 bits and registered on the clock (async active-low reset).

module tiny_matmul (
    input  wire               clk,
    input  wire               rst_n,
    input  wire signed [7:0]  a00,
    input  wire signed [7:0]  a01,
    input  wire signed [7:0]  a10,
    input  wire signed [7:0]  a11,
    input  wire signed [7:0]  b00,
    input  wire signed [7:0]  b01,
    input  wire signed [7:0]  b10,
    input  wire signed [7:0]  b11,
    output reg  signed [17:0] c00,
    output reg  signed [17:0] c01,
    output reg  signed [17:0] c10,
    output reg  signed [17:0] c11
);

    // Signed 8x8 -> 16-bit partial products.
    wire signed [15:0] p00 = a00 * b00;
    wire signed [15:0] p01 = a01 * b10;
    wire signed [15:0] p02 = a00 * b01;
    wire signed [15:0] p03 = a01 * b11;
    wire signed [15:0] p04 = a10 * b00;
    wire signed [15:0] p05 = a11 * b10;
    wire signed [15:0] p06 = a10 * b01;
    wire signed [15:0] p07 = a11 * b11;

    // Dot-products, sign-extended to 18 bits before the add.
    wire signed [17:0] n00 = {{2{p00[15]}}, p00} + {{2{p01[15]}}, p01};
    wire signed [17:0] n01 = {{2{p02[15]}}, p02} + {{2{p03[15]}}, p03};
    wire signed [17:0] n10 = {{2{p04[15]}}, p04} + {{2{p05[15]}}, p05};
    wire signed [17:0] n11 = {{2{p06[15]}}, p06} + {{2{p07[15]}}, p07};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            c00 <= 18'sd0;
            c01 <= 18'sd0;
            c10 <= 18'sd0;
            c11 <= 18'sd0;
        end else begin
            c00 <= n00;
            c01 <= n01;
            c10 <= n10;
            c11 <= n11;
        end
    end

endmodule
