// vip2 fixture twin: the SAME interface, but the response comes 2 cycles after
// the request -- the contract says exactly 1. The VIP must reject it.
module responder (
  input  wire       clk,
  input  wire       rst_n,
  input  wire [7:0] s_q_addr,
  input  wire       s_q_req_valid,
  output wire       s_q_req_gnt,
  output reg        s_q_rsp_valid,
  output reg  [7:0] s_q_rdata
);
  reg       v1;
  reg [7:0] d1;
  assign s_q_req_gnt = 1'b1;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      v1 <= 1'b0; d1 <= 8'd0; s_q_rsp_valid <= 1'b0; s_q_rdata <= 8'd0;
    end else begin
      v1 <= s_q_req_valid; d1 <= s_q_addr + 8'd1;
      s_q_rsp_valid <= v1;  s_q_rdata <= d1;
    end
  end
endmodule
