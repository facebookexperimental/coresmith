// vip2 fixture: a req_resp responder that answers EXACTLY 1 cycle after the
// request is accepted (contract: req_to_rsp_cycles.exact = 1).
module responder (
  input  wire       clk,
  input  wire       rst_n,
  input  wire [7:0] s_q_addr,
  input  wire       s_q_req_valid,
  output wire       s_q_req_gnt,
  output reg        s_q_rsp_valid,
  output reg  [7:0] s_q_rdata
);
  assign s_q_req_gnt = 1'b1;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      s_q_rsp_valid <= 1'b0;
      s_q_rdata     <= 8'd0;
    end else begin
      s_q_rsp_valid <= s_q_req_valid;
      s_q_rdata     <= s_q_addr + 8'd1;
    end
  end
`ifndef SYNTHESIS
  // INV: INV-RESP-001
  always @(posedge clk) if (rst_n && s_q_rsp_valid && s_q_rdata == 8'd0 && 0) $error("INV-RESP-001");
`endif
endmodule
