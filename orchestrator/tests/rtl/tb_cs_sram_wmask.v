// Self-checking TB for the cs_sram_1rw1r optional byte-mask (USE_WMASK).
// Proves: (1) legacy instantiation (wmask0 unconnected, USE_WMASK=0) is
// bit-for-bit the old full-word write; (2) USE_WMASK=1 writes only enabled
// byte lanes and preserves the rest in a single cycle; (3) the write-first
// port0 capture and the port1 same-address bypass both observe the MERGED
// word. Finishes with WMASK_TB_PASS or $fatal.
`timescale 1ns/1ps
module tb_cs_sram_wmask;
    reg clk = 0;
    always #5 clk = ~clk;

    // ---- legacy instance: wmask0 left unconnected, USE_WMASK defaulted ----
    reg         l_ce0 = 0, l_we0 = 0, l_ce1 = 0;
    reg  [8:0]  l_addr0 = 0, l_addr1 = 0;
    reg  [31:0] l_wdata0 = 0;
    wire [31:0] l_rdata0, l_rdata1;
    cs_sram_1rw1r #(.WIDTH(32), .DEPTH(512)) u_legacy (
        .clk(clk),
        .ce0(l_ce0), .we0(l_we0), .addr0(l_addr0), .wdata0(l_wdata0),
        .rdata0(l_rdata0),
        .ce1(l_ce1), .addr1(l_addr1), .rdata1(l_rdata1)
    );

    // ---- masked instance ----
    reg         m_ce0 = 0, m_we0 = 0, m_ce1 = 0;
    reg  [8:0]  m_addr0 = 0, m_addr1 = 0;
    reg  [31:0] m_wdata0 = 0;
    reg  [3:0]  m_wmask0 = 4'hF;
    wire [31:0] m_rdata0, m_rdata1;
    cs_sram_1rw1r #(.WIDTH(32), .DEPTH(512), .USE_WMASK(1)) u_masked (
        .clk(clk),
        .ce0(m_ce0), .we0(m_we0), .addr0(m_addr0), .wdata0(m_wdata0),
        .wmask0(m_wmask0), .rdata0(m_rdata0),
        .ce1(m_ce1), .addr1(m_addr1), .rdata1(m_rdata1)
    );

    task chk(input [31:0] got, input [31:0] exp, input [127:0] tag);
        if (got !== exp) begin
            $display("FAIL %0s: got %h exp %h", tag, got, exp);
            $fatal(1);
        end
    endtask

    initial begin
        @(negedge clk);
        // 1) legacy full-word write + readback (wmask0 unconnected)
        l_ce0 = 1; l_we0 = 1; l_addr0 = 9'd7; l_wdata0 = 32'hA5B6C7D8;
        @(negedge clk);
        chk(l_rdata0, 32'hA5B6C7D8, "legacy write-first capture");
        l_we0 = 0;
        @(negedge clk);
        chk(l_rdata0, 32'hA5B6C7D8, "legacy readback");

        // 2) masked instance: seed full word, then write only lanes 0 and 2
        m_ce0 = 1; m_we0 = 1; m_addr0 = 9'd42; m_wdata0 = 32'h11223344;
        m_wmask0 = 4'hF;
        @(negedge clk);
        m_wdata0 = 32'hDEADBEEF; m_wmask0 = 4'b0101;   // lanes 0,2 only
        // port1 bypass same address in the same cycle must see the MERGE
        m_ce1 = 1; m_addr1 = 9'd42;
        @(negedge clk);
        // expected: keep 11 (lane3), take AD (lane2), keep 33 (lane1), take EF (lane0)
        chk(m_rdata0, 32'h11AD33EF, "masked write-first capture (merged)");
        chk(m_rdata1, 32'h11AD33EF, "port1 same-addr bypass (merged)");
        m_we0 = 0;
        @(negedge clk);
        chk(m_rdata0, 32'h11AD33EF, "masked readback preserves unselected");

        // 3) masked instance with all lanes on == plain full-word write
        m_we0 = 1; m_addr0 = 9'd43; m_wdata0 = 32'h0BADF00D; m_wmask0 = 4'hF;
        @(negedge clk);
        m_we0 = 0;
        @(negedge clk);
        chk(m_rdata0, 32'h0BADF00D, "all-lanes mask == full-word");

        $display("WMASK_TB_PASS");
        $finish;
    end

    // ---- fpmem bit-mask instance (WMASK_GRAN=1 path) ----
    reg         f_ce0 = 0, f_we0 = 0;
    reg  [3:0]  f_addr0 = 0, f_addr1 = 0;
    reg  [35:0] f_wdata0 = 0, f_wmask0 = 0;
    wire [35:0] f_rdata0, f_rdata1;
    cs_fpmem_1rw1r #(.WIDTH(36), .DEPTH(16), .USE_WMASK(1)) u_fp_masked (
        .clk(clk),
        .ce0(f_ce0), .we0(f_we0), .addr0(f_addr0), .wdata0(f_wdata0),
        .wmask0(f_wmask0), .rdata0(f_rdata0),
        .ce1(1'b0), .addr1(f_addr1), .rdata1(f_rdata1)
    );

    task fchk(input [35:0] got, input [35:0] exp, input [127:0] tag);
        if (got !== exp) begin
            $display("FAIL %0s: got %h exp %h", tag, got, exp);
            $fatal(1);
        end
    endtask

    initial begin
        @(negedge clk);
        // seed a full 36-bit word, then masked-commit only bits [17:9]
        f_ce0 = 1; f_we0 = 1; f_addr0 = 4'd3;
        f_wdata0 = 36'h5_5555_5555; f_wmask0 = {36{1'b1}};
        @(negedge clk);
        f_wdata0 = 36'hA_AAAA_AAAA; f_wmask0 = 36'h0_0003_FE00;  // bits 17:9
        @(negedge clk);
        f_we0 = 0;
        @(negedge clk);
        // expect 5s everywhere except bits 17:9 which took the A-pattern
        fchk(f_rdata0,
             (36'h5_5555_5555 & ~36'h0_0003_FE00) |
             (36'hA_AAAA_AAAA &  36'h0_0003_FE00),
             "fpmem bit-masked commit");
        $display("FPMEM_WMASK_TB_PASS");
    end
endmodule
