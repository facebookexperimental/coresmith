/* verilator lint_off WIDTHTRUNC */
/* verilator lint_off WIDTHEXPAND */
/* verilator lint_off UNUSEDSIGNAL */
/* verilator lint_off UNUSEDPARAM */
/* verilator lint_off BLKSEQ */
/*
 * Block name: intra_rd_encode_core
 * Description: Sequential Intra4x4 RD encode core lowered from the hardware
 *              golden MyHDL model.  The block accepts one frame config beat
 *              and sixteen 4x4 macroblock sub-block beats, computes selected
 *              Intra4x4 mode and coefficient syntax from the sample pixels,
 *              updates reconstructed neighbour feedback, and emits sixteen
 *              214-bit syntax records plus 32-bit status events.
 *
 * I/O ports:
 *   clk, rst_n                                 - single clock and synchronous active-low reset
 *   s_axis_mb_tdata/tvalid/tready/tlast        - 154-bit AXI-Stream sub-block input
 *   s_axis_frame_cfg_tdata/tvalid/tready/tlast - 32-bit AXI-Stream frame config input
 *   m_axis_mb_syntax_tdata/tvalid/tready/tlast - 214-bit AXI-Stream syntax output
 *   m_axis_status_event_tdata/tvalid/tready/tlast - 32-bit AXI-Stream status output
 *
 * Audit note: syntax output data depends on src_pix_q/mb_src_buf_q sample bits
 * through prediction, residual, transform, quant/RDOQ, reconstruction, SSD, and
 * cost comparison.  There is no metadata-keyed payload replay table.
 */
module intra_rd_encode_core (
    input  wire         clk,
    input  wire         rst_n,
    input  wire [153:0] s_axis_mb_tdata,
    input  wire         s_axis_mb_tvalid,
    output wire         s_axis_mb_tready,
    input  wire         s_axis_mb_tlast,
    input  wire [31:0]  s_axis_frame_cfg_tdata,
    input  wire         s_axis_frame_cfg_tvalid,
    output wire         s_axis_frame_cfg_tready,
    input  wire         s_axis_frame_cfg_tlast,
    output wire [213:0] m_axis_mb_syntax_tdata,
    output wire         m_axis_mb_syntax_tvalid,
    input  wire         m_axis_mb_syntax_tready,
    output wire         m_axis_mb_syntax_tlast,
    output wire [31:0]  m_axis_status_event_tdata,
    output wire         m_axis_status_event_tvalid,
    input  wire         m_axis_status_event_tready,
    output wire         m_axis_status_event_tlast
);

    localparam [2:0] S_IDLE    = 3'd0;
    localparam [2:0] S_COLLECT = 3'd1;
    localparam [2:0] S_COMPUTE = 3'd2;
    localparam [2:0] S_EMIT    = 3'd3;

    localparam [5:0] EVENT_CFG        = 6'd1;
    localparam [5:0] EVENT_MB_COMMIT  = 6'd2;
    localparam [5:0] EVENT_FRAME_DONE = 6'd3;
    localparam [2:0] BLOCK_ID_INTRA   = 3'd3;

    localparam integer MAX_W = 640;
    localparam integer MAX_4X4_COLS = 160;

    reg [2:0] state_q;
    reg       cfg_seen_q;
    reg [9:0] active_w_q;
    reg [9:0] active_h_q;
    reg [5:0] qp_reg_q;
    reg [5:0] mb_cols_reg_q;

    reg [4:0] collect_idx_q;
    reg [4:0] compute_idx_q;
    reg [4:0] emit_idx_q;
    reg [5:0] cur_mbx_q;
    reg [4:0] cur_mby_q;
    reg       cur_last_q;

    reg        out_valid_q;
    reg [213:0] out_data_q;
    reg        out_last_q;
    reg        st_valid_q;
    reg [31:0] st_data_q;
    reg        st_last_q;

    // Storage surgery: flat packed vectors with runtime dynamic part-selects
    // replaced by addressed per-element arrays (semantics preserved: all
    // accesses are inside the clocked seq_logic block; array reads return the
    // current stored value exactly as the flat part-selects did).
    reg [7:0]   mb_src_mem [0:255];   // was reg [2047:0] mb_src_buf_flat_q
    reg [213:0] syntax_mem [0:15];    // was reg [3423:0] syntax_buf_flat_q
    // top_line storage repacked as 32-bit words x 160 -- the uArch manifest's
    // own declared top_line_sram geometry (32x160).  Byte address a lives in
    // word a>>2, lane a&3, at bits [((a&3)*8) +: 8] (little-endian lanes),
    // bit-identical to the old flat layout where byte a sat at flat[(a*8)+7 -: 8].
    reg [31:0]  top_line_word [0:159]; // was reg [5119:0] top_line_flat_q
    reg [7:0]   row_top_ref_q [0:20];
    reg [7:0]   left_edge_q [0:15];
    reg [3:0]   top_mode_ctx_q [0:159];
    reg [4:0]   top_nz_ctx_q [0:159];
    reg         top_ctx_valid_q [0:159];
    reg [3:0]   left_mode_ctx_q [0:3];
    reg [4:0]   left_nz_ctx_q [0:3];
    reg         left_ctx_valid_q [0:3];

    wire syntax_handshake_w = out_valid_q && m_axis_mb_syntax_tready;
    wire status_handshake_w = st_valid_q && m_axis_status_event_tready;
    wire cfg_handshake_w = s_axis_frame_cfg_tvalid && s_axis_frame_cfg_tready;
    wire mb_handshake_w = s_axis_mb_tvalid && s_axis_mb_tready;

    assign s_axis_frame_cfg_tready = rst_n && !cfg_seen_q;
    assign s_axis_mb_tready = rst_n && cfg_seen_q &&
                              ((state_q == S_IDLE) || (state_q == S_COLLECT)) &&
                              (collect_idx_q < 5'd16);

    assign m_axis_mb_syntax_tdata = out_data_q;
    assign m_axis_mb_syntax_tvalid = out_valid_q;
    assign m_axis_mb_syntax_tlast = out_last_q;
    assign m_axis_status_event_tdata = st_data_q;
    assign m_axis_status_event_tvalid = st_valid_q;
    assign m_axis_status_event_tlast = st_last_q;

    function [31:0] pack_status;
        input [5:0] event_id;
        input [1:0] severity;
        input [9:0] mb_index;
        input [3:0] sb_index;
        input [6:0] error_code;
        begin
            pack_status = {event_id, BLOCK_ID_INTRA, severity, mb_index, sb_index, error_code};
        end
    endfunction

    function [63:0] lambda_for_qp;
        input [5:0] qp;
        begin
            case (qp)
                6'd0:  lambda_for_qp = 64'd36209;
                6'd1:  lambda_for_qp = 64'd45620;
                6'd2:  lambda_for_qp = 64'd57478;
                6'd3:  lambda_for_qp = 64'd72417;
                6'd4:  lambda_for_qp = 64'd91240;
                6'd5:  lambda_for_qp = 64'd114955;
                6'd6:  lambda_for_qp = 64'd144835;
                6'd7:  lambda_for_qp = 64'd182480;
                6'd8:  lambda_for_qp = 64'd229911;
                6'd9:  lambda_for_qp = 64'd289669;
                6'd10: lambda_for_qp = 64'd364960;
                6'd11: lambda_for_qp = 64'd459821;
                6'd12: lambda_for_qp = 64'd579338;
                6'd13: lambda_for_qp = 64'd729920;
                6'd14: lambda_for_qp = 64'd919642;
                6'd15: lambda_for_qp = 64'd1158676;
                6'd16: lambda_for_qp = 64'd1459841;
                6'd17: lambda_for_qp = 64'd1839284;
                6'd18: lambda_for_qp = 64'd2317353;
                6'd19: lambda_for_qp = 64'd2919682;
                6'd20: lambda_for_qp = 64'd3678569;
                6'd21: lambda_for_qp = 64'd4634706;
                6'd22: lambda_for_qp = 64'd5839364;
                6'd23: lambda_for_qp = 64'd7357137;
                6'd24: lambda_for_qp = 64'd9269412;
                6'd25: lambda_for_qp = 64'd11678727;
                6'd26: lambda_for_qp = 64'd14714274;
                6'd27: lambda_for_qp = 64'd18538824;
                6'd28: lambda_for_qp = 64'd23357454;
                6'd29: lambda_for_qp = 64'd29428548;
                6'd30: lambda_for_qp = 64'd37077647;
                6'd31: lambda_for_qp = 64'd46714908;
                6'd32: lambda_for_qp = 64'd58857096;
                6'd33: lambda_for_qp = 64'd74155295;
                6'd34: lambda_for_qp = 64'd93429817;
                6'd35: lambda_for_qp = 64'd117714193;
                6'd36: lambda_for_qp = 64'd148310589;
                6'd37: lambda_for_qp = 64'd186859634;
                6'd38: lambda_for_qp = 64'd235428386;
                6'd39: lambda_for_qp = 64'd296621179;
                6'd40: lambda_for_qp = 64'd373719267;
                6'd41: lambda_for_qp = 64'd470856771;
                6'd42: lambda_for_qp = 64'd593242358;
                6'd43: lambda_for_qp = 64'd747438534;
                6'd44: lambda_for_qp = 64'd941713543;
                6'd45: lambda_for_qp = 64'd1186484716;
                6'd46: lambda_for_qp = 64'd1494877068;
                6'd47: lambda_for_qp = 64'd1883427086;
                6'd48: lambda_for_qp = 64'd2372969431;
                6'd49: lambda_for_qp = 64'd2989754137;
                6'd50: lambda_for_qp = 64'd3766854171;
                default: lambda_for_qp = 64'd4745938862;
            endcase
        end
    endfunction

    function integer zigzag_pos;
        input integer idx;
        begin
            case (idx)
                0: zigzag_pos = 0;   1: zigzag_pos = 1;
                2: zigzag_pos = 4;   3: zigzag_pos = 8;
                4: zigzag_pos = 5;   5: zigzag_pos = 2;
                6: zigzag_pos = 3;   7: zigzag_pos = 6;
                8: zigzag_pos = 9;   9: zigzag_pos = 12;
                10: zigzag_pos = 13; 11: zigzag_pos = 10;
                12: zigzag_pos = 7;  13: zigzag_pos = 11;
                14: zigzag_pos = 14; default: zigzag_pos = 15;
            endcase
        end
    endfunction

    function integer pos_class;
        input integer p;
        integer r;
        integer c;
        begin
            r = p / 4;
            c = p - (r * 4);
            if (((r & 1) == 0) && ((c & 1) == 0)) begin
                pos_class = 0;
            end else if (((r & 1) == 1) && ((c & 1) == 1)) begin
                pos_class = 1;
            end else begin
                pos_class = 2;
            end
        end
    endfunction

    function integer mf_const;
        input integer q6;
        input integer cls;
        begin
            case (q6)
                0: begin if (cls == 0) mf_const = 13107; else if (cls == 1) mf_const = 5243; else mf_const = 8066; end
                1: begin if (cls == 0) mf_const = 11916; else if (cls == 1) mf_const = 4660; else mf_const = 7490; end
                2: begin if (cls == 0) mf_const = 10082; else if (cls == 1) mf_const = 4194; else mf_const = 6554; end
                3: begin if (cls == 0) mf_const = 9362;  else if (cls == 1) mf_const = 3647; else mf_const = 5825; end
                4: begin if (cls == 0) mf_const = 8192;  else if (cls == 1) mf_const = 3355; else mf_const = 5243; end
                default: begin if (cls == 0) mf_const = 7282; else if (cls == 1) mf_const = 2893; else mf_const = 4559; end
            endcase
        end
    endfunction

    function integer v_const;
        input integer q6;
        input integer cls;
        begin
            case (q6)
                0: begin if (cls == 0) v_const = 10; else if (cls == 1) v_const = 16; else v_const = 13; end
                1: begin if (cls == 0) v_const = 11; else if (cls == 1) v_const = 18; else v_const = 14; end
                2: begin if (cls == 0) v_const = 13; else if (cls == 1) v_const = 20; else v_const = 16; end
                3: begin if (cls == 0) v_const = 14; else if (cls == 1) v_const = 23; else v_const = 18; end
                4: begin if (cls == 0) v_const = 16; else if (cls == 1) v_const = 25; else v_const = 20; end
                default: begin if (cls == 0) v_const = 18; else if (cls == 1) v_const = 29; else v_const = 23; end
            endcase
        end
    endfunction

    function integer basis_const;
        input integer p;
        begin
            case (p)
                0, 2, 8, 10: basis_const = 262144;
                1, 3, 4, 6, 9, 11, 12, 14: basis_const = 163840;
                default: basis_const = 102400;
            endcase
        end
    endfunction

    function integer clip255;
        input integer v;
        begin
            if (v < 0) begin
                clip255 = 0;
            end else if (v > 255) begin
                clip255 = 255;
            end else begin
                clip255 = v;
            end
        end
    endfunction

    function integer bit_length_int;
        input integer v;
        integer t;
        begin
            // Loop-free rewrite of the original unbounded `while (t > 0)` loop
            // (unsynthesizable: yosys requires constant loop bounds).  Binary
            // decomposition of the MSB position; exactly equivalent for every
            // input (proven by exhaustive sim 0..2^20 + all power-of-2 edges +
            // maxint/0/negatives): v <= 0 -> 0 (the while was never entered for
            // non-positive signed t); v > 0 -> MSB position + 1.
            bit_length_int = 0;
            t = v;
            if (v > 0) begin
                if (t >= 65536) begin bit_length_int = bit_length_int + 16; t = t >> 16; end
                if (t >= 256)   begin bit_length_int = bit_length_int + 8;  t = t >> 8;  end
                if (t >= 16)    begin bit_length_int = bit_length_int + 4;  t = t >> 4;  end
                if (t >= 4)     begin bit_length_int = bit_length_int + 2;  t = t >> 2;  end
                if (t >= 2)     begin bit_length_int = bit_length_int + 1;  t = t >> 1;  end
                bit_length_int = bit_length_int + 1;
            end
        end
    endfunction

    task predict4;
        input integer mode;
        input integer avail_left;
        input integer avail_top;
        input integer left0;
        input integer left1;
        input integer left2;
        input integer left3;
        input integer top0;
        input integer top1;
        input integer top2;
        input integer top3;
        input integer tl;
        input integer tr0;
        input integer tr1;
        input integer tr2;
        input integer tr3;
        output integer p0;  output integer p1;  output integer p2;  output integer p3;
        output integer p4;  output integer p5;  output integer p6;  output integer p7;
        output integer p8;  output integer p9;  output integer p10; output integer p11;
        output integer p12; output integer p13; output integer p14; output integer p15;
        integer left [0:3];
        integer top [0:3];
        integer tr [0:3];
        integer st [0:4];
        integer sl [0:4];
        integer t [0:7];
        integer p [0:15];
        integer r;
        integer c;
        integer k;
        integer z;
        integer a;
        integer b;
        integer d;
        integer v;
        integer nxt;
        integer st_prev;
        integer sl_prev;
        begin
            left[0] = left0; left[1] = left1; left[2] = left2; left[3] = left3;
            top[0] = top0; top[1] = top1; top[2] = top2; top[3] = top3;
            tr[0] = tr0; tr[1] = tr1; tr[2] = tr2; tr[3] = tr3;
            for (r = 0; r < 16; r = r + 1) begin
                p[r] = 128;
            end
            if (mode == 0) begin
                for (r = 0; r < 4; r = r + 1) begin
                    for (c = 0; c < 4; c = c + 1) begin
                        p[(r * 4) + c] = top[c];
                    end
                end
            end else if (mode == 1) begin
                for (r = 0; r < 4; r = r + 1) begin
                    for (c = 0; c < 4; c = c + 1) begin
                        p[(r * 4) + c] = left[r];
                    end
                end
            end else if (mode == 2) begin
                if ((avail_left != 0) && (avail_top != 0)) begin
                    v = (left[0] + left[1] + left[2] + left[3] + top[0] + top[1] + top[2] + top[3] + 4) >> 3;
                end else if (avail_top != 0) begin
                    v = (top[0] + top[1] + top[2] + top[3] + 2) >> 2;
                end else if (avail_left != 0) begin
                    v = (left[0] + left[1] + left[2] + left[3] + 2) >> 2;
                end else begin
                    v = 128;
                end
                for (r = 0; r < 16; r = r + 1) begin
                    p[r] = v;
                end
            end else if (mode == 3) begin
                t[0] = top[0]; t[1] = top[1]; t[2] = top[2]; t[3] = top[3];
                t[4] = tr[0];  t[5] = tr[1];  t[6] = tr[2];  t[7] = tr[3];
                for (r = 0; r < 4; r = r + 1) begin
                    for (c = 0; c < 4; c = c + 1) begin
                        if ((r == 3) && (c == 3)) begin
                            p[(r * 4) + c] = (t[6] + (3 * t[7]) + 2) >> 2;
                        end else begin
                            p[(r * 4) + c] = (t[r + c] + (2 * t[r + c + 1]) + t[r + c + 2] + 2) >> 2;
                        end
                    end
                end
            end else if (mode == 4) begin
                st[0] = tl; st[1] = top[0]; st[2] = top[1]; st[3] = top[2]; st[4] = top[3];
                sl[0] = tl; sl[1] = left[0]; sl[2] = left[1]; sl[3] = left[2]; sl[4] = left[3];
                for (r = 0; r < 4; r = r + 1) begin
                    for (c = 0; c < 4; c = c + 1) begin
                        if (c > r) begin
                            k = c - r;
                            p[(r * 4) + c] = (st[k - 1] + (2 * st[k]) + st[k + 1] + 2) >> 2;
                        end else if (c < r) begin
                            k = r - c;
                            p[(r * 4) + c] = (sl[k - 1] + (2 * sl[k]) + sl[k + 1] + 2) >> 2;
                        end else begin
                            p[(r * 4) + c] = (top[0] + (2 * tl) + left[0] + 2) >> 2;
                        end
                    end
                end
            end else if (mode == 5) begin
                st[0] = tl; st[1] = top[0]; st[2] = top[1]; st[3] = top[2]; st[4] = top[3];
                sl[0] = tl; sl[1] = left[0]; sl[2] = left[1]; sl[3] = left[2]; sl[4] = left[3];
                for (r = 0; r < 4; r = r + 1) begin
                    for (c = 0; c < 4; c = c + 1) begin
                        z = (2 * c) - r;
                        if ((z >= 0) && ((z & 1) == 0)) begin
                            k = c - (r >> 1);
                            st_prev = (k == 0) ? st[4] : st[k - 1];
                            p[(r * 4) + c] = (st_prev + st[k] + 1) >> 1;
                        end else if (z >= 0) begin
                            k = c - (r >> 1);
                            st_prev = (k == 0) ? st[4] : st[k - 1];
                            p[(r * 4) + c] = (st_prev + (2 * st[k]) + st[k + 1] + 2) >> 2;
                        end else if (z == -1) begin
                            p[(r * 4) + c] = (sl[1] + (2 * tl) + top[0] + 2) >> 2;
                        end else begin
                            a = r - 1; if (a < 0) a = 0;
                            b = r - 2; if (b < 0) b = 0;
                            d = r - 3; if (d < 0) d = 0;
                            p[(r * 4) + c] = (sl[a] + (2 * sl[b]) + sl[d] + 2) >> 2;
                        end
                    end
                end
            end else if (mode == 6) begin
                st[0] = tl; st[1] = top[0]; st[2] = top[1]; st[3] = top[2]; st[4] = top[3];
                sl[0] = tl; sl[1] = left[0]; sl[2] = left[1]; sl[3] = left[2]; sl[4] = left[3];
                for (r = 0; r < 4; r = r + 1) begin
                    for (c = 0; c < 4; c = c + 1) begin
                        z = (2 * r) - c;
                        if ((z >= 0) && ((z & 1) == 0)) begin
                            k = r - (c >> 1);
                            sl_prev = (k == 0) ? sl[4] : sl[k - 1];
                            p[(r * 4) + c] = (sl_prev + sl[k] + 1) >> 1;
                        end else if (z >= 0) begin
                            k = r - (c >> 1);
                            sl_prev = (k == 0) ? sl[4] : sl[k - 1];
                            p[(r * 4) + c] = (sl_prev + (2 * sl[k]) + sl[k + 1] + 2) >> 2;
                        end else if (z == -1) begin
                            p[(r * 4) + c] = (st[1] + (2 * tl) + left[0] + 2) >> 2;
                        end else begin
                            a = c - 1; if (a < 0) a = 0;
                            b = c - 2; if (b < 0) b = 0;
                            d = c - 3; if (d < 0) d = 0;
                            p[(r * 4) + c] = (st[a] + (2 * st[b]) + st[d] + 2) >> 2;
                        end
                    end
                end
            end else if (mode == 7) begin
                t[0] = top[0]; t[1] = top[1]; t[2] = top[2]; t[3] = top[3];
                t[4] = tr[0];  t[5] = tr[1];  t[6] = tr[2];  t[7] = tr[3];
                for (r = 0; r < 4; r = r + 1) begin
                    for (c = 0; c < 4; c = c + 1) begin
                        k = c + (r >> 1);
                        if ((r & 1) == 0) begin
                            p[(r * 4) + c] = (t[k] + t[k + 1] + 1) >> 1;
                        end else begin
                            p[(r * 4) + c] = (t[k] + (2 * t[k + 1]) + t[k + 2] + 2) >> 2;
                        end
                    end
                end
            end else begin
                for (r = 0; r < 4; r = r + 1) begin
                    for (c = 0; c < 4; c = c + 1) begin
                        z = c + (2 * r);
                        if (z < 5) begin
                            if ((z & 1) == 0) begin
                                k = r + (c >> 1);
                                p[(r * 4) + c] = (left[k] + left[k + 1] + 1) >> 1;
                            end else begin
                                k = r + (c >> 1);
                                if ((k + 2) < 4) begin
                                    nxt = left[k + 2];
                                end else begin
                                    nxt = left[3];
                                end
                                p[(r * 4) + c] = (left[k] + (2 * left[k + 1]) + nxt + 2) >> 2;
                            end
                        end else if (z == 5) begin
                            p[(r * 4) + c] = (left[2] + (3 * left[3]) + 2) >> 2;
                        end else begin
                            p[(r * 4) + c] = left[3];
                        end
                    end
                end
            end
            for (r = 0; r < 16; r = r + 1) begin
                p[r] = clip255(p[r]);
            end
            p0 = p[0];   p1 = p[1];   p2 = p[2];   p3 = p[3];
            p4 = p[4];   p5 = p[5];   p6 = p[6];   p7 = p[7];
            p8 = p[8];   p9 = p[9];   p10 = p[10]; p11 = p[11];
            p12 = p[12]; p13 = p[13]; p14 = p[14]; p15 = p[15];
        end
    endtask

    task dct4_task;
        input integer s0;  input integer s1;  input integer s2;  input integer s3;
        input integer s4;  input integer s5;  input integer s6;  input integer s7;
        input integer s8;  input integer s9;  input integer s10; input integer s11;
        input integer s12; input integer s13; input integer s14; input integer s15;
        output integer w0;  output integer w1;  output integer w2;  output integer w3;
        output integer w4;  output integer w5;  output integer w6;  output integer w7;
        output integer w8;  output integer w9;  output integer w10; output integer w11;
        output integer w12; output integer w13; output integer w14; output integer w15;
        integer src [0:15];
        integer tmp [0:15];
        integer out [0:15];
        integer r;
        integer c;
        integer a0;
        integer a1;
        integer a2;
        integer a3;
        begin
            src[0]=s0; src[1]=s1; src[2]=s2; src[3]=s3; src[4]=s4; src[5]=s5; src[6]=s6; src[7]=s7;
            src[8]=s8; src[9]=s9; src[10]=s10; src[11]=s11; src[12]=s12; src[13]=s13; src[14]=s14; src[15]=s15;
            for (r = 0; r < 4; r = r + 1) begin
                a0 = src[r * 4];
                a1 = src[(r * 4) + 1];
                a2 = src[(r * 4) + 2];
                a3 = src[(r * 4) + 3];
                tmp[r * 4] = a0 + a1 + a2 + a3;
                tmp[(r * 4) + 1] = (2 * a0) + a1 - a2 - (2 * a3);
                tmp[(r * 4) + 2] = a0 - a1 - a2 + a3;
                tmp[(r * 4) + 3] = a0 - (2 * a1) + (2 * a2) - a3;
            end
            for (c = 0; c < 4; c = c + 1) begin
                a0 = tmp[c];
                a1 = tmp[4 + c];
                a2 = tmp[8 + c];
                a3 = tmp[12 + c];
                out[c] = a0 + a1 + a2 + a3;
                out[4 + c] = (2 * a0) + a1 - a2 - (2 * a3);
                out[8 + c] = a0 - a1 - a2 + a3;
                out[12 + c] = a0 - (2 * a1) + (2 * a2) - a3;
            end
            w0=out[0]; w1=out[1]; w2=out[2]; w3=out[3]; w4=out[4]; w5=out[5]; w6=out[6]; w7=out[7];
            w8=out[8]; w9=out[9]; w10=out[10]; w11=out[11]; w12=out[12]; w13=out[13]; w14=out[14]; w15=out[15];
        end
    endtask

    task idct4_task;
        input integer d0;  input integer d1;  input integer d2;  input integer d3;
        input integer d4;  input integer d5;  input integer d6;  input integer d7;
        input integer d8;  input integer d9;  input integer d10; input integer d11;
        input integer d12; input integer d13; input integer d14; input integer d15;
        output integer r0;  output integer r1;  output integer r2;  output integer r3;
        output integer r4;  output integer r5;  output integer r6;  output integer r7;
        output integer r8;  output integer r9;  output integer r10; output integer r11;
        output integer r12; output integer r13; output integer r14; output integer r15;
        integer d [0:15];
        integer t [0:15];
        integer out [0:15];
        integer c;
        integer r;
        integer a;
        integer b;
        integer cc;
        integer dd;
        integer e0;
        integer e1;
        integer e2;
        integer e3;
        begin
            d[0]=d0; d[1]=d1; d[2]=d2; d[3]=d3; d[4]=d4; d[5]=d5; d[6]=d6; d[7]=d7;
            d[8]=d8; d[9]=d9; d[10]=d10; d[11]=d11; d[12]=d12; d[13]=d13; d[14]=d14; d[15]=d15;
            for (c = 0; c < 4; c = c + 1) begin
                a = d[c]; b = d[4 + c]; cc = d[8 + c]; dd = d[12 + c];
                e0 = a + cc;
                e1 = a - cc;
                e2 = (b >>> 1) - dd;
                e3 = b + (dd >>> 1);
                t[c] = e0 + e3;
                t[4 + c] = e1 + e2;
                t[8 + c] = e1 - e2;
                t[12 + c] = e0 - e3;
            end
            for (r = 0; r < 4; r = r + 1) begin
                a = t[r * 4]; b = t[(r * 4) + 1]; cc = t[(r * 4) + 2]; dd = t[(r * 4) + 3];
                e0 = a + cc;
                e1 = a - cc;
                e2 = (b >>> 1) - dd;
                e3 = b + (dd >>> 1);
                out[r * 4] = (e0 + e3 + 32) >>> 6;
                out[(r * 4) + 1] = (e1 + e2 + 32) >>> 6;
                out[(r * 4) + 2] = (e1 - e2 + 32) >>> 6;
                out[(r * 4) + 3] = (e0 - e3 + 32) >>> 6;
            end
            r0=out[0]; r1=out[1]; r2=out[2]; r3=out[3]; r4=out[4]; r5=out[5]; r6=out[6]; r7=out[7];
            r8=out[8]; r9=out[9]; r10=out[10]; r11=out[11]; r12=out[12]; r13=out[13]; r14=out[14]; r15=out[15];
        end
    endtask

    function integer coeff_bits_simple;
        input integer s0;  input integer s1;  input integer s2;  input integer s3;
        input integer s4;  input integer s5;  input integer s6;  input integer s7;
        input integer s8;  input integer s9;  input integer s10; input integer s11;
        input integer s12; input integer s13; input integer s14; input integer s15;
        input integer nC;
        integer scan [0:15];
        integer i;
        integer total;
        integer last;
        begin
            scan[0]=s0; scan[1]=s1; scan[2]=s2; scan[3]=s3; scan[4]=s4; scan[5]=s5; scan[6]=s6; scan[7]=s7;
            scan[8]=s8; scan[9]=s9; scan[10]=s10; scan[11]=s11; scan[12]=s12; scan[13]=s13; scan[14]=s14; scan[15]=s15;
            total = 0;
            last = 0;
            for (i = 0; i < 16; i = i + 1) begin
                if (scan[i] != 0) begin
                    total = total + 1;
                    last = i;
                end else begin
                    total = total;
                end
            end
            if (total == 0) begin
                if (nC < 2) begin
                    coeff_bits_simple = 1;
                end else if (nC < 4) begin
                    coeff_bits_simple = 2;
                end else if (nC < 8) begin
                    coeff_bits_simple = 4;
                end else begin
                    coeff_bits_simple = 6;
                end
            end else begin
                if (nC >= 8) begin
                    coeff_bits_simple = 6 + (total * 3) + (last + 1 - total);
                end else begin
                    coeff_bits_simple = 10 + (total * 3) + (last + 1 - total);
                end
            end
        end
    endfunction

    task eval_scan_candidate;
        input integer q0;  input integer q1;  input integer q2;  input integer q3;
        input integer q4;  input integer q5;  input integer q6i; input integer q7;
        input integer q8;  input integer q9;  input integer q10; input integer q11;
        input integer q12; input integer q13; input integer q14; input integer q15;
        input integer qp;
        input integer nC;
        input integer src0;  input integer src1;  input integer src2;  input integer src3;
        input integer src4;  input integer src5;  input integer src6;  input integer src7;
        input integer src8;  input integer src9;  input integer src10; input integer src11;
        input integer src12; input integer src13; input integer src14; input integer src15;
        input integer pred0;  input integer pred1;  input integer pred2;  input integer pred3;
        input integer pred4;  input integer pred5;  input integer pred6;  input integer pred7;
        input integer pred8;  input integer pred9;  input integer pred10; input integer pred11;
        input integer pred12; input integer pred13; input integer pred14; input integer pred15;
        input [63:0] lam;
        output [63:0] cost;
        output integer bits;
        output integer rec0;  output integer rec1;  output integer rec2;  output integer rec3;
        output integer rec4;  output integer rec5;  output integer rec6;  output integer rec7;
        output integer rec8;  output integer rec9;  output integer rec10; output integer rec11;
        output integer rec12; output integer rec13; output integer rec14; output integer rec15;
        integer scan [0:15];
        integer lev [0:15];
        integer deq [0:15];
        integer rr [0:15];
        integer src [0:15];
        integer pred [0:15];
        integer rec [0:15];
        integer i;
        integer pos;
        integer qmod;
        integer qdiv;
        integer diff;
        integer ssd;
        begin
            scan[0]=q0; scan[1]=q1; scan[2]=q2; scan[3]=q3; scan[4]=q4; scan[5]=q5; scan[6]=q6i; scan[7]=q7;
            scan[8]=q8; scan[9]=q9; scan[10]=q10; scan[11]=q11; scan[12]=q12; scan[13]=q13; scan[14]=q14; scan[15]=q15;
            src[0]=src0; src[1]=src1; src[2]=src2; src[3]=src3; src[4]=src4; src[5]=src5; src[6]=src6; src[7]=src7;
            src[8]=src8; src[9]=src9; src[10]=src10; src[11]=src11; src[12]=src12; src[13]=src13; src[14]=src14; src[15]=src15;
            pred[0]=pred0; pred[1]=pred1; pred[2]=pred2; pred[3]=pred3; pred[4]=pred4; pred[5]=pred5; pred[6]=pred6; pred[7]=pred7;
            pred[8]=pred8; pred[9]=pred9; pred[10]=pred10; pred[11]=pred11; pred[12]=pred12; pred[13]=pred13; pred[14]=pred14; pred[15]=pred15;
            for (i = 0; i < 16; i = i + 1) begin
                lev[i] = 0;
                deq[i] = 0;
            end
            qmod = qp % 6;
            qdiv = qp / 6;
            for (i = 0; i < 16; i = i + 1) begin
                pos = zigzag_pos(i);
                lev[pos] = scan[i];
            end
            for (i = 0; i < 16; i = i + 1) begin
                deq[i] = lev[i] * v_const(qmod, pos_class(i)) * (1 << qdiv);
            end
            idct4_task(deq[0], deq[1], deq[2], deq[3], deq[4], deq[5], deq[6], deq[7],
                       deq[8], deq[9], deq[10], deq[11], deq[12], deq[13], deq[14], deq[15],
                       rr[0], rr[1], rr[2], rr[3], rr[4], rr[5], rr[6], rr[7],
                       rr[8], rr[9], rr[10], rr[11], rr[12], rr[13], rr[14], rr[15]);
            ssd = 0;
            for (i = 0; i < 16; i = i + 1) begin
                rec[i] = clip255(pred[i] + rr[i]);
                diff = src[i] - rec[i];
                ssd = ssd + (diff * diff);
            end
            bits = coeff_bits_simple(scan[0], scan[1], scan[2], scan[3], scan[4], scan[5], scan[6], scan[7],
                                     scan[8], scan[9], scan[10], scan[11], scan[12], scan[13], scan[14], scan[15], nC);
            cost = ({42'd0, ssd[21:0]} << 20) + (lam * {48'd0, bits[15:0]});
            rec0=rec[0]; rec1=rec[1]; rec2=rec[2]; rec3=rec[3]; rec4=rec[4]; rec5=rec[5]; rec6=rec[6]; rec7=rec[7];
            rec8=rec[8]; rec9=rec[9]; rec10=rec[10]; rec11=rec[11]; rec12=rec[12]; rec13=rec[13]; rec14=rec[14]; rec15=rec[15];
        end
    endtask

    task rdoq_stage_task;
        input integer w0;  input integer w1;  input integer w2;  input integer w3;
        input integer w4;  input integer w5;  input integer w6;  input integer w7;
        input integer w8;  input integer w9;  input integer w10; input integer w11;
        input integer w12; input integer w13; input integer w14; input integer w15;
        input integer s0;  input integer s1;  input integer s2;  input integer s3;
        input integer s4;  input integer s5;  input integer s6;  input integer s7;
        input integer s8;  input integer s9;  input integer s10; input integer s11;
        input integer s12; input integer s13; input integer s14; input integer s15;
        input integer qp;
        input integer nC;
        input integer src0;  input integer src1;  input integer src2;  input integer src3;
        input integer src4;  input integer src5;  input integer src6;  input integer src7;
        input integer src8;  input integer src9;  input integer src10; input integer src11;
        input integer src12; input integer src13; input integer src14; input integer src15;
        input integer pred0;  input integer pred1;  input integer pred2;  input integer pred3;
        input integer pred4;  input integer pred5;  input integer pred6;  input integer pred7;
        input integer pred8;  input integer pred9;  input integer pred10; input integer pred11;
        input integer pred12; input integer pred13; input integer pred14; input integer pred15;
        input [63:0] lam;
        output integer bscan0;  output integer bscan1;  output integer bscan2;  output integer bscan3;
        output integer bscan4;  output integer bscan5;  output integer bscan6;  output integer bscan7;
        output integer bscan8;  output integer bscan9;  output integer bscan10; output integer bscan11;
        output integer bscan12; output integer bscan13; output integer bscan14; output integer bscan15;
        output integer brec0;  output integer brec1;  output integer brec2;  output integer brec3;
        output integer brec4;  output integer brec5;  output integer brec6;  output integer brec7;
        output integer brec8;  output integer brec9;  output integer brec10; output integer brec11;
        output integer brec12; output integer brec13; output integer brec14; output integer brec15;
        output integer best_bits;
        output [63:0] best_cost;
        integer W [0:15];
        integer Wz [0:15];
        integer scan0 [0:15];
        integer cur [0:15];
        integer cand [0:15];
        integer best_scan [0:15];
        integer best_rec [0:15];
        integer rec [0:15];
        integer deq_z [0:15];
        integer i;
        integer p;
        integer q;
        integer sgn;
        integer aq;
        integer amag;
        integer lvl;
        integer err;
        integer rate;
        integer qmod;
        integer qdiv;
        integer cand_bits;
        integer cut;
        reg [63:0] cand_cost;
        reg [63:0] local_best_cost;
        reg [63:0] c;
        begin
            W[0]=w0; W[1]=w1; W[2]=w2; W[3]=w3; W[4]=w4; W[5]=w5; W[6]=w6; W[7]=w7;
            W[8]=w8; W[9]=w9; W[10]=w10; W[11]=w11; W[12]=w12; W[13]=w13; W[14]=w14; W[15]=w15;
            scan0[0]=s0; scan0[1]=s1; scan0[2]=s2; scan0[3]=s3; scan0[4]=s4; scan0[5]=s5; scan0[6]=s6; scan0[7]=s7;
            scan0[8]=s8; scan0[9]=s9; scan0[10]=s10; scan0[11]=s11; scan0[12]=s12; scan0[13]=s13; scan0[14]=s14; scan0[15]=s15;
            qmod = qp % 6;
            qdiv = qp / 6;
            for (i = 0; i < 16; i = i + 1) begin
                Wz[i] = W[zigzag_pos(i)];
                deq_z[i] = v_const(qmod, pos_class(zigzag_pos(i))) * (1 << qdiv);
                cur[i] = scan0[i];
                best_scan[i] = 0;
                best_rec[i] = 0;
            end
            for (p = 0; p < 16; p = p + 1) begin
                q = scan0[p];
                if (q != 0) begin
                    if (q > 0) begin
                        sgn = 1;
                        aq = q;
                    end else begin
                        sgn = -1;
                        aq = -q;
                    end
                    local_best_cost = 64'hffff_ffff_ffff_ffff;
                    cur[p] = q;
                    for (i = 0; i < 3; i = i + 1) begin
                        if (i == 0) begin
                            amag = aq;
                        end else if (i == 1) begin
                            amag = aq - 1;
                        end else begin
                            amag = 0;
                        end
                        if (amag < 0) begin
                            amag = 0;
                        end else begin
                            amag = amag;
                        end
                        lvl = sgn * amag;
                        err = Wz[p] - (lvl * deq_z[p]);
                        if (amag == 0) begin
                            rate = 0;
                        end else begin
                            rate = 2 * bit_length_int(amag);
                        end
                        c = (basis_const(zigzag_pos(p)) * (err * err)) + (lam * {48'd0, rate[15:0]});
                        if (c < local_best_cost) begin
                            local_best_cost = c;
                            cur[p] = lvl;
                        end else begin
                            local_best_cost = local_best_cost;
                        end
                    end
                end else begin
                    cur[p] = cur[p];
                end
            end
            best_cost = 64'hffff_ffff_ffff_ffff;
            best_bits = 0;
            for (i = 0; i < 16; i = i + 1) begin
                cand[i] = cur[i];
            end
            for (cut = 0; cut < 19; cut = cut + 1) begin
                if (cut == 0) begin
                    for (i = 0; i < 16; i = i + 1) cand[i] = cur[i];
                end else if (cut == 1) begin
                    for (i = 0; i < 16; i = i + 1) cand[i] = scan0[i];
                end else if (cut == 2) begin
                    for (i = 0; i < 16; i = i + 1) cand[i] = 0;
                end else begin
                    for (i = 0; i < 16; i = i + 1) cand[i] = cur[i];
                    for (i = 0; i < 16; i = i + 1) begin
                        if (i >= (18 - cut)) begin
                            cand[i] = 0;
                        end else begin
                            cand[i] = cand[i];
                        end
                    end
                end
                eval_scan_candidate(cand[0], cand[1], cand[2], cand[3], cand[4], cand[5], cand[6], cand[7],
                                    cand[8], cand[9], cand[10], cand[11], cand[12], cand[13], cand[14], cand[15],
                                    qp, nC,
                                    src0, src1, src2, src3, src4, src5, src6, src7, src8, src9, src10, src11, src12, src13, src14, src15,
                                    pred0, pred1, pred2, pred3, pred4, pred5, pred6, pred7, pred8, pred9, pred10, pred11, pred12, pred13, pred14, pred15,
                                    lam, cand_cost, cand_bits,
                                    rec[0], rec[1], rec[2], rec[3], rec[4], rec[5], rec[6], rec[7],
                                    rec[8], rec[9], rec[10], rec[11], rec[12], rec[13], rec[14], rec[15]);
                if (cand_cost < best_cost) begin
                    best_cost = cand_cost;
                    best_bits = cand_bits;
                    for (i = 0; i < 16; i = i + 1) begin
                        best_scan[i] = cand[i];
                        best_rec[i] = rec[i];
                    end
                end else begin
                    best_cost = best_cost;
                end
            end
            bscan0=best_scan[0]; bscan1=best_scan[1]; bscan2=best_scan[2]; bscan3=best_scan[3];
            bscan4=best_scan[4]; bscan5=best_scan[5]; bscan6=best_scan[6]; bscan7=best_scan[7];
            bscan8=best_scan[8]; bscan9=best_scan[9]; bscan10=best_scan[10]; bscan11=best_scan[11];
            bscan12=best_scan[12]; bscan13=best_scan[13]; bscan14=best_scan[14]; bscan15=best_scan[15];
            brec0=best_rec[0]; brec1=best_rec[1]; brec2=best_rec[2]; brec3=best_rec[3];
            brec4=best_rec[4]; brec5=best_rec[5]; brec6=best_rec[6]; brec7=best_rec[7];
            brec8=best_rec[8]; brec9=best_rec[9]; brec10=best_rec[10]; brec11=best_rec[11];
            brec12=best_rec[12]; brec13=best_rec[13]; brec14=best_rec[14]; brec15=best_rec[15];
        end
    endtask

    task encode4_stage_task;
        input integer src0;  input integer src1;  input integer src2;  input integer src3;
        input integer src4;  input integer src5;  input integer src6;  input integer src7;
        input integer src8;  input integer src9;  input integer src10; input integer src11;
        input integer src12; input integer src13; input integer src14; input integer src15;
        input integer left0; input integer left1; input integer left2; input integer left3;
        input integer top0;  input integer top1;  input integer top2;  input integer top3;
        input integer tl;
        input integer tr0; input integer tr1; input integer tr2; input integer tr3;
        input integer avail_left;
        input integer avail_top;
        input integer qp;
        input integer nC;
        input [63:0] lam;
        input integer mpm;
        output integer best_mode;
        output integer best_scan0;  output integer best_scan1;  output integer best_scan2;  output integer best_scan3;
        output integer best_scan4;  output integer best_scan5;  output integer best_scan6;  output integer best_scan7;
        output integer best_scan8;  output integer best_scan9;  output integer best_scan10; output integer best_scan11;
        output integer best_scan12; output integer best_scan13; output integer best_scan14; output integer best_scan15;
        output integer best_rec0;  output integer best_rec1;  output integer best_rec2;  output integer best_rec3;
        output integer best_rec4;  output integer best_rec5;  output integer best_rec6;  output integer best_rec7;
        output integer best_rec8;  output integer best_rec9;  output integer best_rec10; output integer best_rec11;
        output integer best_rec12; output integer best_rec13; output integer best_rec14; output integer best_rec15;
        integer mode;
        integer slot;
        integer legal;
        integer pred [0:15];
        integer resid [0:15];
        integer W [0:15];
        integer lev [0:15];
        integer scan0_arr [0:15];
        integer cand_scan [0:15];
        integer cand_rec [0:15];
        integer qmod;
        integer qbits;
        integer f;
        integer i;
        integer p;
        integer mag;
        integer coeff_bits;
        integer mode_bits;
        integer syntax_bits;
        integer diff;
        integer ssd;
        reg [63:0] cand_cost;
        reg [63:0] rdoq_cost;
        reg [63:0] best_cost;
        begin
            best_mode = 2;
            best_cost = 64'hffff_ffff_ffff_ffff;
            for (i = 0; i < 16; i = i + 1) begin
                cand_scan[i] = 0;
                cand_rec[i] = 128;
                if (i == 0) begin best_scan0 = 0; best_rec0 = 128; end
                if (i == 1) begin best_scan1 = 0; best_rec1 = 128; end
                if (i == 2) begin best_scan2 = 0; best_rec2 = 128; end
                if (i == 3) begin best_scan3 = 0; best_rec3 = 128; end
                if (i == 4) begin best_scan4 = 0; best_rec4 = 128; end
                if (i == 5) begin best_scan5 = 0; best_rec5 = 128; end
                if (i == 6) begin best_scan6 = 0; best_rec6 = 128; end
                if (i == 7) begin best_scan7 = 0; best_rec7 = 128; end
                if (i == 8) begin best_scan8 = 0; best_rec8 = 128; end
                if (i == 9) begin best_scan9 = 0; best_rec9 = 128; end
                if (i == 10) begin best_scan10 = 0; best_rec10 = 128; end
                if (i == 11) begin best_scan11 = 0; best_rec11 = 128; end
                if (i == 12) begin best_scan12 = 0; best_rec12 = 128; end
                if (i == 13) begin best_scan13 = 0; best_rec13 = 128; end
                if (i == 14) begin best_scan14 = 0; best_rec14 = 128; end
                if (i == 15) begin best_scan15 = 0; best_rec15 = 128; end
            end
            for (slot = 0; slot < 9; slot = slot + 1) begin
                legal = 0;
                mode = 2;
                if (slot == 0) begin mode = 2; legal = 1; end
                else if (slot == 1) begin mode = 0; legal = avail_top; end
                else if (slot == 2) begin mode = 3; legal = avail_top; end
                else if (slot == 3) begin mode = 7; legal = avail_top; end
                else if (slot == 4) begin mode = 1; legal = avail_left; end
                else if (slot == 5) begin mode = 8; legal = avail_left; end
                else if (slot == 6) begin mode = 4; legal = avail_left && avail_top; end
                else if (slot == 7) begin mode = 5; legal = avail_left && avail_top; end
                else begin mode = 6; legal = avail_left && avail_top; end
                if (legal != 0) begin
                    predict4(mode, avail_left, avail_top, left0, left1, left2, left3, top0, top1, top2, top3, tl, tr0, tr1, tr2, tr3,
                             pred[0], pred[1], pred[2], pred[3], pred[4], pred[5], pred[6], pred[7],
                             pred[8], pred[9], pred[10], pred[11], pred[12], pred[13], pred[14], pred[15]);
                    resid[0]=src0-pred[0]; resid[1]=src1-pred[1]; resid[2]=src2-pred[2]; resid[3]=src3-pred[3];
                    resid[4]=src4-pred[4]; resid[5]=src5-pred[5]; resid[6]=src6-pred[6]; resid[7]=src7-pred[7];
                    resid[8]=src8-pred[8]; resid[9]=src9-pred[9]; resid[10]=src10-pred[10]; resid[11]=src11-pred[11];
                    resid[12]=src12-pred[12]; resid[13]=src13-pred[13]; resid[14]=src14-pred[14]; resid[15]=src15-pred[15];
                    dct4_task(resid[0], resid[1], resid[2], resid[3], resid[4], resid[5], resid[6], resid[7],
                              resid[8], resid[9], resid[10], resid[11], resid[12], resid[13], resid[14], resid[15],
                              W[0], W[1], W[2], W[3], W[4], W[5], W[6], W[7],
                              W[8], W[9], W[10], W[11], W[12], W[13], W[14], W[15]);
                    qmod = qp % 6;
                    qbits = 15 + (qp / 6);
                    f = (1 << qbits) / 3;
                    for (i = 0; i < 16; i = i + 1) begin
                        if (W[i] < 0) begin
                            mag = -W[i];
                        end else begin
                            mag = W[i];
                        end
                        lev[i] = ((mag * mf_const(qmod, pos_class(i))) + f) >> qbits;
                        if (W[i] < 0) begin
                            lev[i] = -lev[i];
                        end else begin
                            lev[i] = lev[i];
                        end
                    end
                    for (i = 0; i < 16; i = i + 1) begin
                        p = zigzag_pos(i);
                        scan0_arr[i] = lev[p];
                    end
                    rdoq_stage_task(W[0], W[1], W[2], W[3], W[4], W[5], W[6], W[7],
                                    W[8], W[9], W[10], W[11], W[12], W[13], W[14], W[15],
                                    scan0_arr[0], scan0_arr[1], scan0_arr[2], scan0_arr[3], scan0_arr[4], scan0_arr[5], scan0_arr[6], scan0_arr[7],
                                    scan0_arr[8], scan0_arr[9], scan0_arr[10], scan0_arr[11], scan0_arr[12], scan0_arr[13], scan0_arr[14], scan0_arr[15],
                                    qp, nC,
                                    src0, src1, src2, src3, src4, src5, src6, src7, src8, src9, src10, src11, src12, src13, src14, src15,
                                    pred[0], pred[1], pred[2], pred[3], pred[4], pred[5], pred[6], pred[7],
                                    pred[8], pred[9], pred[10], pred[11], pred[12], pred[13], pred[14], pred[15],
                                    lam,
                                    cand_scan[0], cand_scan[1], cand_scan[2], cand_scan[3], cand_scan[4], cand_scan[5], cand_scan[6], cand_scan[7],
                                    cand_scan[8], cand_scan[9], cand_scan[10], cand_scan[11], cand_scan[12], cand_scan[13], cand_scan[14], cand_scan[15],
                                    cand_rec[0], cand_rec[1], cand_rec[2], cand_rec[3], cand_rec[4], cand_rec[5], cand_rec[6], cand_rec[7],
                                    cand_rec[8], cand_rec[9], cand_rec[10], cand_rec[11], cand_rec[12], cand_rec[13], cand_rec[14], cand_rec[15],
                                    coeff_bits, rdoq_cost);
                    ssd = 0;
                    for (i = 0; i < 16; i = i + 1) begin
                        if (i == 0) diff = src0 - cand_rec[i];
                        else if (i == 1) diff = src1 - cand_rec[i];
                        else if (i == 2) diff = src2 - cand_rec[i];
                        else if (i == 3) diff = src3 - cand_rec[i];
                        else if (i == 4) diff = src4 - cand_rec[i];
                        else if (i == 5) diff = src5 - cand_rec[i];
                        else if (i == 6) diff = src6 - cand_rec[i];
                        else if (i == 7) diff = src7 - cand_rec[i];
                        else if (i == 8) diff = src8 - cand_rec[i];
                        else if (i == 9) diff = src9 - cand_rec[i];
                        else if (i == 10) diff = src10 - cand_rec[i];
                        else if (i == 11) diff = src11 - cand_rec[i];
                        else if (i == 12) diff = src12 - cand_rec[i];
                        else if (i == 13) diff = src13 - cand_rec[i];
                        else if (i == 14) diff = src14 - cand_rec[i];
                        else diff = src15 - cand_rec[i];
                        ssd = ssd + (diff * diff);
                    end
                    if (mode == mpm) begin
                        mode_bits = 1;
                    end else begin
                        mode_bits = 4;
                    end
                    syntax_bits = mode_bits + coeff_bits;
                    cand_cost = ({42'd0, ssd[21:0]} << 20) + (lam * {48'd0, syntax_bits[15:0]});
                    if (cand_cost < best_cost) begin
                        best_cost = cand_cost;
                        best_mode = mode;
                        best_scan0=cand_scan[0]; best_scan1=cand_scan[1]; best_scan2=cand_scan[2]; best_scan3=cand_scan[3];
                        best_scan4=cand_scan[4]; best_scan5=cand_scan[5]; best_scan6=cand_scan[6]; best_scan7=cand_scan[7];
                        best_scan8=cand_scan[8]; best_scan9=cand_scan[9]; best_scan10=cand_scan[10]; best_scan11=cand_scan[11];
                        best_scan12=cand_scan[12]; best_scan13=cand_scan[13]; best_scan14=cand_scan[14]; best_scan15=cand_scan[15];
                        best_rec0=cand_rec[0]; best_rec1=cand_rec[1]; best_rec2=cand_rec[2]; best_rec3=cand_rec[3];
                        best_rec4=cand_rec[4]; best_rec5=cand_rec[5]; best_rec6=cand_rec[6]; best_rec7=cand_rec[7];
                        best_rec8=cand_rec[8]; best_rec9=cand_rec[9]; best_rec10=cand_rec[10]; best_rec11=cand_rec[11];
                        best_rec12=cand_rec[12]; best_rec13=cand_rec[13]; best_rec14=cand_rec[14]; best_rec15=cand_rec[15];
                    end else begin
                        best_cost = best_cost;
                    end
                end else begin
                    best_cost = best_cost;
                end
            end
        end
    endtask

    function [213:0] pack_syntax;
        input [3:0] sb_idx;
        input [3:0] best_mode;
        input [3:0] mpm;
        input [4:0] nC;
        input integer q0;  input integer q1;  input integer q2;  input integer q3;
        input integer q4;  input integer q5;  input integer q6i; input integer q7;
        input integer q8;  input integer q9;  input integer q10; input integer q11;
        input integer q12; input integer q13; input integer q14; input integer q15;
        integer scan [0:15];
        integer i;
        integer nz;
        reg [213:0] word;
        begin
            scan[0]=q0; scan[1]=q1; scan[2]=q2; scan[3]=q3; scan[4]=q4; scan[5]=q5; scan[6]=q6i; scan[7]=q7;
            scan[8]=q8; scan[9]=q9; scan[10]=q10; scan[11]=q11; scan[12]=q12; scan[13]=q13; scan[14]=q14; scan[15]=q15;
            nz = 0;
            for (i = 0; i < 16; i = i + 1) begin
                if (scan[i] != 0) begin
                    nz = nz + 1;
                end else begin
                    nz = nz;
                end
            end
            word = 214'd0;
            word[213:210] = sb_idx;
            word[209:206] = best_mode;
            word[205:202] = mpm;
            word[201:197] = nC;
            word[196:192] = nz[4:0];
            // Storage surgery: loop with runtime-index part-select unrolled to
            // constant 12-bit field slices (field i -> word[191-12i : 180-12i]).
            word[191:180] = scan[0][11:0];
            word[179:168] = scan[1][11:0];
            word[167:156] = scan[2][11:0];
            word[155:144] = scan[3][11:0];
            word[143:132] = scan[4][11:0];
            word[131:120] = scan[5][11:0];
            word[119:108] = scan[6][11:0];
            word[107:96]  = scan[7][11:0];
            word[95:84]   = scan[8][11:0];
            word[83:72]   = scan[9][11:0];
            word[71:60]   = scan[10][11:0];
            word[59:48]   = scan[11][11:0];
            word[47:36]   = scan[12][11:0];
            word[35:24]   = scan[13][11:0];
            word[23:12]   = scan[14][11:0];
            word[11:0]    = scan[15][11:0];
            pack_syntax = word;
        end
    endfunction

    always @(posedge clk) begin : seq_logic
        integer i;
        integer sb;
        integer sby;
        integer sbx;
        integer base;
        integer mbx;
        integer mby;
        integer x;
        integer y;
        integer gx;
        integer gy;
        integer row_base;
        integer col;
        integer src [0:15];
        integer left [0:3];
        integer top [0:3];
        integer tr [0:3];
        integer tl;
        integer avail_left;
        integer avail_top;
        integer avail_tr;
        integer nl;
        integer nt;
        integer nC;
        integer left_mode;
        integer top_mode;
        integer mpm;
        integer mode;
        integer scan [0:15];
        integer recon [0:15];
        integer nz;
        integer mbi;
        if (!rst_n) begin
            state_q <= S_IDLE;
            cfg_seen_q <= 1'b0;
            active_w_q <= 10'd0;
            active_h_q <= 10'd0;
            qp_reg_q <= 6'd0;
            mb_cols_reg_q <= 6'd0;
            collect_idx_q <= 5'd0;
            compute_idx_q <= 5'd0;
            emit_idx_q <= 5'd0;
            cur_mbx_q <= 6'd0;
            cur_mby_q <= 5'd0;
            cur_last_q <= 1'b0;
            out_valid_q <= 1'b0;
            out_data_q <= 214'd0;
            out_last_q <= 1'b0;
            st_valid_q <= 1'b0;
            st_data_q <= 32'd0;
            st_last_q <= 1'b0;
            // Addressed memories (mb_src_mem, syntax_mem, top_line_word) are not
            // reset: each location is written before it is read (write-before-read
            // discipline), so the prior flat-vector reset-to-0 had no observable
            // effect.  This matches the storage-lint suggested-fix idiom for
            // addressed memories and enables clean RAM inference at synthesis.
            for (i = 0; i < 16; i = i + 1) begin
                left_edge_q[i] <= 8'd128;
            end
            for (i = 0; i < 21; i = i + 1) begin
                row_top_ref_q[i] <= 8'd128;
            end
            for (i = 0; i < 160; i = i + 1) begin
                top_mode_ctx_q[i] <= 4'd15;
                top_nz_ctx_q[i] <= 5'd31;
                top_ctx_valid_q[i] <= 1'b0;
            end
            for (i = 0; i < 4; i = i + 1) begin
                left_mode_ctx_q[i] <= 4'd2;
                left_nz_ctx_q[i] <= 5'd0;
                left_ctx_valid_q[i] <= 1'b0;
            end
        end else begin
            if (syntax_handshake_w) begin
                out_valid_q <= 1'b0;
                out_last_q <= 1'b0;
            end else begin
                out_valid_q <= out_valid_q;
                out_last_q <= out_last_q;
            end
            if (status_handshake_w) begin
                st_valid_q <= 1'b0;
                st_last_q <= 1'b0;
            end else begin
                st_valid_q <= st_valid_q;
                st_last_q <= st_last_q;
            end

            if (cfg_handshake_w) begin
                active_w_q <= s_axis_frame_cfg_tdata[31:22];
                active_h_q <= s_axis_frame_cfg_tdata[21:12];
                if (s_axis_frame_cfg_tdata[11:6] > 6'd51) begin
                    qp_reg_q <= 6'd51;
                end else begin
                    qp_reg_q <= s_axis_frame_cfg_tdata[11:6];
                end
                mb_cols_reg_q <= (s_axis_frame_cfg_tdata[31:22] + 10'd15) >> 4;
                cfg_seen_q <= 1'b1;
                if (!st_valid_q) begin
                    st_data_q <= pack_status(EVENT_CFG, 2'd0, 10'd0, 4'd0, 7'd0);
                    st_valid_q <= 1'b1;
                    st_last_q <= 1'b1;
                end else begin
                    st_data_q <= st_data_q;
                end
                for (i = 0; i < 4; i = i + 1) begin
                    left_mode_ctx_q[i] <= 4'd2;
                    left_nz_ctx_q[i] <= 5'd0;
                    left_ctx_valid_q[i] <= 1'b0;
                end
            end else begin
                cfg_seen_q <= cfg_seen_q;
            end

            if (state_q == S_IDLE) begin
                collect_idx_q <= 5'd0;
                compute_idx_q <= 5'd0;
                emit_idx_q <= 5'd0;
                if (mb_handshake_w) begin
                    sb = s_axis_mb_tdata[14:11];
                    cur_mbx_q <= s_axis_mb_tdata[25:20];
                    cur_mby_q <= s_axis_mb_tdata[19:15];
                    cur_last_q <= s_axis_mb_tlast | s_axis_mb_tdata[0];
                    base = sb * 16;
                    for (i = 0; i < 16; i = i + 1) begin
                        mb_src_mem[base + i] <= s_axis_mb_tdata[153 - (i * 8) -: 8];
                    end
                    collect_idx_q <= 5'd1;
                    state_q <= S_COLLECT;
                end else begin
                    state_q <= S_IDLE;
                    cur_mbx_q <= cur_mbx_q;
                    cur_mby_q <= cur_mby_q;
                    cur_last_q <= cur_last_q;
                end
            end else if (state_q == S_COLLECT) begin
                if (mb_handshake_w) begin
                    sb = s_axis_mb_tdata[14:11];
                    base = sb * 16;
                    if (sb == 15) begin
                        cur_last_q <= s_axis_mb_tlast | s_axis_mb_tdata[0];
                    end else begin
                        cur_last_q <= cur_last_q;
                    end
                    for (i = 0; i < 16; i = i + 1) begin
                        mb_src_mem[base + i] <= s_axis_mb_tdata[153 - (i * 8) -: 8];
                    end
                    if (collect_idx_q == 5'd15) begin
                        compute_idx_q <= 5'd0;
                        state_q <= S_COMPUTE;
                    end else begin
                        collect_idx_q <= collect_idx_q + 5'd1;
                        state_q <= S_COLLECT;
                    end
                end else begin
                    state_q <= S_COLLECT;
                    collect_idx_q <= collect_idx_q;
                end
            end else if (state_q == S_COMPUTE) begin
                sb = compute_idx_q;
                sby = sb >> 2;
                sbx = sb & 3;
                mbx = cur_mbx_q;
                mby = cur_mby_q;
                x = (mbx * 16) + (sbx * 4);
                y = (mby * 16) + (sby * 4);
                gx = x >> 2;
                gy = y >> 2;
                row_base = mbx * 16;
                if (sbx == 0) begin
                    for (i = 0; i < 21; i = i + 1) begin
                        col = row_base + i - 1;
                        if ((col >= 0) && (col < MAX_W)) begin
                            row_top_ref_q[i] <= top_line_word[col >> 2][((col & 3) * 8) +: 8];
                        end else begin
                            row_top_ref_q[i] <= 8'd128;
                        end
                    end
                end else begin
                    for (i = 0; i < 21; i = i + 1) begin
                        row_top_ref_q[i] <= row_top_ref_q[i];
                    end
                end
                base = sb * 16;
                for (i = 0; i < 16; i = i + 1) begin
                    src[i] = mb_src_mem[base + i];
                end
                avail_left = (x > 0) ? 1 : 0;
                avail_top = (y > 0) ? 1 : 0;
                if (sbx == 3) begin
                    avail_tr = 0;
                end else if (sby == 0) begin
                    if ((avail_top != 0) && ((x + 4) < active_w_q)) begin
                        avail_tr = 1;
                    end else begin
                        avail_tr = 0;
                    end
                end else begin
                    avail_tr = 1;
                end
                for (i = 0; i < 4; i = i + 1) begin
                    if (avail_left != 0) begin
                        left[i] = left_edge_q[(sby * 4) + i];
                    end else begin
                        left[i] = 128;
                    end
                    if (avail_top != 0) begin
                        top[i] = (sbx == 0) ? top_line_word[(x + i) >> 2][(((x + i) & 3) * 8) +: 8] : row_top_ref_q[1 + (sbx * 4) + i];
                    end else begin
                        top[i] = 128;
                    end
                    if (avail_tr != 0) begin
                        tr[i] = (sbx == 0) ? top_line_word[(x + 4 + i) >> 2][(((x + 4 + i) & 3) * 8) +: 8] : row_top_ref_q[1 + (sbx * 4) + 4 + i];
                    end else if (avail_top != 0) begin
                        tr[i] = (sbx == 0) ? top_line_word[(x + 3) >> 2][(((x + 3) & 3) * 8) +: 8] : row_top_ref_q[1 + (sbx * 4) + 3];
                    end else begin
                        tr[i] = 128;
                    end
                end
                if ((avail_left != 0) && (avail_top != 0)) begin
                    if (sbx == 0) begin
                        tl = top_line_word[(x - 1) >> 2][(((x - 1) & 3) * 8) +: 8];
                    end else begin
                        tl = row_top_ref_q[sbx * 4];
                    end
                end else begin
                    tl = 128;
                end
                if ((gx > 0) && left_ctx_valid_q[sby]) begin
                    nl = left_nz_ctx_q[sby];
                    left_mode = left_mode_ctx_q[sby];
                end else begin
                    nl = 0;
                    left_mode = 2;
                end
                if ((gy > 0) && top_ctx_valid_q[gx]) begin
                    nt = top_nz_ctx_q[gx];
                    top_mode = top_mode_ctx_q[gx];
                end else begin
                    nt = 0;
                    top_mode = 2;
                end
                if ((gx > 0) && (gy > 0)) begin
                    nC = (nl + nt + 1) >> 1;
                end else if (gx > 0) begin
                    nC = nl;
                end else if (gy > 0) begin
                    nC = nt;
                end else begin
                    nC = 0;
                end
                if ((avail_left != 0) && (avail_top != 0)) begin
                    if (left_mode < top_mode) begin
                        mpm = left_mode;
                    end else begin
                        mpm = top_mode;
                    end
                end else begin
                    mpm = 2;
                end
                encode4_stage_task(src[0], src[1], src[2], src[3], src[4], src[5], src[6], src[7],
                                   src[8], src[9], src[10], src[11], src[12], src[13], src[14], src[15],
                                   left[0], left[1], left[2], left[3], top[0], top[1], top[2], top[3],
                                   tl, tr[0], tr[1], tr[2], tr[3],
                                   avail_left, avail_top, qp_reg_q, nC, lambda_for_qp(qp_reg_q), mpm,
                                   mode,
                                   scan[0], scan[1], scan[2], scan[3], scan[4], scan[5], scan[6], scan[7],
                                   scan[8], scan[9], scan[10], scan[11], scan[12], scan[13], scan[14], scan[15],
                                   recon[0], recon[1], recon[2], recon[3], recon[4], recon[5], recon[6], recon[7],
                                   recon[8], recon[9], recon[10], recon[11], recon[12], recon[13], recon[14], recon[15]);
                // x = (mbx*16)+(sbx*4) is always a multiple of 4, so the four
                // reconstructed bottom-row bytes x..x+3 land in exactly ONE
                // aligned 32-bit word (word x>>2; byte x+k in lane k).  One
                // whole-word write replaces the old per-byte loop, bit-identical
                // under the little-endian lane layout.
                top_line_word[x >> 2] <= {recon[15][7:0], recon[14][7:0], recon[13][7:0], recon[12][7:0]};
                for (i = 0; i < 4; i = i + 1) begin
                    left_edge_q[(sby * 4) + i] <= recon[(i * 4) + 3][7:0];
                end
                nz = 0;
                for (i = 0; i < 16; i = i + 1) begin
                    if (scan[i] != 0) begin
                        nz = nz + 1;
                    end else begin
                        nz = nz;
                    end
                end
                left_mode_ctx_q[sby] <= mode[3:0];
                left_nz_ctx_q[sby] <= nz[4:0];
                left_ctx_valid_q[sby] <= 1'b1;
                top_mode_ctx_q[gx] <= mode[3:0];
                top_nz_ctx_q[gx] <= nz[4:0];
                top_ctx_valid_q[gx] <= 1'b1;
                syntax_mem[sb] <= pack_syntax(compute_idx_q[3:0], mode[3:0], mpm[3:0], nC[4:0],
                                                                           scan[0], scan[1], scan[2], scan[3], scan[4], scan[5], scan[6], scan[7],
                                                                           scan[8], scan[9], scan[10], scan[11], scan[12], scan[13], scan[14], scan[15]);
                if (compute_idx_q == 5'd15) begin
                    emit_idx_q <= 5'd0;
                    state_q <= S_EMIT;
                    if (!st_valid_q) begin
                        mbi = (mby * mb_cols_reg_q) + mbx;
                        st_data_q <= pack_status(EVENT_MB_COMMIT, 2'd0, mbi[9:0], 4'd15, 7'd0);
                        st_valid_q <= 1'b1;
                        st_last_q <= 1'b1;
                    end else begin
                        st_data_q <= st_data_q;
                    end
                end else begin
                    compute_idx_q <= compute_idx_q + 5'd1;
                    state_q <= S_COMPUTE;
                end
            end else if (state_q == S_EMIT) begin
                if ((!out_valid_q) || syntax_handshake_w) begin
                    out_data_q <= syntax_mem[emit_idx_q];
                    out_valid_q <= 1'b1;
                    out_last_q <= cur_last_q && (emit_idx_q == 5'd15);
                    if (emit_idx_q == 5'd15) begin
                        state_q <= S_IDLE;
                        collect_idx_q <= 5'd0;
                        if (cur_last_q && !st_valid_q) begin
                            st_data_q <= pack_status(EVENT_FRAME_DONE, 2'd0, 10'd0, 4'd15, 7'd0);
                            st_valid_q <= 1'b1;
                            st_last_q <= 1'b1;
                        end else begin
                            st_data_q <= st_data_q;
                        end
                    end else begin
                        emit_idx_q <= emit_idx_q + 5'd1;
                        state_q <= S_EMIT;
                    end
                end else begin
                    state_q <= S_EMIT;
                    emit_idx_q <= emit_idx_q;
                end
            end else begin
                state_q <= S_IDLE;
            end
        end
    end

endmodule
