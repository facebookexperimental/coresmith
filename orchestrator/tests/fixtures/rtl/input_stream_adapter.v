/*
 * Block name: input_stream_adapter
 *
 * Description:
 *   AXI-Stream ingress adapter for 8-bit luma pixels. The block accepts
 *   external pixel beats, latches first-beat frame metadata from the canonical
 *   MSB-first TUSER record, validates width/height/QP/reserved fields, emits
 *   one frame_cfg beat for legal frames, forwards active pixels unchanged, and
 *   reports ingress lifecycle/protocol status events.
 *
 * I/O ports:
 *   clk, rst_n                         : single clock, synchronous active-low reset
 *   pixel_in_tdata/tvalid/tready/tlast : AXI-Stream slave pixel input
 *   pixel_in_tuser[31:0]               : {active_width[9:0], active_height[9:0],
 *                                         qp[5:0], frame_start, reserved_zero[4:0]}
 *   m_axis_pixel_*                     : AXI-Stream master active luma output
 *   m_axis_frame_cfg_*                 : AXI-Stream master canonical frame config
 *   m_axis_status_event_*              : AXI-Stream master status-event output
 */
module input_stream_adapter (
    input  wire        clk,
    input  wire        rst_n,

    input  wire [7:0]  pixel_in_tdata,
    input  wire        pixel_in_tvalid,
    output wire        pixel_in_tready,
    input  wire        pixel_in_tlast,
    input  wire [31:0] pixel_in_tuser,

    output wire [7:0]  m_axis_pixel_tdata,
    output wire        m_axis_pixel_tvalid,
    input  wire        m_axis_pixel_tready,
    output wire        m_axis_pixel_tlast,

    output wire [31:0] m_axis_frame_cfg_tdata,
    output wire        m_axis_frame_cfg_tvalid,
    input  wire        m_axis_frame_cfg_tready,
    output wire        m_axis_frame_cfg_tlast,

    output wire [31:0] m_axis_status_event_tdata,
    output wire        m_axis_status_event_tvalid,
    input  wire        m_axis_status_event_tready,
    output wire        m_axis_status_event_tlast
);

    localparam [1:0] STATE_IDLE   = 2'd0;
    localparam [1:0] STATE_ACTIVE = 2'd1;
    localparam [1:0] STATE_DONE   = 2'd2;

    localparam [9:0] MAX_WIDTH_U10  = 10'd640;
    localparam [9:0] MAX_HEIGHT_U10 = 10'd360;
    localparam [5:0] MAX_QP_U6      = 6'd51;
    localparam [18:0] MAX_PIXELS_U19 = 19'd230400;

    localparam [5:0] EVENT_FIRST_BEAT       = 6'd1;
    localparam [5:0] EVENT_ILLEGAL_METADATA = 6'd2;
    localparam [5:0] EVENT_PREMATURE_TLAST  = 6'd3;
    localparam [5:0] EVENT_LATE_TLAST       = 6'd4;
    localparam [5:0] EVENT_INGRESS_COMPLETE = 6'd5;

    localparam [2:0] BLOCK_ID_INPUT_STREAM_ADAPTER = 3'd0;
    localparam [1:0] SEV_INFO    = 2'd0;
    localparam [1:0] SEV_WARNING = 2'd1;
    localparam [1:0] SEV_ERROR   = 2'd2;

    localparam [6:0] ERR_NONE             = 7'd0;
    localparam [6:0] ERR_BAD_WIDTH        = 7'd1;
    localparam [6:0] ERR_BAD_HEIGHT       = 7'd2;
    localparam [6:0] ERR_BAD_QP           = 7'd3;
    localparam [6:0] ERR_RESERVED_NONZERO = 7'd4;
    localparam [6:0] ERR_TLAST_EARLY      = 7'd5;
    localparam [6:0] ERR_TLAST_LATE       = 7'd6;

    reg  [1:0]  state_q;
    reg         in_frame_q;
    reg         frame_done_q;

    reg  [9:0]  active_width_q;
    reg  [9:0]  active_height_q;
    reg  [5:0]  qp_q;
    reg  [18:0] expected_pixels_q;
    reg  [18:0] pixel_count_q;

    reg         pixel_valid_q;
    reg  [7:0]  pixel_data_q;
    reg         pixel_last_q;

    reg         cfg_valid_q;
    reg  [31:0] cfg_data_q;
    reg         cfg_last_q;

    reg  [31:0] status_data0_q;
    reg  [31:0] status_data1_q;
    reg  [31:0] status_data2_q;
    reg  [31:0] status_data3_q;
    reg         status_last0_q;
    reg         status_last1_q;
    reg         status_last2_q;
    reg         status_last3_q;
    reg  [1:0]  status_rd_q;
    reg  [1:0]  status_wr_q;
    reg  [2:0]  status_count_q;

    reg         local_error_seen_q;

    wire        pixel_room;
    wire        cfg_room;
    wire        status_room;
    wire        pixel_in_fire;
    wire        pop_pixel;
    wire        pop_cfg;
    wire        pop_status;

    wire [9:0]  raw_width;
    wire [9:0]  raw_height;
    wire [5:0]  raw_qp;
    wire        raw_frame_start;
    wire [4:0]  raw_reserved;
    wire        bad_width;
    wire        bad_height;
    wire        bad_qp;
    wire        bad_reserved;
    wire        metadata_legal;

    wire [19:0] raw_product_full;
    wire [18:0] raw_expected_pixels;
    wire [19:0] active_pixel_count_plus_one_full;
    wire [18:0] active_pixel_count_plus_one;
    wire        active_last_expected;
    wire        first_pixel_is_last;
    wire        first_pixel_early_tlast;
    wire        active_pixel_early_tlast;
    wire        active_pixel_late_tlast;

    reg  [1:0]  state_d;
    reg         in_frame_d;
    reg         frame_done_d;
    reg  [9:0]  active_width_d;
    reg  [9:0]  active_height_d;
    reg  [5:0]  qp_d;
    reg  [18:0] expected_pixels_d;
    reg  [18:0] pixel_count_d;
    reg         pixel_valid_d;
    reg  [7:0]  pixel_data_d;
    reg         pixel_last_d;
    reg         cfg_valid_d;
    reg  [31:0] cfg_data_d;
    reg         cfg_last_d;
    reg  [31:0] status_data0_d;
    reg  [31:0] status_data1_d;
    reg  [31:0] status_data2_d;
    reg  [31:0] status_data3_d;
    reg         status_last0_d;
    reg         status_last1_d;
    reg         status_last2_d;
    reg         status_last3_d;
    reg  [1:0]  status_rd_d;
    reg  [1:0]  status_wr_d;
    reg  [2:0]  status_count_d;
    reg         local_error_seen_d;

    reg  [31:0] event_word0;
    reg  [31:0] event_word1;
    reg  [31:0] event_word2;
    reg         event_req0;
    reg         event_req1;
    reg         event_req2;
    reg  [1:0]  event_push_count;

    /* WaveKit-visible metadata extraction follows the MyHDL hardware model:
     * output samples consume pixel_in_tdata[7:0]; metadata consumes
     * pixel_in_tuser[31:0] as the canonical frame_cfg field list.
     */
    assign raw_width       = pixel_in_tuser[31:22];
    assign raw_height      = pixel_in_tuser[21:12];
    assign raw_qp          = pixel_in_tuser[11:6];
    assign raw_frame_start = pixel_in_tuser[5];
    assign raw_reserved    = pixel_in_tuser[4:0];

    assign bad_width        = (raw_width < 10'd1) || (raw_width > MAX_WIDTH_U10);
    assign bad_height       = (raw_height < 10'd1) || (raw_height > MAX_HEIGHT_U10);
    assign bad_qp           = (raw_qp > MAX_QP_U6);
    assign bad_reserved     = (raw_reserved != 5'd0);
    assign metadata_legal   = (!bad_width) && (!bad_height) && (!bad_qp) && (!bad_reserved);

    assign raw_product_full = raw_width * raw_height;
    assign raw_expected_pixels = raw_product_full[18:0];
    assign active_pixel_count_plus_one_full = {1'b0, pixel_count_q} + 20'd1;
    assign active_pixel_count_plus_one = active_pixel_count_plus_one_full[18:0];
    assign active_last_expected = (active_pixel_count_plus_one >= expected_pixels_q);
    assign first_pixel_is_last = (raw_expected_pixels <= 19'd1);
    assign first_pixel_early_tlast = pixel_in_tlast && (raw_expected_pixels > 19'd1);
    assign active_pixel_early_tlast = pixel_in_tlast && (!active_last_expected);
    assign active_pixel_late_tlast = (!pixel_in_tlast) && active_last_expected;

    assign pixel_room = (!pixel_valid_q) || m_axis_pixel_tready;
    assign cfg_room = (!cfg_valid_q) || m_axis_frame_cfg_tready;
    assign status_room = (status_count_q < 3'd4);
    assign pixel_in_tready = pixel_room && cfg_room && status_room && (!frame_done_q);
    assign pixel_in_fire = pixel_in_tvalid && pixel_in_tready;

    assign pop_pixel = pixel_valid_q && m_axis_pixel_tready;
    assign pop_cfg = cfg_valid_q && m_axis_frame_cfg_tready;
    assign pop_status = (status_count_q != 3'd0) && m_axis_status_event_tready;

    assign m_axis_pixel_tvalid = pixel_valid_q;
    assign m_axis_pixel_tdata = pixel_data_q;
    assign m_axis_pixel_tlast = pixel_last_q;

    assign m_axis_frame_cfg_tvalid = cfg_valid_q;
    assign m_axis_frame_cfg_tdata = cfg_data_q;
    assign m_axis_frame_cfg_tlast = cfg_last_q;

    assign m_axis_status_event_tvalid = (status_count_q != 3'd0);
    assign m_axis_status_event_tlast =
        (status_rd_q == 2'd0) ? status_last0_q :
        (status_rd_q == 2'd1) ? status_last1_q :
        (status_rd_q == 2'd2) ? status_last2_q :
                                status_last3_q;
    assign m_axis_status_event_tdata =
        (status_rd_q == 2'd0) ? status_data0_q :
        (status_rd_q == 2'd1) ? status_data1_q :
        (status_rd_q == 2'd2) ? status_data2_q :
                                status_data3_q;

    function [31:0] pack_frame_cfg;
        input [9:0] f_width;
        input [9:0] f_height;
        input [5:0] f_qp;
        input       f_frame_start;
        begin
            pack_frame_cfg = {f_width, f_height, f_qp, f_frame_start, 5'd0};
        end
    endfunction

    function [31:0] pack_status_event;
        input [5:0] f_event_id;
        input [1:0] f_severity;
        input [6:0] f_error_code;
        begin
            pack_status_event = {f_event_id, BLOCK_ID_INPUT_STREAM_ADAPTER,
                                 f_severity, 10'd0, 4'd0, f_error_code};
        end
    endfunction

    task push_status_event;
        input [31:0] t_event_word;
        begin
            if (status_count_d < 3'd4) begin
                if (status_wr_d == 2'd0) begin
                    status_data0_d = t_event_word;
                    status_last0_d = 1'b1;
                end else if (status_wr_d == 2'd1) begin
                    status_data1_d = t_event_word;
                    status_last1_d = 1'b1;
                end else if (status_wr_d == 2'd2) begin
                    status_data2_d = t_event_word;
                    status_last2_d = 1'b1;
                end else begin
                    status_data3_d = t_event_word;
                    status_last3_d = 1'b1;
                end
                status_wr_d = status_wr_d + 2'd1;
                status_count_d = status_count_d + 3'd1;
                event_push_count = event_push_count + 2'd1;
            end else begin
                status_wr_d = status_wr_d;
                status_count_d = status_count_d;
                event_push_count = event_push_count;
            end
        end
    endtask

    always @(*) begin
        state_d = state_q;
        in_frame_d = in_frame_q;
        frame_done_d = frame_done_q;
        active_width_d = active_width_q;
        active_height_d = active_height_q;
        qp_d = qp_q;
        expected_pixels_d = expected_pixels_q;
        pixel_count_d = pixel_count_q;
        pixel_valid_d = pixel_valid_q;
        pixel_data_d = pixel_data_q;
        pixel_last_d = pixel_last_q;
        cfg_valid_d = cfg_valid_q;
        cfg_data_d = cfg_data_q;
        cfg_last_d = cfg_last_q;
        status_data0_d = status_data0_q;
        status_data1_d = status_data1_q;
        status_data2_d = status_data2_q;
        status_data3_d = status_data3_q;
        status_last0_d = status_last0_q;
        status_last1_d = status_last1_q;
        status_last2_d = status_last2_q;
        status_last3_d = status_last3_q;
        status_rd_d = status_rd_q;
        status_wr_d = status_wr_q;
        status_count_d = status_count_q;
        local_error_seen_d = local_error_seen_q;

        event_word0 = 32'd0;
        event_word1 = 32'd0;
        event_word2 = 32'd0;
        event_req0 = 1'b0;
        event_req1 = 1'b0;
        event_req2 = 1'b0;
        event_push_count = 2'd0;

        if (pop_status) begin
            status_rd_d = status_rd_q + 2'd1;
            status_count_d = status_count_q - 3'd1;
        end else begin
            status_rd_d = status_rd_q;
            status_count_d = status_count_q;
        end

        if (pop_pixel) begin
            pixel_valid_d = 1'b0;
            pixel_last_d = 1'b0;
        end else begin
            pixel_valid_d = pixel_valid_q;
            pixel_last_d = pixel_last_q;
        end

        if (pop_cfg) begin
            cfg_valid_d = 1'b0;
            cfg_last_d = 1'b0;
        end else begin
            cfg_valid_d = cfg_valid_q;
            cfg_last_d = cfg_last_q;
        end

        if (pixel_in_fire) begin
            if (!in_frame_q) begin
                active_width_d = (raw_width <= MAX_WIDTH_U10) ? raw_width : 10'd0;
                active_height_d = (raw_height <= MAX_HEIGHT_U10) ? raw_height : 10'd0;
                qp_d = raw_qp;
                expected_pixels_d = metadata_legal ? raw_expected_pixels : 19'd0;
                pixel_count_d = metadata_legal ? 19'd1 : 19'd0;
                in_frame_d = metadata_legal && (!first_pixel_is_last) && (!pixel_in_tlast);
                frame_done_d = metadata_legal && (first_pixel_is_last || pixel_in_tlast);

                if (metadata_legal && (!first_pixel_is_last) && (!pixel_in_tlast)) begin
                    state_d = STATE_ACTIVE;
                end else if (metadata_legal && (first_pixel_is_last || pixel_in_tlast)) begin
                    state_d = STATE_DONE;
                end else begin
                    state_d = STATE_IDLE;
                end

                cfg_data_d = metadata_legal ? pack_frame_cfg(raw_width, raw_height, raw_qp, raw_frame_start) : 32'd0;
                cfg_valid_d = metadata_legal;
                cfg_last_d = metadata_legal;

                if (metadata_legal) begin
                    event_word0 = pack_status_event(EVENT_FIRST_BEAT, SEV_INFO, ERR_NONE);
                    event_req0 = 1'b1;
                end else begin
                    if (bad_width) begin
                        event_word0 = pack_status_event(EVENT_ILLEGAL_METADATA, SEV_ERROR, ERR_BAD_WIDTH);
                    end else if (bad_height) begin
                        event_word0 = pack_status_event(EVENT_ILLEGAL_METADATA, SEV_ERROR, ERR_BAD_HEIGHT);
                    end else if (bad_qp) begin
                        event_word0 = pack_status_event(EVENT_ILLEGAL_METADATA, SEV_ERROR, ERR_BAD_QP);
                    end else begin
                        event_word0 = pack_status_event(EVENT_ILLEGAL_METADATA, SEV_ERROR, ERR_RESERVED_NONZERO);
                    end
                    event_req0 = 1'b1;
                    local_error_seen_d = 1'b1;
                end

                if (metadata_legal) begin
                    pixel_data_d = pixel_in_tdata;
                    pixel_valid_d = 1'b1;
                    pixel_last_d = first_pixel_is_last;
                    if (first_pixel_early_tlast) begin
                        event_word1 = pack_status_event(EVENT_PREMATURE_TLAST, SEV_ERROR, ERR_TLAST_EARLY);
                        event_req1 = 1'b1;
                        local_error_seen_d = 1'b1;
                    end else begin
                        event_word1 = 32'd0;
                        event_req1 = 1'b0;
                    end
                end else begin
                    pixel_data_d = pixel_data_d;
                    pixel_valid_d = pixel_valid_d;
                    pixel_last_d = pixel_last_d;
                    event_word1 = 32'd0;
                    event_req1 = 1'b0;
                end
            end else begin
                pixel_data_d = pixel_in_tdata;
                pixel_valid_d = 1'b1;
                pixel_last_d = active_last_expected;
                pixel_count_d = (active_pixel_count_plus_one <= MAX_PIXELS_U19) ?
                                active_pixel_count_plus_one : MAX_PIXELS_U19;

                if (active_pixel_early_tlast) begin
                    event_word0 = pack_status_event(EVENT_PREMATURE_TLAST, SEV_ERROR, ERR_TLAST_EARLY);
                    event_req0 = 1'b1;
                    local_error_seen_d = 1'b1;
                end else begin
                    event_word0 = 32'd0;
                    event_req0 = 1'b0;
                end

                if (active_last_expected) begin
                    in_frame_d = 1'b0;
                    frame_done_d = 1'b1;
                    state_d = STATE_DONE;
                    if (active_pixel_late_tlast) begin
                        event_word1 = pack_status_event(EVENT_LATE_TLAST, SEV_WARNING, ERR_TLAST_LATE);
                        event_req1 = 1'b1;
                        local_error_seen_d = 1'b1;
                    end else begin
                        event_word1 = 32'd0;
                        event_req1 = 1'b0;
                    end
                    event_word2 = pack_status_event(EVENT_INGRESS_COMPLETE, SEV_INFO, ERR_NONE);
                    event_req2 = 1'b1;
                end else begin
                    in_frame_d = in_frame_q;
                    frame_done_d = frame_done_q;
                    state_d = STATE_ACTIVE;
                    event_word1 = 32'd0;
                    event_req1 = 1'b0;
                    event_word2 = 32'd0;
                    event_req2 = 1'b0;
                end
            end
        end else begin
            event_word0 = 32'd0;
            event_word1 = 32'd0;
            event_word2 = 32'd0;
            event_req0 = 1'b0;
            event_req1 = 1'b0;
            event_req2 = 1'b0;
        end

        if (event_req0) begin
            push_status_event(event_word0);
        end else begin
            status_count_d = status_count_d;
        end

        if (event_req1) begin
            push_status_event(event_word1);
        end else begin
            status_count_d = status_count_d;
        end

        if (event_req2) begin
            push_status_event(event_word2);
        end else begin
            status_count_d = status_count_d;
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            state_q <= STATE_IDLE;
            in_frame_q <= 1'b0;
            frame_done_q <= 1'b0;
            active_width_q <= 10'd0;
            active_height_q <= 10'd0;
            qp_q <= 6'd0;
            expected_pixels_q <= 19'd0;
            pixel_count_q <= 19'd0;
            pixel_valid_q <= 1'b0;
            pixel_data_q <= 8'd0;
            pixel_last_q <= 1'b0;
            cfg_valid_q <= 1'b0;
            cfg_data_q <= 32'd0;
            cfg_last_q <= 1'b0;
            status_data0_q <= 32'd0;
            status_data1_q <= 32'd0;
            status_data2_q <= 32'd0;
            status_data3_q <= 32'd0;
            status_last0_q <= 1'b0;
            status_last1_q <= 1'b0;
            status_last2_q <= 1'b0;
            status_last3_q <= 1'b0;
            status_rd_q <= 2'd0;
            status_wr_q <= 2'd0;
            status_count_q <= 3'd0;
            local_error_seen_q <= 1'b0;
        end else begin
            state_q <= state_d;
            in_frame_q <= in_frame_d;
            frame_done_q <= frame_done_d;
            active_width_q <= active_width_d;
            active_height_q <= active_height_d;
            qp_q <= qp_d;
            expected_pixels_q <= expected_pixels_d;
            pixel_count_q <= pixel_count_d;
            pixel_valid_q <= pixel_valid_d;
            pixel_data_q <= pixel_data_d;
            pixel_last_q <= pixel_last_d;
            cfg_valid_q <= cfg_valid_d;
            cfg_data_q <= cfg_data_d;
            cfg_last_q <= cfg_last_d;
            status_data0_q <= status_data0_d;
            status_data1_q <= status_data1_d;
            status_data2_q <= status_data2_d;
            status_data3_q <= status_data3_d;
            status_last0_q <= status_last0_d;
            status_last1_q <= status_last1_d;
            status_last2_q <= status_last2_d;
            status_last3_q <= status_last3_d;
            status_rd_q <= status_rd_d;
            status_wr_q <= status_wr_d;
            status_count_q <= status_count_d;
            local_error_seen_q <= local_error_seen_d;
        end
    end

endmodule
