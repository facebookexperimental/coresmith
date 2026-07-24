## 1. Block Overview

**Block name:** `intra_rd_encode_core`

`intra_rd_encode_core` is the tier-3 codec-style Intra4x4 luma rate-distortion engine. It consumes one serialized 4x4 sub-block record per `mb_axis` AXI-Stream beat, evaluates the video codec Intra4x4 modes 0 through 8 with one shared sequential candidate datapath, selects the minimum RD-cost legal candidate, commits the selected reconstruction into closed-loop predictor state, updates MPM and nC contexts, and emits one committed 214-bit `mb_syntax_axis` record per sub-block for `entropy_bitstream_engine`.

Latency is fixed at **472 non-stalled cycles** from the cycle a sub-block is popped by the RD sequencer to the corresponding syntax FIFO enqueue. Throughput is **1 sub-block per 472 non-stalled cycles** because the nine candidate modes reuse one datapath. The candidate slot is 50 cycles and has registered boundaries for prediction, residual, forward transform, quant/dequant, inverse transform, reconstruction, SSD, entropy coding bit count, cost multiply, cost add, and best compare.

The interface protocol is **AXI-Stream** because the ERS and canonical edge contracts explicitly define `frame_cfg_axis`, `mb_axis`, `mb_syntax_axis`, and `status_event_axis` as packetized ready/valid streams with `tlast`. Reset is synchronous active-low `rst_n`, matching the single 50 MHz clock-domain convention in the ERS.

This revision addresses the memory-budget rejection by shrinking reconstruction feedback to the **true raster dependency window** consumed by Intra4x4 prediction: one packed top-reference line across the padded width, row-start top-reference latches, and left-edge registers. A 16-line or full-frame random reconstruction store is not required for this block's future predictions, and the final decoded-frame comparison is performed from emitted syntax plus VCD reconstruction-write trace rather than by rereading a local frame store.

## 2. Interface Specification

All records use `msb_first_by_field_list`. A beat transfers on a rising edge where `tvalid && tready`. Once this block asserts an AXI-Stream master `tvalid`, the associated `tdata` and `tlast` remain stable until the downstream handshake occurs.

| Port | Direction | Width | Protocol | Description |
|---|---:|---:|---|---|
| `clk` | input | 1 | clock | Single 50 MHz clock for all logic and memory wrappers. |
| `rst_n` | input | 1 | reset | Synchronous active-low reset. |
| `s_axis_mb_tdata` | input | 154 | AXI-Stream slave | Serialized sub-block record from `macroblock_assembler`. |
| `s_axis_mb_tvalid` | input | 1 | AXI-Stream slave | Asserted by the producer when `s_axis_mb_tdata` and `s_axis_mb_tlast` are valid. |
| `s_axis_mb_tready` | output | 1 | AXI-Stream slave | Asserted when the 16-beat ingress FIFO can accept a beat and a legal frame config has been accepted. |
| `s_axis_mb_tlast` | input | 1 | AXI-Stream slave | Asserted only on the final sub-block beat of the frame; stored atomically with the 154-bit beat. |
| `s_axis_frame_cfg_tdata` | input | 32 | AXI-Stream slave | One frame configuration beat from `input_stream_adapter`. |
| `s_axis_frame_cfg_tvalid` | input | 1 | AXI-Stream slave | Producer valid for the one-beat frame configuration record. |
| `s_axis_frame_cfg_tready` | output | 1 | AXI-Stream slave | Asserted in `STATE_WAIT_CFG` when the config skid is empty. |
| `s_axis_frame_cfg_tlast` | input | 1 | AXI-Stream slave | Must be 1 on the single config beat; 0 raises a status error event. |
| `m_axis_mb_syntax_tdata` | output | 214 | AXI-Stream master | One committed syntax record per 4x4 sub-block for `entropy_bitstream_engine`. |
| `m_axis_mb_syntax_tvalid` | output | 1 | AXI-Stream master | Registered valid from the 16-beat syntax FIFO; held until `m_axis_mb_syntax_tready`. |
| `m_axis_mb_syntax_tready` | input | 1 | AXI-Stream master | Downstream ready from `entropy_bitstream_engine`. |
| `m_axis_mb_syntax_tlast` | output | 1 | AXI-Stream master | Asserted only on the final committed sub-block syntax record of the frame. |
| `m_axis_status_event_tdata` | output | 32 | AXI-Stream master | Single-beat status event for `lifecycle_status_monitor`. |
| `m_axis_status_event_tvalid` | output | 1 | AXI-Stream master | Registered valid from the 4-event status FIFO; held until `m_axis_status_event_tready`. |
| `m_axis_status_event_tready` | input | 1 | AXI-Stream master | Lifecycle monitor ready. |
| `m_axis_status_event_tlast` | output | 1 | AXI-Stream master | Always 1 for each single-beat status event packet. |

`s_axis_frame_cfg_tdata` field packing:

| Field | Slice | Format |
|---|---|---|
| `active_width` | `[31:22]` | 10-bit unsigned integer, legal 1..640 |
| `active_height` | `[21:12]` | 10-bit unsigned integer, legal 1..360 |
| `qp` | `[11:6]` | 6-bit unsigned integer, legal 0..51 |
| `frame_start` | `[5]` | Boolean, must be 1 on a legal config beat |
| `reserved_zero` | `[4:0]` | Constant zero; checked by this block |

`s_axis_mb_tdata` field packing:

| Field | Slice | Format |
|---|---|---|
| `src_pixel_0`..`src_pixel_15` | `[153:146]`, `[145:138]`, ..., `[33:26]` | 16 unsigned 8-bit pixels, row-major `local_y*4+local_x` |
| `mb_x` | `[25:20]` | Unsigned 0..39 |
| `mb_y` | `[19:15]` | Unsigned 0..22 |
| `sb_idx` | `[14:11]` | Unsigned 0..15 |
| `avail_left` | `[10]` | Boolean |
| `avail_top` | `[9]` | Boolean |
| `avail_topright` | `[8]` | Boolean |
| `qp` | `[7:2]` | Unsigned 0..51, must equal `qp_frame_q` |
| `frame_first` | `[1]` | Boolean, first sub-block of the frame only |
| `mb_last` | `[0]` | Boolean, final sub-block of the frame only |

`m_axis_mb_syntax_tdata` field packing:

| Field | Slice | Format |
|---|---|---|
| `sb_idx` | `[213:210]` | Unsigned 0..15 |
| `best_mode` | `[209:206]` | Unsigned 0..8 |
| `mpm` | `[205:202]` | Unsigned 0..8 |
| `nC` | `[201:197]` | Unsigned 0..16 |
| `nz_count` | `[196:192]` | Unsigned 0..16 |
| `quant_level_zz0`..`quant_level_zz15` | `[191:180]`, `[179:168]`, ..., `[11:0]` | Signed 12-bit two's-complement integers in zig-zag order `[0,1,4,8,5,2,3,6,9,12,13,10,7,11,14,15]` |

`m_axis_status_event_tdata` field packing:

| Field | Slice | Format |
|---|---|---|
| `event_id` | `[31:26]` | 6-bit enumerated status event |
| `block_id` | `[25:23]` | 3-bit block id; `intra_rd_encode_core = 3'b011` |
| `severity` | `[22:21]` | 0=info, 1=warning, 2=error, 3=fatal |
| `mb_index` | `[20:11]` | Unsigned 0..919, or zero when not applicable |
| `subblock_index` | `[10:7]` | Unsigned 0..15, or zero when not applicable |
| `error_code` | `[6:0]` | 7-bit enumerated error code |

## 3. Microarchitecture

### 3.1 Top-Level Block Diagram

```text
                         +-----------------------------+
s_axis_frame_cfg ------->| 1-beat cfg skid + geometry  |----+
                         | QP / raster registers       |    |
                         +-----------------------------+    |
                                                            v
                         +-----------------------------+  +----------------+
s_axis_mb -------------> | 16-beat mb ingress FIFO     |->| RD sequencer   |
                         | 155b data+last, SRAM        |  | fixed FSM      |
                         +-----------------------------+  +-------+--------+
                                                                    |
             +--------------------+    +---------------------+      |
             | top mode/nC ctx    |<-->| context read/update |<-----+
             | 9b x 160, fpmem    |    +---------------------+
             +--------------------+
                                                                    |
             +--------------------+    +---------------------+      |
             | top_line_sram      |<-->| row-top prefetch +  |<-----+
             | 32b x 160          |    | left-edge feedback  |
             +--------------------+    +----------+----------+
                                                   |
                                      +------------v-------------+
                                      | one shared candidate     |
                                      | datapath, mode slots 0-8 |
                                      +------------+-------------+
                                                   |
                                      +------------v-------------+
                                      | argmin/best candidate    |
                                      | registers                |
                                      +------------+-------------+
                                                   |
              +------------------------------------+-------------------+
              |                                                        |
              v                                                        v
  +-----------------------------+                         +-----------------------------+
  | 16-beat syntax FIFO         |                         | 4-event status FIFO         |
  | 215b data+last, SRAM        |                         | 33b data+last, flops        |
  +-----------------------------+                         +-----------------------------+
              |                                                        |
              v                                                        v
 m_axis_mb_syntax                                      m_axis_status_event
```

### 3.2 Datapath

**Geometry derivation**

The block derives geometry from the accepted `frame_cfg_axis` beat:

- `active_width`: 10-bit unsigned, legal 1..640.
- `active_height`: 10-bit unsigned, legal 1..360.
- `qp_frame`: 6-bit unsigned, legal 0..51.
- `padded_width = (active_width + 15) & ~15`: 10-bit unsigned, 16..640.
- `padded_height = (active_height + 15) & ~15`: 9-bit unsigned, 16..368.
- `mb_cols = (active_width + 15) >> 4`: 6-bit unsigned, 1..40.
- `mb_rows = (active_height + 15) >> 4`: 5-bit unsigned, 1..23.
- `macroblocks_per_frame = mb_cols * mb_rows`: 10-bit unsigned, 1..920.
- Reconstructed x coordinate range: 0..639, 10 bits.
- Reconstructed y coordinate range: 0..367, 9 bits.
- 4x4 column context range: 0..159, 8 bits.
- Terminal maximum coordinate: `mb_x=39`, `mb_y=22`, `sb_idx=15`.

Sub-block coordinates:

- `sbx = sb_idx[1:0]`, `sby = sb_idx[3:2]`.
- `sub_x = {mb_x,4'b0000} + {sbx,2'b00}`: 10-bit unsigned, 4-pixel aligned.
- `sub_y = {mb_y,4'b0000} + {sby,2'b00}`: 9-bit unsigned.
- `mb_index = mb_y * mb_cols + mb_x`: 10-bit unsigned, range 0..919.

**Reconstruction feedback representation**

The previous rejected specs stored the full 16-line reconstruction window inside this block. That was not the true dependency window needed by raster Intra4x4 prediction. The revised representation is:

- `top_line_sram`: 160 words x 32 bits = 5120 bits. Word address is `x[9:2]`; lane mapping is `x[1:0] == 0 -> [31:24]`, `1 -> [23:16]`, `2 -> [15:8]`, `3 -> [7:0]`.
- `row_top_ref_q[0:19]`: twenty 8-bit registers, loaded at the first sub-block of each 4x4 sub-block row (`sbx=0`). It preserves the row above the current sub-block row across all four `sbx` positions so later commits cannot overwrite top-left/top references before they are consumed.
- `row_top_left_q`: one 8-bit register holding the pixel immediately above-left of the current sub-block row. For `sby==0 && mb_x>0`, it is carried from `next_mb_top_left_seed_q`, captured before the previous pixel_block overwrote that top-line column. For `sby>0 && sbx==0`, it is loaded from `left_edge_q[sby*4-1]`, the previous pixel_block's right-edge pixel on the row immediately above the current sub-block row.
- `left_edge_q[0:15]`: sixteen 8-bit registers holding the current left reference column for the pixel_block. At `sbx=0`, entries contain the right-edge reconstruction from the previous pixel_block in the same pixel_block row, or are ignored when `avail_left=0`. After each committed sub-block, the four entries for its `sby` are updated with the selected rightmost reconstructed pixels.
- `next_mb_top_left_seed_q`: 8-bit register captured from the old top-line word for the current pixel_block's rightmost column before top-line writes for the pixel_block. It becomes `row_top_left_q` for the next pixel_block's top row.

This structure is sufficient because every Intra4x4 prediction reference is either the immediate previous reconstructed row (`top` and `top-right`) or the immediate previous reconstructed column (`left`). There is no legal future prediction that rereads row `y-2` or older after row `y-1` has been latched for the current sub-block row. Thus a full 16-row random-access reconstruction store is oversized for this block's true dependency window.

**Input capture**

On each popped ingress FIFO beat, the sequencer latches:

- `src_pix_q[0:15]`: sixteen unsigned 8-bit row-major pixels.
- `mb_x_q[5:0]`, `mb_y_q[4:0]`, `sb_idx_q[3:0]`.
- `avail_left_q`, `avail_top_q`, `avail_topright_q`.
- `mb_qp_q[5:0]`, checked equal to `qp_frame_q`.
- `frame_first_q`, `mb_last_q`, `subblock_last_q`, where `subblock_last_q` is the stored FIFO `tlast`.

**Top-row prefetch and neighbor construction**

For each sub-block, the control schedule reserves a fixed neighbor setup window. On `sbx=0`, the block issues up to five registered reads from `top_line_sram` to capture `row_top_ref_q[0:19]` for x positions `mb_x*16` through `mb_x*16+19`. Addresses beyond `padded_width-1` are not issued; corresponding bytes load as 128 and `avail_topright_q` must be 0 for legal streams at the right boundary. On `sbx!=0`, the same cycles are NOPs and the existing `row_top_ref_q` row is reused.

Neighbor values are then selected as:

- `top_ref[0:3]`: when `avail_top_q=1`, `row_top_ref_q[sbx*4 + 0..3]`; otherwise 128.
- `topright_ref[0:3]`: when `avail_topright_q=1`, `row_top_ref_q[sbx*4 + 4..7]`; otherwise 128.
- `left_ref[0:3]`: when `avail_left_q=1`, `left_edge_q[sby*4 + 0..3]`; otherwise 128.
- `topleft_ref`: when `avail_left_q && avail_top_q`, `row_top_left_q` for `sbx=0`, otherwise `row_top_ref_q[sbx*4 - 1]`; otherwise 128. For `sbx=0`, `row_top_left_q` is sourced from the saved previous-row seed when `sby==0`, and from `left_edge_q[sby*4-1]` when `sby>0`.

All top-line SRAM reads are registered one-cycle reads. No predictor path consumes SRAM data in the same cycle as address issue. Unavailable references do not issue out-of-range SRAM reads and always substitute `8'd128`.

**Mode legality and prediction**

The mode slot counter visits mode indices 0..8 for every sub-block. Illegal modes still consume the fixed slot and force `candidate_cost = 64'h7fff_ffff_ffff_ffff`, so latency is deterministic. Mode 2 DC is always legal.

Mode legality:

- Mode 0 Vertical: `avail_top_q`.
- Mode 1 Horizontal: `avail_left_q`.
- Mode 2 DC: always legal.
- Mode 3 Diagonal Down Left: `avail_top_q && avail_topright_q`.
- Mode 4 Diagonal Down Right: `avail_left_q && avail_top_q`.
- Mode 5 Vertical Right: `avail_left_q && avail_top_q`.
- Mode 6 Horizontal Down: `avail_left_q && avail_top_q`.
- Mode 7 Vertical Left: `avail_top_q && avail_topright_q`.
- Mode 8 Horizontal Up: `avail_left_q`.

Prediction equations are the frozen the video codec Intra4x4 integer formulas for modes 0..8. All averages use unsigned round-half-up:

- Two-tap average: `(a + b + 1) >> 1`, 9-bit sum to 8-bit output.
- Three-tap weighted average: `(a + 2*b + c + 2) >> 2`, 10-bit sum to 8-bit output.
- DC with left and top: `(sum(left_ref[0:3]) + sum(top_ref[0:3]) + 4) >> 3`, 11-bit sum to 8-bit output.
- DC with top only: `(sum(top_ref[0:3]) + 2) >> 2`, 10-bit sum to 8-bit output.
- DC with left only: `(sum(left_ref[0:3]) + 2) >> 2`, 10-bit sum to 8-bit output.
- DC with neither: literal `8'd128`.

`pred_pix_q[0:15]` are unsigned 8-bit integers.

**Residual**

For each pixel lane `i`:

- Inputs: `src_pix_q[i]` and `pred_pix_q[i]`, unsigned integer 0..255.
- Operation: `residual[i] = signed({1'b0, src}) - signed({1'b0, pred})`.
- Range: -255..255.
- Width: signed 9-bit two's-complement in `residual_q[0:15]`.
- Overflow policy: range-guaranteed.

**Forward transform**

The transform is `Cf * residual * Cf.T`, with:

```text
Cf = [[ 1,  1,  1,  1],
      [ 2,  1, -1, -2],
      [ 1, -1, -1,  1],
      [ 1, -2,  2, -1]]
```

Stage `FWD_H` computes row partials. Each output is a signed sum of four signed 9-bit residuals multiplied by -2, -1, 1, or 2. Worst-case absolute value is `2*255 + 255 + 255 + 2*255 = 1530`, requiring 12 signed bits. `tmp_h_q[0:15]` is signed 12-bit.

Stage `FWD_V` computes column partials from `tmp_h_q`. Worst-case absolute value is `2*1530 + 1530 + 1530 + 2*1530 = 9180`, requiring 15 signed bits. `coeff_q[0:15]` is signed 16-bit. Overflow is range-guaranteed.

**Quantization, zig-zag scan, and dequantization**

The coefficient loop processes one coefficient per cycle for exactly 16 cycles in zig-zag order `[0,1,4,8,5,2,3,6,9,12,13,10,7,11,14,15]`.

The lambda, quant MF, dequant scale, and entropy coding length constants are implemented as compile-time constant decode logic (`localparam` vectors plus `case` functions or equivalent combinational constant muxes), not as addressed memories. This removes the previously priced lambda, quant-scale, and entropy coding length table memories from the memory ledger. The RTL must not instantiate a behavioral table memory for these constants.

For coefficient `c = coeff_q[pos]`:

- `abs_c`: unsigned 16-bit, range 0..9180.
- `qp_mod6`: 3-bit unsigned, 0..5.
- `qp_div6`: 4-bit unsigned, 0..8.
- `mf`: unsigned 16-bit from constant decode indexed by `{qp_mod6,pos}`.
- `dequant_scale`: unsigned 16-bit from constant decode indexed by `{qp_mod6,pos}`.
- `qbits = 15 + qp_div6`: 6-bit unsigned, range 15..23.
- `offset`: unsigned 32-bit golden intra dead-zone offset from QP.
- `product = abs_c * mf`: 16x16 -> unsigned 32-bit.
- `biased = product + offset`: unsigned 33-bit.
- `level_abs = biased >> qbits`: unsigned 18-bit.
- `level_signed_pre = sign(c) ? -level_abs : level_abs`: signed 19-bit.
- `quant_level_zz[k]`: signed 12-bit two's-complement. Legal golden vectors are range-guaranteed to fit -2048..2047. If exceeded, saturate to -2048 or 2047 and enqueue an error status.
- `nz_count`: 5-bit unsigned count of nonzero levels, range 0..16.

Dequantization in the same loop:

- `dq_product = signed(quant_level_zz[k]) * signed({1'b0,dequant_scale})`: 12x17 -> signed 29-bit.
- `dequant_coeff_pre = dq_product <<< qp_div6`: signed 38-bit intermediate.
- `dequant_q[pos]`: signed 20-bit after golden clipping/range check. Legal vectors are range-guaranteed; any clip raises an error status.

Signed widening uses sign extension. Signed right shifts are arithmetic. Quantization rounding is exactly `(abs*mf + offset) >> qbits`; no Verilog division is permitted.

**Inverse transform and reconstruction**

- `inv_tmp_q[0:15]`: signed 24-bit row/column partials. Four signed 20-bit dequant inputs with weights up to 2 fit in 24 bits.
- `rec_residual_q[0:15]`: signed 16-bit inverse residual after the golden final right shift and rounding bias.
- `sum_recon = signed({1'b0,pred_pix_q[i]}) + rec_residual_q[i]`: signed 17-bit.
- `recon_pix_q[i]`: unsigned 8-bit after saturating clip to 0..255.

After best-candidate selection, `best_recon_q[0:15]` writes predictor state:

- Four cycles write four 32-bit top-line words at addresses `(sub_x >> 2)` for each selected reconstructed row. Because `top_line_sram` stores the latest completed row for each x column, the last of the four row writes leaves row `sub_y+3` visible for the next sub-block row.
- The same commit updates `left_edge_q[sby*4 + i]` with `best_recon_q[i*4 + 3]` for `i=0..3`, preserving the rightmost reconstructed column as the next left predictor.
- At `sbx=0`, top references for the current row have already been latched before any commit writes, so overwriting `top_line_sram` cannot corrupt remaining `sbx=1..3` predictions in the same sub-block row.

**SSD and RD cost**

- `diff_i = signed({1'b0,src_pix_q[i]}) - signed({1'b0,recon_pix_q[i]})`: signed 9-bit, -255..255.
- `sq_i = diff_i * diff_i`: 9x9 -> unsigned 18-bit, 0..65025.
- `ssd_sum = sum(sq_0..sq_15)`: unsigned 22-bit, 0..1040400.
- `ssd_term = {ssd_sum,20'b0}`: unsigned 42-bit, equal to `SSD*2^20`.
- `mode_bits`: unsigned 4-bit, `1` when `mode_idx_q == mpm_q`, otherwise `4`.
- `coeff_bits`: unsigned 16-bit from the entropy coding bit-count microsequence.
- `syntax_bits = mode_bits + coeff_bits`: unsigned 17-bit.
- `lambda_q20`: unsigned 32-bit Q12.20 from constant decode indexed by `qp_frame_q`.
- `rate_term = lambda_q20 * syntax_bits`: 32x17 -> unsigned 49-bit Q20.
- `candidate_cost = zero_extend(ssd_term,64) + zero_extend(rate_term,64)`: unsigned 64-bit.

Cost comparison is unsigned. Ties keep the earlier best candidate because best registers update only when `candidate_cost < best_cost_q`, so lower mode index wins exact ties.

**entropy coding bit-count datapath**

The RD core does not emit entropy bits, but its `coeff_bits` must match `entropy_bitstream_engine`. Bit-count uses the same frozen VLC length definitions as entropy emission, implemented as combinational constant-decode functions plus the adaptive suffix-length arithmetic. There is no addressed entropy coding ROM in this block.

The fixed 21-cycle bit-count sequence:

1. Count total coefficients, trailing ones, signs, total zeros, and run-before symbols from `quant_level_zz[0:15]`.
2. Decode coeff-token length from `{nC_category,total_coeff,trailing_ones}`.
3. Add trailing-one sign count.
4. Add level prefix/suffix lengths using golden suffix-length adaptation.
5. Decode total-zeros length only when `total_coeff != 0 && total_coeff != 16`.
6. Decode run-before lengths for nonzero coefficients while `zeros_left != 0`.

All bit counts are unsigned 16-bit and range-checked against 1023 bits per 4x4 block. Overflow raises a status error and clips `coeff_bits` to `16'hffff` for cost calculation.

**MPM and nC context**

Context is sampled before candidate evaluation and updated only after selected reconstruction commit:

- `top_ctx_mem` stores 9-bit words `{mode[3:0], nz_count[4:0]}` for 160 4x4 columns.
- `left_mode_ctx_q[0:3]` and `left_nz_ctx_q[0:3]` store immediately-left context for the current pixel_block row.
- `left_mode = avail_left_q ? left_mode_ctx_q[sby] : 4'd2`.
- `top_mode = avail_top_q ? top_ctx_mode_rdata : 4'd2`.
- `mpm_q = min(left_mode, top_mode)`: unsigned 4-bit, 0..8.
- `left_nz = avail_left_q ? left_nz_ctx_q[sby] : 5'd0`.
- `top_nz = avail_top_q ? top_ctx_nz_rdata : 5'd0`.
- `nC_q = both ? ((left_nz + top_nz + 1) >> 1) : (left_only ? left_nz : (top_only ? top_nz : 5'd0))`: unsigned 5-bit, 0..16.

After selecting a candidate, committed `best_mode_q` and `best_nz_count_q` update the left context for `sby` and the top context at column `mb_x*4 + sbx`. The update occurs before the next sub-block is popped from the RD sequencer.

### 3.3 Control Logic

The core uses a binary-encoded FSM. Binary encoding is selected because state count exceeds 16 and the transition graph is simple; minimizing state FFs matters under the hard 8000 FF cap.

| State | Function | Transitions |
|---|---|---|
| `STATE_RESET_IDLE` | Reset-idle state while `rst_n=0`; all master valids low. | To `STATE_WAIT_CFG` after reset deassertion. |
| `STATE_WAIT_CFG` | Assert `s_axis_frame_cfg_tready` if config skid is empty. Validate `tlast`, `frame_start`, reserved bits, width, height, and QP. | Legal config -> `STATE_FRAME_INIT`; illegal -> `STATE_STATUS_ERROR`. |
| `STATE_FRAME_INIT` | Clear frame-local counters, left contexts, top-line valid state, FIFO controls, and error sticky. | `STATE_READY_FOR_MB`. |
| `STATE_READY_FOR_MB` | Sequencer idle. If ingress FIFO nonempty, pop exactly one sub-block beat. | Pop -> `STATE_POP_SUBBLOCK`; otherwise remain. |
| `STATE_POP_SUBBLOCK` | Latch source pixels and metadata; check raster order, QP, frame_first, and terminal flag consistency. | Checks pass -> `STATE_CTX_ADDR`; error -> `STATE_STATUS_ERROR`. |
| `STATE_CTX_ADDR` | Issue top context read for column `mb_x*4 + sbx`; compute left context values. | `STATE_CTX_WAIT`. |
| `STATE_CTX_WAIT` | Capture registered top context data; compute `mpm_q` and `nC_q`. | `STATE_ROW_TOP_PREFETCH`. |
| `STATE_ROW_TOP_PREFETCH` | Fixed 7-cycle window. If `sbx==0`, issue up to five top-line SRAM reads and capture `row_top_ref_q[0:19]` plus `row_top_left_q`; otherwise hold existing row-top latches. | After 7 cycles -> `STATE_NEIGHBOR_PACK`. |
| `STATE_NEIGHBOR_PACK` | Select left/top/top-right/top-left refs from latches or default 128. | `STATE_MODE_INIT`. |
| `STATE_MODE_INIT` | Set `mode_idx_q=0`, `best_cost_q=MAX`, clear best candidate registers. | `STATE_MODE_LEGAL`. |
| `STATE_MODE_LEGAL` | Determine legality for current mode slot. Illegal modes are marked invalid but keep fixed timing. | `STATE_PREDICT`. |
| `STATE_PREDICT` | Generate `pred_pix_q[0:15]` for the current mode. | `STATE_RESIDUAL`. |
| `STATE_RESIDUAL` | Compute signed residuals. | `STATE_FWD_H`. |
| `STATE_FWD_H` | Compute horizontal `Cf` partials. | `STATE_FWD_V`. |
| `STATE_FWD_V` | Compute vertical transform coefficients. | `STATE_QUANT_INIT`. |
| `STATE_QUANT_INIT` | Set `coeff_idx_q=0`, clear `nz_count`, decode first quant/dequant constants. | `STATE_QUANT_COEFF`. |
| `STATE_QUANT_COEFF` | One coefficient per cycle: quantize in zig-zag order, write level, compute dequant coefficient, count nonzeros. | After 16 coefficients -> `STATE_INV_H`; otherwise remain. |
| `STATE_INV_H` | First inverse-transform registered stage. | `STATE_INV_V_RECON`. |
| `STATE_INV_V_RECON` | Final inverse transform, add prediction, clip reconstructed pixels. | `STATE_SSD`. |
| `STATE_SSD` | Compute 16-term SSD using reconstructed pixels. | `STATE_entropy coding_INIT`. |
| `STATE_entropy coding_INIT` | Initialize fixed entropy coding bit-count microsequence. | `STATE_entropy coding_COUNT`. |
| `STATE_entropy coding_COUNT` | Run 21 fixed cycles of VLC length decode and arithmetic accumulation. | After 21 cycles -> `STATE_COST_MUL`; otherwise remain. |
| `STATE_COST_MUL` | Compute `lambda_q20 * syntax_bits`. | `STATE_COST_ADD`. |
| `STATE_COST_ADD` | Add SSD term and rate term into 64-bit candidate cost. | `STATE_BEST_COMPARE`. |
| `STATE_BEST_COMPARE` | If mode is legal and cost is strictly lower than current best, copy mode, levels, reconstruction, nz, bits, and cost into best registers. | If `mode_idx_q==8` -> `STATE_COMMIT_INIT`; else increment mode and go to `STATE_MODE_LEGAL`. |
| `STATE_COMMIT_INIT` | Stall if syntax FIFO is full because state commit and syntax enqueue are atomic. | `STATE_COMMIT_WRITE`. |
| `STATE_COMMIT_WRITE` | Four-cycle selected reconstruction commit. Write packed top-line word for each selected row and update left-edge entries. | After 4 row writes -> `STATE_CONTEXT_UPDATE`. |
| `STATE_CONTEXT_UPDATE` | Update left and top MPM/nC contexts with selected mode and `nz_count`. | `STATE_SYNTAX_ENQUEUE`. |
| `STATE_SYNTAX_ENQUEUE` | Pack and enqueue 214-bit syntax record plus copied final-frame `tlast`. | `STATE_STATUS_COMMIT`. |
| `STATE_STATUS_COMMIT` | Enqueue sub-block commit event if status FIFO has space. If final sub-block, enqueue frame-complete event. | Final -> `STATE_FRAME_DONE_WAIT`; non-final -> `STATE_READY_FOR_MB`. |
| `STATE_FRAME_DONE_WAIT` | Do not accept more frame data. Wait for lifecycle reset/config for the next frame. | To `STATE_WAIT_CFG` after local frame reset sequencing. |
| `STATE_STATUS_ERROR` | Enqueue error event with severity/code; suppress further RD pops for current frame. | `STATE_FRAME_DONE_WAIT`. |

AXI-Stream handshake rules:

- `s_axis_frame_cfg_tready` is asserted only when the config skid is empty and the FSM is in `STATE_WAIT_CFG`.
- `s_axis_mb_tready` is asserted when the ingress FIFO is not full and a legal frame config has been accepted.
- The ingress FIFO may accept up to 16 beats ahead of the RD sequencer, but the sequencer pops a later beat only after the prior selected reconstruction and contexts are committed.
- `m_axis_mb_syntax_*` and `m_axis_status_event_*` are driven from registered FIFO output beats. There is no hidden pre-output valid.
- FIFO `tready` conditions exactly match the clocked accept branches. FWFT output logic computes post-pop read pointer/occupancy and bypasses same-cycle read/write of the selected address to avoid duplicated or zeroed beats.

### 3.4 Storage Elements

**Registers**

| Register | Width | Reset value | Update condition |
|---|---:|---:|---|
| `state_q` | 5 | `STATE_RESET_IDLE` while reset active | Every clock by FSM next state. |
| `cfg_valid_q` | 1 | 0 | Set on legal config acceptance, cleared by frame reset. |
| `active_width_q` | 10 | 0 | Legal config handshake. |
| `active_height_q` | 10 | 0 | Legal config handshake. |
| `qp_frame_q` | 6 | 0 | Legal config handshake. |
| `padded_width_q` | 10 | 0 | Legal config handshake. |
| `padded_height_q` | 9 | 0 | Legal config handshake. |
| `mb_cols_q` | 6 | 0 | Legal config handshake. |
| `mb_rows_q` | 5 | 0 | Legal config handshake. |
| `macroblocks_per_frame_q` | 10 | 0 | Legal config handshake. |
| `expected_mb_x_q` | 6 | 0 | Increment after committed `sb_idx=15`; clear per frame. |
| `expected_mb_y_q` | 5 | 0 | Increment on pixel_block row end; clear per frame. |
| `expected_sb_idx_q` | 4 | 0 | Increment after each commit; clear at pixel_block boundary/frame reset. |
| `src_pix_q[0:15]` | 16x8 | 0 | FIFO pop in `STATE_POP_SUBBLOCK`. |
| `mb_x_q`, `mb_y_q`, `sb_idx_q` | 6+5+4 | 0 | FIFO pop. |
| `avail_left_q`, `avail_top_q`, `avail_topright_q` | 3 | 0 | FIFO pop. |
| `frame_first_q`, `mb_last_q`, `subblock_last_q` | 3 | 0 | FIFO pop. |
| `row_top_ref_q[0:19]` | 20x8 | 128 | `STATE_ROW_TOP_PREFETCH` when `sbx==0`; otherwise held. |
| `row_top_left_q` | 8 | 128 | `STATE_ROW_TOP_PREFETCH`. |
| `left_edge_q[0:15]` | 16x8 | 128 | Frame row start/default and selected commit. |
| `next_mb_top_left_seed_q` | 8 | 128 | Captured during row-top prefetch for next pixel_block. |
| `left_ref_q[0:3]`, `top_ref_q[0:3]`, `topright_ref_q[0:3]`, `topleft_ref_q` | 13x8 | 128 | `STATE_NEIGHBOR_PACK`. |
| `mode_idx_q` | 4 | 0 | Mode loop. |
| `mode_legal_q` | 1 | 0 | `STATE_MODE_LEGAL`. |
| `pred_pix_q[0:15]` | 16x8 | 0 | `STATE_PREDICT`. |
| `residual_q[0:15]` | 16x9 | 0 | `STATE_RESIDUAL`. |
| `tmp_h_q[0:15]` | 16x12 | 0 | `STATE_FWD_H`. |
| `coeff_q[0:15]` | 16x16 | 0 | `STATE_FWD_V`. |
| `quant_level_q[0:15]` | 16x12 | 0 | `STATE_QUANT_COEFF`. |
| `dequant_q[0:15]` | 16x20 | 0 | `STATE_QUANT_COEFF`. |
| `inv_tmp_q[0:15]` | 16x24 | 0 | `STATE_INV_H`. |
| `recon_pix_q[0:15]` | 16x8 | 0 | `STATE_INV_V_RECON`. |
| `ssd_q` | 22 | 0 | `STATE_SSD`. |
| `coeff_bits_q` | 16 | 0 | `STATE_entropy coding_COUNT` completion. |
| `candidate_cost_q` | 64 | 0 | `STATE_COST_ADD`. |
| `best_cost_q` | 64 | all 1s in mode init | `STATE_MODE_INIT`; updated on strict lower cost. |
| `best_mode_q` | 4 | 0 | `STATE_BEST_COMPARE` on better legal candidate. |
| `best_levels_q[0:15]` | 16x12 | 0 | `STATE_BEST_COMPARE` on better legal candidate. |
| `best_recon_q[0:15]` | 16x8 | 0 | `STATE_BEST_COMPARE` on better legal candidate. |
| `best_nz_count_q` | 5 | 0 | `STATE_BEST_COMPARE` on better legal candidate. |
| `mpm_q` | 4 | 2 | `STATE_CTX_WAIT`. |
| `nC_q` | 5 | 0 | `STATE_CTX_WAIT`. |
| `left_mode_ctx_q[0:3]` | 4x4 | 2 | Frame reset, pixel_block row start, and selected commit. |
| `left_nz_ctx_q[0:3]` | 4x5 | 0 | Frame reset, pixel_block row start, and selected commit. |
| `error_sticky_q` | 1 | 0 | Set on any local protocol/range/arithmetic error; cleared by frame reset. |
| `cycle_ctr_q` | 32 | 0 | Increments while frame active. |
| `mb_cycle_ctr_q` | 16 | 0 | Clear on sub-block 0 of MB, increment until sub-block 15 commit. |

**Storage implementation and budget**

The design stays within the hard 8000 FF cap by evaluating one candidate mode at a time and by limiting priced storage to the true dependency windows. Bulk FIFOs and the top-line predictor buffer use explicit SRAM wrappers. Small context and status state use FFs or `cs_fpmem` with registered reads. No raw combinational `reg [W-1:0] mem [0:N-1]` arrays are permitted.

Storage budget:

- `flip_flop_budget = 7350 FF`, including FSM, geometry/raster registers, datapath pipeline registers, best-candidate registers, row-top/left-edge predictor latches, FIFO controls/output registers, cycle counters, and small status state. This is below the hard 8000 FF allocation.
- `sram_budget = 11040 bits = 1.35 KiB logical SRAM`: `top_line_sram` 5120 bits, `mb_ingress_fifo` 2480 bits, and `syntax_fifo` 3440 bits. The preferred physical macro composition is one `sky130_sram_1kbyte_1rw1r_32x256_8` for `top_line_sram`, width-tiled legal 1rw1r macro wrappers for the two shallow wide FIFOs, or equivalent `cs_sram_1rw1r` wrappers resolved by the flow.
- `area_budget_um2 = 1650000`. Priced memory is expected below 0.55 mm2 using the same gate model that priced a 32x256-class macro at about 0.19 mm2 and the two shallow FIFOs at about 0.25 mm2 total. The remaining area is allocated to the sequential RD datapath/control and integration margin. The rejected full-window reconstruction memory and table memories are intentionally absent.

Memory manifest:

```text
# MEM frame_cfg_skid: 33x1 ports=1rw impl=flop justification=one configuration beat plus tlast is the canonical skid depth; frame geometry and QP are then held in named registers for the whole frame
# MEM mb_ingress_fifo: 155x16 ports=1rw1r impl=sram justification=canonical mb_axis edge requires exactly one pixel_block of elasticity, 16 serialized sub-block beats, between macroblock_assembler and the sequential RD core
# MEM syntax_fifo: 215x16 ports=1rw1r impl=sram justification=canonical mb_syntax_axis edge requires one pixel_block of committed syntax elasticity so entropy backpressure cannot reorder or deadlock RD output
# MEM status_event_fifo: 33x4 ports=1rw1r impl=flop justification=canonical status edge requires a 4-event queue for bursty commit/error/frame events; total size is below 2 Kbit and depth is only four beats
# MEM top_line_sram: 32x160 ports=1rw1r impl=sram justification=true Intra4x4 raster prediction rereads only the immediate previous reconstructed sample row across 640 pixels; row-start latches preserve that row while same-row commits overwrite the line, so a 16-line or full-frame store is not part of the dependency window
# MEM row_top_ref_latch: 8x20 ports=1rw impl=flop justification=twenty bytes cover one pixel_block's 16 top pixels plus four top-right pixels; the latch must hold a sub-block row's top references while top_line_sram is overwritten by committed blocks in that row
# MEM left_edge_ctx: 8x16 ports=1rw impl=flop justification=sixteen bytes hold the immediately-left reconstructed column for the 16 rows of the current pixel_block; prediction never needs columns older than the adjacent left edge
# MEM top_ctx_mem: 9x160 ports=1rw1r impl=fpmem justification=MPM and nC need the previous committed mode/nz context for each of the 160 4x4 columns; row history beyond the current top context is overwritten by raster traversal and is not needed
```

Memory port mapping:

- `top_line_sram` uses a `cs_sram_1rw1r #(WIDTH=32, DEPTH=160)` wrapper or a `sky130_sram_1kbyte_1rw1r_32x256_8` macro with addresses 160..255 unused. Port A performs commit writes and may perform reads during prefetch when not writing. Port B performs registered reads during row-top prefetch. Read data is consumed one cycle after address issue.
- `mb_ingress_fifo` stores `{s_axis_mb_tdata, s_axis_mb_tlast}` as 155 bits. It uses registered reads plus FWFT output control with same-cycle bypass.
- `syntax_fifo` stores `{m_axis_mb_syntax_tdata, m_axis_mb_syntax_tlast}` as 215 bits. It uses registered reads plus FWFT output control with same-cycle bypass.
- `status_event_fifo` stores `{m_axis_status_event_tdata, 1'b1}` as 33 bits in flops.
- `top_ctx_mem` uses `cs_fpmem_1rw1r` with one-cycle registered read. It is below 2 Kbit and shallow.
- Lambda, quant/dequant, and entropy coding length constants are not storage structures. They must be implemented as combinational constant-decode logic or small `localparam` packed constants with constant part-selects only; no runtime-indexed wide dynamic part-select and no addressed memory table is allowed.

## 4. Algorithm Mapping

No Python golden model body was provided in the prompt. This mapping is from the ERS/FRD frozen functional description and canonical interface contracts.

| Functional description | Hardware equivalent |
|---|---|
| Accept one frame configuration and hold QP/geometry | `STATE_WAIT_CFG` AXI handshake, config skid, geometry/QP registers, `cfg_valid_q`. |
| Iterate pixel_blocks in raster order | `expected_mb_x_q`, `expected_mb_y_q`, and `expected_sb_idx_q` check incoming serialized records. |
| One 4x4 sub-block per `mb_axis` beat | 16-deep `mb_ingress_fifo`; RD pops only after prior commit. |
| Use reconstructed neighbors, default 128 if unavailable | `top_line_sram`, row-top latches, left-edge registers, and availability-controlled default 128 muxes. |
| `for mode in range(9)` | Fixed nine-slot FSM loop using `mode_idx_q=0..8`; illegal slots produce max cost. |
| the video codec Intra4x4 prediction formulas | Registered prediction stage using 8-bit references, 9/10/11-bit average sums, and golden right shifts. |
| `residual = src - pred` | 16 parallel signed 9-bit subtractors in `STATE_RESIDUAL`. |
| `Cf @ residual @ Cf.T` | Two registered transform stages: signed 12-bit row partials and signed 16-bit final coefficients. |
| Dead-zone quantization | 16-cycle coefficient loop using constant MF decode, golden offset, 16x16 multiply, and arithmetic sign restoration. |
| Zig-zag scan | Constant position sequence `[0,1,4,8,5,2,3,6,9,12,13,10,7,11,14,15]` in coefficient-loop control. |
| Dequantization and inverse transform | One dequant coefficient per quant-loop cycle, then two registered inverse-transform stages. |
| Clipped reconstruction | Signed add of prediction and inverse residual followed by saturating clamp to unsigned 8-bit. |
| SSD | Registered 16-term SSD stage, unsigned 22-bit result. |
| entropy coding bit count | Fixed 21-cycle microsequence using constant VLC length decode and adaptive arithmetic. |
| `cost = SSD*2^20 + lambda_q20(QP)*(mode_bits+coeff_bits)` | Shift by concatenation, constant lambda decode, 32x17 multiply, 64-bit add. |
| `argmin` | `best_cost_q` and `best_*` registers updated only on strict lower cost; lower mode wins ties. |
| Commit selected reconstruction | Four-cycle write loop updates `top_line_sram` and left-edge predictor registers. |
| Update MPM and nC context | Left-context FF update plus top-context fpmem write in `STATE_CONTEXT_UPDATE`. |
| Emit syntax record | Pack 214-bit record in canonical field order and enqueue to syntax FIFO. |
| Emit status events | Pack 32-bit status records with `block_id=3'b011`, single-beat `tlast=1`. |

### 4a. Cross-Block Semantic Invariants (MANDATORY)

**INV-RD-ORDER-001**

- **Applies to ports/state:** `s_axis_mb_*`, `m_axis_mb_syntax_*`, `mb_x_q`, `mb_y_q`, `sb_idx_q`, `expected_*` counters.
- **Golden reference point:** FRD PERF-006 raster traversal.
- **Tolerance:** Exact ordering; no dropped, duplicated, or reordered sub-blocks.
- **Update/consume timing:** Input beats may enter the ingress FIFO under AXI handshakes, but the RD sequencer pops a later sub-block only after the prior selected reconstruction and contexts are committed.
- **Downstream dependency:** `entropy_bitstream_engine` assumes exactly 16 syntax records per pixel_block in sub-block raster order.
- **Validation hook:** VCD trace `mb_x_q`, `mb_y_q`, `sb_idx_q`, `syntax_fifo_push`, `m_axis_mb_syntax_tdata[213:210]`, and `m_axis_mb_syntax_tlast`.

**INV-RECON-FEEDBACK-002**

- **Applies to ports/state:** `top_line_sram`, `row_top_ref_q`, `left_edge_q`, `best_recon_q`, `STATE_COMMIT_WRITE`.
- **Golden reference point:** FRD PERF-004 reconstructed luma self-consistency and PERF-008 closed-loop arithmetic.
- **Tolerance:** Exact 8-bit reconstructed pixel equality to the golden decoder trace for emitted syntax.
- **Update/consume timing:** Selected reconstructed pixels for sub-block N update top-line and left-edge predictor state during `STATE_COMMIT_WRITE`, and context state updates in `STATE_CONTEXT_UPDATE`, before sub-block N+1 can enter candidate evaluation.
- **Downstream dependency:** Later RD predictions and emitted syntax must describe coefficients generated from the same selected reconstruction the decoder will reconstruct.
- **Validation hook:** Dump `top_line_wr_en`, `top_line_wr_addr`, `top_line_wr_wdata`, `row_top_ref_q`, `left_edge_q`, neighbor refs, `best_mode_q`, and `best_levels_q`.

**INV-MODE-MPM-003**

- **Applies to ports/state:** `top_ctx_mem`, `left_mode_ctx_q`, `mpm_q`, `best_mode_q`, `m_axis_mb_syntax_tdata[209:202]`.
- **Golden reference point:** the video codec Intra4x4 MPM-coded mode syntax in the entropy engine contract.
- **Tolerance:** Exact `best_mode` and `mpm` equality for every sub-block.
- **Update/consume timing:** MPM is sampled in `STATE_CTX_WAIT`; committed best mode updates contexts in `STATE_CONTEXT_UPDATE` before the next sub-block context read.
- **Downstream dependency:** `entropy_bitstream_engine` emits either a 1-bit MPM match or a remapped mode code; wrong MPM changes the bitstream.
- **Validation hook:** Dump `left_mode_ctx_q`, `top_ctx_mem` read/write address/data, `mpm_q`, `best_mode_q`, and syntax bits `[209:202]`.

**INV-NC-entropy coding-004**

- **Applies to ports/state:** `top_ctx_mem`, `left_nz_ctx_q`, `nC_q`, `best_nz_count_q`, `m_axis_mb_syntax_tdata[201:192]`.
- **Golden reference point:** entropy coding nC context and residual block syntax length calculation.
- **Tolerance:** Exact `nC`, `nz_count`, and quantized level equality for every sub-block.
- **Update/consume timing:** `nC_q` is computed before candidate entropy coding bit count; `best_nz_count_q` updates contexts after commit and is emitted in the same syntax record as selected levels.
- **Downstream dependency:** `entropy_bitstream_engine` selects coeff-token VLC table from `nC`; wrong nC can serialize a bit-exact coefficient array incorrectly.
- **Validation hook:** Dump `nC_q`, `coeff_bits_q`, `best_nz_count_q`, top/left nz context, and syntax fields `[201:192]`.

**INV-QP-FRAME-005**

- **Applies to ports/state:** `s_axis_frame_cfg_tdata[11:6]`, `s_axis_mb_tdata[7:2]`, `qp_frame_q`, lambda and quant-scale constant decode indices.
- **Golden reference point:** FRD PERF-010 QP sampling and frozen QP tables.
- **Tolerance:** Exact QP equality; table index must be 0..51.
- **Update/consume timing:** QP is sampled on config beat only and must not change until frame completion; each MB beat QP is checked against `qp_frame_q`.
- **Downstream dependency:** Entropy and decoder expect coefficients generated using the QP carried in the outer container by the frame config path.
- **Validation hook:** Dump `qp_frame_q`, `mb_qp_q`, `lambda_q20`, `quant_const_index`, and QP mismatch status events.

**INV-TLAST-FRAME-006**

- **Applies to ports/state:** `s_axis_mb_tlast`, FIFO stored last bit, `m_axis_mb_syntax_tlast`, `mb_last_q`, `subblock_last_q`.
- **Golden reference point:** Final sub-block of final pixel_block in the frame.
- **Tolerance:** Exact single assertion on final syntax record only.
- **Update/consume timing:** `s_axis_mb_tlast` is latched with the input beat in the ingress FIFO and copied to syntax output only after corresponding committed reconstruction.
- **Downstream dependency:** `entropy_bitstream_engine` terminates payload collection and container generation from final syntax `tlast`; missing TLAST hangs output, early TLAST truncates the frame.
- **Validation hook:** Dump FIFO stored last bit, `subblock_last_q`, syntax FIFO last bit, and `m_axis_mb_syntax_tlast`.

**INV-BITCOUNT-007**

- **Applies to ports/state:** entropy coding constant length decoder, `coeff_bits_q`, `candidate_cost_q`, `best_mode_q`, `best_levels_q`.
- **Golden reference point:** FRD PERF-008 entropy coding bit count and RD cost.
- **Tolerance:** Exact integer bit count and exact unsigned cost comparison.
- **Update/consume timing:** Bit count is computed for each candidate before `STATE_COST_MUL`; best candidate selection uses the same count convention as the entropy engine's emit tables.
- **Downstream dependency:** If RD counts differ from entropy emit lengths, the core can choose a different mode than the golden encoder although residual syntax is locally valid.
- **Validation hook:** Dump candidate `mode_idx_q`, `coeff_bits_q`, entropy coding table class/index, `mode_bits`, `lambda_q20`, `candidate_cost_q`, and `best_cost_q`.

## 5. Reset and Initialization

Reset is synchronous active-low: every sequential block uses `always @(posedge clk)` with `if (!rst_n)`.

On reset assertion:

- All AXI master valids are cleared: `m_axis_mb_syntax_tvalid=0`, `m_axis_status_event_tvalid=0`.
- `s_axis_frame_cfg_tready` and `s_axis_mb_tready` deassert in the reset cycle and become functions of empty registered state after reset deassertion.
- `cfg_valid_q=0`, geometry registers = 0, QP = 0.
- FSM enters `STATE_RESET_IDLE`, then `STATE_WAIT_CFG`.
- FIFO pointers/counts are zeroed; FIFO memory contents are not reset.
- Pipeline/candidate/best registers reset to 0 except `best_cost_q`, which is loaded with all 1s in `STATE_MODE_INIT`.
- `row_top_ref_q`, `row_top_left_q`, `left_edge_q`, and `next_mb_top_left_seed_q` reset to 128.
- Left mode contexts reset to DC mode 2 and left nC contexts reset to 0.
- `top_ctx_mem` contents need not be cleared by reset because first-row availability and raster traversal define validity. If an implementation clears it, it must use a bounded 160-cycle init sequence and hold `s_axis_mb_tready=0` until done.
- `top_line_sram` contents are not reset and are never consumed when top availability is false. Legal first-row traffic has `avail_top=0`; row-top prefetch substitutes 128 for unavailable references.
- `error_sticky_q=0`, cycle counters = 0.

Reset-idle is not protocol completion. `m_axis_mb_syntax_tlast`, `m_axis_status_event_tlast`, frame-completion status events, and any frame-complete mirror must not assert after reset until a real terminal input sub-block has been accepted, processed, committed, and emitted.

Per-frame initialization after a legal config:

1. Validate geometry/QP/reserved fields.
2. Initialize predictor latches (`row_top_ref_q`, `row_top_left_q`, `left_edge_q`) to 128.
3. Initialize expected raster counters to `(mb_x=0, mb_y=0, sb_idx=0)`.
4. Clear `error_sticky_q`, cycle counters, and FIFO state associated with the previous frame.
5. Do not clear `top_line_sram`; availability flags prevent stale consumption until reconstructed rows have been committed.

## 6. Timing and Performance

Target clock is 50 MHz, period 20 ns. The implementation is sequential to meet the hard 8000 FF budget and avoid an unrolled RD combinational cloud.

Critical path estimates by stage:

- Row-top prefetch: SRAM address mux/add (`x >> 2`) plus registered macro access; data is captured one cycle later.
- Prediction average stage: at most an 11-bit add tree plus shift/mux, below one 16-bit add equivalent and less than 20 ns.
- Residual stage: 16 parallel 9-bit subtractors; each is shorter than the characterized 16-bit subtractor of 3.79 ns.
- `FWD_H` and `FWD_V`: each lane uses shifts/sign changes and up to three 16-bit add/sub equivalents, below the five-add 20 ns budget.
- Quant coefficient cycle: one 16x16 multiply (6.77 ns), one 33-bit add, and a registered variable shift selection. The wide rate multiply is not chained in this state.
- SSD stage: one 16-term SAD/SSD-class accumulation is bounded to one stage; if STA shows the square-plus-accumulate path exceeds 20 ns, split square and accumulation across the reserved candidate slot bubble and increase candidate slot by one documented cycle.
- Cost multiply stage: one 32-bit multiply equivalent (11.98 ns) alone in `STATE_COST_MUL`.
- Cost add and compare are split across `STATE_COST_ADD` and `STATE_BEST_COMPARE`.
- SRAM and fpmem reads are registered; prediction/context logic consumes read data only after one-cycle memory latency.

Fixed non-stalled schedule per sub-block from RD FIFO pop:

| Phase | Cycles |
|---|---:|
| Pop/check/context read | 3 |
| Row-top prefetch/neighbor pack | 12 |
| Nine candidate slots, 50 cycles each | 450 |
| Commit init/write 4 packed row words | 5 |
| Context update, syntax enqueue, commit status | 2 |
| **Total from pop to syntax FIFO enqueue** | **472** |

Candidate slot schedule:

| Candidate phase | Cycles |
|---|---:|
| legality + predict + residual + forward transform | 5 |
| quant/dequant coefficient loop | 17 |
| inverse transform + reconstruct + SSD | 3 |
| entropy coding init/count loop | 22 |
| cost multiply/add/compare | 3 |
| **Total** | **50** |

Throughput:

- Input acceptance into the 16-beat `mb_ingress_fifo` can burst at one beat per cycle until full.
- RD processing initiation interval is fixed at 472 non-stalled cycles per sub-block.
- One pixel_block requires 16 sub-blocks, so one pixel_block requires 7552 non-stalled RD cycles plus stalls.
- Downstream syntax FIFO full stalls `STATE_COMMIT_INIT` before reconstruction commit so syntax record and predictor/context state remain atomic.

### 6a. Output Timing Contract (MANDATORY)

| Output port | Type | Pipeline latency | First valid after reset |
|---|---|---:|---:|
| `s_axis_frame_cfg_tready` | combinational from registered FSM/skid-empty state | 0 cycles | 1 cycle after reset deassertion, high in `STATE_WAIT_CFG` if skid empty |
| `s_axis_mb_tready` | combinational from registered FIFO count and `cfg_valid_q` | 0 cycles | 1 cycle after a legal config is accepted, if ingress FIFO is not full |
| `m_axis_mb_syntax_tdata` | registered FIFO output | 472 cycles from RD pop under no stalls | 472 cycles after first legal sub-block is popped when downstream ready |
| `m_axis_mb_syntax_tvalid` | registered FIFO output valid | 472 cycles from RD pop under no stalls | 472 cycles after first legal sub-block pop |
| `m_axis_mb_syntax_tlast` | registered FIFO output sideband | 472 cycles for the corresponding final input sub-block under no stalls | Only for the final frame sub-block |
| `m_axis_status_event_tdata` | registered FIFO output | 2 cycles for config/protocol errors; 472 cycles for commit events | 2 cycles after an immediate error or 472 cycles after first legal sub-block pop |
| `m_axis_status_event_tvalid` | registered FIFO output valid | Same as status data for the event class | Same as status data |
| `m_axis_status_event_tlast` | registered FIFO output sideband | Same as status data for the event class | Asserted with each valid status event only |

Representative primary-path timing:

```wavedrom
{signal: [
  {name: 'clk',                       wave: 'p................'},
  {name: 'rst_n',                     wave: '01...............'},
  {name: 's_axis_frame_cfg_tvalid',   wave: '0.10.............'},
  {name: 's_axis_frame_cfg_tready',   wave: '0.10.............'},
  {name: 's_axis_mb_tvalid',          wave: '0..10............'},
  {name: 's_axis_mb_tready',          wave: '0..10............'},
  {name: 's_axis_mb_tdata',           wave: 'x..=x............', data: ['SB0']},
  {name: 'RD sequencer',              wave: 'x...=............', data: ['472 cycles']},
  {name: 'm_axis_mb_syntax_tdata',    wave: 'x................=', data: ['syntax(SB0)']},
  {name: 'm_axis_mb_syntax_tvalid',   wave: '0................1'}
],
 head: {text: 'Primary datapath latency = 472 non-stalled cycles to registered syntax output'}}
```

Backpressure extends latency but not ordering or payload stability. If `m_axis_mb_syntax_tready=0`, the output FIFO holds `tvalid`, `tdata`, and `tlast` stable. If `m_axis_status_event_tready=0`, the status FIFO holds its current event stable.

## 7. Edge Cases and Corner Conditions

- **First frame after reset:** No top-line SRAM contents are trusted unless top availability is asserted by legal raster position. Unavailable neighbors use 128.
- **First pixel_block/sub-block:** `frame_first` must be high only on `mb_x=0`, `mb_y=0`, `sb_idx=0`. Left/top/top-right unavailable flags drive default 128 references and DC MPM/nC defaults.
- **Top row:** Top and top-right unavailable references must not issue out-of-range memory reads. Modes requiring top are illegal except DC and horizontal modes as listed.
- **Left edge:** Left/topleft unavailable references must not issue out-of-range reads. Modes requiring left are illegal except DC and vertical modes as listed.
- **Right edge/top-right:** If `avail_topright=0`, top-right references are 128 and modes 3 and 7 are illegal. No read may address beyond `padded_width-1`.
- **Row-top overwrite:** Top references for a sub-block row are latched at `sbx=0` before any selected reconstruction in that row overwrites `top_line_sram`; later `sbx` positions must use `row_top_ref_q`, not live top-line memory, for top/top-left/top-right.
- **QP mismatch:** If `s_axis_mb_tdata[7:2] != qp_frame_q`, set an error event and suppress commit for that beat.
- **Unexpected order:** If `mb_x`, `mb_y`, or `sb_idx` does not match expected raster counters, set protocol/range error and stop consuming new sub-blocks for the frame.
- **Arithmetic overflow:** Residual, forward transform, and SSD are range-guaranteed. Quant/dequant emitted-level overflow saturates, raises status error, and is not expected in legal golden vectors. Reconstruction clips to 0..255 by definition.
- **All-zero residual:** Still emits one syntax record with `nz_count=0` and all 16 `quant_level_zz* = 0`.
- **Ties in RD cost:** Lower mode index wins because best registers update only on strict less-than.
- **Status empty vs completion:** Empty status FIFO after reset means no event pending; it is not a frame-complete event.
- **Syntax output TLAST:** Assert only on the final committed sub-block record corresponding to final input `tlast`; never on reset or idle.
- **FIFO full:** `s_axis_mb_tready` deasserts when ingress FIFO is full. The RD FSM stalls before commit if syntax FIFO cannot accept the selected syntax record, preserving atomic reconstruction/syntax state.

## 8. Implementation Notes

- Do not implement nine parallel mode datapaths or a single-cycle mode search. The RTL must structurally instantiate one candidate datapath and sequence it with the FSM.
- Do not use floating point. Lambda is fixed-point Q12.20; transform, quantization, inverse transform, SSD, and cost arithmetic are integer.
- Do not instantiate a 16-line or full-frame reconstruction SRAM in this block. The accepted storage is one 32x160 top-line SRAM plus row-top and left-edge latches.
- Do not implement lambda, quant/dequant, or entropy coding length tables as addressed memories in this block. Use constant-decode logic and keep it pipelined so it does not become a dynamic wide part-select or a large combinational cloud.
- All memory reads are registered one-cycle reads. Prediction must consume top-line/context data only after SRAM/fpmem read latency has elapsed.
- Quantized level packing must use canonical zig-zag order and signed 12-bit two's-complement fields. There is no Intra16x16 branch and no reserved DC-Hadamard slot.
- The status event queue must not drop fatal/error events. If it fills, the RD sequencer stops accepting further sub-blocks until space is available or lifecycle reset occurs.
- Testbench verification points should include: mode loop visited mask, legal-mode mask, predictor references, row-top latch contents, left-edge registers, residuals, transform coefficients, quant levels, dequant coefficients, reconstruction pixels, SSD, mode bits, coeff bits, lambda, candidate cost, best candidate, context updates, top-line writes, syntax FIFO pushes, and TLAST propagation.
- Contract-audit VCD signals: `state_q`, `mode_idx_q`, `mode_legal_q`, `mpm_q`, `nC_q`, `best_mode_q`, `best_levels_q`, `best_nz_count_q`, `top_line_wr_en`, `top_line_wr_addr`, `top_line_wr_wdata`, `row_top_ref_q`, `left_edge_q`, `top_ctx_*`, `left_*_ctx_q`, `syntax_fifo_push`, `m_axis_mb_syntax_*`, and `m_axis_status_event_*`.

## 9. Verilog Interface Stub

```verilog
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
endmodule
```

```json
{
  "block_name": "intra_rd_encode_core",
  "latency_cycles": 472,
  "throughput_samples_per_cycle": 0.00211864406779661,
  "pipeline_stages": 50,
  "register_count": 7350,
  "rom_bits": 0,
  "estimated_gate_count": 95000,
  "fsm_states": [
    "STATE_RESET_IDLE",
    "STATE_WAIT_CFG",
    "STATE_FRAME_INIT",
    "STATE_READY_FOR_MB",
    "STATE_POP_SUBBLOCK",
    "STATE_CTX_ADDR",
    "STATE_CTX_WAIT",
    "STATE_ROW_TOP_PREFETCH",
    "STATE_NEIGHBOR_PACK",
    "STATE_MODE_INIT",
    "STATE_MODE_LEGAL",
    "STATE_PREDICT",
    "STATE_RESIDUAL",
    "STATE_FWD_H",
    "STATE_FWD_V",
    "STATE_QUANT_INIT",
    "STATE_QUANT_COEFF",
    "STATE_INV_H",
    "STATE_INV_V_RECON",
    "STATE_SSD",
    "STATE_entropy coding_INIT",
    "STATE_entropy coding_COUNT",
    "STATE_COST_MUL",
    "STATE_COST_ADD",
    "STATE_BEST_COMPARE",
    "STATE_COMMIT_INIT",
    "STATE_COMMIT_WRITE",
    "STATE_CONTEXT_UPDATE",
    "STATE_SYNTAX_ENQUEUE",
    "STATE_STATUS_COMMIT",
    "STATE_FRAME_DONE_WAIT",
    "STATE_STATUS_ERROR"
  ],
  "data_width_in": 154,
  "data_width_out": 214,
  "fixed_point_format": "integer datapath with lambda_q20 in Q12.20",
  "interface_protocol": "axi_stream",
  "area_budget_um2": 1650000,
  "sram_bits": 11040,
  "output_timing": {
    "s_axis_frame_cfg_tready": {"type": "combinational", "latency_cycles": 0},
    "s_axis_mb_tready": {"type": "combinational", "latency_cycles": 0},
    "m_axis_mb_syntax_tdata": {"type": "registered", "latency_cycles": 472},
    "m_axis_mb_syntax_tvalid": {"type": "registered", "latency_cycles": 472},
    "m_axis_mb_syntax_tlast": {"type": "registered", "latency_cycles": 472},
    "m_axis_status_event_tdata": {"type": "registered", "latency_cycles": 2},
    "m_axis_status_event_tvalid": {"type": "registered", "latency_cycles": 2},
    "m_axis_status_event_tlast": {"type": "registered", "latency_cycles": 2}
  },
  "semantic_invariants": [
    {
      "id": "INV-RD-ORDER-001",
      "description": "Sub-block input and syntax output preserve pixel_block raster order and 16 sub-blocks per pixel_block.",
      "ports_or_state": ["s_axis_mb_tdata", "m_axis_mb_syntax_tdata", "expected_mb_x_q", "expected_mb_y_q", "expected_sb_idx_q"],
      "golden_reference": "FRD PERF-006 raster traversal",
      "tolerance": "exact",
      "validation_hook": "Compare popped mb_x/mb_y/sb_idx to syntax output sb_idx and TLAST position"
    },
    {
      "id": "INV-RECON-FEEDBACK-002",
      "description": "Committed reconstructed pixels are generated from the selected mode and levels and are visible to later prediction through top-line and left-edge predictor state.",
      "ports_or_state": ["top_line_sram", "row_top_ref_q", "left_edge_q", "best_recon_q", "best_mode_q", "best_levels_q"],
      "golden_reference": "FRD PERF-004 internal reconstruction self-consistency",
      "tolerance": "exact 8-bit pixel equality",
      "validation_hook": "Trace top-line writes, row-top latches, left-edge updates, and neighbor reads"
    },
    {
      "id": "INV-MODE-MPM-003",
      "description": "MPM context and best mode are sampled, updated, and emitted atomically for each committed sub-block.",
      "ports_or_state": ["top_ctx_mem", "left_mode_ctx_q", "mpm_q", "best_mode_q", "m_axis_mb_syntax_tdata"],
      "golden_reference": "the video codec Intra4x4 MPM mode syntax",
      "tolerance": "exact",
      "validation_hook": "Trace mpm_q/best_mode_q and syntax fields [209:202]"
    },
    {
      "id": "INV-NC-entropy coding-004",
      "description": "nC context and nz_count match the quantized levels used for RD bit count and downstream entropy coding emission.",
      "ports_or_state": ["top_ctx_mem", "left_nz_ctx_q", "nC_q", "best_nz_count_q", "m_axis_mb_syntax_tdata"],
      "golden_reference": "entropy coding nC context and residual syntax",
      "tolerance": "exact",
      "validation_hook": "Trace nC_q, nz_count, quant_level fields, and entropy coding table id"
    },
    {
      "id": "INV-QP-FRAME-005",
      "description": "Frame QP is sampled once from frame_cfg and drives all lambda, quant, and dequant constant decoding for the frame.",
      "ports_or_state": ["s_axis_frame_cfg_tdata", "s_axis_mb_tdata", "qp_frame_q"],
      "golden_reference": "FRD PERF-010 QP sampling",
      "tolerance": "exact",
      "validation_hook": "Trace qp_frame_q and constant-decode indices"
    },
    {
      "id": "INV-TLAST-FRAME-006",
      "description": "Final input sub-block TLAST is latched and emitted only on the corresponding final syntax record.",
      "ports_or_state": ["s_axis_mb_tlast", "m_axis_mb_syntax_tlast", "subblock_last_q"],
      "golden_reference": "Final sub-block of final pixel_block",
      "tolerance": "exact one TLAST",
      "validation_hook": "Trace ingress FIFO last bit and syntax FIFO last bit"
    },
    {
      "id": "INV-BITCOUNT-007",
      "description": "RD entropy coding bit count uses the same frozen length definitions as entropy emission so mode selection matches the golden cost function.",
      "ports_or_state": ["coeff_bits_q", "candidate_cost_q", "best_cost_q"],
      "golden_reference": "FRD PERF-008 candidate cost",
      "tolerance": "exact integer equality",
      "validation_hook": "Trace candidate coeff_bits, mode_bits, lambda, cost, and selected mode"
    }
  ]
}
```