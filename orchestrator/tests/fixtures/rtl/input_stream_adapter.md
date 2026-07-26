## 1. Block Overview
- **Block name:** `input_stream_adapter`
- **Functional summary:** AXI-Stream ingress adapter for 8-bit luma pixels. It accepts external `pixel_in_axis` beats, latches and validates first-beat frame metadata from `pixel_in_tuser`, emits one canonical 32-bit `frame_cfg_axis` beat for each legal frame, forwards accepted active pixels in raster order on `pixel_axis`, generates internal TLAST on the validated final active pixel, and reports ingress protocol/status events on `status_event_axis`.
- **Latency:** first legal `frame_cfg_axis` beat appears 1 cycle after the first accepted input beat; first `pixel_axis` beat appears 1 cycle after the `frame_cfg_axis` beat is accepted by the broadcast fabric; status events appear 1 cycle after the event is generated when the status queue was empty.
- **Throughput:** accepts up to 1 input pixel/cycle and emits up to 1 pixel/cycle, subject to AXI-Stream backpressure and the 16-beat pixel FIFO.
- **Pipeline depth:** 1 registered ingress/control stage plus output FIFO registers; no arithmetic pipeline beyond the first-beat range/product checks.
- **Interface protocol:** AXI-Stream, because the ERS explicitly defines external `pixel_in_axis` and internal `pixel_axis`, `frame_cfg_axis`, and `status_event_axis` as ready/valid streams with TLAST and backpressure.

## 2. Interface Specification
All ports are synchronous to `clk`. Reset is synchronous active-low `rst_n`, matching the block diagram and ERS convention.

| Port | Direction | Width | Protocol | Description |
|---|---:|---:|---|---|
| `clk` | input | 1 | clock | Single 50 MHz clock. |
| `rst_n` | input | 1 | reset | Synchronous active-low reset. |
| `pixel_in_tdata` | input | 8 | AXI-Stream slave `pixel_in_axis` | External luma sample, unsigned integer `[7:0]`, range 0..255. |
| `pixel_in_tvalid` | input | 1 | AXI-Stream slave `pixel_in_axis` | Source asserts when `pixel_in_tdata`, `pixel_in_tlast`, and `pixel_in_tuser` are valid. |
| `pixel_in_tready` | output | 1 | AXI-Stream slave `pixel_in_axis` | Adapter asserts when it can accept the current beat without pixel FIFO/status queue/config skid overflow. |
| `pixel_in_tlast` | input | 1 | AXI-Stream slave `pixel_in_axis` | External end-of-active-frame marker; must be asserted only on pixel index `active_width*active_height-1`. Checked, not blindly propagated. |
| `pixel_in_tuser` | input | 32 | AXI-Stream slave `pixel_in_axis` | First-beat metadata source: `[9:0] active_width`, `[19:10] active_height`, `[25:20] qp`, `[26] frame_start`, `[31:27] reserved_zero`. Ignored after the first accepted frame beat. |
| `m_axis_pixel_tdata` | output | 8 | AXI-Stream master `pixel_axis` | Accepted active luma sample in raster order. Field `luma_sample[7:0]`, unsigned integer. |
| `m_axis_pixel_tvalid` | output | 1 | AXI-Stream master `pixel_axis` | Registered valid for queued pixel beat. Held until `m_axis_pixel_tready`. |
| `m_axis_pixel_tready` | input | 1 | AXI-Stream master `pixel_axis` | Downstream ready from `macroblock_assembler`. |
| `m_axis_pixel_tlast` | output | 1 | AXI-Stream master `pixel_axis` | Generated TLAST on the validated final active pixel only. |
| `m_axis_frame_cfg_tdata` | output | 32 | AXI-Stream master `frame_cfg_axis` | Canonical config record: `[31:22] active_width`, `[21:12] active_height`, `[11:6] qp`, `[5] frame_start`, `[4:0] reserved_zero=0`. |
| `m_axis_frame_cfg_tvalid` | output | 1 | AXI-Stream master `frame_cfg_axis` | Registered valid for the single config beat of a legal frame. Held until `m_axis_frame_cfg_tready`. |
| `m_axis_frame_cfg_tready` | input | 1 | AXI-Stream master `frame_cfg_axis` | Ready from the frame-config broadcast fabric. Integration must drive this only when all consuming sinks accepted or can accept the single beat. |
| `m_axis_frame_cfg_tlast` | output | 1 | AXI-Stream master `frame_cfg_axis` | Always 1 on the single config beat. |
| `m_axis_status_event_tdata` | output | 32 | AXI-Stream master `status_event_axis` | Status record: `[31:26] event_id`, `[25:23] block_id`, `[22:21] severity`, `[20:11] mb_index`, `[10:7] subblock_index`, `[6:0] error_code`. |
| `m_axis_status_event_tvalid` | output | 1 | AXI-Stream master `status_event_axis` | Registered valid for status event queue head. Held until `m_axis_status_event_tready`. |
| `m_axis_status_event_tready` | input | 1 | AXI-Stream master `status_event_axis` | Ready from `lifecycle_status_monitor` status-event aggregation. |
| `m_axis_status_event_tlast` | output | 1 | AXI-Stream master `status_event_axis` | Always 1 on each single-beat status event packet. |

### Payload Packing Contracts
- `m_axis_pixel_tdata[7:0]`: `luma_sample`, unsigned integer, no padding.
- `m_axis_frame_cfg_tdata[31:0]`: MSB-first field-list order:
  - `[31:22] active_width`, unsigned integer, valid range 1..640.
  - `[21:12] active_height`, unsigned integer, valid range 1..360.
  - `[11:6] qp`, unsigned integer, valid range 0..51.
  - `[5] frame_start`, boolean, must be 1 for a legal frame.
  - `[4:0] reserved_zero`, constant 0.
- `m_axis_status_event_tdata[31:0]`: MSB-first field-list order:
  - `[31:26] event_id`.
  - `[25:23] block_id`, fixed to `3'd0` for `input_stream_adapter`.
  - `[22:21] severity`, `0=info`, `1=warning`, `2=error`, `3=fatal`.
  - `[20:11] mb_index`, fixed 0 because this block does not derive macroblock coordinates.
  - `[10:7] subblock_index`, fixed 0.
  - `[6:0] error_code`.

Status event encodings:
- `EVENT_FIRST_BEAT = 6'd1`, severity info, `ERROR_NONE = 7'd0`.
- `EVENT_ILLEGAL_METADATA = 6'd2`, severity error, error code identifies the failed metadata field.
- `EVENT_TLAST_ERROR = 6'd3`, severity error, error code identifies premature/missing/late TLAST.
- `EVENT_INGRESS_COMPLETE = 6'd4`, severity info, no error.
- `EVENT_FLOW_CONTROL_ERROR = 6'd5`, severity fatal, used only if internal overflow assertions are violated.

Error code encodings:
- `ERROR_NONE = 7'd0`
- `ERROR_WIDTH_RANGE = 7'd1`
- `ERROR_HEIGHT_RANGE = 7'd2`
- `ERROR_QP_RANGE = 7'd3`
- `ERROR_MISSING_FRAME_START = 7'd4`
- `ERROR_RESERVED_NONZERO = 7'd5`
- `ERROR_TLAST_PREMATURE = 7'd6`
- `ERROR_TLAST_MISSING = 7'd7`
- `ERROR_TLAST_LATE = 7'd8`
- `ERROR_PIXEL_FIFO_OVERFLOW = 7'd9`
- `ERROR_STATUS_QUEUE_OVERFLOW = 7'd10`
- `ERROR_FRAME_CFG_OVERWRITE = 7'd11`

## 3. Microarchitecture
### 3.1 Top-Level Block Diagram
```text
                        +-----------------------------+
pixel_in_axis --------->| ingress accept / AXIS slave |----+
                        +--------------+--------------+    |
                                       |                   |
                                       v                   v
                          +------------+---------+   +-----+------+
pixel_in_tuser[first] --->| metadata latch/check |-->| cfg skid   |--> m_axis_frame_cfg
                          +------------+---------+   +------------+
                                       |
                                       | legal_frame, frame_pixels_total
                                       v
                          +------------+---------+
                          | pixel count / TLAST  |
                          | checker / generator  |
                          +------------+---------+
                                       |
                                       v
                          +------------+---------+
                          | 16-deep pixel FIFO   |--> m_axis_pixel
                          +------------+---------+
                                       |
                         status events |
                                       v
                          +------------+---------+
                          | 8-deep status queue  |--> m_axis_status_event
                          +----------------------+
```

### 3.2 Datapath
**Ingress sampling and first-beat metadata**
- A beat is consumed only on `pixel_in_fire = pixel_in_tvalid && pixel_in_tready`.
- The first consumed beat in `STATE_IDLE` samples:
  - `raw_width[9:0] = pixel_in_tuser[9:0]`.
  - `raw_height[9:0] = pixel_in_tuser[19:10]`.
  - `raw_qp[5:0] = pixel_in_tuser[25:20]`.
  - `raw_frame_start = pixel_in_tuser[26]`.
  - `raw_reserved[4:0] = pixel_in_tuser[31:27]`.
- Range checks:
  - width legal iff `1 <= raw_width <= 640`; two unsigned 10-bit compares.
  - height legal iff `1 <= raw_height <= 360`; two unsigned 10-bit compares.
  - QP legal iff `raw_qp <= 51`; one unsigned 6-bit compare.
  - frame_start legal iff `raw_frame_start == 1'b1`.
  - reserved legal iff `raw_reserved == 5'b00000`.
- If all checks pass, latch:
  - `active_width_q[9:0] = raw_width`
  - `active_height_q[9:0] = raw_height`
  - `qp_q[5:0] = raw_qp`
  - `frame_start_q = 1'b1`
  - `frame_cfg_data_q[31:0] = {raw_width, raw_height, raw_qp, 1'b1, 5'b00000}`

**Pixel total calculation**
- `raw_width[9:0] * raw_height[9:0]` uses unsigned 10-bit by 10-bit multiplication.
- Product width is 20 bits because maximum `640*360 = 230400 < 2^18`, but the full multiplier result is stored as `frame_pixels_total_q[19:0]`.
- `frame_pixels_last_index_q[19:0] = frame_pixels_total_q - 20'd1`; subtraction is range-guaranteed because width/height are nonzero for legal frames.
- No saturation is applied; invalid zero dimensions are rejected before the subtract result is used.

**Pixel forwarding**
- Incoming active pixels are stored in `pixel_fifo` as `{generated_tlast, pixel_in_tdata}`.
- `generated_tlast = (pixel_count_q == frame_pixels_last_index_q)` for legal active frames.
- `pixel_in_tlast` is checked against `generated_tlast`:
  - `pixel_in_tlast && !generated_tlast`: premature TLAST event; the pixel may still be forwarded with generated TLAST = 0 if metadata was legal.
  - `!pixel_in_tlast && generated_tlast`: missing TLAST event; the final active pixel is forwarded with generated TLAST = 1 because downstream framing is derived from validated geometry.
  - A further accepted beat after `frame_pixels_total_q` active pixels before an external TLAST is a late TLAST/protocol error and is drained without forwarding.
- `pixel_count_q[19:0]` increments by one on each legal active pixel accepted and forwarded into `pixel_fifo`. It does not increment in metadata-error drain.
- Pixel data is unsigned 8-bit integer; no arithmetic, clipping, saturation, or format conversion is applied.

**Frame config output**
- The frame config skid stores one 32-bit data word plus TLAST=1.
- `m_axis_frame_cfg_tdata` is driven from `frame_cfg_data_q`; it remains stable while `m_axis_frame_cfg_tvalid && !m_axis_frame_cfg_tready`.
- Pixel FIFO output is held inactive until `cfg_sent_q` is set by the `m_axis_frame_cfg_tvalid && m_axis_frame_cfg_tready` handshake. This guarantees downstream blocks can latch frame constants before the first pixel beat becomes visible.

**Status event datapath**
- Status events are packed as `{event_id[5:0], block_id=3'd0, severity[1:0], 10'd0, 4'd0, error_code[6:0]}`.
- Up to two events may be generated by one accepted ingress beat, e.g. first-beat info plus illegal metadata error. The status queue accepts two enqueue lanes in one cycle when at least two free slots are available.
- `m_axis_status_event_tlast` is generated as constant 1 for every queued event.

**Bit-width derivation**
- Active dimensions: 10 bits each cover 0..1023 and legal ranges 1..640 and 1..360.
- QP: 6 bits cover 0..63 and legal range 0..51.
- Pixel count and total: 20 bits cover 0..1,048,575, above maximum 230,400.
- Pixel FIFO occupancy: 5 bits cover 0..16.
- Status queue occupancy: 4 bits cover 0..8.
- All counters use unsigned integer wraparound only at their natural binary width; control prevents reaching the wrap point in legal operation.

### 3.3 Control Logic
The block uses a small binary-encoded FSM. Binary encoding is sufficient because there are only four states and it minimizes FF count.

States:
- `STATE_IDLE = 2'd0`: no active frame locally; wait for first accepted beat.
- `STATE_ACTIVE = 2'd1`: legal frame metadata accepted; accept active pixels until `frame_pixels_total_q` pixels are accepted.
- `STATE_ERROR_DRAIN = 2'd2`: illegal metadata or fatal local protocol error; accept and discard ingress beats until an external TLAST is observed, subject to status queue space.
- `STATE_DONE = 2'd3`: local ingress frame is complete; no additional input is accepted until local clear. Because the block interface has no lifecycle `frame_done` or per-frame-clear input, the RTL may either return to `STATE_IDLE` after one cycle if system integration permits local back-to-back ingress, or hold `STATE_DONE` until synchronous reset in strict one-frame-in-flight builds. This is a uArch integration issue documented below.

Transitions:
- `STATE_IDLE`:
  - If no `pixel_in_fire`, remain idle.
  - If `pixel_in_fire` and metadata legal, latch metadata, enqueue first pixel, enqueue `EVENT_FIRST_BEAT`, assert `m_axis_frame_cfg_tvalid` next cycle, set `pixel_count_q=1`, transition to `STATE_ACTIVE` unless the frame is a one-pixel frame, in which case generate completion and transition to `STATE_DONE`.
  - If `pixel_in_fire` and metadata illegal, enqueue `EVENT_FIRST_BEAT` and `EVENT_ILLEGAL_METADATA`, do not emit config or pixel, transition to `STATE_ERROR_DRAIN` unless the same beat has `pixel_in_tlast`, in which case transition to `STATE_DONE`.
- `STATE_ACTIVE`:
  - On each `pixel_in_fire`, enqueue the pixel if `pixel_count_q < frame_pixels_total_q`.
  - If `pixel_in_tlast` mismatches `generated_tlast`, enqueue `EVENT_TLAST_ERROR` with the specific error code.
  - If accepting the generated final active pixel, enqueue `EVENT_INGRESS_COMPLETE` and transition to `STATE_DONE`.
  - If an unexpected additional beat is accepted after the active pixel total, enqueue late TLAST error, discard the beat, and transition to `STATE_ERROR_DRAIN` until external TLAST is seen.
- `STATE_ERROR_DRAIN`:
  - `pixel_in_tready` may assert only when there is status queue space for any required event.
  - Accepted beats are discarded and never forwarded to `pixel_axis`.
  - If an accepted beat has `pixel_in_tlast`, transition to `STATE_DONE`.
- `STATE_DONE`:
  - No event/completion flag is asserted merely because reset released.
  - `pixel_in_tready=0` in strict mode. If integrated with an external lifecycle clear not present in this port list, local state must be cleared before returning to `STATE_IDLE`.

AXI-Stream handshake rules:
- All master `tvalid` signals are registered.
- Once a master `tvalid` is asserted, its `tdata` and `tlast` remain stable until `tready` is high in the same cycle.
- `pixel_in_tready` is combinational from registered state: current FSM state, pixel FIFO free space, frame config skid availability for first legal beat, and status queue free space for possible event bursts. It does not depend combinationally on downstream `tvalid`.
- No master path double-registers a hidden internal valid ahead of the bus valid. The same registered valid that gates output `tready`/dequeue is the valid visible at the bus.

Open uArch issue:
- `input_stream_adapter` has no input from `lifecycle_status_monitor` indicating global `frame_done` or per-frame reset completion. Therefore the block alone cannot enforce the ERS rule "accept a new frame only after frame_done and clean per-frame lifecycle reset have completed." Either the integration fabric must hold the external source until lifecycle completion, or a future interface revision must add a local lifecycle-clear/accept-enable input. This spec does not invent that port.

### 3.4 Storage Elements
Registers:
- `state_q[1:0]`: reset `STATE_IDLE`; updates every cycle.
- `active_width_q[9:0]`: reset 0; loaded on legal first beat.
- `active_height_q[9:0]`: reset 0; loaded on legal first beat.
- `qp_q[5:0]`: reset 0; loaded on legal first beat.
- `frame_start_q`: reset 0; loaded to 1 on legal first beat.
- `frame_pixels_total_q[19:0]`: reset 0; loaded with `width*height` on legal first beat.
- `frame_pixels_last_index_q[19:0]`: reset 0; loaded with `width*height-1` on legal first beat.
- `pixel_count_q[19:0]`: reset 0; increments on legal active pixel acceptance.
- `cfg_sent_q`: reset 0; set on `m_axis_frame_cfg_tvalid && m_axis_frame_cfg_tready`; cleared for next frame.
- `frame_cfg_data_q[31:0]`: reset 0; loaded on legal first beat.
- `frame_cfg_valid_q`: reset 0; set on legal first beat; cleared on frame config handshake.
- `frame_cfg_tlast_q`: reset 0; set to 1 with the legal config beat; cleared on handshake.
- `local_error_seen_q`: reset 0; set when any error event is generated in the current local frame.

FIFO/buffer storage:
- `pixel_fifo`: 16 entries x 9 bits (`{tlast,data[7:0]}`), implemented as a shallow FF shift queue, not an addressed raw comb-read array. Reset empties occupancy; entry contents need not be reset because valid/occupancy gates reads.
- `status_event_queue`: 8 entries x 33 bits (`{tlast,data[31:0]}`), implemented as a shallow FF queue with two enqueue lanes and one dequeue lane. Reset empties occupancy; entry contents need not be reset.
- `frame_cfg_skid`: 1 entry x 33 bits (`{tlast,data[31:0]}`), implemented in named FFs.

Storage implementation:
- No storage structure is >=2048 bits; `sram_budget = 0` is legal.
- No raw addressed combinational `reg [W-1:0] mem [0:N-1]` arrays are allowed. The pixel and status queues should be implemented as shallow shift queues or an equivalent registered FIFO template with stable output and same-cycle bypass discipline.
- `flip_flop_budget ≈ 900 FF`, within the hard 2000 FF cap. This includes metadata/counters (~150 FF), frame config skid (~33 FF), pixel FIFO and control (~170 FF), status queue and control (~330 FF), event generation/FSM/margin (~220 FF).
- `sram_budget = 0 bits / 0 KiB; macro count = 0`.
- `area_budget_um2 <= 80000`; estimated standard-cell area is far below the 80000 um2 cap because the block has no SRAM macros and no arithmetic datapath beyond compares and a 10x10 product.

Machine-readable memory manifest:
```
# MEM pixel_fifo: 9x16 ports=1w1r impl=flop justification=contract requires a 16-beat elasticity window between ingress and assembler SRAM arbitration; deeper frame/line storage is owned by macroblock_assembler, not this adapter
# MEM frame_cfg_skid: 33x1 ports=1w1r impl=flop justification=single config beat must be held stable until the frame_cfg broadcast fabric accepts it
# MEM status_event_queue: 33x8 ports=2w1r impl=flop justification=status can generate first-beat plus error events in one cycle and the lifecycle monitor arbitrates among producers; 8 beats exceeds the required 4-event burst window without needing SRAM
```

## 4. Algorithm Mapping
No Python golden model body was provided for this block. The hardware mapping is derived from the ERS, FRD, and canonical interface contracts.

| Functional requirement | Hardware equivalent |
|---|---|
| Accept pixels only on `pixel_in_tvalid && pixel_in_tready` | AXI-Stream slave fire signal `pixel_in_fire`; no state updates from input occur without fire. |
| First-beat TUSER latch | In `STATE_IDLE`, `pixel_in_fire` loads metadata registers from fixed `pixel_in_tuser` bit slices. Later `pixel_in_tuser` values are ignored until the local frame is cleared. |
| Width/height/QP/rsv/frame_start validation | Unsigned comparators against 1, 640, 360, and 51; equality checks for `frame_start==1` and reserved bits zero. |
| Emit one `frame_cfg_axis` beat per legal frame | One-entry registered skid loaded with canonical `{width,height,qp,1,0}` on legal first beat; TLAST is 1. |
| Preserve metadata despite later TUSER changes | Downstream config is driven only from latched registers, never from live `pixel_in_tuser`. |
| Count active pixels | 20-bit `pixel_count_q`; increment on each active pixel accepted into the pixel FIFO. |
| Require input TLAST at `active_width*active_height-1` | 20-bit product stored at frame start; compare `pixel_count_q` to `frame_pixels_last_index_q` for each accepted beat. |
| Forward pixel raster order | 16-deep FIFO preserves accepted order; output handshake pops only after `m_axis_pixel_tvalid && m_axis_pixel_tready`. |
| Generate internal pixel TLAST | Store generated TLAST in the FIFO entry for the final active pixel, independent of late changes on external TLAST. |
| Emit status events | Pack event fields into 32-bit records and enqueue into status queue; TLAST=1 on every event packet. |

### 4a. Cross-Block Semantic Invariants
- **Invariant ID:** `INV-ISA-CFG-001`
  - **Applies to ports/state:** `pixel_in_tuser`, `active_width_q`, `active_height_q`, `qp_q`, `m_axis_frame_cfg_tdata`, `m_axis_frame_cfg_tvalid`, `m_axis_frame_cfg_tlast`.
  - **Golden reference point:** `encode_flat(pixels, qp, W, H)` outer container inputs `W`, `H`, and `qp`; ERS first-beat metadata contract.
  - **Tolerance:** exact equality of latched W/H/QP to first accepted TUSER fields; exact zero reserved field.
  - **Update/consume timing:** sampled on the first `pixel_in_tvalid && pixel_in_tready` in `STATE_IDLE`; emitted as a single config beat 1 cycle later and held until handshake.
  - **Downstream dependency:** `macroblock_assembler` derives padding geometry; `intra_rd_encode_core` selects QP tables; `entropy_bitstream_engine` emits outer container header; `lifecycle_status_monitor` starts frame lifecycle.
  - **Validation hook:** VCD `active_width_q`, `active_height_q`, `qp_q`, `frame_cfg_data_q`, `m_axis_frame_cfg_tvalid`, `m_axis_frame_cfg_tready`.

- **Invariant ID:** `INV-ISA-PIXEL-ORDER-002`
  - **Applies to ports/state:** `pixel_in_tdata`, `pixel_count_q`, `pixel_fifo`, `m_axis_pixel_tdata`, `m_axis_pixel_tlast`.
  - **Golden reference point:** input raster pixel list passed to `encode_flat(pixels, qp, W, H)`.
  - **Tolerance:** exact byte equality and exact ordering for all accepted active pixels.
  - **Update/consume timing:** each accepted active input beat enqueues one FIFO entry; each output handshake dequeues exactly one entry in the same order.
  - **Downstream dependency:** `macroblock_assembler` forms source rows and macroblocks; any drop/dup/reorder changes DCT input and corrupts the whole bitstream.
  - **Validation hook:** VCD `pixel_in_fire`, `pixel_count_q`, `pixel_fifo_occ_q`, `m_axis_pixel_tvalid`, `m_axis_pixel_tready`, `m_axis_pixel_tdata`, `m_axis_pixel_tlast`.

- **Invariant ID:** `INV-ISA-TLAST-003`
  - **Applies to ports/state:** `pixel_in_tlast`, `frame_pixels_total_q`, `frame_pixels_last_index_q`, `pixel_count_q`, `m_axis_pixel_tlast`, `m_axis_status_event_tdata`.
  - **Golden reference point:** frame length `W*H` from `encode_flat` input dimensions.
  - **Tolerance:** `m_axis_pixel_tlast` asserts exactly once, on active pixel index `W*H-1`; input TLAST mismatch produces an exact status error.
  - **Update/consume timing:** comparison performed on every accepted active beat; generated TLAST is stored atomically with the pixel FIFO entry.
  - **Downstream dependency:** `macroblock_assembler` needs the final-pixel boundary to complete frame assembly; lifecycle status depends on TLAST error events for clean termination.
  - **Validation hook:** VCD `generated_tlast`, `pixel_in_tlast`, `m_axis_pixel_tlast`, `status_event_queue` head fields.

- **Invariant ID:** `INV-ISA-STATUS-004`
  - **Applies to ports/state:** `status_event_queue`, `m_axis_status_event_tdata`, `m_axis_status_event_tvalid`, `m_axis_status_event_tlast`.
  - **Golden reference point:** ERS lifecycle/status event contract.
  - **Tolerance:** exact event field packing and no dropped/duplicated status events under backpressure.
  - **Update/consume timing:** events are enqueued in the same cycle as the detected condition and emitted in FIFO order.
  - **Downstream dependency:** `lifecycle_status_monitor` uses illegal metadata, TLAST, ingress completion, and fatal flow-control events to drive `status_error`, `frame_start`, and `frame_done`.
  - **Validation hook:** VCD `event_push_count`, `status_event_occ_q`, `m_axis_status_event_tdata`, `m_axis_status_event_tvalid`, `m_axis_status_event_tready`.

## 5. Reset and Initialization
- Reset polarity/type: synchronous active-low `rst_n`; all state updates occur in `always @(posedge clk)` with `if (!rst_n)`.
- Reset values:
  - `state_q = STATE_IDLE`.
  - All metadata registers, product registers, counters, and flags reset to 0.
  - `frame_cfg_valid_q = 0`, `m_axis_pixel_tvalid = 0`, `m_axis_status_event_tvalid = 0`.
  - FIFO occupancies reset to 0; data storage contents are don't-care because valid/occupancy gates outputs.
- No multi-cycle memory initialization is required.
- Reset-idle is not protocol completion:
  - `m_axis_frame_cfg_tvalid`, `m_axis_pixel_tvalid`, and `m_axis_status_event_tvalid` are 0 after reset.
  - No `EVENT_INGRESS_COMPLETE` is emitted on reset release.
  - No TLAST is asserted after reset unless attached to a real config/status/pixel beat.

## 6. Timing and Performance
- Target clock period: 20.00 ns at 50 MHz sky130 TT.
- Critical path estimate:
  - First-beat ready/check path includes FIFO/status queue free-space muxes, 10-bit range compares, a 6-bit compare, and event-count gating. Comparable to a few 16-bit compares and muxes, below 20 ns.
  - Product path is a 10x10 unsigned multiply. It is smaller than the characterized 16-bit multiply delay of 6.77 ns and is used only to load registered `frame_pixels_total_q`.
  - Active pixel path is FIFO push/pop control, counter compare, and 20-bit increment. This is below the 20 ns budget; no exhaustive search or multi-term arithmetic exists in this block.
- Pipeline/stage boundaries:
  - Stage 0: input handshake and combinational metadata legality/TLAST checks.
  - Stage 1: registered metadata/config, FIFO enqueue, counters, and event enqueue.
  - Output stage: registered FIFO/skid outputs hold data stable under backpressure.
- Throughput:
  - Best case after config handshake: 1 accepted pixel per cycle and 1 emitted pixel per cycle.
  - First legal frame beat incurs one config-registration cycle; pixel emission waits until `frame_cfg_axis` handshake completes.
- Backpressure:
  - `pixel_in_tready` deasserts when the pixel FIFO lacks room for an accepted active pixel, when the status queue lacks room for all events that could be generated by the beat, when the frame config skid is occupied on a first legal beat, or when the local FSM blocks input.
  - Output master data and TLAST remain stable while `tvalid=1` and `tready=0`.

### 6a. Output Timing Contract
| Output port | Type | Pipeline latency | First valid cycle after reset |
|---|---|---:|---:|
| `pixel_in_tready` | combinational | 0 from current registered occupancy/FSM state | 1 cycle after reset deassertion; high only if queues are empty and local policy accepts a new frame |
| `m_axis_pixel_tdata` | registered | 1 from pixel FIFO enqueue after config has handshaken | Earliest 2 cycles after first legal input beat if config ready is high |
| `m_axis_pixel_tvalid` | registered | 1 from eligible FIFO enqueue after config has handshaken | Earliest 2 cycles after first legal input beat if config ready is high |
| `m_axis_pixel_tlast` | registered | 1 from final active input beat enqueue | Earliest 2 cycles after a one-pixel legal frame first beat if config ready is high |
| `m_axis_frame_cfg_tdata` | registered | 1 from first legal input beat | Earliest 1 cycle after first legal input beat |
| `m_axis_frame_cfg_tvalid` | registered | 1 from first legal input beat | Earliest 1 cycle after first legal input beat |
| `m_axis_frame_cfg_tlast` | registered | 1 from first legal input beat | Earliest 1 cycle after first legal input beat |
| `m_axis_status_event_tdata` | registered | 1 from event generation when status queue was empty | Earliest 1 cycle after first accepted input beat |
| `m_axis_status_event_tvalid` | registered | 1 from event generation when status queue was empty | Earliest 1 cycle after first accepted input beat |
| `m_axis_status_event_tlast` | registered | 1 from event generation when status queue was empty | Earliest 1 cycle after first accepted input beat |

Representative timing for a legal frame with downstream ready asserted:
```wavedrom
{signal: [
  {name: 'clk',                       wave: 'p..........'},
  {name: 'rst_n',                     wave: '01.........'},
  {name: 'pixel_in_tvalid',           wave: '0.11110....'},
  {name: 'pixel_in_tready',           wave: '0.11110....'},
  {name: 'pixel_in_tdata',            wave: 'x.=.=.=.x..', data: ['P0','P1','P2']},
  {name: 'pixel_in_tuser(first)',      wave: 'x.=xxxxxxxx', data: ['W,H,QP']},
  {name: 'm_axis_frame_cfg_tvalid',    wave: '0..10......'},
  {name: 'm_axis_frame_cfg_tdata',     wave: 'x..=x......', data: ['cfg']},
  {name: 'm_axis_pixel_tvalid',        wave: '0...1110...'},
  {name: 'm_axis_pixel_tdata',         wave: 'x...=.=.=x.', data: ['P0','P1','P2']},
  {name: 'm_axis_pixel_tlast',         wave: '0...0.0.1..'},
  {name: 'm_axis_status_event_tvalid', wave: '0..1..1....'},
  {name: 'm_axis_status_event_tdata',  wave: 'x..=..=x...', data: ['FIRST','DONE']}
],
 head: {text: 'input_stream_adapter registered outputs; pixel output waits for cfg handshake'}}
```

## 7. Edge Cases and Corner Conditions
- Width below 1 or above 640: reject metadata, emit `EVENT_ILLEGAL_METADATA/ERROR_WIDTH_RANGE`, do not emit frame config or pixels for that frame.
- Height below 1 or above 360: reject metadata, emit `ERROR_HEIGHT_RANGE`.
- QP above 51: reject metadata, emit `ERROR_QP_RANGE`.
- Missing first-beat `frame_start`: reject metadata, emit `ERROR_MISSING_FRAME_START`.
- Nonzero `pixel_in_tuser[31:27]`: reject metadata, emit `ERROR_RESERVED_NONZERO`.
- External TUSER changes after first beat: ignored until the next locally accepted frame; no downstream metadata changes.
- Premature external TLAST: emit `ERROR_TLAST_PREMATURE`; generated internal pixel TLAST remains 0 for that pixel.
- Missing external TLAST on the final active pixel: emit `ERROR_TLAST_MISSING`; generated internal pixel TLAST is still 1 on the final active pixel so downstream active-frame geometry can complete.
- Late external TLAST/additional pixels after the active pixel count: discard extra pixels, emit `ERROR_TLAST_LATE`, and drain until external TLAST.
- Pixel FIFO full: deassert `pixel_in_tready`; no accepted beat can be dropped. If an internal overflow ever occurs despite this guard, emit fatal flow-control error.
- Status queue near full: deassert `pixel_in_tready` for beats that could require more event slots than are available.
- Empty/idle after reset: no completion event, no config beat, no pixel beat, no status beat.
- AXI packet boundaries: `m_axis_frame_cfg_tlast=1` on its only beat; `m_axis_status_event_tlast=1` on every single-beat event; `m_axis_pixel_tlast=1` only on the final active pixel.

## 8. Implementation Notes
- Do not propagate live `pixel_in_tlast` directly to `m_axis_pixel_tlast`; generate and store TLAST atomically with the pixel FIFO entry using the latched frame geometry.
- Do not read live `pixel_in_tuser` after the first accepted beat. The latched config is the only source for downstream geometry and QP.
- Use a single visible registered output valid per AXI-Stream master. Avoid a hidden internal valid that is one cycle ahead of the bus valid.
- The pixel FIFO and status queue must preserve order under arbitrary downstream stalls. DV should randomize `m_axis_pixel_tready`, `m_axis_frame_cfg_tready`, and `m_axis_status_event_tready`.
- Suggested verification points:
  - First beat metadata trace: `pixel_in_fire`, `pixel_in_tuser`, `active_width_q`, `active_height_q`, `qp_q`, `frame_cfg_data_q`.
  - Pixel ordering trace: accepted input byte sequence versus `m_axis_pixel` handshake sequence.
  - TLAST trace: `pixel_count_q`, `frame_pixels_last_index_q`, `pixel_in_tlast`, generated FIFO TLAST, and output TLAST.
  - Status trace: generated event fields, queue occupancy, status output handshakes.
- Contract-audit VCD signals: `state_q`, `cfg_sent_q`, `frame_cfg_valid_q`, `pixel_fifo_occ_q`, `status_event_occ_q`, `active_width_q`, `active_height_q`, `qp_q`, `pixel_count_q`, `frame_pixels_total_q`, `local_error_seen_q`.
- Sky130 synthesis considerations: all storage is FF-based small state; no SRAM macro instantiation is required. Avoid latches, tri-states, asynchronous resets, and raw comb-read arrays.

## 9. Verilog Interface Stub
```verilog
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
endmodule
```

```json
{
  "block_name": "input_stream_adapter",
  "latency_cycles": 1,
  "throughput_samples_per_cycle": 1.0,
  "pipeline_stages": 1,
  "register_count": 900,
  "rom_bits": 0,
  "estimated_gate_count": 3500,
  "fsm_states": ["STATE_IDLE", "STATE_ACTIVE", "STATE_ERROR_DRAIN", "STATE_DONE"],
  "data_width_in": 8,
  "data_width_out": 8,
  "fixed_point_format": "N/A",
  "interface_protocol": "axi_stream",
  "output_timing": {
    "pixel_in_tready": {"type": "combinational", "latency_cycles": 0},
    "m_axis_pixel_tdata": {"type": "registered", "latency_cycles": 1},
    "m_axis_pixel_tvalid": {"type": "registered", "latency_cycles": 1},
    "m_axis_pixel_tlast": {"type": "registered", "latency_cycles": 1},
    "m_axis_frame_cfg_tdata": {"type": "registered", "latency_cycles": 1},
    "m_axis_frame_cfg_tvalid": {"type": "registered", "latency_cycles": 1},
    "m_axis_frame_cfg_tlast": {"type": "registered", "latency_cycles": 1},
    "m_axis_status_event_tdata": {"type": "registered", "latency_cycles": 1},
    "m_axis_status_event_tvalid": {"type": "registered", "latency_cycles": 1},
    "m_axis_status_event_tlast": {"type": "registered", "latency_cycles": 1}
  },
  "semantic_invariants": [
    {
      "id": "INV-ISA-CFG-001",
      "description": "First accepted TUSER metadata is latched once, validated, and emitted as the canonical 32-bit frame config record.",
      "ports_or_state": ["pixel_in_tuser", "active_width_q", "active_height_q", "qp_q", "m_axis_frame_cfg_tdata"],
      "golden_reference": "encode_flat(pixels, qp, W, H) frame parameters",
      "tolerance": "exact",
      "validation_hook": "VCD active_width_q/active_height_q/qp_q/frame_cfg_data_q and frame_cfg handshake"
    },
    {
      "id": "INV-ISA-PIXEL-ORDER-002",
      "description": "Every accepted active input pixel is emitted exactly once on pixel_axis in the same raster order.",
      "ports_or_state": ["pixel_in_tdata", "pixel_fifo", "m_axis_pixel_tdata"],
      "golden_reference": "encode_flat input pixel raster",
      "tolerance": "exact byte equality and order",
      "validation_hook": "Compare pixel_in_fire sequence to m_axis_pixel handshake sequence"
    },
    {
      "id": "INV-ISA-TLAST-003",
      "description": "Internal pixel TLAST is generated exactly on active pixel index W*H-1 and input TLAST mismatches produce status errors.",
      "ports_or_state": ["pixel_in_tlast", "frame_pixels_last_index_q", "pixel_count_q", "m_axis_pixel_tlast", "m_axis_status_event_tdata"],
      "golden_reference": "frame length W*H from encode_flat dimensions",
      "tolerance": "exact one TLAST at final active pixel",
      "validation_hook": "VCD generated_tlast/pixel_in_tlast/m_axis_pixel_tlast/status events"
    },
    {
      "id": "INV-ISA-STATUS-004",
      "description": "Ingress status events are packed and delivered in order without loss under backpressure.",
      "ports_or_state": ["status_event_queue", "m_axis_status_event_tdata", "m_axis_status_event_tvalid", "m_axis_status_event_tlast"],
      "golden_reference": "ERS lifecycle/status event contract",
      "tolerance": "exact field packing and event count",
      "validation_hook": "VCD event_push_count/status_event_occ_q/status event handshakes"
    }
  ]
}
```