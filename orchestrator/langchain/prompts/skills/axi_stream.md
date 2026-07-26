# Skill: AXI-Stream

Reference document for any agent designing or implementing AXI-Stream
interfaces in coresmith. Use this when authoring an interface contract,
a uArch spec, or RTL that carries multi-field payloads between blocks.

## What AXI-Stream is

A unidirectional, point-to-point, multi-beat data interface specified
by ARM AMBA AXI4-Stream. One master (producer) drives data into one
slave (consumer). Backpressure is per-beat via a ready/valid handshake.

## Required signals (every AXI-Stream interface has these)

| Signal | Direction | Purpose |
|--------|-----------|---------|
| `<prefix>_tvalid` | master → slave | master asserts when data is presented |
| `<prefix>_tready` | slave → master | slave asserts when it can accept the beat |
| `<prefix>_tdata[N-1:0]` | master → slave | the payload — N is the design choice |

A beat transfers iff `tvalid && tready` on the same rising edge. Both
sides are free to assert their signal independently in any cycle, with
**one critical rule**: once `tvalid` is asserted, it must stay asserted
and `tdata` must stay stable until `tready` is also asserted. The
master is forbidden from waiting for `tready` before asserting
`tvalid` (it would create a combinational loop in protocol-checker
terms).

## Optional sideband signals

Use only if the design needs them; redundant sidebands are noise.

| Signal | Width | Use case |
|--------|-------|----------|
| `<prefix>_tlast` | 1 | marks the last beat of a packet/frame |
| `<prefix>_tuser[U-1:0]` | design-specific | per-beat metadata that travels with the data but is not part of the data itself (e.g., start-of-frame, parity, source-id) |
| `<prefix>_tkeep[N/8-1:0]` | one bit per `tdata` byte | which bytes of `tdata` are valid this beat — used for byte-streams with sparse beats |
| `<prefix>_tstrb[N/8-1:0]` | one bit per `tdata` byte | which bytes are position bytes vs null bytes — almost never used in coresmith designs |
| `<prefix>_tdest[D-1:0]` | route id | demux target on the slave side, for switches |
| `<prefix>_tid[I-1:0]` | source id | identifies the producer when slave aggregates multiple masters |

## `tdata` packing is a design choice, not a standard

This is the most common source of inter-block contract drift in coresmith.
AXI-Stream itself does **not** specify how multi-field payloads pack
inside `tdata`. The bit order, field placement, signedness, and
alignment are all decisions you must make explicitly and freeze in
the interface contract.

When you author or consume an AXI-Stream interface that carries more
than one field, you **must** declare:

1. **Total width** of `tdata` in bits.
2. **Field list**, ordered, with each field's exact `[MSB:LSB]` slice
   inside `tdata`.
3. **Signedness** of each field (`unsigned`, `two's-complement`, etc.).
4. **Encoding** of each field (binary, gray, one-hot, fixed-point Qm.n,
   ASCII byte, etc.). Especially important for non-trivial fields like
   QP indices or mode codes.

Both endpoints **must** reference the same frozen declaration. There
are valid reasons to pick LSB-first packing in some designs (e.g., when
the consumer treats `tdata` as a sliding byte stream and the first
byte of a multi-byte field arrives at `[7:0]`), and equally valid reasons
to pick MSB-first (e.g., when serializing a structured record where the
field order matches network/file byte order). **Neither is universally
correct.** What matters is that producer and consumer agree.

### Default convention for coresmith generated designs

When the architect has no specific reason to pick otherwise:

- **MSB-first by field-list order**: the first field declared in the
  interface contract occupies the highest bits; the last field occupies
  the lowest bits.
- **No padding** between fields unless the contract explicitly declares a
  named `reserved`/`pad` field.
- **Two's-complement** for signed fields.
- **Unsigned** for indices, counters, and addresses.

Example: a 36-bit pixel context with fields `pixel[8]`, `x[10]`,
`y[9]`, `frame_idx[4]`, `qp[2]`, `sof[1]`, `eol[1]`, `frame_last[1]`
packs as:

```
[35:28]  pixel       (8 bits, unsigned)
[27:18]  x           (10 bits, unsigned)
[17:9]   y           (9 bits, unsigned)
[8:5]    frame_idx   (4 bits, unsigned)
[4:3]    qp          (2 bits, unsigned)
[2]      sof         (1 bit)
[1]      eol         (1 bit)
[0]      frame_last  (1 bit)
```

If a design has a strong reason to deviate (e.g., wire-compatibility
with an external IP that expects LSB-first), declare that explicitly
in the interface contract's `packing_convention` field and reference
it from both endpoints' uArch specs.

## Backpressure correctness

Common mistakes that pass lint but fail simulation:

1. **`tvalid` deasserts mid-burst before `tready` fires.** Once
   asserted, hold both `tvalid` and `tdata` stable until `tready`.
2. **`tready` depends combinationally on `tvalid`.** This creates a
   `valid && ready` cycle that some protocol checkers will accept but
   confounds downstream skid buffers. Slave should derive `tready` from
   its internal state only.
3. **Producer asserts `tvalid` before reset deasserts.** Some
   downstream blocks latch this as a phantom beat.

4. **Double-registered output + `tready` gated on the INTERNAL valid =
   silent beat drop under backpressure.** A proven, insidious bug: a block
   keeps an internal output register (`out_valid`/`out_data`) AND *also*
   re-registers it onto the bus (`m_axis_tvalid <= out_valid;
   m_axis_tdata <= out_data;`), then computes its upstream accept as
   `s_axis_tready = ... and ((not out_valid) or m_axis_tready)`. The
   `out_valid` gating `s_tready` is **one cycle ahead** of the
   `m_axis_tvalid` the consumer sees. Under downstream backpressure the
   block accepts a NEW input and overwrites `out_data` before the previous
   beat was ever handshaken — beats vanish (an observed 4-in / 2-out,
   exactly-50%-drop; holding `tready` high hid it entirely). Passes lint
   and any DV that never stalls the consumer.
   - **CORRECT PATTERN:** present exactly ONE registered output beat —
     drive `m_axis_*` combinationally from a single `out_*` reg, or use a
     proper valid/ready skid buffer — but the valid gating `s_tready` MUST
     be the SAME valid the consumer sees (`m_axis_tvalid`), never an inner
     pre-output valid. Do not accept a new input until the current output
     beat is handshaken (`m_axis_tvalid && m_axis_tready`).
   - **ANTI-PATTERN:** `m_axis_tvalid <= out_valid` (an extra register
     stage) while `s_tready = (not out_valid) | m_axis_tready` — accepting
     input based on a valid the consumer hasn't seen yet.

5. **`tready` HIGH on a cycle the slave will not latch the input (an internal
   drain/flush/stall branch wins the if/elif) = ACK-AND-DROP.** This is the
   slave/accept-side dual of rule 4. When a slave's clocked accept logic is a
   PRIORITY chain — `if drain_byte/flush: …  elif token_valid and tready:
   accumulate` — its combinational `tready` must be FALSE on exactly the cycles
   the accept branch will NOT execute, which includes EVERY cycle a
   higher-priority branch (drain/flush/stall) fires. Deriving `tready` from only
   part of that arbitration leaves it asserted on a drain cycle: upstream sees
   `tvalid && tready`, treats the beat as transferred and pops it, but the slave
   took the drain branch and never accumulated it → the token is **acknowledged
   and silently dropped**. PROVEN failure: a `block_packer` drove
   `s_axis_bit_token_tready = (not flushing) and (not byte_valid)` while its
   sequential body was `if next_count >= 8: drain_byte  elif token_valid and
   tready: accumulate`; on every byte-drain cycle `byte_valid` was still low so
   `tready` stayed HIGH, `entropy_enc` popped its token, and the packer dropped it
   — 455 emitted entropy coding bits retained as ~296 (golden 57 bytes truncated to 37,
   byte0 0x81 correct then divergence at bit 14). Every block passed its own
   golden; the loss was entirely in the packer's accept handshake, and because
   it is deterministic the gate's bounded revise produced byte-identical wrong
   output every round.
   - **CORRECT PATTERN:** `tready` must be the EXACT condition under which the
     latch/accept branch runs THIS cycle — the same guard the accept branch
     uses, AND-reduced with "no higher-priority branch is active":
     `s_tready = accept_branch_will_execute` (e.g. `(not draining) and (not
     flushing) and room_for_token`). A token is consumed IFF it is latched;
     never let a drain/flush/stall cycle acknowledge an input it won't store.
   - **ANTI-PATTERN:** `s_tready = (not flushing) and (not byte_valid)` —
     omitting the in-progress drain term that actually steals the cycle from the
     accept branch.
   - **CHECK:** over a known frame, count tokens POPPED by the upstream
     (`tvalid && tready`) vs tokens ACCUMULATED by the slave — they MUST be
     equal; any delta is a dropped beat. Also assert the composed bit-length
     equals the sum of upstream codeword lengths (a short stream = lost beats).

**Mandatory: verify every AXI-Stream producer under RANDOM downstream
backpressure, not just `tready=1`.** A block whose per-block DV only runs
with the consumer always-ready will pass while silently dropping beats in
the real (bursty, back-pressured) chip. Debugging heuristic: if a block is
byte-exact standalone with `tready=1` but wrong/short in-system, suspect
this handshake race FIRST (before blaming the arithmetic) — re-run it
standalone with randomized `tready` and compare beat counts; a delta proves
the skid bug.

## Bootstrap and closed-loop dependencies

If block A's `s_axis` waits for data from block B's `m_axis`, and
block B's `s_axis` waits for data from block A's `m_axis` (a feedback
loop), the design **must** declare an initial-cycle policy in the
interface contract. Options:

- **Reset seed**: one endpoint's `m_axis` emits a default value on
  cycle 0 after reset, valid until consumed. The default value must
  be specified in the contract.
- **Request-driven**: add a separate request channel from consumer
  to producer carrying the coordinate the consumer needs; producer
  responds with current state.
- **Bypass for the first transaction**: declare which endpoint is
  primed externally before the rest of the pipeline starts.

Closed loops with no declared bootstrap policy **will deadlock** on
the first transaction. The contract must call this out.

## Latch the WHOLE beat at the fire edge (payload included)

The tlast-latching rule above generalizes to EVERY field of a beat: consume
``tdata``/``tuser``/sideband payload ON the cycle ``tvalid && tready`` fires,
into a register, and compute from the REGISTER afterwards. Sampling any part
of the beat one cycle later reads whatever the producer legally drives next
(producers may clear or repurpose ``tdata`` immediately after the beat).
Proven live (armD RD core): a response payload read one cycle post-handshake
evaluated every candidate on zeros -- and had previously "passed" only
because that testbench happened to hold ``tdata`` stable after the beat. A
consumer that works only when the producer holds data past the fire edge is
an AXI contract violation waiting for a compliant producer.

## Frame framing across a pipeline (`tlast` must propagate)

When framed data flows through a multi-block pipeline (`tuser` marks the
first beat of a frame, `tlast` the last), **every block must propagate the
frame boundary**: when a block consumes an input beat carrying `tlast`, it
must assert `tlast` on its own LAST output beat for that frame, then return
to idle and emit nothing until the next frame's first beat. A block that
drops `tlast` leaves the chip unable to signal end-of-frame — the egress
free-runs, emits unbounded output, and any consumer or test harness waiting
on output `tlast` hangs forever. Per-frame output beat count must be
deterministic and `tlast` must land exactly on the final beat (never one too
many / one too few).

### LATCH `tlast` at ingress when one input beat expands into a multi-beat burst

> **MANDATORY (most-violated rule).** If your block can emit MORE output beats
> than it consumed input beats (any 1-to-N expansion), you MUST declare a
> registered `last_in` (and `tuser_in`) latch and drive output `tlast` from the
> LATCH. Referencing the live `s_axis_*_tlast` anywhere inside the output-emit /
> last-beat expression is a BUG, full stop — grep your block: if a live input
> `*_tlast` port name appears in the assignment to an output `*_tlast`, you have
> the defect. There is no correct variant of that; the input port is 0 by the
> time the burst's last beat leaves. Latch at accept, read the latch at emit.

A block that consumes **one** input beat and then emits **many** output beats
over the following cycles (1-to-N expansion: an entropy/entropy coding coder turning one
pixel_block word into N bytes, a serializer turning one wide word into N lanes, a
run-length expander, a packetizer adding a multi-word payload) MUST **latch the
input's `tlast` (and `tuser`/frame-id) into a register at the cycle the input
beat is accepted**, and drive its output `tlast` from that LATCHED copy on the
final emitted beat. The upstream `s_axis_*_tlast` wire is only valid for the one
cycle the input beat fires; it deasserts on the very next cycle when the producer
drops `tvalid`. The burst, however, drains over many subsequent cycles. So:

- **CORRECT:** `last_in = Signal(bool(0))`; on input accept
  `last_in.next = s_axis_tlast`; on the final output beat of the burst
  (`out_idx + 1 == out_len`) drive `m_axis_tlast.next = (out_len == 1 and
  last_in) ` for a 1-beat burst, and for beat k>0 drive
  `m_axis_tlast.next = (k + 1 == out_len) and last_in` — always reading the
  **registered** `last_in`, never the live upstream port.
- **ANTI-PATTERN (drops the frame boundary):** re-reading the live
  `s_axis_*_tlast` while emitting beat k of the burst, e.g.
  `m_axis_tlast.next = (byte_idx + 2 == byte_len) and s_axis_selected_mb_tlast`.
  By the time the last byte is emitted, the input beat retired many cycles ago
  and `s_axis_selected_mb_tlast` reads 0, so the output `tlast` NEVER fires. The
  egress never signals end-of-frame: the harness's frame-done counter stays 0,
  recon/stats are never finalized, and the bitstream looks "truncated" only
  because nothing marks its end.

This is the SAME end-of-frame failure as dropping `tlast` outright, but it hides
in 1-to-N blocks because the single-beat input case (`out_len == 1`) often
happens to work — the bug only surfaces when a burst spans >1 cycle. Per-block DV
MUST drive a producer whose burst length is >1 and assert that the LAST emitted
beat (and only it) carries `tlast`, with the input `tvalid`/`tlast` deasserted
during the drain (i.e. the input beat is long gone while the burst is still
streaming).

## When NOT to use AXI-Stream

- Pure 1-cycle latency single-beat handshake: use **sRdy/dRdy**
  (see `skills/srdy_drdy.md`). Lower overhead, no sideband sprawl.
- Address-mapped register access: use AXI4-Lite, not AXI-Stream.
- Wide parallel buses with no ordering: a raw `valid/data` bundle
  with no backpressure may suffice if the consumer can never stall.
