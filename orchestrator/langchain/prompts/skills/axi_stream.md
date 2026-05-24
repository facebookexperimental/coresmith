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

## When NOT to use AXI-Stream

- Pure 1-cycle latency single-beat handshake: use **sRdy/dRdy**
  (see `skills/srdy_drdy.md`). Lower overhead, no sideband sprawl.
- Address-mapped register access: use AXI4-Lite, not AXI-Stream.
- Wide parallel buses with no ordering: a raw `valid/data` bundle
  with no backpressure may suffice if the consumer can never stall.
