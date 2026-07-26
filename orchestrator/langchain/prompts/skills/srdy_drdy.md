# Skill: sRdy/dRdy handshake

Reference document for any agent designing or implementing
source-ready / destination-ready (sRdy/dRdy) interfaces in coresmith.
Use this when authoring an interface contract, a uArch spec, or RTL
that carries a single-beat or low-overhead payload between blocks.

## What sRdy/dRdy is

A unidirectional, point-to-point, per-cycle handshake. The source
asserts `srdy` to indicate it has data ready. The destination asserts
`drdy` to indicate it can accept data this cycle. A transfer happens
iff `srdy && drdy` on the same rising edge.

This is functionally equivalent to AXI-Stream's `tvalid`/`tready` but
intentionally stripped of sidebands. Use it when the design only
needs the handshake itself and a payload bus, with no per-beat
metadata, no packet boundaries, and no demux/aggregation routing.

## Required signals

| Signal | Direction | Purpose |
|--------|-----------|---------|
| `<prefix>_srdy` | source → dest | source has data this cycle |
| `<prefix>_drdy` | dest → source | dest can accept this cycle |
| `<prefix>_data[N-1:0]` | source → dest | payload — N is the design choice |

That's it. No `last`, no `user`, no `keep`, no `dest`, no `id`. If
the design needs any of those, switch to AXI-Stream — don't graft
sidebands onto sRdy/dRdy ad hoc.

## Handshake correctness

The same beat-stability rule as AXI-Stream applies:

- Once `srdy` is asserted with a particular `data` value, both must
  remain stable until `drdy` is also asserted.
- `drdy` must not depend combinationally on `srdy` (no `drdy = srdy && ...`).
  Slave's `drdy` is derived from its internal state only.

## Payload packing

Same rule as AXI-Stream: sRdy/dRdy itself does not specify how
multi-field payloads pack inside `data`. If the payload is a single
scalar (a pixel, an address, an opcode), packing is trivial. If
it's a record with multiple fields, declare:

1. Total `data` width in bits.
2. Field list with `[MSB:LSB]` per field.
3. Signedness and encoding.

Default coresmith convention (MSB-first by field-list order, no
padding, two's-complement for signed) applies — see
`skills/axi_stream.md` for the rationale and the worked codec-style
example. The convention is the same; what differs between the two
protocols is only the sideband surface area.

## When to choose sRdy/dRdy over AXI-Stream

Pick sRdy/dRdy when **all** of the following hold:

- Payload is a single field, or a small record that's atomic at the
  protocol level (no packet boundaries).
- No need for per-beat sidebands (no start-of-frame marker, no
  parity, no source-id, no routing target).
- The interface is a tight 1-cycle local handshake — typically inside
  a single subsystem or between adjacent stages of a pipeline.
- You want the smallest possible interface (no `tlast`, `tuser`,
  `tkeep`, `tstrb`).

Pick AXI-Stream when **any** of the following hold:

- The payload carries packet boundaries (`tlast`) or per-beat
  metadata (`tuser`).
- Multiple producers feed one consumer with source IDs (`tid`).
- One producer feeds multiple consumers via a switch (`tdest`).
- You want the IP to be interoperable with off-the-shelf AXI-Stream
  blocks (FIFOs, width converters, fork/join, AXI DMAs).
- You're at a chip boundary or crossing a subsystem — AXI-Stream's
  larger sideband surface is the right interop default.

## Do NOT handshake a compile-time-enumerable sequence

A per-iteration request/response handshake is for **genuinely data-dependent**
access — where the address/next-item is not known until the current result is
in hand. It is the WRONG tool for a sequence that is enumerable at compile time:
a fixed round order `0..N`, a fixed tap/coefficient sweep, a fixed scan over a
known address range. Wrapping each of those steps in a `req → wait drdy → resp`
handshake pays a full round-trip latency **per element** for an order you already
know — turning an N-element pass into ~N handshake round-trips.

For a compile-time-known sequence:
- **Pre-stage locally.** Read/prepare element `k+1` in parallel with consuming
  element `k` (compute one step ahead), so the datapath never stalls waiting on a
  handshake it did not need. A small local cache / registered look-ahead removes
  the per-step round-trip.
- Drive the known sequence from a counter/FSM, not from a request grant per item.
- **Reserve handshakes for data-dependent access only** — a lookup whose address
  comes from a just-computed value, a variable-length stream, backpressure from a
  consumer whose readiness genuinely varies. There, the handshake earns its
  latency; on an enumerable sequence it is pure overhead.

## Bootstrap and closed-loop dependencies

Identical concern as AXI-Stream: if a feedback loop exists
(consumer waits on data the producer can only emit after consuming
something the consumer hasn't sent yet), the design **must** declare
an initial-cycle policy. The smaller sideband surface of sRdy/dRdy
does not change this requirement.

## Coresmith conventions for sRdy/dRdy ports

Naming:
- Master output signals: `m_srdy`, `m_data`, master input: `m_drdy`
  (mirrors the AXI-Stream `m_axis_*` naming convention).
- Slave input signals: `s_srdy`, `s_data`, slave output: `s_drdy`.
- For multiple interfaces on the same block, qualify with a role:
  `m_pixel_srdy`, `s_neighbor_drdy`, etc.

If multiple sRdy/dRdy interfaces are needed, name them so the role
is clear at the port list — the interface contract's `port_role` field
should match the prefix used in RTL.
