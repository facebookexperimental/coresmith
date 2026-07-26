# Skill: QSPI-slave frontend protocol completeness — the COMPLETE command set + timing

The chassis exposes a register-mapped accelerator to an external host MCU over a
**QSPI-slave** bus on the Caravel `io_in`/`io_out`/`io_oeb` GPIO pins. The host
side of this bus is fixed and standard (it is the grader's host BFM — you never
see it and you cannot co-tune to it). The frontend/IO-subsystem block you author
(`qspi_slave_frontend`, `io_subsystem`, `host_if`, the block that decodes the
QSPI command byte and bridges to the register map / IN / OUT windows) MUST
implement the WHOLE protocol below. Apply this skill whenever you author, spec,
or review the block that owns the external QSPI-slave pin boundary.

This is a **recurring, proven** bug class. Every one of these shipped a frontend
that was protocol-INCOMPLETE and passed the LLM/in-house DV but failed the fixed
grader host on a trivial bus gap — despite a correct compute datapath:

- **Missing `0x05` read_status.** The frontend decoded only `0x02` write and
  `0x03` read. The host's `wait_done()` polls `STATUS.DONE` via cmd `0x05` — with
  no `0x05` decode it reads garbage, never sees DONE, and TIMES OUT. The whole
  chip fails on a missing command.
- **`0x05` serializer misalignment.** `0x05` decoded, but the status byte was
  shifted onto the lanes one nibble off, so DONE/ERROR landed in the wrong bit
  positions and the host read the wrong status.
- **Read launched a nibble early (short/missing dummy turnaround).** On `0x03`
  read (and `0x05` status), the frontend started driving read data BEFORE the
  bus-turnaround dummy phase completed, so every read byte was de-aligned by a
  nibble.

They are one systemic failure: the frontend re-derives the bus protocol per
design and keeps losing a command or a turnaround. Do not re-derive it. Copy the
complete contract below verbatim.

## The COMPLETE command set (all mandatory)

The command byte is the FIRST byte after chip-select (`csn`) goes low. Decode
ALL of these; an unknown opcode sets `STATUS.ERROR`:

| Cmd  | Name          | Frame                                                        |
|------|---------------|-------------------------------------------------------------|
| 0x02 | WRITE         | `cmd, addr[23:0], data...` (auto-incrementing, little-endian) |
| 0x03 | READ          | `cmd, addr[23:0], <1 dummy byte>, data...` (auto-inc, LE)    |
| 0x05 | READ_STATUS   | `cmd, <1 dummy byte>, status_byte`                          |

- `0x02` / `0x03` carry a **24-bit (3-byte) address**, MSB byte first.
- The register/window map is the standard chassis map: `CTRL` (W: bit0 START,
  bit1 SOFT_RESET), `STATUS` (R), `CFG0`/`CFG1` (RW scalars), IN window
  (W, auto-inc), OUT window (R, auto-inc). CFG0/CFG1 are **read/write** — a value
  written via `0x02` must read back byte-identically via `0x03`.

### 0x05 READ_STATUS returns the DONE + ERROR status bits (do NOT omit this)

`STATUS` is a single byte with fixed bit positions:

- bit0 `BUSY`  — high while an operation is in flight.
- bit1 `DONE`  — high when the current operation has completed.
- bit2 `ERROR` — high on a protocol/opcode error (e.g. an unknown command byte).

The host completes EVERY operation by polling `0x05` until `DONE` is set. A
frontend without a `0x05` decode, or one that drives the wrong bits, hangs the
host. `0x05` has NO address; after the command byte it emits exactly the same
number of dummy (turnaround) bytes as `0x03` before driving the status byte.

## The timing (mode 0, quad IO, MSB-nibble-first)

- **Quad IO, 4 lanes.** One transfer moves a NIBBLE (4 bits) per SCK edge; a byte
  is TWO nibbles.
- **MSB-nibble-first.** Within a byte the HIGH nibble `data[7:4]` goes first, then
  the low nibble `data[3:0]`. (This is the exact bug class where reads emitted the
  low nibble twice / in both slots — reproduce MSB-nibble-first on BOTH directions.)
- **Mode 0: drive on the falling edge, sample on the rising edge.** The host
  drives write nibbles while SCK is low and samples read nibbles on the rising
  edge; the frontend MUST do the mirror — sample host writes on the rising SCK
  edge, and present read data while SCK is low so it is stable at the next rising
  edge.
- **Dummy-byte turnaround before read data — and PREFETCH during it.** On `0x03`
  and `0x05`, after the address (for `0x03`) there is exactly ONE dummy byte of
  bus turnaround where NEITHER side drives meaningful data — this is where bus
  ownership flips from host to DUT. Use that dummy phase to **prefetch** the first
  read byte into the output shift register so the FIRST read nibble is already
  aligned and valid at the first post-dummy rising edge. Do NOT start emitting
  read data during the dummy phase (that launches the read a nibble early) and do
  NOT wait until after the dummy phase to begin the fetch (that launches it a
  nibble late / inserts a bubble). The registered-response READ path prefetches
  DURING the dummy phase; the dummy exists precisely to hide the fetch latency.

## The frontend is a bus bridge, not compute

The frontend/IO-subsystem owns ONLY the bus protocol + register map + IN/OUT
window addressing + the START/DONE/ERROR status handshake. It is independent of
the compute lane (JPEG, AES, FFT, raster, …): the SAME complete command set +
timing applies to every accelerator. Author it once, completely, from this
contract — never a subset "just enough to move this design's data".

## Self-check before you finish the frontend

The block-level and integration DV run a **bus-protocol conformance** check that
is INDEPENDENT of the compute lane (it does not need the golden compute model).
It will FAIL your frontend — at generation, not at the secret grader — if any of
these is wrong, so verify them yourself:

1. Write a value to `CFG0`/`CFG1` via `0x02`, read it back via `0x03` through the
   dummy turnaround — it must match byte-for-byte (catches nibble-early reads and
   read-serializer misalignment).
2. `0x05` READ_STATUS decodes and returns a well-formed status byte at idle
   (`BUSY=0`, `DONE=0`).
3. An unknown/bad opcode sets `STATUS.ERROR`, and `0x05` reports it.

If you decode only `0x02`/`0x03`, or start read data before the dummy byte
completes, or drop `0x05`, this conformance DV fails. Implement all three
commands and prefetch the read during the dummy phase.
