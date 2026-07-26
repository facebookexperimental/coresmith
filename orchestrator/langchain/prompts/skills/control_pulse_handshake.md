# Skill: Control-pulse handshake — one-shot commands and back-to-back operations

Command and status handshakes across a control boundary are a recurring source
of silent bugs: a START pulse that re-triggers with stale config, a DONE level
that a fast consumer samples from the PREVIOUS operation. This skill is the
contract for one-shot command pulses and level status across back-to-back
operations. Apply it whenever a block is armed by a START/GO and reports a
DONE/status (a register-mapped accelerator, an FSM launched by a control pulse,
any block that runs discrete operations back to back).

## One-shot command pulses (START / GO / KICK)

A one-shot command must be **cleared by the consumer's OBSERVED ACCEPTANCE**,
never held until the block is idle again:

- The producer asserts START for one cycle (or holds it) and the consumer
  latches the request and its accompanying config on the accepting edge.
- Clear/deassert START when the consumer's **acceptance is observed** — e.g. the
  block asserts `busy`/`ack` and the producer sees it — NOT on "block idle".
- **Never** hold START asserted until the block returns to idle: when the block
  finishes and drops `busy`, a still-asserted START **immediately re-triggers**
  the operation, now with whatever (stale) config is on the bus. That spurious
  re-launch is the classic auto-re-trigger bug.
- Latch the config **with** the accepted START. Do not let the datapath read
  live config registers mid-operation; a back-to-back writer may have already
  changed them for the NEXT operation.

## DONE / status levels across back-to-back boundaries

A DONE/valid/status **level** exposed across operations must be qualified so a
consumer cannot read the PREVIOUS operation's result as this one's:

- Gate any exposed DONE/status on **"busy seen since THIS operation's START"** —
  the block must have entered busy for the current launch before its DONE is
  visible. A bare `done` level that was left high from the prior operation is
  read as an early (wrong-operation) completion by a fast poller.
- Clear DONE at the accepted START of the next operation (before asserting busy),
  so there is no window where the old DONE and the new START coexist.

## Decomposition tax — do NOT pay cycles at registered boundaries

Decomposing a design into modules and control states is correct, but each
throwaway registered boundary adds a cycle to EVERY op and shows up directly in
the delivery-time measured cyc/op (see the throughput_budget_contract skill and
the measured-throughput gate). Three specific wastes, each proven to cost real
cycles against a golden that avoids them:

1. **Accept the command on the FINAL cycle of the incoming write, not a
   registered cycle after it.** When a START/command is delivered as the last
   beat/shift of a bus write, decode and latch it MID-SHIFT — on the same edge
   that completes the write — so the datapath launches the next cycle. A design
   that first registers "write done", THEN in a following state decodes the
   command, pays one full cycle per op for nothing. (The AES golden decodes
   START mid-shift; the delivered design paid +3 by registering first.)
2. **No single-purpose bridge / wait state between two registered
   inter-module handshakes.** If module A registers its output and module B
   registers its input, a state that exists only to "hand the token across" is
   a dead cycle — FOLD the accept into the consuming state (B samples A's
   registered valid directly and advances). A `WAIT_FOR_B` / `BRIDGE` state
   whose only body is `state <= NEXT` is the tell. (This class cost +2.)
3. **Drive status/DONE output PINS combinationally from the status register —
   never re-register them in a pin adapter.** A top-level/port adapter that
   samples the internal `done`/`status` reg into ANOTHER flop before driving the
   pin delays every completion by a cycle for the poller, inflating the measured
   op window. The pin is `assign done_pin = status_q[DONE_BIT];`, not a second
   registered copy. (A prior design paid +5 re-registering status through the
   adapter.)

These are not micro-optimizations to defer — they are the difference between the
declared cyc/op and a delivered design that misses it by a handful of cycles per
op. Record the accept/commit timing in the uArch spec (Section 3.3) so it is a
testable contract, not an accident of decomposition.

## Per-block DV MUST include a back-to-back case

The block's testbench MUST cover a back-to-back scenario, because the single-op
test passes even when both bugs above are present:

- Issue a new START within a few cycles of the prior DONE, with a **DIFFERENT
  config** than the first operation.
- Assert the second operation uses the **new config** (not the latched/stale
  first config) — check an output that depends on the changed config value.
- Assert DONE for the second operation is **not visible early** — the consumer
  must not observe a completion before the block has taken busy for the second
  START (i.e. the first operation's DONE must not leak into the second).
- Include the degenerate timing: START asserted the same cycle DONE deasserts.

Record in the uArch spec (Section 3.3, Control Logic) exactly when START is
accepted, when it is cleared (the observed-acceptance condition), and when DONE
becomes valid and is cleared — as an explicit, testable timing contract.
