# Skill: Pipeline Contract — bound the arithmetic in every stage

A correct datapath that does **too much arithmetic in one clock cycle** is
*functionally* right but **unsynthesizable** (the synthesizer can't close a huge
combinational cloud — yosys times out) and never meets timing. This is a real,
recurring failure: an intra rate-distortion mode-search was emitted as one giant
single-cycle combinational cloud and walled the whole chip.

The discipline: **decide the pipeline depth and what arithmetic happens in each
stage, up front, against the PDK timing budget** — never leave it implicit.

## The rule

1. **No single-cycle exhaustive search / accumulation / mode-decision.** Any
   "evaluate N candidates and pick the best in a cycle", "sum over N terms in a
   cycle", or "unroll the whole transform/search combinationally" is forbidden.
   It is the canonical combinational cloud.
2. **Bound each stage by the period.** Using the *PDK Timing Budget* section
   (the characterized per-op delays + how many chain per stage at this clock),
   partition the datapath so the **chained combinational delay in every stage ≤
   the clock period**. Put a register between stages.
3. **Pick pipeline vs parallel deliberately:**
   - **Pipeline / sequentialize** — one reusable datapath + a small FSM that
     iterates over ~N cycles. Low area, N-cycle latency. Prefer this when the
     `flip_flop_budget` / `area_budget` is tight (search/decision blocks).
   - **Parallelize + reduce** — N copies feeding a compare/add tree, ~1–2
     stages, ~N× area. Prefer only when latency matters and area allows.
   - This choice is not free: **compute cycles/op for the serial form and check
     it against the throughput requirement** (see the *throughput budget
     contract* skill) before committing to sequentialize. A serial loop on one
     resource is `~N` cycles/op; if that misses `PERF-NNN` and flop/area headroom
     exists, widen to K pipelined lanes instead.
4. **A single op that alone exceeds the period must be DECOMPOSED**, not just
   registered (e.g. a too-wide multiply → split into partial products across
   cycles). The budget flags these.

## What to write in the spec (Section 6, Timing)

State explicitly:
- **Pipeline depth** (cycles of latency) and the **initiation interval** (new
  input every cycle? every k cycles?).
- **Per-stage contents**: the arithmetic in each stage and its estimated chained
  delay (from the budget), showing each stage ≤ the period.
- For a search/RD block: the FSM that sequences candidate evaluation over cycles
  **or** the parallel-evaluate-then-reduce structure — never a flat cloud.

The downstream **synth gate** rejects an un-synthesizable combinational cloud,
so an unpipelined search will fail the block — design the stages now.

## Run at the declared II — do not drain item-by-item

A pipeline whose spec says `II = 1` (a new item every cycle) must actually
**sustain** one item per cycle. The recurring bug is a pipeline that fills, then
**drains one item completely before admitting the next** — its effective II is
the whole pipeline depth, so throughput collapses to `depth × N` cycles for `N`
items. Keep the upstream stages at `II = 1`; the only reason to stall is real
backpressure or a genuine resource conflict, never "wait for this item to exit
before starting the next."

**Cross-item recurrences** (a running best/min/max, a running accumulator, any
state carried from item to item) are what tempt an item-by-item drain — do NOT
put them in the streaming stages. Instead:
- Keep the datapath stages `II = 1` and **stage-delay** the item's
  first/last/index tags alongside the data.
- Put the cross-item update in **ONE commit stage at the tail** that consumes the
  stage-delayed tags and updates the running state exactly once per item, in
  order. The recurrence then has distance 1 at the commit stage only; the
  upstream stages never see it and never stall.

This keeps `RecMII` local to the single commit stage and lets the pipeline stream
at its declared II instead of serializing on the accumulator.

## Hoist loop-invariant fetches out of nested loops

A value that does not change across an inner loop must be fetched/computed ONCE
per outer iteration, not re-read every inner step. Re-fetching a loop-invariant
operand inside the inner loop multiplies memory-port pressure and cycles by the
inner trip count for no new data. Instead:

- Load the invariant working set into a **batch-sized local cache** (a small
  register file / line buffer) at the top of the outer iteration, and read it
  from the cache in the inner loop.
- **Declare the cache lifetime in the spec**: what it holds, when it is (re)loaded
  (which outer-loop boundary), and its size — so it is counted in the flop budget
  and its refill cadence is auditable.

This turns an `O(outer × inner)` stream of redundant fetches into `O(outer)`
loads plus cache reads, freeing the shared memory port for real accesses.

## At RTL time (the implementer, not just the spec author)

The uArch spec states the pipeline depth and per-stage arithmetic. The **Verilog
must realize it structurally** — the synth gate checks the *implementation*, not
the spec. A spec that correctly says "28-stage pipeline" still fails if the RTL
collapses it into one combinational block.

- **Each named pipeline stage = a registered boundary.** Put the stage's result
  in a flop (`always @(posedge clk)`) before the next stage reads it. An N-stage
  spec → N register boundaries, **not** one `always @(*)` / one chained `assign`.
- **Never collapse a multi-op search / transform / accumulation into one
  combinational block.** Evaluating N candidates, or summing/processing N terms,
  in a single cycle is the cloud — sequentialize it onto one reusable datapath
  via an FSM (~N cycles) or split it across registered stages.
- **A `function`/`task` is combinational.** A helper that chains many ops
  (predict→transform→quant→reconstruct→cost) is one big combinational cone even
  though it "looks like" a call. Count its op depth against the period; break it
  across stages just like inline logic.
- **Size stages by the *PDK Timing Budget*** (the per-op delays), not by gut
  feel: keep each stage's chained delay ≤ the clock period.
- **Working / FIFO / line storage: addressed memory, NOT a wide flat packed reg
  with a runtime slice.** Do NOT keep buffers as one big `reg [1023:0] buf_q`
  read/written by a DYNAMIC part-select (`buf_q[idx +: 8]`, `buf_q[sel]`). A
  dynamic part-select into a wide flat reg lowers to a giant barrel-shifter /
  decoder tree — synthesis (`proc`/`opt`) cannot elaborate it tractably and
  **times out even when the FSM control is perfectly staged** (this is a real
  failure mode that walled a fully-registered codec block). Store such data as
  a `cs_sram`/`cs_fpmem` addressed memory, or a proper per-element array
  `reg [W-1:0] mem [0:D-1]` indexed by a *registered* address with a registered
  1-cycle read. Constant slices (`cfg[57:42]`) are fine — only a **runtime
  index** into a **wide** reg is the trap.

## Module-per-stage protocol — when the block has a declared stage map (MANDATORY)

When the µArch spec / stage map declares a **multi-stage datapath** (a named
stage map is present AND the declared latency is more than ~10 cycles, or the
block iterates a search/decision over N candidates), you MUST realize it as
**N+1 modules, not one module with "distributed" arithmetic** — because "put a
register between stages, all in one always block" is gameable (four generations
of RD-encoder RTL collapsed the search into one always block and passed a
prose contract). The module boundary is a *structurally checkable* register
boundary. Do exactly this:

1. **One submodule per named stage**, each with **REGISTERED outputs**
   (`always @(posedge clk)`) and a `valid`/`ready` (or `enable`) handshake per
   the spec. The module boundary IS the pipeline register boundary. Put the
   stage submodules in the **same `.v` file** as the top block module — the
   toolflow de-dups multiple modules per file; you do not need separate files.
2. **One controller FSM** that **iterates the modes/candidates over cycles** on
   the shared stage datapath (mode 0..N across N cycles, one reusable datapath),
   driving the stage submodules via the handshake chain. The controller is
   **~arithmetic-free** — indexing, counters, and handshakes only, no datapath
   math.
3. The **top block module instantiates the stage submodules + the controller**
   and wires the handshake chain. It contains no datapath arithmetic itself.

**Explicitly forbidden** (a deterministic census rejects these before yosys):
- A **cross-stage combinational path**: stage K's inputs computed from stage
  K-1's *combinational* outputs with no register between (that is one merged
  stage, not two).
- **Stage bodies written as `task`/`function`s inlined into one always block.**
  A task/function is a *combinational cone*; calling it over a `for` loop
  **multiplies** its arithmetic into that one block (a task called N times = N
  copies of its multipliers), it does **not** pipeline it. The census counts
  `body_ops × call_count × enclosing_loop_trips` — this is exactly how a
  9-mode × 19-cut search became ~10⁴ multipliers in one cycle.
- **Decorative FSM wait / cycle-burn states** wrapped around a body that still
  does all the arithmetic in one state. Registering the *output* is not enough;
  the *arithmetic* must be split across the registered stage submodules.

The deterministic gate (`CORESMITH_STAGE_LINT`, default on) fails any single
`always` block whose effective multipliers (after loop-unroll + task-inline)
exceed `CORESMITH_STAGE_LINT_MUL_CAP` (~64), and — when a stage map is present —
any block whose total effective ops exceed the stage map's op budget by
`CORESMITH_STAGE_LINT_FACTOR` (8×). When the module-per-stage protocol applies
(`CORESMITH_STAGE_MODULES`, default on), it also expects the top module to
instantiate at least one registered submodule per declared stage.

## Exactly ONE implementation per module — never split-brain the RTL

Do NOT "solve" an unsynthesizable datapath by keeping the real algorithm behind
one macro and a lightweight stand-in behind another. Writing **two
implementations of the same module** selected by a global
`` `ifdef ``/`` `ifndef ``/`` `elsif `` (e.g. the verified datapath under
`` `ifndef SYNTHESIS `` and a latency-shell / counter-theater **mock** under
`` `else ``) is FORBIDDEN and **automatically rejected** by a deterministic gate
at generation time. DV (Verilator, no `-DSYNTHESIS`) then verifies one branch
while every synth/backend gate builds the OTHER — they are **different
hardware**, so a green DV proves nothing about the chip. This was the deepest
anti-gaming failure the campaign found: a real 1.9 Mbit-recon encoder under
`` `ifndef SYNTHESIS `` shadowing a non-functional mock (`coeff = input − 128`,
QP unused, memory reads write-only) that existed only to pass the storage lint
and the synth probe.

The discipline instead: make the ONE implementation genuinely synthesizable —
stage the arithmetic (above) and put oversized storage in a `cs_sram`/`cs_fpmem`
wrapper. Conditional compilation may ONLY guard **non-functional** debug/trace/
assertion code (`$display`, `$dumpfile`/`$dumpvars` waveform hooks, SVA
`assert`) — never `always` / continuous `assign` / a driving `initial` / a
module instantiation. The one legitimate sim-body/synth-macro split lives ONLY
inside the provided `cs_*` wrapper library; you just *instantiate* the wrapper.
