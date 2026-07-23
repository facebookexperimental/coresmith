# Skill: Throughput Budget Contract — compute cycles/op before you serialize

A datapath that closes timing and fits the flop budget can still be **several
times too slow** if a fixed-size iteration is scheduled serially on one reusable
resource when the budget had room to pipeline it. Fmax only asks "does one
stage's arithmetic fit the clock?"; it never asks "how many cycles per unit of
work?". This skill is the missing check: **compute cycles/op and compare it to
the throughput requirement BEFORE you accept a serial schedule.**

## The rule

When a datapath contains a **fixed-N iteration on a single reusable resource** —
an N-tap accumulate on one multiplier, an N-round transform on one datapath, an
N-element scan on one comparator, an N-cycle bit loop — do this, in order:

1. **Compute the cycles/op.** For a modulo-scheduled loop:
   `cyc/op = iterations × II + (pipeline_depth − 1) + drain + io_framing`,
   where the initiation interval `II = max(RecMII, ResMII)`:
   - `RecMII = max over loop-carried recurrences of ceil(latency ÷ distance)`
   - `ResMII = max over shared resources of ceil(uses_per_iter ÷ instances)`
2. **Compare to the requirement.** The FRD `PERF-NNN` throughput cap (or, if the
   customer imposed none, the self-imposed roofline-peak × derate budget). A
   serial schedule with one resource has `ResMII = N` — i.e. `cyc/op ≈ N` — which
   is exactly the multiples-too-slow case.
3. **If it misses AND flop/area headroom exists, widen — don't serialize.** Add
   parallel resource instances (K lanes) so `ResMII = ceil(N ÷ K)`, or unroll the
   loop by K with **registered** pipeline stages so successive items overlap at
   `II = 1`. This is a **pipelined** widening (registered boundaries), NOT a
   combinational unroll — a flat unroll trades the throughput miss for an
   unsynthesizable cloud (see the Pipeline Contract skill).

Only accept the serial schedule after step 1 shows it still MEETS the
requirement, or step 3 shows no headroom to widen (record that trade-off
explicitly in the spec).

## Choosing the lever

The **binding constraint** tells you which lever to pull:
- **ResMII binds** (a shared resource is the bottleneck) → add instances / lanes.
  Costs area proportional to K; check it against the flop/area budget.
- **RecMII binds** (a loop-carried recurrence is the bottleneck) → adding lanes
  does NOT help; you must shorten or algebraically break the recurrence (e.g.
  split a running accumulator into K partial sums reduced once at the tail).

## What to write in the spec

- The declared **II** and the **computed cycles/op** (show the arithmetic).
- The **binding constraint** (which recurrence or resource) and the lever.
- A cross-reference to the FRD `PERF-NNN` throughput requirement and an explicit
  MEETS / MISSES verdict. If it would miss, change the microarchitecture before
  emitting the spec — do not declare a number you already know misses the cap.
- The machine-readable `perf` block (see uArch-spec Section 6.1) so the
  throughput roofline checker can price your schedule against the PDK op delays.

"Fits the flop budget" is necessary, not sufficient. A block must ALSO hit its
throughput budget; a serial loop that fails it while lanes were affordable is a
defect, not a valid area/throughput trade.
