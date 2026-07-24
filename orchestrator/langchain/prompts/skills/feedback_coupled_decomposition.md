# Skill: Decomposing feedback-coupled / trial-decision datapaths

Most datapaths decompose cleanly into a **streaming pipeline** — each block
transforms its input and hands off to the next (`A -> B -> C`), and the chip is
the composition of independent stages. Pick that decomposition by default.

But some computations have a **cross-stage dependency that a streaming split
cannot express**. The classic shape is a **trial / rate-distortion (RD) decision
with reconstruction feedback**:

> For each unit (macroblock, tile, frame region, packet, sub-band…), the design
> tries SEVERAL candidate encodings, runs each candidate **all the way through**
> (transform → quantize → de-quantize → inverse-transform → reconstruct →
> measure cost/bits/distortion), then **selects** the best candidate, and the
> **selected candidate's reconstruction feeds the NEXT unit's prediction**.

Recognize this pattern when the reference/golden has any of:
- a per-unit loop that builds 2+ candidates and compares a cost
  (e.g. `cand8` vs `cand4`, `cost = distortion + lambda*bits`, "choose smaller"),
- a reconstruction step (`dequant`/`idct`/`recon`) computed **before** the mode
  is chosen (so the cost can be measured), and/or used as the **neighbor
  prediction** for the next unit (an intra-/recon-feedback loop),
- a selection (`if cand_a.cost <= cand_b.cost`) whose result determines BOTH the
  emitted bits AND the state carried to the next unit.

## The rule

**Do NOT split a trial/RD decision across separate streaming blocks.** A
decomposition like `predict | transform | quantize | mode_decision` as four
independent stream stages CANNOT reproduce the reference, because:
- the mode decision needs the *complete* result of *every* candidate (full
  transform+quant+recon+bitcount of each), so a forward-only stream that has
  already passed data to `transform` cannot go back and choose;
- the per-unit reconstruction must close a feedback loop into the *same* tier
  before the next unit starts, which a pipeline of distinct blocks serializes
  incorrectly (deadlocks or uses stale/zero neighbors).

The composed model will emit near-zero / garbage output and fail the byte-exact
composition gate, no matter how each individual block is re-specced — the gap is
**structural**, not per-block math.

**Instead, CONSOLIDATE the trial-decision into ONE block.** That block owns the
whole per-unit encode: it builds every candidate end-to-end (transform, quant,
dequant, inverse-transform, reconstruct, count bits), compares the cost, selects
the winner, emits the selected syntax, and updates its **own** reconstruction /
neighbor store for the next unit. Surrounding blocks stay simple:
- upstream block: assemble/deliver each unit's input samples (+ geometry/QP
  sideband) in order;
- the consolidated encode block: per-unit trial + select + recon-feedback;
- downstream blocks: entropy-code / pack the already-selected syntax, framing.

In the block diagram, prefer a single `*_encode` / `*_mode_select` block (owning
prediction + transform + quant + recon + RD select for the unit) over a chain of
`predict`/`transform`/`quantize`/`mode` blocks. Keep the recon-feedback edge
**internal** to that block, not an inter-block stream edge. Helper/candidate math
may live in sub-functions, but the decision and the feedback state must be in one
block so the composition is correct.

This applies to any "evaluate N candidates fully, then pick" structure — video/
image codecs (transform-size / intra-mode RD), trellis/Viterbi-style selection,
adaptive quantizer search, etc. When the datapath is a plain feed-forward
transform with no trial/feedback, ignore this and use the normal streaming split.
