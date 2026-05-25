# Skill: Arithmetic Precision in RTL

When a block performs arithmetic — DCT/FFT/IDCT, matrix multiplies, FIR/IIR
filters, accumulators, statistics, divides, fixed-point math, anything that
isn't a pure pass-through — its register widths and number formats are part
of the design contract, not an implementation detail. **Getting this wrong
silently corrupts the algorithm.** The chip will still simulate, lint-pass,
and emit bytes; the bytes will just be wrong.

This skill encodes the rules you must follow when authoring a uArch spec,
an interface contract, or RTL that does math.

## Why this matters

The most common silent bug in LLM-generated RTL is "the algorithm is right,
the bit widths are wrong":

- A 4×4 forward DCT on 8-bit residuals saturates a 12-bit signed coefficient
  register, so the quantizer always outputs ±2047 and the decoder reconstructs
  flat-grey blocks. PSNR drops 25 dB.
- A 16-bit accumulator overflows after 256 8-bit samples; an FIR filter's
  output becomes random.
- A Q1.15 multiplied by a Q1.15 stored back into a Q1.15 register loses the
  sign extension or the high-order bits depending on which slice the LLM
  picked — same algorithm, two different bit-exact behaviors.
- A "fix divide by 16 (QP=36 step)" gets implemented as `>> 4` on a signed
  value with a buggy rounding rule, off-by-one for every negative coefficient.

None of these fail lint. None of these fail isolated block-level cocotb
tests with toy stimulus. They show up only when a real workload exercises
the full range — at validation_dv time, often classified as a non-arithmetic
failure ("structural drift", "reset-seed ambiguity"), wasting debug
iterations.

## When this skill applies

Use this skill whenever the block you are designing computes ANY of:

- Sums, products, dot products, accumulators
- Transforms (DCT/IDCT, FFT/IFFT, wavelet, integer transforms)
- Quantization, scaling, normalization, rounding
- Saturation, clamping, sign-magnitude conversions
- Predictors, subtractors, deltas, residuals
- Filters (FIR, IIR, linear-phase, adaptive)
- Logarithm / exponential / sqrt / reciprocal approximations
- Anything that consumes a number and emits a different number

Skip it for pure handshake / framing / packet routing blocks that just shuffle
bits.

## Rule 1: Derive each register's width from range analysis

Do NOT pick widths by eye. For every arithmetic stage write down:

1. Input range (min, max). Be explicit: signed or unsigned, in what units.
2. The exact operation (sum of N terms, product, transform with known
   expansion factor, etc.).
3. Output range (min, max) computed from (1) + (2).
4. Required width = `ceil(log2(max(|min|, max+1))) + 1` for signed,
   `ceil(log2(max+1))` for unsigned.
5. Round UP to the next standard width (8/9/10/12/16/18/20/24/32) for ease
   of synthesis and debugging. Never round down.

Document this calculation IN the uArch spec, in a block headed
`Bit-width derivation` or similar. The integration review and
`cross_spec_contract_adherence` audit can then check it.

### Concrete examples

**8-bit unsigned pixel minus 8-bit unsigned prediction:**
- input range: [0, 255] each
- diff range: [-255, +255] → 9 bits signed → declare 9-bit signed
- if instead you wrote 8-bit, every negative residual wraps and the
  encode is garbage

**4×4 integer forward DCT (H.264-style) on 9-bit signed residuals:**
- core formula multiplies by ±1, ±2 then sums 4 terms then transposes
  and does it again
- worst-case absolute output ≈ 4 × 2 × 4 × 2 × 255 ≈ 16320 → 15 bits
  signed
- **declare 16-bit signed for the DCT coefficient output, NOT 12-bit**
- a follow-on quantizer that intends to clip can saturate to 12-bit
  AFTER the full coefficient is computed — never store the intermediate
  in 12-bit

**Accumulator of N B-bit signed samples:**
- worst case = N × max(|sample|) → `ceil(log2(N)) + B` bits signed
- include the bit-growth in the spec's "Datapath / Storage" section

**Quantization step Q applied to coefficient C:**
- if Q is a small power of two (8, 16, 32), use `>> log2(Q)` with explicit
  rounding (`(C + Q/2) >> log2(Q)` for round-to-nearest signed) — DO NOT
  use `/` operator; it synthesizes badly and rarely rounds the way the
  golden does
- if Q is arbitrary, use a multiplier with a reciprocal LUT
- declare and document the rounding mode (truncate / floor / round-nearest
  / round-half-to-even). Match the golden bit-for-bit.

## Rule 2: Pick a number format and stick to it

For each port and internal node, declare ONE of:

- **Plain integer**: signed/unsigned N-bit, no implicit scaling. Use for
  pixel data, counters, addresses, configuration values.
- **Fixed-point Q-format**: `Qm.n` means m integer bits + n fractional bits
  (signed unless prefixed `UQ`). e.g. `Q1.15` = 1 integer bit + 15 fractional
  bits, signed, range [-1, 1). Use for filter coefficients, magnitude
  responses, anything where the same multiplier table will be reused at
  multiple QPs.
- **Floating-point**: only if the algorithm genuinely needs huge dynamic
  range. Costs ~6× the area of fixed-point. The default is "don't use it"
  for DSP/video/audio blocks.

If you mix formats on an edge, **the contract must spell out both formats
and the conversion**. Don't write "tdata[31:0] is the result" — write
"tdata[31:0] is the result, Q8.24 signed, two's-complement,
conversion-from-input Q1.15 by signed left-shift of 9 bits".

## Rule 3: Saturation vs wraparound is a policy decision

Default Verilog `+ - *` operators **wrap modulo 2^N**. That is almost
never the behavior you want for a signal-processing block, and almost
always the behavior you want for a counter or address pointer.

For every arithmetic register, document one of:

- **Wraparound**: explicitly mark "expected to wrap" in the spec; e.g.
  a circular FIFO pointer.
- **Saturation**: explicitly mark "saturates to ±MAX" in the spec; the
  RTL must include the saturation logic (`min(max_pos, max(min_neg, x))`).
- **Range-guaranteed**: the upstream block's contract guarantees the
  value cannot exceed the register's range — make the upstream contract
  carry an explicit "max magnitude" field and have the audit check that
  the downstream width covers it.

**Never** leave saturation to "the value won't exceed the width because
the inputs are small." That assumption is exactly what breaks at
validation_dv time when a real workload is run.

## Rule 4: Match the golden bit-for-bit, then optimize

When a block has a Python golden reference, the RTL must produce
bit-exact output for at least the canonical test vectors. Workflow:

1. Read the golden function the spec references. Note every cast,
   every shift, every clip, every rounding mode (`math.floor`,
   `int()`, `np.round` with banker's rounding, etc.).
2. Write the spec's "Algorithm mapping" section to call out each
   of those steps EXPLICITLY with its required bit width and rounding
   mode.
3. Have the per-block testbench drive vectors that exercise both
   sign extremes AND values near the saturation/overflow points
   you derived in Rule 1 — not just zero and small positives.

If the golden uses `int()` (Python's truncate-toward-zero) and your
RTL uses arithmetic right shift (which floors negative values), every
negative coefficient will be off by 1 — silent, never lint-fails,
breaks the bitstream.

## Rule 5: Surface arithmetic limits in the interface contract

When you populate `interface_contracts.json` (the canonical edge contract
produced by the Interface Definition stage), the per-field metadata MUST
include:

```json
{
  "name": "coefficient",
  "msb": 19,
  "lsb": 0,
  "width": 20,
  "signed": true,
  "encoding": "binary",
  "number_format": "integer",          // or "Q4.12", "Q1.15", "float32", ...
  "max_magnitude": 524287,             // 2^19 - 1
  "saturation_policy": "wraparound",   // or "saturating" or "range_guaranteed"
  "rationale": "DCT output before quantizer; downstream is quant.scan"
}
```

The downstream block's spec then has a contract-level reason to size its
input register correctly. The `cross_spec_contract_adherence` audit can
read these fields and reject specs where the consumer's input width is
narrower than the producer's `max_magnitude` needs.

## Quick checklist — paste this into your spec's review section

- [ ] Every arithmetic stage has a documented input range + output range.
- [ ] Every register's width derives from Rule 1 (round UP, not down).
- [ ] Every Q-format edge spells out integer bits, fractional bits, signedness, and conversion to the next stage's format.
- [ ] Every truncation / shift / round names its mode explicitly and matches the golden.
- [ ] Every saturation point is declared as `saturating`, `wraparound`, or `range_guaranteed`.
- [ ] Test vectors exercise sign extremes and near-saturation values.
- [ ] If a golden reference exists, the spec's "Algorithm mapping" cites
      each golden operation with its bit-precision counterpart.

## What to do when you find a width that's too narrow

1. Widen the register at the source of the saturation, not at the consumer.
2. Update the interface contract's `data_width_bits` and field list to match.
3. Cascade the width through any downstream registers that consume the
   value — but stop the widening at any operation that genuinely needs
   to quantize (rounding for a quantizer, mantissa truncation for a
   floating-point intermediate, etc.).
4. Add a regression test case that exercises the value that would have
   saturated.

## Why this isn't a one-time fix

LLM-generated RTL re-picks bit widths every time a per-block spec is
re-rolled. Capturing the rules in this skill means future runs of
`uarch_spec_generator` and `interface_definition` agents see these
rules in their system prompt and apply them consistently across designs
— not just for the one block we hand-patched.
