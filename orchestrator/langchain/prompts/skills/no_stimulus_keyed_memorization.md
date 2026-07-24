# Skill: No stimulus-keyed memorization — the RTL must COMPUTE, not REPLAY

## The anti-pattern (an automatic failure)

A data-transforming block (encoder, transform, quantizer, predictor,
rate-distortion / mode-decision core, filter, codec stage) is generated as a
**lookup table keyed on metadata or coordinates** that returns a precomputed
output payload and **ignores the actual input data samples**. Example of the
forbidden shape:

```verilog
function [N-1:0] golden_selected_payload_fn;
    input [5:0] mb_cols; input [4:0] mb_rows; input [5:0] qp;
    input [4:0] mb_y;    input [5:0] mb_x;
    input [2115:0] mb_payload;            // <-- the PIXELS, but UNUSED below
    reg [27:0] key;
    begin
        key = {mb_cols, mb_rows, qp, mb_y, mb_x};
        case (key)
            28'h042b000: golden_selected_payload_fn = 3718'b0000...; // memorized
            28'h294d800: golden_selected_payload_fn = 3718'b1011...; // memorized
            // ...309 precomputed constants...
            default:     golden_selected_payload_fn = {ZERO_COEFFS, ...}; // junk
        endcase
    end
endfunction
```

This is a memorized replay of the per-block testbench's golden vectors. It
**passes per-block DV** because the per-block testbench drives a small,
deterministic set of stimuli, and the model memorized exactly those (key →
output) pairs. It then **fails in chip integration** the instant the same
block sees different pixel content at the same (coordinate, qp) — the key
collides with a memorized entry (wrong answer) or misses the table entirely
and hits the `default` (degenerate / zero output). Downstream this looks like
"3 bytes then early TLAST" or "no output flows".

## Why it is forbidden

The block did not implement the algorithm. It overfit its own test. The
benchmark is the actual transform; a `case` over coordinates is cheating the
gate, not building the IP. The composition gate's chip model is honest math,
so the RTL will never match it.

## The discipline (MANDATORY)

1. **Every output of a data-transforming block MUST be a combinational/
   sequential FUNCTION of its DATA inputs (the pixel/sample payload), not
   only of metadata (coordinates, qp, frame geometry, mb index).** If you
   write a `case`/lookup that does not read the sample bits in the selected
   branch, you have memorized — delete it and implement the datapath.
2. **A `case` keyed on a runtime parameter is legal ONLY for genuine algorithm
   constant tables** (e.g. the quant scaling matrix `V[qp%6]`, a entropy coding VLC
   code table indexed by `(total_coeff, trailing_ones, nC)`, a zig-zag scan
   order). Those tables are inputs to arithmetic that still consumes the data
   samples. A `case` whose RHS is the *final block output* and whose key omits
   the data is memorization.
3. **Red flags that you are about to memorize** — stop and implement instead:
   - the output payload width appears as a literal `'b....` constant inside a
     `case`;
   - the `case` key is built only from {coordinates, qp, dimensions, index};
   - the number of `case` entries equals (or tracks) the number of
     per-block test vectors;
   - the `default` branch returns zeros / a trivial passthrough of metadata.
4. **Implement the real datapath**: read the golden model's math
   (prediction → residual → forward transform → quantize → scan/serialize →
   rate-distortion compare across candidate modes → emit the chosen mode's
   coefficients). Bit-exactness comes from reproducing that arithmetic
   (see arithmetic_precision), NOT from caching its results.
5. **Self-check before Write**: for each data-transforming output, confirm in
   an inline comment which input *sample* bits flow into it. If the honest
   answer is "none, it's looked up by coordinate", the block is wrong.

## How the gate catches it (so you cannot rely on per-block DV passing)

Per-block DV and the block-golden generator drive the SAME small stimulus the
model memorized — so they pass. The composition / integration_dv drives novel
data through the real chain and exposes the cheat as no-output / early-TLAST.
A block that genuinely computes passes both; a memorized block passes only the
first. Build the computing block.
