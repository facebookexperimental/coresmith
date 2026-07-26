# Skill: buffer stride & packed-record slot contract (writer index == reader index)

When a block stores 2-D data (a line buffer, stripe buffer, frame/tile
buffer, pixel_block window) in a 1-D memory and addresses it with
`row * STRIDE + col`, the **stride used to WRITE must equal the stride used
to READ**. A mismatch silently returns never-written cells (zeros / stale
data) for every row but the first, corrupting the data handed downstream —
and it looks exactly like an arithmetic bug.

## The failure this prevents

A proven case: a `stripe_mb_emitter` filled its line buffer **contiguously**
by an incrementing counter (effective row stride = the runtime frame width,
e.g. 16), but `_pack_macroblock` read it back as a 2-D array with a
**hardcoded** `MAX_WIDTH` row pitch (640): `stripe[ly*MAX_WIDTH + base_x + lx]`.
Only row 0 aligned; rows 1–7 indexed never-written cells = 0. Every
pixel_block delivered to the encoder had a correct row 0 and zeroed rows
1–7, so the DCT input was garbage → wrong coefficients and wrong RD mode
selection from the very first pixel_block → wrong byte 0 of the bitstream.
The transform/quant/entropy coding math was **bit-exact** — it was simply fed bad
pixels. The bug masquerades as "the encoder math is wrong."

## Rules for any 2-D buffer in a 1-D memory

1. **One stride constant, shared by writer and reader.** Define the row
   pitch once and use the SAME symbol on both the write path and every read
   path. Never let the producer use one pitch and the consumer another.

2. **Use the RUNTIME dimension, not a compile-time maximum, unless you pad.**
   If the buffer is sized for a worst case (`MAX_WIDTH`) but holds a smaller
   runtime frame (`width`), you must either:
   - address everything by the **runtime `width`** (`row*width + col`) on
     BOTH sides — the contiguous-fill case — or
   - actually **pad each row to `MAX_WIDTH`** when writing, so the
     `row*MAX_WIDTH + col` reads land on written cells.
   Mixing the two (contiguous fill at `width`, strided read at `MAX_WIDTH`)
   is the bug. Pick one and apply it to writer and reader identically.

3. **Make the stride part of the interface/uArch contract.** Record the
   buffer's logical shape (`rows × cols`), the storage row pitch, and whether
   it is runtime-sized or max-padded, so a downstream/cooperating block reads
   it with the matching stride. Treat it like a width or packing contract.

4. **The math is the last suspect, not the first.** If a datapath block is
   byte-exact when driven directly with known coefficients but produces
   wrong results in-system, check the PIXEL/INPUT addressing before touching
   the transform/quant/scan code. Tap the actual buffer contents the block
   receives (row 0 correct, rows 1+ zero is the signature of this bug).

## Packed-record slot / base-offset contract (same principle, 1-D)

The stride contract generalizes from `row*STRIDE+col` to ANY shared packed
record where the producer reserves fixed slots and the consumer reads them
back by index: **the base offset the producer writes at must equal the base
offset every consumer branch reads at.** A reserved slot is just a base
offset of 1 instead of 0; the same writer-index == reader-index rule holds.

The classic trap is a packed record whose layout has a **reserved leading
slot for one variant** (e.g. a residual-coefficient word that reserves slot
0 for the luma DC block, so the N data blocks live at slots `1..N`). When the
SAME record is shared by two encode paths that differ only in whether the
reserved slot is populated, the consumer must apply the **identical base
offset on BOTH paths** — the data blocks are at slots `1..N` regardless of
whether slot 0 happens to be used in this variant.

### The failure this prevents (proven)

An video_codec `intra_rd_encode_core` packed the per-pixel_block coefficient word
with a **uniform slot convention for both MB types**: slot 0 = luma DC block,
slots `1..16` = the sixteen 4×4 residual blocks (`scans[(sb+1)*16 + i]`,
`totals[sb+1]`) — for the Intra16x16 path AND the Intra4x4 path. The consumer
`entropy_bitstream_engine` read the Intra16x16 residuals correctly with
`scan_idx = sb+1`, but in the **Intra4x4** branch dropped the `+1` and read
`_get_coeff(word, sb, ...)` / `_get_total_coeff(word, sb)`. Result: it
encoded an extra all-zero block from the empty slot 0, shifted every real
block down by one, and **never read the last block (slot 16)** — an
under-produced bitstream that first diverges exactly at the first residual
sym_token. The entropy coding math (sym_token tables, level/run/total_zeros
coding, bit packing) was **byte-exact**; only the slot index was wrong. The
bug masquerades as "the entropy coding entropy math is wrong / it terminates early."

### Rules

5. **One base offset per packed slot, shared by writer and EVERY reader
   branch.** If the producer reserves slot 0 (or any leading slots) and
   stores the N payload records at `base + k`, every consumer access path —
   including each mode/variant branch — must use the SAME `base + k`. Do not
   let one branch (e.g. Intra16x16) use `sb+1` while a sibling branch (e.g.
   Intra4x4) uses `sb`. Define the base offset once as a named constant and
   reuse it on all branches and for ALL parallel fields (coeffs AND their
   per-block totals/counts).

6. **A reserved/DC slot is part of the record contract for ALL variants.**
   If the packing reserves slot 0 for a block that only one MB-type emits,
   the slot still exists (empty) in the other MB-type's record — the data
   blocks do NOT slide down to fill it. Read them at the contracted offset,
   not at 0.

7. **Cross-check the loop bound against the reserved slots.** With a reserved
   slot, reading `k in range(N)` at base `+1` covers slots `1..N`; reading at
   base `0` reads slot 0 and silently DROPS slot N (the under-production
   signature). If the output is shorter than the reference and the first
   divergence is the first record, suspect a dropped base offset before
   touching the per-record math.

8. **MANDATORY: the per-record read EXPRESSION must be byte-identical across
   every variant branch.** When you write two (or more) branches that each
   loop over the same packed records (e.g. an `if mb_type == 0: ... else: ...`
   where both branches iterate `for sb in range(16)` and read residual blocks),
   the slot-index expression MUST be the SAME literal expression in every
   branch. If the record reserves slot 0, the data-block index is `sb + 1` in
   BOTH branches — never `sb + 1` in one and `sb` in the other. The most common
   way this bug ships: the author copies the first branch's loop, then "tidies"
   the copy by deleting the `+ 1` because that branch has no DC block to skip —
   but the producer reserved slot 0 for ALL variants, so the `+ 1` is REQUIRED
   in every branch. Apply the identical offset to the coefficient read AND to
   the per-block total/count read in that same loop.

   ```python
   # ANTI-PATTERN (ships a shifted, under-produced stream):
   if mb_type == 0:                 # Intra16x16
       for sb in range(16):
           scan = [_get_coeff(word, sb + 1, i) for i in range(16)]   # base +1
           totals_update(_get_total_coeff(word, sb + 1))
   else:                            # Intra4x4
       for sb in range(16):
           scan = [_get_coeff(word, sb, i) for i in range(16)]       # BUG: dropped +1
           totals_update(_get_total_coeff(word, sb))                 # BUG: dropped +1

   # CORRECT (one slot expression, both branches, both fields):
   RESIDUAL_BASE = 1                # producer reserves slot 0 for the DC block
   for sb in range(16):             # same in every mb_type branch
       slot = RESIDUAL_BASE + sb
       scan = [_get_coeff(word, slot, i) for i in range(16)]
       totals_update(_get_total_coeff(word, slot))
   ```

   SELF-CHECK before finishing the block: if two branches both read the same
   packed records, diff their slot-index expressions character-by-character —
   any difference (a missing `+ 1`, a different base) is this bug.

## When this does not apply

Purely 1-D streams (byte streams, sample vectors with no row/column
structure, and no reserved/indexed slot layout) have no stride or slot offset
to mismatch. The 2-D rules apply whenever a 1-D memory is indexed as a 2-D
array with a `row * STRIDE + col` expression on more than one access path; the
packed-record rules apply whenever a shared packed word/record reserves fixed
slots read by index on more than one access path.
