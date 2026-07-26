# Skill: Bitstream serialization contracts between blocks

When two blocks exchange a **packed bitstream** (entropy coder → packer, any
variable-length-code producer → a byte/word assembler, a header/syntax emitter →
a stream), the exact way bits are concatenated, ordered, and flushed is a
**shared CONTRACT** — not a per-block implementation detail. Each block can be
internally correct against its own golden and the composition can STILL produce
the wrong bytes, because the blocks disagree on the convention. The per-block DV
cannot catch this (every block passes its own golden); only the full-chip
byte-exact gate catches it, and it cannot localize it. So the convention must be
FROZEN as an interface contract up front.

## The failure this prevents

A real case: a entropy coding coder emitted correct per-symbol codewords and a packer
emitted correct bytes in isolation, but composed they produced a stable WRONG
stream — because the packer **byte-padded each chunk** instead of running ONE
contiguous bit accumulator, and **structural header/syntax bits** (frame type,
block-size flag, mode) were carried in a sideband (`tuser`) and **never
serialized into the stream by any block**, and the final byte waited on a flush
handshake that never arrived. Three independent re-specs reproduced the
byte-identical wrong output: it is structural, invariant to per-block re-spec.

A second real case (proven by isolating the blocks in simulation): the packer
serialized perfectly, but the downstream **elastic byte FIFO duplicated bytes**.
Its registered output mux read `mem[rd_ptr]` and gated `valid` on `count` using
the **pre-pop** read pointer and occupancy — in the *same* clocked process that
advanced `rd_ptr`/`count` on the pop. Because `<= next` does not take effect
until the next edge, the just-popped byte was re-presented for one extra cycle
and re-sampled. Under bursty upstream (occupancy bouncing 0→1→0, the normal
cadence when a packer emits a byte only every few cycles) **every** byte
duplicated: a 57-byte stream became 70 bytes, `81 a7 28 …` → `81 81 a7 …`. The
per-block FIFO DV (continuous traffic) barely shows it (one head duplicate);
only the bursty full-chip path makes it explode.

## Rules for any packed-bitstream edge

1. **One contiguous accumulator, no mid-stream byte padding.** Variable-length
   codewords concatenate into a single bit accumulator MSB-first (or the
   reference's documented order). Do NOT zero-pad each codeword/chunk up to a
   byte boundary unless the reference does. Padding happens ONCE, at end-of-frame.

2. **Every bit the reference emits must be serialized by SOME block — none left
   in sideband.** Structural/header/syntax bits (frame-type, block-size/mode
   flags, lengths) that the golden interleaves into the bitstream must be
   emitted into the stream by a specific block, in the reference's order,
   relative to the payload. Carrying them only in `tuser`/sideband and never
   writing them to the stream drops them from the output. Name the owning block
   and the position in the contract.
   - **Enumerate EVERY syntax element with an explicit producer + stream
     position + per-unit cadence.** Walk the reference's emit order (e.g. per
     pixel_block: `frame_type`, `size_flag`, `mode` bits, THEN the coeff bits)
     and assign each element to exactly one block, in that order. An element the
     reference emits *per MB* must be emitted *per MB* — not once per frame.
   - **A field read for control flow is NOT the same as a field emitted to the
     stream.** Reading `transform_size`/`pred_mode`/`shape` from a sideband word
     to *select* behavior (zigzag order, table choice) does NOT put those bits
     in the output. If the reference also serializes that field, some block must
     additionally EMIT it. (Proven failure: a chain read `shape` to pick the
     zigzag order and read `pred_mode` for control, but never emitted the
     size_flag+mode bits — 33 header bits/MB silently dropped, byte0 0x8D vs
     golden 0x81, stream 33 bits short, while the entropy coding coeff bits were
     bit-exact. The bug lived entirely in the seam.)
   - **Get the NESTING CADENCE right: per-MB syntax vs per-subblock syntax are
     different levels — emit each at its own level, exactly once.** When the unit
     has sub-units (a pixel_block with N sub-blocks), the reference emits some
     syntax ONCE PER MB (e.g. `frame_type`/`I-frame`, `use8x8`/`size_flag`) and
     other syntax ONCE PER SUB-BLOCK (e.g. each sub-block's `pred_mode`, then its
     coeff bits). Walk the reference's nested loop and tag every element with its
     level. Do NOT hoist a per-sub-block field up to per-MB, and do NOT repeat a
     per-MB prefix inside the per-sub-block loop. PROVEN failure (rerun10): the
     chain emitted the `frame_prefix(1)+shape(1)` per SUB-BLOCK instead of once
     per MB, AND omitted the per-sub-block 2-bit `pred_mode` field entirely
     (it was read for control but never serialized) — so the stream was
     content-wrong from byte0/bit5 (golden `81a7…` vs `869c…`, 56B vs 57B) even
     though every per-block coeff golden passed. CHECK: for a ≥2-sub-block MB,
     assert the composed bit layout is `[per-MB prefix bits][ {pred_mode(k),
     coeff_bits(k)} for k in subblocks ]` exactly, and that the per-MB prefix
     appears once, not N times.
   - **Forbid "handler exists but no producer" dead branches.** If a consumer
     block has a case for a token/event (e.g. `EV_HEADER`) but no upstream block
     is contractually required to produce it, the element is silently dropped.
     Every consumer branch must have a named producer; every producer output a
     named consumer.
   - **The chain/composition golden must compare the FULL concatenated
     bitstream** against the reference — per-block coeff goldens each pass while
     a missing header in the seam goes uncaught.

3. **Bit order + width are part of the contract.** Specify MSB-first vs
   LSB-first, the codeword length field, and how partial bytes carry across beats
   (a held partial-byte accumulator with a bit-count, not a fresh byte per beat).

4. **Flush-once, self-terminating on frame end.** The final partial byte must be
   emitted exactly once when the input frame ends (input `tlast`), and the block
   asserts output `tlast` itself — do NOT depend on an external flush pulse that
   may never arrive (a stuck final byte is a classic deadlock). Define who emits
   the terminal byte + `tlast` and on what trigger.

5. **Make the convention an explicit interface contract / system invariant.** In
   the block diagram + interface contracts, record for each packed-bitstream
   edge: accumulator order (MSB/LSB-first), no-intermediate-padding, the exact
   header/syntax bits + their position, and the flush/`tlast` ownership. Treat it
   like a frozen contract the producer and consumer both cite — same as a width
   or handshake contract.

6. **FIFO / elastic-buffer output discipline (read POST-update state).** Any
   queue, skid buffer, or elastic FIFO on the stream (very common as the final
   `output_*_fifo`) must drive its **registered** output from the **post-update**
   read pointer and occupancy, never the pre-pop values, or it re-presents and
   re-emits the just-popped beat → **duplicated bytes**. Concretely, in the
   clocked process:
   - First compute `next_rd` (advance on `pop && count>0`) and
     `next_count = count + push - (pop && count>0)`.
   - Drive the output from those: `valid <= (next_count > 0)`,
     `data <= mem[next_rd]` (NOT `mem[rd_ptr]`), `last <= mem_last[next_rd]`.
   - **Read-during-write bypass — GENERAL, not just empty→1.** A registered
     memory write (`mem[wr_ptr].next = data`) does NOT read back this cycle; a
     same-cycle read of that address returns the OLD/zero cell. So whenever the
     address you are about to read (`next_rd`) equals the address being written
     THIS cycle, you MUST bypass the memory and output the input beat directly:
     `if push and next_rd == wr_ptr: out_data = s_data; out_last = s_last`.
     **Do NOT gate this bypass on `next_count == 1` (or on the FIFO being
     empty).** Under back-to-back `push && pop` the read pointer chases the
     write pointer and `next_rd == wr_ptr` recurs at ANY occupancy — gating the
     bypass on occupancy==1 lets every beat after the first read an uncommitted
     (zero) cell. PROVEN failure: an `output_byte_fifo` whose bypass was
     `next_count==1 and next_rd==wr_ptr` turned a continuous `[10,20,…,80]`
     stream into `[10,0,0,0,0,0,0,0]` — only the first byte survived.
   - Never read a memory/pointer in the same cycle you advance it and assume the
     advance is visible — it is not until the next edge. This is the classic
     first-word-fall-through (FWFT) / show-ahead pointer bug.
   Prefer a known-correct FWFT FIFO template over hand-rolling the pointer math.
   Per-block FIFO DV must include BOTH cadences: (a) **back-to-back push+pop
   with continuous ready** (a known input sequence in == same sequence out,
   no zeros/dups — this exposes the read-during-write bypass hole), and
   (b) **bursty / single-beat-with-gaps** (occupancy bouncing 0→1→0 — this
   exposes the pre-update-read duplication). A FIFO that only passes one cadence
   is not verified.

7. **Multi-element array/struct layout is ONE shared written contract — never let
   producer and consumer re-derive geometry independently.** When block A writes a
   multi-element payload (a coefficient array, a sub-block grid, a packed record)
   and block B reads it, the index/stride/layout formula MUST be a single explicit
   contract both cite verbatim. The consumer's element index = the SAME base/stride
   the producer used (`producer_block_base + scan_order[i]`), not a re-derivation
   from a different geometric assumption. Each block can pass its own block-golden
   and still compose wrong if they disagree on layout. PROVEN failure: a producer
   packed 4×4 coefficients as stacked 16-element raster sub-blocks
   (`sub*16 + y*4+x`) while the consumer read them as one interleaved 8×8 grid
   (`base_y*8 + base_x + zigzag[i]`); both algorithms were individually correct,
   but the layout mismatch corrupted the very first sub-block's coeff tokens →
   wrong from bit 41. Name the array's element-index formula in the interface
   contract; producer and consumer reference that one formula.

8. **Frame-boundary sidebands (`frame_start`/`sof`, `final_mb`/`tlast`) are
   derived once at the true origin and asserted on the exact last element — and a
   MULTI-element block-golden must exercise them.** Two proven failures: (a) a
   per-frame `frame_start` strobe that read as 1 for EVERY pixel_block got OR'd
   into a per-MB reconstruction reset, wiping intra-prediction neighbors before
   every MB → second-MB mode decision flipped → whole downstream stream desynced
   and truncated; (b) `final_mb`/`tlast` left always-0 so the packer never ran its
   terminal flush and the stream ended on a drain timeout mid-frame. A
   single-element (single-MB) block-golden passes both; only a multi-element
   golden catches them. So: derive frame-boundary strobes at the genuine first/last
   element, never broadcast a per-frame flag into per-element control, and make the
   per-block golden drive ≥2 elements (MBs/frames) so boundary logic is verified.

9. **Strip a packed word's base offset ONCE — never re-apply it in the per-field
   accessor.** When a block slices a sub-word out of a wider packed record (e.g.
   `coeff_word = rec >> 55` to drop a 55-bit trailer) and then has a per-element
   accessor for that sub-word, the accessor's shift must be relative to the word
   it ACTUALLY receives (`lane_shift = (N-1-lane)*W`), NOT the original record
   (`55 + (N-1-lane)*W`). Re-adding the base offset reads every element from the
   wrong bit window and silently corrupts counts/values. PROVEN failure: a coeff
   reader stripped 55 bits when slicing then re-applied `+55` in `_get_lane`, so
   lane reads were `[0,0,0,0]` instead of `[10,11,12,13]` and it emitted 16
   symbols vs golden 13 (~2× stream). CHECK: after slicing, assert one known lane
   of a known input equals its known constant in a unit test.

10. **Each preamble / framing / syntax bit has exactly ONE owner — the serializer
    concatenates tokens, it does not inject syntax.** A control bit (I-frame
    prefix, start code, frame-type) must be emitted by a single block. If both a
    producer (e.g. the entropy coding header token) AND the final packer "helpfully" emit
    the same leading bit, it is double-counted and every byte shifts. PROVEN
    failure: `block_packer` pre-seeded the accumulator with the I-frame "1" while
    `entropy_enc` already emitted that bit in MB0's header → stream began `1`+`1000…`
    = `0xc0` instead of golden `0x81`. CORRECT: the packer/FIFO only concatenates
    upstream tokens (no seeded syntax bits); name each preamble bit's owning block
    in the contract. CHECK: assert the composed byte 0 equals the golden byte 0.

11. **A variable-width field packed in a fixed-width word — producer and consumer
    MUST agree on ALIGNMENT (MSB- vs LSB-justified) and on the valid-bit count.**
    When one block emits a code of `n` valid bits inside a wider fixed field
    (e.g. a entropy coding codeword in a 64-bit `code_bits` lane) and another block reads
    it, both must use the SAME justification. If the producer left-justifies
    (MSB-aligned: code occupies the TOP `n` bits) but the consumer reads the
    bottom `n` bits (LSB-aligned), the consumer reads the zero-padding → ALL
    ZEROS out (correct length, no content). PROVEN failure: `entropy` packed
    `code_bits` MSB-aligned in a 64-bit field while `block_packer` read them
    right-aligned → the composed stream was 57 bytes of all `0x00` (every block
    passed its own golden; only the seam was wrong). CORRECT: the contract states
    the justification AND carries the valid bit-count `n` alongside the field;
    the consumer shifts by that `n` from the agreed edge. CHECK: assert a known
    codeword read back by the consumer equals the producer's known bits (and that
    composed output is NOT all-zero) in a unit test.

12. **MANDATORY — the terminal serializer must reproduce the reference's
    OUTERMOST container framing, not just the inner per-unit payload. Emitting
    only the inner payload is a FORBIDDEN, automatic byte-0 failure.** The block
    that owns the chip's output stream is compared by the gate against the
    OUTERMOST reference entry the gate calls (the function NAME passed as the
    reference entry, e.g. `encode()`/`encode_flat()`) — NOT against the inner
    per-frame/per-MB routine. Before writing the block, locate that outermost
    entry in the reference, read its body, and replicate EVERY byte it emits in
    order: typically a header (unit count + global dims/params) then, per unit,
    a LENGTH-PREFIX followed by the inner payload bytes. The inner routine
    (`encode_frame()` and friends) produces ONLY the payload that goes AFTER the
    length-prefix — its output is a strict SUBSET of what the gate expects.
    Stopping at the inner payload is the single most common way this block fails
    the gate. When the gate's reference entry is a top-level wrapper that frames
    one or more payloads with a header (a count of units + global
    dimensions/params + a per-payload LENGTH-PREFIX), the block that owns the
    output stream must emit that ENTIRE container, in the wrapper's exact order —
    emitting only the inner payload is a silently-truncated stream that diverges
    at BYTE 0. Trace the reference from
    its OUTERMOST entry (the name the gate calls, e.g. `encode()`/`encode_flat()`),
    NOT from the inner per-frame/per-MB routine (`encode_frame()`): the outer
    wrapper's header bits are part of the output the gate compares. A header field
    that is a length/size of payload that has not been produced yet REQUIRES the
    block to BUFFER the full payload first, compute its length, emit the header,
    THEN emit the buffered payload — you cannot stream the header before the
    payload length is known. PROVEN failure (codecv4 entropy_bitstream_engine,
    faithful regen): 3 of 5 regens emitted only `encode_frame(...)["bitstream"]`
    (the 61-byte per-frame payload) and omitted the top-level `encode()` framing
    `ue(nframes) ue(W) ue(H) ue(qp) ue(payload_len)` → composed stream 61B vs
    gate-expected 67B, divergent from byte 0; the entropy coding payload bits themselves
    were byte-exact. The 1 regen that buffered the payload and prefixed the
    container header was byte-exact to the 67-byte oracle.
    - **Walk the outermost entry's emit order and tag each element ONCE.** e.g.
      `[count][W][H][qp]` once globally, then per payload `[len(payload)][payload
      bytes]`. Assign the whole container to the terminal block (or name another
      block that owns the header and prefixes the buffered payload) — do not
      leave the container header "implied".
    - **Length-prefix ⇒ buffer-then-emit.** Any `ue(len(payload))`/size field
      that precedes the data it sizes forces a buffer pass: accumulate all
      payload bytes, take `len()`, write the header, then drain the buffer.
      Streaming the header eagerly with a guessed/zero length is wrong.
    - **Flush the WHOLE container as ONE continuous bit stream, exactly ONCE at
      the very end — match the reference's flush boundary bit-for-bit.** If the
      reference builds the entire container (header bits THEN payload) in a single
      bit accumulator and converts to bytes ONCE at the end (one `getbytes()`),
      your block must too: append the header bits, then append the payload bytes
      AS BITS (`put_bits(byte, 8)` each), into the SAME accumulator, and flush
      once. Do NOT introduce an extra intermediate byte-alignment/flush that the
      reference does not have, and do NOT drop or truncate the final PARTIAL byte
      — pad it (`byte <<= 8 - n`) exactly as the reference's `getbytes()` does, so
      a header whose bit-length is not a multiple of 8 still rounds the container
      UP to the same final byte count. PROVEN failure (same codecv4 block): a
      regen wrote the correct header (`payload_len=61`) and correct payload but
      flushed the container 1 BIT short — 528 bits / 66 bytes vs the golden's 529
      bits / 67 bytes — because its terminal flush dropped the last partial byte
      that a 41-bit (non-byte-aligned) header forces. CHECK: assert the composed
      container's TOTAL BIT COUNT equals `header_bits + 8*len(payload)` and that
      `len(output) == ceil(that / 8)`; if the header is not byte-aligned, the
      last output byte is a partial byte and MUST be present (zero-padded), not
      dropped.
    - **Honesty note for missing metadata.** If the outer header needs true
      dimensions/params the block's interface does NOT carry (e.g. true `W`/`H`
      vs padded pixel_block geometry, `nframes` policy), that is an INTERFACE gap
      to escalate (add the metadata port or assign header ownership to a block
      that has it) — never hardcode the dimensions or compare payload-only to
      dodge the container. Derive each header field from a real input field; if
      none exists, flag it.
    - CHECK: assert the composed output LENGTH and BYTE 0 equal the oracle's; a
      length short by exactly the header size with a byte-exact tail is the
      dropped-container signature.
    - SELF-CHECK before finishing the block: find the reference's outermost entry
      function; if it CALLS the inner per-frame routine and ALSO writes any bits
      of its own (a `put_ue`/`put_bits`/header write before or around that call),
      your block MUST emit those bits too. If your block's emit path contains no
      counterpart to the outer wrapper's header writes, you have dropped the
      container — add them. Re-derive header fields from real input fields (never
      hardcode); if a needed field is absent from the interface, flag it as an
      interface gap rather than omitting the header.

When the inter-block data is NOT a packed/variable-length bitstream (fixed-width
samples, byte-per-beat payloads), rules 1–4 do not apply — use the normal stream
contract. Rules 6 (FIFO discipline), 7 (shared array layout), 8 (frame-boundary
sidebands), 9 (strip a packed base-offset once), 10 (single-owner preamble bits),
11 (packed-field MSB/LSB alignment + valid-bit count) and 12 (outermost container
framing + length-prefix buffering) apply to ANY multi-block stream, packed or not.
