# Skill: Global Output-Contract Ownership (decomposition discipline)

When you split a design into blocks, each block gets a **local** job. But some
of what the reference (golden) does is **global and emergent** — a property of
the *whole output*, not of any single block. That behavior lives **in the seams
between blocks**, so a naive decomposition leaves it **unowned**, and the
assembled chip output then fails to match the golden **even though every block
is individually correct**. This is the #1 way a clean per-block decomposition
still produces a wrong end-to-end result.

> **A responsibility owned by "everyone" is owned by no one.**

The tell, every time: each block passes its own model, but the *composed*
output diverges from the golden — because an emergent output property was never
assigned to anybody.

## The emergent responsibilities to look for (domain-agnostic)

For the golden you are decomposing, enumerate which of these exist, then assign
each to exactly **one** owning block:

1. **Output format / framing / container** — headers, length-prefixes, trailers,
   the exact byte/packet/record layout the golden emits around the payload.
   *(Codec: the entropy coding/unit container `ue(nframes), ue(W), ue(H), ue(qp), ue(len), payload`.)*
2. **Global ordering / re-sequencing** — the golden emits results in a specific
   order; parallel or pipelined blocks finish out of order. Who re-sorts/merges
   into the one canonical stream? *(Graph engine: results in vertex-ID order.)*
3. **Coordinate / layout / endianness conventions** — the exact spatial or byte
   layout of the result. *(3D shader: framebuffer tiling/swizzle, origin
   top-left vs bottom-left, raster order, MSAA sample layout.)*
4. **Termination / convergence / end-of-stream** — "is the *whole* computation
   done?" is a global property no single block knows alone. *(Graph engine:
   all vertices stable = the distributed-termination problem; codec: final
   `tlast` for the whole container.)*
5. **Global reductions / aggregation** — a sum / max / count / hash over *all*
   blocks that no single partition owns. *(Shader: final blend/resolve/depth
   merge; analytics: a global accumulator.)*
6. **Global namespaces / ID assignment** — consistent IDs, addresses, or tags
   that must be unique across blocks.

## The rule

For **every** emergent property the golden has:
- Name the **one** block that owns producing it. Prefer a dedicated **terminal
  assembler** block (serializer / output-assembler / resolve / merge stage)
  that sits at the output edge and emits the final result in the golden's exact
  contract.
- Make sure that owner has the **interface to carry the metadata it needs**. The
  emergent property often needs runtime values that live elsewhere (frame
  count, W/H/QP, total length, a done-signal). Add a dedicated config/metadata
  edge (e.g. a `cfg_*` input) so the owner can actually produce the contract —
  do **not** let it guess or hardcode them.
- If **no existing block can own it, ADD a block** (or widen one block's scope).
  This is a *decomposition* decision and must be made now, while block
  boundaries are still movable — it cannot be fixed later by a per-block author,
  because the responsibility has no home to put it in.

## Anti-pattern that caused a real regression

A terminal serializer was split into `token_writer → byte_packer → output_fifo`.
Each block was correct, but **none owned the outermost container** and none had
an interface carrying `nframes/W/H/QP/payload_len`. The chip emitted the bare
payload while the golden's `decode()` expected the full container → divergence
from byte 0, non-decodable, every block "passing." The fix would have been a
**single terminal serializer block owning the container, with a `cfg_*` metadata
interface** — exactly what a *prior, monolithic* decomposition of the same design
did correctly. **Do not split a terminal serializer/assembler across blocks
unless one block still owns the complete output contract.**

## REQUIRED ARTIFACT — emit this in your block-diagram JSON

Add a top-level array `global_output_contract` to the block diagram. One entry
per emergent property of the golden's output:

```json
"global_output_contract": [
  {
    "property": "outermost bitstream container (header + length-prefixed payload)",
    "owner_block": "entropy_bitstream_engine",
    "carries": "cfg interface delivers nframes,W,H,QP; buffers payload to get total length before emit"
  },
  {
    "property": "end-of-stream tlast for the whole container",
    "owner_block": "entropy_bitstream_engine",
    "carries": "asserts tlast on final container byte"
  }
]
```

Rules for the artifact:
- Every emergent property that exists in the golden MUST appear with a **single,
  real `owner_block`** (a block that exists in your `blocks` list).
- If a needed property has no natural owner among your blocks, **add a terminal
  assembler block** and list it — do not leave the property out.
- `carries` must state how the owner gets any runtime metadata it needs
  (which interface/edge), so the property is actually producible, not assumed.

If the design genuinely has no emergent output structure (a pure
stateless element-wise map with identical input/output framing), emit
`"global_output_contract": []` and say so — but think hard first: framing,
ordering, termination, and layout conventions are emergent far more often than
they look.
