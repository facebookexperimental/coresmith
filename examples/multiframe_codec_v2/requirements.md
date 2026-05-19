# H.264-Inspired Soft-IP Codec V2 Requirements

Build a Sky130 Verilog-2005 soft IP block for a grayscale H.264-inspired
I-frame codec, grounded in the Python golden model at:

`examples/multiframe_codec_v2/codec_golden.py`

The architecture must start from this golden model and decompose it into RTL
blocks suitable for SocMate generation. This is not required to be a compliant
H.264 bitstream, but it should preserve the H.264-inspired coding structure.

Required codec features:

- I-frame coding path.
- Intra prediction directions: DC, vertical, and horizontal.
- 8x8 macroblocks.
- Fixed validation geometry for the Mort GIF crop: 640x360 grayscale frames.
  With 8x8 macroblocks this is exactly 80 macroblock columns (`mb_x=0..79`)
  and 45 macroblock rows (`mb_y=0..44`), for 3600 macroblocks per frame in
  raster order. This coordinate contract is a hard invariant; do not transpose
  it to 45 columns by 80 rows. Metadata widths must represent `mb_x=79`.
- Per-macroblock selection between one 8x8 transform block and four 4x4
  transform subblocks.
- Mode decision using a simple rate-distortion proxy consistent with
  `encode_image_v2()`.
- 4x4 and 8x8 DCT-II transform support.
- Scalar QP quantization/dequantization.
- 4x4 and 8x8 zig-zag scan.
- H.264-style CAVLC coefficient coding for TotalCoeff, TrailingOnes,
  TotalZeros, and RunBefore. Keep Exp-Golomb only as an explicitly selectable
  software fallback, not as the hardware entropy target.
- Byte-oriented bitstream packing.
- Simplified H.264-style deblocking filter on every 4-pixel boundary,
  including internal 4x4 boundaries inside selected 8x8 macroblocks.

The block diagram should keep the design tractable for SocMate by using these
blocks unless the architecture stage finds a clearly better decomposition:

- `intra_predict`
- `transform_select`
- `quantize_select`
- `mode_decision`
- `zigzag_rle_select`
- `cavlc_enc`
- `block_packer`
- `deblock_filter`

For each generated block, include `python_source:
examples/multiframe_codec_v2/codec_golden.py`, a Verilog target under
`rtl/multiframe_codec_v2/`, and a cocotb testbench target under `tb/cocotb/`.

The software preflight on the 10-frame Mort GIF crop at 640x360 showed the
CAVLC golden model can achieve:

- QP24: 2.8674 bpp, 49.16 dB, 11330/36000 macroblocks selected as 8x8
- QP36: 1.0233 bpp, 38.76 dB, 13143/36000 macroblocks selected as 8x8
- QP48: 0.2161 bpp, 34.04 dB, 25568/36000 macroblocks selected as 8x8

Optimize for a clean, stream-oriented soft IP rather than MMIO. Do not require
memory map, register spec, or clock-tree elaboration unless explicitly enabled.

Frame-control lifecycle invariant:

- The design must have an explicit frame-start/input-active lifecycle event.
  A control block must not infer a new frame solely from an idle output register,
  ready/valid availability, or the ability to emit a status token.
- `codec_busy` must be tied to real occupancy: accepted input/frame activity,
  live pipeline state, non-empty/draining output FIFO state, valid status inputs,
  and unaccepted held output transactions. After a frame-end flush is accepted and
  the output FIFO reports empty/drained with frame TLAST seen, `codec_busy` must
  be able to deassert and remain low until the next explicit frame-start event.
- Output FIFO `drained` is a terminal frame-completion event, not a synonym for
  reset-idle emptiness. After reset, an empty FIFO must report occupancy zero
  with `drained=0` and no terminal status TLAST/TUSER completion mirror until a
  frame-final TLAST output transaction has been accepted downstream.
- Block, smoke, integration, and validation DV must include a VCD/FST waveform
  check for this lifecycle: frame start, frame end flush, FIFO drain, idle
  deassertion, and no unintended immediate restart.

Geometry/ordering invariant:

- Pixel input is row-major over each 640x360 frame. The macroblock emitter must
  buffer one 640x8 stripe and emit all 80 macroblocks from that stripe before
  accepting/emitting macroblocks from the next stripe. The final macroblock of
  each frame is `(mb_x=79, mb_y=44)`, and the final frame TLAST/terminal status
  may only align with that macroblock.
- Validation DV must check at least the first stripe, a row transition, and the
  final macroblock against the golden model's `for by in range(0, H, 8)` then
  `for bx in range(0, W, 8)` traversal.
