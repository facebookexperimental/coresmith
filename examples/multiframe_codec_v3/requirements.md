# H.264-Inspired Grayscale Encoder — Soft IP for Sky130

**Goal.** Design a synthesizable Verilog-2005 soft IP that encodes
640×360 8-bit grayscale frames into a packed byte stream using an
H.264-inspired intra-only coding pipeline.

**Golden model.** `examples/multiframe_codec_v2/codec_golden.py`. The
generated RTL must be bit-exact against this Python reference on the
10-frame Mort GIF validation crop.

**Targets.**
- 30 fps at 50.0 MHz on `sky130_fd_sc_hd` (nominal corner).
- Output: AXI-Stream 8-bit packed bytes.
- Input: AXI-Stream 8-bit pixels, raster order.

**KPI gates (validation DV must enforce).**
- Packed-byte stream bit-exact against `codec_golden.encode_image_v2()`.
- PSNR ≥ preflight per QP (QP24 ≈ 49 dB, QP36 ≈ 38.5 dB, QP48 ≈ 34 dB).
- Compressed size within ±1% of golden bpp.

**Decomposition.** Up to the architect. Reference the golden model's
functional structure (intra prediction, transform, quantize, mode
decision, entropy coding, byte packing, deblock) but choose the block
boundaries and interfaces that best satisfy the throughput and KPI
targets.
