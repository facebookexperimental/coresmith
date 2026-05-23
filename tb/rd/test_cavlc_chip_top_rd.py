import json
import os
import sys
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge


ROOT = Path(os.environ.get("CORESMITH_PROJECT_ROOT", "/home/ubuntu/coresmith"))
DEFAULT_GIF = (
    Path("/home/ubuntu/dashboards/dashboard/coresmith-llm-bench")
    / "multiframe-codec-handwritten"
    / "mort_original.gif"
)
CLOCK_PERIOD_NS = 20
WIDTH = 640
HEIGHT = 360
PIXELS_PER_FRAME = WIDTH * HEIGHT
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "1"))
MAX_DRAIN_CYCLES = int(os.environ.get("MAX_DRAIN_CYCLES", "2000000"))
QP_SEL = int(os.environ.get("QP_SEL", "0")) & 0x3
QP_VALUE = {0: 24, 1: 36, 2: 48}[QP_SEL]
RESULT_JSON = Path(os.environ.get("RESULT_JSON", str(ROOT / ".coresmith" / "rd_v2" / "qp24.json")))
OUTPUT_BYTES = Path(os.environ.get("OUTPUT_BYTES", str(ROOT / ".coresmith" / "rd_v2" / "qp24_frame0.bin")))
COMPARE_GIF = os.environ.get("COMPARE_GIF")


def load_frames():
    import numpy as np
    from PIL import Image

    gif = Path(os.environ.get("MORT_GIF", str(DEFAULT_GIF)))
    im = Image.open(gif)
    frames = []
    idx = 0
    while True:
        frame = im.convert("L")
        if frame.size != (WIDTH, HEIGHT):
            frame = frame.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
        frames.append(np.asarray(frame, dtype=np.uint8))
        idx += 1
        if MAX_FRAMES and idx >= MAX_FRAMES:
            break
        try:
            im.seek(idx)
        except EOFError:
            break
    return frames


def decode_frames(byte_frames, frames):
    import numpy as np

    sys.path.insert(0, str(ROOT / "examples" / "multiframe_codec_v2"))
    import codec_golden as golden  # type: ignore

    decoded = []
    psnrs = []
    for payload, orig in zip(byte_frames, frames):
        recon = golden.decode_image_v2(bytes(payload), HEIGHT, WIDTH, qp=QP_VALUE, do_deblock=True, entropy="cavlc")
        decoded.append(recon)
        psnrs.append(golden.psnr(orig, recon))
    total_bytes = sum(len(payload) for payload in byte_frames)
    return {
        "qp": QP_VALUE,
        "qp_sel": QP_SEL,
        "frames": len(frames),
        "width": WIDTH,
        "height": HEIGHT,
        "bytes": total_bytes,
        "bpp": 8.0 * total_bytes / (len(frames) * WIDTH * HEIGHT),
        "psnr_db": float(sum(psnrs) / len(psnrs)),
        "per_frame_psnr_db": [float(x) for x in psnrs],
    }, decoded


def maybe_write_compare_gif(frames, decoded):
    if not COMPARE_GIF:
        return
    import numpy as np
    from PIL import Image

    images = []
    spacer = np.full((HEIGHT, 8), 255, dtype=np.uint8)
    for orig, recon in zip(frames, decoded):
        images.append(Image.fromarray(np.concatenate([orig, spacer, recon], axis=1), mode="L"))
    out = Path(COMPARE_GIF)
    out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(out, save_all=True, append_images=images[1:], duration=100, loop=0)


async def reset_dut(dut):
    dut.s_axis_pixel_tdata.value = 0
    dut.s_axis_pixel_tvalid.value = 0
    dut.s_axis_pixel_tlast.value = 0
    dut.s_axis_pixel_tuser.value = 0
    dut.s_axis_status_tdata.value = 0
    dut.s_axis_status_tvalid.value = 0
    dut.s_axis_status_tlast.value = 0
    dut.m_axis_byte_tready.value = 1
    dut.scan_en.value = 1
    dut.debug_mode.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 8)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 4)


async def send_frame(dut, frame):
    flat = frame.reshape(-1)
    accepted = 0
    for idx, px in enumerate(flat):
        dut.s_axis_pixel_tdata.value = int(px)
        dut.s_axis_pixel_tvalid.value = 1
        dut.s_axis_pixel_tlast.value = 1 if idx == PIXELS_PER_FRAME - 1 else 0
        dut.s_axis_pixel_tuser.value = ((QP_SEL & 0x3) << 1) | (1 if idx == 0 else 0)
        while True:
            await RisingEdge(dut.clk)
            if int(dut.s_axis_pixel_tready.value):
                accepted += 1
                break
    dut.s_axis_pixel_tvalid.value = 0
    dut.s_axis_pixel_tlast.value = 0
    dut.s_axis_pixel_tuser.value = 0
    return accepted


async def collect_frame_bytes(dut):
    data = []
    for _ in range(MAX_DRAIN_CYCLES):
        await RisingEdge(dut.clk)
        if int(dut.m_axis_byte_tvalid.value) and int(dut.m_axis_byte_tready.value):
            data.append(int(dut.m_axis_byte_tdata.value) & 0xFF)
            if int(dut.m_axis_byte_tlast.value):
                return data
    raise AssertionError(f"timed out waiting for output tlast after {len(data)} bytes")


@cocotb.test()
async def test_mort_rd_point(dut):
    cocotb.start_soon(Clock(dut.clk, CLOCK_PERIOD_NS, units="ns").start())
    await reset_dut(dut)

    frames = load_frames()
    byte_frames = []
    for frame_idx, frame in enumerate(frames):
        rx_task = cocotb.start_soon(collect_frame_bytes(dut))
        accepted = await send_frame(dut, frame)
        payload = await rx_task
        assert accepted == PIXELS_PER_FRAME
        assert payload, f"frame {frame_idx} produced no bytes"
        byte_frames.append(payload)

    assert int(dut.error_flag.value) == 0
    assert int(dut.ingest_error_flag.value) == 0
    assert int(dut.cavlc_protocol_assert.value) == 0
    assert int(dut.cavlc_code_overflow_assert.value) == 0
    assert int(dut.packer_overflow_assert.value) == 0
    assert int(dut.fifo_overflow_assert.value) == 0

    result, decoded = decode_frames(byte_frames, frames)
    RESULT_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    OUTPUT_BYTES.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_BYTES.write_bytes(bytes(byte_frames[0]))
    maybe_write_compare_gif(frames, decoded)
