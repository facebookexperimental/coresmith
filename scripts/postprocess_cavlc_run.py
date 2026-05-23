#!/usr/bin/env python3
"""Postprocess the CAVLC CoreSmith run for dashboard publication.

This hook is intentionally conservative.  It only marks the run done when the
integrated RTL appears to be the v2 CAVLC top and Verilator collateral exists.
Until then it publishes a software-golden CAVLC R-D preview and records the
remaining blocker for the monitor loop.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DASH_ROOT = Path("/home/ubuntu/dashboards/dashboard/coresmith-llm-bench")
OUT_DIR = DASH_ROOT / "codex-cavlc-multiframe-codec"
SOURCE_GIF = DASH_ROOT / "multiframe-codec-handwritten" / "mort_original.gif"
QPS = [18, 24, 30, 36, 42, 48]
X264_INTRA = [
    (0.1486, 29.712235, "crf42"),
    (0.2185, 32.347475, "crf38"),
    (0.3095, 34.644287, "crf34"),
    (0.4482, 37.410196, "crf30"),
    (0.6362, 40.224391, "crf26"),
    (0.8849, 42.835611, "crf22"),
    (1.2497, 46.002863, "crf18"),
    (1.7242, 49.158111, "crf14"),
]
VT_INTRA = [
    (0.1541, 29.928971, "q10"),
    (0.2447, 33.090240, "q25"),
    (0.3929, 36.121994, "q40"),
    (0.6964, 40.488923, "q55"),
    (1.2648, 45.348185, "q70"),
    (2.3383, 51.760370, "q85"),
]


def import_golden():
    sys.path.insert(0, str(ROOT / "examples" / "multiframe_codec_v2"))
    import codec_golden as golden  # type: ignore

    return golden


def load_frames(gif_path: Path, width: int = 128, height: int = 72, max_frames: int = 10):
    import numpy as np
    from PIL import Image

    frames = []
    gif = Image.open(gif_path)
    idx = 0
    while True:
        frame = gif.convert("L").resize((width, height), Image.Resampling.LANCZOS)
        frames.append(np.asarray(frame, dtype=np.uint8))
        idx += 1
        if max_frames and idx >= max_frames:
            break
        try:
            gif.seek(idx)
        except EOFError:
            break
    return frames


def sweep_golden() -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    golden = import_golden()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = load_frames(SOURCE_GIF, max_frames=int(__import__("os").environ.get("CORESMITH_RD_MAX_FRAMES", "10")))
    height, width = frames[0].shape
    points = []

    for qp in QPS:
        recon_frames = []
        total_bytes = 0
        total_psnr = 0.0
        selected_8x8 = 0
        total_mbs = 0
        for frame in frames:
            encoded, meta, recon = golden.encode_image_v2(frame, qp=qp, do_deblock=True, entropy="cavlc")
            decoded = golden.decode_image_v2(encoded, height, width, qp=qp, do_deblock=True, entropy="cavlc")
            if not np.array_equal(recon, decoded):
                raise RuntimeError(f"golden encode/decode mismatch at qp={qp}")
            total_bytes += len(encoded)
            total_psnr += golden.psnr(frame, decoded)
            selected_8x8 += sum(1 for item in meta if item["block_size"] == 8)
            total_mbs += len(meta)
            recon_frames.append(decoded)

        compare = []
        for orig, recon in zip(frames, recon_frames):
            spacer = np.full((height, 4), 255, dtype=np.uint8)
            compare.append(Image.fromarray(np.concatenate([orig, spacer, recon], axis=1), mode="L"))
        gif_out = OUT_DIR / f"mort_compare_cavlc_qp{qp}.gif"
        compare[0].save(gif_out, save_all=True, append_images=compare[1:], duration=100, loop=0)

        points.append({
            "qp": qp,
            "bytes": total_bytes,
            "bpp": 8.0 * total_bytes / (len(frames) * height * width),
            "psnr_db": total_psnr / len(frames),
            "frames": len(frames),
            "width": width,
            "height": height,
            "selected_8x8": selected_8x8,
            "macroblocks": total_mbs,
            "gif": gif_out.name,
        })

    data = {
        "generated_cavlc_software_golden": points,
        "x264_intra": [{"bpp": bpp, "psnr_db": psnr, "label": label} for bpp, psnr, label in X264_INTRA],
        "videotoolbox_intra": [{"bpp": bpp, "psnr_db": psnr, "label": label} for bpp, psnr, label in VT_INTRA],
        "source": str(SOURCE_GIF),
        "note": "Software golden CAVLC preview; Verilator R-D is still pending until v2 integrated top/harness is ready.",
    }
    (OUT_DIR / "rd_curve_cavlc_preview.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8.5, 5.4), dpi=120)
    ax.plot([p["bpp"] for p in points], [p["psnr_db"] for p in points], "o-", label="Codex CAVLC v2 software golden")
    ax.plot([p[0] for p in X264_INTRA], [p[1] for p in X264_INTRA], "s--", label="x264 intra baseline/CAVLC")
    ax.plot([p[0] for p in VT_INTRA], [p[1] for p in VT_INTRA], "d:", label="VideoToolbox intra baseline/CAVLC")
    ax.set_xlabel("bits per pixel (bpp)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(f"CAVLC v2 R-D preview on Mort GIF ({len(frames)} frames, {width}x{height})")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="lower right", framealpha=0.95)
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "rd_curve_cavlc_preview.png", dpi=140, facecolor="white", bbox_inches="tight")
    return data


def integration_status() -> tuple[bool, str]:
    top = ROOT / "rtl" / "integration" / "chip_top.v"
    if not top.exists():
        return False, "rtl/integration/chip_top.v does not exist yet"
    text = top.read_text(encoding="utf-8", errors="ignore")
    if "cavlc_enc" not in text:
        return False, "integrated chip_top is not the v2 CAVLC top yet"
    if "axis_frame_ingest" not in text and "multiframe_codec_v2" not in text:
        return False, "chip_top mentions CAVLC but not the v2 front-end blocks"
    return True, "v2 CAVLC chip_top detected"


def copy_collateral() -> None:
    coll = OUT_DIR / "collateral"
    coll.mkdir(parents=True, exist_ok=True)
    for rel in [
        "examples/multiframe_codec_v2/requirements.md",
        "examples/multiframe_codec_v2/blocks.yaml",
        "examples/multiframe_codec_v2/codec_golden.py",
        ".coresmith/pipeline_events.jsonl",
        ".coresmith/run-20260513-top-codex-gpt55-cavlc.log",
    ]:
        src = ROOT / rel
        if src.exists():
            dst = coll / rel.replace("/", "__")
            shutil.copy2(src, dst)
    for folder in ["arch/uarch_specs", "rtl/multiframe_codec_v2", "tb/multiframe_codec_v2"]:
        src = ROOT / folder
        if src.exists():
            dst = coll / folder.replace("/", "__")
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)


def write_dashboard(data: dict, ready: bool, status: str) -> None:
    best = max(data["generated_cavlc_software_golden"], key=lambda p: p["psnr_db"])
    rows = "\n".join(
        f"<tr><td>{p['qp']}</td><td class='num'>{p['bytes']:,}</td><td class='num'>{p['bpp']:.3f}</td>"
        f"<td class='num'>{p['psnr_db']:.2f} dB</td><td class='num'>{p['selected_8x8']}/{p['macroblocks']}</td></tr>"
        for p in data["generated_cavlc_software_golden"]
    )
    gifs = "\n".join(
        f"<figure><img src='{p['gif']}' alt='Mort original and CAVLC reconstruction QP {p['qp']}'><figcaption>QP {p['qp']} · {p['psnr_db']:.2f} dB · {p['bpp']:.3f} bpp</figcaption></figure>"
        for p in data["generated_cavlc_software_golden"]
    )
    verdict = "Verilator-ready" if ready else "Verilator pending"
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Codex GPT-5.5 CAVLC multiframe codec</title>
<style>
body{{margin:0;background:#101113;color:#f4f4f5;font-family:Inter,Arial,sans-serif;line-height:1.5}}
.wrap{{max-width:1180px;margin:0 auto;padding:32px 22px 60px}} a{{color:#8db4ff}} code{{background:#191b20;border:1px solid #2b2f38;border-radius:4px;padding:1px 5px}}
.top{{display:flex;justify-content:space-between;gap:16px;border-bottom:1px solid #2b2f38;padding-bottom:18px;margin-bottom:28px}}
h1{{font-size:30px;margin:0 0 10px}} h2{{font-size:17px;margin:0 0 12px}} p{{color:#b7bcc7;max-width:88ch}}
.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0 28px}} .stat,.card{{background:#17191e;border:1px solid #2b2f38;border-radius:8px;padding:16px}}
.label{{font-size:11px;color:#828997;text-transform:uppercase;letter-spacing:.08em}} .value{{font-family:monospace;font-size:23px;margin-top:6px}} .ok{{color:#69c38f}} .warn{{color:#d6a957}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}} figure{{margin:0}} img{{width:100%;display:block;border:1px solid #2b2f38;border-radius:6px;background:#000}} figcaption{{font-family:monospace;font-size:12px;color:#b7bcc7;margin-top:6px}}
table{{width:100%;border-collapse:collapse}} th,td{{padding:10px 12px;border-bottom:1px solid #2b2f38;text-align:left}} th{{font-size:11px;color:#828997;text-transform:uppercase;letter-spacing:.06em}} .num{{font-family:monospace;text-align:right}}
.chart{{background:white}} @media(max-width:850px){{.stats,.grid{{grid-template-columns:1fr 1fr}}}} @media(max-width:560px){{.stats,.grid{{grid-template-columns:1fr}}}}
</style></head><body><main class="wrap">
<div class="top"><strong>coresmith-llm-bench</strong><a href="../">back to overview</a></div>
<h1>Codex GPT-5.5 · CAVLC multiframe codec v2</h1>
<p>This run starts from the top-level architecture prompt and uses GPT-5.5 through the Codex provider in bypass-permissions mode. The target is the improved H.264-inspired soft IP: all-intra frames, selectable 8x8 or 4x4 macroblocks, DC/vertical/horizontal prediction, CAVLC coefficient coding, and a deblocking filter.</p>
<div class="stats">
<div class="stat"><div class="label">Run status</div><div class="value {'ok' if ready else 'warn'}">{verdict}</div><div>{status}</div></div>
<div class="stat"><div class="label">Model</div><div class="value">GPT-5.5</div><div>Codex CLI provider</div></div>
<div class="stat"><div class="label">Preview best</div><div class="value">{best['psnr_db']:.2f} dB</div><div>QP {best['qp']} · software golden</div></div>
<div class="stat"><div class="label">Frames</div><div class="value">{best['frames']}</div><div>{best['width']}x{best['height']} Mort grayscale</div></div>
</div>
<section class="card"><h2>Mort GIF comparison</h2><div class="grid">{gifs}</div><p>Each GIF is original | reconstructed. This preview uses the v2 software golden CAVLC model until the Verilator R-D harness is adapted to the final generated v2 integration top.</p></section>
<section class="card" style="margin-top:20px"><h2>Rate-distortion preview</h2><img class="chart" src="rd_curve_cavlc_preview.png" alt="CAVLC v2 R-D preview"><p>Raw data: <a href="rd_curve_cavlc_preview.json"><code>rd_curve_cavlc_preview.json</code></a>. x264 and VideoToolbox points are the intra-only baseline/CAVLC MacBook measurements supplied for comparison.</p></section>
<section class="card" style="margin-top:20px"><h2>CAVLC software golden points</h2><table><thead><tr><th>QP</th><th class="num">Bytes</th><th class="num">bpp</th><th class="num">PSNR</th><th class="num">8x8 MBs</th></tr></thead><tbody>{rows}</tbody></table></section>
<section class="card" style="margin-top:20px"><h2>Collateral</h2><p>Prompt, CAVLC golden model, block YAML, pipeline logs, microarchitecture specs, testbenches, and generated RTL snapshots are under <code>collateral/</code>.</p></section>
</main></body></html>
"""
    (OUT_DIR / "index.html").write_text(html, encoding="utf-8")


def update_parent_index() -> None:
    index = DASH_ROOT / "index.html"
    if not index.exists():
        return
    text = index.read_text(encoding="utf-8")
    href = './codex-cavlc-multiframe-codec/'
    if href in text:
        return
    needle = '<a class="tab-btn" href="./codex-multiframe-codec/" style="text-decoration:none;display:inline-flex;align-items:center">Codex&nbsp;GPT-5.5&nbsp;·&nbsp;codec ↗</a>'
    insert = needle + '\n  <a class="tab-btn" href="./codex-cavlc-multiframe-codec/" style="text-decoration:none;display:inline-flex;align-items:center">Codex&nbsp;GPT-5.5&nbsp;·&nbsp;CAVLC&nbsp;codec ↗</a>'
    if needle in text:
        index.write_text(text.replace(needle, insert), encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ready, status = integration_status()
    data = sweep_golden()
    copy_collateral()
    write_dashboard(data, ready, status)
    update_parent_index()
    (ROOT / ".coresmith" / "postprocess_cavlc.status.json").write_text(
        json.dumps({"verilator_ready": ready, "status": status, "dashboard": str(OUT_DIR)}, indent=2),
        encoding="utf-8",
    )
    if ready:
        # The monitor keeps running until a real v2 Verilator R-D hook replaces
        # this preview marker.  Avoid a false success on software-only data.
        (ROOT / ".coresmith" / "postprocess_cavlc.partial").write_text(status + "\n", encoding="utf-8")
        return 2
    (ROOT / ".coresmith" / "postprocess_cavlc.partial").write_text(status + "\n", encoding="utf-8")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
