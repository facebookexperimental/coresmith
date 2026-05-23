#!/usr/bin/env python3
"""Run current generated chip_top through a C++ Verilator RD harness."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GIF = Path(
    "/home/ubuntu/dashboards/dashboard/coresmith-llm-bench/"
    "multiframe-codec-handwritten/mort_original.gif"
)
WIDTH = 640
HEIGHT = 360
FRAMES = int(os.environ.get("CORESMITH_RD_FRAMES", "10"))


def make_raw(gif: Path, out_raw: Path) -> Path:
    out_raw.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(gif),
            "-vf",
            f"scale={WIDTH}:{HEIGHT}:flags=lanczos,format=gray",
            "-frames:v",
            str(FRAMES),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            str(out_raw),
        ],
        check=True,
    )
    expected = WIDTH * HEIGHT * FRAMES
    data = np.fromfile(out_raw, dtype=np.uint8)
    if data.size < expected:
        if data.size == 0:
            raise RuntimeError(f"ffmpeg produced no pixels from {gif}")
        frame_count = data.size // (WIDTH * HEIGHT)
        data = data[: frame_count * WIDTH * HEIGHT]
        data = np.concatenate([data, np.tile(data[-WIDTH * HEIGHT :], FRAMES - frame_count)])
        data.tofile(out_raw)
    return out_raw


def build(out_dir: Path) -> Path:
    build_dir = out_dir / "cpp_build"
    build_jobs = int(os.environ.get("CORESMITH_RD_BUILD_JOBS", str(os.cpu_count() or 1)))
    sources = [
        ROOT / "rtl" / "integration" / "chip_top.v",
        ROOT / "rtl" / "integration" / "multiframe_codec_v2_top.v",
        *sorted((ROOT / "rtl" / "multiframe_codec_v2").glob("*.v")),
    ]
    cmd = [
        "verilator",
        "-cc",
        "--exe",
        "--build",
        "--trace",
        "--build-jobs",
        str(max(1, build_jobs)),
        "-Mdir",
        str(build_dir),
        "--top-module",
        "chip_top",
        "-Wno-DECLFILENAME",
        "-Wno-WIDTHEXPAND",
        "-Wno-WIDTHTRUNC",
        "-Wno-UNUSEDSIGNAL",
        "-Wno-UNUSEDPARAM",
        "-Wno-BLKSEQ",
        "-Wno-LATCH",
        *[str(p) for p in sources],
        str(ROOT / "tb" / "rd" / "current_chip_top_rd_main.cpp"),
    ]
    with (out_dir / "build.log").open("w", encoding="utf-8") as log:
        subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    exe = build_dir / "Vchip_top"
    if not exe.exists():
        raise RuntimeError(f"missing Verilator executable {exe}")
    return exe


def qp_sel(qp: int) -> int:
    return {24: 0, 36: 1, 48: 2}[qp]


def load_golden():
    import importlib.util

    path = ROOT / "examples" / "multiframe_codec_v2" / "codec_golden.py"
    spec = importlib.util.spec_from_file_location("codec_golden_v2", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def point(exe: Path, raw: Path, out_dir: Path, qp: int, golden) -> dict:
    log_path = out_dir / f"current_verilator_qp{qp}_cpp.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            [str(exe), str(raw), str(FRAMES), str(qp_sel(qp)), str(out_dir)],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60 * 60,
        )
    base = json.loads((out_dir / f"current_verilator_qp{qp}_cpp.json").read_text())
    base["returncode"] = proc.returncode
    base["log"] = str(log_path)
    if proc.returncode != 0:
        base["error"] = f"harness failed with return code {proc.returncode}"
        return base

    raw_frames = np.fromfile(raw, dtype=np.uint8).reshape(FRAMES, HEIGHT, WIDTH)
    decoded = []
    decode_ok = 0
    for frame in range(FRAMES):
        data_path = out_dir / f"current_verilator_qp{qp}_frame{frame}.bin"
        payload = data_path.read_bytes()
        try:
            dec = golden.decode_image_v2(payload, HEIGHT, WIDTH, qp=qp, do_deblock=True, entropy="cavlc")
            decoded.append(dec)
            decode_ok += 1
        except Exception as exc:
            base.setdefault("decode_errors", []).append({"frame": frame, "error": str(exc)})

    if decode_ok == FRAMES:
        original = raw_frames.reshape(FRAMES * HEIGHT, WIDTH)
        recon = np.stack(decoded).reshape(FRAMES * HEIGHT, WIDTH)
        psnr = float(golden.psnr(original, recon))
    else:
        psnr = float("nan")

    base["bpp"] = 8.0 * base["bytes"] / (FRAMES * WIDTH * HEIGHT)
    base["psnr_db"] = psnr
    base["decode_ok_frames"] = decode_ok
    return base


def main() -> int:
    gif = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_GIF
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / ".coresmith" / "rd_current_cpp"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = make_raw(gif, out_dir / "mort_10f_640x360.raw")
    exe = build(out_dir)
    golden = load_golden()
    qps = [int(v) for v in os.environ.get("CORESMITH_RD_QPS", "24,36,48").split(",") if v]
    jobs = int(os.environ.get("CORESMITH_RD_JOBS", str(os.cpu_count() or 1)))
    points = []
    with ThreadPoolExecutor(max_workers=max(1, min(jobs, len(qps)))) as pool:
        futures = {pool.submit(point, exe, raw, out_dir, qp, golden): qp for qp in qps}
        for fut in as_completed(futures):
            p = fut.result()
            points.append(p)
            qp = p["qp"]
            psnr = p.get("psnr_db", float("nan"))
            print(
                f"QP{qp} rc={p.get('returncode')} bpp={p.get('bpp')} "
                f"psnr={psnr if not math.isnan(psnr) else 'nan'} "
                f"bytes={p.get('bytes')} decode={p.get('decode_ok_frames', 0)}/{FRAMES}",
                flush=True,
            )
    points.sort(key=lambda p: p["qp"])
    for p in points:
        psnr = p.get("psnr_db", float("nan"))
        print(
            f"SUMMARY QP{p['qp']} rc={p.get('returncode')} bpp={p.get('bpp')} "
            f"psnr={psnr if not math.isnan(psnr) else 'nan'} "
            f"bytes={p.get('bytes')} decode={p.get('decode_ok_frames', 0)}/{FRAMES}",
            flush=True,
        )
    summary = {"gif": str(gif), "raw": str(raw), "points": points}
    (out_dir / "current_verilator_rd_cpp_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return 1 if any(p.get("returncode") != 0 for p in points) else 0


if __name__ == "__main__":
    raise SystemExit(main())
