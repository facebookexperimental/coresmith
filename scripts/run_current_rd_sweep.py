#!/usr/bin/env python3
"""Run a Mort GIF RD sweep against the current generated chip_top."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GIF = Path(
    "/home/ubuntu/dashboards/dashboard/coresmith-llm-bench/"
    "multiframe-codec-handwritten/mort_original.gif"
)


def make_frames(gif_path: Path, out_npy: Path, frames: int = 10) -> Path:
    import numpy as np

    raw_path = out_npy.with_suffix(".gray")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(gif_path),
            "-vf",
            "scale=640:360:flags=lanczos,format=gray",
            "-frames:v",
            str(frames),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            str(raw_path),
        ],
        check=True,
    )
    raw = np.fromfile(raw_path, dtype=np.uint8)
    expected = frames * 360 * 640
    if raw.size < expected:
        if raw.size == 0:
            raise RuntimeError(f"ffmpeg produced no frames from {gif_path}")
        frame_count = raw.size // (360 * 640)
        raw = raw[: frame_count * 360 * 640]
        pad = np.repeat(raw[-360 * 640 :], frames - frame_count)
        raw = np.concatenate([raw, pad])
    arr = raw[:expected].reshape(frames, 360, 640)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, arr)
    return out_npy


def source_list() -> list[Path]:
    return [
        ROOT / "rtl" / "integration" / "chip_top.v",
        *sorted((ROOT / "rtl" / "multiframe_codec_v2").glob("*.v")),
    ]


def write_makefile(sim_dir: Path, frames_npy: Path, qp: int, result_json: Path) -> None:
    sources = " ".join(str(p) for p in source_list())
    makefile = sim_dir / "Makefile"
    makefile.write_text(
        "\n".join([
            "SIM = verilator",
            "TOPLEVEL_LANG = verilog",
            "TOPLEVEL = chip_top",
            "MODULE = test_current_chip_top_rd",
            "EXTRA_ARGS += --trace -Wno-DECLFILENAME -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC",
            "EXTRA_ARGS += -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM -Wno-BLKSEQ",
            f"VERILOG_SOURCES = {sources}",
            f"PYTHONPATH := {ROOT}/tb/rd:{ROOT}/tb/validation:{ROOT}:$(PYTHONPATH)",
            f"export CHIP_TOP_VALIDATION_FRAMES := {frames_npy}",
            f"export QP := {qp}",
            f"export RESULT_JSON := {result_json}",
            "include $(shell cocotb-config --makefiles)/Makefile.sim",
            "",
        ]),
        encoding="utf-8",
    )


def run_point(qp: int, out_dir: Path, frames_npy: Path) -> dict:
    sim_dir = out_dir / f"sim_qp{qp}"
    if sim_dir.exists():
        shutil.rmtree(sim_dir)
    sim_dir.mkdir(parents=True)
    shutil.copy2(ROOT / "tb" / "rd" / "test_current_chip_top_rd.py", sim_dir)
    shutil.copy2(ROOT / "tb" / "validation" / "test_chip_top_validation.py", sim_dir)

    result_json = out_dir / f"current_verilator_qp{qp}.json"
    write_makefile(sim_dir, frames_npy, qp, result_json)
    env = os.environ.copy()
    env["PATH"] = f"{ROOT / 'venv' / 'bin'}:{env.get('PATH', '')}"
    log_path = out_dir / f"current_verilator_qp{qp}.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            ["make", "-j", "4"],
            cwd=sim_dir,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60 * 60,
        )
    if proc.returncode != 0:
        return {
            "qp": qp,
            "error": f"make failed with return code {proc.returncode}",
            "log": str(log_path),
        }
    data = json.loads(result_json.read_text(encoding="utf-8"))
    data["log"] = str(log_path)
    return data


def main() -> int:
    gif_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_GIF
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / ".coresmith" / "rd_current"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames_npy = make_frames(gif_path, out_dir / "mort_10f_640x360.npy")
    qps = [int(v) for v in os.environ.get("CORESMITH_RD_QPS", "24,36,48").split(",") if v]
    jobs = int(os.environ.get("CORESMITH_RD_JOBS", "1"))

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futs = {pool.submit(run_point, qp, out_dir, frames_npy): qp for qp in qps}
        for fut in as_completed(futs):
            result = fut.result()
            results.append(result)
            if "error" in result:
                print(f"QP{result['qp']} ERROR: {result['error']} log={result['log']}", flush=True)
            else:
                print(
                    f"QP{result['qp']} bpp={result['bpp']:.6f} "
                    f"psnr={result['psnr_db']} bytes={result['bytes']} "
                    f"decode_ok={result['decode_ok_frames']}/{result['frames']}",
                    flush=True,
                )

    results.sort(key=lambda item: item["qp"])
    summary = {"gif": str(gif_path), "frames_npy": str(frames_npy), "points": results}
    summary_path = out_dir / "current_verilator_rd_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}", flush=True)
    return 1 if any("error" in r for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
