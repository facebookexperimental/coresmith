# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""SystemC toolchain detection, build and smoke run (B2)."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def systemc_home() -> str:
    return os.environ.get("CORESMITH_SYSTEMC_HOME", "").strip()


def detect() -> dict:
    """``{ok, reason, cxx, systemc_home}`` -- a C++17 compiler plus SystemC
    headers and library (system-wide, or under CORESMITH_SYSTEMC_HOME)."""
    cxx = os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++")
    if not cxx:
        return {"ok": False, "reason": "no C++ compiler", "cxx": None, "systemc_home": systemc_home()}
    home = systemc_home()
    cands = [Path(home) / "include" / "systemc"] if home else []
    cands += [Path("/usr/include/systemc"), Path("/usr/local/include/systemc"),
              Path("/usr/local/systemc/include/systemc")]
    if not any(p.exists() for p in cands):
        return {"ok": False, "reason": "systemc headers not found (set CORESMITH_SYSTEMC_HOME)",
                "cxx": cxx, "systemc_home": home}
    src = "#include <systemc>\nint sc_main(int, char**) { return 0; }\n"
    tmp = Path("/tmp") / f"cs_sc_probe_{os.getpid()}"
    tmp.mkdir(exist_ok=True)
    (tmp / "p.cpp").write_text(src)
    inc = ["-I", f"{home}/include"] if home else []
    lib = ["-L", f"{home}/lib", "-L", f"{home}/lib-linux64"] if home else []
    try:
        p = subprocess.run([cxx, "-std=c++17", *inc, str(tmp / "p.cpp"), *lib, "-lsystemc", "-o", str(tmp / "p")],
                           capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "reason": str(exc), "cxx": cxx, "systemc_home": home}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if p.returncode != 0:
        return {"ok": False, "reason": "systemc link probe failed: " + (p.stderr or "")[-400:],
                "cxx": cxx, "systemc_home": home}
    return {"ok": True, "reason": "", "cxx": cxx, "systemc_home": home}


def build(model_dir, *, timeout_s: int = 900) -> dict:
    md = Path(model_dir)
    env = dict(os.environ)
    if systemc_home():
        env["SYSTEMC_HOME"] = systemc_home()
    try:
        p = subprocess.run(["make", "-s", "soc_model"], cwd=md, capture_output=True, text=True,
                           timeout=timeout_s, env=env)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "log": str(exc)}
    (md / "build.log").write_text(p.stdout + p.stderr)
    return {"ok": p.returncode == 0 and (md / "soc_model").exists(), "log": (p.stdout + p.stderr)[-6000:]}


def smoke(model_dir, *, ns: int = 1000, timeout_s: int = 300) -> dict:
    md = Path(model_dir)
    exe = md / "soc_model"
    if not exe.exists():
        return {"ok": False, "log": "soc_model not built"}
    try:
        p = subprocess.run([str(exe), "--ns", str(ns)], cwd=md, capture_output=True, text=True,
                           timeout=timeout_s)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "log": str(exc)}
    out = p.stdout + p.stderr
    (md / "smoke.log").write_text(out)
    return {"ok": p.returncode == 0 and "SOC_MODEL_SMOKE_OK" in out, "log": out[-6000:]}
