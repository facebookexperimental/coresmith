# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""PDK arithmetic-op delay characterization (the logic analog of mem_characterize).

Sweeps the primitive arithmetic operators a uArch datapath is built from
(add / sub / mul / compare / mux / shift, plus a SAD reduction tree -- the
exact op that blew up the video codec RD mode-search) across operand bitwidths,
**synthesizes each to sky130 std cells (yosys) and times its worst
input->output combinational path (OpenROAD STA)**, then fits Google-XLS's
delay form ``delay_ns ~= a*W + b*log2(W) + c`` per operator. The fitted model
answers, for a target clock period, *how much arithmetic fits in one pipeline
stage* -- the input an SDC scheduler needs and the thing CoreSmith currently
lacks (so a single-cycle exhaustive search compiles as one combinational cloud).

Methodology + delay form follow Google XLS's per-op characterization
(``xls/estimators/delay_model``, sky130, Yosys+OpenSTA -- our exact stack).

Reuses mem_characterize's PDK paths + STA constants so both halves of the PDK
characterization stage share one fingerprint/cache key. Results cache to
``~/.coresmith/pdk_char/arith_<pdk_hash>.json``; the cache short-circuits the
(expensive) sweep on every run after the first.
"""
from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Reuse the memory characterizer's resolved PDK paths, STA helper inputs, and
# PDK fingerprint so the two characterizations share one cache key.
from .mem_characterize import (
    _ARRIVAL_RE,
    _YOSYS_AREA_RE,
    CELL_LEF,
    LIBERTY,
    OPENROAD_BIN,
    TECH_LEF,
    pdk_hash,
)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
CACHE_DIR = Path(os.environ.get(
    "CORESMITH_PDK_CHAR_DIR", str(Path.home() / ".coresmith" / "pdk_char")))
SYNTH_TIMEOUT_S = int(os.environ.get("CORESMITH_ARITH_CHAR_SYNTH_TIMEOUT", "180"))
STA_TIMEOUT_S = int(os.environ.get("CORESMITH_ARITH_CHAR_STA_TIMEOUT", "120"))

# Default operand-width sweep. Powers + a couple of odd widths so the linear and
# log2 terms are both observable.
DEFAULT_WIDTHS = [4, 8, 12, 16, 24, 32]
# SAD reduction: number of absolute-difference terms summed in one cloud (the
# recon-core "evaluate K candidates in a cycle" shape). 8-bit pixels.
DEFAULT_SAD_TERMS = [2, 4, 8, 16, 32]
# LUT (ROM/table) address-width sweep. A combinational N-input->N-output table
# is 2**W entries, so the address width -- NOT the data width -- is what its
# delay scales with, and 2**W bounds how big a table we can actually synthesize.
# 8 is the canonical AES S-box (256x8); we keep the sweep <=12 (4096 entries)
# so the exhaustive case-table stays synthesizable in bounded time.
DEFAULT_LUT_WIDTHS = [4, 6, 8, 10, 12]


# --------------------------------------------------------------------------- #
# Operator RTL templates -- each emits a purely combinational top `op_top`
# --------------------------------------------------------------------------- #
def _emit_scalar(op: str, w: int) -> tuple[str, str]:
    """Return (verilog, top_name) for a combinational scalar operator at width w."""
    aw = w
    if op == "add":
        ports = f"input [{aw-1}:0] a, input [{aw-1}:0] b, output [{aw}:0] y"
        body = "assign y = a + b;"
    elif op == "sub":
        ports = f"input [{aw-1}:0] a, input [{aw-1}:0] b, output [{aw}:0] y"
        body = "assign y = a - b;"
    elif op == "mul":
        ports = f"input [{aw-1}:0] a, input [{aw-1}:0] b, output [{2*aw-1}:0] y"
        body = "assign y = a * b;"
    elif op == "cmp":  # unsigned less-than comparator (1-bit decision)
        ports = f"input [{aw-1}:0] a, input [{aw-1}:0] b, output y"
        body = "assign y = (a < b);"
    elif op == "mux":  # 2:1 select (sel is the worst input)
        ports = f"input [{aw-1}:0] a, input [{aw-1}:0] b, input sel, output [{aw-1}:0] y"
        body = "assign y = sel ? a : b;"
    elif op == "shift":  # barrel left-shift by a variable amount
        sw = max(1, (aw - 1).bit_length())
        ports = f"input [{aw-1}:0] a, input [{sw-1}:0] sh, output [{aw-1}:0] y"
        body = "assign y = a << sh;"
    else:
        raise ValueError(f"unknown scalar op {op!r}")
    v = f"module op_top({ports});\n  {body}\nendmodule\n"
    return v, "op_top"


def _emit_sad(terms: int, pix_w: int = 8) -> tuple[str, str]:
    """SAD reduction: y = sum_i |a_i - b_i| over `terms` pixels, packed buses.

    This is the canonical "evaluate the whole candidate in one combinational
    cloud" shape -- an absolute-difference per term feeding an adder tree -- so
    its delay-vs-terms curve is exactly what tells the scheduler how many SAD
    terms fit in a clock period before a register is mandatory.
    """
    n = terms
    accw = pix_w + max(1, (n).bit_length())  # enough bits for the running sum
    lines = [
        f"module op_top(input [{n*pix_w-1}:0] a, input [{n*pix_w-1}:0] b, "
        f"output [{accw-1}:0] y);",
    ]
    diffs = []
    for i in range(n):
        hi, lo = (i + 1) * pix_w - 1, i * pix_w
        lines.append(f"  wire [{pix_w}:0] d{i} = "
                     f"(a[{hi}:{lo}] >= b[{hi}:{lo}]) ? "
                     f"(a[{hi}:{lo}] - b[{hi}:{lo}]) : (b[{hi}:{lo}] - a[{hi}:{lo}]);")
        diffs.append(f"d{i}")
    lines.append(f"  assign y = {' + '.join(diffs)};")
    lines.append("endmodule\n")
    return "\n".join(lines), "op_top"


# --------------------------------------------------------------------------- #
# Non-arithmetic datapath primitives -- the vocabulary CoreSmith was BLIND to.
#
# A crypto round (AES, DES, Camellia, ...), a big VLC/table decode, or a
# Galois-field codec is built from ops that are NOT +/-/*/cmp/mux/shift: an
# N->N table lookup (S-box / ROM), a GF(2^W) multiply, and wide XOR mixing.
# The old model returned None for these, so the scheduler priced them at 0 ns
# and never registered within a round -> the ~27 ns AES round vs a 20 ns clock
# (WNS ~= -7.31 ns) that no amount of ret/opt could close. Characterize them
# the SAME way (synth -> STA -> XLS-form fit) so they are priced, chained, and
# pipelined like every other op.
# --------------------------------------------------------------------------- #

# AES S-box (the canonical 8->8 table). Used as the representative case for the
# lut(W=8) point so the characterized delay is a REAL S-box, not filler logic.
_AES_SBOX = (
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
)

# GF reduction polynomial (low-order feedback taps, below x^W). 0x1B is the AES
# field x^8+x^4+x^3+x+1 for W=8; masked to W bits it is a representative
# low-weight reduction for any width (the delay driver is the W-deep xtime/xor
# chain, not the exact tap set, so a valid field is not required to characterize
# the delay). Kept as a table so W=8 is exactly the AES field.
_GF_POLY = {8: 0x1B}


def _lut_values(w: int) -> list[int]:
    """Deterministic W-bit table contents for a 2**W-entry combinational LUT.

    W=8 is the real AES S-box (so lut(8) is a genuine S-box). Other widths get a
    deterministic nonlinear fill so yosys cannot fold the table to a constant or
    a trivial function -- the point is a representative random-logic ROM delay.
    """
    n = 1 << w
    mask = n - 1
    if w == 8:
        return list(_AES_SBOX)
    vals: list[int] = []
    x = 0x2545F491 & mask
    for i in range(n):
        x = ((i * 0x9E3779B1) ^ (i >> 1) ^ (x * 33 + 0xB5)) & mask
        vals.append(x)
    return vals


def _emit_lut(w: int) -> tuple[str, str]:
    """Combinational N-input -> N-output table (ROM / crypto S-box), 2**W entries.

    Characterized by the INPUT-ADDRESS width W: a 2**W:1 selection tree whose
    delay grows ~linearly in W. This is the op an AES SubBytes / DES S-box / VLC
    decode table is built from -- previously OUTSIDE CoreSmith's vocabulary.
    """
    table = _lut_values(w)
    lines = [
        f"module op_top(input [{w-1}:0] a, output reg [{w-1}:0] y);",
        "  always @* begin",
        "    case (a)",
    ]
    for i, val in enumerate(table):
        lines.append(f"      {w}'d{i}: y = {w}'d{val};")
    lines.append(f"      default: y = {w}'d0;")
    lines.append("    endcase")
    lines.append("  end")
    lines.append("endmodule\n")
    return "\n".join(lines), "op_top"


def _emit_gfmul(w: int) -> tuple[str, str]:
    """GF(2^W) multiply a*b (MixColumns-class), unrolled russian-peasant form.

    Each step is a conditional accumulate (``p ^= b[i] ? a``) plus an ``xtime``
    (``a = (a<<1) reduced by the field poly``); the a-chain and p-chain are both
    W deep, so the delay scales ~linearly in W -- exactly the Galois-field
    arithmetic a codec / AES MixColumns / CRC needs and the model was blind to.
    """
    mask = (1 << w) - 1
    poly = _GF_POLY.get(w, 0x1B) & mask
    lines = [
        f"module op_top(input [{w-1}:0] a, input [{w-1}:0] b, "
        f"output [{w-1}:0] y);",
        f"  wire [{w-1}:0] a0 = a;",
        f"  wire [{w-1}:0] p0 = b[0] ? a0 : {w}'d0;",
    ]
    for i in range(1, w):
        # xtime(a_{i-1}): shift left one, XOR the reduction poly back if MSB set.
        lines.append(
            f"  wire [{w-1}:0] a{i} = a{i-1}[{w-1}] ? "
            f"(({{a{i-1}[{w-2}:0], 1'b0}}) ^ {w}'d{poly}) : "
            f"{{a{i-1}[{w-2}:0], 1'b0}};"
        )
        lines.append(f"  wire [{w-1}:0] p{i} = p{i-1} ^ (b[{i}] ? a{i} : {w}'d0);")
    lines.append(f"  assign y = p{w-1};")
    lines.append("endmodule\n")
    return "\n".join(lines), "op_top"


def _emit_xortree(w: int) -> tuple[str, str]:
    """W-wide XOR reduction (parity) tree -- the wide-XOR mixing an AddRoundKey /
    MixColumns column-sum / CRC folding is built from. Depth ~ log2(W)."""
    ports = f"input [{w-1}:0] a, output y"
    body = "assign y = ^a;"
    return f"module op_top({ports});\n  {body}\nendmodule\n", "op_top"


def _emit_op(op: str, w: int) -> tuple[str, str]:
    """Dispatch to the right combinational-top emitter for an op x width."""
    if op == "lut":
        return _emit_lut(w)
    if op == "gfmul":
        return _emit_gfmul(w)
    if op == "xortree":
        return _emit_xortree(w)
    return _emit_scalar(op, w)


# --------------------------------------------------------------------------- #
# yosys combinational synth -> mapped netlist + area
# --------------------------------------------------------------------------- #
def _synth_comb(src_path: str, top: str, liberty: str,
                timeout_s: int = SYNTH_TIMEOUT_S) -> dict[str, Any]:
    """Map a purely combinational top to sky130 std cells; write the netlist."""
    netlist = str(Path(src_path).with_suffix(".netlist.v"))
    script = (
        f"read_verilog -sv {src_path}\n"
        f"hierarchy -check -top {top}\n"
        "proc; opt; flatten; opt\n"
        # A full case-table (a ROM / S-box) is inferred by ``proc`` as a memory
        # (reg-array + initial block); without mapping it to logic it stays a
        # behavioral ROM that ``abc`` never maps to std cells (0 cells, no area)
        # and OpenROAD's read_verilog cannot parse (escaped-id + initial block)
        # -> the whole ``lut`` op fails to characterize. ``memory_map`` turns the
        # ROM into a combinational address-decode + mux so it maps to gates.
        # No-op for the ops with no memory (add/mul/gfmul/...).
        "memory_collect; memory_map; opt\n"
        "techmap; opt\n"
        f"abc -liberty {liberty}\n"        # no dfflibmap: combinational
        "clean; opt_clean -purge\n"
        f"stat -liberty {liberty}\n"
        f"write_verilog -noattr {netlist}\n"
    )
    yosys = shutil.which("yosys")
    if not yosys:
        return {"error": "yosys not found"}
    with tempfile.NamedTemporaryFile("w", suffix=".ys", delete=False) as fh:
        fh.write(script)
        ys = fh.name
    try:
        r = subprocess.run([yosys, "-s", ys], capture_output=True, text=True,
                           timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {"error": f"yosys timed out ({timeout_s}s)"}
    except OSError as exc:
        return {"error": f"yosys failed: {exc}"}
    finally:
        try:
            Path(ys).unlink()
        except OSError:
            pass
    out = r.stdout + "\n" + r.stderr
    if r.returncode != 0:
        return {"error": f"yosys exit {r.returncode}: {(r.stderr or out)[-200:].strip()}"}
    area = None
    m = _YOSYS_AREA_RE.search(out)
    if m:
        area = float(m.group(1))
    return {"area_um2": area, "netlist": netlist if Path(netlist).exists() else ""}


def _comb_delay_ns(netlist: str, top: str, liberty: str,
                   timeout_s: int = STA_TIMEOUT_S) -> float | None:
    """Worst input->output combinational delay (ns) via OpenROAD STA.

    A combinational op has no clock, so we constrain input->output directly with
    ``set_max_delay`` (no ``create_clock``) and read the worst path's data
    arrival time. Reuses mem_characterize's LEF/liberty + arrival-line regex.
    """
    big = 100000.0
    # ``report_checks``' "single worst path" flag is spelled ``-group_count`` in
    # OpenSTA/OpenROAD; some builds reject ``-group_path_count`` outright
    # ([ERROR STA-0563] ... not a known keyword or flag), which silently failed
    # EVERY point (empty model, per-op pricing disabled). Prefer the recognized
    # flag; fall back to the bare command (one path group here, so the default
    # already reports the single worst input->output path) if a build rejects it.
    for rc_cmd in (
        "report_checks -path_delay max -group_count 1 -fields {} -no_line_splits",
        "report_checks -path_delay max -no_line_splits",
    ):
        script = (
            f"read_lef {TECH_LEF}\n"
            f"read_lef {CELL_LEF}\n"
            f"read_liberty {liberty}\n"
            f"read_verilog {netlist}\n"
            f"link_design {top}\n"
            f"set_max_delay {big} -from [all_inputs] -to [all_outputs]\n"
            f"{rc_cmd}\n"
            "exit\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".tcl", delete=False) as fh:
            fh.write(script)
            tcl = fh.name
        try:
            r = subprocess.run([OPENROAD_BIN, "-no_init", "-exit", tcl],
                               capture_output=True, text=True, timeout=timeout_s)
        except (subprocess.TimeoutExpired, OSError):
            return None
        finally:
            try:
                Path(tcl).unlink()
            except OSError:
                pass
        m = _ARRIVAL_RE.search(r.stdout)
        if m:
            d = float(m.group(1))
            return d if d > 0 else None
        # Only retry with the fallback command when the flag itself was rejected.
        if "not a known keyword or flag" not in r.stdout:
            return None
    return None


# --------------------------------------------------------------------------- #
# Characterization
# --------------------------------------------------------------------------- #
@dataclass
class OpPoint:
    op: str
    width: int           # operand bitwidth (or #terms for SAD)
    delay_ns: float | None
    area_um2: float | None
    error: str = ""


def _measure(verilog: str, top: str, liberty: str) -> tuple[float | None, float | None, str]:
    """Synth + STA one op top. Returns (delay_ns, area_um2, error)."""
    with tempfile.NamedTemporaryFile("w", suffix=".v", delete=False,
                                     dir=str(CACHE_DIR)) as fh:
        fh.write(verilog)
        src = fh.name
    try:
        syn = _synth_comb(src, top, liberty)
        if syn.get("error") or not syn.get("netlist"):
            return None, None, syn.get("error", "no netlist")
        delay = _comb_delay_ns(syn["netlist"], top, liberty)
        return delay, syn.get("area_um2"), ("" if delay else "STA failed")
    finally:
        for p in (src, str(Path(src).with_suffix(".netlist.v"))):
            try:
                Path(p).unlink()
            except OSError:
                pass


# Operators swept over the operand-width grid. Includes the crypto/codec
# primitives -- ``gfmul`` (Galois-field multiply) and ``xortree`` (wide XOR
# mixing) -- that the pure-arithmetic vocabulary was missing. ``lut`` is swept
# separately over DEFAULT_LUT_WIDTHS (its size is 2**W, not W).
WIDTH_SWEEP_OPS = ("add", "sub", "mul", "cmp", "mux", "shift", "gfmul", "xortree")


def characterize_ops(widths: list[int] | None = None,
                     sad_terms: list[int] | None = None,
                     lut_widths: list[int] | None = None,
                     liberty: str = "") -> list[OpPoint]:
    """Sweep every operator x width through synth+STA. Returns the raw points."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    lib = liberty or str(LIBERTY)
    widths = widths or DEFAULT_WIDTHS
    sad_terms = sad_terms or DEFAULT_SAD_TERMS
    lut_widths = lut_widths or DEFAULT_LUT_WIDTHS
    pts: list[OpPoint] = []
    for op in WIDTH_SWEEP_OPS:
        for w in widths:
            v, top = _emit_op(op, w)
            d, a, err = _measure(v, top, lib)
            pts.append(OpPoint(op, w, d, a, err))
    # lut (ROM / S-box): swept by ADDRESS width, bounded (2**W entries).
    for w in lut_widths:
        v, top = _emit_lut(w)
        d, a, err = _measure(v, top, lib)
        pts.append(OpPoint("lut", w, d, a, err))
    for k in sad_terms:
        v, top = _emit_sad(k)
        d, a, err = _measure(v, top, lib)
        pts.append(OpPoint("sad", k, d, a, err))
    return pts


# --------------------------------------------------------------------------- #
# Fit: delay_ns ~= a*W + b*log2(W) + c  (XLS form), per operator
# --------------------------------------------------------------------------- #
def fit_delay_model(points: list[OpPoint]) -> dict[str, dict[str, float]]:
    """Least-squares fit of the XLS delay form per operator.

    Returns ``{op: {"a":.., "b":.., "c":.., "n":int, "max_resid_ns":..}}``.
    """
    import numpy as np
    by_op: dict[str, list[OpPoint]] = {}
    for p in points:
        if p.delay_ns and p.width > 0:
            by_op.setdefault(p.op, []).append(p)
    model: dict[str, dict[str, float]] = {}
    for op, ps in by_op.items():
        W = np.array([p.width for p in ps], dtype=float)
        y = np.array([p.delay_ns for p in ps], dtype=float)
        X = np.column_stack([W, np.log2(W), np.ones_like(W)])
        if len(ps) < 3:  # not enough to fit 3 params -> fall back to mean slope
            coef = [float(y.mean() / max(W.mean(), 1.0)), 0.0, 0.0]
        else:
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = float(np.max(np.abs(X @ np.array(coef) - y))) if len(ps) >= 3 else 0.0
        model[op] = {"a": float(coef[0]), "b": float(coef[1]), "c": float(coef[2]),
                     "n": len(ps), "max_resid_ns": resid}
    return model


# --------------------------------------------------------------------------- #
# Cache + public API
# --------------------------------------------------------------------------- #
def _cache_path(pdk: dict | None = None) -> Path:
    return CACHE_DIR / f"arith_{pdk_hash(pdk)}.json"


logger = logging.getLogger(__name__)


def _model_is_degenerate(doc: dict | None) -> bool:
    """A characterization is degenerate when STA produced no usable delays.

    Happens when OpenROAD/STA isn't available on this box (every synth+STA
    point fails) -- ``fit_delay_model`` then returns an empty model. Such a
    result must NOT count as "characterized": caching it as success masks the
    missing tool and silently disables per-op delay pricing on every later run.
    """
    if not doc:
        return True
    model = doc.get("model") or {}
    if not model:
        return True
    return all(int(m.get("n", 0)) == 0 for m in model.values())


def is_characterized(pdk: dict | None = None) -> bool:
    """True iff a NON-degenerate arithmetic delay model is cached for this PDK."""
    path = _cache_path(pdk)
    if not path.exists():
        return False
    try:
        doc = json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return False
    return not _model_is_degenerate(doc)


def ensure_characterized(pdk: dict | None = None, force: bool = False,
                         widths: list[int] | None = None) -> dict[str, Any]:
    """Load the cached arithmetic delay model, or run the sweep once and cache it.

    This is the entry point the PDK-characterization stage calls. Cheap on every
    run after the first (PDK fingerprint keys the cache). A degenerate result
    (STA unavailable -> empty model) is NOT cached, so a later run on a box
    WITH OpenROAD re-tries and the budget path degrades cleanly to
    "uncharacterized" rather than caching a silent no-op."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(pdk)
    if path.exists() and not force:
        try:
            cached = json.loads(path.read_text())
            if not _model_is_degenerate(cached):
                return cached
            # a previously-cached degenerate doc: fall through and re-try
            # (OpenROAD may be present now); never serve it as success.
        except Exception:  # noqa: BLE001
            pass
    points = characterize_ops(widths=widths)
    model = fit_delay_model(points)
    doc = {
        "pdk_hash": pdk_hash(pdk),
        "liberty": str(LIBERTY),
        "delay_form": "a*W + b*log2(W) + c   (ns)",
        "model": model,
        "degenerate": False,
        "points": [p.__dict__ for p in points],
    }
    if _model_is_degenerate(doc):
        doc["degenerate"] = True
        n_fail = sum(1 for p in points if not p.delay_ns)
        logger.warning(
            "arith characterization DEGENERATE: %d/%d STA points failed "
            "(OpenROAD on PATH? OPENROAD_BIN=%s) -- NOT caching; per-op delay "
            "pricing degrades to uncharacterized (structural stage-map + cycle "
            "reconciliation still work).", n_fail, len(points), OPENROAD_BIN,
        )
        return doc
    path.write_text(json.dumps(doc, indent=2))
    return doc


_MODEL_CACHE: dict[str, dict] = {}


def _load_model(pdk: dict | None = None) -> dict[str, Any] | None:
    key = pdk_hash(pdk)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    path = _cache_path(pdk)
    if not path.exists():
        return None
    try:
        doc = json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None
    _MODEL_CACHE[key] = doc
    return doc


def _fit_delay(m: dict, w: int) -> float:
    """Evaluate the XLS delay form for one op's fitted coefficients at width w."""
    return max(0.0, m["a"] * w + m["b"] * math.log2(w) + m["c"])


def is_op_characterized(op: str, pdk: dict | None = None) -> bool | None:
    """True iff `op` has its OWN fitted delay entry in the characterized model.

    ``None`` when the PDK itself is uncharacterized (no model doc at all) --
    distinct from ``False`` (model present, but this op is outside the swept
    vocabulary and will be priced by the conservative fallback)."""
    doc = _load_model(pdk)
    if not doc:
        return None
    return op in (doc.get("model") or {})


def _conservative_unknown_delay(width: int, model: dict) -> float | None:
    """Delay proxy for an op OUTSIDE the characterized vocabulary.

    An uncharacterized op is NEVER timing-free -- assuming it is (returning
    None -> scheduler 0 ns) is exactly the bug that let an AES round's S-box +
    GF-multiply collapse into one unregistered ~27 ns cloud. Price it at the
    SLOWEST known op at this width: a deliberately pessimistic proxy so the
    scheduler gives an unknown op at least its own pipeline stage instead of
    letting it vanish. Returns None only when the model is empty."""
    if not model:
        return None
    best = max((_fit_delay(m, width) for m in model.values()), default=0.0)
    return best if best > 0 else None


def predict_op_delay(op: str, width: int, pdk: dict | None = None) -> float | None:
    """Predicted combinational delay (ns) of `op` at `width` (or #terms for sad).

    Returns None ONLY when the PDK model isn't characterized yet (no model doc
    -- caller should run ``ensure_characterized`` at the PDK stage first). When
    a model IS characterized but `op` is outside its vocabulary (a crypto
    S-box, a table lookup, a Galois-field op the sweep didn't include), the op
    is priced by a CONSERVATIVE fallback -- the slowest known op at this width
    -- so it is never treated as timing-free. The scheduler uses this to decide
    how many ops chain within one clock period before a register."""
    doc = _load_model(pdk)
    if not doc:
        return None
    model = doc.get("model") or {}
    w = max(1, int(width))
    m = model.get(op)
    if m:
        return _fit_delay(m, w)
    # Uncharacterized op, but the PDK IS characterized -> conservative proxy
    # (never None/0 here; that is only for a fully-uncharacterized PDK above).
    return _conservative_unknown_delay(w, model)


def ops_per_stage(op: str, width: int, period_ns: float,
                  pdk: dict | None = None) -> int | None:
    """How many `op`s of `width` chain within one clock `period_ns` (>=1).

    A convenience for the scheduler / uArch prompt: the per-stage arithmetic
    budget for a chain of identical ops. Returns None ONLY when the PDK is
    uncharacterized. An op OUTSIDE the vocabulary conservatively gets its OWN
    stage (returns 1) -- we never chain an op we cannot price."""
    doc = _load_model(pdk)
    if not doc:
        return None
    model = doc.get("model") or {}
    if op not in model:
        # Unknown op with a characterized PDK: its own stage, never None.
        return 1 if model else None
    d = predict_op_delay(op, width, pdk)
    if not d or d <= 0:
        return None
    return max(1, int(period_ns // d))


def op_width_in_grid(op: str, width: int,
                     pdk: dict | None = None) -> bool | None:
    """Applicability-domain companion to ``predict_op_delay`` (no numeric change).

    The XLS delay form ``a*W + b*log2(W) + c`` extrapolates SANELY (monotone,
    unbounded) unlike the memory regressor, so ``predict_op_delay`` still returns
    a usable number out-of-sweep. But consumers (pipeline_scheduler, budgets)
    should ANNOTATE an out-of-sweep operand width as an extrapolation. This query
    reports it from the CHARACTERIZED widths in the cache:

      * ``True``  -- width within the swept [min, max] operand widths for `op`;
      * ``False`` -- out-of-sweep (delay is an extrapolation of the fitted form);
      * ``None``  -- model uncharacterized / `op` not swept (delay is None too).
    """
    doc = _load_model(pdk)
    if not doc:
        return None
    pts = [p for p in (doc.get("points") or [])
           if p.get("op") == op and p.get("delay_ns") and int(p.get("width", 0)) > 0]
    if not pts:
        return None
    ws = [int(p["width"]) for p in pts]
    return min(ws) <= int(width) <= max(ws)


def predict_op_delay_annotated(op: str, width: int,
                               pdk: dict | None = None) -> dict[str, Any]:
    """``predict_op_delay`` + applicability flags, for consumers that want the
    delay AND its provenance in one call. Numeric ``delay_ns`` is unchanged.

    ``known`` is True when `op` has its OWN fitted delay; when False (but not
    None) the ``delay_ns`` came from the CONSERVATIVE unknown-op fallback --
    surfaced as ``extrapolated`` so a consumer can flag the estimate as a
    pessimistic proxy rather than a measured curve."""
    known = is_op_characterized(op, pdk)
    return {
        "op": op,
        "width": int(width),
        "delay_ns": predict_op_delay(op, width, pdk),
        "in_grid": op_width_in_grid(op, width, pdk),
        "known": known,
        # Extrapolated == priced by the unknown-op fallback (model present, op
        # not in vocabulary). None known-state -> uncharacterized PDK, no proxy.
        "extrapolated": (known is False),
    }


__all__ = [
    "OpPoint",
    "characterize_ops",
    "ensure_characterized",
    "fit_delay_model",
    "is_op_characterized",
    "op_width_in_grid",
    "ops_per_stage",
    "predict_op_delay",
    "predict_op_delay_annotated",
]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from orchestrator.profile import apply as _apply_profile
    _apply_profile()
    import sys
    args = sys.argv[1:]
    if args and args[0] == "predict":
        op, width = args[1], int(args[2])
        print(f"{op} @W={width}: {predict_op_delay(op, width)} ns")
    elif args and args[0] == "ops-per-stage":
        op, width, period = args[1], int(args[2]), float(args[3])
        print(f"{op}@{width} in {period}ns: {ops_per_stage(op, width, period)} chained")
    else:
        force = "--force" in args
        doc = ensure_characterized(force=force)
        print(json.dumps(doc["model"], indent=2))
