# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""RTL Acceptance DV [dv-hardening-16] -- native-speed RTL-vs-GOLDEN at
mission scale.

The missing fourth tier of the DV ladder (armC finding):

  composition gate  (model vs golden, fast stimulus)     -- uarch stage
  full model DV     (model vs golden, mission scale)     -- uarch stage
  integration DV    (RTL vs model, cocotb)               -- RTL stage
  ACCEPTANCE DV     (RTL vs GOLDEN, mission scale)       -- RTL stage  <- this

Integration DV's oracle IS the composed model, so it can never see a
model-vs-golden divergence; and cocotb's per-edge Python overhead makes
mission-scale frames unaffordable (a QCIF frame that takes ~2s under a native
C++ Verilator harness takes >>10min under cocotb). This tier generates a
NATIVE C++ Verilator harness from the chip top's port contract (seeded from
the armC measurement harness that ran 80 full-QCIF encodes in ~3 minutes),
validates it against the composed-model oracle before trusting it, then runs
the FRD's acceptance cases (>=10 recommended: geometry x QP sweeps for image
IPs, audio segments, benchmark programs for CPUs -- the artifact defines the
matrix, the engine stays domain-generic) comparing RTL output to the golden
under the DECLARED criterion (fidelity metric when present, else byte-exact).

HONEST SKIPS, NEVER A FALSE PASS (same charter as rtl_model_equiv): no
acceptance artifact / no verilator / unsupported port shape / harness fails
oracle validation -> ``{"skipped": True, reason}``. A divergence, a fidelity
break, or a watchdog expiry IS a violation.

Env:
- ``CORESMITH_ACCEPTANCE_DV=0`` disables (default on).
- ``CORESMITH_ACCEPTANCE_DV_BUILD_TIMEOUT_S`` (default 900) verilate+build.
- ``CORESMITH_ACCEPTANCE_DV_RUN_TIMEOUT_S`` (default 1800, elastic x1.5 via
  state file) for the full case sweep.
- ``CORESMITH_ACCEPTANCE_DV_MAX_CYCLES`` per-frame watchdog (default 50M).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from orchestrator.processes import run as run_process

logger = logging.getLogger(__name__)


def acceptance_dv_enabled() -> bool:
    return os.environ.get(
        "CORESMITH_ACCEPTANCE_DV", "1"
    ).strip().lower() not in ("0", "false", "no", "off")


def _skip(reason: str) -> dict:
    """The stage does not apply (no oracle declared, shape not covered)."""
    logger.info("acceptance dv: SKIPPED -- %s", reason)
    return {"passed": False, "skipped": True, "reason": reason, "cases": []}


def _incomplete(reason: str, kind: str = "oracle_incomplete") -> dict:
    """WP-38: the oracle IS declared but did not complete -- build failure,
    tool exit, receipt cardinality, an undeclared case setting. Never a skip
    (the pipeline treats a skip as nonblocking) and never a pass: it parks
    with kind in {oracle_incomplete, adapter_defect, infrastructure_error}."""
    logger.warning("acceptance dv: INCOMPLETE (%s) -- %s", kind, reason)
    return {"passed": False, "skipped": False, "oracle_incomplete": True,
            "kind": kind, "reason": reason, "cases": [],
            "violations": [{
                "type": "acceptance_dv_failure",
                "criterion": "acceptance_dv_oracle_incomplete",
                "kind": kind,
                "suggested_fix": "Not an RTL verdict: the acceptance oracle did "
                                 "not complete. Fix the adapter / toolchain, "
                                 "then retry.",
            }]}


# ---------------------------------------------------------------------------
# Port contract discovery (regex over the chip-top Verilog, like rtl_model_equiv)
# ---------------------------------------------------------------------------

_PORT_RE = re.compile(
    r"\b(input|output)\s+(?:wire|reg|logic)?\s*"
    r"(?:\[\s*(\d+)\s*:\s*0\s*\]\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*[,;)]",
)


def discover_ports(rtl_text: str) -> dict[str, dict]:
    """{name: {dir, width}} for every port in the top module header region."""
    ports: dict[str, dict] = {}
    for m in _PORT_RE.finditer(rtl_text):
        direction, msb, name = m.group(1), m.group(2), m.group(3)
        ports[name] = {
            "dir": direction,
            "width": (int(msb) + 1) if msb else 1,
        }
    return ports


def classify_contract(ports: dict[str, dict]) -> dict | None:
    """Classify the framed-stream contract: clk/reset, s_axis group, m_axis
    group, sideband config inputs. None when the shape is unsupported."""
    names = set(ports)

    clk = next((n for n in ("clk", "clock", "clk_in") if n in names), None)
    rst = next((n for n in ("rst_n", "resetn", "rst", "reset") if n in names), None)
    if not clk or not rst:
        return None
    rst_active_low = rst.endswith("n")

    def _group(direction: str) -> dict | None:
        # find <prefix>_tvalid with a matching _tready of the opposite driver
        for n, p in ports.items():
            if not n.endswith("_tvalid") or p["dir"] != direction:
                continue
            prefix = n[: -len("_tvalid")]
            if f"{prefix}_tready" in names and f"{prefix}_tdata" in names:
                return {
                    "prefix": prefix,
                    "data_width": ports[f"{prefix}_tdata"]["width"],
                    "has_tuser": f"{prefix}_tuser" in names,
                    "has_tlast": f"{prefix}_tlast" in names,
                }
        return None

    s_axis = _group("input")
    m_axis = _group("output")
    if not s_axis or not m_axis:
        return None

    # dv-hardening-25 (aes128 OOD): a top with MORE THAN ONE input stream group
    # (e.g. s_axis + s_axis_key on a block cipher) cannot be driven by this
    # single-input-stream harness -- it maps the 2nd stream's ports to constant
    # scalar sidebands, so that stream never loads and every case watchdogs to
    # a FALSE FAILURE (aes128 live: 3/3 rtl_bytes=0 while the RTL was actually
    # byte-exact, proven by an independent two-stream cocotb TB). Honest-skip
    # rather than false-fail; the harness only supports one input stream.
    _in_groups = {
        n[: -len("_tvalid")]
        for n, p in ports.items()
        if n.endswith("_tvalid") and p["dir"] == "input"
        and f"{n[: -len('_tvalid')]}_tready" in names
        and f"{n[: -len('_tvalid')]}_tdata" in names
    }
    if len(_in_groups) > 1:
        return None

    stream_names = {clk, rst}
    for g in (s_axis, m_axis):
        for suf in ("tdata", "tvalid", "tready", "tuser", "tlast"):
            stream_names.add(f"{g['prefix']}_{suf}")
    sidebands = {
        n: p["width"]
        for n, p in ports.items()
        if p["dir"] == "input" and n not in stream_names
    }
    return {
        "kind": "axis",
        "clk": clk,
        "rst": rst,
        "rst_active_low": rst_active_low,
        "s_axis": s_axis,
        "m_axis": m_axis,
        "sidebands": sidebands,
    }


def map_stimulus(stimulus: Any, contract: dict, mapping=None) -> dict | None:
    """Apply task-owned AXIS_MAPPING, never infer payload or sideband meanings.

    AXIS_MAPPING declares payload field, input_width, output_width, packing
    (bytes or words), byte_order (little), and sidebands {port: stimulus_field}.
    A case can supply its own axis_mapping. Values must be flat and in range;
    owners explicitly flatten arrays and declare geometry as scalar sidebands.
    """
    if isinstance(stimulus, dict):
        mapping = stimulus.get("axis_mapping", mapping)
    if not isinstance(mapping, dict):
        return None
    try:
        iw, ow = mapping["input_width"], mapping["output_width"]
        if (type(iw) is not int or type(ow) is not int or iw not in range(8, 65, 8)
                or ow not in range(8, 65, 8) or iw != contract["s_axis"]["data_width"]
                or ow != contract["m_axis"]["data_width"] or mapping["byte_order"] != "little"):
            return None
        packing = mapping["packing"]
        if packing not in ("bytes", "words"):
            return None
        field = mapping["payload"]
        payload = stimulus[field] if isinstance(stimulus, dict) else stimulus if field is None else None
        if hasattr(payload, "tolist"):
            payload = payload.tolist()
        if not isinstance(payload, (list, tuple, bytes, bytearray)):
            return None
        bound = 256 if packing == "bytes" else 1 << iw
        if any(type(v) is not int or not 0 <= v < bound for v in payload):
            return None
        flat = list(payload) if packing == "bytes" else list(b"".join(
            v.to_bytes(iw // 8, "little") for v in payload))
        if len(flat) % (iw // 8):
            return None
        ports = contract["sidebands"]
        sb_map = mapping["sidebands"]
        if not isinstance(sb_map, dict) or set(sb_map) != set(ports):
            return None
        sidebands = {}
        for port, key in sb_map.items():
            value = stimulus[key]
            if type(value) is not int or not 0 <= value < (1 << min(ports[port], 32)):
                return None
            sidebands[port] = value
        return {"payload": flat, "sidebands": sidebands, "unmapped": []}
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def _call_accept(accept_fn, expected, observed, name: str, stimulus: Any):
    """Call the task predicate with the case when it takes one (WP-38): a
    per-case floor (33 dB for one content class, 30 for another) needs the
    case, and a context-free accept(expected, observed) cannot carry it."""
    import inspect as _inspect

    try:
        params = _inspect.signature(accept_fn).parameters
        if "case" in params or any(p.kind == p.VAR_KEYWORD for p in params.values()):
            return accept_fn(expected, observed,
                             case={"name": name, "stimulus": stimulus})
    except (TypeError, ValueError):
        pass
    return accept_fn(expected, observed)


# ---------------------------------------------------------------------------
# Native harness generation (seeded from the armC measurement harness)
# ---------------------------------------------------------------------------

_HARNESS_TEMPLATE = r"""// Auto-generated RTL Acceptance DV harness (coresmith dv-hardening-16).
// Seeded from the armC measurement harness: inputs set up after the falling
// edge, handshakes sampled after the rising edge (mirrors the cocotb TBs).
// Binary protocol (little-endian u32):
//   input:  repeated { n_sidebands, {sb_value}*n, n_pixels, pixel bytes }
//   output: repeated { status, cycles, outlen, out bytes }  (status 0=ok 1=watchdog)
#include "V{TOP}.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

using DUT = V{TOP};
static vluint64_t g_cycle = 0;
static void posedge(DUT *t) { t->{CLK} = 1; t->eval(); g_cycle++; }
static void negedge(DUT *t) { t->{CLK} = 0; t->eval(); }

// [dv-hardening-27] Deterministic xorshift PRNG for randomized output
// backpressure + input gaps. A design that drops out_valid on its own transfer
// edge, or skews in_ready per beat, only fails UNDER backpressure -- with
// ready wired high and no input gaps (the old harness) those bugs pass. The
// A run-through proved the point: CoreSmith's byte/no-backpressure
// harness shipped two handshake bugs an independent backpressuring TB caught.
static uint32_t g_rng = 0x2545F491u;
static inline uint32_t xrng() {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5;
    return g_rng;
}
// 0 = wire ready high + no gaps (legacy); 1 = randomized backpressure + gaps.
static const int BP = {BP_ENABLED};
// Bytes driven/captured per stream beat (= ceil(tdata_width/8), capped at 8).
static const int W_IN = {W_IN};
static const int W_OUT = {W_OUT};

static bool rd_u32(FILE *f, uint32_t *v) {
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4) return false;
    *v = (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    return true;
}
static void wr_u32(FILE *f, uint32_t v) {
    unsigned char b[4] = {(unsigned char)(v), (unsigned char)(v >> 8),
                          (unsigned char)(v >> 16), (unsigned char)(v >> 24)};
    fwrite(b, 1, 4, f);
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    if (argc < 3) { fprintf(stderr, "usage: %s in.bin out.bin [max_cycles]\n", argv[0]); return 2; }
    uint64_t max_cycles = (argc >= 4) ? strtoull(argv[3], nullptr, 10) : {MAX_CYCLES}ULL;
    FILE *fin = fopen(argv[1], "rb");
    FILE *fout = fopen(argv[2], "wb");
    if (!fin || !fout) { fprintf(stderr, "io error\n"); return 2; }
    DUT *top = new DUT;

    uint32_t n_sb;
    while (rd_u32(fin, &n_sb)) {
        std::vector<uint32_t> sb(n_sb);
        for (uint32_t i = 0; i < n_sb; i++) if (!rd_u32(fin, &sb[i])) return 2;
        uint32_t n_pix;
        if (!rd_u32(fin, &n_pix)) return 2;
        std::vector<uint8_t> px(n_pix);
        if (n_pix && fread(px.data(), 1, n_pix, fin) != n_pix) return 2;

        // reset (per case seed so backpressure is deterministic + reproducible)
        g_rng = 0x2545F491u ^ ((uint32_t)n_pix * 2654435761u) ^ (n_sb * 40503u);
        top->{S_TVALID} = 0; top->{M_TREADY} = 1;
{SB_ZERO}
        top->{RST} = {RST_ASSERT};
        for (int i = 0; i < 5; i++) { negedge(top); posedge(top); }
        negedge(top);
        top->{RST} = {RST_DEASSERT};
        for (int i = 0; i < 5; i++) { negedge(top); posedge(top); }
        negedge(top);

        // [dv-hardening-30] Width-aware: pack W_IN payload bytes per input beat
        // into s_tdata (little-endian) and capture W_OUT bytes per output beat.
        // For a byte-wide stream (W_IN=W_OUT=1) this is byte-identical to the
        // old behaviour; for a 32-bit word stream (matmul) it drives whole words
        // instead of a single byte -- the run-through's other limitation.
        size_t n = (W_IN > 0) ? (px.size() / (size_t)W_IN) : px.size();
        size_t idx = 0;
        bool s_done = (n == 0), r_done = false, s_pend = false;
        uint64_t start_c = UINT64_MAX, end_c = UINT64_MAX, wd = 0, m_idle = 0;
        std::vector<uint8_t> out;

        auto load = [&](size_t i) {
            uint64_t w = 0;
            for (int b = 0; b < W_IN; b++) w |= (uint64_t)px[i * W_IN + b] << (8 * b);
            top->{S_TDATA} = w;
{S_TUSER_SET}
{S_TLAST_SET}
{SB_SET}
            top->{S_TVALID} = 1;
        };

        while ((!s_done || !r_done) && wd < max_cycles) {
            // Drive the handshake signals for THIS cycle before the edge.
            // Output backpressure: deassert m_tready ~30% of cycles.
            top->{M_TREADY} = (BP && (xrng() & 7) < 3) ? 0 : 1;
            // Input: hold the current word until accepted; insert ~15% gaps
            // (in_valid low) without advancing -- AXI-Stream master behaviour.
            if (!s_done) {
                // Gap only BETWEEN transfers: once a beat is presented TVALID
                // must stay high until TREADY (AXI-Stream valid stability).
                if (!s_pend && BP && (xrng() % 100) < 15) top->{S_TVALID} = 0;
                else { load(idx); s_pend = true; }
            }
            posedge(top); wd++;
            if (!s_done && top->{S_TVALID} && top->{S_TREADY}) {
                if (start_c == UINT64_MAX) start_c = g_cycle;
                idx++; s_pend = false;
                if (idx >= n) { s_done = true; top->{S_TVALID} = 0; }
            }
            if (!r_done && top->{M_TVALID} && top->{M_TREADY}) {
                uint64_t d = (uint64_t)top->{M_TDATA};
                for (int b = 0; b < W_OUT; b++) out.push_back((uint8_t)((d >> (8 * b)) & 0xff));
                m_idle = 0;
{M_TLAST_CHECK}
            }
{M_IDLE_CHECK}
            negedge(top);
        }
        uint32_t status = (start_c == UINT64_MAX || end_c == UINT64_MAX) ? 1u : 0u;
        uint64_t cycles = (status == 0) ? (end_c - start_c) : 0;
        wr_u32(fout, status);
        wr_u32(fout, (uint32_t)cycles);
        wr_u32(fout, (uint32_t)out.size());
        fwrite(out.data(), 1, out.size(), fout);
        fflush(fout);
    }
    fclose(fin); fclose(fout); delete top;
    return 0;
}
"""


def generate_harness(contract: dict, top_module: str, sb_order: list[str]) -> str:
    s, m = contract["s_axis"], contract["m_axis"]
    sp, mp = s["prefix"], m["prefix"]
    sb_set = "\n".join(
        f"            top->{name} = sb[{i}];" for i, name in enumerate(sb_order)
    ) or "            (void)sb;"
    sb_zero = "\n".join(
        f"        top->{name} = 0;" for name in sb_order
    ) or "        (void)0;"
    tuser = (f"            top->{sp}_tuser = (i == 0) ? 1 : 0;"
             if s["has_tuser"] else "            (void)i;")
    tlast = (f"            top->{sp}_tlast = (i == n - 1) ? 1 : 0;"
             if s["has_tlast"] else "")
    m_tlast = (
        f"                if (top->{mp}_tlast) {{ end_c = g_cycle; r_done = true; }}"
        if m["has_tlast"] else
        "                end_c = g_cycle; /* no tlast: track last beat */"
    )
    # no egress tlast: done when the sender is done and the output has been
    # quiet for 64 cycles SINCE THE LAST ACCEPTED BEAT (m_idle resets on every
    # beat) -- counting wd here ended capture on the first post-drain beat.
    m_idle_check = (
        "            (void)m_idle;" if m["has_tlast"] else
        "            else if (s_done && end_c != UINT64_MAX && ++m_idle > 64)\n"
        "                r_done = true;"
    )
    return (
        _HARNESS_TEMPLATE
        .replace("{TOP}", top_module)
        .replace("{CLK}", contract["clk"])
        .replace("{RST}", contract["rst"])
        .replace("{RST_ASSERT}", "0" if contract["rst_active_low"] else "1")
        .replace("{RST_DEASSERT}", "1" if contract["rst_active_low"] else "0")
        .replace("{S_TDATA}", f"{sp}_tdata")
        .replace("{S_TVALID}", f"{sp}_tvalid")
        .replace("{S_TREADY}", f"{sp}_tready")
        .replace("{M_TDATA}", f"{mp}_tdata")
        .replace("{M_TVALID}", f"{mp}_tvalid")
        .replace("{M_TREADY}", f"{mp}_tready")
        .replace("{S_TUSER_SET}", tuser)
        .replace("{S_TLAST_SET}", tlast)
        .replace("{M_TLAST_CHECK}", m_tlast)
        .replace("{M_IDLE_CHECK}", m_idle_check)
        .replace("{SB_SET}", sb_set)
        .replace("{SB_ZERO}", sb_zero)
        .replace("{MAX_CYCLES}", os.environ.get(
            "CORESMITH_ACCEPTANCE_DV_MAX_CYCLES", "50000000"))
        .replace("{BP_ENABLED}", "0" if os.environ.get(
            "CORESMITH_ACCEPTANCE_DV_BACKPRESSURE", "1").strip().lower()
            in ("0", "false", "no", "off") else "1")
        .replace("{W_IN}", str(min(8, (int(s.get("data_width", 8)) + 7) // 8) or 1))
        .replace("{W_OUT}", str(min(8, (int(m.get("data_width", 8)) + 7) // 8) or 1))
    )


# ---------------------------------------------------------------------------
# Build + run
# ---------------------------------------------------------------------------

def _build(workdir: Path, top_module: str, sources: list[str],
           harness_cpp: Path) -> str | None:
    """verilate + build; returns the binary path or None (reason logged)."""
    timeout = int(float(os.environ.get(
        "CORESMITH_ACCEPTANCE_DV_BUILD_TIMEOUT_S", "900") or 900))
    cmd = [
        "verilator", "--cc", "--exe", "--build", "-j", "2",
        "--top-module", top_module, "-Wno-fatal",
        "-CFLAGS", "-O2",
        str(harness_cpp),
    ] + sources
    try:
        r = run_process(cmd, cwd=workdir, capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning("acceptance dv: verilator build timed out (%ss)", timeout)
        return None
    if r.returncode != 0:
        logger.warning("acceptance dv: verilator build failed:\n%s",
                       (r.stderr or r.stdout)[-1500:])
        return None
    binp = workdir / "obj_dir" / f"V{top_module}"
    return str(binp) if binp.exists() else None


def _pack_cases(cases: list[tuple[str, dict]], sb_order: list[str],
                path: Path) -> None:
    with open(path, "wb") as f:
        for _name, mapped in cases:
            sbv = mapped["sidebands"]
            f.write(struct.pack("<I", len(sb_order)))
            f.writelines(struct.pack("<I", sbv.get(p, 0) & 0xFFFFFFFF) for p in sb_order)
            payload = mapped["payload"]
            f.write(struct.pack("<I", len(payload)))
            f.write(bytes(payload))


def _read_results(path: Path, n: int) -> list[dict]:
    """Parse the harness receipt. A short header ends the list; a short payload
    is recorded as ``truncated``. The caller compares the record count with
    the requested cases and never grades a partial receipt (WP-38)."""
    out = []
    with open(path, "rb") as f:
        for _ in range(n):
            hdr = f.read(12)
            if len(hdr) < 12:
                break
            status, cycles, outlen = struct.unpack("<III", hdr)
            data = f.read(outlen)
            out.append({"status": status, "cycles": cycles, "bytes": data,
                        "truncated": len(data) < outlen})
            if len(data) < outlen:
                break
    return out


def _golden_bytes(expected: Any) -> bytes | None:
    """Coerce a golden return value to comparable bytes, using the SAME
    conventions map_stimulus uses on the input side (numpy arrays -- the
    natural return type for the image IPs this tier targets -- ravel to
    masked bytes). None when the shape is not byte-comparable, which is an
    honest skip at the call site, never a divergence against 0 bytes."""
    if isinstance(expected, (bytes, bytearray)):
        return bytes(expected)
    try:
        import numpy as _np

        return bytes(int(v) & 0xFF
                     for v in _np.asarray(expected).ravel().tolist())
    except Exception:  # noqa: BLE001
        pass
    try:
        return bytes(int(v) & 0xFF for v in expected)
    except Exception:  # noqa: BLE001
        return None


def _first_diff(a: bytes, b: bytes) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else -1


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

def run_acceptance_dv(project_root: str, top_rtl: str,
                      block_rtls: Any = None) -> dict:
    """Returns {"passed", "skipped", "reason", "cases": [...], "violations"}.

    ``cases`` rows: {name, criterion, ok, cycles, rtl_bytes, golden_bytes,
    first_divergence, fidelity}.
    """
    if not acceptance_dv_enabled():
        return _skip("CORESMITH_ACCEPTANCE_DV=0")
    if not shutil.which("verilator"):
        return _skip("verilator not on PATH")

    from orchestrator.architecture.composition import (
        _run_reference,
        resolve_reference_entrypoint,
        resolve_reference_implementation,
    )
    from orchestrator.architecture.fidelity import (
        compute_fidelity_derate,
        fidelity_gate_enabled,
        resolve_fidelity_metric,
    )
    from orchestrator.architecture.reference_oracle import (
        _acceptance_stimulus_path,
        _import_module_from_path,
        _load_reference_module,
    )

    art = _acceptance_stimulus_path(project_root)
    if not art:
        return _skip("no acceptance stimulus artifact (FRD mandate: "
                     "inputs/acceptance_stimulus.py)")
    try:
        art_mod = _import_module_from_path(Path(art), "_cs_acceptance_rtl")
    except Exception as exc:  # noqa: BLE001
        return _skip(f"acceptance artifact failed to import: {exc}")
    raw_cases = getattr(art_mod, "cases", None)
    if not raw_cases:
        stim = getattr(art_mod, "stimulus", None)
        if stim is None:
            return _skip("acceptance artifact exposes neither cases nor stimulus")
        raw_cases = [("acceptance", stim)]
    # WP-38: the case list is taken whole or not at all -- never truncated
    # silently -- and case names must be unique so a receipt row maps to
    # exactly one requested case.
    raw_cases = list(raw_cases)
    _max_cases = int(os.environ.get("CORESMITH_ACCEPTANCE_MAX_CASES", "64") or 64)
    if len(raw_cases) > _max_cases:
        return _incomplete(f"{len(raw_cases)} acceptance cases exceed "
                           f"CORESMITH_ACCEPTANCE_MAX_CASES={_max_cases}",
                           "adapter_defect")
    _names = [str(n) for n, _ in raw_cases]
    if len(set(_names)) != len(_names):
        return _incomplete("duplicate acceptance case names: "
                           + ", ".join(sorted({n for n in _names if _names.count(n) > 1})),
                           "adapter_defect")

    ref_path = resolve_reference_implementation(project_root)
    if not ref_path:
        return _skip("no golden reference resolvable")
    try:
        ref_module = _load_reference_module(ref_path)
        entry_callable, entry_name = resolve_reference_entrypoint(
            project_root, ref_module)
    except Exception as exc:  # noqa: BLE001
        return _skip(f"reference not loadable: {exc}")
    if entry_callable is None:
        return _skip("no callable reference entry")

    from orchestrator.harness.top_module import CandidateError, candidate_for_inputs
    try:
        rec = candidate_for_inputs(project_root, top_rtl, block_rtls)
        if rec["defines"] != "none" or rec["parameters"] != "none":
            raise CandidateError("Native acceptance does not support the candidate configuration")
    except CandidateError as exc:
        return _incomplete(str(exc), exc.kind)
    top_p = Path(rec["top_rtl_path"])
    rtl_text = top_p.read_text(encoding="utf-8")
    top_module = rec["top_module"]
    mname = re.search(rf"\bmodule\s+{re.escape(top_module)}\b", rtl_text)
    if mname is None:
        return _incomplete("Recorded top is missing", "candidate_mismatch")
    # dv-hardening-23 (armD driver-found, defect #9): scope port discovery to
    # the TOP module span. Integration chip-tops carry helper modules in the
    # same file (e.g. rst_sync_2ff); their inputs leaked into the sideband map
    # and the generated harness drove nonexistent top pins (rst_n_in) -->
    # C++ compile failure --> honest-skip at the acceptance gate.
    _end = rtl_text.find("endmodule", mname.start())
    _top_span = rtl_text[mname.start():(_end if _end != -1 else len(rtl_text))]
    contract = classify_contract(discover_ports(_top_span))
    if contract is None:
        return _skip("chip top is not the framed s_axis/m_axis + sideband "
                     "shape this harness covers")

    # Map every case; skip-honest if any is unmappable.
    sb_order = sorted(contract["sidebands"])
    mapped_cases: list[tuple[str, dict]] = []
    for name, stim in raw_cases:
        m = map_stimulus(stim, contract, mapping=getattr(art_mod, "AXIS_MAPPING", None))
        if m is None:
            _why = "missing or unsupported AXIS_MAPPING (payload, widths, packing, byte order, sidebands)"
            return _incomplete(f"acceptance case {name!r} not mappable onto the "
                               f"chip contract (sidebands={sb_order}): {_why}",
                               "oracle_incomplete")
        mapped_cases.append((str(name), m))

    workdir = Path(tempfile.mkdtemp(prefix="cs_acceptance_dv_"))
    harness = workdir / "sim_main.cpp"
    harness.write_text(generate_harness(contract, top_module, sb_order))

    sources = rec["sources"]
    candidate_sha = rec["candidate_sha"]
    binp = _build(workdir, top_module, sources, harness)
    if not binp:
        return _incomplete("native harness build failed (see log)", "infrastructure_error")

    inp, outp = workdir / "in.bin", workdir / "out.bin"
    _pack_cases(mapped_cases, sb_order, inp)

    run_to = int(float(os.environ.get(
        "CORESMITH_ACCEPTANCE_DV_RUN_TIMEOUT_S", "1800") or 1800))
    try:
        # WP-18: run from the PROJECT ROOT so project-relative $readmemh
        # images (inputs/rom_images/*.memh) resolve -- from any other cwd the
        # ROMs load empty and a correct chip stalls after its first frame.
        r = run_process([binp, str(inp), str(outp)], capture_output=True,
                           text=True, timeout=run_to, cwd=str(project_root))
    except subprocess.TimeoutExpired:
        return {"passed": False, "skipped": False,
                "oracle_incomplete": True, "kind": "infrastructure_error",
                "reason": f"native acceptance run exceeded {run_to}s",
                "cases": [], "violations": [{
                    "type": "acceptance_dv_failure",
                    "criterion": "acceptance_dv_timeout",
                    "suggested_fix": "raise CORESMITH_ACCEPTANCE_DV_RUN_TIMEOUT_S "
                                     "or reduce acceptance case count",
                }]}
    if r.returncode != 0:
        return _incomplete(f"native harness exited rc={r.returncode}: "
                           f"{(r.stderr or '')[-400:]}", "infrastructure_error")
    results = _read_results(outp, len(mapped_cases))
    # WP-38: a receipt is graded only when it is COMPLETE. One record for two
    # requested cases used to pass (the zip below dropped the missing case).
    _trunc = any(_r.get("truncated") for _r in results)
    if len(results) != len(mapped_cases) or _trunc:
        return _incomplete(
            f"receipt has {len(results)} complete case record(s) for "
            f"{len(mapped_cases)} requested case(s)"
            + (" (truncated payload)" if _trunc else ""), "oracle_incomplete")
    try:
        _keep = Path(project_root) / ".coresmith" / "acceptance_dv" / candidate_sha[:12]
        _keep.mkdir(parents=True, exist_ok=True)
        for (_n, _m), _r in zip(mapped_cases, results):
            (_keep / f"{_n}.out.bin").write_bytes(bytes(_r["bytes"]))
    except OSError:
        pass

    use_fidelity = (fidelity_gate_enabled()
                    and resolve_fidelity_metric(project_root) is not None)
    # WP-18: a declared accept(expected, observed) predicate (decode + PSNR
    # for a codec, result match for an MCU) is the mission's own judge; byte
    # equality against the golden is only the fallback for bit-exact IPs.
    accept_fn = None
    try:
        from orchestrator.architecture.composition import resolve_functional_acceptance
        accept_fn = resolve_functional_acceptance(project_root)
    except Exception as exc:  # noqa: BLE001
        # WP-38: a declared predicate that fails to load is an adapter defect,
        # never a silent fall-through to a weaker comparison.
        return _incomplete(f"acceptance predicate not loadable: {exc}", "adapter_defect")
    rows, violations = [], []
    for (name, _mapped), res, (rname, rstim) in zip(mapped_cases, results, raw_cases):
        try:
            expected = _run_reference(entry_callable, rstim, reraise=True)
        except Exception as exc:  # noqa: BLE001
            rows.append({"name": name, "ok": None,
                         "note": f"reference not invokable: {exc}"})
            continue
        exp_bytes = _golden_bytes(expected)
        row = {"name": name, "cycles": res["cycles"],
               "rtl_bytes": len(res["bytes"]), "status": res["status"]}
        if res["status"] != 0:
            row["ok"] = False
            row["note"] = "watchdog expired (no completion)"
            violations.append({
                "type": "acceptance_dv_failure",
                "criterion": "acceptance_dv_watchdog",
                "acceptance_case": name,
                "suggested_fix": "RTL never completed the mission-scale case "
                                 "(drain/EOF or stall class at scale).",
            })
        elif accept_fn is not None:
            try:
                ok = bool(_call_accept(
                    accept_fn, exp_bytes if exp_bytes is not None else expected,
                    bytes(res["bytes"]), name, rstim))
                note = ""
            except Exception as exc:  # noqa: BLE001
                ok, note = False, f"acceptance predicate raised: {exc}"
            row["ok"] = ok
            row["criterion"] = "acceptance_predicate"
            if note:
                row["note"] = note
            if not ok:
                violations.append({
                    "type": "acceptance_dv_failure",
                    "criterion": "acceptance_dv_predicate",
                    "acceptance_case": name,
                    "gap_class": "mission",
                    "suggested_fix": (
                        "RTL output fails the declared acceptance predicate at "
                        f"mission scale ({len(res['bytes'])}B streamed"
                        + (f"; {note}" if note else "") + "). Decode/grade the "
                        "captured stream offline (.coresmith/acceptance_dv/) and "
                        "fix the responsible block."
                    ),
                })
        elif use_fidelity:
            fid = compute_fidelity_derate(project_root, expected, res["bytes"])
            row["fidelity"] = fid
            row["ok"] = bool(fid and fid.get("within_budget"))
            if not row["ok"]:
                violations.append({
                    "type": "acceptance_dv_failure",
                    "criterion": "acceptance_dv_below_budget",
                    "acceptance_case": name,
                    "fidelity": fid,
                    "gap_class": "block_math",
                    "suggested_fix": (
                        "The REAL RTL is below the declared fidelity budget "
                        "on a mission-scale acceptance case. Compare against "
                        "the Full Model DV verdict: model-pass + RTL-fail "
                        "here = transcription divergence at scale; both-fail "
                        "= model content math."
                    ),
                })
        elif exp_bytes is None:
            # Unsupported golden shape: leave the row UNJUDGED (-> honest skip
            # below) instead of diverging it against a 0-byte reference.
            row["ok"] = None
            row["note"] = "golden output not byte-comparable"
        else:
            ok = bytes(res["bytes"]) == exp_bytes
            row["ok"] = ok
            if not ok:
                row["first_divergence"] = (
                    _first_diff(bytes(res["bytes"]), exp_bytes or b""))
                row["golden_bytes"] = len(exp_bytes or b"")
                violations.append({
                    "type": "acceptance_dv_failure",
                    "criterion": "acceptance_dv_divergence",
                    "acceptance_case": name,
                    "first_divergence_offset": row["first_divergence"],
                    "gap_class": "block_math",
                    "suggested_fix": (
                        "RTL output diverges from the GOLDEN at mission scale "
                        f"(first diff offset {row['first_divergence']}, "
                        f"{len(res['bytes'])}B vs {len(exp_bytes or b'')}B)."
                    ),
                })
        rows.append(row)

    # HONEST SKIPS, NEVER A FALSE PASS: rows the golden could not judge (the
    # reference raised, or its output is not byte-comparable) used to be
    # FILTERED OUT of the verdict, so a sweep where every case was ungradable
    # reported PASSED without ever comparing RTL to anything.
    graded = [r for r in rows if r.get("ok") is not None]
    ungraded = [r for r in rows if r.get("ok") is None]
    passed = (bool(graded) and not ungraded and not violations
              and all(r["ok"] for r in graded))
    skipped = bool(ungraded) and not violations
    reason = f"{len(rows)} acceptance case(s), {len(violations)} violation(s)"
    if skipped:
        reason = (f"{len(ungraded)}/{len(rows)} acceptance case(s) not "
                  "gradable against the golden: "
                  + "; ".join(str(r.get("note", "")) for r in ungraded[:3]))
    summary = {"passed": passed, "skipped": skipped,
               "reason": reason,
               "cases": rows, "violations": violations,
               # WP-38: identity + completeness receipt
               "candidate_sha": candidate_sha,
               "sources": sources,
               "captured_dir": str(_keep),
               "requested_cases": len(mapped_cases),
               "completed_cases": len(results)}
    try:
        (_keep / "receipt.json").write_text(json.dumps(summary, default=str, indent=1))
    except OSError:
        pass
    try:
        out_json = Path(project_root) / ".coresmith" / "acceptance_dv.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(
            {k: v for k, v in summary.items() if k != "violations"}
            | {"violations": violations}, default=str, indent=1))
    except Exception:  # noqa: BLE001
        pass
    logger.info("acceptance dv: %s (%s)",
                "PASSED" if passed else ("SKIPPED" if skipped else "FAILED"),
                summary["reason"])
    return summary
