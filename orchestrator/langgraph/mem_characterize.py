# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""PDK memory-characterization agent.

Sweeps memory implementations (flop / registered-flop / SRAM macro) across a
grid of (width, depth, ports) geometries, measures **real** PPA (area + Fmax)
with yosys + OpenROAD-STA (macro Fmax/area read from the macro's signed-off
``.lib``/``.lef``), fits a small predictive model, and exposes a query API that
the frontend gate / uArch agent can consult to decide flop-vs-macro and predict
post-synth PPA *at the uArch stage* -- before a wrong choice becomes a
sub-10 MHz combinational read path.

The problem this solves (observed on a real chip run): shallow/wide "SRAMs"
(8x640, 256x16, 161x16) fell below CoreSmith's bit/depth macro threshold and
synthesized into flop-based combinational read muxes -> ~250 ns read paths
(<10 MHz Fmax) + huge-fanout routing congestion, while a deep 80 Kbit memory
composed cleanly into a stock SRAM macro. The frontend gates on bits/FF, which
cannot see that cliff. This module makes Fmax/area/feasibility *predictable*.

Design notes
------------
* Every PPA number is REAL: flop area/FF/Fmax come from a yosys sky130 synth +
  an OpenROAD ideal-clock STA sweep; macro area/Fmax come from reading the
  macro's ``.lef`` SIZE and the ``clk->dout`` access-time arc in its ``.lib``.
  A tool failure is recorded honestly (``error`` field), never faked.
* Per-point work is fast (synth + a few STA evals, seconds each). No full P&R
  in the sweep. Routability risk is a heuristic derived from the flop read-mux
  fan-out (depth) since a deep combinational read mux is exactly what congests.
* Generic: the grid + PDK are inputs; nothing is sky130-specific beyond the
  PDK paths resolved through ``backend_helpers`` (LIBERTY/TECH_LEF/...).
* Reuses engine helpers: ``openram_gen.ensure_macro/find_exact/plan_composition``,
  ``sram_wrapper.wrapper_lib_path``, ``backend_helpers`` PDK consts + the
  OpenROAD binary, ``macro_registry`` for macro ``.lib``/``.lef`` resolution.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Engine helper imports (degrade gracefully if a dependency is missing)
# ---------------------------------------------------------------------------
from orchestrator.langgraph.sram_wrapper import wrapper_lib_path

try:
    from orchestrator.langgraph.backend_helpers import (
        LIBERTY,
        OPENROAD_BIN,
        TECH_LEF,
        CELL_LEF,
    )
except Exception:  # pragma: no cover - only if PDK consts unresolvable
    LIBERTY = TECH_LEF = CELL_LEF = Path("/nonexistent")
    OPENROAD_BIN = "openroad"


def _resolve_openroad() -> str:
    """Real OpenROAD binary for STA.

    backend_helpers resolves OPENROAD_BIN to a ``nix develop`` wrapper script by
    default; on a host without nix that fails (exec: nix: not found). Prefer an
    explicit env var, then a real ``openroad`` on PATH, then the known local
    build, and only fall back to the (possibly-wrapper) backend const.
    """
    import shutil as _sh
    env = os.environ.get("CORESMITH_BACKEND_OPENROAD", "").strip()
    if env and Path(env).exists():
        return env
    on_path = _sh.which("openroad")
    if on_path:
        return on_path
    local = Path.home() / "openroad-src" / "build" / "bin" / "openroad"
    if local.exists():
        return str(local)
    return OPENROAD_BIN


OPENROAD_BIN = _resolve_openroad()

# ---------------------------------------------------------------------------
# Constants / cache location
# ---------------------------------------------------------------------------

CACHE_DIR = Path(os.environ.get(
    "CORESMITH_MEM_CHAR_DIR",
    str(Path.home() / ".coresmith" / "mem_char"),
))

IMPLS = ("flop", "registered_flop", "macro")

# Default sweep grid (small + log-ish). The 6 real geometries from the failing
# run are always appended (see REAL_GEOMS).
DEFAULT_WIDTHS = (8, 16, 32, 64)
DEFAULT_DEPTHS = (16, 64, 256, 1024, 4096)
DEFAULT_PORTS = ("1rw", "1rw1r")

# The six real geometries from the chip run that exposed the flop-memory cliff.
# Tuples are (ports, WIDTH, DEPTH), matching the run's reported spec exactly:
#   (1rw,256,16),(1rw1r,9,160),(1rw1r,8,640),(1rw1r,8,256),(1rw1r,8,10240),
#   (1rw1r,161,16). Several are narrow+DEEP (depth >> width), i.e. the
#   should-be-SRAM memories that flopped into deep N:1 combinational read muxes.
REAL_GEOMS: tuple[tuple[str, int, int], ...] = (
    ("1rw",  256, 16),     # 256 wide, 16 deep (shallow-wide)
    ("1rw1r",  9, 160),    # 9 wide, 160 deep
    ("1rw1r",  8, 640),    # 8 wide, 640 deep  -- deep narrow read mux
    ("1rw1r",  8, 256),    # 8 wide, 256 deep
    ("1rw1r",  8, 10240),  # 8 wide, 10240 deep -- the marquee flop-cliff failure
    ("1rw1r", 161, 16),    # 161 wide, 16 deep (shallow-wide)
)

# Deterministic v0 rule defaults (used with NO model).
D_CRIT_DEFAULT = 256          # depth above which flops stop being viable
WIDE_SHALLOW_DEPTH = 32       # depth below which a wide array is "register file"
WIDE_SHALLOW_WIDTH = 128      # width above which a shallow array is "too wide"

# STA / synth timeouts (seconds) -- per point, keep fast. Deep flop geometries
# (the very failure class we want to catch) are slow-to-elaborate by nature; a
# timeout there is itself a "do not flop this" signal, recorded honestly.
SYNTH_TIMEOUT_S = int(os.environ.get("CORESMITH_MEM_CHAR_SYNTH_TIMEOUT", "240"))
STA_TIMEOUT_S = int(os.environ.get("CORESMITH_MEM_CHAR_STA_TIMEOUT", "180"))


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class MemPoint:
    """One characterized (geometry x implementation) data point."""

    ports: str
    width: int
    depth: int
    impl: str                       # flop | registered_flop | macro
    bits: int = 0
    area_um2: Optional[float] = None
    fmax_mhz: Optional[float] = None
    ff: Optional[int] = None        # flip-flop count (flop impls only)
    cell_count: Optional[int] = None
    macro_feasible: Optional[bool] = None
    macro_impl: str = ""            # exact | compose | openram | "" (n/a)
    routability_risk: str = ""      # low | medium | high
    notes: str = ""
    error: str = ""

    def __post_init__(self) -> None:
        if not self.bits:
            self.bits = self.width * self.depth


# ---------------------------------------------------------------------------
# PDK fingerprint (so the cache key changes when the PDK changes)
# ---------------------------------------------------------------------------

def pdk_hash(pdk: Optional[dict] = None) -> str:
    """Stable short hash of the PDK identity used for this characterization."""
    parts = [str(LIBERTY), str(TECH_LEF)]
    try:
        parts.append(str(int(Path(LIBERTY).stat().st_size)))
    except OSError:
        pass
    if pdk:
        parts.append(json.dumps(pdk, sort_keys=True))
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# RTL emission: a tiny top instantiating the cs_sram wrapper (behavioral)
# ---------------------------------------------------------------------------

def _emit_flop_top(ports: str, width: int, depth: int, comb_read: bool) -> str:
    """Emit a small top instantiating the behavioral cs_sram wrapper.

    ``comb_read=False`` (registered_flop): use the wrapper's registered read as
    the realistic flop case (clk->mem-mux->FF). ``comb_read=True`` (flop):
    additionally route the read through a *combinational* output assign that
    re-reads the memory array, modelling the N:1 comb read mux that the failing
    run actually generated -- which is what blows the read path delay up.

    The wrapper is read behaviorally (NO CORESMITH_SRAM_SYNTH) so its internal
    ``reg mem[]`` is present and synthesizes to flops, exactly as in the run.
    """
    aw = max(1, math.ceil(math.log2(max(2, depth))))
    if ports == "1rw1r":
        inst = (
            f"  cs_sram_1rw1r #(.WIDTH({width}), .DEPTH({depth})) u_mem (\n"
            "    .clk(clk), .ce0(ce0), .we0(we0), .addr0(addr0),\n"
            "    .wdata0(wdata0), .rdata0(rdata0),\n"
            "    .ce1(ce1), .addr1(addr1), .rdata1(rdata1)\n"
            "  );\n"
        )
        ports_decl = (
            "  input wire clk,\n"
            "  input wire ce0, input wire we0,\n"
            f"  input wire [{aw-1}:0] addr0,\n"
            f"  input wire [{width-1}:0] wdata0,\n"
            f"  output wire [{width-1}:0] rdata0,\n"
            "  input wire ce1,\n"
            f"  input wire [{aw-1}:0] addr1,\n"
            f"  output wire [{width-1}:0] rdata1\n"
        )
        # registered read straight out of the wrapper
        wires = (
            f"  wire [{width-1}:0] rdata0; wire [{width-1}:0] rdata1;\n"
            if not comb_read else
            f"  wire [{width-1}:0] q0; wire [{width-1}:0] q1;\n"
        )
        # In comb_read variant we expose the wrapper's registered q AND a comb
        # re-read mux feeding the output -- modelling a comb read path on top of
        # the flop array (the failure mode). Implemented via a shadow reg array
        # mirrored from writes; to keep it a single source of truth we instead
        # bury a comb mux in a helper module so synth sees a real N:1 mux.
        if comb_read:
            inst = inst.replace(".rdata0(rdata0)", ".rdata0(q0)") \
                       .replace(".rdata1(rdata1)", ".rdata1(q1)")
            body = (
                f"{wires}{inst}"
                "  // comb read-mux model: shadow array + combinational select\n"
                f"  reg [{width-1}:0] shadow [0:{depth-1}];\n"
                "  always @(posedge clk) if (ce0 && we0) shadow[addr0] <= wdata0;\n"
                f"  assign rdata0 = shadow[addr0];\n"  # combinational N:1 read mux
                f"  assign rdata1 = shadow[addr1];\n"
            )
            ports_decl = ports_decl.replace("output wire", "output wire")
        else:
            body = f"{wires}{inst}"
    else:  # 1rw
        inst = (
            f"  cs_sram_1rw #(.WIDTH({width}), .DEPTH({depth})) u_mem (\n"
            "    .clk(clk), .ce(ce), .we(we), .addr(addr),\n"
            "    .wdata(wdata), .rdata(rdata)\n"
            "  );\n"
        )
        ports_decl = (
            "  input wire clk,\n"
            "  input wire ce, input wire we,\n"
            f"  input wire [{aw-1}:0] addr,\n"
            f"  input wire [{width-1}:0] wdata,\n"
            f"  output wire [{width-1}:0] rdata\n"
        )
        if comb_read:
            inst = inst.replace(".rdata(rdata)", ".rdata(q)")
            body = (
                f"  wire [{width-1}:0] q;\n{inst}"
                f"  reg [{width-1}:0] shadow [0:{depth-1}];\n"
                "  always @(posedge clk) if (ce && we) shadow[addr] <= wdata;\n"
                f"  assign rdata = shadow[addr];\n"  # combinational N:1 read mux
            )
        else:
            body = f"  wire [{width-1}:0] rdata;\n{inst}"

    return (
        "// auto-generated mem-characterization top\n"
        "module mem_char_top (\n"
        f"{ports_decl}"
        ");\n"
        f"{body}"
        "endmodule\n"
    )


# ---------------------------------------------------------------------------
# yosys flop synthesis (real sky130 mapping) -> area, FF, cell count
# ---------------------------------------------------------------------------

_YOSYS_AREA_RE = re.compile(r"Chip area for module.*?:\s*([\d.]+)")
_YOSYS_CELLS_RE = re.compile(r"Number of cells:\s*(\d+)")
# A yosys `stat -liberty` cell-table row: "  <count> <area> <cellname>"
# (area may be "2.47E+05" or "938.4"); name is the trailing std-cell token.
_STAT_ROW_RE = re.compile(
    r"^\s*(\d+)\s+[\d.eE+-]+\s+(sky130_fd_sc_hd__\w+)\s*$")
# sky130 HD sequential cell families (any flip-flop / latch variant).
_DFF_NAME_RE = re.compile(r"sky130_fd_sc_hd__(?:s?e?df|dlxtp|dlrtp|dlygate)\w*")


def _parse_stat_table(out: str) -> tuple[int, int]:
    """Return (total_cells, ff_count) from a yosys `stat -liberty` dump.

    The liberty stat prints a `<count> <area> <cellname>` table with no
    "Number of cells:" summary line, so we sum the rows ourselves and pick out
    sequential-cell families for the FF count.
    """
    total = 0
    ff = 0
    for line in out.splitlines():
        m = _STAT_ROW_RE.match(line)
        if not m:
            continue
        cnt, name = int(m.group(1)), m.group(2)
        total += cnt
        if _DFF_NAME_RE.fullmatch(name):
            ff += cnt
    return total, ff


def _synth_flop(top_path: str, top: str, liberty: str, yosys_bin: str = "yosys",
                timeout_s: int = SYNTH_TIMEOUT_S) -> dict[str, Any]:
    """yosys synth of the flop top to sky130 std cells. Returns area/ff/cells.

    Writes the mapped netlist next to the source for the STA step.
    """
    netlist = str(Path(top_path).with_suffix(".netlist.v"))
    script = (
        f"read_verilog -sv {wrapper_lib_path()}\n"
        f"read_verilog -sv {top_path}\n"
        f"hierarchy -check -top {top}\n"
        "proc; opt; flatten; opt; memory; opt\n"
        "techmap; opt\n"
        f"dfflibmap -liberty {liberty}\n"
        f"abc -liberty {liberty}\n"
        "clean; opt_clean -purge\n"
        f"stat -liberty {liberty}\n"
        f"write_verilog -noattr {netlist}\n"
    )
    import shutil as _sh
    yosys = _sh.which(yosys_bin)
    if not yosys:
        return {"error": "yosys not found"}
    with tempfile.NamedTemporaryFile("w", suffix=".ys", delete=False) as fh:
        fh.write(script)
        ys = fh.name
    try:
        r = subprocess.run([yosys, "-s", ys], capture_output=True, text=True,
                           timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {"error": f"yosys timed out ({timeout_s}s) -- likely an oversized "
                         "flop array that does not elaborate in bounded time"}
    except OSError as exc:
        return {"error": f"yosys failed: {exc}"}
    finally:
        try:
            Path(ys).unlink()
        except OSError:
            pass
    out = r.stdout + "\n" + r.stderr
    if r.returncode != 0:
        return {"error": f"yosys exit {r.returncode}: {(r.stderr or out)[-300:].strip()}"}
    area = None
    m = _YOSYS_AREA_RE.search(out)
    if m:
        area = float(m.group(1))
    cells, ff = _parse_stat_table(out)
    return {"area_um2": area, "cell_count": cells or None, "ff": ff,
            "netlist": netlist if Path(netlist).exists() else "",
            "stdout": out}


# ---------------------------------------------------------------------------
# OpenROAD ideal-clock STA: worst-path delay -> Fmax in one report
# ---------------------------------------------------------------------------

# "       8.98   data arrival time" -- the worst-path arrival in report_checks.
_ARRIVAL_RE = re.compile(r"([\d.]+)\s+data arrival time")
# "Period" header line not needed; we parse arrival directly.


def _worst_path_delay_ns(netlist: str, top: str, liberty: str,
                         openroad_bin: str = OPENROAD_BIN,
                         timeout_s: int = STA_TIMEOUT_S) -> Optional[float]:
    """Worst-path data-delay (ns), ideal-clock pre-layout, via OpenROAD STA.

    The critical path for a flop memory is the combinational read mux
    (addr -> N:1 mux -> rdata), an input->output path; for a registered read it
    is reg->reg / reg->out. To time BOTH path classes in one report we:
      * define a slow real clock (so reg->reg never bounds the result), and
      * ``set_max_delay`` on input->output paths (so the comb read mux IS timed).
    ``report_checks -path_delay max`` then prints the single worst path's
    ``data arrival time`` = the worst delay through the design. Fmax = 1/delay.

    Reuses OpenROAD's built-in OpenSTA (no standalone ``sta`` on this host) and
    needs no wire RC -- optimistic, but it nails the gross flop read-mux cliff
    (a hundreds-of-ns path shows up plainly). Returns None if STA fails.
    """
    big = 100000.0  # ns -- larger than any path so it never bounds arrival
    script = (
        f"read_lef {TECH_LEF}\n"
        f"read_lef {CELL_LEF}\n"
        f"read_liberty {liberty}\n"
        f"read_verilog {netlist}\n"
        f"link_design {top}\n"
        f"create_clock -name clk -period {big} [get_ports clk]\n"
        f"set_max_delay {big} -from [all_inputs] -to [all_outputs]\n"
        "report_checks -path_delay max -group_path_count 1 "
        "-fields {} -no_line_splits\n"
        "exit\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".tcl", delete=False) as fh:
        fh.write(script)
        tcl = fh.name
    try:
        r = subprocess.run([openroad_bin, "-no_init", "-exit", tcl],
                           capture_output=True, text=True, timeout=timeout_s)
    except (subprocess.TimeoutExpired, OSError):
        return None
    finally:
        try:
            Path(tcl).unlink()
        except OSError:
            pass
    m = _ARRIVAL_RE.search(r.stdout)
    if not m:
        return None
    delay = float(m.group(1))
    return delay if delay > 0 else None


def _find_fmax(netlist: str, top: str, liberty: str,
               timeout_s: int = STA_TIMEOUT_S) -> Optional[float]:
    """Fmax (MHz) = 1000 / worst_path_delay_ns, measured in a single STA report."""
    delay = _worst_path_delay_ns(netlist, top, liberty, timeout_s=timeout_s)
    if delay is None or delay <= 0:
        return None
    return 1000.0 / delay


# ---------------------------------------------------------------------------
# Macro resolution: read real area + access-time from .lef / .lib
# ---------------------------------------------------------------------------

_LEF_SIZE_RE = re.compile(r"SIZE\s+([\d.]+)\s+BY\s+([\d.]+)", re.IGNORECASE)


def _lef_area_um2(lef_path: str) -> Optional[float]:
    try:
        text = Path(lef_path).read_text(errors="ignore")
    except OSError:
        return None
    m = _LEF_SIZE_RE.search(text)
    if not m:
        return None
    return float(m.group(1)) * float(m.group(2))


def _lib_access_time_ns(lib_path: str) -> Optional[float]:
    """Worst clk->dout access time (ns) from the macro's .lib timing arcs.

    Reads the largest ``cell_rise``/``cell_fall`` value among the data-out pin
    arcs related to the clock (rising_edge/falling_edge). This is the real
    registered-read access time the SRAM achieves -- typically ~0.3-0.6 ns,
    vs the hundreds of ns a deep flop comb-read mux takes.
    """
    try:
        text = Path(lib_path).read_text(errors="ignore")
    except OSError:
        return None
    # Pull all cell_rise/cell_fall value tuples in the file; the data-out arcs
    # dominate the max. (Setup/hold are in separate constraint tables with
    # rise_constraint/fall_constraint keys, so they don't pollute this.)
    worst = 0.0
    for m in re.finditer(r"cell_(?:rise|fall)\s*\([^)]*\)\s*\{\s*values\(([^)]*)\)",
                         text, re.DOTALL):
        for tok in re.findall(r"-?\d+\.\d+", m.group(1)):
            worst = max(worst, abs(float(tok)))
    return worst if worst > 0 else None


def _resolve_macro(ports: str, width: int, depth: int) -> MemPoint:
    """Resolve a macro for (ports,width,depth) and read its real area/Fmax.

    Uses ensure_macro (exact -> compose -> OpenRAM). For exact/openram we read
    the macro's own .lef/.lib. For a composition plan we scale the base macro's
    area by the tile count and use the base macro's access time (the read path
    of one tile; bank muxing adds a small select that we surface as a note).
    """
    from orchestrator.langgraph.openram_gen import (
        ensure_macro, CompositionPlan,
    )
    from orchestrator.langgraph.macro_registry import MacroInfo

    pt = MemPoint(ports=ports, width=width, depth=depth, impl="macro")
    try:
        # ensure_macro takes (words=depth, data_bits=width). Do NOT generate
        # with OpenRAM inside the sweep by default (slow); allow exact+compose.
        allow_gen = os.environ.get("CORESMITH_MEM_CHAR_OPENRAM", "").strip() in {
            "1", "true", "yes", "on"}
        res = ensure_macro(words=depth, data_bits=width, allow_generate=allow_gen)
    except Exception as exc:
        pt.macro_feasible = False
        pt.error = f"ensure_macro raised: {exc}"
        return pt

    if res is None:
        pt.macro_feasible = False
        pt.macro_impl = ""
        pt.notes = ("no stock macro, not composable, OpenRAM disabled/failed -- "
                    "reshape memory (too wide/shallow for a macro; split banks "
                    "or register the read)")
        return pt

    if isinstance(res, CompositionPlan):
        base = res.base
        tiles = res.tiles_wide * res.tiles_deep
        area = (_lef_area_um2(base.lef) or base.area_um2 or 0.0) * tiles
        acc = _lib_access_time_ns(base.lib)
        pt.macro_feasible = True
        pt.macro_impl = "compose"
        pt.area_um2 = area if area > 0 else None
        if acc:
            # bank select mux adds ~one extra mux level; small penalty
            pt.fmax_mhz = 1000.0 / (acc + 0.10)
        pt.notes = res.describe()
        return pt

    if isinstance(res, MacroInfo):
        area = _lef_area_um2(res.lef) or res.area_um2 or None
        acc = _lib_access_time_ns(res.lib)
        pt.macro_feasible = True
        pt.macro_impl = "openram" if res.name.startswith("sram_") else "exact"
        pt.area_um2 = area
        if acc:
            pt.fmax_mhz = 1000.0 / acc
        pt.notes = f"macro {res.name} ({res.words}x{res.data_bits})"
        if pt.fmax_mhz is None:
            pt.error = "macro resolved but .lib access-time unreadable"
        return pt

    pt.macro_feasible = False
    pt.error = f"unexpected ensure_macro result type {type(res)}"
    return pt


# ---------------------------------------------------------------------------
# Routability-risk heuristic (no P&R in the sweep)
# ---------------------------------------------------------------------------

def _routability_risk(impl: str, depth: int, fmax_mhz: Optional[float]) -> str:
    """Heuristic risk from the flop read-mux fan-out (depth) + measured Fmax.

    A deep combinational/registered read mux is an N:1 mux whose fan-out grows
    with depth; that is exactly what congested the failing run. Macros isolate
    the array behind a fixed-pin abstract -> low risk.
    """
    if impl == "macro":
        return "low"
    if depth >= 1024:
        return "high"
    if depth >= 256:
        return "medium"
    if fmax_mhz is not None and fmax_mhz < 50:
        return "high"
    return "low"


# ---------------------------------------------------------------------------
# Single-point characterization
# ---------------------------------------------------------------------------

def characterize_point(ports: str, width: int, depth: int, impl: str,
                       liberty: str = "") -> MemPoint:
    """Characterize one (geometry x impl). Real measurements only."""
    lib = liberty or str(LIBERTY)
    if impl == "macro":
        pt = _resolve_macro(ports, width, depth)
        pt.routability_risk = _routability_risk("macro", depth, pt.fmax_mhz)
        return pt

    comb = (impl == "flop")  # flop=comb read mux model; registered_flop=registered
    pt = MemPoint(ports=ports, width=width, depth=depth, impl=impl)
    # A comb-read N:1 mux over a very deep array is the exact failure class; it
    # is also expensive to elaborate. Give it a SHORTER synth budget so a deep
    # one records an honest timeout fast (the timeout IS the "do not flop this"
    # signal) instead of stalling the sweep for minutes. Registered/macro paths
    # keep the full budget.
    syn_timeout = SYNTH_TIMEOUT_S
    deep_budget = int(os.environ.get("CORESMITH_MEM_CHAR_DEEP_FLOP_TIMEOUT", "90"))
    if (comb and depth >= 1024) or depth >= 4096:
        syn_timeout = min(SYNTH_TIMEOUT_S, deep_budget)
    with tempfile.TemporaryDirectory() as td:
        top_path = str(Path(td) / "mem_char_top.v")
        Path(top_path).write_text(_emit_flop_top(ports, width, depth, comb))
        syn = _synth_flop(top_path, "mem_char_top", lib, timeout_s=syn_timeout)
        if syn.get("error"):
            pt.error = syn["error"]
            if "timed out" in syn["error"] and depth >= 1024:
                pt.notes = (f"{impl} over {depth} words did not elaborate "
                            f"in {syn_timeout}s -- infeasible as flops; use a macro")
            pt.routability_risk = _routability_risk(impl, depth, None)
            return pt
        pt.area_um2 = syn.get("area_um2")
        pt.ff = syn.get("ff")
        pt.cell_count = syn.get("cell_count")
        netlist = syn.get("netlist")
        if netlist:
            # copy netlist out of the tmpdir before it's removed for STA
            persistent = str(Path(td) / "nl.v")
            Path(persistent).write_text(Path(netlist).read_text())
            sta_to = STA_TIMEOUT_S if depth < 1024 else min(STA_TIMEOUT_S, 90)
            fmax = _find_fmax(persistent, "mem_char_top", lib, timeout_s=sta_to)
            pt.fmax_mhz = fmax
            if fmax is None:
                pt.notes = (f"STA did not return a path in {sta_to}s "
                            "(deep read mux) -- area+FF measured, Fmax infeasible")
        else:
            pt.notes = "no netlist emitted -- area from stat only"
    pt.routability_risk = _routability_risk(impl, depth, pt.fmax_mhz)
    return pt


# ---------------------------------------------------------------------------
# Sweep harness
# ---------------------------------------------------------------------------

def build_grid(widths=DEFAULT_WIDTHS, depths=DEFAULT_DEPTHS,
               ports=DEFAULT_PORTS,
               include_real: bool = True) -> list[tuple[str, int, int]]:
    """Build the (ports,width,depth) grid + the 6 real geometries (dedup)."""
    grid: list[tuple[str, int, int]] = []
    for p in ports:
        for w in widths:
            for d in depths:
                grid.append((p, w, d))
    if include_real:
        grid.extend(REAL_GEOMS)
    # dedup preserving order
    seen: set[tuple[str, int, int]] = set()
    out: list[tuple[str, int, int]] = []
    for g in grid:
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out


def characterize_memories(grid: Optional[list[tuple[str, int, int]]] = None,
                          pdk: Optional[dict] = None,
                          impls: tuple[str, ...] = IMPLS,
                          liberty: str = "",
                          verbose: bool = True) -> list[MemPoint]:
    """Sweep the grid x impls, measure real PPA, cache the table.

    Returns the list of MemPoint rows and writes them to
    ``<CACHE_DIR>/<pdk_hash>.json``.
    """
    if grid is None:
        grid = build_grid()
    lib = liberty or str(LIBERTY)
    rows: list[MemPoint] = []
    total = len(grid) * len(impls)
    n = 0
    for (ports, width, depth) in grid:
        for impl in impls:
            n += 1
            if verbose:
                print(f"[{n}/{total}] {impl:16s} {ports:6s} {width}x{depth} ...",
                      flush=True)
            pt = characterize_point(ports, width, depth, impl, liberty=lib)
            if verbose:
                a = f"{pt.area_um2:,.0f}" if pt.area_um2 else "-"
                f = f"{pt.fmax_mhz:,.0f}" if pt.fmax_mhz else "-"
                tag = pt.error or pt.macro_impl or pt.routability_risk
                print(f"        area={a} um2  Fmax={f} MHz  ff={pt.ff}  {tag}",
                      flush=True)
            rows.append(pt)
    save_table(rows, pdk)
    return rows


def cache_path(pdk: Optional[dict] = None) -> Path:
    return CACHE_DIR / f"{pdk_hash(pdk)}.json"


def save_table(rows: list[MemPoint], pdk: Optional[dict] = None) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = cache_path(pdk)
    payload = {
        "pdk_hash": pdk_hash(pdk),
        "liberty": str(LIBERTY),
        "rows": [asdict(r) for r in rows],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_table(pdk: Optional[dict] = None) -> list[MemPoint]:
    path = cache_path(pdk)
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return [MemPoint(**{k: v for k, v in r.items()
                        if k in MemPoint.__dataclass_fields__})
            for r in data.get("rows", [])]


# ---------------------------------------------------------------------------
# Model fit
# ---------------------------------------------------------------------------

def _features(ports: str, width: int, depth: int, impl: str) -> list[float]:
    bits = width * depth
    return [
        float(width),
        float(depth),
        float(bits),
        math.log2(max(2, depth)),
        math.log2(max(2, width)),
        1.0 if ports == "1rw1r" else 0.0,
        1.0 if impl == "flop" else 0.0,
        1.0 if impl == "registered_flop" else 0.0,
        1.0 if impl == "macro" else 0.0,
    ]


FEATURE_NAMES = ["width", "depth", "bits", "log_depth", "log_width",
                 "ports_1rw1r", "impl_flop", "impl_regflop", "impl_macro"]


class MemModel:
    """Predicts (area_um2, fmax_mhz) from geometry+impl.

    Prefers sklearn GradientBoostingRegressor; falls back to a transparent
    nearest-neighbour + monotone interpolation if sklearn is unavailable.
    """

    def __init__(self) -> None:
        self.kind = "none"
        self._area_model = None
        self._fmax_model = None
        self._rows: list[MemPoint] = []
        self.area_importances: dict[str, float] = {}
        self.fmax_importances: dict[str, float] = {}

    def fit(self, rows: list[MemPoint]) -> "MemModel":
        # only rows with a usable target
        self._rows = [r for r in rows if r.area_um2 or r.fmax_mhz]
        X, ya, yf, mask_a, mask_f = [], [], [], [], []
        for r in self._rows:
            X.append(_features(r.ports, r.width, r.depth, r.impl))
            ya.append(r.area_um2 if r.area_um2 else 0.0)
            mask_a.append(r.area_um2 is not None and r.area_um2 > 0)
            yf.append(r.fmax_mhz if r.fmax_mhz else 0.0)
            mask_f.append(r.fmax_mhz is not None and r.fmax_mhz > 0)
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            import numpy as np
            Xn = np.array(X)
            self.kind = "gbt"
            if sum(mask_a) >= 4:
                self._area_model = GradientBoostingRegressor(
                    n_estimators=200, max_depth=3, learning_rate=0.05)
                self._area_model.fit(Xn[mask_a], np.array(ya)[mask_a])
                self.area_importances = dict(zip(
                    FEATURE_NAMES, self._area_model.feature_importances_))
            if sum(mask_f) >= 4:
                self._fmax_model = GradientBoostingRegressor(
                    n_estimators=200, max_depth=3, learning_rate=0.05)
                self._fmax_model.fit(Xn[mask_f], np.array(yf)[mask_f])
                self.fmax_importances = dict(zip(
                    FEATURE_NAMES, self._fmax_model.feature_importances_))
        except Exception:
            self.kind = "knn"  # transparent fallback
        return self

    # ---- prediction ----
    def _knn_predict(self, ports: str, width: int, depth: int, impl: str,
                     target: str) -> Optional[float]:
        # nearest neighbours in (log_depth, log_width, ports, impl) space among
        # rows with that target present; inverse-distance weighted.
        cand = [r for r in self._rows
                if r.impl == impl and r.ports == ports
                and getattr(r, target) is not None and getattr(r, target) > 0]
        if not cand:
            cand = [r for r in self._rows if r.impl == impl
                    and getattr(r, target) and getattr(r, target) > 0]
        if not cand:
            return None
        ld, lw = math.log2(max(2, depth)), math.log2(max(2, width))
        scored = []
        for r in cand:
            d = (math.log2(max(2, r.depth)) - ld) ** 2 + \
                (math.log2(max(2, r.width)) - lw) ** 2
            scored.append((d, getattr(r, target)))
        scored.sort(key=lambda x: x[0])
        scored = scored[:3]
        if scored[0][0] == 0:
            return scored[0][1]
        wsum = sum(1.0 / (d + 1e-6) for d, _ in scored)
        return sum((1.0 / (d + 1e-6)) * v for d, v in scored) / wsum

    def predict_area(self, ports: str, width: int, depth: int, impl: str) -> Optional[float]:
        if self.kind == "gbt" and self._area_model is not None:
            import numpy as np
            return float(self._area_model.predict(
                np.array([_features(ports, width, depth, impl)]))[0])
        return self._knn_predict(ports, width, depth, impl, "area_um2")

    def predict_fmax(self, ports: str, width: int, depth: int, impl: str) -> Optional[float]:
        if self.kind == "gbt" and self._fmax_model is not None:
            import numpy as np
            return float(self._fmax_model.predict(
                np.array([_features(ports, width, depth, impl)]))[0])
        return self._knn_predict(ports, width, depth, impl, "fmax_mhz")


def fit_model(table: Optional[list[MemPoint]] = None,
              pdk: Optional[dict] = None) -> MemModel:
    rows = table if table is not None else load_table(pdk)
    return MemModel().fit(rows)


# ---------------------------------------------------------------------------
# Deterministic v0 rules (usable with NO model)
# ---------------------------------------------------------------------------

def recommend_impl_rule(width: int, depth: int, ports: str = "1rw",
                        d_crit: int = D_CRIT_DEFAULT) -> dict[str, Any]:
    """Transparent rule-based recommendation (no measurements needed).

    * Registered read is mandatory for any flop memory (never comb read mux).
    * depth > d_crit  => macro (or pipelined), regardless of bits.
    * wide+shallow (can't tile a macro) => reshape / register-read.
    """
    bits = width * depth
    if depth > d_crit:
        return {"recommended_impl": "macro",
                "reason": f"depth {depth} > D_crit {d_crit}: a flop read mux this "
                          "deep is a multi-ns N:1 path -> use an SRAM macro (or "
                          "deeply pipelined register read)."}
    if depth <= WIDE_SHALLOW_DEPTH and width >= WIDE_SHALLOW_WIDTH:
        return {"recommended_impl": "registered_flop",
                "reason": f"wide-shallow ({width}x{depth}, {bits}b): too "
                          "shallow/wide for a stock SRAM macro -- keep as flops "
                          "but the read MUST be registered (no comb mux), or "
                          "reshape (split banks)."}
    return {"recommended_impl": "registered_flop",
            "reason": f"small/shallow ({width}x{depth}, {bits}b): flops are "
                      "cheaper than a macro; registered read is mandatory."}


# ---------------------------------------------------------------------------
# Applicability domain (is-the-query-inside-the-characterized-grid) + the
# out-of-grid analytic extrapolation used instead of the saturated regressor.
#
# The learned model (GBR/kNN) is only trustworthy INSIDE the geometries actually
# characterized in the cache. Outside that box it saturates at the training edge
# -- e.g. a 1.9 Mbit store predicts the same ~2 mm^2 the model learned for its
# largest 64 Kbit grid point (a ~20x under-estimate). These helpers detect the
# out-of-grid case (from the REAL cache rows, not the DEFAULT_* sweep constants,
# which differ per box) and price/clock it analytically instead.
# ---------------------------------------------------------------------------

def _fallback_um2_per_bit() -> float:
    """Analytic per-bit area fallback (env-tunable), shared with the pricing gate.

    Prefers ``mem_price.flop_um2_per_bit`` (honours ``CORESMITH_FLOP_UM2_PER_BIT``)
    so the extrapolation floor and the mem_price backstop use ONE ruler; degrades
    to the 25 um^2/bit default if mem_price is unimportable.
    """
    try:
        from orchestrator.langgraph.mem_price import flop_um2_per_bit
        return flop_um2_per_bit()
    except Exception:  # noqa: BLE001 - never fail the predictor on an import hiccup
        return 25.0


def grid_bounds(table: list[MemPoint]) -> Optional[tuple[int, int, int, int, int]]:
    """(min_w, max_w, min_d, max_d, max_bits) over rows with a usable measurement.

    Bounds the actual characterized geometries (any row carrying an area OR an
    Fmax). Returns ``None`` when the table has no measured row (nothing to bound).
    """
    usable = [r for r in table if (r.area_um2 or r.fmax_mhz)]
    if not usable:
        return None
    ws = [r.width for r in usable]
    ds = [r.depth for r in usable]
    bits = [r.width * r.depth for r in usable]
    return (min(ws), max(ws), min(ds), max(ds), max(bits))


def is_in_grid(width: int, depth: int, table: list[MemPoint]) -> bool:
    """True iff (width, depth) is inside the CHARACTERIZED grid's bounding box.

    Requires width AND depth within the measured bounds AND the requested
    bit-capacity within the largest measured capacity -- the last clause guards
    the corner where each dimension is individually in-range but their product
    (e.g. 256x1024) exceeds anything ever characterized. Uses the real cache
    rows so it is correct regardless of which grid a given box swept. An empty /
    unmeasured table reads as "not in grid" -> analytic path (never the saturated
    regressor).
    """
    b = grid_bounds(table)
    if b is None:
        return False
    min_w, max_w, min_d, max_d, max_bits = b
    return (min_w <= width <= max_w
            and min_d <= depth <= max_d
            and width * depth <= max_bits)


def _per_bit_area_cost(table: list[MemPoint], impl: str) -> float:
    """Marginal area/bit measured at the LARGEST characterized geometries for `impl`.

    Averages ``area/bits`` over the top quartile (by bit-count, >=1 row) of `impl`
    rows -- i.e. the grid EDGE, where the per-bit cost has stopped shrinking with
    amortized peripherals and best approximates the slope out into the
    extrapolation region. Falls back to the analytic flop-bit cost when the cache
    has no usable row for this impl.
    """
    rows = [r for r in table
            if r.impl == impl and r.area_um2 and (r.width * r.depth) > 0]
    if rows:
        rows.sort(key=lambda r: r.width * r.depth)
        k = max(1, len(rows) // 4)
        edge = rows[-k:]
        costs = [r.area_um2 / (r.width * r.depth) for r in edge]
        return sum(costs) / len(costs)
    return _fallback_um2_per_bit()


def _extrapolate_area(width: int, depth: int, impl: str,
                      table: list[MemPoint]) -> float:
    """Out-of-grid area = (edge per-bit cost) x (requested bits). Anchored at the
    largest characterized geometry's marginal cost, scaled to the query."""
    return _per_bit_area_cost(table, impl) * float(width * depth)


def _worst_in_grid_fmax(table: list[MemPoint], impl: str) -> Optional[float]:
    """Slowest (min) Fmax among characterized rows for `impl`, or None if none.

    The conservative clamp for an out-of-grid Fmax: a memory LARGER than anything
    we measured cannot be FASTER than the slowest instance of that impl we saw, so
    we return that floor rather than the regressor's saturated-optimistic number.
    """
    fs = [r.fmax_mhz for r in table
          if r.impl == impl and r.fmax_mhz and r.fmax_mhz > 0]
    return min(fs) if fs else None


def _extrapolate_fmax(width: int, depth: int, impl: str,
                      table: list[MemPoint]) -> Optional[float]:
    """Out-of-grid Fmax: clamp to the worst (slowest) in-grid Fmax for this impl.

    We deliberately pick the SIMPLER defensible option (clamp-to-worst) over a
    mux-depth degrade: for a deep flop read mux the honest engineering signal is
    "at best as slow as the slowest thing we characterized, and flagged
    out-of-grid" -- which already fails any real target and steers the choice to
    a macro. Returns None if this impl has no measured Fmax to clamp to.
    """
    return _worst_in_grid_fmax(table, impl)


# ---------------------------------------------------------------------------
# Query API
# ---------------------------------------------------------------------------

def predict_mem(width: int, depth: int, ports: str = "1rw",
                target_mhz: float = 100.0,
                model: Optional[MemModel] = None,
                pdk: Optional[dict] = None) -> dict[str, Any]:
    """Recommend an impl + predict PPA for a memory geometry.

    Returns ``{recommended_impl, pred_area_um2, pred_fmax_mhz, macro_feasible,
    in_grid, estimate_source, reason, candidates, rule_v0}``. Each candidate also
    carries a per-impl ``source``. Recommendation logic:
      * pick the impl that meets target_mhz at least area;
      * if flops can't meet target (deep read mux) -> recommend macro (or
        registered+pipelined);
      * if macro infeasible for the geometry -> flag "reshape memory".

    PPA precedence (why the number is what it is):
      * **macro, deterministic-first** -- for a macro-FEASIBLE geometry the LEF
        banking area and the ``.lib`` access-time are AUTHORITATIVE; the learned
        regressor never overrides them (it saturates far outside the sweep). Order
        is: cached measured row -> live LEF/.lib resolve -> regressor (only to
        fill a hole neither could).
      * **flop / registered_flop, in-grid** -> the regressor, verbatim (unchanged).
      * **flop / registered_flop, out-of-grid** -> analytic extrapolation
        (edge per-bit area cost x bits; Fmax clamped to worst-in-grid), tagged
        ``estimate_source='analytic_extrapolation'`` so the pricing gate is honest
        about the ruler used.
    """
    if model is None:
        model = fit_model(pdk=pdk)

    # macro feasibility from the cached table (measured) when available; else a
    # LIVE resolve whose LEF/.lib numbers we REUSE (deterministic-first routing).
    table = model._rows if model._rows else load_table(pdk)
    macro_row = next((r for r in table if r.impl == "macro"
                      and r.ports == ports and r.width == width
                      and r.depth == depth), None)
    resolved_macro: Optional[MemPoint] = None
    if macro_row is not None:
        macro_feasible = bool(macro_row.macro_feasible)
    else:
        # not in table -> resolve live (exact/compose only, no OpenRAM). Keep the
        # resolved point: its LEF area + .lib Fmax are the authoritative answer.
        try:
            resolved_macro = _resolve_macro(ports, width, depth)
            macro_feasible = resolved_macro.macro_feasible or False
        except Exception:
            macro_feasible = False

    in_grid = is_in_grid(width, depth, table)
    reg_src = "regressor_in_grid" if in_grid else "analytic_extrapolation"

    cands: dict[str, dict[str, Any]] = {}
    for impl in IMPLS:
        if impl == "macro":
            if not macro_feasible:
                continue
            area: Optional[float] = None
            fmax: Optional[float] = None
            src = ""
            # 1) cached measured row (byte-identical to prior behavior)
            if macro_row is not None:
                if macro_row.area_um2:
                    area, src = macro_row.area_um2, "pdk_measured"
                if macro_row.fmax_mhz:
                    fmax, src = macro_row.fmax_mhz, src or "pdk_measured"
            # 2) live LEF/.lib banking arithmetic -- AUTHORITATIVE for a
            #    macro-feasible geometry; the regressor must NOT override it.
            if resolved_macro is not None:
                if area is None and resolved_macro.area_um2:
                    area, src = resolved_macro.area_um2, src or "lef_exact"
                if fmax is None and resolved_macro.fmax_mhz:
                    fmax, src = resolved_macro.fmax_mhz, src or "lef_exact"
            # 3) last resort: regressor, only to fill a hole LEF/row could not.
            if area is None:
                area, src = model.predict_area(ports, width, depth, impl), src or reg_src
            if fmax is None:
                fmax, src = model.predict_fmax(ports, width, depth, impl), src or reg_src
            cands[impl] = {"area_um2": area, "fmax_mhz": fmax, "source": src or "lef_exact"}
        elif in_grid:
            # in-grid flop/registered_flop -> the learned model, verbatim.
            cands[impl] = {
                "area_um2": model.predict_area(ports, width, depth, impl),
                "fmax_mhz": model.predict_fmax(ports, width, depth, impl),
                "source": "regressor_in_grid",
            }
        else:
            # out-of-grid flop/registered_flop -> analytic extrapolation, never
            # the saturated regressor.
            cands[impl] = {
                "area_um2": _extrapolate_area(width, depth, impl, table),
                "fmax_mhz": _extrapolate_fmax(width, depth, impl, table),
                "source": "analytic_extrapolation",
            }

    # choose: among impls meeting target, least area; else highest Fmax.
    def meets(c):
        return c["fmax_mhz"] is not None and c["fmax_mhz"] >= target_mhz

    meeting = {k: v for k, v in cands.items() if meets(v)}
    rule = recommend_impl_rule(width, depth, ports)

    if meeting:
        # never recommend a bare comb-read "flop"; promote to registered_flop
        order = sorted(meeting.items(),
                       key=lambda kv: (kv[1]["area_um2"] or float("inf")))
        rec = order[0][0]
        if rec == "flop" and "registered_flop" in meeting:
            rec = "registered_flop"
        reason = (f"{rec} meets {target_mhz:.0f} MHz at least predicted area "
                  f"({meeting[rec]['area_um2']:.0f} um2, "
                  f"{meeting[rec]['fmax_mhz']:.0f} MHz).")
    else:
        # nothing meets target with flops -> macro if feasible, else reshape
        if macro_feasible and "macro" in cands:
            rec = "macro"
            reason = (f"no flop impl reaches {target_mhz:.0f} MHz (deep read "
                      f"mux); recommend SRAM macro.")
        else:
            rec = "reshape"
            reason = ("no flop impl reaches target AND no macro is feasible "
                      "(too wide/shallow): reshape the memory -- split banks, "
                      "register the read, or deepen the array to clear the "
                      "macro threshold. " + rule["reason"])

    pred = cands.get(rec if rec != "reshape" else "registered_flop", {})
    return {
        "recommended_impl": rec,
        "pred_area_um2": pred.get("area_um2"),
        "pred_fmax_mhz": pred.get("fmax_mhz"),
        "macro_feasible": macro_feasible,
        "in_grid": in_grid,
        "estimate_source": pred.get("source", reg_src),
        "reason": reason,
        "candidates": cands,
        "rule_v0": rule,
    }


# ---------------------------------------------------------------------------
# Standalone entry
# ---------------------------------------------------------------------------

def _cmd_sweep(args: list[str]) -> int:
    small = "--full" not in args
    if small:
        widths = (8, 32, 64)
        depths = (16, 64, 256, 1024)
        grid = build_grid(widths=widths, depths=depths)
    else:
        grid = build_grid()
    print(f"PDK hash: {pdk_hash()}  liberty: {LIBERTY}")
    print(f"Sweeping {len(grid)} geometries x {len(IMPLS)} impls ...")
    rows = characterize_memories(grid=grid)
    model = fit_model(rows)
    print(f"\nModel kind: {model.kind}")
    if model.fmax_importances:
        imp = sorted(model.fmax_importances.items(), key=lambda x: -x[1])
        print("Fmax feature importances:",
              ", ".join(f"{k}={v:.2f}" for k, v in imp[:5]))
    if model.area_importances:
        imp = sorted(model.area_importances.items(), key=lambda x: -x[1])
        print("Area feature importances:",
              ", ".join(f"{k}={v:.2f}" for k, v in imp[:5]))
    print(f"\nCached table: {cache_path()}")
    _print_real_geom_report(rows, model)
    return 0


def _print_real_geom_report(rows: list[MemPoint], model: MemModel) -> None:
    print("\n=== Real-geometry measured-vs-predicted report ===")
    hdr = (f"{'geom':>14} {'impl':>16} {'meas_area':>10} {'meas_Fmax':>10} "
           f"{'pred_area':>10} {'pred_Fmax':>10} {'risk':>7}")
    print(hdr)
    print("-" * len(hdr))
    for ports, width, depth in REAL_GEOMS:
        for impl in IMPLS:
            r = next((x for x in rows if x.ports == ports and x.width == width
                      and x.depth == depth and x.impl == impl), None)
            if r is None:
                continue
            pa = model.predict_area(ports, width, depth, impl)
            pf = model.predict_fmax(ports, width, depth, impl)
            ma = f"{r.area_um2:,.0f}" if r.area_um2 else "-"
            mf = f"{r.fmax_mhz:,.0f}" if r.fmax_mhz else ("FAIL" if r.error else "-")
            print(f"{ports+' '+str(width)+'x'+str(depth):>14} {impl:>16} "
                  f"{ma:>10} {mf:>10} "
                  f"{(f'{pa:,.0f}' if pa else '-'):>10} "
                  f"{(f'{pf:,.0f}' if pf else '-'):>10} {r.routability_risk:>7}")
        rec = predict_mem(width, depth, ports, target_mhz=100.0, model=model)
        print(f"  -> recommend @100MHz: {rec['recommended_impl']}  "
              f"(macro_feasible={rec['macro_feasible']})")
        print(f"     {rec['reason']}")
    print()


def _cmd_predict(args: list[str]) -> int:
    if len(args) < 3:
        print("usage: predict <ports> <width> <depth> [target_mhz]")
        return 2
    ports, width, depth = args[0], int(args[1]), int(args[2])
    target = float(args[3]) if len(args) > 3 else 100.0
    res = predict_mem(width, depth, ports, target_mhz=target)
    print(json.dumps(res, indent=2))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: python -m orchestrator.langgraph.mem_characterize "
              "{sweep [--full] | predict <ports> <width> <depth> [target_mhz]}")
        return 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "sweep":
        return _cmd_sweep(rest)
    if cmd == "predict":
        return _cmd_predict(rest)
    print(f"unknown command: {cmd}")
    return 2


if __name__ == "__main__":
    from orchestrator.profile import apply as _apply_profile
    _apply_profile()
    raise SystemExit(main())
