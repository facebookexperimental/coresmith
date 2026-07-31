# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Backend helper functions for the ASIC physical design pipeline.

Provides Tcl script generation (parameterized from validated templates),
subprocess wrappers for Nix-wrapped EDA tools, and report parsers.

Tools:
  - OpenROAD 26Q1 (PnR, STA, SPEF estimate)
  - OpenRCX via OpenROAD (accurate SPEF extraction)
  - Magic VLSI (DRC, GDS, SPICE extraction)
  - Netgen (LVS)

All Tcl templates are derived from the validated adder_16bit flow that
produced zero-violation results on Sky130 HD at 50 MHz.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from orchestrator.langgraph.pipeline_helpers import (
    GREEN,
    PDK_ROOT,
    PROJECT_ROOT,
    RED,
    YELLOW,
    _write_step_log,
    _write_step_log_error,
    load_config,
    log,
)

# ---------------------------------------------------------------------------
# PDK path resolution
# ---------------------------------------------------------------------------

def _pdk_variant() -> str:
    """Return the PDK variant directory name (sky130A or sky130B)."""
    for v in ("sky130A", "sky130B"):
        if (PDK_ROOT / v).is_dir():
            return v
    return "sky130A"


_PDK_VAR = _pdk_variant()
_PDK_PATH = PDK_ROOT / _PDK_VAR
_STD_CELL = "sky130_fd_sc_hd"

TECH_LEF = _PDK_PATH / "libs.ref" / _STD_CELL / "techlef" / f"{_STD_CELL}__nom.tlef"
CELL_LEF = _PDK_PATH / "libs.ref" / _STD_CELL / "lef" / f"{_STD_CELL}.lef"
LIBERTY = _PDK_PATH / "libs.ref" / _STD_CELL / "lib" / f"{_STD_CELL}__tt_025C_1v80.lib"
CELL_GDS = _PDK_PATH / "libs.ref" / _STD_CELL / "gds" / f"{_STD_CELL}.gds"
CELL_SPICE = _PDK_PATH / "libs.ref" / _STD_CELL / "spice" / f"{_STD_CELL}.spice"
MAGIC_RC = _PDK_PATH / "libs.tech" / "magic" / f"{_PDK_VAR}.magicrc"
NETGEN_SETUP = _PDK_PATH / "libs.tech" / "netgen" / "setup.tcl"
RCX_RULES = _PDK_PATH / "libs.tech" / "rcx" / "sky130hd_rcx_patterns.rules"


def _resolve_tool(config_key: str, default_script: str) -> str:
    """Resolve an EDA tool binary path.

    Resolution order (first match wins):

    1. ``CORESMITH_BACKEND_<NAME>`` env var (e.g. ``CORESMITH_BACKEND_OPENROAD``)
       -- used by the ``nix develop`` shellHook and the Docker image to
       point at the bare binary on ``$PATH`` and skip the per-call
       ``nix shell`` re-entry.
    2. ``backend.<config_key>`` in ``orchestrator/config.yaml`` -- the
       checked-in default points at ``scripts/*-nix.sh`` wrappers.
    3. ``default_script`` relative to the project root.
    4. ``default_script`` as-is (lets the OS resolve it via ``$PATH``).
    """
    env_key = "CORESMITH_BACKEND_" + config_key.removesuffix("_binary").upper()
    env_val = os.environ.get(env_key, "").strip()
    if env_val:
        return env_val

    try:
        cfg = load_config()
        backend = cfg.get("backend", {})
        path = backend.get(config_key, "")
        if path:
            p = Path(path)
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            if p.exists():
                return str(p)
    except Exception:
        pass
    p = PROJECT_ROOT / default_script
    if p.exists():
        return str(p)
    return default_script


OPENROAD_BIN = _resolve_tool("openroad_binary", "scripts/openroad-nix.sh")
MAGIC_BIN = _resolve_tool("magic_binary", "scripts/magic-nix.sh")
NETGEN_BIN = _resolve_tool("netgen_binary", "scripts/netgen-nix.sh")
KLAYOUT_BIN = _resolve_tool("klayout_binary", "scripts/klayout-nix.sh")
RENDER_SCRIPT = str(PROJECT_ROOT / "scripts" / "render_layout.rb")


# ---------------------------------------------------------------------------
# Layout image rendering (best-effort)
# ---------------------------------------------------------------------------

def render_layout_image(
    input_path: str,
    output_path: str,
    width: int = 2048,
    height: int = 1536,
    timeout: int = 120,
) -> bool:
    """Render a GDS or DEF file to PNG using KLayout.

    Best-effort: returns True on success, False on any failure.
    Never raises -- image rendering must not break the build flow.
    """
    if not Path(input_path).exists():
        return False

    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        KLAYOUT_BIN, "-z",
        "-r", RENDER_SCRIPT,
        "-rd", f"input={input_path}",
        "-rd", f"output={output_path}",
        "-rd", f"width={width}",
        "-rd", f"height={height}",
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode == 0 and Path(output_path).exists():
            log(f"  [IMG] Rendered {Path(output_path).name}", GREEN)
            return True
        if result.stderr:
            log(f"  [IMG] KLayout stderr: {result.stderr[:200]}", YELLOW)
        return False
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        log(f"  [IMG] Skipped image render: {exc}", YELLOW)
        return False


# ---------------------------------------------------------------------------
# Synthesis attempt history (self-recovery must never be silent)
# ---------------------------------------------------------------------------
#
# The flat-synth driver runs Yosys inside an LLM step that retries on its own.
# On a live run it reported "Synthesis succeeded on attempt 2" -- and attempt 1's
# failure reason was retained NOWHERE: not in ``attempt_history``, not in
# ``previous_error``, not in ``synth_result.json``. A driver that heals itself in
# silence teaches nobody: the same script defect recurs every run, and the only
# trace is a sentence in a chat transcript that is discarded.
#
# Everything below is a PURE function of (result dict, on-disk logs, reply text)
# so it can be unit-tested without an LLM, and it NEVER raises: retaining history
# is diagnostics, and diagnostics must not be able to fail a synthesis that
# succeeded.

#: Yosys prints hard failures as ``ERROR: ...``; Verilog front-end syntax errors
#: come through as ``... syntax error ...``. Both are attempt-ending.
_SYNTH_ERROR_RE = re.compile(
    r"^(?:\s*)(ERROR:.*|.*\bsyntax error\b.*)$", re.MULTILINE | re.IGNORECASE
)
#: "succeeded on attempt 3", "attempt 2 succeeded", "on the 2nd attempt", ...
_ATTEMPT_CLAIM_RE = re.compile(r"attempt\s*#?\s*(\d+)", re.IGNORECASE)
_SYNTH_LOG_GLOBS = ("*.log", "*.txt", "*.out")
_MAX_SUMMARY_CHARS = 800


def _clip(text: str, limit: int = _MAX_SUMMARY_CHARS) -> str:
    t = " ".join(str(text or "").split())
    return t if len(t) <= limit else t[: limit - 3] + "..."


def _first_error_line(text: str) -> str:
    m = _SYNTH_ERROR_RE.search(str(text or ""))
    return _clip(m.group(1)) if m else ""


def collect_synth_attempt_history(
    result: dict | None,
    *,
    output_dir: str = "",
    llm_reply: str = "",
    prior_error: str = "",
    node_attempt: int = 1,
    now: str = "",
) -> list[dict]:
    """Per-attempt failure records for ONE synthesis-driver invocation.

    Merges every source of attempt evidence that exists, de-duplicated and
    ordered by attempt number. Each record is
    ``{attempt, error_summary, source, timestamp}``.

      * ``result["attempt_history"]`` / ``["attempts"]`` -- what the driver
        itself recorded (the prompt now asks for it explicitly).
      * on-disk logs under ``output_dir`` -- any log carrying a Yosys ``ERROR:``
        line. Deterministic evidence that does not depend on the driver being
        honest about its own retries.
      * ``prior_error`` -- the failure this NODE-level attempt is retrying, which
        otherwise only survives until the next node overwrites it.
      * ``llm_reply`` -- the driver's own summary. When it claims success on
        attempt N and fewer than N-1 failures were recovered above, the missing
        attempts are recorded EXPLICITLY as ``not retained``. That entry is the
        point: a gap we can see beats a gap we cannot.

    Returns ``[]`` when nothing failed and nothing claims anything did.
    """
    try:
        return _collect_synth_attempt_history(
            result or {}, output_dir, llm_reply, prior_error, node_attempt, now)
    except Exception:  # noqa: BLE001 - diagnostics never fail a synthesis
        return []


def _collect_synth_attempt_history(
    result: dict, output_dir: str, llm_reply: str, prior_error: str,
    node_attempt: int, now: str,
) -> list[dict]:
    import time as _time

    stamp = now or _time.strftime("%Y-%m-%dT%H:%M:%S%z")
    by_attempt: dict[int, dict] = {}

    def _record(attempt: int, summary: str, source: str, timestamp: str = "") -> None:
        summary = _clip(summary)
        if not summary:
            return
        n = max(1, int(attempt))
        prev = by_attempt.get(n)
        # First writer wins per attempt, EXCEPT that a real error summary always
        # displaces a "not retained" placeholder.
        if prev is not None and not prev.get("unrecorded"):
            return
        by_attempt[n] = {
            "attempt": n,
            "error_summary": summary,
            "source": source,
            "timestamp": timestamp or stamp,
            "unrecorded": source == "unrecorded",
        }

    # 1. What the driver recorded about itself.
    declared = result.get("attempt_history")
    if not isinstance(declared, list):
        declared = result.get("attempts")
    if isinstance(declared, list):
        for i, entry in enumerate(declared):
            if isinstance(entry, dict):
                n = entry.get("attempt", entry.get("n", i + 1))
                summary = (
                    entry.get("error_summary") or entry.get("error")
                    or entry.get("reason") or entry.get("summary") or ""
                )
                try:
                    n = int(n)
                except (TypeError, ValueError):
                    n = i + 1
                _record(n, summary, "driver_reported",
                        str(entry.get("timestamp", "")))
            elif isinstance(entry, str):
                _record(i + 1, entry, "driver_reported")

    # 2. Deterministic on-disk evidence.
    log_hits: list[tuple[float, str, str]] = []
    if output_dir:
        d = Path(output_dir)
        if d.is_dir():
            seen: set[str] = set()
            for pattern in _SYNTH_LOG_GLOBS:
                for p in sorted(d.glob(pattern)):
                    if str(p) in seen or not p.is_file():
                        continue
                    seen.add(str(p))
                    try:
                        line = _first_error_line(
                            p.read_text(encoding="utf-8", errors="replace"))
                    except OSError:
                        continue
                    if line:
                        log_hits.append((p.stat().st_mtime, p.name, line))
    log_hits.sort()
    for i, (mtime, name, line) in enumerate(log_hits):
        import time as _t
        _record(i + 1, f"{line}  [{name}]", "yosys_log",
                _t.strftime("%Y-%m-%dT%H:%M:%S", _t.localtime(mtime)))

    # 3. The node-level failure this invocation is retrying.
    if prior_error and str(prior_error).strip().lower() not in ("", "none"):
        _record(max(1, int(node_attempt) - 1), prior_error, "node_previous_error")

    # 4. The driver's own claim about how many attempts it took.
    claimed = 0
    for m in _ATTEMPT_CLAIM_RE.finditer(str(llm_reply or "")):
        try:
            claimed = max(claimed, int(m.group(1)))
        except ValueError:
            continue
    if claimed > 1:
        for n in range(1, claimed):
            if n not in by_attempt:
                _record(
                    n,
                    f"(not retained) the synthesis driver reported reaching "
                    f"attempt {claimed}, so attempt {n} failed, but no error "
                    f"text for it was written to the result JSON or to any log "
                    f"under the output dir",
                    "unrecorded",
                )

    return [by_attempt[k] for k in sorted(by_attempt)]


def persist_synth_attempt_history(
    result_json_path: str, history: list[dict]
) -> bool:
    """Merge ``history`` into the synthesis result artifact. True when written.

    The artifact is the thing a later reader (diagnose, the final report, a
    human) actually opens, so the record has to live THERE and not only in
    graph state that dies with the process. Never raises.
    """
    if not result_json_path or not history:
        return False
    try:
        import json as _json

        p = Path(result_json_path)
        data: dict = {}
        if p.exists():
            try:
                loaded = _json.loads(p.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    data = loaded
            except (OSError, ValueError):
                data = {}
        data["attempt_history"] = history
        data["attempt_failures"] = len(history)
        data["attempt_history_unrecorded"] = sum(
            1 for h in history if h.get("unrecorded"))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_json.dumps(data, indent=2), encoding="utf-8")
        return True
    except Exception:  # noqa: BLE001
        return False


def describe_synth_attempt_history(history: list[dict]) -> str:
    """One-line-per-attempt human summary ('' when nothing failed)."""
    if not history:
        return ""
    lines = [
        f"{len(history)} failed synthesis attempt(s) BEFORE the reported outcome:"
    ]
    for h in history:
        tag = " [NOT RETAINED]" if h.get("unrecorded") else ""
        lines.append(
            f"  attempt {h.get('attempt')}{tag} ({h.get('source')}, "
            f"{h.get('timestamp')}): {h.get('error_summary')}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Flat Top-Level Synthesis (Yosys)
# ---------------------------------------------------------------------------

def generate_flat_synthesis_script(
    design_name: str,
    top_rtl_path: str,
    block_rtl_paths: dict[str, str],
    target_clock_mhz: float = 50.0,
    output_dir: str = "",
) -> str:
    """Generate a Yosys synthesis script for the flat top-level design.

    Reads all block RTL + top-level, synthesises to Sky130 gates.

    Returns the path to the generated .ys file.
    """
    if not output_dir:
        output_dir = str(PROJECT_ROOT / "syn" / "output" / design_name)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    script_path = out / f"{design_name}_flat.ys"

    period_ns = 1000.0 / target_clock_mhz

    read_cmds = [f"read_verilog {top_rtl_path}"]
    for bp in block_rtl_paths.values():
        if Path(bp).exists() and bp != top_rtl_path:
            read_cmds.append(f"read_verilog {bp}")
    reads = "\n".join(read_cmds)

    top_module = Path(top_rtl_path).stem

    # --- SRAM macro awareness (5th fix) ---------------------------------
    # If the RTL instantiates a pre-built macro, read its verilog as a
    # blackbox (so `hierarchy -check` resolves the instance instead of
    # erroring) + its liberty, and map constants to tie cells so the macro's
    # tie-offs don't leave unroutable one_/zero_ nets. Macro blackbox reads
    # MUST precede the design reads.
    macro_bb = macro_libs = macro_hilomap = ""
    try:
        from orchestrator.langgraph.macro_backend import synth_injection
        from orchestrator.langgraph.macro_registry import (
            detect_instantiated_macros,
            discover_macros,
        )
        registry = discover_macros()
        rtl_files = [top_rtl_path, *block_rtl_paths.values()]
        used: dict[str, object] = {}
        for rf in rtl_files:
            if Path(rf).exists():
                for mi in detect_instantiated_macros(rf, registry):
                    used[mi.name] = mi
        macro_bb, macro_libs, macro_hilomap = synth_injection(list(used.values()))
    except Exception as exc:  # never let macro logic break std-cell synth
        log(f"  [MACRO] synth injection skipped: {exc}", YELLOW)

    bb_block = (macro_bb + "\n") if macro_bb else ""
    lib_block = (macro_libs + "\n") if macro_libs else ""
    hilomap_block = (macro_hilomap + "\n") if macro_hilomap else ""

    script = f"""# Flat top-level synthesis for {design_name} (Sky130 HD)
# Generated by coresmith backend_helpers.generate_flat_synthesis_script

{bb_block}{lib_block}{reads}

hierarchy -check -top {top_module}
proc; opt
flatten
opt; fsm; opt; memory; opt
techmap; opt
dfflibmap -liberty {LIBERTY}
abc -liberty {LIBERTY}
{hilomap_block}clean
opt_clean -purge

stat -liberty {LIBERTY}

write_verilog -noattr {out / f"{design_name}_netlist.v"}

"""
    # Generate SDC
    sdc_path = out / f"{design_name}.sdc"
    sdc_path.write_text(
        f"create_clock -name clk -period {period_ns} [get_ports clk]\n"
        f"set_input_delay {period_ns * 0.2:.1f} -clock clk [all_inputs]\n"
        f"set_output_delay {period_ns * 0.2:.1f} -clock clk [all_outputs]\n"
    )

    script_path.write_text(script)
    return str(script_path)


def run_flat_synthesis(
    design_name: str,
    top_rtl_path: str,
    block_rtl_paths: dict[str, str],
    target_clock_mhz: float = 50.0,
    project_root: str = "",
    timeout: int = 600,
) -> dict:
    """Run Yosys flat synthesis on the integrated design.

    Returns dict with: success, netlist_path, sdc_path, gate_count, area_um2,
    log_path, error.
    """
    if not project_root:
        project_root = str(PROJECT_ROOT)
    root = Path(project_root)
    output_dir = str(root / "syn" / "output" / design_name)

    script_path = generate_flat_synthesis_script(
        design_name, top_rtl_path, block_rtl_paths,
        target_clock_mhz=target_clock_mhz,
        output_dir=output_dir,
    )

    cmd = ["yosys", "-s", script_path]

    log(f"  [FLAT-SYNTH] Running Yosys flat synthesis for {design_name}...", YELLOW)

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=project_root,
        )
        log_path = _write_step_log(design_name, "synthesize", cmd, result)
        stdout = result.stdout
        stderr = result.stderr

        has_error = result.returncode != 0
        if has_error:
            error_text = stderr or stdout
            log(f"  [FLAT-SYNTH] FAILED: {error_text[:200]}", RED)
            return {
                "success": False,
                "error": error_text[-3000:],
                "log_path": log_path,
            }

        # Parse gate count from Yosys stat output
        gate_count = 0
        area_um2 = 0.0
        m = re.search(r"Number of cells:\s+(\d+)", stdout)
        if m:
            gate_count = int(m.group(1))
        m = re.search(r"Chip area.*?:\s+([\d.]+)", stdout)
        if m:
            area_um2 = float(m.group(1))

        netlist_path = str(Path(output_dir) / f"{design_name}_netlist.v")
        sdc_path = str(Path(output_dir) / f"{design_name}.sdc")

        log(f"  [FLAT-SYNTH] SUCCESS: {gate_count:,} cells, "
            f"{area_um2:,.1f} µm²", GREEN)

        return {
            "success": True,
            "netlist_path": netlist_path,
            "sdc_path": sdc_path,
            "gate_count": gate_count,
            "area_um2": area_um2,
            "log_path": log_path,
        }
    except subprocess.TimeoutExpired:
        log_path = _write_step_log_error(
            design_name, "synthesize", cmd,
            f"Yosys timed out ({timeout}s)",
        )
        return {
            "success": False,
            "error": f"Yosys timed out ({timeout}s)",
            "log_path": log_path,
        }
    except FileNotFoundError:
        log_path = _write_step_log_error(
            design_name, "synthesize", cmd,
            "Yosys not found",
        )
        return {
            "success": False,
            "error": "Yosys binary not found",
            "log_path": log_path,
        }


def _generate_floorplan_tcl(block_name: str, utilization: int, gate_count: int) -> str:
    """Generate the floorplan section of the PnR TCL script.

    Uses gate-count-based minimum die sizing to prevent power-strap
    failures on small designs (OpenROAD IFP-0024).
    """
    import math

    avg_cell_area_um2 = 10
    min_edge = 60.0
    if gate_count > 0:
        estimated_edge = math.sqrt(gate_count * avg_cell_area_um2 / (utilization / 100.0)) * 2.0
        min_edge = max(60.0, estimated_edge)

    needs_explicit_die = gate_count > 0 and gate_count < 500

    tracks = (
        'make_tracks li1  -x_offset 0.23 -x_pitch 0.46 -y_offset 0.17 -y_pitch 0.34\n'
        'make_tracks met1 -x_offset 0.17 -x_pitch 0.34 -y_offset 0.17 -y_pitch 0.34\n'
        'make_tracks met2 -x_offset 0.23 -x_pitch 0.46 -y_offset 0.23 -y_pitch 0.46\n'
        'make_tracks met3 -x_offset 0.34 -x_pitch 0.68 -y_offset 0.34 -y_pitch 0.68\n'
        'make_tracks met4 -x_offset 0.46 -x_pitch 0.92 -y_offset 0.46 -y_pitch 0.92\n'
        'make_tracks met5 -x_offset 1.70 -x_pitch 3.40 -y_offset 1.70 -y_pitch 3.40\n'
    )

    if needs_explicit_die:
        core_margin = 2.5
        core_edge = min_edge - 2 * core_margin
        floorplan = (
            f'# Small design ({gate_count} gates) -- use explicit die area\n'
            f'# to ensure enough space for power straps (avoid IFP-0024).\n'
            f'initialize_floorplan \\\n'
            f'    -die_area "0 0 {min_edge:.1f} {min_edge:.1f}" \\\n'
            f'    -core_area "{core_margin} {core_margin} {core_edge:.1f} {core_edge:.1f}" \\\n'
            f'    -site unithd\n'
        )
    else:
        floorplan = (
            f'initialize_floorplan \\\n'
            f'    -utilization {utilization} \\\n'
            f'    -aspect_ratio 1.0 \\\n'
            f'    -core_space 2 \\\n'
            f'    -site unithd\n'
        )

    relaxed_util = max(utilization - 10, 15)

    return (
        f'{floorplan}\n'
        f'{tracks}\n'
        f'place_pins -hor_layers met3 -ver_layers met2\n\n'
        f'tapcell \\\n'
        f'    -distance 14 \\\n'
        f'    -tapcell_master {_STD_CELL}__tapvpwrvgnd_1\n\n'
        f'set die_area [ord::get_die_area]\n'
        f'puts "Die area: $die_area"\n\n'
        f'# Post-init die size check\n'
        f'set die_w [expr {{[lindex $die_area 2] - [lindex $die_area 0]}}]\n'
        f'set die_h [expr {{[lindex $die_area 3] - [lindex $die_area 1]}}]\n'
        f'if {{$die_w < 50.0 || $die_h < 50.0}} {{\n'
        f'    puts "WARNING: Die ${{die_w}} x ${{die_h}} um too small for PDN."\n'
        f'    initialize_floorplan -die_area "0 0 {min_edge:.1f} {min_edge:.1f}" '
        f'-core_area "2.5 2.5 {min_edge - 2.5:.1f} {min_edge - 2.5:.1f}" -site unithd\n'
        f'    {tracks}'
        f'    place_pins -hor_layers met3 -ver_layers met2\n'
        f'    set die_area [ord::get_die_area]\n'
        f'    puts "Resized die area: $die_area"\n'
        f'}}\n\n'
        f'# Post-floorplan utilization sanity check\n'
        f'set fp_die_w [expr {{[lindex $die_area 2] - [lindex $die_area 0]}}]\n'
        f'set fp_die_h [expr {{[lindex $die_area 3] - [lindex $die_area 1]}}]\n'
        f'set fp_core_area [expr {{$fp_die_w * $fp_die_h}}]\n'
        f'set fp_cell_count [llength [get_cells *]]\n'
        f'set fp_est_cell_area [expr {{$fp_cell_count * 10.0}}]\n'
        f'if {{$fp_core_area > 0}} {{\n'
        f'    set fp_actual_util [expr {{$fp_est_cell_area / $fp_core_area * 100.0}}]\n'
        f'    puts "Floorplan check: die ${{fp_die_w}}x${{fp_die_h}} um, '
        f'target util: {utilization}%, actual: ${{fp_actual_util}}%"\n'
        f'    if {{$fp_actual_util > {utilization * 1.5}}} {{\n'
        f'        puts "WARNING: utilization ${{fp_actual_util}}% exceeds 1.5x target '
        f'({utilization}%) -- re-floorplanning with {relaxed_util}%"\n'
        f'        initialize_floorplan -utilization {relaxed_util} '
        f'-aspect_ratio 1.0 -core_space 2 -site unithd\n'
        f'        {tracks}'
        f'        place_pins -hor_layers met3 -ver_layers met2\n'
        f'        set die_area [ord::get_die_area]\n'
        f'        puts "Re-floorplanned die area: $die_area"\n'
        f'    }}\n'
        f'}}\n'
    )


# ---------------------------------------------------------------------------
# Tcl Generation: PnR
# ---------------------------------------------------------------------------

def _detect_top_module(netlist_path: str, preferred: str) -> str:
    """Read ``netlist_path`` and return its REAL top module -- the module that
    is defined but not instantiated by any other -- preferring ``preferred``
    when it already is that top (the common case) and never returning a
    sub-block.

    Replaces the old "first ``module (\\w+)``" scan, which picked a SUB-BLOCK
    as the PnR top on a macro-rewritten netlist (yosys emits sub-blocks first,
    the integration top last) and routed/"signed off" a fragment as the chip.
    Falls back to ``preferred`` unchanged when the netlist can't be read
    (matches the pre-fix fallback used by the callers' tests).
    """
    try:
        with open(netlist_path, encoding="utf-8", errors="replace") as _nf:
            text = _nf.read()
    except OSError:
        return preferred
    from orchestrator.langgraph.macro_backend import detect_top_module
    return detect_top_module(text, preferred)


def generate_pnr_tcl(
    block_name: str,
    netlist_path: str,
    sdc_path: str,
    output_dir: str,
    utilization: int = 45,
    density: float = 0.6,
    gate_count: int = 0,
) -> str:
    """Generate an OpenROAD PnR Tcl script from the validated template.

    Returns the path to the generated .tcl file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tcl_path = out / f"pnr_{block_name}.tcl"

    # Use absolute paths so the script works from any location (including tmp_path)
    abs_netlist = str(Path(netlist_path).resolve()) if not os.path.isabs(netlist_path) else netlist_path
    abs_sdc = str(Path(sdc_path).resolve()) if not os.path.isabs(sdc_path) else sdc_path

    # Resolve the REAL top module (defined-but-not-instantiated). Preserves the
    # slug-mangling intent (preferred name != the netlist's real module name,
    # e.g. prd___video_codec_..._top vs video_encode_pipeline_top) but never links a
    # sub-block as the top.
    actual_module = _detect_top_module(netlist_path, block_name)

    script = f"""# Auto-generated PnR flow for {block_name} (Sky130 HD)
# Generated by coresmith backend_helpers.generate_pnr_tcl

set script_dir [file dirname [file normalize [info script]]]

# ----- PDK paths (absolute) -----
set tech_lef   "{TECH_LEF}"
set cell_lef   "{CELL_LEF}"
set liberty    "{LIBERTY}"

# ----- Design paths (absolute) -----
set netlist    "{abs_netlist}"
set sdc_file   "{abs_sdc}"
set out_dir    "$script_dir"

# =====================================================================
# 1. READ DESIGN
# =====================================================================
puts "========== 1. Reading design =========="

read_lef $tech_lef
read_lef $cell_lef
read_liberty $liberty
read_verilog $netlist
link_design {actual_module}
read_sdc $sdc_file

# Fix DRT-0305: Yosys constant nets (zero_, one_) typed as GROUND/POWER
# are not routable by TritonRoute. Connect them to the power grid so they
# become special nets handled by the PDN, not by the signal router.
catch {{
    add_global_connection -net VGND -inst_pattern ".*" -pin_pattern "zero_" -ground
}}
catch {{
    add_global_connection -net VPWR -inst_pattern ".*" -pin_pattern "one_" -power
}}

puts "Design linked. Cell count: [llength [get_cells *]]"

# =====================================================================
# 2. FLOORPLAN
# =====================================================================
puts "\\n========== 2. Floorplan =========="

{_generate_floorplan_tcl(block_name, utilization, gate_count)}

# =====================================================================
# 3. POWER DISTRIBUTION NETWORK (PDN)
# =====================================================================
puts "\\n========== 3. Power grid =========="

add_global_connection -net VPWR -pin_pattern "VPWR" -power
add_global_connection -net VGND -pin_pattern "VGND" -ground
add_global_connection -net VPWR -pin_pattern "VPB" -power
add_global_connection -net VGND -pin_pattern "VNB" -ground

global_connect

set_voltage_domain -name CORE -power VPWR -ground VGND

define_pdn_grid -name stdcell_grid \\
    -starts_with POWER \\
    -voltage_domain CORE \\
    -pins met4

add_pdn_stripe -grid stdcell_grid -layer met1 -width 0.48 -followpins -starts_with POWER
add_pdn_stripe -grid stdcell_grid -layer met4 -width 1.6 -pitch 27.14 -offset 13.57 -starts_with POWER
add_pdn_connect -grid stdcell_grid -layers {{met1 met4}}

pdngen

puts "PDN generated."

# =====================================================================
# 4. GLOBAL PLACEMENT
# =====================================================================
puts "\\n========== 4. Global Placement =========="

global_placement -density {density} -pad_left 2 -pad_right 2

puts "Global placement done."

# =====================================================================
# 5. DETAILED PLACEMENT
# =====================================================================
puts "\\n========== 5. Detailed Placement =========="

detailed_placement
check_placement -verbose

# NO filler insertion here -- fillers are inserted after CTS to avoid
# DPL-0036 failures when CTS buffers need placement sites occupied by
# pre-CTS fillers.

puts "Detailed placement done (fillers deferred until after CTS)."

# =====================================================================
# 6. SET WIRE RC (needed for CTS, timing repair, and STA)
# =====================================================================
puts "\\n========== 6. Set wire RC parasitics =========="

set_wire_rc -signal -layer met2
set_wire_rc -clock  -layer met3

puts "Wire RC set: signal=met2, clock=met3"

# =====================================================================
# 6b. PRE-CTS DESIGN REPAIR (buffer high-fanout nets, resize weak drivers)
# =====================================================================
# A ~200-fanout net left on a single min-size driver measures ~12 ns of pure
# cell delay pre-repair (live run: reset net, WNS -1.92 ns); repair_design
# with a fanout cap moved it to WNS 0.00 at 50 MHz. Also exclude the sky130
# probe/lpflow cells: the resizer otherwise picks them and they break LVS
# and skew timing.
puts "\n========== 6b. Pre-CTS repair_design =========="

set_dont_use {{{_STD_CELL}__probe_p_* {_STD_CELL}__probec_p_* {_STD_CELL}__lpflow_*}}
estimate_parasitics -placement
set_max_fanout 16 [current_design]
repair_design
detailed_placement
check_placement -verbose

puts "Pre-CTS repair_design done."

# =====================================================================
# 7. CLOCK TREE SYNTHESIS
# =====================================================================
puts "\\n========== 7. Clock Tree Synthesis =========="

clock_tree_synthesis \\
    -buf_list {{{_STD_CELL}__clkbuf_4 {_STD_CELL}__clkbuf_8}} \\
    -root_buf {_STD_CELL}__clkbuf_8 \\
    -sink_clustering_enable

set_propagated_clock [all_clocks]

repair_clock_nets

remove_fillers
detailed_placement
filler_placement -prefix FILLER {{{_STD_CELL}__decap_12 {_STD_CELL}__decap_8 {_STD_CELL}__decap_6 {_STD_CELL}__decap_4 {_STD_CELL}__decap_3 {_STD_CELL}__fill_2 {_STD_CELL}__fill_1}}

puts "CTS done."

# =====================================================================
# 8. TIMING REPAIR (post-CTS)
# =====================================================================
puts "\\n========== 8. Post-CTS Timing Repair =========="

estimate_parasitics -placement

repair_timing -setup
repair_timing -hold

remove_fillers
detailed_placement
check_placement -verbose
filler_placement -prefix FILLER {{{_STD_CELL}__decap_12 {_STD_CELL}__decap_8 {_STD_CELL}__decap_6 {_STD_CELL}__decap_4 {_STD_CELL}__decap_3 {_STD_CELL}__fill_2 {_STD_CELL}__fill_1}}

puts "Post-CTS repair done."

# =====================================================================
# 9. GLOBAL ROUTING
# =====================================================================
puts "\\n========== 9. Global Routing =========="

set_routing_layers -signal met1-met4 -clock met3-met4

global_route -guide_file "$out_dir/route_guide.guide" \\
    -congestion_iterations 50

puts "Global routing done."

# =====================================================================
# 10. DETAILED ROUTING
# =====================================================================
puts "\\n========== 10. Detailed Routing =========="

# Fix DRT-0305: Yosys/OpenROAD may create constant nets (zero_, one_)
# typed as GROUND/POWER that TritonRoute refuses to route as signal nets.
# Reclassify any non-special GROUND/POWER nets to SIGNAL before routing.
set block [ord::get_db_block]
foreach net [$block getNets] {{
    set sig_type [$net getSigType]
    set special [$net isSpecial]
    if {{($sig_type == "GROUND" || $sig_type == "POWER") && !$special}} {{
        set net_name [$net getName]
        if {{$net_name ne "VPWR" && $net_name ne "VGND" && $net_name ne "VPB" && $net_name ne "VNB"}} {{
            puts "Reclassifying net '$net_name' ($sig_type, special=$special) to SIGNAL"
            $net setSigType SIGNAL
        }}
    }}
}}

detailed_route \\
    -output_drc "$out_dir/route_drc.rpt" \\
    -verbose 1

puts "Detailed routing done."

# =====================================================================
# 11. SPEF PARASITIC ESTIMATION (in-flow)
# =====================================================================
puts "\\n========== 11. SPEF Parasitic Estimation =========="

estimate_parasitics -global_routing

# write_spef may produce empty file if estimate_parasitics didn't populate
# the RCX data store (expected -- use standalone RCX for accurate SPEF)
catch {{write_spef "$out_dir/{block_name}.spef"}}

puts "SPEF estimation done (use standalone RCX for accurate extraction)."

# =====================================================================
# 12. REPORTS (post-route STA)
# =====================================================================
puts "\\n========== 12. Reports =========="

report_checks -path_delay max -format full_clock_expanded > "$out_dir/timing_setup.rpt"
report_checks -path_delay min -format full_clock_expanded > "$out_dir/timing_hold.rpt"
report_tns > "$out_dir/timing_tns.rpt"
report_wns > "$out_dir/timing_wns.rpt"
report_power > "$out_dir/power.rpt"
puts "Reports written to $out_dir"

# Print key metrics to stdout for parsing
puts "\\n========== SUMMARY =========="
report_design_area
report_wns
report_tns
report_power

# =====================================================================
# 13. METAL DENSITY FILL (Efabless shuttle requirement)
# =====================================================================
puts "\\n========== 13. Metal Density Fill =========="

density_fill -rules $tech_lef

puts "Density fill done."

# =====================================================================
# 14. WRITE OUTPUTS
# =====================================================================
puts "\\n========== 14. Writing outputs =========="

write_def "$out_dir/{block_name}_routed.def"
write_verilog "$out_dir/{block_name}_pnr.v"
write_verilog -include_pwr_gnd "$out_dir/{block_name}_pwr.v"

puts "\\n========== FLOW COMPLETE =========="
puts "DEF:              $out_dir/{block_name}_routed.def"
puts "Verilog:          $out_dir/{block_name}_pnr.v"
puts "Power Verilog:    $out_dir/{block_name}_pwr.v"
puts "SPEF:             $out_dir/{block_name}.spef"

exit
"""
    tcl_path.write_text(script)
    return str(tcl_path)


# ---------------------------------------------------------------------------
# PnR reference template: prepare a working copy for LLM iteration
# ---------------------------------------------------------------------------

_PNR_REFERENCE_TCL = Path(__file__).resolve().parent.parent / "pdk_templates" / "sky130" / "pnr_reference.tcl"


def _estimate_std_cell_area(netlist_path: str) -> float:
    """Rough std-cell area (um^2) for floorplan sizing: count gate instances and
    multiply by a nominal sky130 HD average cell area. Best-effort; the planner
    sizes the die at ~50% util so an over/under-estimate still routes."""
    try:
        text = Path(netlist_path).read_text(errors="ignore")
    except OSError:
        return 150_000.0
    n = len(re.findall(r"\bsky130_fd_sc_hd__\w+\s+\S+\s*\(", text))
    # ~6 um^2 average per HD cell (incl. fill headroom); floor to a sane minimum
    return max(n * 6.0, 50_000.0)


def _tech_lef_dbu(tech_lef: str | Path = TECH_LEF) -> int | None:
    """Read the tech LEF's own `DATABASE MICRONS` value (do NOT hardcode 1000).

    Returns None when the tech LEF is missing or has no UNITS header, in which
    case the caller leaves macro LEFs untouched (the pre-fix behavior) -- we
    never guess a DBU to down-convert against.
    """
    from orchestrator.langgraph.macro_backend import lef_database_units
    try:
        text = Path(tech_lef).read_text(errors="ignore")
    except OSError:
        return None
    return lef_database_units(text)


def _normalize_macro_lefs(macros: list, out_dir: Path, tech_dbu: int | None):
    """Return the macro list with each LEF pointed at a DBU-normalized COPY.

    For every macro whose LEF DBU exceeds the tech DBU, write a copy under
    ``<out_dir>/macro_lefs/<name>.lef`` with ONLY its `DATABASE MICRONS` header
    rewritten to the tech DBU (coordinates unchanged) and swap the MacroInfo's
    ``lef`` to the copy. LEFs already at/below the tech DBU (e.g. an efabless
    prebuilt macro) and those with no DATABASE header are emitted unchanged --
    so we never assume all macro LEFs are 2000 and never touch the registry's
    cached MacroInfo objects (a fresh ``dataclasses.replace`` copy is used).

    Propagates :class:`LefDbuError` (a coordinate not representable at the tech
    DBU) so a geometry-corrupting rewrite fails hard instead of silently
    shipping.
    """
    import dataclasses

    from orchestrator.langgraph.macro_backend import normalize_lef_dbu
    if tech_dbu is None:
        return macros
    lef_dir = out_dir / "macro_lefs"
    result = []
    for m in macros:
        lef = getattr(m, "lef", "")
        if not lef or not Path(lef).exists():
            result.append(m)
            continue
        text = Path(lef).read_text(errors="ignore")
        new_text, changed = normalize_lef_dbu(text, tech_dbu)  # may raise LefDbuError
        if not changed:
            result.append(m)
            continue
        lef_dir.mkdir(parents=True, exist_ok=True)
        dst = lef_dir / Path(lef).name
        dst.write_text(new_text, encoding="utf-8")
        log(f"  [MACRO] normalized {Path(lef).name} DBU -> tech DBU {tech_dbu} "
            f"(header-only; coordinates unchanged)", GREEN)
        result.append(dataclasses.replace(m, lef=str(dst)))
    return result


def prepare_pnr_working_copy(
    design_name: str,
    netlist_path: str,
    sdc_path: str,
    output_dir: str,
    utilization: int = 35,
    density: float = 0.6,
    macro_bindings: list[dict] | None = None,
) -> str:
    """Copy the reference PnR TCL template and prepend design-specific variables.

    The LLM agent reads, modifies, and runs this working copy.
    Returns the path to the working TCL script.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    abs_netlist = str(Path(netlist_path).resolve()) if not os.path.isabs(netlist_path) else netlist_path
    abs_sdc = str(Path(sdc_path).resolve()) if not os.path.isabs(sdc_path) else sdc_path

    # Resolve the REAL top module (defined-but-not-instantiated); never a
    # sub-block (see _detect_top_module). Preserves slug-mangling handling.
    actual_module = _detect_top_module(netlist_path, design_name)

    header = (
        f'# Design-specific variables (auto-generated)\n'
        f'set tech_lef   "{TECH_LEF}"\n'
        f'set cell_lef   "{CELL_LEF}"\n'
        f'set liberty    "{LIBERTY}"\n'
        f'set netlist    "{abs_netlist}"\n'
        f'set sdc_file   "{abs_sdc}"\n'
        f'set out_dir    "{out.resolve()}"\n'
        f'set design_name "{actual_module}"\n'
        f'set utilization {utilization}\n'
        f'set density     {density}\n'
        f'\n'
    )

    # --- SRAM macro collateral + floorplan plan (5th fix) ----------------
    # If the netlist instantiates pre-built macros, emit the macro variables
    # the guarded blocks in pnr_reference.tcl consume: LEF/lib lists, an
    # explicit die sized to fit macro+std area, and per-instance placement.
    #
    # TWO source paths (regression-safe):
    #   (a) CONCRETE-NAME: the netlist already instantiates a named macro
    #       (e.g. a directly-instantiated MCU-A macro) -> detect + place as
    #       before (unchanged).
    #   (b) SHELL BINDING: the netlist carries empty `cs_mem_macro_shell` stubs
    #       resolved to concrete macros at synth (`macro_bindings`). Materialize
    #       each shell into its concrete macro (with the active-low pin adapter)
    #       so OpenROAD reads a netlist that REALLY instantiates the macro, then
    #       point `read_verilog` at that rewritten netlist. Gated by
    #       CORESMITH_PNR_MACRO_PLACEMENT (default ON).
    pnr_netlist = abs_netlist
    from orchestrator.langgraph.macro_backend import LefDbuError
    from orchestrator.langgraph.sram_wrapper import pnr_macro_placement_enabled
    # Hardening (fix 2): when memories were BOUND at synth and placement is
    # enabled, reading + placing the macro is NON-OPTIONAL. A failure here must
    # NOT be swallowed into a macro-less (memory-absent) layout -- that is the
    # path the LLM previously reward-hacked around the LEF discard. Fail hard so
    # the run cannot "succeed" with the SRAM physically absent.
    macro_placement_required = bool(macro_bindings) and pnr_macro_placement_enabled()
    try:
        from orchestrator.langgraph.macro_backend import (
            expand_placements_to_flattened,
            extract_macro_instances,
            materialize_macro_netlist,
            plan_floorplan,
            pnr_header_vars,
        )
        from orchestrator.langgraph.macro_registry import (
            detect_instantiated_macros,
            discover_macros,
        )
        registry = discover_macros()
        macros = detect_instantiated_macros(abs_netlist, registry)
        placed: list = []
        if macros:
            # (a) concrete-name path -- unchanged, then sized for the FLATTENED
            # leaf count (a macro inside a sub-block instantiated K times
            # resolves to K leaves at link -> size the die + place all K).
            placed = extract_macro_instances(abs_netlist, macros)
            _src_txt = Path(abs_netlist).read_text(encoding="utf-8", errors="replace")
            placed = expand_placements_to_flattened(_src_txt, design_name, placed)
        elif macro_bindings and pnr_macro_placement_enabled():
            # (b) shell-materialization path
            _src = Path(abs_netlist).read_text(encoding="utf-8", errors="replace")
            _rewritten, placed = materialize_macro_netlist(_src, macro_bindings)
            if placed:
                pnr_netlist = str(out / f"{design_name}_macro.v")
                Path(pnr_netlist).write_text(_rewritten, encoding="utf-8")
                log(f"  [MACRO] materialized {len(placed)} shell instance(s) -> "
                    f"concrete macro(s) in {Path(pnr_netlist).name}", GREEN)
        if placed:
            std_area = _estimate_std_cell_area(abs_netlist)
            die_box, core_box, placements = plan_floorplan(placed, std_area)
            uniq = {p[4].name: p[4] for p in placements}
            macros_for_vars = list(uniq.values())
            # Fix 1: normalize each macro LEF's DBU to the tech LEF's own DBU
            # (read, not hardcoded) so OpenROAD does not DISCARD a
            # higher-precision macro LEF (ODB-0292) and place 0 macros. Gated
            # (default ON); env-off emits the original LEFs = pre-fix behavior.
            if pnr_macro_placement_enabled():
                macros_for_vars = _normalize_macro_lefs(
                    macros_for_vars, out, _tech_lef_dbu()
                )
            header += pnr_header_vars(
                macros_for_vars, die_box, core_box, placements
            )
            # Point read_verilog at the materialized netlist (later `set`
            # overrides the header default) so PnR reads the concrete macro.
            if pnr_netlist != abs_netlist:
                header += f'set netlist    "{pnr_netlist}"\n'
            # Macro designs congest at high placement density; cap it so the
            # std cells get routing room around the macro (later `set`
            # overrides the header default).
            if density > 0.5:
                header += "set density 0.5\n"
            log(
                f"  [MACRO] {len(placements)} macro instance(s) -> die {die_box}",
                GREEN,
            )
    except LefDbuError:
        # A macro LEF cannot be normalized to the tech DBU without rounding a
        # coordinate -- fail hard rather than silently corrupt geometry OR drop
        # the macro into a memory-absent layout.
        raise
    except Exception as exc:
        if macro_placement_required:
            # Non-optional macro placement failed -- do not degrade to a
            # macro-less layout. Surface it so the PnR step reports failure.
            raise RuntimeError(
                f"macro placement is required (memories bound at synth) but the "
                f"macro read/placement failed: {exc}"
            ) from exc
        log(f"  [MACRO] PnR injection skipped: {exc}", YELLOW)

    template_src = _PNR_REFERENCE_TCL.read_text(encoding="utf-8")
    # Remove the variable-declaration comments from the template header
    # since we provide concrete values
    body_start = template_src.find("set script_dir")
    if body_start > 0:
        template_body = template_src[body_start:]
    else:
        template_body = template_src

    tcl_path = out / f"pnr_{design_name}.tcl"
    tcl_path.write_text(header + template_body, encoding="utf-8")
    return str(tcl_path)


# ---------------------------------------------------------------------------
# Tcl Generation: DRC + GDS
# ---------------------------------------------------------------------------

def generate_drc_tcl(
    block_name: str,
    routed_def_path: str,
    output_dir: str,
    macro_bindings: list[dict] | None = None,
) -> str:
    """Generate a Magic DRC + GDS + hierarchical SPICE extraction Tcl script.

    Returns the path to the generated .tcl file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tcl_path = out / f"drc_{block_name}.tcl"

    abs_def = str(Path(routed_def_path).resolve()) if not os.path.isabs(routed_def_path) else routed_def_path

    # --- SRAM macro black-box block (5th fix) ---
    # Read each instantiated macro's LEF + mark it LEFview (skip its internal
    # DRC/extraction -> no 1.1M false violations + clean LVS black box) while
    # GDS_FILE streams the macro's real signed-off layout into the merged GDS.
    macro_block = ""
    try:
        from orchestrator.langgraph.macro_backend import (
            _macro_info_from_binding,
            drc_macro_block,
            extract_macro_instances,
        )
        from orchestrator.langgraph.macro_registry import (
            detect_instantiated_macros,
            discover_macros,
        )
        macros = detect_instantiated_macros(abs_def, discover_macros())
        placed = extract_macro_instances(abs_def, macros) if macros else []
        # Also include the concrete macros the cs_mem/cs_rom shells were bound
        # to at synth: a memory that entered PnR as a materialized shell places
        # its concrete master in the DEF, so DRC/LVS must treat it as the same
        # signed-off black box even when the DEF name-scan misses it.
        seen = {m.name for m, _ in placed}
        for _b in (macro_bindings or []):
            if _b.get("name") and _b["name"] not in seen and _b.get("lef"):
                _mi = _macro_info_from_binding(_b)
                placed.append((_mi, _b["name"]))
                seen.add(_b["name"])
        if placed:
            macro_block = drc_macro_block(placed)
        elif macros:
            macro_block = drc_macro_block([(m, m.name) for m in macros])
    except Exception as exc:
        log(f"  [MACRO] DRC injection skipped: {exc}", YELLOW)
    macro_block = (macro_block + "\n\n") if macro_block else ""

    script = f"""# Auto-generated Magic DRC + GDS for {block_name} (Sky130)
# Generated by coresmith backend_helpers.generate_drc_tcl

set script_dir [file dirname [file normalize [info script]]]

set def_file   "{abs_def}"
set cell_lef   "{CELL_LEF}"
set tech_lef   "{TECH_LEF}"
set cell_gds   "{CELL_GDS}"
set out_dir    "$script_dir"

puts "========== Magic DRC + GDS: {block_name} =========="

# Read LEF abstracts for standard cells
lef read $tech_lef
lef read $cell_lef

# Read cell GDS for full layouts
gds read $cell_gds

{macro_block}# Read the routed DEF
def read $def_file

# Load the design
load {block_name}

# ---- DRC on flattened version ----
flatten {block_name}_flat
load {block_name}_flat
select top cell
drc catchup
drc count
set drc_count [drc listall count]

set drc_rpt [open "$out_dir/magic_drc.rpt" w]
puts $drc_rpt "Design: {block_name}"
puts $drc_rpt "DRC count: $drc_count"
set drc_result [drc listall why]
puts $drc_rpt $drc_result
close $drc_rpt
puts "DRC violations: $drc_count"

# ---- GDS from flattened ----
gds write "$out_dir/{block_name}.gds"

# ---- Hierarchical SPICE extraction (for LVS) ----
load {block_name}
select top cell
extract all
ext2spice lvs
ext2spice -o "$out_dir/{block_name}.spice"

puts "DRC violations: $drc_count"
puts "DRC report: $out_dir/magic_drc.rpt"
puts "GDS: $out_dir/{block_name}.gds"
puts "SPICE: $out_dir/{block_name}.spice"

quit -noprompt
"""
    tcl_path.write_text(script)
    return str(tcl_path)


# ---------------------------------------------------------------------------
# Tcl Generation: RCX SPEF extraction
# ---------------------------------------------------------------------------

def generate_rcx_tcl(
    block_name: str,
    routed_def_path: str,
    sdc_path: str,
    output_dir: str,
) -> str:
    """Generate an OpenRCX SPEF extraction Tcl script.

    Uses extract_parasitics with the ORFS sky130hd production rules file
    for accurate parasitics. Includes via resistance calibration.

    Returns the path to the generated .tcl file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tcl_path = out / f"rcx_{block_name}.tcl"

    rcx_rules_path = str(RCX_RULES)

    abs_def = str(Path(routed_def_path).resolve()) if not os.path.isabs(routed_def_path) else routed_def_path
    abs_sdc = str(Path(sdc_path).resolve()) if not os.path.isabs(sdc_path) else sdc_path

    script = f"""# Auto-generated OpenRCX SPEF extraction for {block_name} (Sky130)
# Generated by coresmith backend_helpers.generate_rcx_tcl

set script_dir [file dirname [file normalize [info script]]]

set tech_lef   "{TECH_LEF}"
set cell_lef   "{CELL_LEF}"
set liberty    "{LIBERTY}"
set sdc_file   "{abs_sdc}"
set def_file   "{abs_def}"
set rcx_rules  "{rcx_rules_path}"
set out_dir    "$script_dir"

puts "========== RCX SPEF Extraction: {block_name} =========="

# 1. Read design from DEF
read_lef $tech_lef
read_lef $cell_lef
read_liberty $liberty
read_def $def_file
read_sdc $sdc_file

set_propagated_clock [all_clocks]

# 2. Set via resistances (Sky130 calibration from OpenROAD-flow-scripts)
set tech [ord::get_db_tech]
[$tech findLayer mcon] setResistance 9.249146
[$tech findLayer via]  setResistance 4.5
[$tech findLayer via2] setResistance 3.368786
[$tech findLayer via3] setResistance 0.376635
[$tech findLayer via4] setResistance 0.00580

# 3. Run OpenRCX extraction
puts "Running OpenRCX extraction..."
define_process_corner -ext_model_index 0 X
extract_parasitics -ext_model_file $rcx_rules

# 4. Write SPEF
puts "Writing SPEF..."
write_spef "$out_dir/{block_name}.spef"

# 5. Write power-aware Verilog for LVS
write_verilog -include_pwr_gnd "$out_dir/{block_name}_pwr.v"

# 6. Post-extraction STA
puts "\\n========== Post-extraction STA =========="
report_checks -path_delay max -format full_clock_expanded
report_checks -path_delay min -format full_clock_expanded
report_tns
report_wns
report_power

puts "\\nSPEF: $out_dir/{block_name}.spef"
puts "Power Verilog: $out_dir/{block_name}_pwr.v"
puts "Done."

exit
"""
    tcl_path.write_text(script)
    return str(tcl_path)


# ---------------------------------------------------------------------------
# Subprocess Wrappers
# ---------------------------------------------------------------------------

def run_openroad(
    tcl_script: str,
    block_name: str,
    step: str,
    attempt: int = 1,
    timeout: int = 1800,
) -> dict:
    """Run OpenROAD with the given Tcl script.

    Returns dict with: success, stdout, stderr, log_path, and any
    parsed metrics from stdout.
    """
    # `-exit` so a Tcl error TERMINATES the process instead of dropping to an
    # interactive `openroad>` prompt that hangs (leaked procs observed on macro
    # LEF-discard errors); `-no_init` skips any user ~/.openroad init file.
    cmd = [OPENROAD_BIN, "-no_init", "-exit", tcl_script]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=str(PROJECT_ROOT),
        )
        log_path = _write_step_log(block_name, step, cmd, result, attempt)

        stdout = result.stdout
        stderr = result.stderr

        has_error = (
            result.returncode != 0
            or "[ERROR" in stderr
            or "[ERROR" in stdout
        )

        return {
            "success": not has_error,
            "stdout": stdout,
            "stderr": stderr,
            "returncode": result.returncode,
            "log_path": log_path,
        }
    except subprocess.TimeoutExpired:
        log_path = _write_step_log_error(
            block_name, step, cmd,
            f"OpenROAD timed out ({timeout}s)", attempt,
        )
        return {
            "success": False,
            "stdout": "",
            "stderr": f"OpenROAD timed out ({timeout}s)",
            "log_path": log_path,
        }
    except FileNotFoundError:
        log_path = _write_step_log_error(
            block_name, step, cmd,
            f"OpenROAD binary not found: {OPENROAD_BIN}", attempt,
        )
        return {
            "success": False,
            "stdout": "",
            "stderr": f"OpenROAD binary not found: {OPENROAD_BIN}",
            "log_path": log_path,
        }


def run_magic(
    tcl_script: str,
    block_name: str,
    step: str = "drc",
    attempt: int = 1,
    timeout: int = 600,
) -> dict:
    """Run Magic VLSI with the given Tcl script.

    Returns dict with: success, drc_count, gds_path, spice_path, log_path.
    """
    cmd = [
        MAGIC_BIN,
        "-dnull", "-noconsole",
        "-rcfile", str(MAGIC_RC),
        tcl_script,
    ]

    env = os.environ.copy()
    env["PDK_ROOT"] = str(PDK_ROOT)

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=str(PROJECT_ROOT), env=env,
        )
        log_path = _write_step_log(block_name, step, cmd, result, attempt)

        # Clean up .ext files left by Magic extraction in the CWD
        for ext_file in PROJECT_ROOT.glob("*.ext"):
            try:
                ext_file.unlink()
            except OSError:
                pass

        stdout = result.stdout
        output_dir = str(Path(tcl_script).parent)

        drc_count = _parse_magic_drc_count(stdout)
        # Honesty fallback: when the stdout count is blank/zero, recount from the
        # report file (drc listall why) so a blank Magic count never masks real
        # violation rects. Only reads the report when the stdout count is suspect.
        if drc_count <= 0:
            _rpt_path = os.path.join(output_dir, "magic_drc.rpt")
            if os.path.isfile(_rpt_path):
                try:
                    _rpt_text = Path(_rpt_path).read_text(errors="replace")
                except OSError:
                    _rpt_text = ""
                if _rpt_text:
                    _recount = _parse_magic_drc_count(stdout, _rpt_text)
                    if _recount > drc_count:
                        log(
                            f"  [DRC] stdout count blank/zero but report holds "
                            f"{_recount} violation rect(s) -- using report count "
                            "(false-clean guard)", RED,
                        )
                    drc_count = _recount

        return {
            "success": result.returncode == 0,
            "drc_count": drc_count,
            "gds_path": os.path.join(output_dir, f"{block_name}.gds"),
            "spice_path": os.path.join(output_dir, f"{block_name}.spice"),
            "drc_report_path": os.path.join(output_dir, "magic_drc.rpt"),
            "stdout": stdout,
            "stderr": result.stderr,
            "log_path": log_path,
        }
    except subprocess.TimeoutExpired:
        log_path = _write_step_log_error(
            block_name, step, cmd,
            f"Magic timed out ({timeout}s)", attempt,
        )
        return {
            "success": False,
            "drc_count": -1,
            "stdout": "",
            "stderr": f"Magic timed out ({timeout}s)",
            "log_path": log_path,
        }
    except FileNotFoundError:
        log_path = _write_step_log_error(
            block_name, step, cmd,
            f"Magic binary not found: {MAGIC_BIN}", attempt,
        )
        return {
            "success": False,
            "drc_count": -1,
            "stdout": "",
            "stderr": f"Magic binary not found: {MAGIC_BIN}",
            "log_path": log_path,
        }


# ---------------------------------------------------------------------------
# LVS benign-pin reconciliation (openframe GPIO / power tie triage)
# ---------------------------------------------------------------------------
# netgen prints "Top level cell failed pin matching." for EVERY openframe-
# wrapped block even when the layout is correct. The shared-constant GPIO bus
# (io_in/io_out/io_oeb tied to common tie nets) is structural symmetry netgen
# reports as unmatched / "shorted" top-level pins -- the SAME verdict appears on
# a design the engine independently recorded as an LVS match. The raw
# ``"match uniquely" in final_line`` test therefore hard-fails openframe blocks,
# forcing a manual/LLM override.
#
# The reconciliation below is CONSERVATIVE and fail-closed: it upgrades the raw
# netgen mismatch to a match ONLY when the report proves the failure is limited
# to a known-benign top-level pin set. A real short (two independently-driven
# signal nets merged), a missing/extra device, a device-class inequality, a
# top-level "Netlists do not match", or ANY unmatched INTERNAL (non-top-pin)
# node keeps ``match=False``. Note NET-count deltas are the EXPECTED tie-collapse
# artifact and are deliberately NOT disqualifying; only DEVICE counts must be
# equal. Gate: ``CORESMITH_LVS_BENIGN_CLASSIFY`` (default ON -- see
# ``lvs_benign_classify_enabled``). Pure/string-only; unit-testable.

# openframe GPIO bus bits -- PDK/openframe-standard names, not design-specific.
_LVS_GPIO_BUS_RE = re.compile(r"^(?:io_in|io_out|io_oeb)\s*\[\s*\d+\s*\]$")
# SRAM/openframe power & body-bias annotation pins (optionally bus-indexed).
_LVS_POWER_PINS = ("VPWR", "VGND", "VPB", "VNB")
# "Pins <a> and <b> are shorted in cell <top> (n)" (netgen stdout short report).
_LVS_SHORTED_RE = re.compile(
    r"Pins\s+(\S+)\s+and\s+(\S+)\s+are\s+shorted\s+in\s+cell"
)
# "Device classes <a> and <b> are [not] equivalent".
_LVS_DEVCLASS_RE = re.compile(
    r"Device classes\s+(\S+)\s+and\s+(\S+)\s+are\s+(not\s+)?equivalent",
    re.IGNORECASE,
)
# "Circuit 1 contains <n> devices, Circuit 2 contains <m> devices".
_LVS_DEV_COUNT_RE = re.compile(
    r"Circuit\s+1\s+contains\s+(\d+)\s+devices,\s+"
    r"Circuit\s+2\s+contains\s+(\d+)\s+devices",
    re.IGNORECASE,
)
# A single `name[idx]` bit reference, for reference-declared tie membership.
_LVS_BIT_RE = re.compile(r"([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]")
# "Contents of circuit N:  Circuit: '<name>'" -- the per-circuit device-class
# summary header netgen prints (twice: pre- and post-reduction).
_LVS_CIRCUIT_HDR_RE = re.compile(
    r"Contents of circuit\s+([12])\s*:\s*Circuit:\s*'([^']+)'", re.IGNORECASE,
)
# "  Class: <cellname> instances: <n>" -- a per-class device tally line.
_LVS_CLASS_COUNT_RE = re.compile(
    r"Class:\s+(\S+)\s+instances:\s*(\d+)", re.IGNORECASE,
)
# Zero-transistor PHYSICAL-ONLY cells (tap / fill / decap / antenna-diode /
# endcap / constant-tie). Magic's ext2spice legitimately drops these signal-less
# fillers that the reference `_pwr.v` still emits, so an unmatched-DEVICE delta
# composed ONLY of these is benign -- but a real logic/sequential cell delta
# (dfxtp/nand/mux/inv/...) is NOT. Tokens are sky130/OpenROAD physical-cell
# names, not design-specific.
_LVS_PHYSICAL_CELL_TOKENS = (
    "tapvpwrvgnd", "tapcell", "tap_", "_tap", "fill", "filler", "decap",
    "diode", "antenna", "endcap", "fakediode", "conb",
)


def _is_physical_only_cell(cell: str) -> bool:
    """True iff a cell/instance name is a zero-transistor physical-only cell
    (tap, fill, decap, antenna diode, endcap, constant-tie) -- never a real
    logic/sequential cell."""
    n = (cell or "").lower()
    return any(tok in n for tok in _LVS_PHYSICAL_CELL_TOKENS)


def lvs_benign_classify_enabled() -> bool:
    """Whether the deterministic benign-pin LVS reconciliation is active.

    Default ON. The classifier is strict/fail-closed -- it upgrades a netgen
    "Top level cell failed pin matching" verdict to a match ONLY when the top
    cell's device classes are equivalent, every device count is equal, and every
    unmatched/shorted item is in the openframe GPIO/power benign set. Being
    default-on lets the engine sign off openframe blocks deterministically
    instead of always-failing-then-manual-override; it mirrors the sibling
    ``CORESMITH_LVS_VERIFY_TIES`` gate (also default-on) on the LangGraph LVS
    path. Set ``CORESMITH_LVS_BENIGN_CLASSIFY=0`` to restore the raw
    "match uniquely"-only verdict verbatim.
    """
    from orchestrator.profile import ensure_applied, flag_enabled
    ensure_applied()
    return flag_enabled("CORESMITH_LVS_BENIGN_CLASSIFY", default=True)


def _lvs_norm_bit(text: str) -> str | None:
    """Normalize a `name[idx]` token to `name[idx]` (no inner whitespace)."""
    m = _LVS_BIT_RE.search(text or "")
    return f"{m.group(1)}[{int(m.group(2))}]" if m else None


def _lvs_pin_is_benign(pin: str, tie_bits: set[str]) -> bool:
    """True iff a top-level pin token is provably-benign to see unmatched:
    an openframe GPIO bus bit (io_in/io_out/io_oeb[*]), a VPWR/VGND/VPB/VNB
    power annotation, or a reference-declared constant-tie/replicated bit."""
    p = (pin or "").strip()
    if not p or p.startswith("("):
        return False
    if _LVS_GPIO_BUS_RE.match(p):
        return True
    base = p.split("[", 1)[0].strip().upper()
    if base in _LVS_POWER_PINS:
        return True
    if tie_bits:
        nb = _lvs_norm_bit(p)
        if nb is not None and nb in tie_bits:
            return True
    return False


def _lvs_iter_pin_table_rows(report_text: str):
    """Yield ``(left, right)`` stripped cells from every netgen
    ``Subcircuit pins:`` block (the top-level port-matching table)."""
    in_pins = False
    enders = ("Cell pin lists", "Netlists ", "Final result", "Device classes",
              "Subcircuit summary", "Number of", "Circuit 1 contains",
              "Circuit 2 contains")
    for line in (report_text or "").splitlines():
        s = line.strip()
        if s.startswith("Subcircuit pins:"):
            in_pins = True
            continue
        if not in_pins:
            continue
        if any(s.startswith(e) for e in enders):
            in_pins = False
            continue
        if "|" not in line:
            continue
        left, _, right = line.partition("|")
        ls, rs = left.strip(), right.strip()
        # Skip the "Circuit 1: .. |Circuit 2: .." header + dashed separators.
        if ls.startswith("Circuit 1") or set(ls) <= set("- "):
            continue
        yield ls, rs


def _lvs_unmatched_pins(report_text: str) -> list[str]:
    """Return the RAW present-side token for every pin-table row where exactly
    one side is ``(no matching pin)`` -- indexed OR not, so a non-benign
    internal name (e.g. ``net241``) is surfaced rather than silently dropped."""
    out: list[str] = []
    for ls, rs in _lvs_iter_pin_table_rows(report_text):
        l_missing = ls == "" or ls.startswith("(")
        r_missing = rs == "" or rs.startswith("(")
        if l_missing and not r_missing:
            out.append(rs)
        elif r_missing and not l_missing:
            out.append(ls)
    return out


def _lvs_shorted_pins(report_text: str) -> list[str]:
    """Return both pins from every ``Pins <a> and <b> are shorted in cell ...``
    line (netgen's stdout short report)."""
    out: list[str] = []
    for m in _LVS_SHORTED_RE.finditer(report_text or ""):
        out.extend([m.group(1), m.group(2)])
    return out


def _parse_netgen_class_counts(
    report_text: str, top_cell: str = "",
) -> tuple[dict[str, int], dict[str, int]]:
    """Return ``(c1_classes, c2_classes)`` -- the per-device-class instance
    tallies from the LAST top-cell ``Contents of circuit 1/2:`` block on each
    side (the post-reduction blocks that produce netgen's final device counts).

    netgen prints these blocks TWICE (pre- and post-series/parallel reduction);
    keeping the LAST for each circuit takes the reduced tally. Blocks for a
    different circuit name (e.g. an SRAM sub-comparison) are skipped when
    ``top_cell`` is given, so only the top cell's devices are compared.
    """
    c1: dict[str, int] = {}
    c2: dict[str, int] = {}
    cur: dict[str, int] | None = None
    for line in (report_text or "").splitlines():
        hm = _LVS_CIRCUIT_HDR_RE.search(line)
        if hm:
            which, name = hm.group(1), hm.group(2)
            if top_cell and name != top_cell:
                cur = None  # a sub-circuit (e.g. SRAM macro) -- ignore
            elif which == "1":
                c1 = {}       # fresh dict -> LAST block for circuit 1 wins
                cur = c1
            else:
                c2 = {}
                cur = c2
            continue
        s = line.strip()
        if s.startswith(("Contents of circuit", "Circuit 1 contains",
                         "Circuit 2 contains", "Final result")):
            cur = None
            continue
        if cur is None:
            continue
        cm = _LVS_CLASS_COUNT_RE.search(line)
        if cm:
            cur[cm.group(1)] = int(cm.group(2))
    return c1, c2


def _lvs_device_delta_classes(
    report_text: str, top_cell: str = "",
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Diff the two circuits' per-class device tallies and split the imbalanced
    classes into ``(physical_only, non_physical)`` -- each ``(class, abs_delta)``.

    A device-count delta is benign ONLY when ``non_physical`` is empty (every
    imbalanced class is a zero-transistor filler the extractor legitimately
    drops) AND the caller confirms the physical deltas sum to the reported
    device delta. Returns ``([], [])`` when the per-class blocks are absent, so
    the caller fails closed.
    """
    c1, c2 = _parse_netgen_class_counts(report_text, top_cell)
    physical: list[tuple[str, int]] = []
    non_physical: list[tuple[str, int]] = []
    for cls in set(c1) | set(c2):
        d = abs(c1.get(cls, 0) - c2.get(cls, 0))
        if d == 0:
            continue
        (physical if _is_physical_only_cell(cls) else non_physical).append(
            (cls, d)
        )
    return physical, non_physical


def classify_netgen_lvs_benign(
    report_text: str,
    top_cell: str = "",
    reference_verilog_text: str = "",
) -> dict:
    """Triage a netgen LVS report: is a "failed pin matching" verdict a benign
    openframe GPIO/power/tie artifact, or a real mismatch?

    Returns a dict:
      ``benign``            -- True ONLY when every guard below holds.
      ``netgen_match``      -- netgen itself reported "Circuits match uniquely".
      ``reconciled_pins``   -- benign unmatched/shorted top-level pins.
      ``non_benign_pins``   -- unmatched/shorted items NOT in the benign set.
      ``device_classes_equivalent`` / ``device_counts_equal`` -- guard results.
      ``reasons``           -- why acceptance was refused (empty when benign).
      ``analysis``          -- human-readable one-liner.

    Benign iff ALL hold (fail-closed on any doubt):
      1. the reduced netlists are structurally equivalent -- signalled by an
         explicit top-cell "Device classes <top> and <top> are equivalent"
         (report-file form) OR a "Top level cell failed pin matching" verdict
         (stdout form) -- AND nothing is reported "not equivalent",
      2. device counts are equal OR any device-count delta is composed ENTIRELY
         of zero-transistor physical-only filler cells (tap/fill/decap/diode)
         that fully account for the delta; a real logic/sequential-cell delta
         fails. Net-count deltas are the expected tie-collapse artifact and are
         IGNORED. No top-level "Netlists do not match",
      3. at least one unmatched/shorted item exists and EVERY one is a benign
         top-level pin (openframe GPIO bus bit, VPWR/VGND/VPB/VNB power pin, or a
         reference-declared constant-tie bit).
    """
    text = report_text or ""
    low = text.lower()

    # (1) Structural (device-level) equivalence. Two report formats express it:
    #     the report-FILE table form prints an explicit top-cell
    #     "Device classes <top> and <top> are equivalent"; the STDOUT form omits
    #     that line and instead ends with "Top level cell failed pin matching"
    #     -- netgen's own statement that the reduced netlists MATCHED and ONLY
    #     the port pins failed (a real structural divergence prints "Netlists do
    #     not match" instead, caught below). Accept EITHER signal; reject if
    #     anything is reported "not equivalent". A real logic-cell device delta
    #     is independently caught by guard (2), so this relaxation cannot mask a
    #     structural mismatch.
    top_equiv_line = False
    any_not_equiv = False
    for m in _LVS_DEVCLASS_RE.finditer(text):
        g1, g2, neg = m.group(1), m.group(2), m.group(3)
        if neg:
            any_not_equiv = True
            continue
        if g1 == g2 and (not top_cell or g1 == top_cell):
            top_equiv_line = True
    failed_pin_matching = "failed pin matching" in low
    top_equiv = top_equiv_line or failed_pin_matching

    # (2) Device counts. A NET-count "*** MISMATCH ***" is the expected
    #     tie-collapse artifact and is IGNORED. A "Netlists do not match" is a
    #     real structural divergence. A DEVICE-count delta is NOT auto-rejected:
    #     it is reconciled ONLY when every imbalanced device class is a
    #     zero-transistor physical-only filler (tap/fill/decap/diode) that the
    #     extractor legitimately drops -- and the physical deltas fully account
    #     for the reported delta. A real logic/sequential-cell delta stays a
    #     mismatch.
    counts_equal = True          # no device delta at all
    device_delta_benign = False  # delta exists but is all physical-only filler
    device_delta = 0
    delta_physical: list[tuple[str, int]] = []
    delta_nonphysical: list[tuple[str, int]] = []
    structural_mismatch = "netlists do not match" in low
    for line in text.splitlines():
        if "*** MISMATCH ***" in line or "**Mismatch**" in line:
            ll = line.lower()
            if "device" in ll or "instance" in ll:
                counts_equal = False
        m = _LVS_DEV_COUNT_RE.search(line)
        if m and int(m.group(1)) != int(m.group(2)):
            counts_equal = False
            device_delta = abs(int(m.group(1)) - int(m.group(2)))
    if not counts_equal:
        delta_physical, delta_nonphysical = _lvs_device_delta_classes(
            text, top_cell,
        )
        phys_sum = sum(d for _, d in delta_physical)
        device_delta_benign = bool(
            delta_physical
            and not delta_nonphysical
            # the physical fillers must FULLY account for the reported delta;
            # if a numeric delta is present it must match exactly (fail-closed
            # on any unexplained imbalance).
            and (device_delta == 0 or phys_sum == device_delta)
        )
    device_ok = counts_equal or device_delta_benign

    # (3) Benign-pin membership. Reference-declared constant-ties (optional)
    #     extend the name-based openframe/power set.
    tie_bits: set[str] = set()
    if reference_verilog_text:
        try:
            from orchestrator.langgraph.macro_backend import (
                parse_output_tie_classes,
            )
            ties = parse_output_tie_classes(reference_verilog_text)
            tie_bits = (
                set(ties.get("tied_bits", set()))
                | set(ties.get("const_bits", set()))
                | set(ties.get("alias_targets", set()))
            )
        except Exception:
            tie_bits = set()

    flagged = _lvs_unmatched_pins(text) + _lvs_shorted_pins(text)
    reconciled: list[str] = []
    non_benign: list[str] = []
    for pin in flagged:
        if _lvs_pin_is_benign(pin, tie_bits):
            reconciled.append(pin)
        else:
            non_benign.append(pin)

    benign = bool(
        top_equiv
        and not any_not_equiv
        and device_ok
        and not structural_mismatch
        and reconciled
        and not non_benign
    )

    reasons: list[str] = []
    if not top_equiv:
        reasons.append(
            "no structural-equivalence signal (neither a top-cell "
            "device-classes-equivalent line nor a failed-pin-matching verdict)"
        )
    if any_not_equiv:
        reasons.append("a device class is reported NOT equivalent")
    if not device_ok:
        if delta_nonphysical:
            reasons.append(
                "a REAL (non-filler) device count differs: "
                + ", ".join(f"{c} (Δ{d})" for c, d in delta_nonphysical[:6])
            )
        else:
            reasons.append(
                f"a device count differs (Δ{device_delta}) and could not be "
                "proven to be physical-only filler"
            )
    if structural_mismatch:
        reasons.append("top-level netlists do not match (structural)")
    if non_benign:
        reasons.append(
            f"{len(non_benign)} unmatched/shorted item(s) outside the benign "
            f"set: {', '.join(sorted(set(non_benign))[:8])}"
        )
    if not reconciled and not non_benign:
        reasons.append("no unmatched/shorted pins found to reconcile")

    if benign:
        _uniq = sorted(set(reconciled))
        if device_delta_benign:
            _dev = (
                f"device delta of {device_delta} accepted as physical-only "
                "filler ("
                + ", ".join(f"{c} ×{d}" for c, d in sorted(delta_physical))
                + ")"
            )
        else:
            _dev = "device counts equal"
        analysis = (
            f"Benign openframe/power pin mismatch: reconciled "
            f"{len(_uniq)} top-level pin(s) as benign "
            f"({', '.join(_uniq[:12])}{' ...' if len(_uniq) > 12 else ''}); "
            f"device classes equivalent, {_dev}, no internal short."
        )
    else:
        analysis = "Not reconciled as benign: " + "; ".join(reasons)

    return {
        "benign": benign,
        "netgen_match": ("match uniquely" in low) and ("do not match" not in low),
        "reconciled_pins": reconciled,
        "non_benign_pins": non_benign,
        "device_classes_equivalent": bool(top_equiv and not any_not_equiv),
        "device_counts_equal": bool(counts_equal and not structural_mismatch),
        "device_ok": bool(device_ok and not structural_mismatch),
        "device_delta": device_delta,
        "device_delta_benign": device_delta_benign,
        "device_delta_physical": delta_physical,
        "device_delta_nonphysical": delta_nonphysical,
        "reasons": reasons,
        "analysis": analysis,
    }


def reconcile_lvs_match(
    raw_match: bool,
    report_text: str,
    top_cell: str = "",
    reference_verilog_text: str = "",
) -> dict:
    """Gate-aware wrapper around :func:`classify_netgen_lvs_benign`.

    Returns the structured record fields the LVS result carries:
      ``lvs_raw_match``            -- netgen's raw "match uniquely" verdict.
      ``lvs_match``                -- reconciled verdict (== raw unless upgraded).
      ``benign_reconciled_pins``   -- count of pins reconciled as benign.
      ``benign_reconciled_pin_names`` / ``lvs_benign_analysis``.

    When the gate is OFF, or netgen already matched, this is a no-op that returns
    the raw verdict unchanged (preserving legacy behavior exactly). Every
    reconciliation is logged, enumerating the pins accepted as benign.
    """
    info = {
        "lvs_raw_match": bool(raw_match),
        "lvs_match": bool(raw_match),
        "benign_reconciled_pins": 0,
        "benign_reconciled_pin_names": [],
        "lvs_benign_analysis": "",
    }
    if raw_match or not lvs_benign_classify_enabled():
        return info

    verdict = classify_netgen_lvs_benign(
        report_text, top_cell, reference_verilog_text,
    )
    info["lvs_benign_analysis"] = verdict["analysis"]
    if verdict["benign"]:
        info["lvs_match"] = True
        info["benign_reconciled_pins"] = len(verdict["reconciled_pins"])
        info["benign_reconciled_pin_names"] = list(verdict["reconciled_pins"])
        log(
            "  [LVS] benign-classify: raw netgen verdict=FAIL reconciled to "
            f"MATCH -- {verdict['analysis']}",
            GREEN,
        )
    else:
        log(
            "  [LVS] benign-classify: raw netgen verdict=FAIL kept (not "
            f"benign) -- {verdict['analysis']}",
            YELLOW,
        )
    return info


def run_netgen_lvs(
    spice_path: str,
    verilog_path: str,
    block_name: str,
    report_path: str = "",
    attempt: int = 1,
    timeout: int = 600,
) -> dict:
    """Run Netgen LVS comparison.

    Returns dict with: match, device_delta, net_delta, report_path, log_path.
    """
    if not report_path:
        report_path = str(
            Path(spice_path).parent / f"lvs_{block_name}.rpt"
        )

    cmd = [
        NETGEN_BIN, "-batch", "lvs",
        f"{spice_path} {block_name}",
        f"{verilog_path} {block_name}",
        str(NETGEN_SETUP),
        report_path,
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=str(PROJECT_ROOT),
        )
        log_path = _write_step_log(block_name, "lvs", cmd, result, attempt)

        stdout = result.stdout

        report_text = ""
        if Path(report_path).exists():
            try:
                report_text = Path(report_path).read_text()
            except OSError:
                pass

        combined = report_text or stdout
        final_line = ""
        for line in reversed(combined.split("\n")):
            if "final result" in line.lower():
                final_line = line.lower()
                break
        if final_line:
            match = "match uniquely" in final_line
        else:
            match = (
                "match" in stdout.lower()
                and "do not match" not in stdout.lower()
                and "failed" not in stdout.lower()
            )

        # Deterministic benign-pin reconciliation (gate default ON). netgen
        # fails top-level pin matching on the openframe GPIO/power tie bus even
        # for a correct layout; upgrade a raw FAIL to a match ONLY when the
        # report proves the failure is limited to that benign pin set. Reads the
        # reference power-Verilog to also honor declared constant-ties.
        ref_v = ""
        try:
            if verilog_path and Path(verilog_path).exists():
                ref_v = Path(verilog_path).read_text(errors="replace")
        except OSError:
            ref_v = ""
        recon = reconcile_lvs_match(
            match, combined, top_cell=block_name, reference_verilog_text=ref_v,
        )
        match = recon["lvs_match"]

        # Parse device/net counts from stdout
        device_delta, net_delta = _parse_lvs_deltas(stdout)

        return {
            "match": match,
            "device_delta": device_delta,
            "net_delta": net_delta,
            "report_path": report_path,
            "stdout": stdout,
            "stderr": result.stderr,
            "log_path": log_path,
            "lvs_raw_match": recon["lvs_raw_match"],
            "lvs_match": recon["lvs_match"],
            "benign_reconciled_pins": recon["benign_reconciled_pins"],
            "benign_reconciled_pin_names": recon["benign_reconciled_pin_names"],
            "lvs_benign_analysis": recon["lvs_benign_analysis"],
        }
    except subprocess.TimeoutExpired:
        log_path = _write_step_log_error(
            block_name, "lvs", cmd,
            f"Netgen timed out ({timeout}s)", attempt,
        )
        return {
            "match": False,
            "device_delta": -1,
            "net_delta": -1,
            "stdout": "",
            "stderr": f"Netgen timed out ({timeout}s)",
            "log_path": log_path,
            "lvs_raw_match": False,
            "lvs_match": False,
            "benign_reconciled_pins": 0,
            "benign_reconciled_pin_names": [],
            "lvs_benign_analysis": "",
        }
    except FileNotFoundError:
        log_path = _write_step_log_error(
            block_name, "lvs", cmd,
            f"Netgen binary not found: {NETGEN_BIN}", attempt,
        )
        return {
            "match": False,
            "device_delta": -1,
            "net_delta": -1,
            "stdout": "",
            "stderr": f"Netgen binary not found: {NETGEN_BIN}",
            "log_path": log_path,
            "lvs_raw_match": False,
            "lvs_match": False,
            "benign_reconciled_pins": 0,
            "benign_reconciled_pin_names": [],
            "lvs_benign_analysis": "",
        }


# ---------------------------------------------------------------------------
# Report Parsers
# ---------------------------------------------------------------------------

def parse_openroad_reports(output_dir: str) -> dict:
    """Parse timing, power, and area reports from OpenROAD output directory.

    Returns dict with timing, power, and area metrics.
    """
    out = Path(output_dir)
    metrics: dict = {
        "wns_ns": 0.0,
        "tns_ns": 0.0,
        "setup_slack_ns": 0.0,
        "hold_slack_ns": 0.0,
        "total_power_mw": 0.0,
        "dynamic_power_mw": 0.0,
        "leakage_power_mw": 0.0,
        "die_area_um2": 0.0,
        "design_area_um2": 0.0,
        "utilization_pct": 0.0,
        # None until a WNS is actually read: an unmeasured timing result is
        # "cannot judge", not a pass (fail-closed, A-Fix 2g). Consumers treat
        # it falsy-safe (``.get("timing_met", False)`` / ``and`` chains).
        "timing_met": None,
    }

    # Parse WNS
    wns_file = out / "timing_wns.rpt"
    if wns_file.exists():
        text = wns_file.read_text().strip()
        m = re.search(r"wns\s+(?:max\s+)?(-?[\d.]+)", text, re.IGNORECASE)
        if m:
            metrics["wns_ns"] = float(m.group(1))
            metrics["timing_met"] = metrics["wns_ns"] >= 0

    # Parse TNS
    tns_file = out / "timing_tns.rpt"
    if tns_file.exists():
        text = tns_file.read_text().strip()
        m = re.search(r"tns\s+(?:max\s+)?(-?[\d.]+)", text, re.IGNORECASE)
        if m:
            metrics["tns_ns"] = float(m.group(1))

    # Parse setup slack
    setup_file = out / "timing_setup.rpt"
    if setup_file.exists():
        text = setup_file.read_text()
        m = re.search(r"(-?[\d.]+)\s+slack\s+\((?:MET|VIOLATED)\)", text)
        if m:
            metrics["setup_slack_ns"] = float(m.group(1))

    # Parse hold slack
    hold_file = out / "timing_hold.rpt"
    if hold_file.exists():
        text = hold_file.read_text()
        m = re.search(r"(-?[\d.]+)\s+slack\s+\((?:MET|VIOLATED)\)", text)
        if m:
            metrics["hold_slack_ns"] = float(m.group(1))

    # Parse power report
    power_file = out / "power.rpt"
    if power_file.exists():
        text = power_file.read_text()
        # "Total  1.41e-04   3.24e-05   7.52e-10   1.74e-04 100.0%"
        m = re.search(
            r"Total\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)",
            text,
        )
        if m:
            internal = float(m.group(1))
            switching = float(m.group(2))
            leakage = float(m.group(3))
            total = float(m.group(4))
            # Values from OpenROAD are in Watts, convert to mW
            metrics["total_power_mw"] = total * 1000.0
            metrics["dynamic_power_mw"] = (internal + switching) * 1000.0
            metrics["leakage_power_mw"] = leakage * 1000.0

    # Parse area report
    area_file = out / "area.rpt"
    if area_file.exists():
        text = area_file.read_text()
        m = re.search(r"Design area\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%", text)
        if m:
            metrics["design_area_um2"] = float(m.group(1))
            metrics["die_area_um2"] = float(m.group(2))
            metrics["utilization_pct"] = float(m.group(3))

    return metrics


# ---------------------------------------------------------------------------
# DRC report parsing + signed-off hard-macro interior exclusion
# ---------------------------------------------------------------------------

# Magic emits `drc listall why` coordinates in its INTERNAL units, which for a
# DEF imported at the sky130 manufacturing grid are 0.005 um each (um x 200) --
# NOT the DEF DBU (um/1000). VERIFIED against a real routed report vs the die
# size: a met4.4a min-area tile of 76x76 internal units is 76 * 0.005 = 0.38 um
# on a side = 0.1444 um^2 (< the 0.24 um^2 Metal4 minimum), which is exactly the
# 0.38x0.38 um met4 signal-pin footprint of the small sky130B OpenRAM SRAM. At
# um/1000 the same tile would read 0.076 um (0.0058 um^2), a physically
# impossible pin, and would mis-locate every violation -- the exact unit bug
# behind an earlier misdiagnosis. The DEF's own coordinates (COMPONENTS
# placements) are separately in DBU = UNITS DISTANCE MICRONS (typ. 1000/um) and
# are converted with that scale in `macro_bboxes_from_def`.
MAGIC_DRC_UM_PER_INTERNAL_UNIT = 0.005

# Layers a sky130 hard-macro (SRAM/ROM) LEF abstract obstructs. met5 is
# deliberately EXCLUDED from this set: the top-level PDN runs power straps over
# macros on met5, so a genuine met5-over-macro violation must still be counted
# (keeps the gate honest). Even if a macro LEF's OBS section listed met5, the
# exclusion caps here at met1-met4.
_MACRO_OBSTRUCTED_LAYERS = frozenset({"met1", "met2", "met3", "met4"})

# Orientations whose footprint keeps the LEF SIZE (w,h) as-is; the 90/270
# rotation classes swap w<->h. Covers DEF/LEF tokens and OpenROAD R/M spellings.
_ORIENT_SWAP_WH = frozenset({"E", "W", "FE", "FW", "R90", "R270"})

# A DRC error tile is a line of exactly four coordinates "x1 y1 x2 y2". The
# `drc listall why` script is LLM-authored per run, so the SAME rule is emitted
# in two coordinate conventions across runs: raw Magic INTERNAL units (integers,
# x0.005 um) OR already-scaled MICRONS (decimals, via `cif scale out`). Accept
# BOTH -- an integer-only tile regex silently dropped every micron tile, which
# made `parse_drc_report` return a FALSE CLEAN (count 0) on a micron report and
# never run the macro-interior exclusion. Presence of a decimal point in a tile
# means the report is already in um (scale 1.0); all-integer means internal units.
_DRC_TILE_RE = re.compile(r"^-?\d+(?:\.\d+)?(?:\s+-?\d+(?:\.\d+)?){3}$")


def drc_macro_interior_exclude_enabled() -> bool:
    """Whether the DRC gate drops violations that fall INSIDE a signed-off
    hard-macro's bounding box on a layer the macro obstructs (met1-met4).

    Default ON. Set ``CORESMITH_DRC_MACRO_INTERIOR_EXCLUDE=0`` to restore the
    pre-fix behavior (count every violation, including DEF+abstract artifacts:
    a hard macro stays a LEF abstract in the top-level Magic DRC, so its
    sub-min-area LEF pins are DRC'd as top-level metal even though the real,
    signed-off macro GDS -- merged into the shipped GDS -- is clean, so those
    in-interior met1-4 shapes are not on the fabricated die).
    """
    from orchestrator.profile import ensure_applied, flag_enabled
    ensure_applied()
    return flag_enabled("CORESMITH_DRC_MACRO_INTERIOR_EXCLUDE", default=True)


def _drc_rule_layer(rule: str) -> str | None:
    """Best-effort map a Magic DRC why-string to its metal layer (met1..met5).

    Recognizes Magic layer names (``met4.4a``) and human names (``Metal4``).
    Returns None when no metal layer is identifiable -- such a violation is then
    NEVER excluded (fail-closed).
    """
    if not rule:
        return None
    m = re.search(r"met(\d)", rule) or re.search(r"[Mm]etal\s*(\d)", rule)
    if m:
        return f"met{m.group(1)}"
    return None


def placed_macro_bboxes(placements) -> list[dict]:
    """Pure helper: turn placed hard macros into axis-aligned bboxes in MICRONS.

    Each placement carries the macro-placement ORIGIN (x,y in um -- the DEF /
    `macro_place` lower-left placement point), the LEF ``SIZE`` (w,h in um), an
    orientation token, and optionally the layers the macro obstructs + a tag.
    Accepts a mapping ``{x,y,w,h,orient,layers,tag/name}`` or a sequence
    ``(x, y, orient, w, h[, layers[, tag]])``.

    Returns ``[{"x1","y1","x2","y2","layers","tag"}, ...]`` in um. The bbox uses
    the placement point as the footprint's lower-left corner -- matching the
    engine's own `plan_floorplan`, which places every macro at (x,y) with bbox
    [x, x+w] x [y, y+h] at R0 -- and swaps w<->h for the 90/270 rotation classes.
    """
    out: list[dict] = []
    for p in placements:
        tag = None
        if isinstance(p, dict):
            x = float(p["x"])
            y = float(p["y"])
            w = float(p["w"])
            h = float(p["h"])
            orient = str(p.get("orient", "N")).upper()
            layers = p.get("layers")
            tag = p.get("tag") or p.get("name")
        else:
            x = float(p[0])
            y = float(p[1])
            orient = str(p[2]).upper()
            w = float(p[3])
            h = float(p[4])
            layers = p[5] if len(p) > 5 else None
            tag = p[6] if len(p) > 6 else None
        if orient in _ORIENT_SWAP_WH:
            w, h = h, w
        lset = (
            frozenset(str(lyr).lower() for lyr in layers)
            if layers else _MACRO_OBSTRUCTED_LAYERS
        )
        out.append({
            "x1": x, "y1": y, "x2": x + w, "y2": y + h,
            "layers": lset, "tag": tag,
        })
    return out


def _macro_obstructed_layers(info) -> frozenset:
    """Layers a macro obstructs, read from its LEF ``OBS`` section and capped at
    met1-met4 (met5+ dropped so a real top-PDN-over-macro met5 violation is
    still counted). Falls back to the known sky130 SRAM/ROM set (met1-met4)."""
    try:
        lef = getattr(info, "lef", "")
        if lef and Path(lef).exists():
            text = Path(lef).read_text(errors="ignore")
            mobs = re.search(r"\bOBS\b(.*?)\bEND\b", text, re.DOTALL)
            if mobs:
                layers = {
                    lyr.lower()
                    for lyr in re.findall(r"LAYER\s+(\w+)", mobs.group(1))
                }
                layers &= set(_MACRO_OBSTRUCTED_LAYERS)
                if layers:
                    return frozenset(layers)
    except Exception:  # noqa: BLE001 - OBS parse is best-effort
        pass
    return _MACRO_OBSTRUCTED_LAYERS


def macro_bboxes_from_def(def_path: str, registry: dict | None = None) -> list[dict]:
    """Build placed hard-macro bboxes (in um) from a routed DEF.

    Reads the DEF ``UNITS DISTANCE MICRONS`` scale + the ``COMPONENTS`` macro
    placements (origin + orientation), resolves each master's LEF ``SIZE`` and
    obstructed layers from the macro registry (`_parse_lef`), and returns the
    bboxes. This is the placement the engine actually handed to PnR/Magic, so it
    is authoritative for what Magic DRC'd. Only masters KNOWN to the macro
    registry (signed-off hard IP) are included; std cells / unknown components
    are ignored. Returns ``[]`` on any parse failure or when no macro is placed
    (this doubles as the "guard on macro presence" -- no macro => no exclusion).
    """
    try:
        p = Path(def_path)
        if not def_path or not p.exists():
            return []
        text = p.read_text(errors="ignore")
    except OSError:
        return []

    m = re.search(r"UNITS\s+DISTANCE\s+MICRONS\s+(\d+)", text)
    dbu = float(m.group(1)) if m else 1000.0

    if registry is None:
        try:
            from orchestrator.langgraph.macro_registry import discover_macros
            registry = discover_macros()
        except Exception:  # noqa: BLE001 - registry discovery is best-effort
            registry = {}
    if not registry:
        return []

    # Restrict to the COMPONENTS section to bound the scan on a mm-scale DEF.
    cstart = text.find("COMPONENTS")
    cend = text.find("END COMPONENTS")
    section = text[cstart:cend] if (cstart >= 0 and cend > cstart) else text

    # `- <inst> <master> ... + (FIXED|PLACED) ( x y ) <orient>` on one entry.
    # `[^;\n]*?` keeps the match on a single component entry (no DOTALL) so a
    # 20 MB DEF cannot trigger catastrophic backtracking.
    comp_re = re.compile(
        r"-\s+\S+\s+(\S+)[^;\n]*?\+\s+(?:FIXED|PLACED)\s+"
        r"\(\s*(-?\d+)\s+(-?\d+)\s*\)\s+(\w+)"
    )
    obs_cache: dict[str, frozenset] = {}
    placements: list[dict] = []
    for cm in comp_re.finditer(section):
        master, xs, ys, orient = cm.group(1), cm.group(2), cm.group(3), cm.group(4)
        info = registry.get(master)
        if info is None or not getattr(info, "width_um", 0) or not getattr(info, "height_um", 0):
            continue
        layers = obs_cache.get(master)
        if layers is None:
            layers = _macro_obstructed_layers(info)
            obs_cache[master] = layers
        placements.append({
            "x": float(xs) / dbu,
            "y": float(ys) / dbu,
            "w": info.width_um,
            "h": info.height_um,
            "orient": orient,
            "layers": layers,
            "tag": master,
        })
    return placed_macro_bboxes(placements)


def _violation_in_macro_interior(
    record: dict,
    macro_bboxes: list[dict],
    um_per_unit: float = MAGIC_DRC_UM_PER_INTERNAL_UNIT,
):
    """Return the tag of the placed macro whose interior contains this
    coordinate-bearing violation's CENTER on a layer that macro obstructs; else
    None. The record's coords are scaled to um by ``um_per_unit`` --
    ``MAGIC_DRC_UM_PER_INTERNAL_UNIT`` for a Magic INTERNAL-unit report, or 1.0
    when the report already emits microns (see `parse_drc_report`).
    """
    rect = record.get("rect")
    if not rect:
        return None
    layer = record.get("layer")
    if not layer:
        return None  # unknown layer -> never exclude (fail-closed)
    x1, y1, x2, y2 = rect
    cx = (x1 + x2) / 2.0 * um_per_unit
    cy = (y1 + y2) / 2.0 * um_per_unit
    for i, b in enumerate(macro_bboxes):
        layers = b.get("layers") or _MACRO_OBSTRUCTED_LAYERS
        if layer not in layers:
            continue
        if b["x1"] <= cx <= b["x2"] and b["y1"] <= cy <= b["y2"]:
            return b.get("tag") or f"macro#{i}"
    return None


def parse_drc_report(report_path: str, macro_bboxes: list[dict] | None = None) -> dict:
    """Parse a Magic DRC report into a verdict.

    Returns ``{clean, violation_count, violations}`` and, when an exclusion
    runs, also ``{excluded_count, excluded_detail}``.

    Understands three report shapes:
      1. Magic native ``drc listall why <file>`` (what the backend DRC step
         writes, and the only coordinate-bearing form)::

             <cellname> <count>
             ----------------------------------------
             <why-string>
             ----------------------------------------
              x1 y1 x2 y2        # one error tile per line
              ...

         The tile coordinates come in EITHER of two conventions -- the DRC
         script is LLM-authored per run: raw Magic INTERNAL units (integers,
         0.005 um each) or already-scaled MICRONS (decimals, via
         ``cif scale out``). Both are parsed; a decimal in any tile flags the
         whole report as microns so the macro-interior test scales correctly.
         (An integer-only tile regex previously dropped every micron tile,
         silently returning a false-clean count of 0 on a micron report.)
      2. ``Design: X`` / ``DRC count: N``  -- explicit numeric count.
      3. ``{rule} {coords} ...``           -- Tcl-list brace form.

    When ``macro_bboxes`` is supplied (list of ``{x1,y1,x2,y2,layers,tag}`` in
    um, e.g. from `macro_bboxes_from_def`) AND
    ``CORESMITH_DRC_MACRO_INTERIOR_EXCLUDE`` is ON, any coordinate-bearing
    violation whose CENTER (converted internal-units -> um) falls inside a macro
    bbox on a layer that macro obstructs (met1-met4) is DROPPED and not counted
    -- a signed-off hard-macro interior artifact. met5 is never excluded (top
    PDN runs over macros there), so a genuine met5-over-macro violation, and any
    violation outside every macro, still count. The exclusion is logged (count +
    per-macro), never silent. ``macro_bboxes=None`` -> pre-fix counting.

    Note: for the native form the count is derived from the parsed error tiles
    (so per-tile exclusions subtract exactly); Magic's header count can differ
    from the tile count by boundary tiles, but the gate only cares 0 vs nonzero.
    """
    p = Path(report_path)
    if not p.exists():
        return {"clean": False, "violation_count": -1, "violations": []}

    text = p.read_text()
    lines = text.split("\n")

    # Native "<cellname> <count>" header is only the FIRST non-empty line; a
    # later why-string that happens to look like "word 5" must not be dropped.
    native_count: int | None = None
    start_idx = 0
    for idx, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        hm = re.match(r"^(\S+)\s+(\d+)$", s)
        if hm and not s.startswith(("Design:", "DRC count:")):
            native_count = int(hm.group(2))
            start_idx = idx + 1
        break

    records: list[dict] = []
    cur_rule: str | None = None
    coords_in_um = False  # set once any tile is emitted in microns (has a decimal)
    for line in lines[start_idx:]:
        s = line.strip()
        if not s:
            continue
        if set(s) <= {"-"}:  # dashed separator line
            continue
        if _DRC_TILE_RE.match(s):  # error tile "x1 y1 x2 y2" (int OR decimal-um)
            if "." in s:
                coords_in_um = True
            nums = [float(t) for t in s.split()]
            records.append({
                "rule": cur_rule,
                "layer": _drc_rule_layer(cur_rule or ""),
                "rect": (nums[0], nums[1], nums[2], nums[3]),
            })
            continue
        if "{" in line:  # Tcl brace form: {rule} {coords} ...
            for rm in re.finditer(r"\{([^{}]+)\}\s*\{", line):
                r = rm.group(1).strip()
                records.append({"rule": r, "layer": _drc_rule_layer(r), "rect": None})
            continue
        if s.startswith("Design:") or s.startswith("DRC count:"):
            continue
        cur_rule = s  # a why-string / rule header

    m = re.search(r"DRC count:\s*(\d+)\s*$", text, re.MULTILINE)
    explicit = int(m.group(1)) if m else native_count

    coord_recs = [r for r in records if r["rect"] is not None]
    noncoord_recs = [r for r in records if r["rect"] is None and r["rule"]]

    if coord_recs:
        pre_count = len(coord_recs)
    elif explicit is not None:
        pre_count = explicit
    elif noncoord_recs:
        pre_count = len(noncoord_recs)
    else:
        pre_count = 0

    excluded: list[dict] = []
    kept_coord = coord_recs
    do_exclude = (
        bool(macro_bboxes)
        and bool(coord_recs)
        and drc_macro_interior_exclude_enabled()
    )
    if do_exclude:
        # Micron report -> coords are already um (scale 1.0); internal-unit
        # report -> scale by 0.005 um/unit. Detected from the parsed tiles.
        um_per_unit = 1.0 if coords_in_um else MAGIC_DRC_UM_PER_INTERNAL_UNIT
        kept_coord = []
        for r in coord_recs:
            hit = _violation_in_macro_interior(r, macro_bboxes, um_per_unit)
            if hit is not None:
                excluded.append({**r, "excluded_by": hit})
            else:
                kept_coord.append(r)

    post_count = max(pre_count - len(excluded), 0)
    kept_rules = [r["rule"] for r in (kept_coord + noncoord_recs) if r["rule"]]

    result: dict = {
        "clean": post_count == 0 and not kept_rules,
        "violation_count": post_count,
        "violations": kept_rules[:50],
    }
    if do_exclude:
        from collections import Counter
        per_macro = Counter(r["excluded_by"] for r in excluded)
        result["excluded_count"] = len(excluded)
        result["excluded_detail"] = dict(per_macro)
        if excluded:
            detail = ", ".join(f"{n}x {tag}" for tag, n in per_macro.items())
            log(
                f"  [DRC] excluded {len(excluded)} in-macro-interior met1-4 "
                f"artifact(s) across {len(per_macro)} placed macro(s) "
                f"[{detail}]; {post_count} violation(s) remain",
                GREEN,
            )
    return result


def parse_pnr_stdout(stdout: str) -> dict:
    """Parse key metrics directly from OpenROAD stdout.

    This is a fallback when report files aren't available. Extracts
    design area, WNS, TNS, and power from the SUMMARY section.
    """
    metrics: dict = {
        "design_area_um2": 0.0,
        "utilization_pct": 0.0,
        "wns_ns": 0.0,
        "tns_ns": 0.0,
        "total_power_mw": 0.0,
        "wire_length_um": 0,
        "via_count": 0,
    }

    # Format 1: "Design area  955  1830  49%" (3-column)
    m = re.search(r"Design area\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%", stdout)
    if m:
        metrics["design_area_um2"] = float(m.group(1))
        metrics["utilization_pct"] = float(m.group(3))
    else:
        # Format 2: "Design area 955 um^2 49% utilization."
        m = re.search(r"Design area\s+([\d.]+)\s+um\^?2?\s+([\d.]+)%", stdout)
        if m:
            metrics["design_area_um2"] = float(m.group(1))
            metrics["utilization_pct"] = float(m.group(2))

    m = re.search(r"wns\s+(?:max\s+)?(-?[\d.]+)", stdout, re.IGNORECASE)
    if m:
        metrics["wns_ns"] = float(m.group(1))

    m = re.search(r"tns\s+(?:max\s+)?(-?[\d.]+)", stdout, re.IGNORECASE)
    if m:
        metrics["tns_ns"] = float(m.group(1))

    m = re.search(
        r"Total\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)",
        stdout,
    )
    if m:
        metrics["total_power_mw"] = float(m.group(4)) * 1000.0

    m = re.search(r"Total wire length\s*=\s*([\d.]+)\s*um", stdout)
    if m:
        metrics["wire_length_um"] = int(float(m.group(1)))

    m = re.search(r"Total number of vias\s*=\s*(\d+)", stdout)
    if m:
        metrics["via_count"] = int(m.group(1))

    return metrics


# A concrete DRC error rect: Tcl-brace form ``{x1 y1 x2 y2}`` (what
# ``drc listall why`` writes) OR a bare ``x1 y1 x2 y2`` tile line. Ints
# (Magic internal units) or decimals (microns) both count.
_DRC_RECT_BRACE_RE = re.compile(
    r"\{\s*-?\d+(?:\.\d+)?(?:\s+-?\d+(?:\.\d+)?){3}\s*\}"
)


def drc_report_fallback_enabled() -> bool:
    """Whether a blank/zero stdout DRC count falls back to counting the rects in
    the report file. Default ON.

    Magic emits an empty ``DRC violations:`` line (and a blank ``DRC count:`` in
    the report header) when ``drc listall count`` returns nothing, even though
    ``drc listall why`` still writes every violation rect to the report -- so the
    stdout-only parser reports a FALSE-CLEAN 0 over a report holding thousands of
    real violations. Being default-on closes that honesty hole; it only ever
    fires when the report actually contains rects, so a genuinely clean report is
    unaffected. Set ``CORESMITH_DRC_REPORT_FALLBACK=0`` to restore the raw
    stdout-only count.
    """
    from orchestrator.profile import ensure_applied, flag_enabled
    ensure_applied()
    return flag_enabled("CORESMITH_DRC_REPORT_FALLBACK", default=True)


def _count_drc_report_violations(report_text: str) -> int:
    """Count concrete DRC violation rects in a Magic ``drc listall why`` report.

    Prefers the ``{x1 y1 x2 y2}`` Tcl-brace tuples (the coordinate-bearing form);
    falls back to bare ``x1 y1 x2 y2`` tile lines; else counts the per-rule-class
    ``why``-string entries. Returns 0 for a genuinely clean report (no rects).
    """
    if not report_text:
        return 0
    braces = _DRC_RECT_BRACE_RE.findall(report_text)
    if braces:
        return len(braces)
    tiles = 0
    rule_entries = 0
    for line in report_text.splitlines():
        s = line.strip()
        if not s or set(s) <= {"-"}:
            continue
        if s.startswith(("Design:", "DRC count:")):
            continue
        if _DRC_TILE_RE.match(s):
            tiles += 1
        else:
            rule_entries += 1  # a why-string / rule header naming a violation
    return tiles or rule_entries


def _parse_magic_drc_count(stdout: str, report_text: str = "") -> int:
    """Extract DRC violation count from Magic stdout.

    When the stdout count is blank/absent or zero and ``report_text`` is
    supplied, FALL BACK to counting the concrete violation rects in the report
    (gated by ``CORESMITH_DRC_REPORT_FALLBACK``, default ON) -- Magic prints an
    empty ``DRC violations:`` when ``drc listall count`` returns nothing, so the
    stdout-only count would falsely read 0 over a report full of real rects.
    Never returns a false-clean 0 while the report holds rects; a genuinely clean
    report still returns 0.
    """
    count = -1
    for line in reversed(stdout.split("\n")):
        m = re.search(r"DRC violations:\s*(\d+)", line)
        if m:
            count = int(m.group(1))
            break
        # Magic prints "DRC violations: " (empty) when the count is blank.
        if re.match(r"DRC violations:\s*$", line.strip()):
            count = 0
            break
    if count < 0:
        # Fallback: look for "DRC count:"
        m = re.search(r"DRC count:\s*(\d+)", stdout)
        if m:
            count = int(m.group(1))
        elif "DRC violations:" in stdout:
            count = 0
    # Honesty fallback: a blank/zero stdout count must not mask real rects.
    if count <= 0 and report_text and drc_report_fallback_enabled():
        rpt = _count_drc_report_violations(report_text)
        if rpt > 0:
            return rpt
    return count


def _parse_lvs_deltas(stdout: str) -> tuple[int, int]:
    """Extract device and net count deltas from Netgen stdout.

    Returns (device_delta, net_delta) where 0 means match.
    """
    device_delta = 0
    net_delta = 0

    # Look for "N devices" in circuit comparison
    devices = re.findall(r"(\d+)\s+devices", stdout)
    if len(devices) >= 2:
        device_delta = abs(int(devices[0]) - int(devices[1]))

    nets = re.findall(r"(\d+)\s+nets", stdout)
    if len(nets) >= 2:
        net_delta = abs(int(nets[0]) - int(nets[1]))

    return device_delta, net_delta


# ---------------------------------------------------------------------------
# High-level convenience functions (called by graph nodes)
# ---------------------------------------------------------------------------

def pnr_route_drc_gate_enabled() -> bool:
    """Honest gate: a routed design that OpenROAD left with unresolved
    detailed-route DRC violations is NOT a passing PnR.

    ``CORESMITH_PNR_ROUTE_DRC_GATE`` (default ON, matching the honest-by-default
    synth/DRC/cell-count gates). Set it to ``0``/``false``/``no``/``off`` to
    restore the pre-fix behavior (PnR success reported unconditionally).
    """
    from orchestrator.profile import ensure_applied, flag_enabled
    ensure_applied()
    return flag_enabled("CORESMITH_PNR_ROUTE_DRC_GATE", default=True)


def count_route_drc_violations(route_drc_path) -> int:
    """Robustly count detailed-route DRC violation ENTRIES in an OpenROAD
    ``route_drc.rpt``.

    OpenROAD's ``detailed_route -output_drc <file>`` writes one entry per
    violation as a block that BEGINS with a marker line::

        violation type: <TYPE>
          srcs: <net/obj> <net/obj>
          bbox = ( <x1> <y1> ) - ( <x2> <y2> ) on Layer <layer>

    We count the ``violation type:`` marker lines -- the true per-entry count.
    The pre-fix code used ``content.count("violation")`` (a raw substring count
    of the word "violation"), which mis-counts: it double-counts any detail
    text mentioning "violation" and can also under/over-count against summary
    headers. A missing or empty report -> 0 (a clean route still passes).
    """
    p = Path(route_drc_path)
    if not p.exists():
        return 0
    try:
        content = p.read_text()
    except OSError:
        return 0
    n = 0
    for line in content.splitlines():
        if line.strip().lower().startswith("violation type:"):
            n += 1
    return n


def run_pnr_flow(
    block_name: str,
    netlist_path: str,
    sdc_path: str,
    output_dir: str,
    attempt: int = 1,
    utilization: int = 45,
    density: float = 0.6,
    timeout: int = 1800,
    gate_count: int = 0,
) -> dict:
    """Run complete PnR flow deterministically and return structured results.

    LEGACY: The backend graph now uses the LLM-driven flow via
    ``prepare_pnr_working_copy()`` + ``_run_llm_eda_step()``.
    This function is retained for ``run_step()`` debugging and tests.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tcl_path = generate_pnr_tcl(
        block_name, netlist_path, sdc_path, output_dir,
        utilization=utilization, density=density, gate_count=gate_count,
    )

    log(f"  [PNR] Running OpenROAD PnR flow for {block_name}...", YELLOW)
    result = run_openroad(tcl_path, block_name, "pnr", attempt, timeout)

    if not result["success"]:
        error = result.get("stderr", "") or result.get("stdout", "")
        log(f"  [PNR] FAILED: {error[:200]}", RED)
        return {
            "success": False,
            "error": error[-3000:],
            "log_path": result.get("log_path", ""),
        }

    # Parse reports
    metrics = parse_openroad_reports(output_dir)
    # Also parse stdout for metrics not in report files
    stdout_metrics = parse_pnr_stdout(result.get("stdout", ""))

    # Merge: report files take priority, stdout as fallback
    for k, v in stdout_metrics.items():
        if k not in metrics or (metrics.get(k) == 0.0 and v != 0.0):
            metrics[k] = v

    # Check for routing DRC file. Robustly count detailed-route DRC violation
    # ENTRIES (one "violation type:" marker per violation), not a raw substring
    # count of the word "violation".
    route_drc = out / "route_drc.rpt"
    route_drc_violations = count_route_drc_violations(route_drc)

    routed_def = str(out / f"{block_name}_routed.def")
    pnr_verilog = str(out / f"{block_name}_pnr.v")
    pwr_verilog = str(out / f"{block_name}_pwr.v")
    spef_path = str(out / f"{block_name}.spef")

    log(f"  [PNR] Complete: area={metrics['design_area_um2']:.0f} um², "
        f"WNS={metrics['wns_ns']:.2f} ns, "
        f"power={metrics['total_power_mw']:.3f} mW", GREEN)

    # Best-effort: render floorplan image from routed DEF
    img_dir = PROJECT_ROOT / ".coresmith" / "images"
    floorplan_png = str(img_dir / f"{block_name}_floorplan.png")
    render_layout_image(routed_def, floorplan_png)

    # Honest gate: detailed routing that left unresolved DRC markers is NOT a
    # passing PnR (same false-pass class as the synth/DRC honest gates). When
    # the gate is ON (default) and route_drc.rpt has >0 violation entries,
    # report success=False with an actionable error + the entry count so the
    # caller routes to diagnose/fail instead of signing off a die with open
    # DRC markers. Gate OFF restores the pre-fix behavior (success uncondit-
    # ional) so both branches are testable.
    pnr_success = True
    route_drc_error = ""
    if pnr_route_drc_gate_enabled() and route_drc_violations > 0:
        pnr_success = False
        route_drc_error = (
            f"Route-DRC gate: detailed routing left {route_drc_violations} "
            f"unresolved DRC violation(s) in {route_drc}. A routed design with "
            f"open detailed-route DRC markers is not a passing PnR -- refusing "
            f"to report success. Re-route to convergence (raise detailed_route "
            f"iterations / fix the offending nets), or disable the gate with "
            f"CORESMITH_PNR_ROUTE_DRC_GATE=0."
        )
        log(f"  [PNR] {route_drc_error}", RED)

    result_out = {
        "success": pnr_success,
        "routed_def_path": routed_def,
        "pnr_verilog_path": pnr_verilog,
        "pwr_verilog_path": pwr_verilog,
        "spef_path": spef_path,
        "route_drc_violations": route_drc_violations,
        "floorplan_image": floorplan_png if Path(floorplan_png).exists() else "",
        "log_path": result.get("log_path", ""),
        **metrics,
    }
    if route_drc_error:
        result_out["error"] = route_drc_error
    return result_out


def run_drc_flow(
    block_name: str,
    routed_def_path: str,
    output_dir: str,
    attempt: int = 1,
    timeout: int = 600,
) -> dict:
    """Run Magic DRC + GDS + SPICE extraction.

    Returns structured results including DRC count and artifact paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tcl_path = generate_drc_tcl(block_name, routed_def_path, output_dir)

    log(f"  [DRC] Running Magic DRC for {block_name}...", YELLOW)
    result = run_magic(tcl_path, block_name, "drc", attempt, timeout)

    if not result["success"]:
        error = result.get("stderr", "") or result.get("stdout", "")
        log(f"  [DRC] FAILED: {error[:200]}", RED)
        return {
            "clean": False,
            "violation_count": -1,
            "error": error[-2000:],
            "log_path": result.get("log_path", ""),
        }

    drc_count = result["drc_count"]
    gds_path = result["gds_path"]
    spice_path = result["spice_path"]

    if drc_count == 0:
        log("  [DRC] Clean -- no violations", GREEN)
    else:
        log(f"  [DRC] {drc_count} violations found", RED)

    # Best-effort: render GDS layout image
    img_dir = PROJECT_ROOT / ".coresmith" / "images"
    gds_png = str(img_dir / f"{block_name}_gds.png")
    if gds_path and Path(gds_path).exists():
        render_layout_image(gds_path, gds_png)

    return {
        "clean": drc_count == 0,
        "violation_count": drc_count,
        "gds_path": gds_path,
        "spice_path": spice_path,
        "gds_image": gds_png if Path(gds_png).exists() else "",
        "drc_report_path": result.get("drc_report_path", ""),
        "log_path": result.get("log_path", ""),
    }


def run_lvs_flow(
    block_name: str,
    spice_path: str,
    verilog_path: str,
    output_dir: str,
    attempt: int = 1,
    timeout: int = 600,
) -> dict:
    """Run Netgen LVS comparison.

    Returns structured results.
    """
    report_path = str(Path(output_dir) / f"lvs_{block_name}.rpt")

    log(f"  [LVS] Running Netgen LVS for {block_name}...", YELLOW)
    result = run_netgen_lvs(
        spice_path, verilog_path, block_name,
        report_path=report_path, attempt=attempt, timeout=timeout,
    )

    if result["match"]:
        log("  [LVS] Match", GREEN)
    else:
        log(f"  [LVS] Mismatch: device_delta={result['device_delta']}, "
            f"net_delta={result['net_delta']}", RED)

    return result
