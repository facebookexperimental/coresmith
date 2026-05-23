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
    PROJECT_ROOT,
    PDK_ROOT,
    _write_step_log,
    _write_step_log_error,
    load_config,
    log,
    GREEN,
    RED,
    YELLOW,
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

    script = f"""# Flat top-level synthesis for {design_name} (Sky130 HD)
# Generated by coresmith backend_helpers.generate_flat_synthesis_script

{reads}

hierarchy -check -top {top_module}
proc; opt; fsm; opt; memory; opt
techmap; opt
dfflibmap -liberty {LIBERTY}
abc -liberty {LIBERTY}
clean
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

    # Extract actual top-level module name from the netlist to avoid mangled
    # PRD title slugs (e.g. prd___h_264_..._top vs h264_encode_pipeline_top).
    actual_module = block_name
    try:
        with open(netlist_path, encoding="utf-8", errors="replace") as _nf:
            for _line in _nf:
                _m = re.match(r'\s*module\s+(\w+)', _line)
                if _m:
                    actual_module = _m.group(1)
                    break
    except OSError:
        pass

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


def prepare_pnr_working_copy(
    design_name: str,
    netlist_path: str,
    sdc_path: str,
    output_dir: str,
    utilization: int = 35,
    density: float = 0.6,
) -> str:
    """Copy the reference PnR TCL template and prepend design-specific variables.

    The LLM agent reads, modifies, and runs this working copy.
    Returns the path to the working TCL script.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    abs_netlist = str(Path(netlist_path).resolve()) if not os.path.isabs(netlist_path) else netlist_path
    abs_sdc = str(Path(sdc_path).resolve()) if not os.path.isabs(sdc_path) else sdc_path

    actual_module = design_name
    try:
        with open(netlist_path, encoding="utf-8", errors="replace") as _nf:
            for _line in _nf:
                _m = re.match(r'\s*module\s+(\w+)', _line)
                if _m:
                    actual_module = _m.group(1)
                    break
    except OSError:
        pass

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
) -> str:
    """Generate a Magic DRC + GDS + hierarchical SPICE extraction Tcl script.

    Returns the path to the generated .tcl file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tcl_path = out / f"drc_{block_name}.tcl"

    abs_def = str(Path(routed_def_path).resolve()) if not os.path.isabs(routed_def_path) else routed_def_path

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

# Read the routed DEF
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
    cmd = [OPENROAD_BIN, tcl_script]

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
        "timing_met": True,
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


def parse_drc_report(report_path: str) -> dict:
    """Parse Magic DRC report.

    Returns dict with: clean, violation_count, violations.

    Magic's report format has two variants:
      1. ``DRC count: 42``  -- explicit numeric count
      2. ``DRC count:\\n{rule} {{coords}} ...`` -- violations inline, no number

    We try the numeric form first, then fall back to counting ``{rule_name}``
    entries in the text after the ``DRC count:`` header.
    """
    p = Path(report_path)
    if not p.exists():
        return {"clean": False, "violation_count": -1, "violations": []}

    text = p.read_text()
    violations: list[str] = []

    m = re.search(r"DRC count:\s*(\d+)\s*$", text, re.MULTILINE)
    if m:
        count = int(m.group(1))
    else:
        count = len(re.findall(
            r"\{[A-Za-z][\w\s.<>()]+\}", text,
        ))

    for line in text.split("\n"):
        line = line.strip()
        if line and not line.startswith("Design:") and not line.startswith("DRC count:"):
            for rm in re.finditer(r"\{([^{}]+)\}\s*\{", line):
                violations.append(rm.group(1).strip())

    return {
        "clean": count == 0 and not violations,
        "violation_count": count if count else len(violations),
        "violations": violations[:50],
    }


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


def _parse_magic_drc_count(stdout: str) -> int:
    """Extract DRC violation count from Magic stdout."""
    for line in reversed(stdout.split("\n")):
        m = re.search(r"DRC violations:\s*(\d+)", line)
        if m:
            return int(m.group(1))
        # Magic prints "DRC violations: " (empty) when count is 0
        if re.match(r"DRC violations:\s*$", line.strip()):
            return 0
    # Fallback: look for "DRC count:"
    m = re.search(r"DRC count:\s*(\d+)", stdout)
    if m:
        return int(m.group(1))
    # If "DRC violations:" appears at all without a number, assume 0
    if "DRC violations:" in stdout:
        return 0
    return -1


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

    # Check for routing DRC file
    route_drc = out / "route_drc.rpt"
    route_drc_violations = 0
    if route_drc.exists():
        content = route_drc.read_text().strip()
        if content:
            route_drc_violations = content.count("violation")

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

    return {
        "success": True,
        "routed_def_path": routed_def,
        "pnr_verilog_path": pnr_verilog,
        "pwr_verilog_path": pwr_verilog,
        "spef_path": spef_path,
        "route_drc_violations": route_drc_violations,
        "floorplan_image": floorplan_png if Path(floorplan_png).exists() else "",
        "log_path": result.get("log_path", ""),
        **metrics,
    }


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
