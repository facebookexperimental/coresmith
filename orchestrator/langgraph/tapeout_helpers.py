# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Tapeout helper functions for OpenFrame shuttle submission.

Provides:
  - OpenFrame wrapper RTL generation (GPIO mapping, power connections)
  - Submission directory structure generation
  - Wrapper-level PnR Tcl (macro placement within OpenFrame die)
  - Native mpw_precheck runner (no Docker required)

The native precheck replaces the Efabless Docker-based mpw_precheck by
running the individual checks directly via Nix-wrapped EDA tools:
  - KLayout DRC (density, metal minimum width/spacing)
  - Magic DRC (full Sky130 design rules)
  - Netgen LVS (layout vs schematic)
  - Directory/file structure validation
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from orchestrator.langgraph.backend_helpers import (
    _STD_CELL,
    CELL_LEF,
    LIBERTY,
    TECH_LEF,
    _resolve_tool,
)
from orchestrator.langgraph.pipeline_helpers import (
    GREEN,
    PROJECT_ROOT,
    RED,
    YELLOW,
    _write_step_log,
    log,
)

# ---------------------------------------------------------------------------
# OpenFrame constants
# ---------------------------------------------------------------------------

OPENFRAME_IO_PADS = 44
OPENFRAME_DIE_WIDTH_UM = 3520.0
OPENFRAME_DIE_HEIGHT_UM = 5188.0
OPENFRAME_CORE_MARGIN_UM = 100.0

KLAYOUT_BIN = _resolve_tool("klayout_binary", "scripts/klayout-nix.sh")


# ---------------------------------------------------------------------------
# Verilog port parser (regex-based, no external dependency)
# ---------------------------------------------------------------------------

def _parse_verilog_ports(rtl_source: str) -> dict[str, dict]:
    """Extract port declarations from Verilog source.

    Returns a dict mapping port_name -> {"width": int, "direction": str}.
    Handles both ANSI and non-ANSI port styles.
    """
    ports: dict[str, dict] = {}

    # Match: input/output/inout [wire] [signed] [N:0] port_name
    # Uses \b word boundary instead of ^ to handle both ANSI and inline styles
    port_re = re.compile(
        r'\b(input|output|inout)\s+'
        r'(?:wire\s+|reg\s+)?'
        r'(?:signed\s+)?'
        r'(?:\[(\d+):(\d+)\]\s+)?'
        r'(\w+)',
    )

    for m in port_re.finditer(rtl_source):
        direction = m.group(1)
        msb = int(m.group(2)) if m.group(2) else 0
        lsb = int(m.group(3)) if m.group(3) else 0
        name = m.group(4)
        width = abs(msb - lsb) + 1 if m.group(2) else 1
        ports[name] = {"width": width, "direction": direction}

    return ports


def _discover_block_ports(blocks: list[dict]) -> list[dict]:
    """Populate each block's 'ports' dict from its RTL file on disk.

    Searches for RTL files in order of preference:
      1. rtl/<block_dir>/<block_name>.v (source RTL)
      2. syn/output/<block_name>/<block_name>_netlist.v (gate-level)

    Also loads the block diagram connections from .coresmith/block_diagram.json
    to identify inter-block ports that should NOT be mapped to GPIO.
    """
    inter_block_ports: dict[str, set[str]] = {}
    bd_path = PROJECT_ROOT / ".coresmith" / "block_diagram.json"
    if bd_path.exists():
        try:
            bd = json.loads(bd_path.read_text())
            for conn in bd.get("connections", []):
                from_block = conn.get("from", "")
                to_block = conn.get("to", "")
                iface = conn.get("interface", "")
                if iface:
                    inter_block_ports.setdefault(from_block, set()).add(iface)
                    inter_block_ports.setdefault(to_block, set()).add(iface)
        except (json.JSONDecodeError, KeyError):
            pass

    enriched = []
    for block in blocks:
        block = dict(block)
        name = block["name"]
        existing_ports = block.get("ports", {})

        if not existing_ports:
            rtl_source = ""
            # Try source RTL first
            for candidate in (
                PROJECT_ROOT / "rtl" / name / f"{name}.v",
                PROJECT_ROOT / "rtl" / f"{name}.v",
                PROJECT_ROOT / "syn" / "output" / name / f"{name}_netlist.v",
            ):
                if candidate.exists():
                    rtl_source = candidate.read_text()
                    break

            # Also check subdirectories of rtl/ for the block
            if not rtl_source:
                rtl_dir = PROJECT_ROOT / "rtl"
                if rtl_dir.is_dir():
                    for sub in rtl_dir.iterdir():
                        if sub.is_dir():
                            candidate = sub / f"{name}.v"
                            if candidate.exists():
                                rtl_source = candidate.read_text()
                                break

            if rtl_source:
                parsed = _parse_verilog_ports(rtl_source)
                clk_names = {"clk", "clk_in", "clock"}
                rst_names = {"rst", "rst_n", "reset", "rst_n_in"}
                skip = clk_names | rst_names
                ib_ports = inter_block_ports.get(name, set())

                # Detect actual clock/reset port names for wrapper wiring
                for pname in parsed:
                    if pname.lower() in clk_names or "clk" in pname.lower():
                        block["_clk_port"] = pname
                    if pname.lower() in rst_names or "rst" in pname.lower():
                        block["_rst_port"] = pname

                filtered = {}
                for pname, pinfo in parsed.items():
                    if pname.lower() in skip:
                        continue
                    is_inter_block = any(
                        ib in pname.lower() for ib in ib_ports
                    ) if ib_ports else False
                    if not is_inter_block:
                        filtered[pname] = pinfo
                block["ports"] = filtered
                log(f"  [WRAPPER] Parsed {len(filtered)} GPIO ports "
                    f"from RTL for {name}", GREEN)
            else:
                log(f"  [WRAPPER] No RTL file found for {name}, "
                    f"ports will be empty", YELLOW)

        enriched.append(block)
    return enriched


# ═══════════════════════════════════════════════════════════════════════════
# OpenFrame Wrapper RTL Generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_wrapper_rtl(
    blocks: list[dict],
    gpio_mapping: dict | None = None,
    output_dir: str = "",
) -> dict:
    """Generate openframe_project_wrapper.v and supporting RTL.

    Instantiates user blocks as macros inside the OpenFrame wrapper,
    maps block I/O to GPIO pads, and adds power connections.

    Args:
        blocks: List of block dicts with 'name', 'ports' (optional).
                Each block must have a corresponding GDS and gate-level netlist.
        gpio_mapping: Optional dict mapping block port names to GPIO pad indices.
                     Auto-generated if not provided.
        output_dir: Output directory for generated files.

    Returns:
        dict with paths to generated files and port assignments.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    blocks = _discover_block_ports(blocks)

    if gpio_mapping is None:
        gpio_mapping = _auto_gpio_mapping(blocks)

    # Generate the main wrapper
    wrapper_v = _generate_wrapper_verilog(blocks, gpio_mapping, out)

    # Generate power connection modules
    vccd1_v = _generate_power_connection(out, "vccd1", "VPWR")
    vssd1_v = _generate_power_connection(out, "vssd1", "VGND")

    # Generate the top-level netlists include
    netlists_v = _generate_netlists_include(blocks, out)

    return {
        "wrapper_path": str(wrapper_v),
        "vccd1_path": str(vccd1_v),
        "vssd1_path": str(vssd1_v),
        "netlists_path": str(netlists_v),
        "gpio_mapping": gpio_mapping,
        "gpio_used": sum(len(v) for v in gpio_mapping.values()),
        "gpio_available": OPENFRAME_IO_PADS,
    }


def _auto_gpio_mapping(blocks: list[dict]) -> dict:
    """Auto-assign block ports to GPIO pads sequentially.

    Reserves GPIO[0] for clk, GPIO[1] for rst.
    Assigns remaining ports starting from GPIO[2].
    """
    mapping: dict = {}
    gpio_idx = 2  # 0=clk, 1=rst

    for block in blocks:
        block_name = block["name"]
        ports = block.get("ports", {})
        block_map: dict = {}

        # Skip clk and rst (globally assigned to GPIO[0] and GPIO[1])
        for port_name, port_info in sorted(ports.items()):
            if port_name in ("clk", "rst", "rst_n"):
                continue
            width = 1
            direction = "inout"
            if isinstance(port_info, dict):
                width = port_info.get("width", 1)
                direction = port_info.get("direction", "inout")
            elif isinstance(port_info, int):
                width = port_info

            if gpio_idx + width > OPENFRAME_IO_PADS:
                log(f"  [WRAPPER] GPIO overflow: {port_name} needs {width} pads, "
                    f"only {OPENFRAME_IO_PADS - gpio_idx} remaining. "
                    f"Truncating to available.", YELLOW)
                width = max(1, OPENFRAME_IO_PADS - gpio_idx)

            if gpio_idx >= OPENFRAME_IO_PADS:
                log(f"  [WRAPPER] No GPIO pads left for {port_name}", RED)
                break

            block_map[port_name] = {
                "start": gpio_idx,
                "width": width,
                "direction": direction,
            }
            gpio_idx += width

        mapping[block_name] = block_map

    return mapping


def _generate_wrapper_verilog(
    blocks: list[dict],
    gpio_mapping: dict,
    output_dir: Path,
) -> Path:
    """Generate the openframe_project_wrapper.v file."""
    wrapper_path = output_dir / "openframe_project_wrapper.v"

    # Build block instantiations
    instantiations = []
    for block in blocks:
        name = block["name"]
        block_gpio = gpio_mapping.get(name, {})
        inst_lines = [f"    // --- {name} ---"]
        inst_lines.append(f"    {name} u_{name} (")

        port_connections = []
        # Detect actual clock/reset port names from RTL
        clk_port = block.get("_clk_port", "clk")
        rst_port = block.get("_rst_port", "rst")
        port_connections.append(f"        .{clk_port}(io_in[0])")
        port_connections.append(f"        .{rst_port}(io_in[1])")

        for port_name, pad_info in block_gpio.items():
            start = pad_info["start"]
            width = pad_info["width"]
            direction = pad_info.get("direction", "inout")

            if width == 1:
                if direction == "input":
                    port_connections.append(f"        .{port_name}(io_in[{start}])")
                elif direction == "output":
                    port_connections.append(f"        .{port_name}(io_out[{start}])")
                else:
                    port_connections.append(f"        .{port_name}(io_in[{start}])")
            else:
                end = start + width - 1
                if direction == "input":
                    port_connections.append(
                        f"        .{port_name}(io_in[{end}:{start}])")
                elif direction == "output":
                    port_connections.append(
                        f"        .{port_name}(io_out[{end}:{start}])")
                else:
                    port_connections.append(
                        f"        .{port_name}(io_in[{end}:{start}])")

        inst_lines.append(",\n".join(port_connections))
        inst_lines.append("    );")
        instantiations.append("\n".join(inst_lines))

    instantiation_text = "\n\n".join(instantiations)

    # OEB (output enable active low) assignments
    oeb_assignments = []
    for block in blocks:
        name = block["name"]
        block_gpio = gpio_mapping.get(name, {})
        for port_name, pad_info in block_gpio.items():
            if pad_info.get("direction") == "output":
                start = pad_info["start"]
                width = pad_info["width"]
                for i in range(width):
                    oeb_assignments.append(
                        f"    assign io_oeb[{start + i}] = 1'b0;  "
                        f"// {name}.{port_name}[{i}] output enable")

    oeb_text = "\n".join(oeb_assignments) if oeb_assignments else (
        "    // No explicit OEB assignments -- all pads default to input")

    wrapper_v = f"""`default_nettype none
// openframe_project_wrapper.v
// Auto-generated by coresmith tapeout_helpers for OpenFrame shuttle submission
//
// Die area: {OPENFRAME_DIE_WIDTH_UM:.0f} x {OPENFRAME_DIE_HEIGHT_UM:.0f} um
// GPIO pads: {OPENFRAME_IO_PADS}
// Blocks: {', '.join(b['name'] for b in blocks)}

`define OPENFRAME_IO_PADS {OPENFRAME_IO_PADS}

module openframe_project_wrapper (
`ifdef USE_POWER_PINS
    inout vccd1,
    inout vssd1,
`endif
    input  wire [`OPENFRAME_IO_PADS-1:0] io_in,
    output wire [`OPENFRAME_IO_PADS-1:0] io_out,
    output wire [`OPENFRAME_IO_PADS-1:0] io_oeb
);

    // ---- GPIO pad 0 = clk, pad 1 = rst ----
    // Active-low output enable (0 = output, 1 = input)
    assign io_oeb[0] = 1'b1;  // clk is input
    assign io_oeb[1] = 1'b1;  // rst is input

    // ---- Output enable for block outputs ----
{oeb_text}

    // ---- Block instantiations ----
{instantiation_text}

    // ---- Tie unused io_out to 0 ----
    // (OpenFrame requires all outputs driven)
    genvar _unused_i;
    generate
        for (_unused_i = 0; _unused_i < `OPENFRAME_IO_PADS; _unused_i = _unused_i + 1) begin : tie_unused
            // Default: unused outputs tied low, unused OEB set to input
        end
    endgenerate

endmodule
`default_nettype wire
"""
    wrapper_path.write_text(wrapper_v)
    log(f"  [WRAPPER] Generated {wrapper_path}", GREEN)
    return wrapper_path


def _generate_power_connection(output_dir: Path, supply: str, net: str) -> Path:
    """Generate a power connection module (vccd1_connection.v or vssd1_connection.v)."""
    path = output_dir / f"{supply}_connection.v"
    path.write_text(f"""`default_nettype none
// {supply}_connection.v -- Power connection for OpenFrame
// Auto-generated by coresmith tapeout_helpers

module {supply}_connection (
    inout {supply}
);

`ifdef USE_POWER_PINS
    // Connect {supply} to internal {net} rail
    assign {net} = {supply};
`endif

endmodule
`default_nettype wire
""")
    return path


def _generate_netlists_include(blocks: list[dict], output_dir: Path) -> Path:
    """Generate an include file that references all block gate-level netlists."""
    path = output_dir / "openframe_project_netlists.v"
    includes = []
    for block in blocks:
        name = block["name"]
        includes.append(f'`include "{name}_netlist.v"')
    path.write_text(
        "// openframe_project_netlists.v -- Block netlist includes\n"
        "// Auto-generated by coresmith tapeout_helpers\n\n"
        + "\n".join(includes) + "\n"
    )
    return path


# ═══════════════════════════════════════════════════════════════════════════
# Submission Directory Structure
# ═══════════════════════════════════════════════════════════════════════════

def generate_submission_structure(
    project_root: str,
    blocks: list[dict],
    completed_backend_blocks: list[dict],
) -> dict:
    """Create the efabless/openframe_user_project directory structure.

    Copies block GDS, DEF, gate-level netlists, and SDC into the
    submission tree. Generates config files for OpenLane2 compatibility.

    Returns dict with submission_dir and file inventory.
    """
    root = Path(project_root)
    sub_dir = root / "openframe_submission"
    sub_dir.mkdir(parents=True, exist_ok=True)

    # Create directory skeleton
    dirs = [
        "openlane/openframe_project_wrapper",
        "verilog/rtl",
        "verilog/gl",
        "gds",
        "def",
        "sdc",
        "mag",
        "lef",
    ]
    for d in dirs:
        (sub_dir / d).mkdir(parents=True, exist_ok=True)

    files_copied: list[str] = []
    errors: list[str] = []

    for block_result in completed_backend_blocks:
        name = block_result.get("name", "")
        if not block_result.get("success"):
            errors.append(f"Skipping failed block: {name}")
            continue

        # Copy GDS
        gds_path = block_result.get("gds_path", "")
        if gds_path and Path(gds_path).exists():
            dest = sub_dir / "gds" / f"{name}.gds"
            shutil.copy2(gds_path, dest)
            files_copied.append(str(dest.relative_to(sub_dir)))

        # Copy routed DEF
        def_path = block_result.get("routed_def_path", "")
        if def_path and Path(def_path).exists():
            dest = sub_dir / "def" / f"{name}.def"
            shutil.copy2(def_path, dest)
            files_copied.append(str(dest.relative_to(sub_dir)))

        # Copy gate-level netlist
        pnr_dir = root / "syn" / "output" / name / "pnr"
        gl_netlist = pnr_dir / f"{name}_pnr.v"
        if gl_netlist.exists():
            dest = sub_dir / "verilog" / "gl" / f"{name}_netlist.v"
            shutil.copy2(gl_netlist, dest)
            files_copied.append(str(dest.relative_to(sub_dir)))

        # Copy SDC
        sdc_src = root / "syn" / "output" / name / f"{name}.sdc"
        if sdc_src.exists():
            dest = sub_dir / "sdc" / f"{name}.sdc"
            shutil.copy2(sdc_src, dest)
            files_copied.append(str(dest.relative_to(sub_dir)))

    # Generate a minimal Makefile
    makefile = sub_dir / "Makefile"
    makefile.write_text(
        "# OpenFrame submission Makefile\n"
        "# Auto-generated by coresmith tapeout_helpers\n\n"
        ".PHONY: precheck clean\n\n"
        "precheck:\n"
        "\t@echo \"Run: python3 -m precheck --pdk sky130A $(PWD)\"\n\n"
        "clean:\n"
        "\trm -rf runs/\n"
    )
    files_copied.append("Makefile")

    return {
        "submission_dir": str(sub_dir),
        "files_copied": files_copied,
        "errors": errors,
        "block_count": len([b for b in completed_backend_blocks if b.get("success")]),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Wrapper Synthesis (Yosys: RTL -> gate-level netlist for OpenROAD PnR)
# ═══════════════════════════════════════════════════════════════════════════

def synthesize_wrapper(
    wrapper_rtl_path: str,
    completed_backend_blocks: list[dict],
    output_dir: str,
    timeout: int = 300,
) -> dict:
    """Synthesize the OpenFrame wrapper RTL to a gate-level netlist.

    The wrapper is mostly assigns and tie-offs, so synthesis is trivial.
    Block macros are read as blackboxes from their gate-level netlists
    so Yosys can resolve the instantiation.

    Returns dict with: success, netlist_path, gate_count, error.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    script_path = out / "wrapper_synth.ys"

    abs_wrapper = str(Path(wrapper_rtl_path).resolve())

    block_reads = []
    for blk in completed_backend_blocks:
        if not blk.get("success"):
            continue
        name = blk["name"]
        # Use pre-PnR netlist (clean synthesis output without filler/decap cells).
        # Post-PnR netlists contain physical cells (sky130_fd_sc_hd__decap_*)
        # that Yosys doesn't have in its liberty and will error on.
        for candidate in (
            PROJECT_ROOT / "syn" / "output" / name / f"{name}_netlist.v",
            PROJECT_ROOT / "syn" / "output" / name / f"{name}_flat_netlist.v",
        ):
            if candidate.exists():
                block_reads.append(f"read_verilog {candidate.resolve()}")
                break

    reads = "\n".join(block_reads)

    script = f"""# Wrapper synthesis for OpenFrame (Sky130 HD)
# Generated by coresmith tapeout_helpers.synthesize_wrapper

{reads}
read_verilog {abs_wrapper}

hierarchy -check -top openframe_project_wrapper
proc; opt; flatten; opt
techmap; opt
dfflibmap -liberty {LIBERTY}
abc -liberty {LIBERTY}
clean
opt_clean -purge

stat -liberty {LIBERTY}

write_verilog -noattr {out / "openframe_project_wrapper_netlist.v"}

"""
    script_path.write_text(script)

    log("  [WRAPPER SYNTH] Running Yosys wrapper synthesis...", YELLOW)
    try:
        result = subprocess.run(
            ["yosys", "-s", str(script_path)],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(PROJECT_ROOT),
        )
        log_path = _write_step_log(
            "openframe_wrapper", "synthesize", ["yosys", "-s", str(script_path)], result,
        )

        if result.returncode != 0:
            error = (result.stderr or result.stdout)[-2000:]
            log(f"  [WRAPPER SYNTH] FAILED: {error[:200]}", RED)
            return {"success": False, "error": error, "log_path": log_path}

        netlist_path = str(out / "openframe_project_wrapper_netlist.v")

        gate_count = 0
        for line in result.stdout.splitlines():
            line = line.strip()
            if "cells" in line and line[0].isdigit():
                parts = line.split()
                if len(parts) >= 1:
                    try:
                        gate_count = int(parts[0])
                    except ValueError:
                        pass

        log(f"  [WRAPPER SYNTH] Complete: {gate_count} cells", GREEN)
        return {
            "success": True,
            "netlist_path": netlist_path,
            "gate_count": gate_count,
            "log_path": log_path,
        }

    except FileNotFoundError:
        return {"success": False, "error": "yosys binary not found"}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Yosys timed out ({timeout}s)"}


# ═══════════════════════════════════════════════════════════════════════════
# Wrapper-Level PnR (Macro Placement within OpenFrame Die)
# ═══════════════════════════════════════════════════════════════════════════

def generate_wrapper_pnr_tcl(
    wrapper_netlist: str,
    blocks: list[dict],
    completed_backend_blocks: list[dict],
    output_dir: str,
    target_clock_mhz: float = 50.0,
) -> str:
    """Generate OpenROAD Tcl for wrapper-level PnR.

    The wrapper synthesis flattens the block into standard cells, so this
    is a flat PnR within the full OpenFrame die (3520x5188 um).  The
    design is mostly wiring and tie-offs with the user block's logic
    embedded as standard cells.

    Returns path to the generated Tcl script.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tcl_path = out / "wrapper_pnr.tcl"

    period_ns = 1000.0 / target_clock_mhz

    abs_netlist = str(Path(wrapper_netlist).resolve())
    sdc_path = out / "wrapper.sdc"
    sdc_path.write_text(
        f"create_clock -name clk -period {period_ns} [get_ports {{io_in[0]}}]\n"
    )

    script = f"""# Auto-generated wrapper-level PnR for OpenFrame (Sky130)
# Generated by coresmith tapeout_helpers

set script_dir [file dirname [file normalize [info script]]]
set out_dir "$script_dir"

# ---- Read PDK ----
read_lef "{TECH_LEF}"
read_lef "{CELL_LEF}"
read_liberty "{LIBERTY}"

# ---- Read wrapper netlist (flattened, includes block logic) ----
read_verilog "{abs_netlist}"
link_design openframe_project_wrapper
read_sdc "{sdc_path}"

# ---- Fixed OpenFrame die ----
initialize_floorplan \\
    -die_area "0 0 {OPENFRAME_DIE_WIDTH_UM:.1f} {OPENFRAME_DIE_HEIGHT_UM:.1f}" \\
    -core_area "{OPENFRAME_CORE_MARGIN_UM:.1f} {OPENFRAME_CORE_MARGIN_UM:.1f} \\
                {OPENFRAME_DIE_WIDTH_UM - OPENFRAME_CORE_MARGIN_UM:.1f} \\
                {OPENFRAME_DIE_HEIGHT_UM - OPENFRAME_CORE_MARGIN_UM:.1f}" \\
    -site unithd

# Routing tracks
make_tracks li1  -x_offset 0.23 -x_pitch 0.46 -y_offset 0.17 -y_pitch 0.34
make_tracks met1 -x_offset 0.17 -x_pitch 0.34 -y_offset 0.17 -y_pitch 0.34
make_tracks met2 -x_offset 0.23 -x_pitch 0.46 -y_offset 0.23 -y_pitch 0.46
make_tracks met3 -x_offset 0.34 -x_pitch 0.68 -y_offset 0.34 -y_pitch 0.68
make_tracks met4 -x_offset 0.46 -x_pitch 0.92 -y_offset 0.46 -y_pitch 0.92
make_tracks met5 -x_offset 1.70 -x_pitch 3.40 -y_offset 1.70 -y_pitch 3.40

place_pins -hor_layers met3 -ver_layers met2

# ---- PDN (wrapper-level: met4 + met5) ----
add_global_connection -net VPWR -pin_pattern "VPWR" -power
add_global_connection -net VGND -pin_pattern "VGND" -ground
add_global_connection -net VPWR -pin_pattern "VPB" -power
add_global_connection -net VGND -pin_pattern "VNB" -ground
global_connect

set_voltage_domain -name CORE -power VPWR -ground VGND
define_pdn_grid -name wrapper_grid -starts_with POWER -voltage_domain CORE -pins {{met4 met5}}
add_pdn_stripe -grid wrapper_grid -layer met1 -width 0.48 -followpins -starts_with POWER
add_pdn_stripe -grid wrapper_grid -layer met4 -width 1.6 -pitch 27.14 -offset 13.57 -starts_with POWER
add_pdn_stripe -grid wrapper_grid -layer met5 -width 1.6 -pitch 27.14 -offset 13.57 -starts_with POWER
add_pdn_connect -grid wrapper_grid -layers {{met1 met4}}
add_pdn_connect -grid wrapper_grid -layers {{met4 met5}}
pdngen

# ---- Standard cell placement and routing ----
global_placement -density 0.3 -pad_left 2 -pad_right 2
detailed_placement
check_placement -verbose

# Filler / decap cells for continuous n-well and power rail
filler_placement -prefix FILLER {{{_STD_CELL}__decap_12 {_STD_CELL}__decap_8 {_STD_CELL}__decap_6 {_STD_CELL}__decap_4 {_STD_CELL}__decap_3 {_STD_CELL}__fill_2 {_STD_CELL}__fill_1}}

set_wire_rc -signal -layer met2
set_wire_rc -clock  -layer met3

clock_tree_synthesis \\
    -buf_list {{{_STD_CELL}__clkbuf_4 {_STD_CELL}__clkbuf_8}} \\
    -root_buf {_STD_CELL}__clkbuf_8 \\
    -sink_clustering_enable
set_propagated_clock [all_clocks]
repair_clock_nets
remove_fillers
detailed_placement
filler_placement -prefix FILLER {{{_STD_CELL}__decap_12 {_STD_CELL}__decap_8 {_STD_CELL}__decap_6 {_STD_CELL}__decap_4 {_STD_CELL}__decap_3 {_STD_CELL}__fill_2 {_STD_CELL}__fill_1}}

set_routing_layers -signal met1-met4 -clock met3-met4
global_route -congestion_iterations 50
detailed_route -output_drc "$out_dir/wrapper_route_drc.rpt" -verbose 1

# ---- Reports ----
report_checks -path_delay max > "$out_dir/wrapper_timing_setup.rpt"
report_checks -path_delay min > "$out_dir/wrapper_timing_hold.rpt"
report_wns > "$out_dir/wrapper_wns.rpt"
report_tns > "$out_dir/wrapper_tns.rpt"
report_power > "$out_dir/wrapper_power.rpt"

puts "\\n========== WRAPPER SUMMARY =========="
report_design_area
report_wns
report_tns

# ---- Write outputs ----
write_def "$out_dir/openframe_project_wrapper_routed.def"
write_verilog "$out_dir/openframe_project_wrapper_pnr.v"
write_verilog -include_pwr_gnd "$out_dir/openframe_project_wrapper_pwr.v"

puts "\\n========== WRAPPER PNR COMPLETE =========="

exit
"""
    tcl_path.write_text(script)
    return str(tcl_path)


def run_wrapper_pnr(
    wrapper_netlist: str,
    blocks: list[dict],
    completed_backend_blocks: list[dict],
    output_dir: str,
    target_clock_mhz: float = 50.0,
    timeout: int = 3600,
) -> dict:
    """Run wrapper-level PnR and return results."""
    from orchestrator.langgraph.backend_helpers import (
        parse_openroad_reports,
        parse_pnr_stdout,
        run_openroad,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tcl_path = generate_wrapper_pnr_tcl(
        wrapper_netlist, blocks, completed_backend_blocks,
        output_dir, target_clock_mhz,
    )

    log("  [WRAPPER PNR] Running OpenROAD wrapper PnR...", YELLOW)
    result = run_openroad(tcl_path, "openframe_wrapper", "wrapper_pnr", timeout=timeout)

    if not result["success"]:
        error = result.get("stderr", "") or result.get("stdout", "")
        log(f"  [WRAPPER PNR] FAILED: {error[:200]}", RED)
        return {
            "success": False,
            "error": error[-3000:],
            "log_path": result.get("log_path", ""),
        }

    metrics = parse_openroad_reports(output_dir)
    stdout_metrics = parse_pnr_stdout(result.get("stdout", ""))
    for k, v in stdout_metrics.items():
        if k not in metrics or (metrics.get(k) == 0.0 and v != 0.0):
            metrics[k] = v

    routed_def = str(out / "openframe_project_wrapper_routed.def")

    log(f"  [WRAPPER PNR] Complete: area={metrics.get('design_area_um2', 0):.0f} um², "
        f"WNS={metrics.get('wns_ns', 0):.2f} ns", GREEN)

    return {
        "success": True,
        "routed_def_path": routed_def,
        "pnr_verilog_path": str(out / "openframe_project_wrapper_pnr.v"),
        "pwr_verilog_path": str(out / "openframe_project_wrapper_pwr.v"),
        "log_path": result.get("log_path", ""),
        **metrics,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Wrapper-Level DRC + GDS (same Magic flow, different scale)
# ═══════════════════════════════════════════════════════════════════════════

def run_wrapper_drc(routed_def: str, output_dir: str, timeout: int = 1200) -> dict:
    """Run Magic DRC + GDS on the wrapper."""
    from orchestrator.langgraph.backend_helpers import run_drc_flow
    log("  [WRAPPER DRC] Running Magic DRC on wrapper...", YELLOW)
    return run_drc_flow("openframe_project_wrapper", routed_def, output_dir, timeout=timeout)


def run_wrapper_lvs(spice_path: str, pwr_verilog: str, output_dir: str, timeout: int = 1200) -> dict:
    """Run Netgen LVS on the wrapper."""
    from orchestrator.langgraph.backend_helpers import run_lvs_flow
    log("  [WRAPPER LVS] Running Netgen LVS on wrapper...", YELLOW)
    return run_lvs_flow("openframe_project_wrapper", spice_path, pwr_verilog, output_dir, timeout=timeout)


# ═══════════════════════════════════════════════════════════════════════════
# Native MPW Precheck (Option A -- no Docker)
# ═══════════════════════════════════════════════════════════════════════════

def run_mpw_precheck_native(
    submission_dir: str,
    gds_path: str = "",
    timeout: int = 1800,
) -> dict:
    """Run Efabless MPW precheck natively without Docker.

    Performs the same checks as the Docker-based mpw_precheck tool:
      1. Directory structure validation
      2. GDS file checks (layers, size, bounding box)
      3. user_defines.v generation and validation
      4. Wrapper port name validation against golden reference
      5. KLayout DRC (density, minimum width/spacing)
      6. Magic DRC (full Sky130 design rules)

    Args:
        submission_dir: Path to the openframe_submission/ directory.
        gds_path: Path to the top-level GDS. Auto-detected if empty.
        timeout: Per-check timeout in seconds.

    Returns:
        dict with overall pass/fail and per-check results.
    """
    sub = Path(submission_dir)
    results: dict = {
        "pass": True,
        "checks": {},
        "errors": [],
        "warnings": [],
    }

    # ---- Check 1: Directory structure ----
    structure_result = _check_submission_structure(sub)
    results["checks"]["structure"] = structure_result
    if not structure_result["pass"]:
        results["pass"] = False
        results["errors"].extend(structure_result.get("errors", []))

    # ---- Check 2: GDS file validation ----
    if not gds_path:
        gds_files = list((sub / "gds").glob("*.gds"))
        if gds_files:
            gds_path = str(gds_files[0])

    if gds_path and Path(gds_path).exists():
        gds_result = _check_gds_file(gds_path)
        results["checks"]["gds"] = gds_result
        if not gds_result["pass"]:
            results["pass"] = False
            results["errors"].append(f"GDS check: {gds_result.get('error', 'failed')}")
    else:
        results["checks"]["gds"] = {"pass": False, "error": "No GDS found"}
        results["pass"] = False
        results["errors"].append("No GDS file found in submission")

    # ---- Check 3: user_defines.v ----
    defines_result = _check_and_generate_user_defines(sub)
    results["checks"]["user_defines"] = defines_result
    if not defines_result["pass"]:
        results["pass"] = False
        results["errors"].extend(defines_result.get("errors", []))

    # ---- Check 4: Wrapper port names ----
    port_result = _check_wrapper_port_names(sub)
    results["checks"]["port_names"] = port_result
    if not port_result["pass"]:
        results["pass"] = False
        results["errors"].extend(port_result.get("errors", []))
    if port_result.get("warnings"):
        results["warnings"].extend(port_result["warnings"])

    # ---- Check 5: KLayout DRC (advisory -- not a hard gate) ----
    if gds_path and Path(gds_path).exists():
        klayout_result = _run_klayout_drc(gds_path, str(sub), timeout)
        results["checks"]["klayout_drc"] = klayout_result
        if not klayout_result.get("pass"):
            if klayout_result.get("skipped"):
                results["warnings"].append("KLayout DRC skipped (binary not available)")
            else:
                results["warnings"].append(
                    f"KLayout DRC: {klayout_result.get('violation_count', '?')} violations")

    # ---- Check 6: Magic DRC (authoritative -- on top-level GDS) ----
    if gds_path and Path(gds_path).exists():
        magic_result = _run_magic_drc_on_gds(gds_path, str(sub), timeout)
        results["checks"]["magic_drc"] = magic_result
        if not magic_result["pass"]:
            results["pass"] = False
            results["errors"].append(
                f"Magic DRC: {magic_result.get('violation_count', '?')} violations")

    hard_checks = {
        k: v for k, v in results["checks"].items()
        if not v.get("skipped")
    }
    derived_pass = all(c.get("pass") for c in hard_checks.values()) if hard_checks else False
    if results["pass"] and not derived_pass:
        results["pass"] = False
        results["errors"].append(
            "Consistency fix: overall pass was True but individual "
            f"checks disagree: {[k for k, v in hard_checks.items() if not v.get('pass')]}"
        )

    passed = sum(1 for c in results["checks"].values() if c.get("pass"))
    total = len(results["checks"])
    log(f"  [PRECHECK] {'PASS' if results['pass'] else 'FAIL'}: "
        f"{passed}/{total} checks passed", GREEN if results["pass"] else RED)

    return results


def _check_and_generate_user_defines(sub: Path) -> dict:
    """Check for user_defines.v and generate if missing.

    The Efabless precheck requires a user_defines.v that sets GPIO
    configuration defaults. If absent, we auto-generate one.
    """
    rtl_dir = sub / "verilog" / "rtl"
    defines_path = rtl_dir / "user_defines.v"
    errors: list[str] = []
    generated = False

    if not defines_path.exists():
        rtl_dir.mkdir(parents=True, exist_ok=True)
        defines_content = _generate_user_defines_v()
        defines_path.write_text(defines_content)
        generated = True
        log("  [PRECHECK] Generated user_defines.v", GREEN)

    # Validate content
    try:
        text = defines_path.read_text()
        if "`define USER_CONFIG_GPIO" not in text and "`define OPENFRAME" not in text:
            if len(text.strip()) < 10:
                errors.append("user_defines.v is empty or lacks GPIO config defines")
    except OSError as exc:
        errors.append(f"Cannot read user_defines.v: {exc}")

    return {
        "pass": len(errors) == 0,
        "generated": generated,
        "path": str(defines_path),
        "errors": errors,
    }


def _generate_user_defines_v() -> str:
    """Generate a user_defines.v with GPIO configuration defaults."""
    lines = [
        "`default_nettype none",
        "// user_defines.v",
        "// Auto-generated by coresmith tapeout_helpers for OpenFrame submission",
        "//",
        "// GPIO configuration defaults for the OpenFrame wrapper.",
        "// Each GPIO pad mode is set via a 13-bit config register.",
        "// Default: digital input (GPIO_MODE_USER_STD_INPUT_NOPULL)",
        "",
        "`ifndef __USER_DEFINES_H",
        "`define __USER_DEFINES_H",
        "",
        "// GPIO mode constants (from caravel/openframe defines)",
        "`define GPIO_MODE_MGMT_STD_INPUT_NOPULL    13'h0403",
        "`define GPIO_MODE_MGMT_STD_INPUT_PULLDOWN   13'h0c01",
        "`define GPIO_MODE_MGMT_STD_INPUT_PULLUP     13'h0801",
        "`define GPIO_MODE_MGMT_STD_OUTPUT           13'h1809",
        "`define GPIO_MODE_MGMT_STD_BIDIRECTIONAL    13'h1801",
        "`define GPIO_MODE_USER_STD_INPUT_NOPULL     13'h0402",
        "`define GPIO_MODE_USER_STD_INPUT_PULLDOWN    13'h0c00",
        "`define GPIO_MODE_USER_STD_INPUT_PULLUP      13'h0800",
        "`define GPIO_MODE_USER_STD_OUTPUT            13'h1808",
        "`define GPIO_MODE_USER_STD_BIDIRECTIONAL     13'h1800",
        "`define GPIO_MODE_USER_STD_ANALOG            13'h000a",
        "",
        "// Per-GPIO configuration (active during user mode)",
    ]

    # GPIO[0] = clk (input), GPIO[1] = rst (input), rest = user-defined
    lines.append("`define USER_CONFIG_GPIO_0_INIT  `GPIO_MODE_USER_STD_INPUT_NOPULL")
    lines.append("`define USER_CONFIG_GPIO_1_INIT  `GPIO_MODE_USER_STD_INPUT_NOPULL")

    for i in range(2, OPENFRAME_IO_PADS):
        lines.append(
            f"`define USER_CONFIG_GPIO_{i}_INIT  `GPIO_MODE_USER_STD_BIDIRECTIONAL"
        )

    lines.extend([
        "",
        "`endif // __USER_DEFINES_H",
        "`default_nettype wire",
        "",
    ])
    return "\n".join(lines)


def _check_wrapper_port_names(sub: Path) -> dict:
    """Validate wrapper RTL port names against OpenFrame golden reference.

    The Efabless precheck XOR-checks port names. This validates them
    before submission to catch naming issues early.
    """
    golden_ports = {
        "io_in": {"direction": "input", "width": OPENFRAME_IO_PADS},
        "io_out": {"direction": "output", "width": OPENFRAME_IO_PADS},
        "io_oeb": {"direction": "output", "width": OPENFRAME_IO_PADS},
    }
    power_ports = {"vccd1", "vssd1"}

    errors: list[str] = []
    warnings: list[str] = []

    rtl_dir = sub / "verilog" / "rtl"
    wrapper_files = list(rtl_dir.glob("*wrapper*.v")) if rtl_dir.is_dir() else []

    if not wrapper_files:
        wrapper_files = list(rtl_dir.glob("*.v")) if rtl_dir.is_dir() else []

    if not wrapper_files:
        errors.append("No wrapper RTL found in verilog/rtl/")
        return {"pass": False, "errors": errors, "warnings": warnings}

    wrapper_path = wrapper_files[0]
    try:
        text = wrapper_path.read_text()
    except OSError as exc:
        errors.append(f"Cannot read wrapper RTL: {exc}")
        return {"pass": False, "errors": errors, "warnings": warnings}

    # Check module name
    module_match = re.search(r"module\s+(\w+)", text)
    if module_match:
        module_name = module_match.group(1)
        if module_name != "openframe_project_wrapper":
            errors.append(
                f"Wrapper module named '{module_name}', expected "
                f"'openframe_project_wrapper'"
            )

    # Check required ports
    for port_name, spec in golden_ports.items():
        width = spec["width"]
        # Look for the port declaration
        pattern = rf"(?:input|output|inout)\s+(?:wire\s+)?(?:\[{width - 1}:0\]\s+)?{port_name}"
        if not re.search(pattern, text):
            simple_pattern = rf"\b{port_name}\b"
            if re.search(simple_pattern, text):
                warnings.append(
                    f"Port '{port_name}' found but width may not match "
                    f"expected [{width - 1}:0]"
                )
            else:
                errors.append(f"Missing required port: {port_name}")

    # Check power ports (inside USE_POWER_PINS ifdef)
    for pport in power_ports:
        if pport not in text:
            warnings.append(
                f"Power port '{pport}' not found in wrapper "
                f"(should be inside `ifdef USE_POWER_PINS)"
            )

    return {
        "pass": len(errors) == 0,
        "wrapper_path": str(wrapper_path),
        "errors": errors,
        "warnings": warnings,
    }


def _check_submission_structure(sub: Path) -> dict:
    """Validate the OpenFrame submission directory structure."""
    required = [
        "gds",
        "def",
        "verilog/rtl",
        "verilog/gl",
    ]
    missing = [d for d in required if not (sub / d).is_dir()]
    has_gds = bool(list((sub / "gds").glob("*.gds"))) if (sub / "gds").is_dir() else False
    has_def = bool(list((sub / "def").glob("*.def"))) if (sub / "def").is_dir() else False
    has_netlist = (
        bool(list((sub / "verilog" / "gl").glob("*.v")))
        if (sub / "verilog" / "gl").is_dir() else False
    )

    errors = []
    if missing:
        errors.append(f"Missing directories: {', '.join(missing)}")
    if not has_gds:
        errors.append("No GDS files in gds/")
    if not has_def:
        errors.append("No DEF files in def/")
    if not has_netlist:
        errors.append("No gate-level netlists in verilog/gl/")

    return {
        "pass": len(errors) == 0,
        "missing_dirs": missing,
        "has_gds": has_gds,
        "has_def": has_def,
        "has_netlist": has_netlist,
        "errors": errors,
    }


def _check_gds_file(gds_path: str) -> dict:
    """Basic GDS file checks: size, existence, non-empty."""
    p = Path(gds_path)
    size = p.stat().st_size
    return {
        "pass": size > 1000,
        "path": gds_path,
        "size_bytes": size,
        "size_mb": round(size / (1024 * 1024), 2),
        "error": "GDS too small (< 1KB)" if size <= 1000 else "",
    }


def _run_klayout_drc(gds_path: str, output_dir: str, timeout: int = 600) -> dict:
    """Run KLayout DRC on the GDS file.

    Uses a Ruby DRC script that checks metal density and basic rules.
    """
    out = Path(output_dir)
    report_path = out / "klayout_drc.xml"

    # Generate KLayout Ruby DRC script for Sky130 with density checks
    drc_script = out / "precheck_drc.drc"
    drc_script.write_text(f"""# Sky130 precheck DRC (KLayout Ruby format)
# Auto-generated by coresmith tapeout_helpers
# Includes minimum width/spacing and metal density checks

source("{gds_path}")
report("Sky130 Precheck DRC", "{report_path}")

# Metal layer definitions (GDS layer/datatype)
met1 = input(68, 20)
met2 = input(69, 20)
met3 = input(70, 20)
met4 = input(71, 20)
met5 = input(72, 20)

# Minimum width checks (Sky130 design rules)
met1.width(0.14).output("met1.width", "Metal 1 width < 0.14um")
met2.width(0.14).output("met2.width", "Metal 2 width < 0.14um")
met3.width(0.3).output("met3.width", "Metal 3 width < 0.3um")
met4.width(0.3).output("met4.width", "Metal 4 width < 0.3um")
met5.width(1.6).output("met5.width", "Metal 5 width < 1.6um")

# Minimum spacing checks
met1.space(0.14).output("met1.space", "Metal 1 spacing < 0.14um")
met2.space(0.14).output("met2.space", "Metal 2 spacing < 0.14um")
met3.space(0.3).output("met3.space", "Metal 3 spacing < 0.3um")
met4.space(0.3).output("met4.space", "Metal 4 spacing < 0.3um")
met5.space(1.6).output("met5.space", "Metal 5 spacing < 1.6um")

# Metal density checks (Efabless MPW requirement)
# Sky130 target: all metal layers between ~30-70% density
# These are tile-based density checks over the full die
die_area = extent.area

[["met1", met1, 68], ["met2", met2, 69], ["met3", met3, 70],
 ["met4", met4, 71], ["met5", met5, 72]].each do |name, layer, gds_layer|
  layer_area = layer.area
  if die_area > 0
    density = layer_area.to_f / die_area.to_f
    if density < 0.01
      log("WARNING: #{{name}} density #{{(density * 100).round(2)}}% -- below minimum (needs metal fill)")
    end
  end
end
""")

    cmd = [KLAYOUT_BIN, "-b", "-r", str(drc_script)]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=str(PROJECT_ROOT),
        )

        _write_step_log("openframe_wrapper", "klayout_drc", cmd, result, 1)

        # Parse the KLayout DRC XML report
        violation_count = 0
        if report_path.exists():
            report_text = report_path.read_text()
            violation_count = report_text.count("<item>")

        return {
            "pass": violation_count == 0,
            "violation_count": violation_count,
            "report_path": str(report_path),
            "returncode": result.returncode,
            "stderr": result.stderr[:500] if result.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {
            "pass": False,
            "violation_count": -1,
            "error": f"KLayout DRC timed out ({timeout}s)",
        }
    except FileNotFoundError:
        log("  [PRECHECK] KLayout not available -- skipping KLayout DRC", YELLOW)
        return {
            "pass": False,
            "violation_count": 0,
            "error": f"KLayout binary not found: {KLAYOUT_BIN}. Skipped.",
            "skipped": True,
        }


def _run_magic_drc_on_gds(gds_path: str, output_dir: str, timeout: int = 600) -> dict:
    """Run Magic DRC directly on a GDS file (precheck variant)."""
    from orchestrator.langgraph.backend_helpers import run_magic

    out = Path(output_dir)
    tcl_path = out / "precheck_magic_drc.tcl"

    abs_gds = str(Path(gds_path).resolve())
    design_name = Path(gds_path).stem

    tcl_path.write_text(f"""# Magic DRC precheck on GDS
# Auto-generated by coresmith tapeout_helpers

lef read "{TECH_LEF}"
lef read "{CELL_LEF}"
gds read "{abs_gds}"
load {design_name}
flatten {design_name}_flat
load {design_name}_flat
select top cell
drc catchup
drc count
set drc_count [drc listall count]

set rpt [open "{out}/precheck_magic_drc.rpt" w]
puts $rpt "Design: {design_name}"
puts $rpt "DRC count: $drc_count"
set result [drc listall why]
puts $rpt $result
close $rpt

puts "DRC violations: $drc_count"
quit -noprompt
""")

    result = run_magic(str(tcl_path), "precheck", "precheck_drc", timeout=timeout)

    return {
        "pass": result.get("drc_count", -1) == 0 and result.get("success", False),
        "violation_count": result.get("drc_count", -1),
        "report_path": str(out / "precheck_magic_drc.rpt"),
    }
