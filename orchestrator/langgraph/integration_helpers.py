# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Chip-level integration helpers for the ASIC pipeline.

Provides:
- parse_verilog_ports(): extract port declarations from Verilog RTL files
- check_integration_compatibility(): verify all block-to-block connections
- generate_top_level_rtl(): create the chip top-level module wiring all blocks
- lint_top_level(): run Verilator lint on the integrated design
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from orchestrator.langgraph.pipeline_helpers import (
    PROJECT_ROOT,
    _write_step_log,
    _write_step_log_error,
    run_wavekit_vcd_audit,
)


# ---------------------------------------------------------------------------
# Port parsing
# ---------------------------------------------------------------------------

@dataclass
class VerilogPort:
    """A single port extracted from a Verilog module."""
    name: str
    direction: str          # "input", "output", "inout"
    width: int = 1          # bit width (e.g. [7:0] -> 8)
    msb: int = 0            # upper bound of range
    lsb: int = 0            # lower bound of range
    is_reg: bool = False
    is_signed: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VerilogModule:
    """Parsed module header from a Verilog file."""
    name: str
    ports: list[VerilogPort] = field(default_factory=list)
    parameters: dict[str, str] = field(default_factory=dict)
    filepath: str = ""

    def port_by_name(self, name: str) -> Optional[VerilogPort]:
        for p in self.ports:
            if p.name == name:
                return p
        return None

    def inputs(self) -> list[VerilogPort]:
        return [p for p in self.ports if p.direction == "input"]

    def outputs(self) -> list[VerilogPort]:
        return [p for p in self.ports if p.direction == "output"]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "ports": [p.to_dict() for p in self.ports],
            "parameters": self.parameters,
            "filepath": self.filepath,
        }


def parse_verilog_ports(rtl_path: str) -> VerilogModule:
    """Parse a Verilog file and extract the module name, ports, and parameters.

    Handles both ANSI-style (ports in header) and non-ANSI (separate
    declarations) Verilog modules. Extracts the *first* module found.

    Returns:
        VerilogModule with parsed port list.
    """
    path = Path(rtl_path)
    if not path.exists():
        return VerilogModule(name="", filepath=rtl_path)

    source = path.read_text(encoding="utf-8", errors="replace")

    # Strip comments (line and block)
    source = re.sub(r'//.*?$', '', source, flags=re.MULTILINE)
    source = re.sub(r'/\*.*?\*/', '', source, flags=re.DOTALL)

    # Find module declaration
    mod_match = re.search(
        r'module\s+(\w+)\s*(?:#\s*\(([^)]*)\))?\s*\(([^;]*)\)\s*;',
        source, re.DOTALL
    )
    if not mod_match:
        # Try module without port list (empty or parameterised)
        mod_match = re.search(r'module\s+(\w+)\s*;', source)
        if mod_match:
            return VerilogModule(name=mod_match.group(1), filepath=rtl_path)
        return VerilogModule(name="", filepath=rtl_path)

    module_name = mod_match.group(1)
    param_text = mod_match.group(2) or ""
    port_text = mod_match.group(3) or ""

    # Parse parameters
    parameters: dict[str, str] = {}
    if param_text.strip():
        for pm in re.finditer(
            r'parameter\s+(?:\w+\s+)?(\w+)\s*=\s*([^,\)]+)', param_text
        ):
            parameters[pm.group(1).strip()] = pm.group(2).strip()

    ports: list[VerilogPort] = []

    # Try ANSI-style ports (direction in header)
    ansi_port_re = re.compile(
        r'(input|output|inout)\s+'
        r'(?:(reg|wire)\s+)?'
        r'(?:(signed)\s+)?'
        r'(?:\[\s*(\d+)\s*:\s*(\d+)\s*\]\s*)?'
        r'(\w+)',
        re.MULTILINE
    )

    ansi_ports = list(ansi_port_re.finditer(port_text))

    if ansi_ports:
        for m in ansi_ports:
            direction = m.group(1)
            is_reg = m.group(2) == "reg"
            is_signed = m.group(3) == "signed"
            msb = int(m.group(4)) if m.group(4) else 0
            lsb = int(m.group(5)) if m.group(5) else 0
            name = m.group(6)
            width = abs(msb - lsb) + 1 if m.group(4) else 1

            ports.append(VerilogPort(
                name=name,
                direction=direction,
                width=width,
                msb=msb,
                lsb=lsb,
                is_reg=is_reg,
                is_signed=is_signed,
            ))
    else:
        # Non-ANSI: port names in header, declarations in body
        port_names = [n.strip() for n in port_text.split(',') if n.strip()]
        # Get the body after the module header
        body_start = mod_match.end()
        endmodule_match = re.search(r'\bendmodule\b', source[body_start:])
        body = source[body_start:body_start + endmodule_match.start()] if endmodule_match else source[body_start:]

        for pname in port_names:
            # Clean up any remaining brackets/whitespace
            pname = re.sub(r'\s+', '', pname)
            if not pname:
                continue

            # Find direction declaration in body
            decl_re = re.compile(
                rf'(input|output|inout)\s+'
                rf'(?:(reg|wire)\s+)?'
                rf'(?:(signed)\s+)?'
                rf'(?:\[\s*(\d+)\s*:\s*(\d+)\s*\]\s*)?'
                rf'\b{re.escape(pname)}\b'
            )
            dm = decl_re.search(body)
            if dm:
                direction = dm.group(1)
                is_reg = dm.group(2) == "reg"
                is_signed = dm.group(3) == "signed"
                msb = int(dm.group(4)) if dm.group(4) else 0
                lsb = int(dm.group(5)) if dm.group(5) else 0
                width = abs(msb - lsb) + 1 if dm.group(4) else 1
                ports.append(VerilogPort(
                    name=pname, direction=direction, width=width,
                    msb=msb, lsb=lsb, is_reg=is_reg, is_signed=is_signed,
                ))
            else:
                ports.append(VerilogPort(name=pname, direction="input", width=1))

    return VerilogModule(
        name=module_name, ports=ports, parameters=parameters, filepath=rtl_path
    )


# ---------------------------------------------------------------------------
# Compatibility checking
# ---------------------------------------------------------------------------

@dataclass
class IntegrationMismatch:
    """A single integration compatibility issue."""
    from_block: str
    to_block: str
    issue_type: str      # "width_mismatch", "missing_port", "direction_error", "naming_mismatch"
    severity: str        # "error", "warning"
    description: str
    suggested_fix: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _find_port_fuzzy(
    module: VerilogModule,
    port_name: str,
    connection_name: str,
) -> Optional[VerilogPort]:
    """Find a port by exact name, then by common naming conventions.

    Tries: exact match, snake_case variants, with/without block prefix.
    """
    # Exact match
    p = module.port_by_name(port_name)
    if p:
        return p

    # Try connection name directly
    p = module.port_by_name(connection_name)
    if p:
        return p

    # Try common suffixes/prefixes
    variants = [
        port_name,
        f"{port_name}_o", f"{port_name}_i",
        f"o_{port_name}", f"i_{port_name}",
        port_name.replace("_data", ""), port_name.replace("_out", ""),
        port_name.replace("_in", ""),
    ]

    for v in variants:
        p = module.port_by_name(v)
        if p:
            return p

    # Substring match (last resort) -- check if any port contains the key term
    key_terms = [t for t in port_name.split('_') if len(t) > 2]
    for term in key_terms:
        for port in module.ports:
            if term in port.name:
                return port

    return None


def check_integration_compatibility(
    connections: list[dict],
    modules: dict[str, VerilogModule],
) -> list[IntegrationMismatch]:
    """Check all block-to-block connections for compatibility.

    Args:
        connections: List of connection dicts from architecture block diagram.
            Each has: from_block, from_port, to_block, to_port, data_width,
            interface (connection name/label).
        modules: Dict mapping block_name -> VerilogModule (parsed RTL).

    Returns:
        List of IntegrationMismatch objects for all issues found.
    """
    mismatches: list[IntegrationMismatch] = []

    for conn in connections:
        from_block = conn.get("from_block", conn.get("from", ""))
        to_block = conn.get("to_block", conn.get("to", ""))
        from_port = conn.get("from_port", "")
        to_port = conn.get("to_port", "")
        interface_name = conn.get("interface", conn.get("name", ""))
        expected_width = conn.get("data_width", 0)

        # Parse data_width from string if needed (e.g. "8b" -> 8)
        if isinstance(expected_width, str):
            w_match = re.match(r'(\d+)', str(expected_width))
            expected_width = int(w_match.group(1)) if w_match else 0

        # Check source block exists
        src_module = modules.get(from_block)
        if not src_module:
            mismatches.append(IntegrationMismatch(
                from_block=from_block, to_block=to_block,
                issue_type="missing_block", severity="error",
                description=f"Source block '{from_block}' RTL not found",
                suggested_fix=f"Ensure {from_block} RTL was generated and passed synthesis",
            ))
            continue

        # Check destination block exists
        dst_module = modules.get(to_block)
        if not dst_module:
            mismatches.append(IntegrationMismatch(
                from_block=from_block, to_block=to_block,
                issue_type="missing_block", severity="error",
                description=f"Destination block '{to_block}' RTL not found",
                suggested_fix=f"Ensure {to_block} RTL was generated and passed synthesis",
            ))
            continue

        # Find source output port
        src_port = _find_port_fuzzy(src_module, from_port, interface_name)
        if not src_port:
            mismatches.append(IntegrationMismatch(
                from_block=from_block, to_block=to_block,
                issue_type="missing_port", severity="error",
                description=(
                    f"Source port '{from_port}' not found on {from_block} "
                    f"(available outputs: {[p.name for p in src_module.outputs()]})"
                ),
                suggested_fix=(
                    f"Add output port '{from_port}' to {from_block} RTL, "
                    f"or update the architecture connection to use an existing port"
                ),
                details={
                    "connection": interface_name,
                    "available_ports": [p.name for p in src_module.ports],
                },
            ))
            continue

        # Check source port direction (should be output)
        if src_port.direction == "input":
            mismatches.append(IntegrationMismatch(
                from_block=from_block, to_block=to_block,
                issue_type="direction_error", severity="error",
                description=(
                    f"Port '{src_port.name}' on {from_block} is an input, "
                    f"but connection expects it to be an output"
                ),
                suggested_fix=f"Change '{src_port.name}' direction to output in {from_block}",
            ))

        # Find destination input port
        dst_port = _find_port_fuzzy(dst_module, to_port, interface_name)
        if not dst_port:
            mismatches.append(IntegrationMismatch(
                from_block=from_block, to_block=to_block,
                issue_type="missing_port", severity="error",
                description=(
                    f"Destination port '{to_port}' not found on {to_block} "
                    f"(available inputs: {[p.name for p in dst_module.inputs()]})"
                ),
                suggested_fix=(
                    f"Add input port '{to_port}' to {to_block} RTL, "
                    f"or update the architecture connection to use an existing port"
                ),
                details={
                    "connection": interface_name,
                    "available_ports": [p.name for p in dst_module.ports],
                },
            ))
            continue

        # Check destination port direction (should be input)
        if dst_port.direction == "output":
            mismatches.append(IntegrationMismatch(
                from_block=from_block, to_block=to_block,
                issue_type="direction_error", severity="error",
                description=(
                    f"Port '{dst_port.name}' on {to_block} is an output, "
                    f"but connection expects it to be an input"
                ),
                suggested_fix=f"Change '{dst_port.name}' direction to input in {to_block}",
            ))

        # Width compatibility check
        if src_port.width != dst_port.width:
            mismatches.append(IntegrationMismatch(
                from_block=from_block, to_block=to_block,
                issue_type="width_mismatch", severity="error",
                description=(
                    f"Width mismatch: {from_block}.{src_port.name} is "
                    f"{src_port.width}-bit but {to_block}.{dst_port.name} "
                    f"is {dst_port.width}-bit"
                ),
                suggested_fix=(
                    "Widen the narrower port or add explicit "
                    "truncation/extension in the top-level wiring"
                ),
                details={
                    "src_width": src_port.width,
                    "dst_width": dst_port.width,
                    "expected_width": expected_width,
                },
            ))
        elif expected_width > 0 and src_port.width != expected_width:
            mismatches.append(IntegrationMismatch(
                from_block=from_block, to_block=to_block,
                issue_type="width_mismatch", severity="warning",
                description=(
                    f"Architecture specifies {expected_width}-bit for "
                    f"'{interface_name}', but both ports are {src_port.width}-bit"
                ),
                suggested_fix=(
                    f"Update architecture connection width to {src_port.width} "
                    f"or adjust both block ports to {expected_width} bits"
                ),
                details={
                    "actual_width": src_port.width,
                    "expected_width": expected_width,
                },
            ))

    return mismatches


# ---------------------------------------------------------------------------
# Shared signal detection (clk, rst, etc.)
# ---------------------------------------------------------------------------

_SHARED_SIGNAL_PATTERNS = {
    "clk": re.compile(r'^(clk|clock|i_clk|clk_i)$', re.IGNORECASE),
    "rst": re.compile(r'^(rst|reset|rstn|rst_n|i_rst|rst_i|i_rstn|rstn_i|arst_n)$', re.IGNORECASE),
}


def _is_shared_signal(port_name: str) -> Optional[str]:
    """Check if a port name is a shared infrastructure signal (clk, rst).

    Returns the canonical signal name ('clk', 'rst') or None.
    """
    for canonical, pattern in _SHARED_SIGNAL_PATTERNS.items():
        if pattern.match(port_name):
            return canonical
    return None


def _detect_reset_convention(modules: dict[str, VerilogModule]) -> dict:
    """Detect reset naming convention across all blocks.

    Returns dict with 'name', 'active_low' fields representing the
    most common reset convention.
    """
    reset_names: dict[str, int] = {}
    for mod in modules.values():
        for p in mod.ports:
            sig = _is_shared_signal(p.name)
            if sig == "rst":
                reset_names[p.name] = reset_names.get(p.name, 0) + 1

    if not reset_names:
        return {"name": "rst_n", "active_low": True}

    most_common = max(reset_names, key=reset_names.get)
    active_low = 'n' in most_common.lower()
    return {"name": most_common, "active_low": active_low}


def _detect_clock_name(modules: dict[str, VerilogModule]) -> str:
    """Detect the most common clock port name across all blocks."""
    clk_names: dict[str, int] = {}
    for mod in modules.values():
        for p in mod.ports:
            sig = _is_shared_signal(p.name)
            if sig == "clk":
                clk_names[p.name] = clk_names.get(p.name, 0) + 1

    if not clk_names:
        return "clk"
    return max(clk_names, key=clk_names.get)


# ---------------------------------------------------------------------------
# Top-level RTL generation
# ---------------------------------------------------------------------------

def generate_top_level_rtl(
    design_name: str,
    connections: list[dict],
    modules: dict[str, VerilogModule],
    mismatches: list[IntegrationMismatch] | None = None,
) -> dict:
    """Generate the top-level Verilog module that instantiates and wires all blocks.

    Only includes blocks that have parsed RTL (are in the ``modules`` dict).
    Shared signals (clk, rst) are connected to top-level ports.
    Block-to-block connections use internal wires.

    Args:
        design_name: Name for the top-level module (e.g. "h264_encoder_top").
        connections: Architecture connection list.
        modules: Parsed block modules.
        mismatches: Known mismatches (used to skip broken connections).

    Returns:
        dict with keys: verilog, rtl_path, module_name, block_count,
        wire_count, skipped_connections.
    """
    safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', design_name).lower()
    if not safe_name or safe_name[0].isdigit():
        safe_name = f"top_{safe_name}"

    clk_name = _detect_clock_name(modules)
    rst_info = _detect_reset_convention(modules)
    rst_name = rst_info["name"]

    # Collect all error-level mismatched connections to skip
    error_connections: set[tuple[str, str]] = set()
    if mismatches:
        for m in mismatches:
            if m.severity == "error":
                error_connections.add((m.from_block, m.to_block))

    # Build wire declarations and connection map
    wires: list[str] = []
    wire_connections: dict[str, list[tuple[str, str]]] = {}  # wire_name -> [(block, port)]
    skipped: list[str] = []

    for i, conn in enumerate(connections):
        from_block = conn.get("from_block", conn.get("from", ""))
        to_block = conn.get("to_block", conn.get("to", ""))
        from_port = conn.get("from_port", "")
        to_port = conn.get("to_port", "")
        interface_name = conn.get("interface", conn.get("name", f"conn_{i}"))

        # Skip connections with error-level mismatches
        if (from_block, to_block) in error_connections:
            skipped.append(f"{from_block}->{to_block} ({interface_name}): has errors")
            continue

        src_mod = modules.get(from_block)
        dst_mod = modules.get(to_block)
        if not src_mod or not dst_mod:
            skipped.append(f"{from_block}->{to_block}: missing block RTL")
            continue

        # Find actual ports
        src_port = _find_port_fuzzy(src_mod, from_port, interface_name)
        dst_port = _find_port_fuzzy(dst_mod, to_port, interface_name)
        if not src_port or not dst_port:
            skipped.append(f"{from_block}->{to_block} ({interface_name}): port not found")
            continue

        # Determine wire width (use source port width)
        width = src_port.width
        wire_name = f"w_{from_block}_{src_port.name}_to_{to_block}_{dst_port.name}"
        wire_name = re.sub(r'[^a-zA-Z0-9_]', '_', wire_name)

        if width > 1:
            wires.append(f"  wire [{width-1}:0] {wire_name};")
        else:
            wires.append(f"  wire {wire_name};")

        # Track connections for each block's port
        wire_connections.setdefault(f"{from_block}.{src_port.name}", []).append(
            ("wire", wire_name)
        )
        wire_connections.setdefault(f"{to_block}.{dst_port.name}", []).append(
            ("wire", wire_name)
        )

    # Collect top-level I/O ports (ports not connected to other blocks)
    top_inputs: list[str] = []
    top_outputs: list[str] = []
    top_port_lines: list[str] = []

    # Always include clk and rst
    top_inputs.append(f"  input  wire {clk_name}")
    top_inputs.append(f"  input  wire {rst_name}")

    # Find unconnected ports across all blocks
    for block_name, mod in sorted(modules.items()):
        for port in mod.ports:
            if _is_shared_signal(port.name):
                continue  # handled globally

            key = f"{block_name}.{port.name}"
            if key in wire_connections:
                continue  # connected to another block

            # This port is unconnected -- expose it at top level
            top_port_name = f"{block_name}_{port.name}"
            width_decl = f"[{port.msb}:{port.lsb}] " if port.width > 1 else ""

            if port.direction == "input":
                top_inputs.append(f"  input  wire {width_decl}{top_port_name}")
                wire_connections[key] = [("top", top_port_name)]
            elif port.direction == "output":
                top_outputs.append(f"  output wire {width_decl}{top_port_name}")
                wire_connections[key] = [("top", top_port_name)]
            else:  # inout
                top_inputs.append(f"  inout  wire {width_decl}{top_port_name}")
                wire_connections[key] = [("top", top_port_name)]

    top_port_lines = top_inputs + top_outputs

    # Build module header
    lines: list[str] = []
    lines.append("// Auto-generated top-level integration module")
    lines.append(f"// Design: {design_name}")
    lines.append(f"// Blocks: {len(modules)}")
    lines.append("// Generated by coresmith integration pipeline")
    lines.append("")
    lines.append(f"module {safe_name} (")
    lines.append(",\n".join(top_port_lines))
    lines.append(");")
    lines.append("")

    # Wire declarations
    if wires:
        lines.append(f"  // Internal wires ({len(wires)} connections)")
        lines.extend(wires)
        lines.append("")

    # Block instantiations
    for block_name, mod in sorted(modules.items()):
        lines.append(f"  // {block_name}")
        lines.append(f"  {mod.name} u_{block_name} (")

        port_connections: list[str] = []
        for port in mod.ports:
            key = f"{block_name}.{port.name}"
            sig = _is_shared_signal(port.name)

            if sig == "clk":
                port_connections.append(f"    .{port.name}({clk_name})")
            elif sig == "rst":
                # Handle reset polarity mismatch
                if port.name == rst_name:
                    port_connections.append(f"    .{port.name}({rst_name})")
                else:
                    # Different naming -- might need inversion
                    port_is_active_low = 'n' in port.name.lower()
                    top_is_active_low = rst_info["active_low"]
                    if port_is_active_low == top_is_active_low:
                        port_connections.append(f"    .{port.name}({rst_name})")
                    else:
                        port_connections.append(f"    .{port.name}(~{rst_name})")
            elif key in wire_connections:
                conns = wire_connections[key]
                # Use the first wire/top connection
                _, wire_name = conns[0]
                port_connections.append(f"    .{port.name}({wire_name})")
            else:
                # Unconnected -- tie off
                if port.direction == "input":
                    tie_val = f"{port.width}'b0" if port.width > 1 else "1'b0"
                    port_connections.append(f"    .{port.name}({tie_val})  // UNCONNECTED")
                else:
                    port_connections.append(f"    .{port.name}()  // UNCONNECTED")

        lines.append(",\n".join(port_connections))
        lines.append("  );")
        lines.append("")

    lines.append("endmodule")
    lines.append("")

    verilog = "\n".join(lines)

    # Write to disk
    rtl_dir = PROJECT_ROOT / "rtl" / "integration"
    rtl_dir.mkdir(parents=True, exist_ok=True)
    rtl_path = rtl_dir / f"{safe_name}.v"
    rtl_path.write_text(verilog, encoding="utf-8")

    return {
        "verilog": verilog,
        "rtl_path": str(rtl_path),
        "module_name": safe_name,
        "block_count": len(modules),
        "wire_count": len(wires),
        "skipped_connections": skipped,
    }


# ---------------------------------------------------------------------------
# Integration lint
# ---------------------------------------------------------------------------

def lint_top_level(
    top_rtl_path: str,
    block_rtl_paths: list[str],
    design_name: str = "integration",
) -> dict:
    """Run Verilator lint on the top-level module with all block RTL files.

    Includes all block Verilog files so Verilator can resolve instantiations.

    Returns:
        dict with: clean (bool), errors (str), warnings (str), log_path (str).
    """
    cmd = [
        "verilator", "--lint-only", "-Wall", "-Wno-fatal",
        "-Wno-EOFNEWLINE",
        "--top-module", Path(top_rtl_path).stem,
        top_rtl_path,
    ]
    # Add all block RTL files
    for bp in block_rtl_paths:
        if Path(bp).exists() and bp != top_rtl_path:
            cmd.append(bp)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        log_path = _write_step_log(design_name, "integration_lint", cmd, result)
        stderr = result.stderr.strip()
        has_errors = "%Error" in stderr
        if result.returncode == 0 and not has_errors:
            return {"clean": True, "warnings": stderr, "log_path": log_path}
        else:
            return {"clean": False, "errors": stderr[-3000:], "log_path": log_path}
    except subprocess.TimeoutExpired:
        log_path = _write_step_log_error(
            design_name, "integration_lint", cmd, "Verilator lint timed out"
        )
        return {"clean": False, "errors": "Verilator lint timed out", "log_path": log_path}
    except FileNotFoundError:
        log_path = _write_step_log_error(
            design_name, "integration_lint", cmd, "Verilator not installed"
        )
        return {"clean": False, "errors": "Verilator not installed", "log_path": log_path}


# ---------------------------------------------------------------------------
# Load architecture connections
# ---------------------------------------------------------------------------

def load_architecture_connections(project_root: str) -> tuple[list[dict], str]:
    """Load block-to-block connections from architecture state.

    Tries architecture_state.json first, then block_diagram_viz.json.

    Returns:
        (connections_list, design_name)
    """
    root = Path(project_root)
    design_name = "chip_top"

    # Try architecture_state.json (primary source)
    arch_path = root / ".coresmith" / "architecture_state.json"
    if arch_path.exists():
        try:
            data = json.loads(arch_path.read_text(encoding="utf-8"))
            bd = data.get("block_diagram", {})
            connections = bd.get("connections", [])
            # Extract design name: prefer actual module name from
            # integration RTL on disk, fall back to block_diagram title,
            # and only use PRD title as last resort.
            _int_dir = root / "rtl" / "integration"
            _found_module = ""
            if _int_dir.is_dir():
                for _vf in sorted(_int_dir.glob("*.v")):
                    try:
                        _src = _vf.read_text(encoding="utf-8", errors="replace")
                        _mm = re.search(r'^\s*module\s+(\w+)', _src, re.MULTILINE)
                        if _mm:
                            _found_module = _mm.group(1)
                            break
                    except OSError:
                        pass
            if _found_module:
                design_name = _found_module
            else:
                # Fall back to a clean name from PRD title
                prd = data.get("prd_spec", data.get("ers_spec", {}))
                prd_doc = prd.get("prd", prd.get("ers", {})) if isinstance(prd, dict) else {}
                if prd_doc.get("title"):
                    _raw = prd_doc["title"]
                    # Strip common prefixes like "PRD — " or "ERS — "
                    _raw = re.sub(r'^(?:PRD|ERS)\s*[—–-]\s*', '', _raw)
                    design_name = re.sub(r'[^a-zA-Z0-9_]', '_', _raw).strip('_').lower()
                    design_name = re.sub(r'_+', '_', design_name)
                    design_name = f"{design_name}_top"
            if connections:
                return connections, design_name
        except (json.JSONDecodeError, OSError):
            pass

    # Fallback: block_diagram_viz.json (ReactFlow format)
    viz_path = root / ".coresmith" / "block_diagram_viz.json"
    if viz_path.exists():
        try:
            data = json.loads(viz_path.read_text(encoding="utf-8"))
            # ReactFlow edges -> connections
            edges = data.get("edges", [])
            connections = []
            for edge in edges:
                conn = {
                    "from_block": edge.get("source", ""),
                    "to_block": edge.get("target", ""),
                    "interface": edge.get("data", {}).get("label", ""),
                    "from_port": edge.get("sourceHandle", ""),
                    "to_port": edge.get("targetHandle", ""),
                    "data_width": edge.get("data", {}).get("data_width", 0),
                }
                connections.append(conn)

            # Design name from viz metadata
            nodes = data.get("nodes", [])
            if nodes:
                design_name = "chip_top"

            return connections, design_name
        except (json.JSONDecodeError, OSError):
            pass

    return [], design_name


# ---------------------------------------------------------------------------
# Integration testbench generation + simulation
# ---------------------------------------------------------------------------

async def generate_integration_testbench(
    design_name: str,
    top_rtl_path: str,
    modules: dict[str, "VerilogModule"],
    connections: list[dict],
    block_rtl_paths: dict[str, str],
    prd_summary: str = "",
    prior_failure: str = "",
) -> dict:
    """Generate a cocotb integration testbench via the Lead DV agent.

    ``prior_failure`` is non-empty when this is a retry. It carries a brief
    description of why the previous integration DV attempt failed so the
    LLM can avoid repeating the same mistake. The underlying
    ``IntegrationTestbenchGenerator.generate`` accepts the same kwarg.

    Returns:
        dict with: tb_path (str), testbench_path (str), test_count (int).
    """
    from orchestrator.langchain.agents.coresmith_llm import DEFAULT_MODEL
    from orchestrator.langchain.agents.integration_testbench_generator import (
        IntegrationTestbenchGenerator,
    )

    top_rtl_source = Path(top_rtl_path).read_text(encoding="utf-8")

    block_summaries = []
    for name, mod in sorted(modules.items()):
        block_summaries.append({
            "name": name,
            "port_count": len(mod.ports),
            "ports": [p.to_dict() for p in mod.ports],
        })

    tb_dir = PROJECT_ROOT / "tb" / "integration"
    tb_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(tb_dir / f"test_{design_name}.py")

    agent = IntegrationTestbenchGenerator(model=DEFAULT_MODEL, temperature=0.1)
    result = await agent.generate(
        design_name=design_name,
        top_rtl_source=top_rtl_source,
        block_summaries=block_summaries,
        connections=connections,
        prd_summary=prd_summary,
        block_rtl_paths=block_rtl_paths,
        output_path=output_path,
        prior_failure=prior_failure,
    )

    result["testbench_path"] = result.get("tb_path", output_path)
    return result


async def generate_validation_testbench(
    design_name: str,
    top_rtl_path: str,
    modules: dict[str, "VerilogModule"],
    connections: list[dict],
    block_rtl_paths: dict[str, str],
    ers_context: str,
    prior_failure: str = "",
) -> dict:
    """Generate an ERS/KPI validation cocotb testbench via Lead Validation DV.

    Returns:
        dict with: tb_path (str), testbench_path (str), test_count (int).
    """
    from orchestrator.langchain.agents.coresmith_llm import DEFAULT_MODEL
    from orchestrator.langchain.agents.validation_dv_generator import (
        ValidationDVGenerator,
    )

    top_rtl_source = Path(top_rtl_path).read_text(encoding="utf-8")

    block_summaries = []
    for name, mod in sorted(modules.items()):
        block_summaries.append({
            "name": name,
            "port_count": len(mod.ports),
            "ports": [p.to_dict() for p in mod.ports],
        })

    tb_dir = PROJECT_ROOT / "tb" / "validation"
    tb_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(tb_dir / f"test_{design_name}_validation.py")

    agent = ValidationDVGenerator(model=DEFAULT_MODEL, temperature=0.1)
    result = await agent.generate(
        design_name=design_name,
        top_rtl_path=top_rtl_path,
        top_rtl_source=top_rtl_source,
        block_summaries=block_summaries,
        connections=connections,
        ers_context=ers_context,
        block_rtl_paths=block_rtl_paths,
        output_path=output_path,
        prior_failure=prior_failure,
    )

    result["testbench_path"] = result.get("tb_path", output_path)
    return result


def run_integration_simulation(
    design_name: str,
    top_rtl_path: str,
    block_rtl_paths: dict[str, str],
    tb_path: str,
    attempt: int = 1,
) -> dict:
    """Run cocotb simulation on the integrated top-level design.

    Sets up a Makefile with all block Verilog sources + top-level,
    runs cocotb via Verilator, and returns pass/fail + logs.

    Returns:
        dict with: passed (bool), log (str), returncode (int), log_path (str).
    """
    import os
    import shutil
    from orchestrator.langgraph.pipeline_helpers import (
        _normalize_cocotb_timing_keywords,
        _parse_cocotb_summary,
    )

    sim_dir = PROJECT_ROOT / "sim_build" / "integration"
    sim_dir.mkdir(parents=True, exist_ok=True)

    all_sources = [top_rtl_path]
    for bp in block_rtl_paths.values():
        if Path(bp).exists() and bp != top_rtl_path:
            all_sources.append(bp)
    sources_str = " ".join(all_sources)

    safe_name = Path(top_rtl_path).stem

    makefile_content = f"""
SIM = verilator
TOPLEVEL_LANG = verilog
VERILOG_SOURCES = {sources_str}
TOPLEVEL = {safe_name}
MODULE = {Path(tb_path).stem}
WAVES = 1
EXTRA_ARGS += --trace --trace-structs
include $(shell cocotb-config --makefiles)/Makefile.sim
"""
    (sim_dir / "Makefile").write_text(makefile_content)

    # Copy the TB under its ORIGINAL stem so the cocotb MODULE name (which is
    # Path(tb_path).stem) resolves on PYTHONPATH=sim_dir.  Validation DV TBs
    # are named test_<design>_validation.py, so the previous hardcoded
    # f"test_{design_name}.py" copy renamed them to test_<design>.py and
    # cocotb failed import at 0 ns with `No module named test_<design>_validation`.
    sim_tb_path = sim_dir / f"{Path(tb_path).stem}.py"
    shutil.copy2(tb_path, sim_tb_path)
    _normalize_cocotb_timing_keywords(sim_tb_path)

    env = os.environ.copy()
    import sys
    venv_bin = str(Path(sys.prefix) / "bin")
    env["PATH"] = f"{venv_bin}:{env.get('PATH', '/usr/bin:/bin')}"
    env["SHELL"] = shutil.which("bash") or "/bin/bash"
    env["PYTHONPATH"] = f"{sim_dir}:{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"

    make_bin = shutil.which("make") or "make"

    try:
        result = subprocess.run(
            [make_bin, "-C", str(sim_dir)],
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
        )
        log_path = _write_step_log(
            "integration", "integration_sim", [make_bin, "-C", str(sim_dir)],
            result, attempt,
        )
        full_output = result.stdout + "\n" + result.stderr
        output = full_output[-5000:]
        no_tests = "No tests were discovered" in output
        summary = _parse_cocotb_summary(full_output)
        if no_tests:
            output = (
                "COCOTB ERROR: No tests were discovered. Treating simulation "
                "as failed to prevent DV false pass.\n" + output
            )
        if summary["found"] and summary["tests_failed"]:
            output = (
                "COCOTB ERROR: Regression summary reports failing tests. "
                "Treating simulation as failed even if make returned 0.\n" + output
            )

        vcd_path = sim_dir / "dump.vcd"
        audit_path = sim_dir / "wavekit_audit.json"
        wavekit_audit = run_wavekit_vcd_audit(vcd_path, audit_path)
        passed = (
            result.returncode == 0
            and not no_tests
            and (
                not summary["found"]
                or (summary["tests_total"] > 0 and summary["tests_failed"] == 0)
            )
            and wavekit_audit.get("ok") is True
        )
        if not wavekit_audit.get("ok"):
            output = (
                "WAVEKIT VCD AUDIT FAILED: "
                f"{wavekit_audit.get('error', 'unknown error')}\n" + output
            )
        return {
            "passed": passed,
            "log": output,
            "returncode": result.returncode,
            "tests_passed": summary["tests_passed"],
            "tests_total": summary["tests_total"],
            "tests_failed": summary["tests_failed"],
            "log_path": log_path,
            "vcd_path": str(vcd_path) if vcd_path.exists() else "",
            "wavekit_audit_path": str(audit_path),
            "wavekit_audit": wavekit_audit,
        }
    except subprocess.TimeoutExpired:
        cmd = [make_bin, "-C", str(sim_dir)]
        log_path = _write_step_log_error(
            "integration", "integration_sim", cmd,
            "Integration simulation timed out (10 min)", attempt,
        )
        return {"passed": False, "log": "Integration simulation timed out (10 min)", "log_path": log_path}
    except FileNotFoundError as e:
        cmd = [make_bin, "-C", str(sim_dir)]
        log_path = _write_step_log_error(
            "integration", "integration_sim", cmd, f"Tool not found: {e}", attempt,
        )
        return {"passed": False, "log": f"Tool not found: {e}", "log_path": log_path}


def discover_block_rtl(
    project_root: str,
    completed_blocks: list[dict],
) -> dict[str, str]:
    """Discover RTL file paths for all completed blocks.

    Uses completed_blocks state to find RTL paths, falling back to
    convention-based discovery.  Also searches subdirectories of rtl/.

    Returns:
        Dict mapping block_name -> rtl_file_path.
    """
    root = Path(project_root)
    rtl_paths: dict[str, str] = {}

    for block in completed_blocks:
        name = block.get("name", block.get("block_name", ""))
        if not name:
            continue

        # Skip failed/aborted blocks
        if block.get("aborted") or block.get("skipped"):
            continue

        # Try rtl_path from block result
        rtl_path = block.get("rtl_path", "")
        if rtl_path and Path(rtl_path).exists():
            rtl_paths[name] = rtl_path
            continue

        # Convention-based discovery
        candidates = [
            root / "rtl" / name / f"{name}.v",
            root / "rtl" / f"{name}.v",
            root / f"{name}.v",
        ]
        for c in candidates:
            if c.exists():
                rtl_paths[name] = str(c)
                break
        else:
            # Search subdirectories of rtl/
            rtl_dir = root / "rtl"
            if rtl_dir.is_dir():
                for sub in rtl_dir.iterdir():
                    if sub.is_dir():
                        candidate = sub / f"{name}.v"
                        if candidate.exists():
                            rtl_paths[name] = str(candidate)
                            break

    return rtl_paths


def detect_glue_block_needs(
    connections: list[dict],
    modules: dict[str, "VerilogModule"],
) -> list[dict]:
    """Detect where glue/adapter blocks are needed between connected modules.

    Scans connections for width mismatches or protocol incompatibilities
    that require a bridge module.

    Returns a list of dicts, each describing a glue block need:
      {"from_block", "to_block", "type", "from_width", "to_width", "name"}
    """
    needs: list[dict] = []

    for conn in connections:
        from_block = conn.get("from_block", conn.get("from", ""))
        to_block = conn.get("to_block", conn.get("to", ""))
        from_port = conn.get("from_port", "")
        to_port = conn.get("to_port", "")
        interface_name = conn.get("interface", conn.get("name", ""))

        src_mod = modules.get(from_block)
        dst_mod = modules.get(to_block)
        if not src_mod or not dst_mod:
            continue

        src_port = _find_port_fuzzy(src_mod, from_port, interface_name)
        dst_port = _find_port_fuzzy(dst_mod, to_port, interface_name)
        if not src_port or not dst_port:
            continue

        if src_port.width != dst_port.width:
            if src_port.width > dst_port.width and src_port.width % dst_port.width == 0:
                glue_type = "parallel_to_serial"
            elif dst_port.width > src_port.width and dst_port.width % src_port.width == 0:
                glue_type = "serial_to_parallel"
            else:
                glue_type = "width_adapter"

            glue_name = f"{glue_type}_{from_block}_{to_block}"
            needs.append({
                "from_block": from_block,
                "to_block": to_block,
                "type": glue_type,
                "from_width": src_port.width,
                "to_width": dst_port.width,
                "name": glue_name,
                "interface": interface_name,
            })

    return needs
