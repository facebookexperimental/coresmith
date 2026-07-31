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
from dataclasses import asdict, dataclass, field
from pathlib import Path

from orchestrator.langgraph.pipeline_helpers import (
    PROJECT_ROOT,
    RED,
    _write_step_log,
    _write_step_log_error,
    apply_build_fingerprint,
    clear_build_products,
    log,
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

    def port_by_name(self, name: str) -> VerilogPort | None:
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


def parse_verilog_ports(rtl_path: str, module: str | None = None) -> VerilogModule:
    """Parse a Verilog file and extract the module name, ports, and parameters.

    Handles both ANSI-style (ports in header) and non-ANSI (separate
    declarations) Verilog modules.

    ``module`` selects WHICH module to parse; without it the first one wins,
    which is frequently the wrong answer. A generated block file commonly
    declares its internal stages first -- raster_scan_pipeline.v declares four
    sub-stages before the block itself at line 668 -- so integration judged a
    sub-stage's ports as the block's interface and raised 22 wiring hazards for
    a block that conforms perfectly. Callers that know the block name should
    pass it.

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

    # Narrow to the requested module, else the file stem -- the same precedence
    # rtl_module_name uses. Sliced to its endmodule so a later module's
    # non-ANSI port declarations cannot leak in.
    for _want in [w for w in (module, path.stem) if w]:
        _m = re.search(r'\bmodule\s+' + re.escape(str(_want)) + r'\b', source)
        if _m:
            _end = source.find('endmodule', _m.start())
            source = source[_m.start():_end if _end != -1 else len(source)]
            break

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
    prefer_direction: str | None = None,
) -> VerilogPort | None:
    """Find a port by exact name, then by common naming conventions.

    Tries: exact match, snake_case variants, with/without block prefix.

    ``prefer_direction`` ("input"/"output") disambiguates the fuzzy fallbacks
    on blocks with multiple same-prefix interfaces. dv-hardening-28: a block
    with BOTH an ``s_axis_*`` (input) and an ``m_axis_*`` (output) interface fed
    a direction-blind substring match that returned whichever port was declared
    first -- often the wrong direction -- so a producer's OUTPUT lookup resolved
    to an INPUT port and the checker manufactured a "port is an input but
    connection expects output" storm (false-positive on armD + aes, both with
    multi-interface blocks). Preferring the port whose direction matches the
    connection role fixes it. Exact matches (unambiguous) are unaffected.
    """
    def _pick(cands: list[VerilogPort]) -> VerilogPort | None:
        cands = [c for c in cands if c is not None]
        if not cands:
            return None
        if prefer_direction:
            for c in cands:
                if getattr(c, "direction", None) == prefer_direction:
                    return c
        return cands[0]

    # Exact match (unambiguous)
    p = module.port_by_name(port_name)
    if p:
        return p
    p = module.port_by_name(connection_name)
    if p:
        return p

    # Common suffix/prefix variants -- prefer the requested direction.
    variants = [
        f"{port_name}_o", f"{port_name}_i",
        f"o_{port_name}", f"i_{port_name}",
        port_name.replace("_data", ""), port_name.replace("_out", ""),
        port_name.replace("_in", ""),
    ]
    picked = _pick([module.port_by_name(v) for v in variants])
    if picked:
        return picked

    # Substring match (last resort) -- direction-aware among the candidates.
    key_terms = [t for t in port_name.split('_') if len(t) > 2]
    # rung3-fixes-1: a port-less connection (only an ``interface`` name) yields
    # no key terms; fall back to the connection/interface name so it can still
    # resolve. Named-port lookups never reach here with an empty key_terms.
    if not key_terms and connection_name:
        key_terms = [t for t in connection_name.split('_') if len(t) > 2]
    substr_cands: list[VerilogPort] = []
    for term in key_terms:
        for port in module.ports:
            if term in port.name and port not in substr_cands:
                substr_cands.append(port)
    return _pick(substr_cands)


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
        src_port = _find_port_fuzzy(src_module, from_port, interface_name, prefer_direction="output")
        if not src_port:
            if not str(from_port).strip():
                # rung3-fixes-1: PORT-LESS connection (the block diagram carried
                # only an ``interface`` name, no ``from_port``) whose interface
                # name also matched no port on the source module. There is no
                # signal to check -- cannot-check != error. SKIP with an
                # informational note instead of manufacturing a bogus
                # "Source port '' not found" error. An empty-string port must
                # never inflate error_count.
                mismatches.append(IntegrationMismatch(
                    from_block=from_block, to_block=to_block,
                    issue_type="unresolved_portless_connection", severity="info",
                    description=(
                        f"Connection '{interface_name or '(unnamed)'}' from "
                        f"{from_block} has no source-port attribution and its "
                        f"interface name matched no port on {from_block}; "
                        f"skipping compatibility check (cannot check != error)"
                    ),
                    suggested_fix=(
                        "Add 'from_port' to the architecture connection (or name "
                        "the interface to match an RTL port) to enable the check"
                    ),
                    details={
                        "connection": interface_name,
                        "available_ports": [p.name for p in src_module.ports],
                    },
                ))
                continue
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
        dst_port = _find_port_fuzzy(dst_module, to_port, interface_name, prefer_direction="input")
        if not dst_port:
            if not str(to_port).strip():
                # rung3-fixes-1: port-less destination (see the source branch
                # above) whose interface name matched no port -- SKIP with an
                # informational note, never a manufactured "port '' not found".
                mismatches.append(IntegrationMismatch(
                    from_block=from_block, to_block=to_block,
                    issue_type="unresolved_portless_connection", severity="info",
                    description=(
                        f"Connection '{interface_name or '(unnamed)'}' to "
                        f"{to_block} has no destination-port attribution and its "
                        f"interface name matched no port on {to_block}; "
                        f"skipping compatibility check (cannot check != error)"
                    ),
                    suggested_fix=(
                        "Add 'to_port' to the architecture connection (or name "
                        "the interface to match an RTL port) to enable the check"
                    ),
                    details={
                        "connection": interface_name,
                        "available_ports": [p.name for p in dst_module.ports],
                    },
                ))
                continue
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


def _is_shared_signal(port_name: str) -> str | None:
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
        design_name: Name for the top-level module (e.g. "video_encoder_top").
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
        src_port = _find_port_fuzzy(src_mod, from_port, interface_name, prefer_direction="output")
        dst_port = _find_port_fuzzy(dst_mod, to_port, interface_name, prefer_direction="input")
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
# Caravel / user_project_wrapper deterministic top assembly (Defect 4)
# ---------------------------------------------------------------------------
#
# THE GAP: the per-block generator emits the ``user_project_wrapper`` block as a
# bare *pad adapter* (io_in/io_out/io_oeb <-> a few packed bundle edges). But the
# MPW/Caravel harness + the grader require ``user_project_wrapper`` to BE the
# wired chip_top -- a locked-port module that INSTANTIATES and wires every block.
# So the daemon never produced a gradeable chip_top and each chip-lead had to
# hand-assemble one. This assembler closes the gap deterministically: it emits
# ``module user_project_wrapper`` with the locked Caravel port list, instantiates
# the pad-adapter block (renamed to dodge the name collision) + all core blocks,
# and wires block<->block bundles by normalized signal key. No LLM, no hand-work.

# The locked Caravel `user_project_wrapper` port list -- the harness instantiates
# this module BY NAME with these EXACT ports. (dir, width_decl, name); width_decl
# is "" for a scalar.
_CARAVEL_TOP_PORTS: list[tuple[str, str, str]] = [
    ("input",  "",         "wb_clk_i"),
    ("input",  "",         "wb_rst_i"),
    ("input",  "",         "wbs_stb_i"),
    ("input",  "",         "wbs_cyc_i"),
    ("input",  "",         "wbs_we_i"),
    ("input",  "[3:0] ",   "wbs_sel_i"),
    ("input",  "[31:0] ",  "wbs_dat_i"),
    ("input",  "[31:0] ",  "wbs_adr_i"),
    ("output", "",         "wbs_ack_o"),
    ("output", "[31:0] ",  "wbs_dat_o"),
    ("input",  "[127:0] ", "la_data_in"),
    ("output", "[127:0] ", "la_data_out"),
    ("input",  "[127:0] ", "la_oenb"),
    ("input",  "[37:0] ",  "io_in"),
    ("output", "[37:0] ",  "io_out"),
    ("output", "[37:0] ",  "io_oeb"),
    ("inout",  "[28:0] ",  "analog_io"),
    ("input",  "",         "user_clock2"),
    ("output", "[2:0] ",   "user_irq"),
]

# Non-accelerator Caravel interfaces the wrapper drives to constants when the
# design does not use them (matches the hand-assembled tops).
_CARAVEL_TIEOFFS: list[tuple[str, str]] = [
    ("wbs_ack_o",   "1'b0"),
    ("wbs_dat_o",   "32'b0"),
    ("la_data_out", "128'b0"),
    ("user_irq",    "3'b0"),
]

CARAVEL_TOP_MODULE = "user_project_wrapper"
_PAD_IO_PORTS = ("io_in", "io_out", "io_oeb")
_CARAVEL_TOP_PORT_NAMES = {name for _d, _w, name in _CARAVEL_TOP_PORTS}

# Caravel supply pins live inside `ifdef USE_POWER_PINS on both the top and the
# wrapper block. They MUST be emitted under the same guard on both sides: a
# parser sees them as ports even with the define off, so a blind `.vdda1(vdda1)`
# connection to a module compiled WITHOUT the define is a PINNOTFOUND error.
_CARAVEL_POWER_PORTS = [
    "vdda1", "vdda2", "vssa1", "vssa2", "vccd1", "vccd2", "vssd1", "vssd2",
]
_POWER_PIN_RE = re.compile(
    r"^(?:vdda\d?|vssa\d?|vccd\d?|vssd\d?|vddio\d?|vssio\d?|vdd|vss|vgnd|vpwr|"
    r"vpb|vnb)$",
    re.IGNORECASE,
)


def _is_power_pin(port_name: str) -> bool:
    return bool(_POWER_PIN_RE.match(port_name))


def detect_wrapper_block(modules: dict[str, VerilogModule]) -> str | None:
    """Identify the pad-adapter / Caravel wrapper block among the parsed modules.

    Prefers a block literally named ``user_project_wrapper``; else the block
    whose ports carry the Caravel GPIO pad vector (io_in / io_out / io_oeb); else
    a block whose *module* name is ``user_project_wrapper``. Returns the block
    key or None (non-Caravel design)."""
    if CARAVEL_TOP_MODULE in modules:
        return CARAVEL_TOP_MODULE
    for bn, mod in modules.items():
        names = {p.name for p in mod.ports}
        if all(io in names for io in _PAD_IO_PORTS):
            return bn
    for bn, mod in modules.items():
        if mod.name == CARAVEL_TOP_MODULE:
            return bn
    return None


def _contract_signal_names(edge: dict) -> list[str]:
    """Every signal a contract edge declares: payload fields + sideband.

    The schema splits them across two keys, which is why several readers
    concluded the contract did not record signal names at all. Their union is
    the channel's port set.
    """
    out: list[str] = []
    for f in (edge.get("fields") or []):
        n = f.get("name") if isinstance(f, dict) else f
        if n:
            out.append(str(n))
    for s in (edge.get("sideband_signals") or []):
        n = s.get("name") if isinstance(s, dict) else s
        if n:
            out.append(str(n))
    seen, uniq = set(), []
    for n in out:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq


def _resolve_by_contract(edge, pb, cb, port_exact, modules):
    """Wire an edge signal-by-signal from the contract's own declaration.

    Returns ``(paired, hazards)``, or ``None`` when the edge declares no
    signals (the caller then falls back to the legacy name/positional path, so
    contracts that predate signal lists are unaffected).

    Each side is resolved INDEPENDENTLY against the declared name, accepting
    either ``<channel>_<signal>`` or bare ``<signal>``. Because both ends are
    matched to the same contract signal, they are never matched to each other:
    the two sides may legitimately spell their ports differently and still bind
    correctly, which the positional path could not express.
    """
    signals = _contract_signal_names(edge)
    if not signals:
        return None
    if pb not in modules or cb not in modules:
        return None

    def _one(block: str, chan: str, sig: str):
        pref = port_exact(block, f"{chan}_{sig}")
        bare = port_exact(block, sig)
        if pref is not None and bare is not None:
            return None, (f"{block}.{chan}_{sig} and {block}.{sig} both exist "
                          f"-- ambiguous which implements '{sig}'")
        got = pref or bare
        if got is None:
            return None, (f"{block} implements no port for declared signal "
                          f"'{sig}' (expected '{chan}_{sig}' or '{sig}')")
        return got, None

    pchan = str(edge.get("producer_port") or "")
    cchan = str(edge.get("consumer_port") or "")
    eid = edge.get("edge_id")
    paired, hazards = [], []
    for sig in signals:
        pp, perr = _one(pb, pchan, sig)
        cp, cerr = _one(cb, cchan, sig)
        for err in (perr, cerr):
            if err:
                hazards.append(f"edge {eid}: {err}")
        if pp is None or cp is None:
            continue
        if _wrap_shared_signal(pp.name) or _wrap_shared_signal(cp.name):
            continue
        if pp.width != cp.width:
            hazards.append(
                f"edge {eid}: signal '{sig}' is {pp.width}b on {pb}.{pp.name} "
                f"but {cp.width}b on {cb}.{cp.name} -- refusing to short "
                f"mismatched-width ports")
            continue
        paired.append((pp, cp))
    return paired, hazards


def load_interface_contract_edges(project_root: str) -> list[dict]:
    """Load endpoint-resolved block<->block edges from
    ``.coresmith/interface_contracts.json`` (producer/consumer block + port).

    The block_diagram ``connections`` are frequently interface-keyed with NULL
    endpoints, so the contracts file is the authoritative producer/consumer map.
    Returns a list of ``{producer_block, consumer_block, producer_port,
    consumer_port, data_width, edge_id}``. The exact per-edge ``producer_port`` /
    ``consumer_port`` (kept, not discarded) let the deterministic top assembler
    resolve a specific field when a block exposes two ports that normalize to the
    same signal key (e.g. ``rd_addr`` 24b and ``in_rd_addr`` 8b) -- without them
    the assembler would short distinct nets.
    """
    path = Path(project_root) / ".coresmith" / "interface_contracts.json"
    edges: list[dict] = []
    if not path.exists():
        return edges
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return edges
    contracts = data.get("contracts", data) if isinstance(data, dict) else data
    if not isinstance(contracts, list):
        return edges
    for c in contracts:
        if not isinstance(c, dict):
            continue
        pb = c.get("producer_block") or c.get("from_block")
        cb = c.get("consumer_block") or c.get("to_block")
        if not pb or not cb or pb == cb:
            continue
        edges.append({
            "producer_block": pb,
            "consumer_block": cb,
            "producer_port": c.get("producer_port") or c.get("from_port") or "",
            "consumer_port": c.get("consumer_port") or c.get("to_port") or "",
            "data_width": c.get("data_width_bits") or c.get("data_width") or 0,
            "edge_id": c.get("edge_id") or f"{pb}__to__{cb}",
            # Carry the channel SIGNAL LIST through. The contract declares every
            # signal on the edge (fields = payload, sideband_signals =
            # everything else, union = the port set); dropping them here forced
            # the assembler to re-derive the port set from a naming convention
            # and to pair the two ends positionally against each other. With the
            # list present each end resolves against the CONTRACT instead, so
            # the two ends never have to agree on spelling.
            "fields": c.get("fields") or [],
            "sideband_signals": c.get("sideband_signals") or [],
        })
    return edges


# A leading direction prefix on a bundle port (m_axis_reg_req_tdata /
# s_axis_reg_req_tdata both normalize to reg_req_tdata) so a producer OUTPUT and
# a consumer INPUT of the same interface share one normalized signal key.
_DIR_PREFIX_RE = re.compile(
    r"^(?:m_axis_|s_axis_|axis_|m_|s_|i_|o_|in_|out_)", re.IGNORECASE
)
_DIR_SUFFIX_RE = re.compile(r"(?:_i|_o|_in|_out)$", re.IGNORECASE)


def _signal_key(port_name: str) -> str:
    """Normalize a bundle port name to a direction-agnostic signal key so a
    producer master port and its consumer slave port collide on one wire."""
    s = port_name.lower()
    s = _DIR_PREFIX_RE.sub("", s, count=1)
    s = _DIR_SUFFIX_RE.sub("", s)
    return s.strip("_")


# The global _SHARED_SIGNAL_PATTERNS omit Caravel's wb_clk_i / wb_rst_i, so the
# assembler recognizes clk/rst locally (without perturbing that shared table,
# which other integration paths depend on). Falls back to the global patterns.
_WRAP_CLK_RE = re.compile(
    r"^(?:wb_clk_i|clk|clock|clk_i|i_clk|clk_in|core_clk|sys_clk)$", re.IGNORECASE
)
_WRAP_RST_RE = re.compile(
    r"^(?:wb_rst_i|wb_rst_n_i|rst|reset|rst_n|rstn|rst_i|i_rst|rstn_i|arst_n|"
    r"resetn|sys_rst|core_rst)$",
    re.IGNORECASE,
)


def _wrap_shared_signal(port_name: str) -> str | None:
    if _WRAP_CLK_RE.match(port_name):
        return "clk"
    if _WRAP_RST_RE.match(port_name):
        return "rst"
    return _is_shared_signal(port_name)


class _WireUnion:
    """Tiny union-find over (block, port) nodes: ports across CONNECTED blocks
    that share a normalized signal key are merged onto one internal wire (handles
    fan-out and producer->consumer chains)."""

    def __init__(self) -> None:
        self._parent: dict[tuple[str, str], tuple[str, str]] = {}

    def find(self, x: tuple[str, str]) -> tuple[str, str]:
        self._parent.setdefault(x, x)
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: tuple[str, str], b: tuple[str, str]) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra

    def groups(self) -> dict[tuple[str, str], list[tuple[str, str]]]:
        out: dict[tuple[str, str], list[tuple[str, str]]] = {}
        for node in list(self._parent):
            out.setdefault(self.find(node), []).append(node)
        return out


def generate_caravel_wrapper_top(
    modules: dict[str, VerilogModule],
    edges: list[dict],
    rtl_paths: dict[str, str],
    output_dir: str,
    wrapper_block: str | None = None,
    pin_map=None,
) -> dict:
    """Assemble a wired ``user_project_wrapper`` chip_top deterministically.

    Emits ``module user_project_wrapper`` with the locked Caravel port list that
    instantiates the pad-adapter block (renamed to avoid the module-name
    collision) + every core block, wiring block<->block bundles by normalized
    signal key (union-find over the interface_contracts edges). The pad-adapter's
    io_in/io_out/io_oeb route straight to the top GPIO ports; unused Caravel
    interfaces are tied off; dangling block inputs are tied to 0 and dangling
    outputs left open -- so the result INSTANTIATES ALL BLOCKS and ELABORATES as
    ``user_project_wrapper`` with no hand-assembly.

    Returns dict: verilog, rtl_path, module_name, block_count, wire_count,
    instantiated (list), lint_block_paths (block RTL paths with the wrapper's
    original file swapped for the renamed pad copy), renamed_pad_path.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if wrapper_block is None:
        wrapper_block = detect_wrapper_block(modules)

    # A declared pin map REPLACES the pin-adapter block. The adapter existed
    # only to translate pad bits into named signals, and the top now does that
    # itself from data. Keeping it would instantiate a module that duplicates
    # the routing -- and, in the case that motivated this, the entire rest of
    # the chip, because its mandated module name means "the whole design" and
    # the generator built one.
    #
    # The contract edges that reference it describe that same translation, so
    # they are dropped with it: they are no longer block-to-block channels.
    dropped_adapter = ""
    if (pin_map is not None and getattr(pin_map, "ok", False)
            and wrapper_block is not None):
        dropped_adapter = wrapper_block
        modules = {k: v for k, v in modules.items() if k != wrapper_block}
        # Its FILE has to go too, not just its instantiation. That file declares
        # stub versions of the sibling blocks, and it sorts first, so leaving it
        # in the source list makes _dedup_module_sources treat those stubs as
        # the first definitions and strip the real implementations out of every
        # other file -- emptying the design while still looking assembled.
        rtl_paths = {k: v for k, v in rtl_paths.items() if k != wrapper_block}
        edges = [e for e in edges
                 if wrapper_block not in (e.get("producer_block"),
                                          e.get("consumer_block"))]
        wrapper_block = None

    clk_name = "wb_clk_i"
    rst_name = "wb_rst_i"          # Caravel wb_rst_i is active-high
    # Per-block reset polarity here is decided port-by-port below (a port
    # named with a trailing "n" is treated as active-low), independent of
    # the whole-design consensus computed by _detect_reset_convention() --
    # unlike generate_chip_top's single blended top-level reset above.

    # ---- connected pairs from the contract edges ----
    pairs: set[frozenset] = set()
    for e in edges:
        pb, cb = e.get("producer_block"), e.get("consumer_block")
        if pb in modules and cb in modules and pb != cb:
            pairs.add(frozenset((pb, cb)))

    # ---- union bundle ports across connected pairs ----
    # Precompute, per block, {signal_key -> [port]} for non-clk/rst, non-pad-io.
    def _bundle_ports(bn: str) -> dict[str, list[VerilogPort]]:
        out: dict[str, list[VerilogPort]] = {}
        for p in modules[bn].ports:
            if _wrap_shared_signal(p.name) or _is_power_pin(p.name):
                continue
            if bn == wrapper_block and p.name in _PAD_IO_PORTS:
                continue
            out.setdefault(_signal_key(p.name), []).append(p)
        return out

    def _port_exact(bn: str, pname: str) -> VerilogPort | None:
        if not pname:
            return None
        for p in modules[bn].ports:
            if p.name.lower() == str(pname).lower():
                return p
        return None

    def _channel_fields(bn: str, chan: str) -> list[VerilogPort]:
        """Ports implementing a CHANNEL named ``chan``, ordered by suffix.

        A contract edge names a channel; the RTL exposes it prefix-expanded
        (contract ``framebuffer_read`` -> ports ``framebuffer_read_req_addr``,
        ``framebuffer_read_read_enable``, ``framebuffer_read_rdata``,
        ``framebuffer_read_rvalid``, ``framebuffer_read_fault``). Without this,
        every such edge failed ``_port_exact``, counted as a wiring hazard, and
        took the whole deterministic Caravel assembly down with it -- on this
        engine's own RTL conventions, i.e. always.

        Ordered by the suffix FOLLOWING the prefix, so when both sides name the
        same channel the caller's positional pairing is suffix-to-suffix. Shared
        clock/reset, power pins and the wrapper's pad vector are excluded for the
        same reasons ``_bundle_ports`` excludes them: they are wired globally,
        not per edge.

        Returns [] when nothing matches, which the caller already treats as a
        hazard. This only ever ADDS resolutions -- every width, count and suffix
        check downstream still applies.
        """
        pre = str(chan).lower() + "_"
        out: list[tuple[str, VerilogPort]] = []
        for p in modules[bn].ports:
            low = p.name.lower()
            if not low.startswith(pre) or len(low) == len(pre):
                continue
            if _wrap_shared_signal(p.name) or _is_power_pin(p.name):
                continue
            if bn == wrapper_block and p.name in _PAD_IO_PORTS:
                continue
            out.append((low[len(pre):], p))
        out.sort(key=lambda t: t[0])
        return [p for _sfx, p in out]

    keyed = {bn: _bundle_ports(bn) for bn in modules}
    uf = _WireUnion()
    # Wiring hazards that make the deterministic top UNSAFE. When non-empty the
    # caller must fall back to the LLM Integration-Lead rather than ship a
    # mis-wired top (a shorted / multi-driven / width-truncated net).
    wiring_errors: list[str] = []

    # (1) STRUCTURED contract first: union the EXACT producer_port<->consumer_port
    #     named by each edge. This is what disambiguates a normalized-key
    #     collision -- e.g. a block exposing both `rd_addr` (24b) and
    #     `in_rd_addr` (8b) both key to "rd_addr"; the edge names which one this
    #     connection instantiates, so we bind that specific field instead of
    #     [0]-picking. A width mismatch across a NAMED edge is a real contract
    #     break -> refuse (fall back), never truncate.
    def _resolve_fields(bn: str, pname) -> list[VerilogPort] | None:
        """Resolve an edge port name to the block's RTL ports.

        A contract may name a channel's fields as a SLASH-COMPOUND string
        (``"m_x_srdy/m_x_data"``); each field must resolve individually.
        Returns None when the edge names no port (normalized-key fallback
        applies), or [] when the edge NAMES ports that don't resolve -- which
        the caller must treat as a wiring hazard, never a silent skip: a
        skipped named edge leaves the whole channel dangling and the
        deterministic top ties it to constants (observed: 11 cross-named
        srdy/drdy channels tied to 1'b0 -> START never reached the engine).
        """
        if not pname:
            return None
        parts = [s for s in re.split(r"[/,\s]+", str(pname)) if s]
        out: list[VerilogPort] = []
        for part in parts:
            p = _port_exact(bn, part)
            if p is not None:
                out.append(p)
                continue
            # Not a port name -- try it as a CHANNEL whose fields the RTL
            # prefix-expanded. Exact match is still preferred, so a block that
            # really has a port of this name is unaffected.
            grp = _channel_fields(bn, part)
            if not grp:
                return []
            out.extend(grp)
        return out

    edge_bound: set[frozenset] = set()
    for e in edges:
        pb, cb = e.get("producer_block"), e.get("consumer_block")
        if pb not in modules or cb not in modules or pb == cb:
            continue
        # Contract-directed resolution: match each END against the contract's
        # declared signal list, never against the other end. See
        # _resolve_by_contract for why this removes positional pairing.
        _by_contract = _resolve_by_contract(e, pb, cb, _port_exact, modules)
        if _by_contract is not None:
            _paired, _hazards = _by_contract
            if _hazards:
                wiring_errors.extend(_hazards)
                continue
            for pp, cp in _paired:
                uf.union((pb, pp.name), (cb, cp.name))
            if _paired:
                edge_bound.add(frozenset((pb, cb)))
            continue

        pfields = _resolve_fields(pb, e.get("producer_port", ""))
        cfields = _resolve_fields(cb, e.get("consumer_port", ""))
        # A NAMED port that fails to resolve is a hazard (fall back to the
        # LLM integrator), not a silent skip; an UN-named side defers to the
        # normalized-key fallback exactly as before.
        if (pfields is not None and not pfields) or (
                cfields is not None and not cfields):
            _side = "producer" if not pfields else "consumer"
            wiring_errors.append(
                f"edge {e.get('edge_id')}: {_side} port name "
                f"{(e.get('producer_port') if _side == 'producer' else e.get('consumer_port'))!r} "
                f"does not resolve to RTL port(s) -- refusing to silently "
                f"skip a NAMED contract edge (the channel would be tied to "
                f"constants)")
            continue
        if pfields is None or cfields is None:
            continue
        if len(pfields) != len(cfields):
            wiring_errors.append(
                f"edge {e.get('edge_id')}: producer names {len(pfields)} "
                f"field(s) but consumer names {len(cfields)} -- cannot pair")
            continue
        # Pair fields positionally; the trailing suffix token must agree so a
        # cross-NAMED channel (m_work_write_* <-> s_butterfly_write_*) binds
        # srdy<->srdy / data<->data / drdy<->drdy, never crosswise.
        paired: list[tuple] = []
        ok = True
        for pp, cp in zip(pfields, cfields):
            if _wrap_shared_signal(pp.name) or _wrap_shared_signal(cp.name):
                continue
            _sp = pp.name.rsplit("_", 1)[-1].lower()
            _sc = cp.name.rsplit("_", 1)[-1].lower()
            if len(pfields) > 1 and _sp != _sc:
                wiring_errors.append(
                    f"edge {e.get('edge_id')}: field pairing {pb}.{pp.name} "
                    f"<-> {cb}.{cp.name} suffix mismatch ('{_sp}' vs '{_sc}')")
                ok = False
                break
            if pp.width != cp.width:
                wiring_errors.append(
                    f"edge {e.get('edge_id')}: producer {pb}.{pp.name} "
                    f"[{pp.width}] width != consumer {cb}.{cp.name} "
                    f"[{cp.width}] -- refusing to short mismatched-width ports")
                ok = False
                break
            paired.append((pp, cp))
        if not ok:
            continue
        for pp, cp in paired:
            uf.union((pb, pp.name), (cb, cp.name))
        if paired:
            edge_bound.add(frozenset((pb, cb)))

    # (2) Normalized-key fallback for connected pairs the contract did NOT bind
    #     port-by-port -- but ONLY when the key is UNAMBIGUOUS (exactly one port
    #     each side) AND the widths match. A key that maps to MULTIPLE ports on a
    #     side is a genuine collision: a [0]-pick would short distinct nets
    #     (rd_addr 24b vs in_rd_addr 8b), so record a hazard and let the LLM
    #     integrator resolve it. A width mismatch is likewise a hazard, not a
    #     max-width truncation.
    for pair in pairs:
        a, b = tuple(pair)
        for key in sorted(set(keyed[a]) & set(keyed[b])):
            pa, pbp = keyed[a][key], keyed[b][key]
            if len(pa) != 1 or len(pbp) != 1:
                if frozenset((a, b)) not in edge_bound:
                    wiring_errors.append(
                        f"pair {a}<->{b}: signal key '{key}' is ambiguous "
                        f"({a} exposes {len(pa)}, {b} exposes {len(pbp)} ports) "
                        f"and no interface_contract edge names the exact ports -- "
                        f"refusing to [0]-pick a wire")
                continue
            if pa[0].width != pbp[0].width:
                wiring_errors.append(
                    f"pair {a}<->{b}: '{key}' width {pa[0].width} "
                    f"({a}.{pa[0].name}) != {pbp[0].width} ({b}.{pbp[0].name}) -- "
                    f"refusing to truncate")
                continue
            uf.union((a, pa[0].name), (b, pbp[0].name))

    # ---- assign one wire per union group ----
    port_wire: dict[tuple[str, str], str] = {}
    wire_decls: list[str] = []
    used_names: set[str] = set()
    for root, members in sorted(uf.groups().items(), key=lambda kv: kv[0]):
        if len(members) < 2:
            continue  # unmatched -> handled as dangling below
        # widths in a group are guaranteed equal by the union guards above; a
        # residual mismatch (transitive merge) is a hazard, not a silent max.
        widths = {modules[bn].port_by_name(pn).width
                  for (bn, pn) in members if modules[bn].port_by_name(pn)}
        if len(widths) > 1:
            wiring_errors.append(
                f"wire group {sorted(members)} mixes widths {sorted(widths)} -- "
                f"refusing to short different-width nets")
            continue
        width = max(widths) if widths else 1
        rb, rp = root
        wname = re.sub(r"[^a-zA-Z0-9_]", "_", f"w_{rb}_{rp}")
        base = wname
        i = 1
        while wname in used_names:
            i += 1
            wname = f"{base}_{i}"
        used_names.add(wname)
        wire_decls.append(
            f"  wire [{width-1}:0] {wname};" if width > 1 else f"  wire {wname};"
        )
        for m in members:
            port_wire[m] = wname

    # ---- pad-adapter module rename (dodge collision with the top name) ----
    renamed_pad_path = ""
    lint_block_paths = dict(rtl_paths)
    wrap_inst_module = ""
    if wrapper_block is not None:
        wrap_mod_name = modules[wrapper_block].name
        wrap_inst_module = wrap_mod_name
        src_path = rtl_paths.get(wrapper_block, "")
        try:
            src = Path(src_path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            src = ""
        # `\b` after the name means this does NOT match `user_project_wrapper_io`
        # (an underscore is a word character), so the block's own module is safe.
        _top_decl = re.search(
            rf"\bmodule\s+{re.escape(CARAVEL_TOP_MODULE)}\b", src) if src else None
        renamed = ""
        if wrap_mod_name == CARAVEL_TOP_MODULE:
            # The pad block IS named like the graded top. Rename it so the top
            # this function emits can take that name.
            wrap_inst_module = f"{CARAVEL_TOP_MODULE}_pads"
            renamed = re.sub(
                rf"\bmodule\s+{re.escape(CARAVEL_TOP_MODULE)}\b",
                f"module {wrap_inst_module}",
                src,
                count=1,
            )
        elif _top_decl:
            # The block's own module is named something else, but its FILE also
            # declares the graded module -- a thin alias wrapper emitted next to
            # the block. Left alone it collides with the top emitted below: two
            # definitions of CARAVEL_TOP_MODULE, which is a MODDUP abort at
            # elaboration and, worse, makes resolve_netlist_top see TWO
            # un-instantiated roots so the chip gate-sim cannot resolve a top at
            # all. The alias is redundant -- this function emits that module --
            # so drop it from a COPY. The block's own source is never modified.
            _end = src.find("endmodule", _top_decl.start())
            if _end != -1:
                renamed = (src[:_top_decl.start()]
                           + f"// [coresmith] removed a redundant "
                             f"`module {CARAVEL_TOP_MODULE}` alias: the "
                             f"integration stage emits that module itself, and "
                             f"two definitions collide.\n"
                           + src[_end + len("endmodule"):])
        if renamed:
            renamed_pad_path = str(out_dir / f"{wrap_inst_module}.v")
            Path(renamed_pad_path).write_text(renamed, encoding="utf-8")
            lint_block_paths[wrapper_block] = renamed_pad_path

    # ---- pin routing (from the PRD's structured pin_map) ----
    # With the assignment available as data the top slices io_in itself and
    # drives io_out/io_oeb itself, so the design needs no pin-adapter block --
    # the block an LLM could not generate because its mandated module name
    # means "the entire chip".
    pin_decls: list[str] = []
    pin_assigns: list[str] = []
    pin_sigs: dict = {}
    pin_block_driven: dict = {}
    if pin_map is not None and getattr(pin_map, "ok", False):
        from orchestrator.architecture.pin_map import (
            emit_pin_routing,
            mapped_signals,
        )
        pin_decls, pin_assigns = emit_pin_routing(pin_map)
        pin_sigs = mapped_signals(pin_map)
        # Output-side signals (and any output-enable) are DRIVEN BY BLOCKS, so
        # the top declares a wire for each and the producing block connects to
        # it. Input-side signals are already assigned from io_in by pin_decls.
        for e in pin_map.entries:
            if e.dir == "out":
                pin_block_driven[e.signal] = e.width
            if e.oe:
                pin_block_driven[e.oe] = 1
        pin_sigs.update(pin_block_driven)

    # ---- emit ----
    lines: list[str] = []
    lines.append("// Auto-generated Caravel user_project_wrapper chip_top")
    lines.append(f"// Blocks: {len(modules)} (wired by coresmith integration)")
    lines.append("// Instantiates + wires all blocks; pad adapter routes GPIO.")
    lines.append("")
    lines.append(f"module {CARAVEL_TOP_MODULE} (")
    lines.append("`ifdef USE_POWER_PINS")
    lines.append(
        ",\n".join(f"    inout  wire {pn}" for pn in _CARAVEL_POWER_PORTS) + ","
    )
    lines.append("`endif")
    port_lines = [
        f"    {d:<6} wire {w}{name}" for (d, w, name) in _CARAVEL_TOP_PORTS
    ]
    lines.append(",\n".join(port_lines))
    lines.append(");")
    lines.append("")

    # tie off unused Caravel interfaces
    lines.append("  // ---- tie off unused Caravel interfaces ----")
    for name, val in _CARAVEL_TIEOFFS:
        lines.append(f"  assign {name} = {val};")
    lines.append("")

    if pin_decls or pin_block_driven:
        lines.append("  // ---- pin map (declared in the PRD) ----")
        for sig, w in sorted(pin_block_driven.items()):
            rng = "" if w == 1 else f"[{w - 1}:0] "
            lines.append(f"  wire {rng}{sig};")
        for d in pin_decls:
            lines.append(f"  {d}")
        lines.append("")

    if wire_decls:
        lines.append(f"  // ---- internal block<->block wires ({len(wire_decls)}) ----")
        lines.extend(sorted(wire_decls))
        lines.append("")

    instantiated: list[str] = []
    for bn in sorted(modules):
        mod = modules[bn]
        inst_module = wrap_inst_module if bn == wrapper_block else mod.name
        lines.append(f"  // {bn}")
        lines.append(f"  {inst_module} u_{bn} (")
        conns: list[str] = []
        power_conns: list[str] = []
        for p in mod.ports:
            if _is_power_pin(p.name):
                # Supply pins only exist under USE_POWER_PINS on BOTH sides;
                # connect same-named (top exposes the identical guarded pins).
                power_conns.append(p.name)
                continue
            sig = _wrap_shared_signal(p.name)
            if sig == "clk":
                conns.append(f"    .{p.name}({clk_name})")
                continue
            if sig == "rst":
                port_active_low = "n" in p.name.lower()
                expr = f"~{rst_name}" if port_active_low else rst_name
                conns.append(f"    .{p.name}({expr})")
                continue
            if bn == wrapper_block and p.name in _PAD_IO_PORTS:
                conns.append(f"    .{p.name}({p.name})")  # straight to top GPIO
                continue
            if p.name in pin_sigs:
                # A block port named after a pin-map signal binds to it. The
                # contract already uses these names, so this needs no new
                # convention.
                conns.append(f"    .{p.name}({p.name})")
                continue
            w = port_wire.get((bn, p.name))
            if w:
                conns.append(f"    .{p.name}({w})")
            elif p.direction == "input":
                tie = f"{p.width}'b0" if p.width > 1 else "1'b0"
                conns.append(f"    .{p.name}({tie})")
            else:  # dangling output -> leave open
                conns.append(f"    .{p.name}()")
        lines.append(",\n".join(conns))
        if power_conns:
            # Leading-comma continuation so it stitches onto the list whether or
            # not USE_POWER_PINS is defined (only omit the lead comma if there
            # were no regular conns, which never happens -- every block has clk).
            lead = "" if not conns else ", "
            pc_lines = [f"    {lead if i == 0 else ', '}.{pn}({pn})"
                        for i, pn in enumerate(power_conns)]
            lines.append("`ifdef USE_POWER_PINS")
            lines.extend(pc_lines)
            lines.append("`endif")
        lines.append("  );")
        lines.append("")
        instantiated.append(bn)

    if pin_assigns:
        lines.append("")
        lines.append("  // ---- pin map: drive the pads ----")
        for a in pin_assigns:
            lines.append(f"  {a}")
    lines.append("endmodule")
    lines.append("")
    verilog = "\n".join(lines)

    rtl_path = out_dir / f"{CARAVEL_TOP_MODULE}.v"
    rtl_path.write_text(verilog, encoding="utf-8")

    return {
        "verilog": verilog,
        "rtl_path": str(rtl_path),
        "module_name": CARAVEL_TOP_MODULE,
        "block_count": len(modules),
        "wire_count": len(wire_decls),
        "instantiated": instantiated,
        "wrapper_block": wrapper_block,
        "renamed_pad_path": renamed_pad_path,
        "dropped_adapter": dropped_adapter,
        "lint_block_paths": lint_block_paths,
        # Non-empty => the deterministic wiring is UNSAFE (an ambiguous key that
        # would be [0]-picked, or a width mismatch that would be shorted). The
        # caller must NOT ship this top -- fall back to the LLM Integration-Lead.
        "wiring_errors": wiring_errors,
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
    # Assemble the full source list (top + blocks), in the same order the sim
    # uses, so lint and sim see an identical set of definitions.
    lint_sources: list[str] = [top_rtl_path]
    for bp in block_rtl_paths:
        if Path(bp).exists() and bp != top_rtl_path:
            lint_sources.append(bp)

    # Include the generic SRAM wrapper lib if any block instantiates cs_sram, so
    # Verilator can resolve cs_sram_1rw/1rw1r at the chip level (per-block lint
    # already does this; integration lint must too, else "Cannot find module
    # cs_sram_1rw1r"). Best-effort.
    try:
        from orchestrator.langgraph.sram_wrapper import (
            uses_wrapper as _uses_wrapper,
        )
        from orchestrator.langgraph.sram_wrapper import (
            wrapper_lib_path as _wrapper_lib_path,
        )
        _all_rtl = "".join(
            Path(p).read_text() for p in lint_sources if Path(p).exists()
        )
        if _uses_wrapper(_all_rtl):
            lint_sources.append(_wrapper_lib_path())
    except Exception:
        pass

    # Dedup duplicate module definitions BEFORE linting, exactly as
    # run_integration_simulation does, so lint and sim agree. Without this an
    # LLM-authored empty cs_sram_* blackbox in chip_top + the real rtl_lib body
    # would either MODDUP-abort lint or (worse) lint clean against the stub while
    # sim keeps the lib body -- the two stages must see the same sources. Writes
    # deduped copies into a scratch dir alongside the run logs.
    try:
        _dd_dir = PROJECT_ROOT / "sim_build" / "integration_lint"
        _dd_dir.mkdir(parents=True, exist_ok=True)
        lint_sources = _dedup_module_sources(lint_sources, _dd_dir)
    except Exception:
        pass

    cmd = [
        "verilator", "--lint-only", "-Wall", "-Wno-fatal",
        "-Wno-EOFNEWLINE",
        "--top-module", Path(top_rtl_path).stem,
        *lint_sources,
    ]

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
    modules: dict[str, VerilogModule],
    connections: list[dict],
    block_rtl_paths: dict[str, str],
    prd_summary: str = "",
    prior_failure: str = "",
    chip_model_path: str = "",
    parameter_table: str = "",
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
        chip_model_path=chip_model_path,
        parameter_table=parameter_table,
    )

    result["testbench_path"] = result.get("tb_path", output_path)
    return result


async def generate_validation_testbench(
    design_name: str,
    top_rtl_path: str,
    modules: dict[str, VerilogModule],
    connections: list[dict],
    block_rtl_paths: dict[str, str],
    ers_context: str,
    prior_failure: str = "",
    reference_path: str = "",
    reference_entry: str = "",
    parameter_table: str = "",
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
        reference_path=reference_path,
        reference_entry=reference_entry,
        parameter_table=parameter_table,
    )

    result["testbench_path"] = result.get("tb_path", output_path)
    return result


def _rtl_lib_module_names() -> set[str]:
    """Module names defined by the shared rtl_lib memory library.

    These are AUTHORITATIVE: their real (behavioral) body lives in
    ``rtl_lib/cs_sram.v`` and must survive dedup regardless of source order. A
    chip_top that (incorrectly) authors its own empty/blackbox copy of one of
    these names must lose to the lib copy, not win by being first. Detected by
    parsing the lib so the set never drifts from the file.
    """
    names: set[str] = set()
    try:
        from orchestrator.langgraph.sram_wrapper import wrapper_lib_path
        lib_src = Path(wrapper_lib_path()).read_text(errors="ignore")
        for m in re.finditer(r"^\s*module\s+([A-Za-z_]\w*)", lib_src, re.MULTILINE):
            names.add(m.group(1))
    except Exception:
        # Fallback to the known family if the lib can't be read.
        names = {
            "cs_mem_1rw", "cs_mem_1rw1r", "cs_mem_macro_shell",
            "cs_sram_1rw", "cs_sram_1rw1r",
            "cs_fpmem_1rw", "cs_fpmem_1rw1r",
        }
    return names


def _is_rtl_lib_path(src: str) -> bool:
    """True if ``src`` is the shared rtl_lib memory library file."""
    try:
        return "rtl_lib" in Path(src).parts
    except Exception:
        return "rtl_lib" in str(src)


_INCLUDE_DIRECTIVE_RE = re.compile(r'^\s*`include\s+"([^"]+)"', re.MULTILINE)


def _drop_include_provided_sources(sources: list[str]) -> list[str]:
    """Drop listed sources that another listed source already provides via
    ``\\`include``.

    yosys/Verilator expand an ``\\`include`` at read time, so a source list
    that carries BOTH a self-contained wrapper (which ``\\`include``s its block
    RTLs) AND those block files as separate entries double-defines every
    included module -- a MODDUP the module-level deduper below cannot see
    (it reads each listed file's own text; the include expansion happens
    inside the tool). Matching is by resolved include path, then by basename
    (a wrapper generated against a different engine checkout ``\\`include``s
    the same-named library file from another absolute path). The FIRST entry
    (the top) is never dropped.
    """
    texts: dict[str, str] = {}
    for src in sources:
        try:
            texts[src] = Path(src).read_text(errors="ignore")
        except OSError:
            texts[src] = ""
    provided: set[str] = set()
    for src in sources:
        for m in _INCLUDE_DIRECTIVE_RE.finditer(texts[src]):
            inc = m.group(1)
            ip = Path(inc)
            if not ip.is_absolute():
                ip = Path(src).parent / inc
            try:
                provided.add(str(ip.resolve()))
            except OSError:
                provided.add(str(ip))
            provided.add(Path(inc).name)
    if not provided:
        return list(sources)
    out: list[str] = []
    for i, src in enumerate(sources):
        if i > 0:
            try:
                rp = str(Path(src).resolve())
            except OSError:
                rp = src
            if rp in provided or Path(src).name in provided:
                continue
        out.append(src)
    return out


def _dedup_module_sources(
    sources: list[str], out_dir, lib_module_names: set[str] | None = None
) -> list[str]:
    """Return a source list with duplicate ``module`` definitions removed.

    ``lib_module_names`` (default: the names parsed from ``rtl_lib/cs_sram.v``)
    are LIBRARY cells whose body is authoritative: for those names the copy that
    lives in the ``rtl_lib`` file is always kept and any duplicate appearing in a
    NON-lib source (e.g. an empty blackbox the Integration Lead authored into
    chip_top) is stripped -- regardless of source order. This is independent of
    the first-wins rule below, which still governs all other (non-library)
    shared modules.

    Blocks frequently each bundle the SAME shared macro (e.g. a Sky130 SRAM
    behavioural model), so the chip-level sim sees the module declared multiple
    times -> Verilator ``%Error: MODDUP`` at elaboration. Verilog ``\\`ifndef``
    guards do NOT carry across files (separate compilation units), so the robust
    fix is to keep the FIRST definition of each module name across all sources
    and strip the rest, writing deduped copies to ``out_dir``. Files with no
    duplicate are passed through unchanged.

    MODDUP is a *cross-file* hazard: it only fires when two separate compilation
    units both define the same module. A module name appearing twice *within one
    file* is almost always a ``\\`ifdef SYNTHESIS`` blackbox / ``\\`else``
    behavioural pair where exactly one branch is active per compile. This
    deduper is intentionally NOT preprocessor-aware, so stripping the second
    in-file occurrence would delete the only definition visible under the active
    define (e.g. the behavioural SRAM model under simulation, since SYNTHESIS is
    not defined) and break Verilator elaboration with "Cannot find module".

    Therefore we only strip occurrences whose name was first defined in an
    EARLIER file, and never the second+ occurrence within the same file. A name
    becomes "seen" for cross-file purposes only after the whole file is
    processed, so intra-file ifdef/else variants are preserved verbatim.
    """
    import re

    if lib_module_names is None:
        lib_module_names = _rtl_lib_module_names()

    mod_re = re.compile(r"^\s*module\s+([A-Za-z_]\w*)")
    end_re = re.compile(r"^\s*endmodule\b")
    seen: set[str] = set()  # module names defined in an EARLIER file
    out_paths: list[str] = []
    for src in sources:
        p = Path(src)
        is_lib = _is_rtl_lib_path(src)
        try:
            lines = p.read_text(errors="ignore").split("\n")
        except OSError:
            out_paths.append(src)
            continue
        out: list[str] = []
        i = 0
        changed = False
        defined_here: set[str] = set()
        # C23: track `ifdef SYNTHESIS nesting. A module declared inside an
        # active `ifdef SYNTHESIS region is a SYNTHESIS-ONLY blackbox -- it is
        # NOT compiled under the Verilator lint/sim (which does not define
        # SYNTHESIS), so it must NOT become "seen" for cross-file dedup and
        # thereby shadow/strip the REAL definition that lives in another file.
        # Bug: a pad wrapper's `ifdef SYNTHESIS blackbox of a child module came
        # FIRST, so every real child .v was rewritten to "removed duplicate",
        # and under lint (SYNTHESIS undefined) the blackbox was absent too ->
        # %Error-MODMISSING for every child. Stack holds, per open `ifdef, True
        # when the CURRENT branch is synthesis-only (inactive under lint).
        synth_stack: list[bool] = []

        def _lint_active() -> bool:
            return not any(synth_stack)
        while i < len(lines):
            _s = lines[i].lstrip()
            if _s.startswith("`ifdef") or _s.startswith("`ifndef"):
                _neg = _s.startswith("`ifndef")
                _is_synth = bool(re.match(r"`ifn?def\s+SYNTHESIS\b", _s))
                # synthesis-only branch (inactive under lint): `ifdef SYNTHESIS
                # -> True; `ifndef SYNTHESIS -> False (active under lint); any
                # non-SYNTHESIS guard -> assume active (False).
                synth_stack.append(_is_synth and not _neg)
                out.append(lines[i])
                i += 1
                continue
            if _s.startswith("`else"):
                if synth_stack:
                    synth_stack[-1] = not synth_stack[-1]
                out.append(lines[i])
                i += 1
                continue
            if _s.startswith("`endif"):
                if synth_stack:
                    synth_stack.pop()
                out.append(lines[i])
                i += 1
                continue
            m = mod_re.match(lines[i])
            if m:
                name = m.group(1)
                j = i
                while j < len(lines) and not end_re.match(lines[j]):
                    j += 1
                # Strip in two cases:
                #  (1) LIBRARY-CELL OVERRIDE: a cs_mem_*/cs_sram_*/cs_fpmem_*
                #      definition that appears OUTSIDE the rtl_lib file. Its real
                #      behavioral body is authoritative and lives in rtl_lib, so
                #      a copy authored into chip_top (often an empty blackbox the
                #      Integration Lead wrote) must be removed regardless of
                #      source order -- the lib copy is always the one kept. This
                #      is the regression that made SRAM-backed blocks read 0.
                #  (2) CROSS-FILE duplicate (name already defined by an earlier
                #      file). A repeat within THIS file is an ifdef/else variant
                #      and must be kept so the active branch survives.
                lib_override = (name in lib_module_names) and not is_lib
                cross_file_dup = (name in seen and name not in defined_here)
                if lib_override or cross_file_dup:
                    why = "library cell (authoritative copy is in rtl_lib)" \
                        if lib_override else "duplicate module"
                    out.append(f"// [coresmith dedup] removed {why} {name}")
                    changed = True
                else:
                    # C23: a SYNTHESIS-only blackbox (inside an active
                    # `ifdef SYNTHESIS) is kept verbatim but must NOT be
                    # promoted to the cross-file "seen" set -- it is not the
                    # lint/sim-active definition and must not strip the real
                    # def in a later file.
                    if _lint_active():
                        defined_here.add(name)
                    out.extend(lines[i:j + 1])
                i = j + 1
            else:
                out.append(lines[i])
                i += 1
        # Promote this file's definitions to cross-file scope only now, so that
        # a same-file second occurrence above was never treated as a duplicate.
        # (Library names stripped under the override rule were never added to
        # defined_here, so they don't poison the cross-file set either.)
        seen |= defined_here
        if changed:
            dp = Path(out_dir) / ("_dd_" + p.name)
            dp.write_text("\n".join(out))
            out_paths.append(str(dp))
        else:
            out_paths.append(src)
    return out_paths


def chip_rtl_sources(
    top_rtl_path: str,
    block_rtl_paths: dict[str, str],
    dedup_dir=None,
) -> list[str]:
    """Every Verilog source needed to elaborate the ASSEMBLED chip, top first.

    Single definition on purpose. The integration/validation DV sims and the
    chip_top gate-sim's REFERENCE run must elaborate the identical source set:
    the gate compares the flat netlist against that reference, so a reference
    built from a different source list is not a reference at all -- it is a
    second design, and any verdict against it is meaningless.

    Top-first ordering matters: callers resolve the Verilator TOPLEVEL from the
    first entry.
    """
    sources = [top_rtl_path]
    for bp in block_rtl_paths.values():
        if Path(bp).exists() and bp != top_rtl_path:
            sources.append(bp)
    # Include the generic SRAM wrapper lib if any block instantiates cs_sram, so
    # the chip-level Verilator build can resolve cs_sram_1rw/1rw1r (without it
    # the sim hard-fails with "Cannot find module cs_sram_1rw1r"). Best-effort.
    try:
        from orchestrator.langgraph.sram_wrapper import (
            uses_wrapper as _uses_wrapper,
        )
        from orchestrator.langgraph.sram_wrapper import (
            wrapper_lib_path as _wrapper_lib_path,
        )
        _all_rtl = "".join(
            Path(p).read_text(errors="replace")
            for p in sources if Path(p).exists()
        )
        _wlib = _wrapper_lib_path()
        if _uses_wrapper(_all_rtl) and _wlib not in sources:
            sources.append(_wlib)
    except Exception:
        pass
    # A deterministically-assembled Caravel top and the pad-adapter BLOCK it was
    # built from both declare `module user_project_wrapper`, and blocks commonly
    # each bundle the same shared macro. Two compilation units defining one
    # module is a Verilator MODDUP abort before any transaction runs, so a caller
    # that hands this list straight to a simulator must dedup. Pass a scratch dir
    # to get an elaborable list back; omit it to get raw paths.
    if dedup_dir is not None:
        # _dedup_module_sources WRITES the stripped copies and does not create
        # its own output dir, so a caller passing a fresh scratch path would get
        # back paths to files that do not exist.
        Path(dedup_dir).mkdir(parents=True, exist_ok=True)
        sources = _dedup_module_sources(sources, dedup_dir)
    return sources


def run_integration_simulation(
    design_name: str,
    top_rtl_path: str,
    block_rtl_paths: dict[str, str],
    tb_path: str,
    attempt: int = 1,
    sim_scope: str = "integration",
) -> dict:
    """Run cocotb simulation on the integrated top-level design.

    Sets up a Makefile with all block Verilog sources + top-level,
    runs cocotb via Verilator, and returns pass/fail + logs.

    ``sim_scope`` namespaces the sim build dir and the step log so the
    integration_dv and validation_dv runs (both driven by this function) do not
    clobber each other. It defaults to ``"integration"`` (byte-identical to the
    historical behavior); validation_dv passes ``"validation"`` so it gets
    ``sim_build/validation`` + ``step_logs/integration/validation_sim_attempt<N>.log``
    -- preserving the integration run's raw sim log for forensics and avoiding
    build-fingerprint churn between the two runs.

    Returns:
        dict with: passed (bool), log (str), returncode (int), log_path (str).
    """
    import os
    import shutil

    from orchestrator.langgraph.pipeline_helpers import (
        _normalize_cocotb_timing_keywords,
        _parse_cocotb_summary,
    )

    # Distinct sim build dir per scope (avoids fingerprint churn: the two runs
    # differ only in MODULE, which would otherwise trigger a full rebuild on
    # every integration<->validation switch through a shared dir).
    sim_dir = PROJECT_ROOT / "sim_build" / sim_scope
    sim_dir.mkdir(parents=True, exist_ok=True)
    # Distinct step-log name per scope (validation -> validation_sim_attempt<N>.log)
    # so validation_dv never overwrites integration_dv's raw sim log.
    _log_step = f"{sim_scope}_sim"

    all_sources = chip_rtl_sources(top_rtl_path, block_rtl_paths)
    # Dedup shared macro modules (e.g. SRAM) bundled by multiple blocks, else
    # Verilator MODDUP-aborts elaboration before any transaction.
    all_sources = _dedup_module_sources(all_sources, sim_dir)
    sources_str = " ".join(all_sources)

    # The integration file may deliberately contain more than one wrapper
    # module (for example a 44-pad OpenFrame parent and the graded Caravel
    # user_project_wrapper).  A filename-derived TOPLEVEL always selects the
    # first/file-stem wrapper even when the pipeline explicitly chose the real
    # graded module. Honor design_name when that module is declared in the file;
    # retain the historical file-stem fallback for ordinary generated tops.
    safe_name = Path(top_rtl_path).stem
    try:
        _top_source = Path(top_rtl_path).read_text(
            encoding="utf-8", errors="replace"
        )
        if re.search(
            rf"^\s*module\s+{re.escape(design_name)}\b",
            _top_source,
            re.MULTILINE,
        ):
            safe_name = design_name
    except OSError:
        pass

    # TRACE IS MANDATORY on the integration/validation sim path (2026-07-02 fix).
    # This function backs BOTH integration_dv and validation_dv, whose PASS verdict
    # HARD-REQUIRES a WaveKit VCD audit: `run_wavekit_vcd_audit` fail-closes on a
    # missing/empty VCD and the caller gates `passed` on `wavekit_audit["ok"] is
    # True`. Previously the trace was gated behind CORESMITH_SIM_TRACE=1 (default
    # OFF), so a healthy 6/6-passing sim could STRUCTURALLY never pass integration
    # DV -- it emitted no dump.vcd and the audit fail-closed on it.
    #
    # The trace was only ever gated to dodge an OOM from `--trace --trace-structs`
    # C++ built in PARALLEL fork-storming a 4-core host (2026-07-01). That storm is
    # already prevented by the SERIAL build below (`--build-jobs 1` here + `make
    # -j1` in the caller), so tracing on/off can never fork-storm. Force it ON
    # UNCONDITIONALLY here so the mandatory VCD audit is satisfiable; per-block
    # sims (run_simulation) already trace unconditionally -- only this path was
    # env-gated. (Per-block sims may stay env-gated for speed if ever needed; the
    # integration/validation path may not, because its audit is fail-closed.)
    _waves = "1"
    _trace_args = "--trace --trace-structs"
    makefile_content = f"""
SIM = verilator
TOPLEVEL_LANG = verilog
VERILOG_SOURCES = {sources_str}
TOPLEVEL = {safe_name}
MODULE = {Path(tb_path).stem}
WAVES = {_waves}
EXTRA_ARGS += {_trace_args}
EXTRA_ARGS += --build-jobs 1
include $(shell cocotb-config --makefiles)/Makefile.sim
"""
    # Pre-run hygiene: the INTEGRATION/VALIDATION sims are engine-authoritative,
    # rare, and correctness-critical (their PASS verdict hard-requires a WaveKit
    # VCD audit). Unconditionally wipe any pre-existing build products in this dir
    # before writing our traced Makefile: an agent's in-context `verify` (or an
    # earlier aborted run) may have left a stale/traceless Vtop here, and cocotb's
    # make would REUSE it (its mtime predates our fresh Makefile), emit no
    # dump.vcd, and fail-close the mandatory audit on a phantom-missing VCD
    # (2026-07-02 integration-DV failure). Per-block sims (run_simulation) keep the
    # fingerprint fast-path -- they are frequent and cheap; only this path forces a
    # clean rebuild.
    clear_build_products(sim_dir)
    # Fingerprint the (now clean) build inputs so a later flag/source change is
    # still caught by the mismatch path (and so the fingerprint file stays current).
    apply_build_fingerprint(sim_dir, makefile_content, all_sources)
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
    # SERIAL make (-j1): with `--build-jobs 1` in the Makefile this keeps the
    # full-chip Verilator build single-threaded so it can never fork-storm the
    # host (the 2026-07-01 incident). Raise only on a big box via
    # CORESMITH_SIM_MAKE_J.
    env["MAKEFLAGS"] = (
        env.get("MAKEFLAGS", "")
        + f" -j{os.environ.get('CORESMITH_SIM_MAKE_J', '1')}"
    ).strip()

    make_bin = shutil.which("make") or "make"

    # ELASTIC SIM TIMEOUT [dv-hardening-13]: the RTL-stage DV wrappers had a
    # HARD 600s cap while the block tier auto-extends (x1.5 on prior timeout,
    # capped) -- exactly backwards, since integration/validation sims are the
    # biggest. Same mechanism as run_simulation: base from env, x1.5 per
    # recorded prior timeout for THIS scope, capped; state persists across
    # attempts/restarts.
    _to_cap = int(float(os.environ.get(
        "CORESMITH_INTEGRATION_SIM_TIMEOUT_CAP_S", "3600") or 3600))
    _to_base = int(float(os.environ.get(
        "CORESMITH_INTEGRATION_SIM_TIMEOUT_S", "600") or 600))
    _to_state = sim_dir / "sim_timeout_state.json"
    _prior_to = 0
    try:
        if _to_state.exists():
            _prior_to = int(json.loads(_to_state.read_text()).get("timeouts", 0))
    except Exception:  # noqa: BLE001
        _prior_to = 0
    _sim_timeout = min(int(_to_base * (1.5 ** _prior_to)), _to_cap)
    print(f"[{sim_scope.upper()}-SIM] timeout={_sim_timeout}s "
          f"(base={_to_base}s prior_timeouts={_prior_to} cap={_to_cap}s)",
          flush=True)

    try:
        # Own process group so a timeout kills the WHOLE tree (make -> verilator
        # -> g++ / vvp) instead of orphaning compilers/sims (the block-tier
        # fork-bomb fix, applied here too).
        import signal as _signal

        _proc = subprocess.Popen(
            [make_bin, "-C", str(sim_dir)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,
        )
        try:
            _out, _err = _proc.communicate(timeout=_sim_timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(_proc.pid), _signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            _proc.wait()
            raise
        result = subprocess.CompletedProcess(
            [make_bin, "-C", str(sim_dir)], _proc.returncode, _out, _err,
        )
        log_path = _write_step_log(
            "integration", _log_step, [make_bin, "-C", str(sim_dir)],
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
        # Persist the timeout so the NEXT attempt auto-extends (x1.5, capped).
        try:
            _to_state.write_text(json.dumps({"timeouts": _prior_to + 1}))
        except Exception:  # noqa: BLE001
            pass
        log_path = _write_step_log_error(
            "integration", _log_step, cmd,
            f"SIM_TIMEOUT: {sim_scope} simulation exceeded {_sim_timeout}s. "
            f"No functional verdict produced. Next attempt auto-extends "
            f"(x1.5, cap {_to_cap}s).", attempt,
        )
        return {"passed": False, "log": "Integration simulation timed out (10 min)", "log_path": log_path}
    except FileNotFoundError as e:
        cmd = [make_bin, "-C", str(sim_dir)]
        log_path = _write_step_log_error(
            "integration", _log_step, cmd, f"Tool not found: {e}", attempt,
        )
        return {"passed": False, "log": f"Tool not found: {e}", "log_path": log_path}


def _eligible_blocks(completed_blocks: list[dict]) -> list[tuple[str, dict]]:
    """(name, block) for every block that is supposed to be in the chip."""
    out = []
    for block in completed_blocks:
        name = block.get("name", block.get("block_name", ""))
        if not name:
            continue
        if block.get("aborted") or block.get("skipped"):
            continue
        out.append((name, block))
    return out


def discover_block_rtl(
    project_root: str,
    completed_blocks: list[dict],
) -> dict[str, str]:
    """Discover RTL file paths for all completed blocks.

    Resolution order: the block result's own ``rtl_path``, then the block
    spec's ``rtl_target``, then filename convention.

    ``rtl_target`` is NOT optional politeness -- it is the only correct answer
    whenever a block's Verilog file is not named after the block, which is
    exactly the case for a block whose module name is locked by an interface
    contract (a Caravel ``user_project_wrapper``, a vendor-mandated top). Before
    it was consulted, such a block resolved to nothing and was dropped from the
    returned dict with no error, which structurally DELETED it from the
    assembled chip: the pad adapter never reached ``modules``, so
    ``detect_wrapper_block`` returned None, the deterministic Caravel assembler
    never ran, and the resulting top -- and netlist, and GDS -- carried no
    io_in/io_out/io_oeb at all.

    An eligible block that still resolves to nothing is logged as an ERROR
    rather than omitted quietly. Callers that must not assemble a partial chip
    should gate on :func:`unresolved_block_rtl`.

    Returns:
        Dict mapping block_name -> rtl_file_path.
    """
    root = Path(project_root)
    rtl_paths: dict[str, str] = {}

    for name, block in _eligible_blocks(completed_blocks):
        # Try rtl_path from block result
        rtl_path = block.get("rtl_path", "")
        if rtl_path and Path(rtl_path).exists():
            rtl_paths[name] = rtl_path
            continue

        # The block spec's declared target. Accepts absolute or
        # project-root-relative, since both forms appear in block_specs.json.
        target = block.get("rtl_target", "") or ""
        if target:
            for cand in (Path(target), root / target):
                if cand.is_file():
                    rtl_paths[name] = str(cand)
                    break
            if name in rtl_paths:
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

        if name not in rtl_paths:
            log(f"  [INTEGRATION] NO RTL FOUND for block '{name}' "
                f"(rtl_target={target or '<unset>'}) -- it will be ABSENT from "
                f"the assembled chip. A block that silently stops existing is "
                f"how a locked pad adapter was dropped and the chip shipped "
                f"with no GPIO boundary.", RED)

    return rtl_paths


def module_for_block(rtl_path, block_name: str) -> str:
    """Module name to parse for ``block_name``. Mirrors rtl_module_name."""
    from pathlib import Path as _P
    try:
        text = _P(rtl_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return block_name
    import re as _re
    for cand in (block_name, _P(rtl_path).stem):
        if cand and _re.search(r"\bmodule\s+" + _re.escape(cand) + r"\b", text):
            return cand
    return block_name


def merge_block_specs(blocks: list[dict], block_queue: list) -> list[dict]:
    """Join per-block RESULT dicts to their SPEC entries by name.

    A block RESULT records what happened ({name, success, attempts, ...}); the
    block SPEC records what the block IS, including ``rtl_target``. Only the
    result reaches discovery, so a block whose Verilog file is not named after
    it -- the contract-locked case ``rtl_target`` exists for -- resolved to
    nothing and was dropped.

    Result keys win on conflict: nothing a block actually reported is
    overwritten by its spec.
    """
    by_name: dict[str, dict] = {}
    for spec in block_queue or []:
        if isinstance(spec, dict):
            n = spec.get("name") or spec.get("block_name")
            if n:
                by_name[str(n)] = spec
    out: list[dict] = []
    for b in blocks or []:
        if not isinstance(b, dict):
            continue
        n = b.get("name") or b.get("block_name")
        spec = by_name.get(str(n)) if n else None
        out.append({**spec, **b} if spec else dict(b))
    return out


def missing_from(
    rtl_paths: dict[str, str],
    completed_blocks: list[dict],
) -> list[str]:
    """Eligible blocks absent from a resolution the CALLER already has.

    This is the form a gate wants. It judges the exact dict the caller is about
    to assemble from, rather than re-deriving one -- so a caller (or a test)
    that supplied or patched its own discovery cannot end up gated on a second,
    different answer.
    """
    return [n for n, _ in _eligible_blocks(completed_blocks) if n not in rtl_paths]


def unresolved_block_rtl(
    project_root: str,
    completed_blocks: list[dict],
) -> list[str]:
    """Eligible blocks whose RTL could not be located, so a caller can fail
    closed instead of assembling a chip that is missing a block.

    Standalone query: it runs its own discovery. A caller that ALREADY has a
    resolution should use :func:`missing_from` on that, so the gate and the
    assembler can never be judging two different dicts.
    """
    return missing_from(
        discover_block_rtl(project_root, completed_blocks), completed_blocks)


def detect_glue_block_needs(
    connections: list[dict],
    modules: dict[str, VerilogModule],
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

        src_port = _find_port_fuzzy(src_mod, from_port, interface_name, prefer_direction="output")
        dst_port = _find_port_fuzzy(dst_mod, to_port, interface_name, prefer_direction="input")
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
