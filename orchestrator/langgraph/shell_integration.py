# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Shell integration (A4): port-passing from day one.

The chip top used to be written by an LLM after every block had passed, so
the first time two real blocks met was the end of the run. Here the top is
assembled DETERMINISTICALLY from the frozen interface contracts:

* ``stub_module`` -- a stand-in for a block that has no RTL yet: exactly the
  ports the contract demands (the conformance gate's canonical
  ``<channel>_<signal>`` names), outputs tied low;
* ``assemble_top`` -- instantiates every block (real RTL or stub) and wires
  each contract edge signal by signal: the producer's ``<pchan>_<sig>`` to the
  consumer's ``<cchan>_<sig>``. Ports on no edge become chip boundary ports
  (a block whose module IS the declared top contributes its pins under their
  own names; its instance is renamed ``<top>_core``);
* the assembled top is elaborated with Verilator after every block passes,
  so interface drift is caught at the block that introduced it, not at the
  end.

Caravel designs keep their dedicated assembler; this one covers every
declared or synthesized top.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from orchestrator.langgraph.contract_conformance import (
    canonical_port,
    channel_base,
    signal_specs,
    strip_preprocessor,
)

_CLK_NAMES = ("clk", "clk_i", "clock")
_RST_NAMES = ("rst_n", "rst_ni", "rst", "reset", "rst_i", "resetn", "reset_n")


def _dir_from_role(role: str, sig_dir: str, kind: str, name: str) -> str:
    """The block-side direction of one contract signal (input|output)."""
    d = (sig_dir or "").lower()
    to_producer = ("consumer->producer" in d) or (name.lower() in ("tready", "drdy", "ready", "req_gnt", "gnt",
                                                                     "rsp_valid", "rvalid", "rdata", "pready",
                                                                     "prdata", "pslverr") and not d)
    if d in ("producer->consumer", "m2s"):
        to_producer = False
    if role == "producer":
        return "input" if to_producer else "output"
    return "output" if to_producer else "input"


@dataclass
class PortSpec:
    name: str
    dir: str
    width: int = 1


@dataclass
class BlockPorts:
    block: str
    ports: dict[str, PortSpec] = field(default_factory=dict)   # port name -> spec
    edge_ports: set[str] = field(default_factory=set)          # ports on some contract edge


def contract_ports(edges: list[dict], block: str) -> BlockPorts:
    """Every port the contracts declare for ``block`` (both roles)."""
    bp = BlockPorts(block=block)
    for e in edges:
        for role, key in (("producer", "producer_port"), ("consumer", "consumer_port")):
            if e.get("producer_block" if role == "producer" else "consumer_block") != block:
                continue
            chan = channel_base(str(e.get(key) or "")) or ""
            for s in signal_specs(e):
                port, _ = canonical_port(chan, s["name"])
                if not port:
                    continue
                try:
                    w = int(s["width"]) if s.get("width") not in ("", None) else 1
                except ValueError:
                    w = 1
                d = _dir_from_role(role, s.get("dir", ""), s.get("kind", ""), s["name"])
                bp.ports[port] = PortSpec(port, d, max(1, w))
                bp.edge_ports.add(port)
    return bp


def stub_module(block: str, ports: BlockPorts, *, clk: str = "clk", rst: str = "rst_n") -> str:
    """A compilable stand-in: contract ports + clk/rst, outputs tied to 0."""
    lines = [f"// STUB for block {block} (A4 shell integration): generated from the interface",
             "// contracts; replaced by the real RTL as soon as the block passes DV.",
             f"module {block} ("]
    decl = [f"  input  wire {clk}", f"  input  wire {rst}"]
    for p in ports.ports.values():
        if p.name in (clk, rst):
            continue
        w = f"[{p.width - 1}:0] " if p.width > 1 else ""
        decl.append(f"  {p.dir:<6} wire {w}{p.name}")
    lines.append(",\n".join(decl))
    lines.append(");")
    for p in ports.ports.values():
        if p.dir == "output" and p.name not in (clk, rst):
            lines.append(f"  assign {p.name} = {p.width}'d0;")
    lines.append("endmodule")
    return "\n".join(lines) + "\n"


def _rtl_ports(rtl_text: str, module: str | None = None) -> dict[str, PortSpec]:
    """Ports declared by a Verilog module header (ANSI style)."""
    text = re.sub(r"//[^\n]*", " ", strip_preprocessor(rtl_text or "", defines=()))
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    m = None
    for cand in re.finditer(r"\bmodule\s+(\w+)\s*(#\s*\(.*?\))?\s*\((.*?)\)\s*;", text, re.S):
        if module is None or cand.group(1) == module:
            m = cand
            break
    if m is None:
        return {}
    out: dict[str, PortSpec] = {}
    for chunk in m.group(3).split(","):
        chunk = chunk.strip()
        pm = re.match(r"(input|output|inout)\s+(?:wire|reg|logic)?\s*(?:signed\s+)?(?:\[\s*([^:\]]+)\s*:\s*([^\]]+)\])?\s*(\w+)", chunk)
        if not pm:
            continue
        w = 1
        if pm.group(2) is not None:
            try:
                w = abs(int(eval(pm.group(2), {"__builtins__": {}}, {})) - int(eval(pm.group(3), {"__builtins__": {}}, {}))) + 1  # noqa: S307
            except Exception:  # noqa: BLE001 - parametric width: keep 1, the wire is still declared
                w = 1
        out[pm.group(4)] = PortSpec(pm.group(4), pm.group(1), w)
    return out


def _non_ansi_ports(path, module: str) -> dict[str, PortSpec]:
    """Ports of a non-ANSI header (e.g. the generated fabric primitive)."""
    from orchestrator.langgraph.integration_helpers import parse_verilog_ports
    try:
        vm = parse_verilog_ports(str(path), module)
    except Exception:  # noqa: BLE001 - reported as an unparsable header
        return {}
    return {p.name: PortSpec(p.name, p.direction, p.width) for p in vm.ports}


def _module_name(rtl_text: str) -> str | None:
    m = re.search(r"\bmodule\s+(\w+)", strip_preprocessor(rtl_text or "", defines=()))
    return m.group(1) if m else None


@dataclass
class Assembly:
    verilog: str
    module_name: str
    rtl_path: str
    instantiated: list[str]
    stubs: list[str]
    boundary_ports: list[dict]
    wiring_errors: list[str]
    wires: int
    sources: list[str]


def assemble_top(project_root, *, top_name: str, blocks: list[str], edges: list[dict],
                 rtl_paths: dict[str, str], out_dir, clk: str = "clk", rst: str = "rst_n",
                 boundary_block: str | None = None) -> Assembly:
    """Assemble ``top_name`` from real RTL (``rtl_paths``) and stubs for the rest.

    ``boundary_block``: a block whose module is the chip boundary (e.g. a
    locked pad adapter named like the top); its non-edge ports become the
    top's ports verbatim and its instance module is renamed ``<top>_core``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stub_dir = out / "stubs"
    stub_dir.mkdir(exist_ok=True)
    errors: list[str] = []
    sources: list[str] = []
    stubs: list[str] = []
    ports_by_block: dict[str, dict[str, PortSpec]] = {}
    inst_module: dict[str, str] = {}
    for b in blocks:
        cp = contract_ports(edges, b)
        path = rtl_paths.get(b)
        if path and Path(path).exists():
            text = Path(path).read_text(errors="replace")
            mod = _module_name(text) or b
            rp = _rtl_ports(text, mod) or _non_ansi_ports(path, mod)
            if not rp:
                errors.append(f"{b}: could not parse the module header of {path}")
                rp = {}
            # every contract port must exist in the real RTL
            for name in cp.edge_ports:
                if name not in rp:
                    errors.append(f"{b}: contract port {name!r} missing from {Path(path).name}")
            if mod == top_name:
                # the block is the chip boundary: rename its module for the instance
                renamed = out / f"{mod}_core.v"
                renamed.write_text(re.sub(rf"\bmodule\s+{re.escape(mod)}\b", f"module {mod}_core", text, count=1))
                sources.append(str(renamed))
                inst_module[b] = f"{mod}_core"
                boundary_block = boundary_block or b
            else:
                sources.append(str(path))
                inst_module[b] = mod
            ports_by_block[b] = rp
        else:
            sp = stub_dir / f"{b}.v"
            sp.write_text(stub_module(b, cp, clk=clk, rst=rst))
            sources.append(str(sp))
            stubs.append(b)
            inst_module[b] = b
            ports_by_block[b] = dict(cp.ports)
            ports_by_block[b].setdefault(clk, PortSpec(clk, "input", 1))
            ports_by_block[b].setdefault(rst, PortSpec(rst, "input", 1))

    # ---- nets from the contract edges
    net_of: dict[tuple[str, str], str] = {}
    wires: dict[str, int] = {}
    for e in edges:
        pb, cb = str(e.get("producer_block") or ""), str(e.get("consumer_block") or "")
        if pb not in ports_by_block or cb not in ports_by_block:
            continue
        pchan = channel_base(str(e.get("producer_port") or "")) or ""
        cchan = channel_base(str(e.get("consumer_port") or "")) or ""
        eid = re.sub(r"[^A-Za-z0-9_]", "_", str(e.get("edge_id") or f"{pb}__to__{cb}"))
        for s in signal_specs(e):
            pport, _ = canonical_port(pchan, s["name"])
            cport, _ = canonical_port(cchan, s["name"])
            pp = ports_by_block[pb].get(pport)
            cp_ = ports_by_block[cb].get(cport)
            if pp is None or cp_ is None:
                continue  # already reported as missing above (real RTL) -- stubs always have them
            if pp.dir == cp_.dir and pp.dir != "inout":
                errors.append(f"edge {eid}: {pb}.{pport} and {cb}.{cport} are both {pp.dir}")
                continue
            if pp.width != cp_.width:
                errors.append(f"edge {eid}: width mismatch {pb}.{pport}[{pp.width}] vs {cb}.{cport}[{cp_.width}]")
                continue
            # A producer output legitimately fans out to several consumers: the
            # first edge names the net, later edges join it. Two different
            # drivers on one consumer input is the real hazard.
            net = net_of.get((pb, pport)) or f"w_{eid}__{s['name']}"
            if pp.dir == "output" and (cb, cport) in net_of and net_of[(cb, cport)] != net:
                errors.append(f"{cb}.{cport} driven from two edges ({net_of[(cb, cport)]}, {net})")
                continue
            if cp_.dir == "output" and (pb, pport) in net_of and net_of[(pb, pport)] != net:
                errors.append(f"{pb}.{pport} driven from two edges ({net_of[(pb, pport)]}, {net})")
                continue
            net_of[(pb, pport)] = net
            net_of[(cb, cport)] = net
            wires[net] = pp.width

    # ---- boundary ports: unconnected block ports
    boundary: list[dict] = []
    for b in blocks:
        for name, p in ports_by_block[b].items():
            if (b, name) in net_of:
                continue
            if name in _CLK_NAMES:
                net_of[(b, name)] = clk
                continue
            if name in _RST_NAMES:
                net_of[(b, name)] = rst
                continue
            if b == boundary_block:
                top_port = name
            else:
                top_port = f"{b}_{name}"
            boundary.append({"name": top_port, "dir": p.dir, "width": p.width, "block": b, "port": name})
            net_of[(b, name)] = top_port

    # ---- emit
    L = [f"// {top_name}: deterministic shell assembly (A4). Real blocks: "
         f"{', '.join(x for x in blocks if x not in stubs) or '-'}; stubs: {', '.join(stubs) or '-'}.",
         "// Wired from the frozen interface contracts; regenerated as blocks pass. Do not edit.",
         f"module {top_name} ("]
    decl = [f"  input  wire {clk}", f"  input  wire {rst}"]
    seen = {clk, rst}
    for bp in boundary:
        if bp["name"] in seen:
            errors.append(f"boundary port {bp['name']!r} declared twice")
            continue
        seen.add(bp["name"])
        w = f"[{bp['width'] - 1}:0] " if bp["width"] > 1 else ""
        decl.append(f"  {bp['dir']:<6} wire {w}{bp['name']}")
    L.append(",\n".join(decl))
    L.append(");")
    for net, w in wires.items():
        L.append(f"  wire {'[' + str(w - 1) + ':0] ' if w > 1 else ''}{net};")
    for b in blocks:
        conns = [f".{name}({net_of[(b, name)]})" for name in ports_by_block[b]]
        L.append(f"  {inst_module[b]} u_{b} (")
        L.append("    " + ",\n    ".join(conns))
        L.append("  );")
    L.append("endmodule")
    verilog = "\n".join(L) + "\n"
    rtl_path = out / f"{top_name}.v"
    rtl_path.write_text(verilog)
    return Assembly(verilog=verilog, module_name=top_name, rtl_path=str(rtl_path),
                    instantiated=list(blocks), stubs=stubs, boundary_ports=boundary,
                    wiring_errors=errors, wires=len(wires), sources=sources)


def elaborate(assembly: Assembly, *, timeout_s: int = 600) -> dict:
    """Verilator lint of the assembled top with its sources, plus the engine
    primitive library the blocks were verified against (cs_sram_* etc.)."""
    import shutil
    import subprocess

    from orchestrator.langgraph.sram_wrapper import engine_lib_sources
    vb = shutil.which("verilator")
    if not vb:
        return {"ran": False, "ok": None, "reason": "verilator not installed"}
    cmd = [vb, "--lint-only", "-Wno-fatal", "-Wno-WIDTH", "-Wno-UNUSED", "-Wno-UNOPTFLAT",
           "-Wno-PINMISSING", "-Wno-DECLFILENAME", "--top-module", assembly.module_name,
           assembly.rtl_path, *assembly.sources, *engine_lib_sources(assembly.sources)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ran": False, "ok": None, "reason": str(exc)}
    errs = [ln for ln in (p.stdout + p.stderr).splitlines() if "%Error" in ln]
    return {"ran": True, "ok": p.returncode == 0 and not errs, "errors": errs[:20],
            "log": (p.stdout + p.stderr)[-4000:]}


_MODMISSING_RE = re.compile(r"Cannot find file containing module:\s*'?(\w+)'?")


def attribute_errors(errors: list[str], real_blocks: list[str]) -> tuple[list[tuple[str, str]], list[str]]:
    """Split assembly/elaboration errors into ``[(block, error)]`` the block
    must answer for, and engine/tool errors that no block caused.

    A missing module the engine itself supplies (``cs_sram_*``, ``cs_fabric_*``,
    ...) is an engine/tool problem -- the block passed its own gates against
    that library -- so it must not send the block back for another round.
    """
    from orchestrator.langgraph.sram_wrapper import is_engine_module
    blamed: list[tuple[str, str]] = []
    engine: list[str] = []
    for err in errors:
        m = _MODMISSING_RE.search(err)
        if m and is_engine_module(m.group(1)):
            engine.append(err)
            continue
        blk = next((b for b in real_blocks if err.startswith(f"{b}:") or f"u_{b}" in err or f"{b}." in err), None)
        if blk:
            blamed.append((blk, err))
    return blamed, engine


def write_snapshot(project_root, assembly: Assembly, elab: dict, *, tier=None) -> dict:
    """Record the assembly in the DB (``integration_snapshots``) and a view."""
    snap = {"ts": time.time(), "tier": tier, "top": assembly.module_name, "rtl_path": assembly.rtl_path,
            "real_blocks": [b for b in assembly.instantiated if b not in assembly.stubs],
            "stub_blocks": list(assembly.stubs), "wires": assembly.wires,
            "boundary_ports": len(assembly.boundary_ports), "wiring_errors": assembly.wiring_errors,
            "elaborated": elab.get("ok"), "elab_errors": elab.get("errors") or []}
    try:
        from orchestrator.state_store.project_db import open_project
        db = open_project(project_root)
        snap["id"] = db.add_integration_snapshot(snap)
    except Exception:  # noqa: BLE001 - the view still tells the story
        pass
    try:
        p = Path(project_root) / ".coresmith" / "integration_snapshot.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(snap, indent=2, default=str))
    except OSError:
        pass
    return snap
