# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Bind memory-wrapper instances in a mapped netlist to concrete macros so
pre-layout STA times the paths through them.

A block netlist keeps ``cs_sram_1rw1r #(.WIDTH(32), .DEPTH(512)) u_x (...)``
as a hierarchical placeholder: correct for synthesis (a macro, 0 flops), but
OpenSTA has no definition for it, prints "Creating black box" and gives the
pins no timing arcs -- so address setup and clk->rdata into the hit logic,
the critical paths of any cache, were simply not timed and the gate recorded
0.0. This module resolves each (kind, width, depth) to the macro the backend
would bind (``macro_registry``), emits one structural wrapper module per
geometry that instantiates the macro, renames the instances to it, and
returns the macro liberty files to link. Geometries with no macro are
reported as unresolved: timing is then *unmeasured*, never a pass.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

_WRAPPERS = {"cs_sram_1rw1r": 2, "cs_sram_1rw": 1}
_INST_RE = re.compile(r"\b(cs_sram_1rw1r|cs_sram_1rw)\s*#\s*\(", re.S)
_PARAM_RE = re.compile(r"\.\s*(WIDTH|DEPTH)\s*\(\s*([^)]*?)\s*\)")


def enabled() -> bool:
    return (os.environ.get("CORESMITH_STA_BIND_MACROS", "1") or "1").strip().lower() not in {"0", "false", "no", "off"}


def _int_literal(v: str) -> int | None:
    v = v.strip().replace("_", "")
    m = re.match(r"^(?:\d+)?'s?[dD](\d+)$", v)
    if m:
        return int(m.group(1))
    m = re.match(r"^(?:\d+)?'s?[hH]([0-9a-fA-F]+)$", v)
    if m:
        return int(m.group(1), 16)
    m = re.match(r"^(\d+)$", v)
    return int(m.group(1)) if m else None


def _balanced_close(s: str, open_at: int) -> int:
    depth = 0
    for i in range(open_at, len(s)):
        if s[i] == "(":
            depth += 1
        elif s[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def wrapper_name(kind: str, width: int, depth: int) -> str:
    return f"{kind}__w{width}_d{depth}"


@dataclass
class MacroBinding:
    netlist: str
    wrappers: str = ""
    libs: list = field(default_factory=list)
    lefs: list = field(default_factory=list)
    bound: list = field(default_factory=list)       # (kind, width, depth, macro_name)
    unresolved: list = field(default_factory=list)  # (kind, width, depth)
    instances: int = 0

    @property
    def ok(self) -> bool:
        return not self.unresolved


def find_instances(netlist: str) -> list[tuple[int, int, str, int, int]]:
    """``(start, end, kind, width, depth)`` for every parameterised wrapper
    instance; ``end`` is the index just past the ``#( ... )`` block."""
    out = []
    for m in _INST_RE.finditer(netlist):
        open_at = m.end() - 1
        close = _balanced_close(netlist, open_at)
        if close < 0:
            continue
        params = netlist[open_at + 1:close]
        w = d = None
        for pm in _PARAM_RE.finditer(params):
            v = _int_literal(pm.group(2))
            if pm.group(1) == "WIDTH":
                w = v
            else:
                d = v
        if w is None or d is None:
            continue
        out.append((m.start(), close + 1, m.group(1), w, d))
    return out


def _wrapper_module(kind: str, width: int, depth: int, macro) -> str:
    """A structural wrapper (no expressions: OpenSTA's reader is netlist-only)
    exposing the cs_sram wrapper ports and instantiating the macro. Pin
    polarity is irrelevant to timing arcs; only pins the macro declares are
    connected."""
    from orchestrator.langgraph.macro_prebind import macro_ports
    aw = max(1, (depth - 1).bit_length())
    nb = (width + 7) // 8
    have = macro_ports(getattr(macro, "verilog", "") or "")
    nport = _WRAPPERS[kind]
    if nport == 2:
        ports = ["input clk", "input ce0", "input we0", f"input [{aw - 1}:0] addr0", f"input [{width - 1}:0] wdata0",
                 f"input [{nb - 1}:0] wmask0", f"output [{width - 1}:0] rdata0", "input ce1",
                 f"input [{aw - 1}:0] addr1", f"output [{width - 1}:0] rdata1"]
        conns = [("clk0", "clk"), ("csb0", "ce0"), ("web0", "we0"), ("addr0", "addr0"), ("din0", "wdata0"),
                 ("dout0", "rdata0"), ("clk1", "clk"), ("csb1", "ce1"), ("addr1", "addr1"), ("dout1", "rdata1")]
    else:
        ports = ["input clk", "input ce", "input we", f"input [{aw - 1}:0] addr", f"input [{width - 1}:0] wdata",
                 f"output [{width - 1}:0] rdata"]
        conns = [("clk0", "clk"), ("csb0", "ce"), ("web0", "we"), ("addr0", "addr"), ("din0", "wdata"), ("dout0", "rdata")]
    if have:
        conns = [(p, s) for p, s in conns if p in have]
    lines = [f"module {wrapper_name(kind, width, depth)} (", "  " + ",\n  ".join(ports), ");",
             f"  {macro.name} u_macro (", "    " + ",\n    ".join(f".{p}({s})" for p, s in conns), "  );", "endmodule", ""]
    return "\n".join(lines)


def bind_netlist_macros(netlist: str, *, registry=None, allow_generate: bool = False) -> MacroBinding:
    """Rewrite wrapper instances to per-geometry wrapper modules bound to
    concrete macros. Never raises; an unavailable registry yields every
    geometry unresolved."""
    inst = find_instances(netlist)
    res = MacroBinding(netlist=netlist, instances=len(inst))
    if not inst:
        return res
    try:
        from orchestrator.langgraph.macro_registry import ShellSpec, discover_macros, resolve_shell
        if registry is None:
            registry = discover_macros()
    except Exception as exc:  # noqa: BLE001
        res.unresolved = sorted({(k, w, d) for _s, _e, k, w, d in inst})
        res.wrappers = f"// macro registry unavailable: {exc}\n"
        return res
    macros: dict[tuple, object] = {}
    for _s, _e, kind, w, d in inst:
        key = (kind, w, d)
        if key in macros:
            continue
        spec = ShellSpec(kind="sram", width=w, depth=d, nport=_WRAPPERS[kind])
        try:
            m = resolve_shell(spec, registry=registry, allow_generate=allow_generate)
        except Exception:  # noqa: BLE001
            m = None
        if m is None or not getattr(m, "name", "") or not getattr(m, "lib", ""):
            res.unresolved.append(key)
            macros[key] = None
            continue
        macros[key] = m
        res.bound.append((kind, w, d, m.name))
        if m.lib not in res.libs:
            res.libs.append(m.lib)
        lef = getattr(m, "lef", "") or ""
        if lef and lef not in res.lefs:
            res.lefs.append(lef)
    res.unresolved = sorted(set(res.unresolved))
    # rewrite from the end so offsets stay valid
    text = netlist
    for start, end, kind, w, d in sorted(inst, key=lambda t: t[0], reverse=True):
        if macros.get((kind, w, d)) is None:
            continue
        rest = text[end:].lstrip()
        text = text[:start] + wrapper_name(kind, w, d) + " " + rest
    res.netlist = text
    seen = set()
    parts = ["// GENERATED (macro_sta): per-geometry wrappers binding cs_sram instances to macros for STA."]
    for kind, w, d, _name in res.bound:
        if (kind, w, d) in seen:
            continue
        seen.add((kind, w, d))
        parts.append(_wrapper_module(kind, w, d, macros[(kind, w, d)]))
    res.wrappers = "\n".join(parts) + "\n"
    return res


def describe_unresolved(res: MacroBinding) -> str:
    return ", ".join(f"{k} {w}x{d}" for k, w, d in res.unresolved)
