# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The chip's physical pin assignment, as data rather than prose.

The shuttle fixes the package pinout: a design gets `io_in`/`io_out`/`io_oeb`
and must decide which bit carries which signal. That decision is a REQUIREMENT --
it comes from the host protocol the part has to speak -- so it belongs in the
PRD, and it belongs there structured.

Today it is a sentence:

    "GPIO mapping is locked: io[0]=qspi_csn input, io[1]=qspi_sck input,
     io[5:2]=bidirectional qspi_io[3:0], and io[6]=irq output. io_oeb[0]=
     io_oeb[1]=1, io_oeb[6]=0, and io_oeb[5:2]=0 only during QSPI read-data
     drive phases; otherwise io_oeb[5:2]=1..."

Everything downstream then re-derives that by reading it. The architecture phase
invented a whole pin-adapter BLOCK to hold the translation, an LLM generated that
block's RTL, and because the shuttle also mandates the module name
`user_project_wrapper` -- which conventionally means "the entire chip" -- the
generator produced a competing chip top instead of an adapter. Four regenerations
with explicit corrective feedback produced the identical result each time.

None of that is a modelling problem. A pin map is a permutation of bit ranges.
Given it as data, the integration stage emits the routing directly and there is
no adapter block to get wrong.

Schema (``prd["pin_map"]``)::

    {
      "bus_width": 38,
      "entries": [
        {"signal": "qspi_csn",    "dir": "in",  "msb": 0, "lsb": 0},
        {"signal": "qspi_io_in",  "dir": "in",  "msb": 5, "lsb": 2},
        {"signal": "qspi_io_out", "dir": "out", "msb": 5, "lsb": 2,
         "oe": "qspi_drive_en"},
        {"signal": "irq_level",   "dir": "out", "msb": 6, "lsb": 6}
      ]
    }

``dir`` is from the CHIP's point of view: ``in`` reads ``io_in``, ``out`` drives
``io_out``. ``oe`` names an active-high enable; the emitter inverts it, because
``io_oeb`` is active-low. An ``out`` entry with no ``oe`` drives permanently.
Unmapped bits are tied off, so every bit of every bus has exactly one driver.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

IN_BUS, OUT_BUS, OE_BUS = "io_in", "io_out", "io_oeb"


@dataclass
class PinEntry:
    signal: str
    dir: str                 # "in" | "out"
    msb: int
    lsb: int
    oe: str = ""             # active-high enable for an "out" entry

    @property
    def width(self) -> int:
        return abs(self.msb - self.lsb) + 1

    @property
    def bits(self) -> range:
        lo, hi = sorted((self.lsb, self.msb))
        return range(lo, hi + 1)


@dataclass
class PinMap:
    bus_width: int = 38
    entries: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and bool(self.entries)


def load_pin_map(project_root) -> PinMap | None:
    """Read ``prd["pin_map"]``. None when the PRD declares none.

    Deliberately reads ONLY the structured field. Parsing the prose form would
    put a guess back on the production path, which is the thing this replaces.
    """
    p = Path(project_root) / ".coresmith" / "prd_spec.json"
    try:
        doc = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    prd = doc.get("prd") if isinstance(doc, dict) else None
    raw = (prd or {}).get("pin_map") if isinstance(prd, dict) else None
    if not isinstance(raw, dict) or not raw.get("entries"):
        return None
    return parse_pin_map(raw)


def parse_pin_map(raw: dict) -> PinMap:
    """Validate a pin-map document. Errors are returned, never raised."""
    pm = PinMap(bus_width=int(raw.get("bus_width") or 38))
    for i, e in enumerate(raw.get("entries") or []):
        if not isinstance(e, dict):
            pm.errors.append(f"entry {i} is not an object")
            continue
        sig, d = str(e.get("signal") or ""), str(e.get("dir") or "")
        if not sig:
            pm.errors.append(f"entry {i} has no signal name")
            continue
        if d not in ("in", "out"):
            pm.errors.append(f"{sig}: dir must be 'in' or 'out', got {d!r}")
            continue
        try:
            msb, lsb = int(e["msb"]), int(e["lsb"])
        except (KeyError, TypeError, ValueError):
            pm.errors.append(f"{sig}: msb/lsb must be integers")
            continue
        if min(msb, lsb) < 0 or max(msb, lsb) >= pm.bus_width:
            pm.errors.append(
                f"{sig}: bits [{msb}:{lsb}] fall outside the {pm.bus_width}-bit bus")
            continue
        oe = str(e.get("oe") or "")
        if oe and d != "out":
            pm.errors.append(f"{sig}: 'oe' is only meaningful on an out entry")
            continue
        pm.entries.append(PinEntry(sig, d, msb, lsb, oe))

    # A bit driven twice is a short; a signal named twice is a typo. Both are
    # refused rather than resolved, because either guess produces a wrong chip.
    for bus, sel in (("input", "in"), ("output", "out")):
        seen: dict = {}
        for ent in [x for x in pm.entries if x.dir == sel]:
            for b in ent.bits:
                if b in seen:
                    pm.errors.append(
                        f"{bus} bit {b} is claimed by both '{seen[b]}' and "
                        f"'{ent.signal}'")
                seen[b] = ent.signal
    names = [e.signal for e in pm.entries]
    for n in sorted(set(names)):
        if names.count(n) > 1:
            pm.errors.append(f"signal '{n}' is mapped {names.count(n)} times")
    return pm


def _ranges(bits: set, width: int) -> list:
    """Contiguous [msb, lsb] ranges covering `bits`, high to low."""
    out, run = [], []
    for b in range(width):
        if b in bits:
            run.append(b)
        elif run:
            out.append((run[-1], run[0]))
            run = []
    if run:
        out.append((run[-1], run[0]))
    return list(reversed(out))


def emit_pin_routing(pm: PinMap) -> tuple:
    """Verilog that connects the pad buses to named signals.

    Returns ``(declarations, assignments)``. Declarations are the internal wires
    a chip top should declare for INPUT-side signals; output-side signals are
    driven by blocks and only read here.

    Every bit of io_out and io_oeb ends up with exactly one driver: mapped bits
    from their signal, everything else tied to the safe default (output 0,
    output-enable de-asserted). A floating pad bit is a real defect and tying
    them here makes it impossible.
    """
    decls, assigns = [], []
    w = pm.bus_width

    for e in [x for x in pm.entries if x.dir == "in"]:
        rng = "" if e.width == 1 else f"[{e.width - 1}:0] "
        sel = f"[{e.msb}]" if e.width == 1 else f"[{max(e.msb, e.lsb)}:{min(e.msb, e.lsb)}]"
        decls.append(f"wire {rng}{e.signal} = {IN_BUS}{sel};")

    driven_out, driven_oe = set(), set()
    for e in [x for x in pm.entries if x.dir == "out"]:
        sel = (f"[{e.msb}]" if e.width == 1
               else f"[{max(e.msb, e.lsb)}:{min(e.msb, e.lsb)}]")
        assigns.append(f"assign {OUT_BUS}{sel} = {e.signal};")
        driven_out |= set(e.bits)
        if e.oe:
            # io_oeb is ACTIVE LOW: drive the pad when the enable is high.
            rep = "" if e.width == 1 else f"{{{e.width}{{"
            tail = "" if e.width == 1 else "}}"
            assigns.append(f"assign {OE_BUS}{sel} = {rep}~{e.oe}{tail};")
        else:
            assigns.append(
                f"assign {OE_BUS}{sel} = {e.width}'b" + "0" * e.width + ";")
        driven_oe |= set(e.bits)

    for msb, lsb in _ranges(set(range(w)) - driven_out, w):
        n = msb - lsb + 1
        sel = f"[{msb}]" if n == 1 else f"[{msb}:{lsb}]"
        assigns.append(f"assign {OUT_BUS}{sel} = {n}'b" + "0" * n + ";")
    for msb, lsb in _ranges(set(range(w)) - driven_oe, w):
        n = msb - lsb + 1
        sel = f"[{msb}]" if n == 1 else f"[{msb}:{lsb}]"
        # Default is 1: do NOT drive a pad nobody asked for.
        assigns.append(f"assign {OE_BUS}{sel} = {n}'b" + "1" * n + ";")

    return decls, assigns


def mapped_signals(pm: PinMap) -> dict:
    """signal name -> width, for wiring the routing to block ports."""
    return {e.signal: e.width for e in pm.entries}
