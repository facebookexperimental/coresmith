# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Backend collateral injection for SRAM macros (5th fix).

Turns the macros a netlist instantiates (from `macro_registry`) into the
concrete tool inputs each backend step needs:

* `plan_floorplan()` -- size a die that fits the macros + std cells and lay the
  macros out in a row with halos; returns die/core boxes + per-instance
  placement (inst name, x, y, orient).
* `synth_injection()` -- yosys lines to read each macro's verilog as a blackbox
  + its liberty, and a `hilomap` tie-cell line (constants -> conb_1, else
  TritonRoute rejects `one_`/`zero_` nets, DRT-0305).
* `pnr_header_vars()` -- tcl `set` lines (macro LEF/lib lists, placement list,
  explicit die/core) consumed by the guarded macro blocks in pnr_reference.tcl.
* `drc_macro_block()` -- magic lines using the OpenLane idiom
  `property LEFview true` + `property GDS_FILE`: the macro is a black box for
  DRC + extraction (no 1.1M false internal violations, clean LVS black box) but
  its real GDS still streams into the merged output.

All functions are pure/string-producing -- no EDA tools, safe to unit-test.
"""
from __future__ import annotations

import math
import re

from orchestrator.langgraph.macro_registry import MacroInfo

# A std-cell-area allowance used only when the real post-synth area is unknown,
# so the auto-sized die still leaves room around the macros.
_DEFAULT_STD_AREA_UM2 = 150_000.0

# sky130 HD site (unithd): cells sit on a 0.46um x 2.72um grid. OpenROAD snaps
# the core box to this; a macro must be placed on it too (and at/after the
# snapped core corner) or place_macro errors MPL-0034 / leaves a thin channel.
_SITE_W = 0.46
_SITE_H = 2.72


# ---------------------------------------------------------------------------
# LEF database-unit (DBU) normalization for PnR macro read
# ---------------------------------------------------------------------------
# OpenROAD refuses any macro LEF whose `DATABASE MICRONS` precision EXCEEDS the
# tech LEF's: it warns ODB-0205 then ERRORs ODB-0292 "LEF data ... is
# discarded", the macro never enters the DB ("Found 0 macro blocks",
# MacroInstsArea 0), and the memory is physically absent from the layout.
# OpenRAM-generated macro LEFs ship at `DATABASE MICRONS 2000` while the sky130
# HD tech LEF is `1000` (efabless-prebuilt macro LEFs are already 1000).
#
# The fix rewrites ONLY the `DATABASE MICRONS` header line to the tech DBU.
# CRITICAL: in LEF, all geometry (SIZE, ORIGIN, FOREIGN, RECT, PORT, OBS, ...)
# is expressed in MICRONS -- `DATABASE MICRONS` only sets the grid precision --
# so NO coordinate value is scaled (they are byte-identical after the rewrite).
# Down-conversion is lossless ONLY if every micron coordinate is an integer
# multiple of 1/target_dbu um; if ANY value would round, we FAIL HARD with the
# offending value rather than silently corrupt the macro geometry.

# `UNITS ... DATABASE MICRONS <integer> ;` -- capture the integer to rewrite it.
_DB_MICRONS_RE = re.compile(r"(DATABASE\s+MICRONS\s+)(\d+)", re.IGNORECASE)
# Every fractional micron coordinate in the file (real LEF writers always emit a
# leading digit, e.g. `0.005`, `1.380`, `-0.5`). Pure integers are trivially
# representable at any DBU, so they need no check.
_LEF_FLOAT_RE = re.compile(r"[-+]?\d+\.\d+")


class LefDbuError(ValueError):
    """A macro LEF cannot be down-converted to the tech DBU without rounding a
    coordinate (would corrupt geometry). Raised instead of silently rewriting.
    """


def lef_database_units(text: str) -> int | None:
    """Return the LEF's `DATABASE MICRONS <N>` value, or None if it has no
    UNITS/DATABASE MICRONS header."""
    m = _DB_MICRONS_RE.search(text)
    return int(m.group(2)) if m else None


def normalize_lef_dbu(text: str, target_dbu: int) -> tuple[str, bool]:
    """Rewrite ONLY the `DATABASE MICRONS` header to ``target_dbu`` when the
    LEF's DBU exceeds it (so OpenROAD does not discard the macro, ODB-0292).

    Returns ``(new_text, changed)``. Leaves the text untouched (``changed ==
    False``) when the LEF has no DATABASE header or its DBU is already <=
    ``target_dbu`` (e.g. an already-1000 prebuilt macro LEF -- normalize
    per-LEF, never assume 2000).

    Down-conversion rewrites ONLY the header integer; no coordinate is scaled.
    Raises :class:`LefDbuError` (with the offending value) if any fractional
    micron coordinate is not representable at ``target_dbu`` -- refusing to
    silently corrupt geometry.
    """
    src_dbu = lef_database_units(text)
    if src_dbu is None or src_dbu <= int(target_dbu):
        return text, False
    target = int(target_dbu)
    # Losslessness gate: every fractional micron coordinate must land exactly on
    # the 1/target_dbu grid. Scan the whole file EXCEPT the DATABASE MICRONS line
    # (that integer is the DBU being rewritten, not a coordinate).
    for line in text.splitlines():
        if _DB_MICRONS_RE.search(line):
            continue
        for fm in _LEF_FLOAT_RE.finditer(line):
            scaled = float(fm.group(0)) * target
            if abs(scaled - round(scaled)) > 1e-6:
                raise LefDbuError(
                    f"LEF coordinate {fm.group(0)} um is not representable at "
                    f"{target} DBU (grid 1/{target} um); refusing to "
                    f"down-convert from {src_dbu} DBU -- rewriting the header "
                    f"would corrupt the macro geometry."
                )
    new_text = _DB_MICRONS_RE.sub(rf"\g<1>{target}", text, count=1)
    return new_text, True


# ---------------------------------------------------------------------------
# Netlist module-hierarchy parsing: real top detection + flattened leaf counts
# ---------------------------------------------------------------------------
# A gutted structural netlist (yosys `write_verilog -noattr`) with macros emits
# MANY modules: `$paramod...` parameterized wrappers, macro/shell leaves, and
# one real integration top -- and yosys emits the leaves FIRST, the top LAST.
# The old "first `module (\w+)`" scan therefore picked a SUB-BLOCK as the PnR
# top (e.g. a small memory sub-block, a 154-cell fragment, not the 4,682-cell
# chip), routing + "signing off" a fragment mislabeled as the chip. These pure
# helpers replace that scan with real top detection (defined-but-not-
# instantiated) and resolve the FLATTENED macro leaf count (a macro inside a
# sub-block that is itself instantiated K times resolves to K leaves once
# OpenROAD flattens).

# A Verilog identifier: an escaped id (`\...`, ends at whitespace -- covers the
# yosys `\$paramod$..\name` form) OR a plain id.
_VLOG_ID = r"(?:\\[^\s]+|[A-Za-z_][\w$]*)"
_MODULE_BLOCK_RE = re.compile(
    rf"\bmodule\b\s+({_VLOG_ID})(.*?)\bendmodule\b", re.DOTALL
)
_INSTANCE_RE = re.compile(
    rf"^\s*({_VLOG_ID})\s+{_VLOG_ID}\s*(?:#\s*\(.*?\)\s*)?\(", re.DOTALL
)
# Verilog keywords that begin a statement but are NOT module instantiations.
_VLOG_STMT_KEYWORDS = frozenset({
    "input", "output", "inout", "wire", "reg", "logic", "assign", "parameter",
    "localparam", "supply0", "supply1", "wand", "wor", "tri", "tri0", "tri1",
    "trireg", "generate", "endgenerate", "begin", "end", "always", "always_ff",
    "always_comb", "always_latch", "initial", "function", "endfunction", "task",
    "endtask", "specify", "endspecify", "genvar", "integer", "real", "realtime",
    "time", "defparam", "if", "else", "case", "casex", "casez", "endcase",
    "for", "while", "repeat", "forever", "signed", "unsigned", "event",
    "module", "endmodule", "primitive", "endprimitive",
})


def _strip_vlog_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", " ", text)
    return text


def parse_module_graph(netlist_text: str) -> tuple[list[str], dict[str, list[str]]]:
    """Parse a structural netlist into its module-instantiation graph.

    Returns ``(defined, children)`` where ``defined`` is the list of module
    names DEFINED in the netlist (in file order) and ``children`` maps each
    defined module to the (multiplicity-preserving) list of module/cell TYPE
    names it instantiates in its body. Purely textual -- no EDA tools; robust to
    yosys escaped identifiers and `$paramod` wrappers.
    """
    text = _strip_vlog_comments(netlist_text or "")
    defined: list[str] = []
    children: dict[str, list[str]] = {}
    for mm in _MODULE_BLOCK_RE.finditer(text):
        name = mm.group(1)
        rest = mm.group(2)
        defined.append(name)
        # Drop the header (port list) up to its terminating ';' so the port list
        # is never mistaken for an instantiation.
        semi = rest.find(";")
        body = rest[semi + 1:] if semi >= 0 else rest
        kids: list[str] = []
        # Each statement ends at ';'; an instantiation is `TYPE INST ( ... )`.
        for stmt in body.split(";"):
            im = _INSTANCE_RE.match(stmt)
            if not im:
                continue
            typ = im.group(1)
            if typ in _VLOG_STMT_KEYWORDS:
                continue
            kids.append(typ)
        children[name] = kids
    return defined, children


def _normalize_module_name(s: str) -> str:
    """Normalize for slug-tolerant matching: strip a leading `prd_`/`prd___`
    prefix, collapse repeated underscores, lowercase (mirrors the PRD-title slug
    mangling commit 263d821 targeted)."""
    s = s.lstrip("\\")
    s = re.sub(r"^prd_+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"_+", "_", s).strip("_")
    return s.lower()


def detect_top_module(netlist_text: str, preferred: str) -> str:
    """Return the real top module of ``netlist_text``, never a sub-block.

    The top is a module that is DEFINED but NOT INSTANTIATED by any other
    module. Preference order:
      (a) ``preferred`` names a defined, uninstantiated module -> use it (the
          common case: the passed design_name already IS the netlist top);
      (b) exactly one module is uninstantiated -> it is the top (covers the
          PRD-slug mangling case where preferred != the netlist's real name);
      (c) several uninstantiated -> the one best matching ``preferred``
          (normalized), else the largest by instantiation count;
      (d) fall back to ``preferred`` unchanged (empty/unparseable netlist).
    NEVER returns a module that is instantiated by another module.
    """
    defined, children = parse_module_graph(netlist_text)
    if not defined:
        return preferred
    defined_set = set(defined)
    instantiated: set[str] = set()
    for kids in children.values():
        instantiated.update(kids)
    tops = [m for m in defined if m not in instantiated]
    # (a) preferred is itself a valid (uninstantiated) top.
    if preferred in defined_set and preferred not in instantiated:
        return preferred
    # (b) exactly one uninstantiated module = the real top.
    if len(tops) == 1:
        return tops[0]
    if len(tops) > 1:
        # (c) best name match to preferred, else most-connected.
        npref = _normalize_module_name(preferred or "")
        if npref:
            def _score(m: str) -> tuple[int, int]:
                nm = _normalize_module_name(m)
                if nm == npref:
                    s = 3
                elif npref in nm or nm in npref:
                    s = 2
                else:
                    s = 0
                return (s, len(children.get(m, [])))
            best = max(tops, key=_score)
            if _score(best)[0] > 0:
                return best
        return max(tops, key=lambda m: len(children.get(m, [])))
    # (d) no uninstantiated module (unparseable / cyclic) -> preferred as-is.
    return preferred


def flattened_type_counts(netlist_text: str, top: str) -> dict[str, int]:
    """Return ``{type_name: flattened_instance_count}`` under ``top``.

    Elaborates the module hierarchy the way OpenROAD flattens it: a type
    instantiated inside a sub-module that is itself instantiated K times
    resolves to K leaves. Counts every instantiated TYPE (defined sub-modules
    AND leaf cells/macros), so a macro's flattened count is robust to N greater
    than the shell-module text-instantiation count.
    """
    from collections import Counter, defaultdict

    defined, children = parse_module_graph(netlist_text)
    defined_set = set(defined)
    if top not in defined_set:
        return {}
    child_ctr = {p: Counter(ch) for p, ch in children.items()}

    # Topological order (parents before children) via reverse post-order DFS.
    order: list[str] = []
    visited: set[str] = set()

    def _dfs(n: str) -> None:
        if n in visited or n not in defined_set:
            return
        visited.add(n)
        for c in children.get(n, []):
            if c in defined_set:
                _dfs(c)
        order.append(n)

    _dfs(top)
    order.reverse()

    # Elaboration multiplicity of each DEFINED module.
    mult: dict[str, int] = defaultdict(int)
    mult[top] = 1
    for parent in order:
        base = mult.get(parent, 0)
        if not base:
            continue
        for child, k in child_ctr.get(parent, {}).items():
            if child in defined_set:
                mult[child] += base * k

    # Flattened count of every type = sum over defined parents of mult * count.
    flat: dict[str, int] = defaultdict(int)
    for parent in defined_set:
        base = mult.get(parent, 0)
        if not base:
            continue
        for child, k in child_ctr.get(parent, {}).items():
            flat[child] += base * k
    return dict(flat)


def expand_placements_to_flattened(
    netlist_text: str,
    preferred_top: str,
    placed: list[tuple[MacroInfo, str]],
) -> list[tuple[MacroInfo, str]]:
    """Re-size a ``[(MacroInfo, inst), ...]`` list (one entry per TEXT
    instantiation) to the FLATTENED leaf count per macro master, so the
    floorplan + placement plan cover every leaf OpenROAD resolves (not just the
    shell-module text count). Placement positions are what matter downstream
    (the template resolves real instance names via ``get_cells``), so extra
    entries get synthetic instance names. Falls back to the input on any parse
    failure or when the flattened count is not larger."""
    if not placed:
        return placed
    try:
        top = detect_top_module(netlist_text, preferred_top)
        flat = flattened_type_counts(netlist_text, top)
    except Exception:
        return placed
    if not flat:
        return placed
    from collections import OrderedDict

    grouped: OrderedDict[str, tuple[MacroInfo, list[str]]] = OrderedDict()
    for mi, inst in placed:
        entry = grouped.get(mi.name)
        if entry is None:
            grouped[mi.name] = (mi, [inst])
        else:
            entry[1].append(inst)

    out: list[tuple[MacroInfo, str]] = []
    for name, (mi, insts) in grouped.items():
        k = max(int(flat.get(name, 0)), len(insts))
        stem = re.sub(r"[^\w]", "_", name).strip("_") or "macro"
        for i in range(k):
            inst = insts[i] if i < len(insts) else f"{stem}_{i}"
            out.append((mi, inst))
    return out


def _snap_up(v: float, grid: float) -> float:
    import math as _m
    return round(_m.ceil(v / grid) * grid, 3)


def extract_macro_instances(
    netlist_path: str, macros: list[MacroInfo]
) -> list[tuple[MacroInfo, str]]:
    """Return [(macro, instance_name), ...] for each macro instantiation found.

    Handles yosys escaped identifiers (``\\u_foo.u_bar``) and plain names. The
    instance name is returned WITHOUT the leading backslash / trailing space so
    it can be passed straight to OpenROAD `place_macro -macro_name`.
    """
    try:
        with open(netlist_path, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return []
    out: list[tuple[MacroInfo, str]] = []
    for m in macros:
        # <macro_name> <inst> (   where <inst> may be \escaped\ (ends at ws) or plain
        for im in re.finditer(
            rf"(?<![\w]){re.escape(m.name)}\s+(\\?\S+)\s*\(", text
        ):
            inst = im.group(1).lstrip("\\")
            out.append((m, inst))
    return out


def plan_floorplan(
    placed_macros: list[tuple[MacroInfo, str]],
    std_cell_area_um2: float | None = None,
    target_util: float = 0.4,
    margin_um: float = 30.0,
    halo_um: float = 10.0,
    spacing_um: float = 20.0,
):
    """Compute (die_box, core_box, placements) for a macro-bearing design.

    die_box/core_box are "llx lly urx ury" strings (OpenROAD -die_area form).
    placements is [(inst, x, y, orient, macro), ...].

    The die is sized to hold the macro area + a std-cell allowance at
    `target_util`, but never smaller than what the macro row physically needs.
    Macros are tiled left->right in rows along the bottom of the core, each with
    a halo so the router has channels and the PDN can ring them.
    """
    if std_cell_area_um2 is None:
        std_cell_area_um2 = _DEFAULT_STD_AREA_UM2
    macro_area = sum(m.area_um2 for m, _ in placed_macros)
    # Reserve halo ring area around macros too.
    halo_area = sum(
        (m.width_um + 2 * halo_um) * (m.height_um + 2 * halo_um) - m.area_um2
        for m, _ in placed_macros
    )
    needed_area = (macro_area + halo_area + std_cell_area_um2) / max(target_util, 0.1)
    side = math.sqrt(needed_area)

    # Must at least fit the widest macro + margins, and the tallest.
    widest = max((m.width_um for m, _ in placed_macros), default=0.0)
    tallest = max((m.height_um for m, _ in placed_macros), default=0.0)
    min_side = max(widest, tallest) + 2 * (margin_um + halo_um) + spacing_um
    side = max(side, min_side, 120.0)
    side = math.ceil(side / 10.0) * 10.0  # round up to 10um

    core_llx = core_lly = margin_um
    core_urx = core_ury = side - margin_um

    # Place macros FLUSH to the SNAPPED core corner, site-aligned. Flush (0
    # channel to the die edge) avoids PDN-0179 "unable to repair channels";
    # snapping up to the site grid keeps the origin inside the snapped core
    # (else MPL-0034) and aligns macro pins to tracks. Halo/spacing applies
    # only BETWEEN macros.
    corner_x = _snap_up(core_llx, _SITE_W)
    corner_y = _snap_up(core_lly, _SITE_H)
    placements = []
    x = corner_x
    y = corner_y
    row_h = 0.0
    for m, inst in placed_macros:
        if x + m.width_um > core_urx and x > corner_x:
            x = corner_x
            y = _snap_up(y + row_h + spacing_um + 2 * halo_um, _SITE_H)
            row_h = 0.0
        placements.append((inst, _snap_up(x, _SITE_W), _snap_up(y, _SITE_H), "R0", m))
        x = _snap_up(x + m.width_um + spacing_um + 2 * halo_um, _SITE_W)
        row_h = max(row_h, m.height_um)

    die_box = f"0 0 {side} {side}"
    core_box = f"{core_llx} {core_lly} {core_urx} {core_ury}"
    return die_box, core_box, placements


def synth_injection(macros: list[MacroInfo]) -> tuple[str, str, str]:
    """Return (blackbox_reads, liberty_reads, hilomap_line) for yosys.

    blackbox_reads: `read_verilog -lib <macro.v>` so `hierarchy -check` resolves
    the instance without trying to synthesize the macro.
    liberty_reads: `read_liberty -lib <macro.lib>` so `stat`/abc see it.
    hilomap_line: map constants to conb_1 tie cells (empty if no macros).
    """
    if not macros:
        return "", "", ""
    bb = "\n".join(
        f"read_verilog -lib -sv {m.verilog}" for m in macros if m.verilog
    )
    libs = "\n".join(
        f"read_liberty -lib {m.lib}" for m in macros if m.lib
    )
    hilomap = (
        "hilomap -hicell sky130_fd_sc_hd__conb_1 HI "
        "-locell sky130_fd_sc_hd__conb_1 LO"
    )
    return bb, libs, hilomap


def pnr_header_vars(
    macros: list[MacroInfo],
    die_box: str,
    core_box: str,
    placements: list,
) -> str:
    """Tcl `set` lines for the macro-aware blocks in pnr_reference.tcl.

    Defines: macro_lefs, macro_libs, macro_names, macro_place, macro_pg
    (power/ground pin tuples), macro_die_area, macro_core_area. The template
    guards every macro block on `[info exists ...]`, so emitting these turns the
    macro flow on; omitting them leaves the std-cell-only flow unchanged.
    """
    if not placements:
        return ""
    lefs = " ".join(f'"{m.lef}"' for m in macros if m.lef)
    libs = " ".join(f'"{m.lib}"' for m in macros if m.lib)
    names = " ".join(m.name for m in macros)
    # Positions only (no instance name): the template resolves real instances by
    # their master via get_cells, so this is robust to hierarchical flattening.
    place_items = " ".join(
        f'{{{x} {y} {orient}}}' for _inst, x, y, orient, _m in placements
    )
    # unique {power_pin ground_pin} pairs; connected by pin pattern (only macros
    # carry these pins, so inst_pattern .* is safe).
    pg_pairs = {(m.power_pin, m.ground_pin) for m in macros}
    pg_items = " ".join(f"{{{p} {g}}}" for p, g in sorted(pg_pairs))
    return (
        f'set macro_lefs [list {lefs}]\n'
        f'set macro_libs [list {libs}]\n'
        f'set macro_names [list {names}]\n'
        f'set macro_place [list {place_items}]\n'
        f'set macro_pg [list {pg_items}]\n'
        f'set macro_die_area "{die_box}"\n'
        f'set macro_core_area "{core_box}"\n'
    )


def drc_macro_block(placed_macros: list[tuple[MacroInfo, str]]) -> str:
    """Magic lines to treat each macro as a signed-off black box.

    Uses `property LEFview true` (skip internal DRC/extraction -> no false
    1.1M violations, clean LVS black box) + `property GDS_FILE` (stream the
    macro's real GDS into the merged output). Emitted BEFORE `def read`.
    """
    if not placed_macros:
        return ""
    seen = set()
    lines = ["# --- SRAM macro black-boxes (abstract for DRC/LVS, real GDS streamed) ---"]
    for m, _inst in placed_macros:
        if m.name in seen:
            continue
        seen.add(m.name)
        lines.append(f"lef read {m.lef}")
        lines.append(f"load {m.name}")
        lines.append("property LEFview true")
        if m.gds:
            lines.append(f"property GDS_FILE {m.gds}")
    return "\n".join(lines)


def lvs_macro_spice(macros: list[MacroInfo]) -> list[str]:
    """Macro spice files to include on the LVS source side (as black boxes)."""
    return [m.spice for m in macros if m.spice]


def lvs_verify_ties_enabled() -> bool:
    """Whether the LVS node deterministically PROVES a netgen pin/net mismatch
    is a benign constant-tie/replication (see ``classify_lvs_report``) and, only
    then, accepts it as a match -- grounding the acceptance in the report+netlist
    instead of the inner LLM's free-form judgement.

    Default ON. Set ``CORESMITH_LVS_VERIFY_TIES=0`` to restore the pre-fix
    behavior (trust the LLM's ``match`` verdict verbatim). The proof only ever
    turns a reported mismatch into a match when it is provably benign; it never
    forces a netgen ``match uniquely`` to fail, so designs that already sign off
    are unaffected.
    """
    from orchestrator.profile import ensure_applied, flag_enabled
    ensure_applied()
    return flag_enabled("CORESMITH_LVS_VERIFY_TIES", default=True)


# ---------------------------------------------------------------------------
# LVS constant-tie / port-equivalence PROOF (honest netgen mismatch triage)
# ---------------------------------------------------------------------------
# Netgen's top-level pin matching fails on two structurally-benign patterns a
# caravel/OpenFrame chip legitimately produces, even though device_delta == 0:
#
#   (1) CONSTANT-TIED / REPLICATED TOP OUTPUT BITS.  A wrapper drives its unused
#       GPIO outputs from shared constants, e.g.
#           assign io_out  = { 31'b0, done, qspi_o, 2'b0 };   // 33 bits -> 0
#           assign io_oeb  = { 31'b1, 1'b0, {4{oe}}, 2'b1 };  // 33 bits -> 1
#       yosys `write_verilog` lowers this to per-bit aliases in the reference
#       netlist:
#           assign io_oeb[37] = io_oeb[0];   // ... constant-1 group
#           assign io_out[0]  = io_oeb[6];   // ... constant-0 group
#           assign io_oeb[5]  = io_oeb[2];   // replicated real enable
#       The LAYOUT collapses every same-constant bit onto ONE physical tie net
#       (named after a different representative bit than the reference picks), so
#       netgen cannot 1:1-match the port pins and reports "failed pin matching".
#       This is NEVER a two-driver short: `assign portbit = X` is single-driver
#       fan-out, so tying the bits is electrically identical to the schematic.
#
#   (2) CONSTANT-TIED UNUSED MACRO INPUT PINS.  An over-provisioned SRAM macro
#       (an 8-bit cell holding 1-bit data, or a full write-mask always enabled)
#       ties its unused inputs to a shared yosys constant net:
#           .din0 ({ {7{1'b0}}, wdata }) -> `u_mem/zero_`  on din0[1:7]
#           .wmask0 ('hf)                -> `u_mem/one_`    on wmask0[*]
#       The reference keeps those pins on the single `zero_`/`one_` net; the
#       black-boxed macro's unused INPUT pins float per-pin in the extraction, so
#       each shows as its own layout net -> a positive net_delta. An open unused
#       INPUT tied to a constant cannot be a functional short.
#
# The classifier below PROVES a reported mismatch belongs to (1) or (2) from the
# report text + the reference Verilog, and refuses to bless anything else -- an
# independent real-signal net merged in the layout (a genuine short) is NOT in
# any tie/constant class and stays a mismatch. Pure/string-only; unit-testable.

# A single verilog port-bit reference, e.g. `io_out[0]` or `io_oeb [ 12 ]`.
_LVS_PORTBIT_RE = re.compile(r"([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]")
# `assign <portbit> = <rhs> ;`
_LVS_ASSIGN_RE = re.compile(
    r"assign\s+([A-Za-z_]\w*\s*\[\s*\d+\s*\])\s*=\s*([^;]+?)\s*;"
)
# A verilog constant literal, e.g. `1'b0`, `4'hf`, `0`.
_LVS_CONST_RE = re.compile(r"^(?:\d+\s*'\s*[bBhHdDoO]\s*[0-9a-fA-FxXzZ_]+|\d+)$")
# yosys constant tie nets are named `zero_` / `one_` (optionally hierarchical,
# `\path/to/zero_`), the sinks of the `hilomap` conb_1 tie cells.
_LVS_CONST_NET_RE = re.compile(r"(?:^|[\\/])(?:zero_|one_)\s*$")
# Unused-input pins a black-boxed SRAM macro legitimately leaves constant-tied.
# (The OpenRAM 1rw / 1rw1r input-pin set -- generic, not design-specific.)
_LVS_MACRO_INPUT_PINS = (
    "din0", "din1", "wmask0", "wmask1", "addr0", "addr1",
    "csb0", "csb1", "web0", "web1", "clk0", "clk1",
)


def _norm_bit(text: str) -> str | None:
    """Normalize a single `name[idx]` token to `name[idx]` (no inner spaces)."""
    m = _LVS_PORTBIT_RE.search(text)
    return f"{m.group(1)}[{int(m.group(2))}]" if m else None


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def parse_output_tie_classes(verilog_text: str) -> dict:
    """Parse a structural top netlist's port-bit aliasing into tie classes.

    Returns ``{"tied_bits": set[str], "const_bits": set[str]}`` where
    ``tied_bits`` are port bits that are the LHS of an ``assign portbit = X``
    (single-driver slaved -- never an independent output), and ``const_bits``
    are the port bits transitively connected to a constant literal. Both are
    the set of bits it is provably-benign to see collapsed/renamed in the
    layout. Purely textual; robust to escaped ids and whitespace.
    """
    text = _strip_vlog_comments(verilog_text or "")
    uf = _UnionFind()
    tied_bits: set[str] = set()
    alias_targets: set[str] = set()  # RHS port bits others alias TO (repr nets)
    const_seed: set[str] = set()  # bits DIRECTLY assigned a constant literal
    for am in _LVS_ASSIGN_RE.finditer(text):
        lhs = _norm_bit(am.group(1))
        if lhs is None:
            continue
        rhs_raw = am.group(2).strip()
        tied_bits.add(lhs)  # LHS is slaved to rhs -> single driver -> benign
        uf.find(lhs)  # register the node
        if _LVS_CONST_RE.match(rhs_raw.replace(" ", "")):
            const_seed.add(lhs)
            continue
        rhs_bit = _norm_bit(rhs_raw)
        if rhs_bit is not None and _LVS_PORTBIT_RE.fullmatch(rhs_raw.strip()):
            uf.union(lhs, rhs_bit)
            # The RHS is the shared representative net (constant or a single
            # replicated driver) the LHS bits tie to. `assign a = b` is always
            # single-driver fan-out, so the target can never be a two-driver
            # short -- it is the collapse target the layout renamed.
            alias_targets.add(rhs_bit)
        # else: rhs is a real net/expr -> lhs is single-driver fan-out (benign),
        # captured by tied_bits; not marked constant.
    # Resolve constant-ness AFTER all unions (roots move as classes merge): a
    # bit is constant iff it shares a class with a directly-constant-assigned
    # seed bit.
    const_roots = {uf.find(c) for c in const_seed}
    const_bits = {b for b in uf.parent if uf.find(b) in const_roots}
    return {
        "tied_bits": tied_bits,
        "const_bits": const_bits,
        "alias_targets": alias_targets,
    }


def _iter_report_pin_pairs(report_text: str):
    """Yield (left, right) cells from the netgen top-level pin-matching table.

    The report renders two columns split by a ``|``; each side is a pin name,
    ``(no matching pin)``, or ``(no pin, node is ...)``. Only lines inside a
    ``Subcircuit pins:`` block are considered.
    """
    in_pins = False
    _enders = ("Cell pin lists", "Netlists ", "Final result", "Device classes",
               "Subcircuit summary", "Number of")
    for line in report_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Subcircuit pins:"):
            in_pins = True
            continue
        if not in_pins:
            continue
        if any(stripped.startswith(e) for e in _enders):
            in_pins = False
            continue
        if "|" not in line:
            continue
        left, _, right = line.partition("|")
        ls, rs = left.strip(), right.strip()
        # Skip the column header + the dashed separator rows.
        if ls.startswith("Circuit 1") or set(ls) <= set("- "):
            continue
        yield ls, rs


def unmatched_top_pins(report_text: str) -> list[str]:
    """Return every top-level port bit netgen could NOT match on either side.

    A pin is 'unmatched' when its own cell is a real ``name[idx]`` but the
    facing cell is ``(no matching pin)`` / ``(no pin, node is ...)``.
    """
    out: list[str] = []
    for left, right in _iter_report_pin_pairs(report_text):
        left_bit = _norm_bit(left) if not left.startswith("(") else None
        right_bit = _norm_bit(right) if not right.startswith("(") else None
        left_missing = left.startswith("(") or left == ""
        right_missing = right.startswith("(") or right == ""
        if left_bit and right_missing:
            out.append(left_bit)
        if right_bit and left_missing:
            out.append(right_bit)
    return out


def _iter_report_net_fragments(report_text: str):
    """Yield ``(c1_net, c2_net, c1_pins, c2_pins)`` for each NET-mismatch class
    fragment. Each fragment is a ``Net: <c1> |Net: <c2>`` header followed by the
    indented member-pin rows (accumulated into ``c1_pins``/``c2_pins``) up to the
    next header or the section end."""
    in_frag = False
    cur = None  # [c1_net, c2_net, c1_pins, c2_pins]
    for line in report_text.splitlines():
        if "NET mismatches" in line and "Class fragments" in line:
            in_frag = True
            continue
        if not in_frag:
            continue
        s = line.strip()
        if s.startswith("Netlists do not match") or s.startswith("Netlists match"):
            if cur:
                yield tuple(cur)
                cur = None
            in_frag = False
            continue
        if "|" not in line:
            continue
        left, _, right = line.partition("|")
        if "Net:" in left or "Net:" in right:
            if cur:
                yield tuple(cur)
            c1 = left.split("Net:", 1)[1].strip() if "Net:" in left else ""
            c2 = right.split("Net:", 1)[1].strip() if "Net:" in right \
                else right.strip()
            cur = [c1, c2, "", ""]
        elif cur is not None:  # member-pin row under the current header
            ls, rs = left.strip(), right.strip()
            if ls and not (set(ls) <= set("- ")) and not ls.startswith("Circuit"):
                cur[2] += " " + ls
            if rs and not (set(rs) <= set("- ")) and not rs.startswith("Circuit"):
                cur[3] += " " + rs
    if cur:
        yield tuple(cur)


def _pin_is_macro_input(pin_line: str) -> bool:
    """True when a fragment pin line names a known SRAM-macro INPUT pin."""
    return any(re.search(rf"(?:^|[\\/\s]){re.escape(p)}\s*\[", pin_line)
               for p in _LVS_MACRO_INPUT_PINS)


def classify_lvs_report(
    report_text: str,
    reference_verilog_text: str = "",
) -> dict:
    """Triage a netgen LVS report into match / provably-benign / unresolved.

    Returns a dict:
      ``netgen_match``  -- netgen itself reported "Circuits match uniquely".
      ``benign``        -- every reported mismatch is a PROVEN constant-tie /
                           replication (class 1) or constant-tied macro input
                           (class 2); safe to accept as an LVS match.
      ``accept``        -- ``netgen_match or benign``.
      ``classes``       -- counts per benign class.
      ``unresolved``    -- mismatches NOT proven benign (block ``benign``).
      ``analysis``      -- human-readable one-liner.

    A genuine short (two independently-driven nets merged in the layout) is not
    in any tie/constant class, so it lands in ``unresolved`` and ``accept`` is
    False -- the honest gate is preserved.
    """
    rl = (report_text or "").lower()
    netgen_match = ("match uniquely" in rl) and ("do not match" not in rl)

    ties = parse_output_tie_classes(reference_verilog_text)
    benign_output = (
        ties["tied_bits"] | ties["const_bits"] | ties["alias_targets"]
    )

    classes = {
        "constant_tied_output": 0,
        "constant_macro_input": 0,
    }
    unresolved: list[str] = []

    # Class 1: unmatched top-level output/port bits must be provably tied.
    for bit in unmatched_top_pins(report_text):
        if bit in benign_output:
            classes["constant_tied_output"] += 1
        else:
            unresolved.append(f"top-pin {bit}")

    # Class 2: NET-mismatch fragments must be constant nets or unused macro
    # input pins (an open/tied unused INPUT can never be a functional short).
    for c1_net, c2_net, c1_pins, c2_pins in _iter_report_net_fragments(report_text):
        const_net = bool(
            _LVS_CONST_NET_RE.search(c1_net) or _LVS_CONST_NET_RE.search(c2_net)
        )
        macro_in = _pin_is_macro_input(c1_pins) or _pin_is_macro_input(c2_pins)
        if const_net or macro_in:
            classes["constant_macro_input"] += 1
        else:
            unresolved.append(f"net {c1_net or c2_net}")

    benign = (not unresolved) and (
        classes["constant_tied_output"] + classes["constant_macro_input"] > 0
    )
    accept = netgen_match or benign

    if netgen_match:
        analysis = "Circuits match uniquely."
    elif benign:
        analysis = (
            "Circuits match after accepting provably-benign equivalences: "
            f"{classes['constant_tied_output']} constant-tied/replicated top "
            f"output bit(s), {classes['constant_macro_input']} constant-tied "
            "unused macro input net(s). No independently-driven net is shorted."
        )
    else:
        analysis = (
            "Circuits do not match; unresolved (not provably constant-tie): "
            + "; ".join(unresolved[:8])
            + (" ..." if len(unresolved) > 8 else "")
        )

    return {
        "netgen_match": netgen_match,
        "benign": benign,
        "accept": accept,
        "classes": classes,
        "unresolved": unresolved,
        "analysis": analysis,
    }


# ---------------------------------------------------------------------------
# Shell -> concrete-macro NETLIST MATERIALIZATION + active-low PIN ADAPTER
# ---------------------------------------------------------------------------
# The Part-C binding resolves each `cs_mem_macro_shell` leaf to a concrete
# on-disk macro (`state["macro_bindings"]`), but the shell that reaches PnR is
# still an EMPTY tie-0 stub -- OpenROAD reads it, finds no placeable macro, and
# emits a memory-ABSENT layout. This section rewrites the PnR/DRC/LVS netlist so
# each shell MODULE really instantiates its bound concrete macro, adapting the
# generic active-HIGH shell control pins to the sky130 OpenRAM macro's
# active-LOW csb/web pins.
#
# The concrete-macro pin NAMES + polarity were read from the actual sky130
# collateral (efabless `$PDK_ROOT/sky130B/libs.ref/sky130_sram_macros/`):
#   verilog sky130_sram_1kbyte_1rw1r_8x1024_8.v / sram_1rw1r_32_64_8_sky130.v
#   + the matching .lef PIN list. Both naming schemes are the SAME OpenRAM 1rw1r
#   cell, ports (verified identical in .v and .lef):
#     port0 (RW): clk0, csb0 (active-LOW CS), web0 (active-LOW WE), wmask0[NB-1:0]
#                 (byte-mask, granularity = write_size), addr0[MA-1:0],
#                 din0[MW-1:0], dout0[MW-1:0]
#     port1 (R) : clk1, csb1 (active-LOW CS), addr1[MA-1:0], dout1[MW-1:0]
#     power     : vccd1 / vssd1 (inout; wired physically by global_connect)
#   behavioral sense: read when (!csb0 && web0), write when (!csb0 && !web0).
# The generic shell ports are active-HIGH (from rtl_lib/cs_sram.v):
#     clk, ce0 (1=selected), we0 (1=write), addr0[AW-1:0], wdata0[W-1:0],
#     rdata0[W-1:0], ce1 (1=selected), addr1[AW-1:0], rdata1[W-1:0]  (no wmask).
# So the adapter is:  csb0 = ~ce0,  web0 = ~we0,  csb1 = ~ce1,  clk0 = clk1 = clk,
#     din0 = wdata0,  rdata0 = dout0,  rdata1 = dout1,  wmask0 = all-ones (the
#     shell has no byte mask -> every write is a full-word write).

# Active-low adaptation uses a REAL inverter cell, not a behavioral `~` (which
# OpenROAD's structural verilog reader rejects). sky130 HD inverter -- the same
# family the rest of pnr_reference.tcl hardcodes (clkbuf/decap/tapcell). Its
# power pins are wired by the template's global_connect, so only .A/.Y here.
_INV_CELL = "sky130_fd_sc_hd__inv_2"

_MACRO_ADAPTER_MARK = "macro pin adapter (shell active-HIGH -> SRAM active-LOW)"
_MACRO_COMPOSITION_MARK = (
    "macro composition tiling (shell -> N pre-built tiles; active-HIGH -> "
    "active-LOW)"
)


def _clog2b(n: int) -> int:
    """Verilog $clog2 with the cs_sram.v AW convention (DEPTH<=1 -> 1)."""
    if n <= 1:
        return 1
    b, v = 0, n - 1
    while v:
        b += 1
        v >>= 1
    return b


def _zext(expr: str, extra_bits: int) -> str:
    """Zero-extend `expr` by `extra_bits` MSBs (structural concat form yosys uses
    and OpenROAD reads). No-op when extra_bits <= 0."""
    if extra_bits <= 0:
        return expr
    return "{ {%d{1'b0}}, %s }" % (extra_bits, expr)


def build_macro_adapter_instance(
    shell_width: int,
    shell_depth: int,
    *,
    macro_name: str,
    macro_data_bits: int,
    macro_words: int,
    macro_mask_bits: int = 8,
    inst_name: str = "u_macro",
) -> str:
    """Verilog fragment that instantiates a concrete sky130 1rw1r SRAM macro and
    adapts the generic active-HIGH shell ports to it (active-LOW csb/web).

    Inserted into a `cs_mem_macro_shell` module body (which already declares
    clk, ce0, we0, addr0[AW-1:0], wdata0[W-1:0], rdata0[W-1:0], ce1,
    addr1[AW-1:0], rdata1[W-1:0]) in place of the tie-0 assigns. Widths adapt
    generically: an over-provisioned macro (deeper/wider than requested) gets
    its extra address/data MSBs tied to 0 and its extra read MSBs dropped.
    """
    W = int(shell_width)
    AW = _clog2b(int(shell_depth))
    MW = int(macro_data_bits) or W
    MA = _clog2b(int(macro_words) or shell_depth)
    NB = max(1, MW // macro_mask_bits) if macro_mask_bits else 1
    # all lanes enabled -> full-word write (the shell has no byte mask)
    wmask_const = "%d'h%x" % (NB, (1 << NB) - 1)

    addr0_expr = _zext("addr0", MA - AW)
    addr1_expr = _zext("addr1", MA - AW)
    din0_expr = _zext("wdata0", MW - W)

    lines = [f"  // --- {_MACRO_ADAPTER_MARK} ---",
             "  wire _csb0, _web0, _csb1;",
             f"  {_INV_CELL} _u_csb0 (.A(ce0), .Y(_csb0));",  # csb0 = ~ce0
             f"  {_INV_CELL} _u_web0 (.A(we0), .Y(_web0));",  # web0 = ~we0
             f"  {_INV_CELL} _u_csb1 (.A(ce1), .Y(_csb1));"]  # csb1 = ~ce1
    if MW == W:
        dout0_conn, dout1_conn, post = "rdata0", "rdata1", []
    else:
        lines.append(f"  wire [{MW - 1}:0] _dout0;")
        lines.append(f"  wire [{MW - 1}:0] _dout1;")
        dout0_conn, dout1_conn = "_dout0", "_dout1"
        post = [f"  assign rdata0 = _dout0[{W - 1}:0];",
                f"  assign rdata1 = _dout1[{W - 1}:0];"]
    lines.append(f"  {macro_name} {inst_name} (")
    lines.append(f"    .clk0(clk), .csb0(_csb0), .web0(_web0), .wmask0({wmask_const}),")
    lines.append(f"    .addr0({addr0_expr}), .din0({din0_expr}), .dout0({dout0_conn}),")
    lines.append(f"    .clk1(clk), .csb1(_csb1), .addr1({addr1_expr}), .dout1({dout1_conn})")
    lines.append("  );")
    lines.extend(post)
    return "\n".join(lines)


def build_macro_composition_instance(
    shell_width: int,
    shell_depth: int,
    *,
    macro_name: str,
    macro_data_bits: int,
    macro_words: int,
    tiles_wide: int,
    tiles_deep: int,
    macro_mask_bits: int = 8,
    inst_prefix: str = "u_tile",
) -> str:
    """Verilog fragment that realizes ONE `cs_mem_macro_shell` from an N-tile
    COMPOSITION of a single pre-built 1rw1r macro, with the data/address tiling
    wrapper DERIVED FROM THE PLAN (never a per-design hardcode).

    A composition plan (openram_gen.CompositionPlan) carries: the tile macro
    (``macro_name`` + its ``macro_data_bits`` x ``macro_words`` geometry), and
    the array shape ``tiles_wide`` (columns concatenated for WIDTH) x
    ``tiles_deep`` (banks stacked for DEPTH). This builder consumes exactly
    that:

    * WIDTH tiling (``tiles_wide`` columns): column ``w`` owns logical data bits
      ``[macro_data_bits*w +: used]`` (the last column may be partial when the
      provisioned width over-runs the shell width); its ``din0`` is that slice
      (zero-extended to the tile width when partial) and its ``dout0``/``dout1``
      low ``used`` bits reassemble ``rdata0``/``rdata1`` by concatenation
      (LVS then validates ``din0[hi:lo] -> column w``).
    * DEPTH over-provision into a SINGLE bank (``tiles_deep == 1``): the logical
      ``addr0``/``addr1`` (``$clog2(shell_depth)`` bits) is zero-extended into
      the deeper tile's address (``$clog2(macro_words)`` bits) -- surplus rows
      addressed by tying the high address bits low. Control (``csb0``/``web0``/
      ``csb1``/``wmask0``/``clk``) is SHARED across every tile (all columns are
      one logical word -- written/read together), reusing the same active-LOW
      inverter adapter the single-macro path uses.

    Genuine multi-BANK depth (``tiles_deep > 1``) needs per-bank select gating +
    a registered read mux and is NOT realized by this shared-control tiling; the
    binder surfaces such a plan as a hard blocker rather than emit a silently
    mis-banked memory (see ``backend_graph._bind_macro_shells_for_backend``). So
    this builder asserts a single bank.
    """
    W = int(shell_width)
    AW = _clog2b(int(shell_depth))
    MW = int(macro_data_bits) or W
    MA = _clog2b(int(macro_words) or shell_depth)
    tw = max(1, int(tiles_wide))
    td = max(1, int(tiles_deep))
    if td != 1:
        raise ValueError(
            f"build_macro_composition_instance realizes single-bank tilings "
            f"only (tiles_deep=1); got tiles_deep={td} for {macro_name}. A "
            f"multi-bank plan must be surfaced as a blocker upstream."
        )
    NB = max(1, MW // macro_mask_bits) if macro_mask_bits else 1
    wmask_const = "%d'h%x" % (NB, (1 << NB) - 1)

    # Per-column logical bit range [lo, hi] and the USED width (planner
    # guarantees lo < W for every column: tiles_wide = ceil(W/MW), so the
    # surplus is < MW and lands entirely in the last column).
    cols = []
    for w in range(tw):
        lo = MW * w
        hi = min(MW * (w + 1), W) - 1
        used = hi - lo + 1
        cols.append((w, lo, hi, max(0, used)))

    # Shared active-LOW controls via REAL inverter cells (the structural PnR
    # verilog reader rejects behavioral `~`); mirrors the single-macro adapter.
    lines = [
        f"  // --- {_MACRO_COMPOSITION_MARK}: {tw}w x {td}d of {macro_name} ---",
        "  wire _csb0, _web0, _csb1;",
        f"  {_INV_CELL} _u_csb0 (.A(ce0), .Y(_csb0));",   # csb0 = ~ce0
        f"  {_INV_CELL} _u_web0 (.A(we0), .Y(_web0));",   # web0 = ~we0
        f"  {_INV_CELL} _u_csb1 (.A(ce1), .Y(_csb1));",   # csb1 = ~ce1
    ]
    # Depth over-provision: zero-extend the logical address into the deeper tile.
    addr0_expr = _zext("addr0", MA - AW)
    addr1_expr = _zext("addr1", MA - AW)

    # One tile per width column; declare its dout wires, wire the macro.
    rdata0_parts: list[str] = []
    rdata1_parts: list[str] = []
    for (w, lo, hi, used) in cols:
        d0, d1 = f"_dout0_c{w}", f"_dout1_c{w}"
        lines.append(f"  wire [{MW - 1}:0] {d0};")
        lines.append(f"  wire [{MW - 1}:0] {d1};")
        if used >= MW:
            din_expr = f"wdata0[{hi}:{lo}]"
        else:
            # partial column: the used shell bits, zero-extended to tile width.
            din_expr = _zext(f"wdata0[{hi}:{lo}]", MW - used)
        inst = f"{inst_prefix}_r0_c{w}"
        lines.append(f"  {macro_name} {inst} (")
        lines.append(
            f"    .clk0(clk), .csb0(_csb0), .web0(_web0), .wmask0({wmask_const}),"
        )
        lines.append(f"    .addr0({addr0_expr}), .din0({din_expr}), .dout0({d0}),")
        lines.append(
            f"    .clk1(clk), .csb1(_csb1), .addr1({addr1_expr}), .dout1({d1})"
        )
        lines.append("  );")
        # low `used` bits of each column, MSB column first -> rebuild rdata.
        rdata0_parts.append(f"{d0}[{used - 1}:0]")
        rdata1_parts.append(f"{d1}[{used - 1}:0]")

    # Reassemble the logical read words by concatenation (structural net alias):
    # column tw-1 supplies the MSBs, column 0 the LSBs. Total width == W.
    lines.append("  assign rdata0 = { %s };" % ", ".join(reversed(rdata0_parts)))
    lines.append("  assign rdata1 = { %s };" % ", ".join(reversed(rdata1_parts)))
    return "\n".join(lines)


# One `cs_mem_macro_shell` module DEFINITION (base or yosys `$paramod$..\` form).
_SHELL_MODULE_RE = re.compile(
    r"module\s+(?P<name>[\\\w$.]*cs_mem_macro_shell)\s*\((?P<ports>[^;]*)\)\s*;"
    r"(?P<body>.*?)\bendmodule",
    re.DOTALL,
)
_TIEOFF_ASSIGN_RE = re.compile(r"(?m)^[ \t]*assign[ \t]+rdata[01]\b[^;]*;[ \t]*\n?")


def _macro_info_from_binding(b: dict) -> MacroInfo:
    """Reconstruct a MacroInfo from a `state["macro_bindings"]` dict. Geometry
    comes from the binding (self-contained -- no PDK needed for the netlist
    rewrite); physical LEF dims + power pins are parsed from the macro LEF when
    it exists (best-effort, only needed for floorplan sizing)."""
    from pathlib import Path as _P

    from orchestrator.langgraph.macro_registry import _parse_lef
    mi = MacroInfo(
        name=b.get("name", ""), lef=b.get("lef", ""), gds=b.get("gds", ""),
        lib=b.get("lib", ""), spice=b.get("spice", ""),
        verilog=b.get("verilog", ""),
        data_bits=int(b.get("macro_data_bits") or 0),
        words=int(b.get("macro_words") or 0),
        mask_bits=int(b.get("macro_mask_bits") or 0),
        ports=b.get("ports", "") or "1rw1r",
    )
    mi.bits = mi.data_bits * mi.words
    if mi.lef and _P(mi.lef).exists():
        w, h, pp, gp = _parse_lef(_P(mi.lef))
        mi.width_um, mi.height_um, mi.power_pin, mi.ground_pin = w, h, pp, gp
    return mi


def materialize_macro_netlist(
    netlist_text: str, bindings: list[dict]
) -> tuple[str, list[tuple[MacroInfo, str]]]:
    """Rewrite a synthesized netlist so every `cs_mem_macro_shell` MODULE
    instantiates its bound CONCRETE SRAM macro instead of an empty tie-0 stub.
    A shell bound to a single macro gets the active-low pin adapter; a shell
    bound to an N-tile COMPOSITION plan (binding carries a ``composition`` dict)
    gets the data/address tiling wrapper (:func:`build_macro_composition_instance`)
    that instantiates all N concrete tiles. The module NAME (hence every parent
    instance site) is preserved, so only the leaf definition changes.

    Returns ``(rewritten_text, placed)`` where ``placed`` is one
    ``(MacroInfo, synthetic_inst)`` per PHYSICAL macro instance -- a composition
    shell contributes ``tiles_wide*tiles_deep`` tiles per shell instance -- so
    the floorplan is sized for the real count. Returns the text unchanged with
    ``placed == []`` when there are no SRAM shells or no matching SRAM binding
    (a ROM shell, or an unmatched geometry, is left untouched -- the PnR
    memory-absent assertion then catches a truly unplaced macro).
    """
    from orchestrator.langgraph.macro_registry import detect_macro_shells

    by_geom: dict[tuple[int, int], dict] = {}
    for b in bindings or []:
        if b.get("kind", "sram") == "sram" and b.get("name"):
            try:
                by_geom[(int(b["width"]), int(b["depth"]))] = b
            except (KeyError, TypeError, ValueError):
                continue
    if not by_geom:
        return netlist_text, []

    replaced: list[tuple[str, dict]] = []

    def _sub(m: re.Match) -> str:
        block = m.group(0)
        specs = detect_macro_shells(block)
        if not specs:
            return block
        sp = specs[0]
        b = by_geom.get((sp.width, sp.depth))
        if b is None:
            return block  # no binding for this geometry -> leave as-is
        comp = b.get("composition")
        if comp:
            # Shell resolved to an N-tile COMPOSITION: instantiate the concrete
            # tiles (data/address tiling derived from the plan), not one macro.
            body = build_macro_composition_instance(
                sp.width, sp.depth,
                macro_name=b["name"],
                macro_data_bits=int(b.get("macro_data_bits") or sp.width),
                macro_words=int(b.get("macro_words") or sp.depth),
                tiles_wide=int(comp.get("tiles_wide") or 1),
                tiles_deep=int(comp.get("tiles_deep") or 1),
                macro_mask_bits=int(b.get("macro_mask_bits") or 8),
            )
        else:
            body = build_macro_adapter_instance(
                sp.width, sp.depth,
                macro_name=b["name"],
                macro_data_bits=int(b.get("macro_data_bits") or sp.width),
                macro_words=int(b.get("macro_words") or sp.depth),
                macro_mask_bits=int(b.get("macro_mask_bits") or 8),
            )
        stripped = _TIEOFF_ASSIGN_RE.sub("", block)
        cut = stripped.rindex("endmodule")
        replaced.append((m.group("name"), b))
        return stripped[:cut] + body + "\nendmodule"

    new_text = _SHELL_MODULE_RE.sub(_sub, netlist_text)

    # Physical instance count per shell = the FLATTENED leaf count. A raw text
    # count (occurrences - 1) undercounts a shell instantiated inside a
    # sub-module that is itself instantiated K times (e.g. 3 shell modules
    # -> 14 leaves once OpenROAD flattens). Elaborate the hierarchy to get K.
    flat: dict[str, int] = {}
    try:
        _top = detect_top_module(netlist_text, "")
        flat = flattened_type_counts(netlist_text, _top)
    except Exception:
        flat = {}
    placed: list[tuple[MacroInfo, str]] = []
    for name, b in replaced:
        mi = _macro_info_from_binding(b)
        n_inst = int(flat.get(name, 0)) or max(1, netlist_text.count(name) - 1)
        n_inst = max(1, n_inst)
        # A composition shell materializes into tiles_wide*tiles_deep PHYSICAL
        # tile macros per shell instance -- size the floorplan for every tile.
        comp = b.get("composition") or {}
        per_shell = max(
            1,
            int(comp.get("tiles_wide") or 1) * int(comp.get("tiles_deep") or 1),
        )
        stem = re.sub(r"[^\w]", "_", name).strip("_") or "macro"
        for i in range(n_inst * per_shell):
            placed.append((mi, f"{stem}_{i}"))
    return new_text, placed
