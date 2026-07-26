# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Idempotently repair a pip-installed OpenRAM so its sky130 compilers run.

The PyPI ``openram==1.2.48`` wheel has two defects that block sky130 macro
generation on a stock modern install:

1. It omits ``openram/__main__.py``, so ``python -m openram <cfg>`` (how the
   engine invokes it) dies with "cannot be directly executed" -- even though
   ``import openram`` succeeds.
2. Its gdsMill bounding-box path calls ``float()`` on size-one NumPy arrays,
   which NumPy >= 2 removed (``TypeError: only 0-dimensional arrays can be
   converted to Python scalars``). OpenRAM only pins ``numpy>=1.17.4``, so pip
   selects NumPy 2 and generation crashes mid-layout.
3. Its ROM compiler crashes for ``words_per_row=1`` (required by odd logical
   depths such as 1141): the one-output column decoder is built with zero
   address inputs. Keep one internal decoder input, tie it physically and
   schematically to ground, and expose no extra logical address pin.
4. Several sky130 hard-cell SPICE views express transistor W/L values as bare
   micron numbers, and eight views used by the 1RW+1R compiler are stale
   relative to the pinned PDK's extracted hard cells. Netgen otherwise sees
   false property, device-class, and drain-only-device mismatches.
5. The 1RW+1R top-level placer leaves a channel-route M2 track only 0.055um
   from the adjacent write-mask DFF via, and its clock spine necks to minimum
   width at a wider M2-M3 enclosure. Both violate sky130 met2.2. Separating
   the DFF arrays must also preserve the technology's 1.27um N-well spacing.
6. Parametric 0.36um NMOS devices are extracted by the pinned Magic tech as
   ``special_nfet_01v8``, but the wheel always writes the regular NMOS model.
   Select the schematic model using the same pinned width boundary.

These are patched HERE, in-repo, so a fresh checkout on any box self-heals the
first time the macro flow runs (see ``openram_gen.ensure_openram_patched``) --
rather than depending on a manual venv edit that would not travel with a
public commit. Every edit is idempotent (a marker/exact-pattern check), backs
the target up once, and is a no-op when already applied. Downgrading NumPy is
deliberately NOT attempted: a shared venv's SciPy/scikit-learn need NumPy 2.

Usable as a library (``patch_openram() -> PatchResult``) or a CLI
(``python scripts/patch_openram.py``; ``--check`` reports status without
writing).
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

_MAIN_MARKER = "coresmith openram __main__ shim"
_MAIN_SHIM = f'''"""Module entry point for ``python -m openram`` ({_MAIN_MARKER}).

The PyPI 1.2.48 wheel ships sram_compiler.py but omits __main__.py; recreate
it with the compiler's script semantics so the package is executable.
"""
from pathlib import Path
import runpy
import sys

_pkg = Path(__file__).resolve().parent
sys.path.insert(0, str(_pkg))
runpy.run_path(str(_pkg / "sram_compiler.py"), run_name="__main__")
'''

# The NumPy-2-incompatible bbox pattern, matched INDENTATION- and
# variable-name-agnostically: any `vector(boundary[i][j], boundary[k][l])`
# whose args are bare (no .item()). Adds `.item()` to each subscript so the
# size-one ndarray converts under NumPy 2. Idempotent: an already-fixed call
# has `.item()` before the comma/paren, so the bare-arg group won't match.
_BBOX_RE = re.compile(
    r"vector\(\s*"
    r"(boundary\[\d+\]\[\d+\])\s*,\s*"
    r"(boundary\[\d+\]\[\d+\])\s*\)"
)
_HORIZONTAL_RENAME_MARKER = "coresmith preserve route_horizontal_pins new_name"
_SPICE_DIMENSION_RE = re.compile(
    r"(?P<prefix>\b[WwLl]=)"
    r"(?P<value>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"(?=\s|$)"
)

_PDK_HARDCELL_MAP = {
    "sky130_fd_bd_sram__openram_dp_cell":
        "sky130_fd_bd_sram__openram_dp_cell",
    "sky130_fd_bd_sram__openram_dp_cell_replica":
        "sky130_fd_bd_sram__openram_dp_cell_replica",
    "sky130_fd_bd_sram__openram_dp_cell_dummy":
        "sky130_fd_bd_sram__openram_dp_cell_dummy",
    "sky130_fd_bd_sram__openram_dff": "dff",
    "sky130_fd_bd_sram__openram_dp_nand2_dec": "nand2_dec",
    "sky130_fd_bd_sram__openram_dp_nand3_dec": "nand3_dec",
    "sky130_fd_bd_sram__openram_sense_amp": "sense_amp",
    "sky130_fd_bd_sram__openram_write_driver": "write_driver",
}
_DP_SPICE_MARKER = "coresmith pinned-PDK hard-cell schematic"
_PDK_DP_SPICE_REL = Path(
    "sky130A/libs.ref/sky130_sram_macros/spice/"
    "sram_1rw1r_32_256_8_sky130.spice"
)

_SRAM_M2_DFF_GAP_MARKER = "coresmith reserve M2 track between DFF arrays"
_SRAM_M2_CLK_WIDEN_MARKER = "coresmith widen M2 clock spine at via"
_SRAM_M2_DFF_GAP_BROKEN = (
    "            self.col_addr_dff_insts[port].place(self.col_addr_pos[port])\n"
    "            x_offset = self.col_addr_dff_insts[port].rx()\n"
    "        else:\n"
)
_SRAM_M2_DFF_GAP_FIXED = (
    "            self.col_addr_dff_insts[port].place(self.col_addr_pos[port])\n"
    "            x_offset = self.col_addr_dff_insts[port].rx()\n"
    "            # Leave a legal routing track before the next DFF array; the\n"
    "            # adjacent channel route otherwise has only 0.055um of M2\n"
    "            # clearance from its edge clock via in sky130.\n"
    "            x_offset += self.m2_pitch + self.nwell_space  # "
    + _SRAM_M2_DFF_GAP_MARKER
    + "\n"
    "        else:\n"
)
_SRAM_M2_DFF_GAP_TRACK_ONLY = _SRAM_M2_DFF_GAP_FIXED.replace(
    "self.m2_pitch + self.nwell_space", "self.m2_pitch"
)
_SRAM_M2_DFF_GAP_TRACK_ONLY_LINE = (
    "            x_offset += self.m2_pitch  # "
    + _SRAM_M2_DFF_GAP_MARKER
    + "\n"
)
_SRAM_M2_DFF_GAP_FIXED_LINE = (
    "            x_offset += self.m2_pitch + self.nwell_space  # "
    + _SRAM_M2_DFF_GAP_MARKER
    + "\n"
)
_SRAM_M2_CLK_WIDEN_BROKEN = (
    "                mid_pos = vector(clk_steiner_pos.x, dff_clk_pos.y)\n"
    "                self.add_wire(self.m2_stack[::-1],\n"
)
_SRAM_M2_CLK_WIDEN_FIXED = (
    "                mid_pos = vector(clk_steiner_pos.x, dff_clk_pos.y)\n"
    "                # Keep the M2 spine as wide as its M2-M3 enclosure. A\n"
    "                # min-width neck leaves a sub-rule notch at the via.\n"
    "                self.add_path(\"m2\",\n"
    "                              [mid_pos, clk_steiner_pos],\n"
    "                              width=max(self.m2_via.width,\n"
    "                                        self.m2_via.height))  # "
    + _SRAM_M2_CLK_WIDEN_MARKER
    + "\n"
    "                self.add_wire(self.m2_stack[::-1],\n"
)


def _bbox_sub(text: str) -> tuple[str, int]:
    """Return (patched_text, n_substitutions) -- adds .item() to bare
    ``vector(boundary[..], boundary[..])`` calls; leaves fixed ones alone."""
    return _BBOX_RE.subn(
        lambda m: f"vector({m.group(1)}.item(), {m.group(2)}.item())", text)


def _horizontal_pin_rename_sub(text: str) -> tuple[str, int]:
    """Honor ``new_name`` when a horizontal pin bin has only one member.

    OpenRAM 1.2.48 copies the child pin's old name in this fallback, so callers
    requesting ``new_name="vdd_tmp"`` immediately fail to find ``vdd_tmp``.
    Restrict the source edit to the route_horizontal_pins function; the similar
    route_vertical_pins code has no rename argument and must remain unchanged.
    """
    if _HORIZONTAL_RENAME_MARKER in text:
        return text, 0
    start = text.find("def route_horizontal_pins")
    if start < 0:
        return text, 0
    end = text.find("\n    def ", start + 4)
    if end < 0:
        end = len(text)
    block = text[start:end]
    broken = "self.add_layout_pin(pin.name,"
    if broken not in block:
        return text, 0
    fixed = (
        "self.add_layout_pin(pin_name,  # "
        + _HORIZONTAL_RENAME_MARKER
    )
    block = block.replace(broken, fixed, 1)
    return text[:start] + block + text[end:], 1


def _spice_micron_units_sub(text: str) -> tuple[str, int]:
    """Suffix bare sky130 hard-cell transistor W/L values with microns."""
    out: list[str] = []
    substitutions = 0
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if (
            stripped[:1].upper() in {"X", "M"}
            and "sky130_fd_pr__" in stripped
        ):
            line, count = _SPICE_DIMENSION_RE.subn(
                lambda match: (
                    f"{match.group('prefix')}{match.group('value')}u"
                ),
                line,
            )
            substitutions += count
        out.append(line)
    return "".join(out), substitutions


def _extract_spice_subckt(text: str, name: str) -> str | None:
    """Return one complete named SPICE subcircuit, case-insensitively."""
    match = re.search(
        rf"^\.subckt\s+{re.escape(name)}\b.*?^\.ends(?:\s+{re.escape(name)})?\s*$",
        text,
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    return match.group(0) if match else None


def _pdk_dp_spice_sub(
    old: str,
    canonical_source: str,
    name: str,
    signature_source: str | None = None,
    canonical_name: str | None = None,
) -> tuple[str, int]:
    """Replace one stale wheel hard-cell schematic with the pinned PDK view.

    The wheel GDS is byte-for-byte geometrically identical to the copy embedded
    in the pinned PDK SRAM macro, but its three dual-port SPICE views predate the
    PDK's special-pfet classification and drain-only devices.  Preserve the
    wheel's license header and synchronize only the named subcircuit.
    """
    canonical = _extract_spice_subckt(
        canonical_source, canonical_name or name
    )
    current = _extract_spice_subckt(old, name)
    signature = _extract_spice_subckt(signature_source or old, name)
    if canonical is None or current is None or signature is None:
        return old, 0
    canonical, _ = _spice_micron_units_sub(canonical)
    # In the pinned Magic extraction deck, drain-only SRAM PFET geometry is
    # classified as special_pfet_latch. ``special_pfet_pass`` survives only
    # as a Netgen compatibility model and is never emitted by extraction.
    canonical = canonical.replace(
        "sky130_fd_pr__special_pfet_pass",
        "sky130_fd_pr__special_pfet_latch",
    )
    canonical_lines = canonical.splitlines()
    # OpenRAM treats custom-cell pin spelling as case-sensitive even though
    # SPICE does not. Keep its shipped declaration byte-for-byte while using
    # the PDK-authoritative device body.
    canonical_lines[0] = signature.splitlines()[0]
    canonical = "\n".join(canonical_lines)
    prefix = old[:old.lower().find(".subckt")]
    prefix = "\n".join(
        line for line in prefix.splitlines()
        if _DP_SPICE_MARKER not in line
    ).rstrip()
    desired = (
        prefix
        + "\n\n* "
        + _DP_SPICE_MARKER
        + "\n"
        + canonical.strip()
        + "\n"
    )
    if old == desired:
        return old, 0
    return desired, 1


def _sram_m2_drc_sub(text: str) -> tuple[str, int]:
    """Repair reproduced M2 defects and preserve legal N-well spacing."""
    substitutions = 0
    if _SRAM_M2_DFF_GAP_FIXED_LINE not in text:
        if _SRAM_M2_DFF_GAP_TRACK_ONLY_LINE in text:
            text = text.replace(
                _SRAM_M2_DFF_GAP_TRACK_ONLY_LINE,
                _SRAM_M2_DFF_GAP_FIXED_LINE,
                1,
            )
            substitutions += 1
        elif _SRAM_M2_DFF_GAP_BROKEN in text:
            text = text.replace(
                _SRAM_M2_DFF_GAP_BROKEN,
                _SRAM_M2_DFF_GAP_FIXED,
                1,
            )
            substitutions += 1
    if _SRAM_M2_CLK_WIDEN_MARKER not in text:
        if _SRAM_M2_CLK_WIDEN_BROKEN in text:
            text = text.replace(
                _SRAM_M2_CLK_WIDEN_BROKEN,
                _SRAM_M2_CLK_WIDEN_FIXED,
                1,
            )
            substitutions += 1
    return text, substitutions


_PTX_WIDTH_MODEL_MARKER = "coresmith match pinned Magic width taxonomy"
_PTX_WIDTH_MODEL_ANCHOR = (
    "        self.add_pin_list(pin_list, dir_list)\n\n"
    "        # Just make a guess since these will actually\n"
)
_PTX_WIDTH_MODEL_FIXED = (
    "        self.add_pin_list(pin_list, dir_list)\n\n"
    "        # The pinned Magic deck uses a special device class below a\n"
    "        # technology-defined width. Keep generated SPICE/LVS in the same\n"
    "        # taxonomy instead of relying on a Netgen class equivalence.\n"
    "        tx_model = spice[self.tx_type]\n"
    "        width_limit = spice.get(self.tx_type + \"_model_width_limit\")\n"
    "        if width_limit is not None and self.tx_width < width_limit:\n"
    "            tx_model = spice[self.tx_type + \"_model_below_width_limit\"]\n"
    "        # " + _PTX_WIDTH_MODEL_MARKER + "\n\n"
    "        # Just make a guess since these will actually\n"
)
_SKY130_WIDTH_MODEL_MARKER = "coresmith pinned Magic NMOS width taxonomy"
_SKY130_WIDTH_MODEL_ANCHOR = (
    'spice["nmos"] = "sky130_fd_pr__nfet_01v8"\n'
)
_SKY130_WIDTH_MODEL_FIXED = (
    _SKY130_WIDTH_MODEL_ANCHOR
    + '# ' + _SKY130_WIDTH_MODEL_MARKER + '\n'
    + 'spice["nmos_model_width_limit"] = 0.42\n'
    + 'spice["nmos_model_below_width_limit"] = '
      '"sky130_fd_pr__special_nfet_01v8"\n'
)


def _ptx_width_model_sub(
    ptx_text: str,
    tech_text: str,
) -> tuple[str, str, int]:
    """Match schematic device classes to the pinned Magic width taxonomy."""
    substitutions = 0
    if _PTX_WIDTH_MODEL_MARKER not in ptx_text:
        if _PTX_WIDTH_MODEL_ANCHOR not in ptx_text:
            return ptx_text, tech_text, substitutions
        ptx_text = ptx_text.replace(
            _PTX_WIDTH_MODEL_ANCHOR, _PTX_WIDTH_MODEL_FIXED, 1
        )
        # Use the selected model in every generated SPICE/LVS format string.
        ptx_text, count = re.subn(
            r"\.format\(spice\[self\.tx_type\],",
            ".format(tx_model,",
            ptx_text,
        )
        if count < 4:
            return ptx_text, tech_text, substitutions
        substitutions += 1
    if _SKY130_WIDTH_MODEL_MARKER not in tech_text:
        if _SKY130_WIDTH_MODEL_ANCHOR not in tech_text:
            return ptx_text, tech_text, substitutions
        tech_text = tech_text.replace(
            _SKY130_WIDTH_MODEL_ANCHOR,
            _SKY130_WIDTH_MODEL_FIXED,
            1,
        )
        substitutions += 1
    return ptx_text, tech_text, substitutions


_ROM_NOMUX_MARKER = "coresmith ROM words_per_row=1"
_SIGNAL_ESCAPE_MARKER = "coresmith keep signal pin already at perimeter"
_SIGNAL_OVERLAP_MARKER = "coresmith keep overlapping escape target"
_SIGNAL_ZERO_WIRE_MARKER = "coresmith keep zero-wire escape target"
_SIGNAL_RECT_MARKER = "coresmith accept graph_shape rect"
_SIGNAL_EDGE_FALLBACK_MARKER = "coresmith preserve accessible pin without strip"
_ROM_DECODER_BROKEN = "self.num_inputs = ceil(log(num_outputs, 2))"
_ROM_DECODER_FIXED = (
    "self.num_inputs = max(1, ceil(log(num_outputs, 2)))  "
    f"# {_ROM_NOMUX_MARKER}"
)
_ROM_BANK_ADDR_BROKEN = (
    'addr_lsb = ["addr0[{}]".format(addr) '
    "for addr in range(self.col_bits)]"
)
_ROM_BANK_ADDR_FIXED = (
    _ROM_BANK_ADDR_BROKEN
    + "\n        if self.words_per_row == 1:\n"
    + "            # The one-output column decoder has one internal address\n"
    + "            # input only to avoid a zero-input layout crash. It is not\n"
    + "            # a logical address bit and is tied to ground.\n"
    + '            addr_lsb = ["vssd1"]  # ' + _ROM_NOMUX_MARKER
)
_ROM_BANK_ROUTE_BROKEN = (
    "        self.connect_row_pins(self.wordline_layer, sel_pins, round=True)\n"
)
_ROM_BANK_ROUTE_FIXED = (
    _ROM_BANK_ROUTE_BROKEN
    + "\n"
    + "        if self.words_per_row == 1:\n"
    + "            # Match the schematic vssd1 tie above in layout by routing\n"
    + "            # the decoder's internal A0 pin to its nearest ground pin.\n"
    + '            addr = self.col_decode_inst.get_pin("A0")\n'
    + '            grounds = self.col_decode_inst.get_pins("gnd")\n'
    + "            ground = min(grounds, key=lambda p: abs(p.cx() - addr.cx())\n"
    + "                         + abs(p.cy() - addr.cy()))\n"
    + "            route_layer = self.route_stack[0]\n"
    + "            self.add_via_stack_center(addr.center(), addr.layer,\n"
    + "                                      route_layer)\n"
    + "            self.add_via_stack_center(ground.center(), ground.layer,\n"
    + "                                      route_layer)\n"
    + "            corner = vector(ground.cx(), addr.cy())\n"
    + "            self.add_path(route_layer,\n"
    + "                          [addr.center(), corner, ground.center()])\n"
    + "            # " + _ROM_NOMUX_MARKER + "\n"
)


def _rom_nomux_sub(
    decoder_text: str,
    bank_text: str,
) -> tuple[str, str, int]:
    """Patch OpenRAM 1.2.48's zero-input one-column ROM decoder.

    Returns ``(decoder, bank, substitutions)``. Exact anchors make a future
    upstream implementation a no-op rather than a guessed source rewrite.
    """
    n = 0
    if _ROM_NOMUX_MARKER not in decoder_text:
        if _ROM_DECODER_BROKEN in decoder_text:
            decoder_text = decoder_text.replace(
                _ROM_DECODER_BROKEN, _ROM_DECODER_FIXED, 1
            )
            n += 1
    if _ROM_NOMUX_MARKER not in bank_text:
        if (
            _ROM_BANK_ADDR_BROKEN in bank_text
            and _ROM_BANK_ROUTE_BROKEN in bank_text
        ):
            bank_text = bank_text.replace(
                _ROM_BANK_ADDR_BROKEN, _ROM_BANK_ADDR_FIXED, 1
            )
            bank_text = bank_text.replace(
                _ROM_BANK_ROUTE_BROKEN, _ROM_BANK_ROUTE_FIXED, 1
            )
            n += 1
    return decoder_text, bank_text, n


_SIGNAL_ESCAPE_ANCHOR = (
    "        for source, target, _ in self.get_route_pairs(pin_names):\n"
    "            # Change fake pin's name so the graph will treat it as routable\n"
)
_SIGNAL_ESCAPE_FIXED = (
    "        for source, target, _ in self.get_route_pairs(pin_names):\n"
    "            # A pin that already intersects the perimeter needs no escape\n"
    "            # wire. Trying to graph-route its overlapping fake target can\n"
    "            # return no path even though it is already externally usable.\n"
    "            if any(source.intersection(fake) for fake in self.fake_pins):\n"
    "                self.new_pins[source.name] = source\n"
    "                routed_count += 1\n"
    "                continue  # " + _SIGNAL_ESCAPE_MARKER + "\n"
    "            # Change fake pin's name so the graph will treat it as routable\n"
)


def _signal_escape_sub(text: str) -> tuple[str, int]:
    if _SIGNAL_ESCAPE_MARKER in text:
        return text, 0
    if _SIGNAL_ESCAPE_ANCHOR not in text:
        return text, 0
    return text.replace(
        _SIGNAL_ESCAPE_ANCHOR, _SIGNAL_ESCAPE_FIXED, 1
    ), 1


_SIGNAL_OVERLAP_ANCHOR = (
    "            target.name = source.name\n"
    "            # Create the graph\n"
)
_SIGNAL_OVERLAP_FIXED = (
    "            target.name = source.name\n"
    "            # The generated target can already overlap a source that\n"
    "            # straddles the boundary. Their overlap is the escape wire;\n"
    "            # graph routing this zero-distance case can report no path.\n"
    "            if source.overlaps(target):\n"
    "                self.new_pins[source.name] = target\n"
    "                routed_count += 1\n"
    "                continue  # " + _SIGNAL_OVERLAP_MARKER + "\n"
    "            # Create the graph\n"
)


def _signal_overlap_sub(text: str) -> tuple[str, int]:
    if _SIGNAL_OVERLAP_MARKER in text:
        return text, 0
    if _SIGNAL_OVERLAP_ANCHOR not in text:
        return text, 0
    return text.replace(
        _SIGNAL_OVERLAP_ANCHOR, _SIGNAL_OVERLAP_FIXED, 1
    ), 1


_SIGNAL_ZERO_WIRE_BROKEN = (
    "            self.new_pins[source.name] = new_wires[-1]\n"
)
_SIGNAL_ZERO_WIRE_FIXED = (
    "            self.new_pins[source.name] = (new_wires[-1] if new_wires "
    "else target)  # " + _SIGNAL_ZERO_WIRE_MARKER + "\n"
)


def _signal_zero_wire_sub(text: str) -> tuple[str, int]:
    if _SIGNAL_ZERO_WIRE_MARKER in text:
        return text, 0
    if _SIGNAL_ZERO_WIRE_BROKEN not in text:
        return text, 0
    return text.replace(
        _SIGNAL_ZERO_WIRE_BROKEN, _SIGNAL_ZERO_WIRE_FIXED, 1
    ), 1


_SIGNAL_RECT_BROKEN = (
    "            pin = graph_shape(pin.name, pin.boundary, pin.lpp)\n"
)
_SIGNAL_RECT_FIXED = (
    "            rect = pin.boundary if hasattr(pin, \"boundary\") else pin.rect\n"
    "            pin = graph_shape(pin.name, rect, pin.lpp)  # "
    + _SIGNAL_RECT_MARKER + "\n"
)


def _signal_rect_sub(text: str) -> tuple[str, int]:
    if _SIGNAL_RECT_MARKER in text:
        return text, 0
    if _SIGNAL_RECT_BROKEN not in text:
        return text, 0
    return text.replace(_SIGNAL_RECT_BROKEN, _SIGNAL_RECT_FIXED, 1), 1


_SIGNAL_EDGE_FALLBACK_BROKEN = (
    "            self.design.replace_layout_pin(name, edge)\n"
)
_SIGNAL_EDGE_FALLBACK_FIXED = (
    "            if edge is None:\n"
    "                edge = pin  # " + _SIGNAL_EDGE_FALLBACK_MARKER + "\n"
    "            self.design.replace_layout_pin(name, edge)\n"
)


def _signal_edge_fallback_sub(text: str) -> tuple[str, int]:
    if _SIGNAL_EDGE_FALLBACK_MARKER in text:
        return text, 0
    if _SIGNAL_EDGE_FALLBACK_BROKEN not in text:
        return text, 0
    return text.replace(
        _SIGNAL_EDGE_FALLBACK_BROKEN,
        _SIGNAL_EDGE_FALLBACK_FIXED,
        1,
    ), 1


@dataclass
class PatchResult:
    ok: bool                       # openram is runnable after this call
    applied: list[str] = field(default_factory=list)
    already: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = []
        if self.applied:
            parts.append("applied: " + ", ".join(self.applied))
        if self.already:
            parts.append("already-ok: " + ", ".join(self.already))
        if self.errors:
            parts.append("errors: " + "; ".join(self.errors))
        return f"openram runnable={self.ok} [{'; '.join(parts) or 'no-op'}]"


def _openram_pkg_dir() -> Path | None:
    try:
        spec = importlib.util.find_spec("openram")
    except Exception:
        return None
    if not spec or not spec.origin:
        return None
    return Path(spec.origin).resolve().parent


def _find_hierarchy_layout(pkg: Path) -> Path | None:
    cand = pkg / "compiler" / "base" / "hierarchy_layout.py"
    if cand.exists():
        return cand
    hits = list(pkg.rglob("hierarchy_layout.py"))
    return hits[0] if hits else None


def patch_openram(*, check_only: bool = False) -> PatchResult:
    """Apply (or, with ``check_only``, report) the OpenRAM repairs."""
    res = PatchResult(ok=False)
    pkg = _openram_pkg_dir()
    if pkg is None:
        res.errors.append("openram package not importable")
        return res

    # (1) __main__.py launcher --------------------------------------------
    main_py = pkg / "__main__.py"
    main_ok = main_py.exists() and _MAIN_MARKER in main_py.read_text(
        errors="ignore") if main_py.exists() else False
    # A wheel that already ships a working __main__ (future fix) also counts.
    stock_main_ok = main_py.exists() and _MAIN_MARKER not in main_py.read_text(
        errors="ignore")
    if main_ok or stock_main_ok:
        res.already.append("__main__.py")
    elif check_only:
        res.errors.append("__main__.py missing")
    else:
        try:
            main_py.write_text(_MAIN_SHIM, encoding="utf-8")
            res.applied.append("__main__.py")
        except OSError as e:
            res.errors.append(f"__main__.py write failed ({e}); "
                              "venv may be read-only")

    # (2) NumPy-2 bbox .item() fix ----------------------------------------
    hl = _find_hierarchy_layout(pkg)
    if hl is None:
        res.errors.append("hierarchy_layout.py not found")
    else:
        text = hl.read_text(errors="ignore")
        patched, n = _bbox_sub(text)
        has_fixed = ".item()" in text and "vector(boundary" in text.replace(
            ".item()", "")  # a fixed call still contains vector(boundary..
        if n == 0:
            # No bare vector(boundary..) call. Either already fixed (a
            # .item() form present) or an unexpected version (neither).
            if "vector(boundary" in text or has_fixed:
                res.already.append("bbox-item")
            else:
                res.errors.append(
                    "bbox lines not found (unexpected openram version -- "
                    "not patched; verify generation manually)")
        elif check_only:
            res.errors.append("bbox-item not applied")
        else:
            try:
                bak = hl.with_suffix(".py.coresmith-bak")
                if not bak.exists():
                    bak.write_text(text, encoding="utf-8")
                hl.write_text(patched, encoding="utf-8")
                res.applied.append("bbox-item")
            except OSError as e:
                res.errors.append(f"bbox patch write failed ({e})")

        # A second hierarchy-layout defect is independent of NumPy: honor the
        # requested renamed pin even for singleton alignment bins.
        current = hl.read_text(errors="ignore")
        rename_patched, rename_n = _horizontal_pin_rename_sub(current)
        if _HORIZONTAL_RENAME_MARKER in current:
            res.already.append("horizontal-pin-rename")
        elif rename_n == 0:
            res.errors.append(
                "route_horizontal_pins rename anchor not found "
                "(unexpected OpenRAM version)"
            )
        elif check_only:
            res.errors.append("horizontal-pin-rename not applied")
        else:
            try:
                bak = hl.with_suffix(".py.coresmith-bak")
                if not bak.exists():
                    bak.write_text(current, encoding="utf-8")
                hl.write_text(rename_patched, encoding="utf-8")
                res.applied.append("horizontal-pin-rename")
            except OSError as e:
                res.errors.append(f"horizontal pin rename patch failed ({e})")

    # (3) ROM words_per_row=1 / odd-depth fix ----------------------------
    rom_decoder = pkg / "compiler" / "modules" / "rom_decoder.py"
    rom_bank = pkg / "compiler" / "modules" / "rom_bank.py"
    if not rom_decoder.exists() or not rom_bank.exists():
        # Some older/future wheels do not ship the ROM compiler. SRAM remains
        # runnable; ROM availability is checked separately by openram_gen.
        res.already.append("rom-nomux-unavailable")
    else:
        decoder_text = rom_decoder.read_text(errors="ignore")
        bank_text = rom_bank.read_text(errors="ignore")
        already_fixed = (
            _ROM_NOMUX_MARKER in decoder_text
            and _ROM_NOMUX_MARKER in bank_text
        )
        patched_decoder, patched_bank, n = _rom_nomux_sub(
            decoder_text, bank_text
        )
        if already_fixed:
            res.already.append("rom-nomux")
        elif n != 2:
            res.errors.append(
                "ROM no-mux anchors not found (unexpected OpenRAM version; "
                "odd-depth ROM generation requires manual verification)"
            )
        elif check_only:
            res.errors.append("rom-nomux not applied")
        else:
            try:
                for path, old, new in (
                    (rom_decoder, decoder_text, patched_decoder),
                    (rom_bank, bank_text, patched_bank),
                ):
                    bak = path.with_suffix(".py.coresmith-bak")
                    if not bak.exists():
                        bak.write_text(old, encoding="utf-8")
                    path.write_text(new, encoding="utf-8")
                res.applied.append("rom-nomux")
            except OSError as e:
                res.errors.append(f"ROM no-mux patch write failed ({e})")

    # (4) Signal already on perimeter is already escaped -----------------
    signal_escape = (
        pkg / "compiler" / "router" / "signal_escape_router.py"
    )
    if not signal_escape.exists():
        res.already.append("signal-escape-unavailable")
    else:
        escape_text = signal_escape.read_text(errors="ignore")
        escape_patched, escape_n = _signal_escape_sub(escape_text)
        if _SIGNAL_ESCAPE_MARKER in escape_text:
            res.already.append("signal-escape-edge-pin")
        elif escape_n == 0:
            res.errors.append(
                "signal escape edge-pin anchor not found "
                "(unexpected OpenRAM version)"
            )
        elif check_only:
            res.errors.append("signal-escape-edge-pin not applied")
        else:
            try:
                bak = signal_escape.with_suffix(".py.coresmith-bak")
                if not bak.exists():
                    bak.write_text(escape_text, encoding="utf-8")
                signal_escape.write_text(escape_patched, encoding="utf-8")
                res.applied.append("signal-escape-edge-pin")
            except OSError as e:
                res.errors.append(f"signal escape patch write failed ({e})")

        # The target itself can overlap an edge-straddling source even when
        # purpose-layer metadata prevents the broader perimeter-strip test.
        overlap_text = signal_escape.read_text(errors="ignore")
        overlap_patched, overlap_n = _signal_overlap_sub(overlap_text)
        if _SIGNAL_OVERLAP_MARKER in overlap_text:
            res.already.append("signal-escape-overlap")
        elif overlap_n == 0:
            res.errors.append(
                "signal escape overlap anchor not found "
                "(unexpected OpenRAM version)"
            )
        elif check_only:
            res.errors.append("signal-escape-overlap not applied")
        else:
            try:
                bak = signal_escape.with_suffix(".py.coresmith-bak")
                if not bak.exists():
                    bak.write_text(overlap_text, encoding="utf-8")
                signal_escape.write_text(overlap_patched, encoding="utf-8")
                res.applied.append("signal-escape-overlap")
            except OSError as e:
                res.errors.append(
                    f"signal escape overlap patch write failed ({e})"
                )

        # A valid zero-distance/abutment path can contain no newly-created
        # wire shape. Use its perimeter target instead of indexing an empty
        # new_wires list.
        zero_text = signal_escape.read_text(errors="ignore")
        zero_patched, zero_n = _signal_zero_wire_sub(zero_text)
        if _SIGNAL_ZERO_WIRE_MARKER in zero_text:
            res.already.append("signal-escape-zero-wire")
        elif zero_n == 0:
            res.errors.append(
                "signal escape zero-wire anchor not found "
                "(unexpected OpenRAM version)"
            )
        elif check_only:
            res.errors.append("signal-escape-zero-wire not applied")
        else:
            try:
                bak = signal_escape.with_suffix(".py.coresmith-bak")
                if not bak.exists():
                    bak.write_text(zero_text, encoding="utf-8")
                signal_escape.write_text(zero_patched, encoding="utf-8")
                res.applied.append("signal-escape-zero-wire")
            except OSError as e:
                res.errors.append(
                    f"signal escape zero-wire patch write failed ({e})"
                )

        rect_text = signal_escape.read_text(errors="ignore")
        rect_patched, rect_n = _signal_rect_sub(rect_text)
        if _SIGNAL_RECT_MARKER in rect_text:
            res.already.append("signal-escape-graph-rect")
        elif rect_n == 0:
            res.errors.append(
                "signal escape graph-shape anchor not found "
                "(unexpected OpenRAM version)"
            )
        elif check_only:
            res.errors.append("signal-escape-graph-rect not applied")
        else:
            try:
                bak = signal_escape.with_suffix(".py.coresmith-bak")
                if not bak.exists():
                    bak.write_text(rect_text, encoding="utf-8")
                signal_escape.write_text(rect_patched, encoding="utf-8")
                res.applied.append("signal-escape-graph-rect")
            except OSError as e:
                res.errors.append(
                    f"signal escape graph-shape patch write failed ({e})"
                )

        edge_text = signal_escape.read_text(errors="ignore")
        edge_patched, edge_n = _signal_edge_fallback_sub(edge_text)
        if _SIGNAL_EDGE_FALLBACK_MARKER in edge_text:
            res.already.append("signal-escape-edge-fallback")
        elif edge_n == 0:
            res.errors.append(
                "signal escape replacement anchor not found "
                "(unexpected OpenRAM version)"
            )
        elif check_only:
            res.errors.append("signal-escape-edge-fallback not applied")
        else:
            try:
                bak = signal_escape.with_suffix(".py.coresmith-bak")
                if not bak.exists():
                    bak.write_text(edge_text, encoding="utf-8")
                signal_escape.write_text(edge_patched, encoding="utf-8")
                res.applied.append("signal-escape-edge-fallback")
            except OSError as e:
                res.errors.append(
                    f"signal escape edge fallback patch write failed ({e})"
                )

    # (5) Keep generated M2 geometry legal at DFF-array/via boundaries ------
    sram_1bank = pkg / "compiler" / "modules" / "sram_1bank.py"
    if sram_1bank.exists():
        old = sram_1bank.read_text(errors="ignore")
        new, count = _sram_m2_drc_sub(old)
        markers = (
            _SRAM_M2_DFF_GAP_FIXED_LINE in old
            and _SRAM_M2_CLK_WIDEN_MARKER in old
        )
        repaired = (
            _SRAM_M2_DFF_GAP_FIXED_LINE in new
            and _SRAM_M2_CLK_WIDEN_MARKER in new
        )
        if markers:
            res.already.append("sram-m2-drc")
        elif not repaired or count == 0:
            res.errors.append(
                "sram_1bank M2 DRC anchors not found "
                "(unexpected OpenRAM version)"
            )
        elif check_only:
            res.errors.append("sram-m2-drc not applied")
        else:
            try:
                bak = sram_1bank.with_suffix(".py.coresmith-bak")
                if not bak.exists():
                    bak.write_text(old, encoding="utf-8")
                sram_1bank.write_text(new, encoding="utf-8")
                res.applied.append("sram-m2-drc")
            except OSError as e:
                res.errors.append(f"sram M2 DRC patch write failed ({e})")

    # (6) Match parametric-device models to pinned Magic extraction --------
    ptx_path = pkg / "compiler" / "modules" / "ptx.py"
    tech_path = pkg / "technology" / "sky130" / "tech" / "tech.py"
    if not ptx_path.exists() or not tech_path.exists():
        res.already.append("sky130-width-model-unavailable")
    else:
        ptx_old = ptx_path.read_text(errors="ignore")
        tech_old = tech_path.read_text(errors="ignore")
        ptx_new, tech_new, count = _ptx_width_model_sub(
            ptx_old, tech_old
        )
        markers = (
            _PTX_WIDTH_MODEL_MARKER in ptx_old
            and _SKY130_WIDTH_MODEL_MARKER in tech_old
        )
        if markers:
            res.already.append("sky130-width-model")
        elif count != 2:
            res.errors.append(
                "sky130 width-model anchors not found "
                "(unexpected OpenRAM version)"
            )
        elif check_only:
            res.errors.append("sky130-width-model not applied")
        else:
            try:
                for path, old, new in (
                    (ptx_path, ptx_old, ptx_new),
                    (tech_path, tech_old, tech_new),
                ):
                    bak = path.with_suffix(".py.coresmith-bak")
                    if not bak.exists():
                        bak.write_text(old, encoding="utf-8")
                    path.write_text(new, encoding="utf-8")
                res.applied.append("sky130-width-model")
            except OSError as e:
                res.errors.append(
                    f"sky130 width-model patch write failed ({e})"
                )

    # (7) Synchronize stale dual-port hard-cell schematics to the PDK -------
    # This is deliberately conditional on an explicit PDK_ROOT. General
    # OpenRAM setup can run before a PDK is selected; a pinned physical run
    # always supplies this variable and therefore gets strict validation.
    pdk_root_text = os.environ.get("PDK_ROOT")
    if pdk_root_text:
        canonical_path = Path(pdk_root_text) / _PDK_DP_SPICE_REL
        sp_lib_dir = pkg / "technology" / "sky130" / "sp_lib"
        if not canonical_path.is_file():
            res.errors.append(
                f"pinned PDK dual-port SPICE source missing: {canonical_path}"
            )
        elif not sp_lib_dir.is_dir():
            res.errors.append("OpenRAM sky130 sp_lib missing")
        else:
            canonical_source = canonical_path.read_text(errors="ignore")
            pending: list[tuple[Path, str, str]] = []
            missing: list[str] = []
            for name, canonical_name in _PDK_HARDCELL_MAP.items():
                path = sp_lib_dir / f"{name}.sp"
                if not path.is_file():
                    missing.append(name)
                    continue
                old = path.read_text(errors="ignore")
                bak = path.with_suffix(".sp.coresmith-bak")
                signature_source = (
                    bak.read_text(errors="ignore") if bak.is_file() else old
                )
                new, count = _pdk_dp_spice_sub(
                    old,
                    canonical_source,
                    name,
                    signature_source,
                    canonical_name,
                )
                if count:
                    pending.append((path, old, new))
                elif _DP_SPICE_MARKER not in old:
                    missing.append(name)
            if missing:
                res.errors.append(
                    "hard-cell SPICE sync failed for: " + ", ".join(missing)
                )
            elif not pending:
                res.already.append("pdk-hardcell-spice")
            elif check_only:
                res.errors.append(
                    f"{len(pending)} hard-cell SPICE view(s) stale"
                )
            else:
                try:
                    for path, old, new in pending:
                        bak = path.with_suffix(".sp.coresmith-bak")
                        if not bak.exists():
                            bak.write_text(old, encoding="utf-8")
                        path.write_text(new, encoding="utf-8")
                    res.applied.append("pdk-hardcell-spice")
                except OSError as e:
                    res.errors.append(
                        f"hard-cell SPICE sync failed ({e})"
                    )

    # (8) Magic blackbox files use the extension verifier requests --------
    # The 1.2.48 wheel ships `maglef_lib/*.maglef`, while magic.py explicitly
    # requires/copies `maglef_lib/<cell>.mag`. The file contents are already
    # valid Magic abstracts; install extension aliases for every shipped cell.
    maglef_dir = pkg / "technology" / "sky130" / "maglef_lib"
    if not maglef_dir.exists():
        res.already.append("maglef-alias-unavailable")
    else:
        sources = sorted(maglef_dir.glob("*.maglef"))
        missing = [
            (source, source.with_suffix(".mag"))
            for source in sources
            if not source.with_suffix(".mag").exists()
        ]
        if not missing:
            res.already.append("maglef-mag-alias")
        elif check_only:
            res.errors.append(
                f"{len(missing)} Magic blackbox .mag alias(es) missing"
            )
        else:
            try:
                for source, target in missing:
                    target.write_bytes(source.read_bytes())
                res.applied.append("maglef-mag-alias")
            except OSError as e:
                res.errors.append(f"Magic blackbox alias write failed ({e})")

    # (9) Make sky130 hard-cell SPICE W/L units explicit for Netgen --------
    sp_lib_dir = pkg / "technology" / "sky130" / "sp_lib"
    if not sp_lib_dir.exists():
        res.already.append("spice-micron-units-unavailable")
    else:
        sp_files = sorted(sp_lib_dir.glob("*.sp"))
        pending: list[tuple[Path, str, str]] = []
        for path in sp_files:
            old = path.read_text(errors="ignore")
            new, count = _spice_micron_units_sub(old)
            if count:
                pending.append((path, old, new))
        if not pending:
            res.already.append("spice-micron-units")
        elif check_only:
            res.errors.append(
                f"{len(pending)} hard-cell SPICE view(s) have bare W/L units"
            )
        else:
            try:
                for path, old, new in pending:
                    bak = path.with_suffix(".sp.coresmith-bak")
                    if not bak.exists():
                        bak.write_text(old, encoding="utf-8")
                    path.write_text(new, encoding="utf-8")
                res.applied.append("spice-micron-units")
            except OSError as e:
                res.errors.append(f"SPICE micron-unit patch failed ({e})")

    # Runnable iff __main__ importable now.
    try:
        importlib.invalidate_caches()
        res.ok = importlib.util.find_spec("openram.__main__") is not None
    except Exception:
        res.ok = False
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Idempotently repair pip OpenRAM")
    ap.add_argument("--check", action="store_true",
                    help="report status without writing")
    args = ap.parse_args(argv)
    r = patch_openram(check_only=args.check)
    print(r.summary())
    return 0 if r.ok else 1


if __name__ == "__main__":
    sys.exit(main())
