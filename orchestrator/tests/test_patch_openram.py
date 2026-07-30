# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""scripts/patch_openram.py: idempotent self-heal of a pip OpenRAM wheel."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import patch_openram as po

_BROKEN_BBOX = (
    "class hierarchy_layout:\n"
    "    def get_bbox(self):\n"
    "        ll = vector(boundary[0][0], boundary[0][1])\n"
    "        ur = vector(boundary[1][0], boundary[1][1])\n"
    "        return ll, ur\n"
    "    def route_horizontal_pins(self, name, new_name=None):\n"
    "        pin_name = new_name if new_name is not None else name\n"
    "        for inst,pin in v:\n"
    "            self.add_layout_pin(pin.name,\n"
    "                                pin.layer, pin.ll(),\n"
    "                                pin.width(), pin.height())\n"
    "    def route_vertical_pins(self, name):\n"
    "        self.add_layout_pin(pin.name, pin.layer, pin.ll())\n"
)
_BROKEN_ROM_DECODER = (
    "from math import ceil, log\n"
    "class rom_decoder:\n"
    "    def f(self, num_outputs):\n"
    "        self.num_inputs = ceil(log(num_outputs, 2))\n"
)
_BROKEN_ROM_BANK = (
    "class rom_bank:\n"
    "    def create_instances(self):\n"
    '        addr_lsb = ["addr0[{}]".format(addr) for addr in range(self.col_bits)]\n'
    "    def route_decode_outputs(self):\n"
    "        self.connect_row_pins(self.wordline_layer, sel_pins, round=True)\n"
)
_BROKEN_SIGNAL_ESCAPE = (
    "class signal_escape_router:\n"
    "    def route(self, pin_names):\n"
    "        routed_count = 0\n"
    "        for source, target, _ in self.get_route_pairs(pin_names):\n"
    "            # Change fake pin's name so the graph will treat it as routable\n"
    "            target.name = source.name\n"
    "            # Create the graph\n"
    "            graph = object()\n"
    "            self.new_pins[source.name] = new_wires[-1]\n"
    "    def replace_layout_pins(self):\n"
    "        for name, pin in self.new_pins.items():\n"
    "            pin = graph_shape(pin.name, pin.boundary, pin.lpp)\n"
    "            edge = None\n"
    "            for fake in self.fake_pins:\n"
    "                edge = pin.intersection(fake)\n"
    "                if edge:\n"
    "                    break\n"
    "            self.design.replace_layout_pin(name, edge)\n"
)
_BROKEN_SRAM_1BANK = (
    "class sram_1bank:\n"
    "    def place_dffs(self):\n"
    "        if self.col_addr_dff:\n"
    "            self.col_addr_dff_insts[port].place(self.col_addr_pos[port])\n"
    "            x_offset = self.col_addr_dff_insts[port].rx()\n"
    "        else:\n"
    "            pass\n"
    "    def route_clk(self):\n"
    "        for port in self.all_ports:\n"
    "            if self.col_addr_dff:\n"
    "                mid_pos = vector(clk_steiner_pos.x, dff_clk_pos.y)\n"
    "                self.add_wire(self.m2_stack[::-1],\n"
    "                              [dff_clk_pos, mid_pos, clk_steiner_pos])\n"
)
_BROKEN_PTX = (
    "class ptx:\n"
    "    def create_netlist(self):\n"
    "        self.add_pin_list(pin_list, dir_list)\n\n"
    "        # Just make a guess since these will actually\n"
    "        main_str = 'a'.format(spice[self.tx_type], x)\n"
    "        main_str = 'b'.format(spice[self.tx_type], x)\n"
    "        # old = 'c'.format(spice[self.tx_type], x)\n"
    "        self.lvs_device = 'd'.format(spice[self.tx_type], x)\n"
)
_BROKEN_SKY130_TECH = (
    'spice = {}\n'
    'spice["nmos"] = "sky130_fd_pr__nfet_01v8"\n'
    'spice["pmos"] = "sky130_fd_pr__pfet_01v8"\n'
)


def _fake_openram(tmp_path: Path, *, with_main: bool, broken_bbox: bool) -> Path:
    pkg = tmp_path / "openram"
    (pkg / "compiler" / "base").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "sram_compiler.py").write_text("print('compile')\n")
    if with_main:
        (pkg / "__main__.py").write_text("# stock\n")
    hl = pkg / "compiler" / "base" / "hierarchy_layout.py"
    fixed_hl = po._bbox_sub(_BROKEN_BBOX)[0]
    fixed_hl = po._horizontal_pin_rename_sub(fixed_hl)[0]
    hl.write_text(_BROKEN_BBOX if broken_bbox else fixed_hl)
    modules = pkg / "compiler" / "modules"
    modules.mkdir()
    decoder, bank, _ = po._rom_nomux_sub(
        _BROKEN_ROM_DECODER, _BROKEN_ROM_BANK
    )
    (modules / "rom_decoder.py").write_text(
        _BROKEN_ROM_DECODER if broken_bbox else decoder
    )
    (modules / "rom_bank.py").write_text(
        _BROKEN_ROM_BANK if broken_bbox else bank
    )
    fixed_sram = po._sram_m2_drc_sub(_BROKEN_SRAM_1BANK)[0]
    (modules / "sram_1bank.py").write_text(
        _BROKEN_SRAM_1BANK if broken_bbox else fixed_sram
    )
    fixed_ptx, fixed_tech, _ = po._ptx_width_model_sub(
        _BROKEN_PTX, _BROKEN_SKY130_TECH
    )
    (modules / "ptx.py").write_text(
        _BROKEN_PTX if broken_bbox else fixed_ptx
    )
    tech_dir = pkg / "technology" / "sky130" / "tech"
    tech_dir.mkdir(parents=True)
    (tech_dir / "tech.py").write_text(
        _BROKEN_SKY130_TECH if broken_bbox else fixed_tech
    )
    router = pkg / "compiler" / "router"
    router.mkdir()
    fixed_escape = po._signal_escape_sub(_BROKEN_SIGNAL_ESCAPE)[0]
    fixed_escape = po._signal_overlap_sub(fixed_escape)[0]
    fixed_escape = po._signal_zero_wire_sub(fixed_escape)[0]
    fixed_escape = po._signal_rect_sub(fixed_escape)[0]
    fixed_escape = po._signal_edge_fallback_sub(fixed_escape)[0]
    (router / "signal_escape_router.py").write_text(
        _BROKEN_SIGNAL_ESCAPE if broken_bbox else fixed_escape
    )
    maglef = pkg / "technology" / "sky130" / "maglef_lib"
    maglef.mkdir(parents=True)
    for name in ("cell_a", "cell_b"):
        (maglef / f"{name}.maglef").write_text(f"magic\\n{name}\\n")
        if not broken_bbox:
            (maglef / f"{name}.mag").write_text(f"magic\\n{name}\\n")
    sp_lib = pkg / "technology" / "sky130" / "sp_lib"
    sp_lib.mkdir()
    spice = (
        ".subckt hard A Z VDD GND\n"
        "X0 Z A VDD VDD sky130_fd_pr__pfet_01v8 W=1.12 L=0.15\n"
        ".ends\n"
    )
    if not broken_bbox:
        spice = po._spice_micron_units_sub(spice)[0]
    (sp_lib / "hard.sp").write_text(spice)
    return pkg


def _point_at(monkeypatch, pkg: Path):
    real = importlib.util.find_spec

    def fake(name, *a, **k):
        if name == "openram":
            class _S:
                origin = str(pkg / "__init__.py")
            return _S()
        if name == "openram.__main__":
            return object() if (pkg / "__main__.py").exists() else None
        return real(name, *a, **k)
    monkeypatch.setattr(po.importlib.util, "find_spec", fake)


def test_applies_both_fixes(tmp_path, monkeypatch):
    pkg = _fake_openram(tmp_path, with_main=False, broken_bbox=True)
    _point_at(monkeypatch, pkg)
    r = po.patch_openram()
    assert r.ok is True
    assert set(r.applied) == {
        "__main__.py", "bbox-item", "horizontal-pin-rename", "rom-nomux",
        "signal-escape-edge-pin", "signal-escape-overlap",
        "signal-escape-zero-wire", "signal-escape-graph-rect",
        "signal-escape-edge-fallback",
        "sram-m2-drc",
        "sky130-width-model",
        "maglef-mag-alias",
        "spice-micron-units",
    }
    assert po._MAIN_MARKER in (pkg / "__main__.py").read_text()
    hl = (pkg / "compiler" / "base" / "hierarchy_layout.py").read_text()
    assert ".item()" in hl
    assert po._HORIZONTAL_RENAME_MARKER in hl
    # The vertical helper has no new_name argument and is untouched.
    assert "route_vertical_pins" in hl
    assert "self.add_layout_pin(pin.name, pin.layer, pin.ll())" in hl
    assert po._ROM_NOMUX_MARKER in (
        pkg / "compiler" / "modules" / "rom_decoder.py"
    ).read_text()
    assert po._ROM_NOMUX_MARKER in (
        pkg / "compiler" / "modules" / "rom_bank.py"
    ).read_text()
    assert po._SIGNAL_ESCAPE_MARKER in (
        pkg / "compiler" / "router" / "signal_escape_router.py"
    ).read_text()
    assert po._SIGNAL_OVERLAP_MARKER in (
        pkg / "compiler" / "router" / "signal_escape_router.py"
    ).read_text()
    assert po._SIGNAL_ZERO_WIRE_MARKER in (
        pkg / "compiler" / "router" / "signal_escape_router.py"
    ).read_text()
    assert po._SIGNAL_RECT_MARKER in (
        pkg / "compiler" / "router" / "signal_escape_router.py"
    ).read_text()
    assert po._SIGNAL_EDGE_FALLBACK_MARKER in (
        pkg / "compiler" / "router" / "signal_escape_router.py"
    ).read_text()
    sram = (pkg / "compiler" / "modules" / "sram_1bank.py").read_text()
    assert po._SRAM_M2_DFF_GAP_MARKER in sram
    assert po._SRAM_M2_CLK_WIDEN_MARKER in sram
    ptx = (pkg / "compiler" / "modules" / "ptx.py").read_text()
    tech = (pkg / "technology" / "sky130" / "tech" / "tech.py").read_text()
    assert po._PTX_WIDTH_MODEL_MARKER in ptx
    assert ".format(tx_model," in ptx
    assert po._SKY130_WIDTH_MODEL_MARKER in tech
    assert 'spice["nmos_model_width_limit"] = 0.42' in tech
    assert (
        pkg / "technology" / "sky130" / "maglef_lib" / "cell_a.mag"
    ).read_text() == "magic\\ncell_a\\n"
    assert "W=1.12u L=0.15u" in (
        pkg / "technology" / "sky130" / "sp_lib" / "hard.sp"
    ).read_text()
    # backup written
    assert (pkg / "compiler" / "base"
            / "hierarchy_layout.py.coresmith-bak").exists()


def test_patches_regardless_of_indentation(tmp_path, monkeypatch):
    # regression: the real openram file indents the bbox lines 12 spaces, not
    # 8. An exact-string matcher missed it; the regex matcher must not.
    pkg = _fake_openram(tmp_path, with_main=False, broken_bbox=True)
    hl = pkg / "compiler" / "base" / "hierarchy_layout.py"
    hl.write_text(
        "class hierarchy_layout:\n"
        "    def get_bbox(self):\n"
        "            ll = vector(boundary[0][0], boundary[0][1])\n"
        "            ur = vector(boundary[1][0], boundary[1][1])\n"
        "            return ll, ur\n")
    _point_at(monkeypatch, pkg)
    r = po.patch_openram()
    assert "bbox-item" in r.applied
    txt = hl.read_text()
    assert "vector(boundary[0][0].item(), boundary[0][1].item())" in txt
    assert "vector(boundary[1][0].item(), boundary[1][1].item())" in txt
    # idempotent: a second pass makes no further change
    r2 = po.patch_openram()
    assert "bbox-item" in r2.already


def test_idempotent_second_run_is_noop(tmp_path, monkeypatch):
    pkg = _fake_openram(tmp_path, with_main=False, broken_bbox=True)
    _point_at(monkeypatch, pkg)
    po.patch_openram()
    r2 = po.patch_openram()
    assert r2.ok is True
    assert r2.applied == []
    assert set(r2.already) == {
        "__main__.py", "bbox-item", "horizontal-pin-rename", "rom-nomux",
        "signal-escape-edge-pin", "signal-escape-overlap",
        "signal-escape-zero-wire", "signal-escape-graph-rect",
        "signal-escape-edge-fallback",
        "sram-m2-drc",
        "sky130-width-model",
        "maglef-mag-alias",
        "spice-micron-units",
        # repair (10): only freepdk45 ships nand4_leakage; gf180mcu and
        # scn3me_subm still omit it (sky130 was fixed upstream in 3f1f580, but
        # the pinned 1.2.48 wheel predates that). This fixture package carries
        # no technology tech.py at all, so the repair reports it had nothing to
        # act on instead of claiming a patch it never made -- which is why the
        # name here is the -unavailable variant and why r.applied (asserted in
        # test_applies_both_fixes) is deliberately unchanged.
        "nand4-leakage-unavailable",
    }


def test_stock_working_wheel_is_left_alone(tmp_path, monkeypatch):
    # a hypothetical future wheel that already ships __main__ + fixed bbox
    pkg = _fake_openram(tmp_path, with_main=True, broken_bbox=False)
    _point_at(monkeypatch, pkg)
    r = po.patch_openram()
    assert r.ok is True
    assert r.applied == []
    assert "__main__.py" in r.already and "bbox-item" in r.already


def test_check_only_never_writes(tmp_path, monkeypatch):
    pkg = _fake_openram(tmp_path, with_main=False, broken_bbox=True)
    _point_at(monkeypatch, pkg)
    r = po.patch_openram(check_only=True)
    assert r.applied == []
    assert not (pkg / "__main__.py").exists()
    assert any("missing" in e for e in r.errors)


def test_missing_openram_reports_error(monkeypatch):
    monkeypatch.setattr(po, "_openram_pkg_dir", lambda: None)
    r = po.patch_openram()
    assert r.ok is False and any("not importable" in e for e in r.errors)


def test_unknown_bbox_pattern_not_guessed(tmp_path, monkeypatch):
    pkg = _fake_openram(tmp_path, with_main=False, broken_bbox=True)
    hl = pkg / "compiler" / "base" / "hierarchy_layout.py"
    hl.write_text("class x:\n    pass\n")  # neither broken nor fixed pattern
    _point_at(monkeypatch, pkg)
    r = po.patch_openram()
    assert any("bbox lines not found" in e for e in r.errors)


def test_pdk_dp_spice_sync_uses_canonical_devices():
    old = (
        "* license\n\n"
        ".SUBCKT sky130_fd_bd_sram__openram_dp_cell BL0 BR0 BL1 BR1 "
        "WL0 WL1 VDD GND\n"
        "X8 VDD Q QB VDD sky130_fd_pr__special_pfet_pass W=0.14u L=0.15u\n"
        ".ENDS\n"
    )
    canonical = (
        ".SUBCKT sky130_fd_bd_sram__openram_dp_cell bl0 br0 bl1 br1 "
        "wl0 wl1 vdd gnd\n"
        "X8 vdd Q QB vdd sky130_fd_pr__special_pfet_latch W=0.14 L=0.15\n"
        "X10 QB wl1 QB vdd sky130_fd_pr__special_pfet_pass L=0.08 W=0.14\n"
        ".ENDS\n"
    )
    new, count = po._pdk_dp_spice_sub(
        old, canonical, "sky130_fd_bd_sram__openram_dp_cell"
    )
    assert count == 1
    assert po._DP_SPICE_MARKER in new
    assert (
        ".SUBCKT sky130_fd_bd_sram__openram_dp_cell BL0 BR0 BL1 BR1 "
        "WL0 WL1 VDD GND"
    ) in new
    assert "special_pfet_latch W=0.14u L=0.15u" in new
    assert "special_pfet_pass" not in new
    assert "special_pfet_latch L=0.08u W=0.14u" in new
    assert po._pdk_dp_spice_sub(
        new, canonical, "sky130_fd_bd_sram__openram_dp_cell"
    ) == (new, 0)


def test_sram_m2_patch_migrates_track_only_fix_to_nwell_spacing():
    old = po._sram_m2_drc_sub(_BROKEN_SRAM_1BANK)[0]
    track_only = old.replace(
        "self.m2_pitch + self.nwell_space", "self.m2_pitch"
    )
    new, count = po._sram_m2_drc_sub(track_only)
    assert count == 1
    assert po._SRAM_M2_DFF_GAP_FIXED_LINE in new
    assert po._SRAM_M2_DFF_GAP_TRACK_ONLY_LINE not in new


def test_ensure_openram_patched_reaches_patcher(tmp_path, monkeypatch):
    """Regression: ensure_openram_patched used sys.path but sys was not a
    module-level import in openram_gen -> NameError swallowed as False. A
    fake scripts/patch_openram on the resolved path must be reached and its
    .ok returned (proves the sys.path line executes)."""
    import orchestrator.langgraph.openram_gen as og
    fake_scripts = tmp_path / "scripts"
    fake_scripts.mkdir()
    (fake_scripts / "patch_openram.py").write_text(
        "class _R:\n    ok = True\n"
        "def patch_openram(*a, **k):\n    return _R()\n")
    monkeypatch.setattr(og, "__file__",
                        str(tmp_path / "orchestrator" / "langgraph" / "x.py"))
    import sys as _sys
    monkeypatch.setitem(_sys.modules, "patch_openram", None)
    _sys.modules.pop("patch_openram", None)
    assert og.ensure_openram_patched() is True
