# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Wide-macro intra-macro pin-short guard (LVS wide-macro-pins fix).

Root cause proven on E6: a wide OpenRAM sky130 SRAM macro can ship two DISTINCT
pins whose PORT metal OVERLAPS on the same layer in the LEF abstract (and, for
these macros, in the real GDS the abstract mirrors) -- a genuine galvanic short
between two independently-driven nets that no honest LVS can pass. This suite
uses only synthetic, geometry-named macro fixtures (no design/benchmark names):
a "wide" macro with an overlapping din0/addr1 pin pair, plus clean macros.

Both branches of the env gate ``CORESMITH_MACRO_PIN_SHORT_GUARD`` are tested.
"""
from __future__ import annotations

import pytest

from orchestrator.langgraph import macro_registry as mr
from orchestrator.langgraph import openram_gen as og

_GUARD = "CORESMITH_MACRO_PIN_SHORT_GUARD"

# A pin block with a single met4 PORT rect at [x0, x0+0.38] x [0, 0.38].
_PIN = """\
  PIN {pin}
    DIRECTION {dir} ;
    PORT
      LAYER {layer} ;
      RECT  {x0} 0.0 {x1} 0.38 ;
    END
  END {pin}
"""

_HEAD = """\
MACRO {name}
  CLASS BLOCK ;
  SIZE {w} BY {h} ;
  PIN vccd1
    DIRECTION INOUT ;
    USE POWER ;
  END vccd1
  PIN vssd1
    DIRECTION INOUT ;
    USE GROUND ;
  END vssd1
"""


def _pin(pin, x0, *, layer="met4", dir="INPUT"):
    return _PIN.format(pin=pin, dir=dir, layer=layer, x0=f"{x0:.2f}", x1=f"{x0 + 0.38:.2f}")


def _lef_clean(name, w=200.0, h=100.0):
    """din0[0] and addr1[0] well-separated -> no short."""
    return (
        _HEAD.format(name=name, w=w, h=h)
        + _pin("din0[0]", 10.00)
        + _pin("addr1[0]", 40.00)
        + f"END {name}\n"
    )


def _lef_shorted(name, w=200.0, h=100.0):
    """din0[50] [441.72,442.10] and addr1[4] [441.64,442.02] overlap on met4 --
    the exact defect class found in the real sram_1rw1r_64_32_8 macro."""
    return (
        _HEAD.format(name=name, w=w, h=h)
        + _pin("din0[50]", 441.72)
        + _pin("addr1[4]", 441.64)
        + f"END {name}\n"
    )


def _make_pdk(tmp_path, lef_bodies, variant="sky130B"):
    """lef_bodies: {macro_name: lef_text}. Writes a full synthetic PDK tree."""
    sram = tmp_path / variant / "libs.ref" / "sky130_sram_macros"
    for sub in ("lef", "gds", "lib", "spice", "verilog"):
        (sram / sub).mkdir(parents=True, exist_ok=True)
    for name, body in lef_bodies.items():
        (sram / "lef" / f"{name}.lef").write_text(body)
        (sram / "gds" / f"{name}.gds").write_text("stub")
        (sram / "lib" / f"{name}_TT_1p8V_25C.lib").write_text("stub")
        (sram / "spice" / f"{name}.spice").write_text("stub")
        (sram / "verilog" / f"{name}.v").write_text(f"module {name}(); endmodule")
    return str(tmp_path)


@pytest.fixture(autouse=True)
def _clear_caches():
    mr.discover_macros.cache_clear()
    yield
    mr.discover_macros.cache_clear()


# --------------------------------------------------------------------------
# lef_pin_shorts: the pure geometric detector
# --------------------------------------------------------------------------
class TestLefPinShorts:
    def test_detects_overlapping_distinct_pins(self):
        shorts = mr.lef_pin_shorts(_lef_shorted("sram_1rw1r_8_4_8_sky130"))
        assert shorts == (("addr1[4]", "din0[50]"),)

    def test_clean_macro_has_no_short(self):
        assert mr.lef_pin_shorts(_lef_clean("sram_1rw1r_4_4_8_sky130")) == ()

    def test_same_pin_multi_rect_is_not_a_short(self):
        # One pin drawn as two overlapping stripes is NOT an inter-pin short.
        body = (
            _HEAD.format(name="m", w=50.0, h=50.0)
            + "  PIN din0[0]\n    DIRECTION INPUT ;\n    PORT\n"
            + "      LAYER met4 ;\n      RECT 10.00 0.0 10.40 0.38 ;\n"
            + "      RECT 10.20 0.0 10.60 15.0 ;\n    END\n  END din0[0]\n"
            + _pin("addr1[0]", 40.00)
            + "END m\n"
        )
        assert mr.lef_pin_shorts(body) == ()

    def test_different_layers_do_not_short(self):
        body = (
            _HEAD.format(name="m", w=50.0, h=50.0)
            + _pin("din0[0]", 10.00, layer="met4")
            + _pin("addr1[0]", 10.00, layer="met3")  # same x, DIFFERENT layer
            + "END m\n"
        )
        assert mr.lef_pin_shorts(body) == ()

    def test_obstruction_metal_is_ignored(self):
        body = (
            _HEAD.format(name="m", w=50.0, h=50.0)
            + _pin("din0[0]", 10.00)
            + "  OBS\n    LAYER met4 ;\n    RECT 10.10 0.0 10.50 0.38 ;\n  END\n"
            + "END m\n"
        )
        assert mr.lef_pin_shorts(body) == ()

    def test_accepts_a_path(self, tmp_path):
        p = tmp_path / "m.lef"
        p.write_text(_lef_shorted("sram_1rw1r_8_4_8_sky130"))
        assert mr.lef_pin_shorts(p) == (("addr1[4]", "din0[50]"),)


# --------------------------------------------------------------------------
# discovery populates the flag
# --------------------------------------------------------------------------
class TestDiscoveryFlag:
    def test_shorted_macro_flagged_clean_macro_not(self, tmp_path):
        root = _make_pdk(tmp_path, {
            "sram_1rw1r_8_4_8_sky130": _lef_shorted("sram_1rw1r_8_4_8_sky130"),
            "sram_1rw1r_4_4_8_sky130": _lef_clean("sram_1rw1r_4_4_8_sky130"),
        })
        reg = mr.discover_macros(root)
        assert reg["sram_1rw1r_8_4_8_sky130"].pin_shorts == (("addr1[4]", "din0[50]"),)
        assert reg["sram_1rw1r_8_4_8_sky130"].lvs_clean_pins is False
        assert reg["sram_1rw1r_4_4_8_sky130"].pin_shorts == ()
        assert reg["sram_1rw1r_4_4_8_sky130"].lvs_clean_pins is True


# --------------------------------------------------------------------------
# selection: both branches of the env gate
# --------------------------------------------------------------------------
class TestSelectionGuard:
    def _reg(self, tmp_path):
        root = _make_pdk(tmp_path, {
            # exact 4-words x 8-bits match, but SHORTED
            "sram_1rw1r_8_4_8_sky130": _lef_shorted("sram_1rw1r_8_4_8_sky130"),
            # clean half-width macro -> composes the 8-bit width as 2x4
            "sram_1rw1r_4_4_8_sky130": _lef_clean("sram_1rw1r_4_4_8_sky130"),
        })
        return mr.discover_macros(root)

    def test_guard_on_excludes_shorted_from_find_exact(self, tmp_path, monkeypatch):
        monkeypatch.delenv(_GUARD, raising=False)  # default ON
        reg = self._reg(tmp_path)
        assert og.find_exact(words=4, data_bits=8, registry=reg) is None

    def test_guard_off_returns_shorted_from_find_exact(self, tmp_path, monkeypatch):
        monkeypatch.setenv(_GUARD, "0")
        reg = self._reg(tmp_path)
        m = og.find_exact(words=4, data_bits=8, registry=reg)
        assert m is not None and m.name == "sram_1rw1r_8_4_8_sky130"

    def test_guard_on_resolves_to_clean_composition(self, tmp_path, monkeypatch):
        monkeypatch.delenv(_GUARD, raising=False)
        # Tiling is OFF by default, so the composition fallback needs the
        # explicit opt-in; the default path is asserted separately below.
        monkeypatch.setenv("CORESMITH_ALLOW_MACRO_TILING", "1")
        reg = self._reg(tmp_path)
        # allow_generate=False so we exercise the prebuilt fallback deterministically
        res = og.ensure_macro(words=4, data_bits=8, allow_generate=False, registry=reg)
        assert isinstance(res, og.CompositionPlan)
        assert res.base.name == "sram_1rw1r_4_4_8_sky130"
        assert res.base.lvs_clean_pins is True

    def test_guard_on_escalates_when_tiling_disabled(self, tmp_path, monkeypatch):
        """Default path: a shorted exact macro is still never selected.

        With tiling off there is no composition to fall back to, so the
        resolution escalates (None) rather than placing the shorted part.
        """
        monkeypatch.delenv(_GUARD, raising=False)
        monkeypatch.delenv("CORESMITH_ALLOW_MACRO_TILING", raising=False)
        reg = self._reg(tmp_path)
        res = og.ensure_macro(words=4, data_bits=8, allow_generate=False, registry=reg)
        assert res is None

    def test_guard_off_resolves_to_shorted_exact(self, tmp_path, monkeypatch):
        monkeypatch.setenv(_GUARD, "0")
        reg = self._reg(tmp_path)
        res = og.ensure_macro(words=4, data_bits=8, allow_generate=False, registry=reg)
        assert isinstance(res, mr.MacroInfo)
        assert res.name == "sram_1rw1r_8_4_8_sky130"

    def test_clean_exact_macro_still_selected_with_guard_on(self, tmp_path, monkeypatch):
        monkeypatch.delenv(_GUARD, raising=False)
        root = _make_pdk(tmp_path, {
            "sram_1rw1r_4_4_8_sky130": _lef_clean("sram_1rw1r_4_4_8_sky130"),
        })
        reg = mr.discover_macros(root)
        m = og.find_exact(words=4, data_bits=4, registry=reg)
        assert m is not None and m.name == "sram_1rw1r_4_4_8_sky130"


# --------------------------------------------------------------------------
# menu: don't advertise a macro we will refuse to place
# --------------------------------------------------------------------------
class TestMenu:
    def test_menu_hides_shorted_macro_when_guard_on(self, tmp_path, monkeypatch):
        monkeypatch.delenv(_GUARD, raising=False)
        reg = mr.discover_macros(_make_pdk(tmp_path, {
            "sram_1rw1r_8_4_8_sky130": _lef_shorted("sram_1rw1r_8_4_8_sky130"),
            "sram_1rw1r_4_4_8_sky130": _lef_clean("sram_1rw1r_4_4_8_sky130"),
        }))
        menu = mr.macro_menu_markdown(reg)
        assert "sram_1rw1r_8_4_8_sky130" not in menu
        assert "sram_1rw1r_4_4_8_sky130" in menu

    def test_menu_shows_shorted_macro_when_guard_off(self, tmp_path, monkeypatch):
        monkeypatch.setenv(_GUARD, "0")
        reg = mr.discover_macros(_make_pdk(tmp_path, {
            "sram_1rw1r_8_4_8_sky130": _lef_shorted("sram_1rw1r_8_4_8_sky130"),
        }))
        menu = mr.macro_menu_markdown(reg)
        assert "sram_1rw1r_8_4_8_sky130" in menu
