# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the SRAM-macro registry (5th fix: backend macro discovery).

Hermetic -- builds a synthetic PDK tree so it runs without volare/EDA tools.
"""
from __future__ import annotations

from orchestrator.langgraph import macro_registry as mr

_LEF = """\
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
  PIN clk0
    DIRECTION INPUT ;
  END clk0
END {name}
"""


def _make_pdk(tmp_path, names, variant="sky130B", w=479.78, h=397.5):
    sram = tmp_path / variant / "libs.ref" / "sky130_sram_macros"
    for sub in ("lef", "gds", "lib", "spice", "verilog"):
        (sram / sub).mkdir(parents=True, exist_ok=True)
    for name in names:
        (sram / "lef" / f"{name}.lef").write_text(_LEF.format(name=name, w=w, h=h))
        (sram / "gds" / f"{name}.gds").write_text("stub")
        (sram / "lib" / f"{name}_TT_1p8V_25C.lib").write_text("stub")
        (sram / "spice" / f"{name}.spice").write_text("stub")
        (sram / "verilog" / f"{name}.v").write_text(f"module {name}(); endmodule")
    return tmp_path


class TestDiscovery:
    def test_discovers_and_parses_efabless_name(self, tmp_path):
        _make_pdk(tmp_path, ["sky130_sram_1kbyte_1rw1r_32x256_8"])
        reg = mr.discover_macros(str(tmp_path))
        assert "sky130_sram_1kbyte_1rw1r_32x256_8" in reg
        m = reg["sky130_sram_1kbyte_1rw1r_32x256_8"]
        assert m.words == 256 and m.data_bits == 32
        assert m.bits == 256 * 32 and abs(m.kib - 1.0) < 1e-6
        assert m.ports == "1rw1r" and m.mask_bits == 8
        assert m.width_um == 479.78 and m.height_um == 397.5
        assert m.power_pin == "vccd1" and m.ground_pin == "vssd1"
        assert m.collateral_complete()

    def test_parses_openram_raw_name(self, tmp_path):
        _make_pdk(tmp_path, ["sram_1rw1r_32_256_8_sky130"])
        reg = mr.discover_macros(str(tmp_path))
        m = reg["sram_1rw1r_32_256_8_sky130"]
        assert m.words == 256 and m.data_bits == 32 and m.ports == "1rw1r"

    def test_absent_pdk_returns_empty(self, tmp_path):
        assert mr.discover_macros(str(tmp_path / "nope")) == {}

    def test_incomplete_collateral_flagged(self, tmp_path):
        _make_pdk(tmp_path, ["sky130_sram_2kbyte_1rw1r_32x512_8"])
        # delete the gds -> not complete
        name = "sky130_sram_2kbyte_1rw1r_32x512_8"
        (tmp_path / "sky130B" / "libs.ref" / "sky130_sram_macros" / "gds"
         / f"{name}.gds").unlink()
        mr.discover_macros.cache_clear()
        reg = mr.discover_macros(str(tmp_path))
        assert not reg[name].collateral_complete()


class TestDetection:
    def test_detects_instantiated_macro(self, tmp_path):
        _make_pdk(tmp_path, [
            "sky130_sram_1kbyte_1rw1r_32x256_8",
            "sky130_sram_2kbyte_1rw1r_32x512_8",
        ])
        reg = mr.discover_macros(str(tmp_path))
        netlist = tmp_path / "net.v"
        netlist.write_text(
            "module top();\n"
            "  sky130_sram_1kbyte_1rw1r_32x256_8 u_mem (.clk0(c));\n"
            "endmodule\n"
        )
        found = mr.detect_instantiated_macros(str(netlist), reg)
        assert [m.name for m in found] == ["sky130_sram_1kbyte_1rw1r_32x256_8"]

    def test_no_false_substring_match(self, tmp_path):
        _make_pdk(tmp_path, ["sky130_sram_1kbyte_1rw1r_32x256_8"])
        reg = mr.discover_macros(str(tmp_path))
        netlist = tmp_path / "net.v"
        # name appears only as a substring of a longer identifier
        netlist.write_text("wire my_sky130_sram_1kbyte_1rw1r_32x256_8_x;\n")
        assert mr.detect_instantiated_macros(str(netlist), reg) == []


class TestMenu:
    def test_menu_lists_macros_sorted_by_size(self, tmp_path):
        _make_pdk(tmp_path, [
            "sky130_sram_2kbyte_1rw1r_32x512_8",
            "sky130_sram_1kbyte_1rw1r_32x256_8",
        ])
        reg = mr.discover_macros(str(tmp_path))
        md = mr.macro_menu_markdown(reg)
        assert "sky130_sram_1kbyte_1rw1r_32x256_8" in md
        # 1KiB row precedes 2KiB row
        assert md.index("32x256") < md.index("32x512")

    def test_empty_menu_message(self):
        md = mr.macro_menu_markdown({})
        assert "No pre-built SRAM macros" in md
