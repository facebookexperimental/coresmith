# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""OpenROM (mask-ROM) enablement: naming/registry, data conversion, pricing,
and cs_rom_1r detection. Pure/deterministic -- never invokes OpenRAM itself
(the generation path is validated by the E6 smoke run)."""

from __future__ import annotations

from pathlib import Path

from orchestrator.langgraph import macro_registry as mr
from orchestrator.langgraph import mem_price as mp
from orchestrator.langgraph import openram_gen as og
from orchestrator.langgraph import sram_wrapper as sw


class TestRomNaming:
    def test_rom_name_roundtrips_through_registry_parse(self):
        name = og.rom_name_for(words=4096, data_bits=32)
        assert name == "rom_1r_32_4096_sky130"
        geom = mr._parse_geometry(name)
        assert geom == ("1r", 32, 4096, 0)
        assert mr._macro_kind(name) == "rom"

    def test_sram_names_still_parse_as_sram(self):
        assert mr._macro_kind("sram_1rw1r_32_256_8_sky130") == "sram"
        g = mr._parse_geometry("sram_1rw1r_32_256_8_sky130")
        assert g == ("1rw1r", 32, 256, 8)


class TestRomCollateral:
    def _mk(self, root: Path, name: str, with_lib: bool):
        for sub, fn in (("lef", f"{name}.lef"), ("gds", f"{name}.gds"),
                        ("verilog", f"{name}.v"), ("spice", f"{name}.spice")):
            d = root / sub
            d.mkdir(parents=True, exist_ok=True)
            (d / fn).write_text("SIZE 175.13 BY 122.55 ;\n" if sub == "lef"
                                else "x")
        if with_lib:
            (root / "lib").mkdir(exist_ok=True)
            (root / "lib" / f"{name}.lib").write_text("x")

    def test_rom_complete_without_lib(self, tmp_path):
        """rom_compiler never emits a .lib (upstream TODO) -- a ROM's
        collateral is complete without one; an SRAM's is not."""
        rom = og.rom_name_for(1024, 8)
        self._mk(tmp_path, rom, with_lib=False)
        info = mr._build_macro(rom, tmp_path)
        assert info is not None and info.kind == "rom"
        assert info.collateral_complete() is True

        sram = "sram_1rw_8_1024_8_sky130"
        self._mk(tmp_path, sram, with_lib=False)
        s = mr._build_macro(sram, tmp_path)
        assert s is not None and s.kind == "sram"
        assert s.collateral_complete() is False  # SRAM still needs the lib


class TestMemhConversion:
    def test_memh_to_rom_bytes_big_endian(self, tmp_path):
        f = tmp_path / "t.memh"
        f.write_text("00a06666\n// comment\n04808000  // trailing\n")
        out = og.memh_to_rom_bytes(f, 32)
        assert out == bytes([0x00, 0xA0, 0x66, 0x66, 0x04, 0x80, 0x80, 0x00])

    def test_non_byte_aligned_width_rejected(self, tmp_path):
        f = tmp_path / "t.memh"
        f.write_text("1\n")
        try:
            og.memh_to_rom_bytes(f, 12)
            assert False, "expected ValueError"
        except ValueError:
            pass


class TestRomPricing:
    def test_impl_rom_synonyms_normalize(self):
        for k in ("rom", "mask_rom", "maskrom", "cs_rom"):
            assert mp._normalize_impl(k) == "rom"

    def test_mem_manifest_impl_rom_parses(self):
        d = mp.parse_mem_manifest(
            "# MEM quant_rom: 32x4096 ports=1r impl=rom "
            "justification=constant quant tables")[0]
        assert d.impl == "rom"
        assert d.bits == 32 * 4096

    def test_rom_priced_with_rom_ruler_not_sram(self):
        d = mp.MemDecl(name="rom0", width=32, depth=4096, ports="1r",
                       impl="rom")
        p = mp.price_mem_decl(d, warm=False)
        assert p.estimate_source == "analytic_rom_bits"
        assert p.area_um2 == sw.rom_area_um2(d.bits)
        # sanity: far below the SRAM ruler for the same bits
        from orchestrator.langgraph.sram_wrapper import um2_per_bit
        assert p.area_um2 < d.bits * um2_per_bit()


class TestRomRtlDetection:
    RTL = (
        "module t(input clk, input [9:0] a, output [31:0] q);\n"
        "  cs_rom_1r #(.WIDTH(32), .DEPTH(1024),\n"
        "              .INIT_FILE(\"inputs/r.memh\")) u_rom (\n"
        "      .clk(clk), .ce(1'b1), .addr(a), .rdata(q));\n"
        "endmodule\n"
    )

    def test_uses_wrapper_detects_cs_rom(self):
        assert sw.uses_wrapper(self.RTL) is True

    def test_rom_instances_and_area(self):
        assert sw.rom_instances(self.RTL) == [(32, 1024)]
        assert sw.rom_bits(self.RTL) == 32 * 1024
        assert sw.estimate_rom_area_um2(self.RTL) == sw.rom_area_um2(32 * 1024)

    def test_rom_module_names_are_macroish(self):
        assert sw._is_macro_module("cs_rom_1r") is True
        assert sw._is_macro_module("rom_1r_32_4096_sky130") is True
