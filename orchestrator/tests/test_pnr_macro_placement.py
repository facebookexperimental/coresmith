# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""PnR-stage completion of the macro flow: materialize each bound
`cs_mem_macro_shell` into its CONCRETE sky130 SRAM macro (active-low pin
adapter), feed the macro vars + rewritten netlist into PnR, and hard-fail a
memory-absent layout.

Every observable change is env-gated default-ON and tested on BOTH branches.
FAIRNESS: only generic synthetic geometries + mocked macro pin lists -- no
benchmark/exercise/golden names, no PDK collateral required.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

from orchestrator.langgraph import backend_graph as bg
from orchestrator.langgraph import backend_helpers as bh
from orchestrator.langgraph import macro_backend as mb
from orchestrator.langgraph import macro_registry as mr
from orchestrator.langgraph import sram_wrapper as sw

_HAVE_YOSYS = shutil.which("yosys") is not None


# --- generic synthetic collateral -----------------------------------------

def _binding(name, width, depth, *, mdata=None, mwords=None, mask=8, kind="sram",
             lef=""):
    """A `state["macro_bindings"]` entry with the geometry PnR needs. `mdata`/
    `mwords` default to an EXACT-match concrete macro."""
    return {
        "name": name, "lef": lef, "gds": "", "lib": "", "spice": "", "verilog": "",
        "kind": kind, "width": width, "depth": depth, "nport": 2, "ports": "1rw1r",
        "macro_data_bits": mdata or width, "macro_words": mwords or depth,
        "macro_mask_bits": mask,
    }


def _shell_netlist(width=32, depth=1024, aw=10, mod=r"\$paramod$abc\cs_mem_macro_shell"):
    """A yosys-style derived netlist: a top instantiating an empty
    cs_mem_macro_shell leaf (tie-0), exactly what reaches PnR pre-fix."""
    return (
        "module top(clk, ce0, we0, a0, wd, q0, ce1, a1, q1);\n"
        "  input clk; input ce0; input we0;\n"
        f"  input [{aw-1}:0] a0; input [{width-1}:0] wd; output [{width-1}:0] q0;\n"
        f"  input ce1; input [{aw-1}:0] a1; output [{width-1}:0] q1;\n"
        f"  {mod}  u_mem (\n"
        "    .addr0(a0), .addr1(a1), .ce0(ce0), .ce1(ce1), .clk(clk),\n"
        "    .rdata0(q0), .rdata1(q1), .wdata0(wd), .we0(we0)\n"
        "  );\n"
        "endmodule\n"
        f"module {mod} (clk, ce0, we0, addr0, wdata0, rdata0, ce1, addr1, rdata1);\n"
        "  input clk; wire clk;\n"
        "  input ce0; wire ce0;\n"
        "  input we0; wire we0;\n"
        f"  input [{aw-1}:0] addr0; wire [{aw-1}:0] addr0;\n"
        f"  input [{width-1}:0] wdata0; wire [{width-1}:0] wdata0;\n"
        f"  output [{width-1}:0] rdata0; wire [{width-1}:0] rdata0;\n"
        "  input ce1; wire ce1;\n"
        f"  input [{aw-1}:0] addr1; wire [{aw-1}:0] addr1;\n"
        f"  output [{width-1}:0] rdata1; wire [{width-1}:0] rdata1;\n"
        f"  assign rdata1 = {width}'d0;\n"
        f"  assign rdata0 = {width}'d0;\n"
        "endmodule\n"
    )


def _stub_lef(tmp_path, name, w=200.0, h=300.0):
    p = tmp_path / f"{name}.lef"
    p.write_text(
        f"MACRO {name}\n  SIZE {w} BY {h} ;\n"
        "  PIN vccd1\n    USE POWER ;\n  END vccd1\n"
        "  PIN vssd1\n    USE GROUND ;\n  END vssd1\nEND {name}\n"
    )
    return str(p)


# A generic macro LEF WITH a UNITS/DATABASE header and 0.005-grid micron coords.
def _macro_lef_text(name, dbu=2000, *, bad_coord=None):
    obs = "    RECT 1.380 0.005 197.240 296.115 ;\n"
    if bad_coord is not None:
        obs += f"    RECT {bad_coord} 0.010 1.500 2.000 ;\n"
    return (
        "VERSION 5.7 ;\n"
        "BUSBITCHARS \"[]\" ;\n"
        "UNITS\n"
        f"  DATABASE MICRONS {dbu} ;\n"
        "END UNITS\n"
        f"MACRO {name}\n"
        "  CLASS BLOCK ;\n"
        "  FOREIGN {name} 0 0 ;\n"
        "  ORIGIN 0.000 0.000 ;\n"
        "  SIZE 200.005 BY 300.010 ;\n"
        "  SYMMETRY X Y R90 ;\n"
        "  PIN vccd1\n    DIRECTION INOUT ;\n    USE POWER ;\n"
        "    PORT\n      LAYER met4 ;\n" + obs + "    END\n  END vccd1\n"
        "  PIN vssd1\n    DIRECTION INOUT ;\n    USE GROUND ;\n  END vssd1\n"
        "  OBS\n    LAYER met1 ;\n    RECT 0.000 0.000 200.005 300.010 ;\n  END\n"
        "END {name}\n"
    )


def _stub_lef_dbu(tmp_path, name, dbu=2000, *, bad_coord=None):
    p = tmp_path / f"{name}.lef"
    p.write_text(_macro_lef_text(name, dbu, bad_coord=bad_coord))
    return str(p)


def _stub_tech_lef(tmp_path, dbu=1000):
    p = tmp_path / "tech.tlef"
    p.write_text(
        "VERSION 5.7 ;\nUNITS\n"
        f"  DATABASE MICRONS {dbu} ;\nEND UNITS\n"
        "LAYER met1\n  TYPE ROUTING ;\n  WIDTH 0.140 ;\nEND met1\n"
    )
    return p


# ---------------------------------------------------------------------------
# Fix env gate (both branches)
# ---------------------------------------------------------------------------

class TestPlacementGate:
    def test_flag_both_branches(self, monkeypatch):
        monkeypatch.delenv("CORESMITH_PNR_MACRO_PLACEMENT", raising=False)
        assert sw.pnr_macro_placement_enabled() is True          # default ON
        monkeypatch.setenv("CORESMITH_PNR_MACRO_PLACEMENT", "0")
        assert sw.pnr_macro_placement_enabled() is False          # pre-fix restored


# ---------------------------------------------------------------------------
# Fix 2a: the active-low pin adapter (read from real sky130 1rw1r collateral)
# ---------------------------------------------------------------------------

class TestPinAdapter:
    def test_exact_match_active_low_mapping(self):
        frag = mb.build_macro_adapter_instance(
            32, 1024, macro_name="synth_sram_a",
            macro_data_bits=32, macro_words=1024, macro_mask_bits=8)
        # active-LOW controls come from REAL inverter cells (not behavioral ~).
        assert f"{mb._INV_CELL} _u_csb0 (.A(ce0), .Y(_csb0));" in frag  # csb0 = ~ce0
        assert f"{mb._INV_CELL} _u_web0 (.A(we0), .Y(_web0));" in frag  # web0 = ~we0
        assert f"{mb._INV_CELL} _u_csb1 (.A(ce1), .Y(_csb1));" in frag  # csb1 = ~ce1
        # concrete macro wired with the real OpenRAM pin names + polarity.
        assert ".csb0(_csb0)" in frag and ".web0(_web0)" in frag
        assert ".csb1(_csb1)" in frag
        assert ".din0(wdata0)" in frag and ".dout0(rdata0)" in frag
        assert ".dout1(rdata1)" in frag
        assert ".clk0(clk)" in frag and ".clk1(clk)" in frag
        # 32b / 8b write_size -> 4 mask lanes, all-ones (full-word write).
        assert ".wmask0(4'hf)" in frag
        assert "synth_sram_a u_macro (" in frag

    def test_wmask_width_tracks_geometry(self):
        # 8b wide / 8b write_size -> a single mask lane, all-ones.
        frag = mb.build_macro_adapter_instance(
            8, 256, macro_name="synth_sram_b",
            macro_data_bits=8, macro_words=256, macro_mask_bits=8)
        assert ".wmask0(1'h1)" in frag

    def test_over_provisioned_macro_extends_and_truncates(self):
        # macro deeper (2048 vs 1024) AND wider (48 vs 32): extra addr/data MSBs
        # tied to 0, extra read bits dropped.
        frag = mb.build_macro_adapter_instance(
            32, 1024, macro_name="synth_sram_c",
            macro_data_bits=48, macro_words=2048, macro_mask_bits=8)
        assert "{ {1{1'b0}}, addr0 }" in frag       # 11-bit macro addr vs 10-bit
        assert "{ {16{1'b0}}, wdata0 }" in frag      # 48-bit macro din vs 32-bit
        assert "assign rdata0 = _dout0[31:0];" in frag  # truncate 48 -> 32
        assert "assign rdata1 = _dout1[31:0];" in frag


# ---------------------------------------------------------------------------
# Fix 2b: materialize the shell -> concrete macro in the PnR-read netlist
# ---------------------------------------------------------------------------

class TestMaterialize:
    def test_shell_becomes_concrete_macro(self):
        net = _shell_netlist(32, 1024, 10)
        new, placed = mb.materialize_macro_netlist(
            net, [_binding("synth_sram_x", 32, 1024)])
        # the empty tie-0 shell body is gone; the concrete macro is instantiated.
        assert "assign rdata0 = 32'd0;" not in new
        assert "synth_sram_x u_macro (" in new
        assert ".csb0(_csb0)" in new and ".din0(wdata0)" in new
        # the shell MODULE NAME is preserved -> the parent instance still binds.
        assert r"\$paramod$abc\cs_mem_macro_shell  u_mem (" in new
        # one physical macro instance recorded for the floorplan.
        assert len(placed) == 1
        assert placed[0][0].name == "synth_sram_x"

    def test_no_binding_for_geometry_leaves_stub(self):
        # a shell whose geometry no binding matches is left untouched (the PnR
        # memory-absent assertion then catches a genuinely unplaced macro).
        net = _shell_netlist(16, 512, 9)
        new, placed = mb.materialize_macro_netlist(
            net, [_binding("synth_sram_x", 32, 1024)])
        assert placed == []
        assert "assign rdata0 = 16'd0;" in new       # unchanged
        assert "u_macro" not in new

    def test_rom_kind_binding_ignored(self):
        # only SRAM shells are materialized here (ROM pins differ); a ROM binding
        # must not corrupt a cs_mem shell.
        net = _shell_netlist(32, 1024, 10)
        new, placed = mb.materialize_macro_netlist(
            net, [_binding("synth_rom_x", 32, 1024, kind="rom")])
        assert placed == []
        assert new == net

    def test_multiple_instances_counted(self):
        # same-geometry memory instantiated twice -> two physical macros planned.
        net = _shell_netlist(32, 1024, 10)
        # add a second instantiation of the same shell module in the top.
        net = net.replace(
            "  );\nendmodule\nmodule",
            "  );\n"
            r"  \$paramod$abc\cs_mem_macro_shell  u_mem2 (.clk(clk), .ce0(ce0),"
            " .we0(we0), .addr0(a0), .wdata0(wd), .rdata0(q0), .ce1(ce1),"
            " .addr1(a1), .rdata1(q1));\nendmodule\nmodule", 1)
        _new, placed = mb.materialize_macro_netlist(
            net, [_binding("synth_sram_x", 32, 1024)])
        assert len(placed) == 2


# ---------------------------------------------------------------------------
# Fix 1: PnR variable emission from macro_bindings (+ regression on the old path)
# ---------------------------------------------------------------------------

class TestPnrHeaderEmission:
    def test_bindings_emit_vars_and_materialized_netlist(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mr, "discover_macros", lambda *a, **k: {})
        monkeypatch.setenv("CORESMITH_PNR_MACRO_PLACEMENT", "1")
        net = tmp_path / "flat.v"
        net.write_text(_shell_netlist(32, 1024, 10))
        lef = _stub_lef(tmp_path, "synth_sram_x")
        tcl = bh.prepare_pnr_working_copy(
            design_name="blk", netlist_path=str(net), sdc_path=str(tmp_path / "s.sdc"),
            output_dir=str(tmp_path / "out"),
            macro_bindings=[_binding("synth_sram_x", 32, 1024, lef=lef)])
        content = Path(tcl).read_text()
        # (i) macro vars reach the PnR tcl ...
        assert "set macro_names [list synth_sram_x]" in content
        assert "set macro_place" in content and "set macro_die_area" in content
        assert lef in content                                     # LEF var carried
        # ... and read_verilog is pointed at the materialized netlist.
        assert 'set netlist    "' in content and "blk_macro.v" in content
        mat = tmp_path / "out" / "blk_macro.v"
        assert mat.exists()
        assert "synth_sram_x u_macro (" in mat.read_text()

    def test_env_off_no_materialization(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mr, "discover_macros", lambda *a, **k: {})
        monkeypatch.setenv("CORESMITH_PNR_MACRO_PLACEMENT", "0")     # pre-fix
        net = tmp_path / "flat.v"
        net.write_text(_shell_netlist(32, 1024, 10))
        tcl = bh.prepare_pnr_working_copy(
            design_name="blk", netlist_path=str(net), sdc_path=str(tmp_path / "s.sdc"),
            output_dir=str(tmp_path / "out"),
            macro_bindings=[_binding("synth_sram_x", 32, 1024)])
        content = Path(tcl).read_text()
        assert "set macro_names" not in content
        assert "blk_macro.v" not in content
        assert not (tmp_path / "out" / "blk_macro.v").exists()

    def test_concrete_name_path_unchanged(self, tmp_path, monkeypatch):
        # REGRESSION: a netlist that already instantiates a concrete macro takes
        # the old detect+place path -- vars emitted, netlist NOT rewritten.
        info = mr.MacroInfo(
            name="synth_sram_direct", lef=_stub_lef(tmp_path, "synth_sram_direct"),
            gds="", lib="", spice="", verilog="", width_um=200.0, height_um=300.0,
            words=1024, data_bits=32, bits=32 * 1024, ports="1rw1r", mask_bits=8)
        monkeypatch.setattr(mr, "discover_macros",
                            lambda *a, **k: {"synth_sram_direct": info})
        net = tmp_path / "flat.v"
        net.write_text(
            "module top(input clk);\n"
            "  synth_sram_direct u_mem (.clk0(clk));\nendmodule\n")
        tcl = bh.prepare_pnr_working_copy(
            design_name="blk", netlist_path=str(net), sdc_path=str(tmp_path / "s.sdc"),
            output_dir=str(tmp_path / "out"),
            macro_bindings=None)
        content = Path(tcl).read_text()
        assert "set macro_names [list synth_sram_direct]" in content
        assert "blk_macro.v" not in content                       # not rewritten
        assert not (tmp_path / "out" / "blk_macro.v").exists()


# ---------------------------------------------------------------------------
# Fix 3: memory-absent PnR assertion (analogue of the synth flop gate)
# ---------------------------------------------------------------------------

def _def_with(tmp_path, *masters):
    p = tmp_path / "routed.def"
    comps = "\n".join(
        f"- u{i} {m} + PLACED ( {i*10} {i*10} ) N ;" for i, m in enumerate(masters))
    p.write_text(
        "DESIGN blk ;\n"
        f"COMPONENTS {len(masters)} ;\n{comps}\nEND COMPONENTS\nEND DESIGN\n")
    return str(p)


class TestMemoryAbsentAssertion:
    def test_fires_when_bound_but_not_placed(self, tmp_path):
        d = _def_with(tmp_path, "sky130_fd_sc_hd__and2_1", "sky130_fd_sc_hd__inv_2")
        err = bg.memory_absent_pnr_error([_binding("synth_sram_x", 32, 1024)], d)
        assert err is not None
        assert "0 were PLACED" in err and "synth_sram_x" in err

    def test_passes_when_macro_placed(self, tmp_path):
        d = _def_with(tmp_path, "synth_sram_x", "sky130_fd_sc_hd__inv_2")
        assert bg.memory_absent_pnr_error(
            [_binding("synth_sram_x", 32, 1024)], d) is None

    def test_none_when_no_bindings(self, tmp_path):
        d = _def_with(tmp_path, "sky130_fd_sc_hd__inv_2")
        assert bg.memory_absent_pnr_error([], d) is None
        assert bg.memory_absent_pnr_error(None, d) is None

    def test_none_when_def_missing(self):
        # can't verify -> never a false-fail.
        assert bg.memory_absent_pnr_error(
            [_binding("synth_sram_x", 32, 1024)], "/no/such/routed.def") is None


# ---------------------------------------------------------------------------
# Fix 1: normalize each macro LEF's DBU to the tech LEF's DBU (header-only)
# ---------------------------------------------------------------------------

class TestLefDbuNormalization:
    def test_reads_dbu_header(self):
        assert mb.lef_database_units(_macro_lef_text("m", 2000)) == 2000
        assert mb.lef_database_units(_macro_lef_text("m", 1000)) == 1000
        # no UNITS/DATABASE header -> None (leave untouched)
        assert mb.lef_database_units("MACRO m\n  SIZE 1.0 BY 2.0 ;\nEND m\n") is None

    def test_downconvert_is_header_only_coords_unchanged(self):
        src = _macro_lef_text("synth_sram_x", 2000)
        out, changed = mb.normalize_lef_dbu(src, 1000)
        assert changed is True
        assert "DATABASE MICRONS 1000" in out
        assert "DATABASE MICRONS 2000" not in out
        # EVERY line except the DATABASE MICRONS line is byte-identical -> no
        # coordinate value was scaled (that would corrupt the macro geometry).
        def mask(t):
            return re.sub(r"DATABASE MICRONS \d+", "DATABASE MICRONS _", t)
        assert mask(src) == mask(out)
        # spot-check a coordinate survived verbatim.
        assert "SIZE 200.005 BY 300.010 ;" in out

    def test_already_at_or_below_tech_dbu_untouched(self):
        # an efabless-prebuilt macro LEF is already 1000 -- normalize per-LEF,
        # never assume 2000, never rewrite when src <= target.
        src = _macro_lef_text("synth_sram_y", 1000)
        out, changed = mb.normalize_lef_dbu(src, 1000)
        assert changed is False and out == src
        # a coarser macro (500) than the tech (1000) is also left alone.
        src2 = _macro_lef_text("synth_sram_z", 500)
        out2, changed2 = mb.normalize_lef_dbu(src2, 1000)
        assert changed2 is False and out2 == src2

    def test_no_header_untouched(self):
        src = "MACRO m\n  SIZE 1.0 BY 2.0 ;\nEND m\n"
        out, changed = mb.normalize_lef_dbu(src, 1000)
        assert changed is False and out == src

    def test_non_representable_coord_fails_hard(self):
        # 0.0001 um is not an integer multiple of 1/1000 um -> refuse to rewrite
        # (silently down-converting would corrupt the geometry).
        src = _macro_lef_text("synth_sram_bad", 2000, bad_coord=0.0001)
        with pytest.raises(mb.LefDbuError) as ei:
            mb.normalize_lef_dbu(src, 1000)
        assert "0.0001" in str(ei.value)          # names the offending value
        assert "1000 DBU" in str(ei.value)

    def test_tech_dbu_read_from_stub_not_hardcoded(self, tmp_path):
        # Fix reads the tech LEF's OWN DATABASE MICRONS (not a hardcoded 1000).
        assert bh._tech_lef_dbu(_stub_tech_lef(tmp_path, 1000)) == 1000
        assert bh._tech_lef_dbu(_stub_tech_lef(tmp_path, 2000)) == 2000
        assert bh._tech_lef_dbu(tmp_path / "does_not_exist.tlef") is None


class TestPrepareNormalizesMacroLefs:
    def _wire_tech(self, monkeypatch, tmp_path, dbu=1000):
        # tech DBU comes from a stub tech LEF read through the real reader
        # (capture the real fn first so the patched default arg doesn't recurse).
        stub = _stub_tech_lef(tmp_path, dbu)
        real = bh._tech_lef_dbu
        monkeypatch.setattr(bh, "_tech_lef_dbu", lambda *a, **k: real(stub))

    def test_prepare_points_macro_lefs_at_normalized_copy(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mr, "discover_macros", lambda *a, **k: {})
        monkeypatch.setenv("CORESMITH_PNR_MACRO_PLACEMENT", "1")
        self._wire_tech(monkeypatch, tmp_path, 1000)
        net = tmp_path / "flat.v"
        net.write_text(_shell_netlist(32, 1024, 10))
        lef = _stub_lef_dbu(tmp_path, "synth_sram_x", 2000)      # macro at 2000
        tcl = bh.prepare_pnr_working_copy(
            design_name="blk", netlist_path=str(net), sdc_path=str(tmp_path / "s.sdc"),
            output_dir=str(tmp_path / "out"),
            macro_bindings=[_binding("synth_sram_x", 32, 1024, lef=lef)])
        content = Path(tcl).read_text()
        copy = tmp_path / "out" / "macro_lefs" / "synth_sram_x.lef"
        assert copy.exists()                                      # (i) copy written
        assert str(copy) in content                               # macro_lefs -> copy
        assert lef not in content                                 # NOT the 2000 orig
        norm = copy.read_text()
        assert "DATABASE MICRONS 1000" in norm                    # normalized to tech
        assert "DATABASE MICRONS 2000" not in norm
        # coordinates byte-identical vs the original 2000-DBU LEF.
        def mask(t):
            return re.sub(r"DATABASE MICRONS \d+", "DATABASE MICRONS _", t)
        assert mask(Path(lef).read_text()) == mask(norm)

    def test_env_off_no_normalization(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mr, "discover_macros", lambda *a, **k: {})
        monkeypatch.setenv("CORESMITH_PNR_MACRO_PLACEMENT", "0")   # pre-fix
        self._wire_tech(monkeypatch, tmp_path, 1000)
        net = tmp_path / "flat.v"
        net.write_text(_shell_netlist(32, 1024, 10))
        lef = _stub_lef_dbu(tmp_path, "synth_sram_x", 2000)
        bh.prepare_pnr_working_copy(
            design_name="blk", netlist_path=str(net), sdc_path=str(tmp_path / "s.sdc"),
            output_dir=str(tmp_path / "out"),
            macro_bindings=[_binding("synth_sram_x", 32, 1024, lef=lef)])
        # env-off: shell path disabled entirely -> no macro_lefs dir at all.
        assert not (tmp_path / "out" / "macro_lefs").exists()

    def test_non_representable_macro_lef_fails_prepare(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mr, "discover_macros", lambda *a, **k: {})
        monkeypatch.setenv("CORESMITH_PNR_MACRO_PLACEMENT", "1")
        self._wire_tech(monkeypatch, tmp_path, 1000)
        net = tmp_path / "flat.v"
        net.write_text(_shell_netlist(32, 1024, 10))
        lef = _stub_lef_dbu(tmp_path, "synth_sram_x", 2000, bad_coord=0.0001)
        with pytest.raises(mb.LefDbuError):
            bh.prepare_pnr_working_copy(
                design_name="blk", netlist_path=str(net), sdc_path=str(tmp_path / "s.sdc"),
                output_dir=str(tmp_path / "out"),
                macro_bindings=[_binding("synth_sram_x", 32, 1024, lef=lef)])


# ---------------------------------------------------------------------------
# Fix 2: hardening -- non-optional placement + failure-path memory-absent park
# ---------------------------------------------------------------------------

class TestNonOptionalPlacement:
    def test_required_placement_failure_raises(self, tmp_path, monkeypatch):
        # bindings present + enabled -> macro placement is NON-OPTIONAL. A failure
        # during macro injection must NOT be swallowed into a macro-less layout.
        monkeypatch.setattr(mr, "discover_macros", lambda *a, **k: {})
        monkeypatch.setenv("CORESMITH_PNR_MACRO_PLACEMENT", "1")
        def _boom(*a, **k):
            raise RuntimeError("plan_floorplan blew up")
        monkeypatch.setattr(mb, "plan_floorplan", _boom)
        net = tmp_path / "flat.v"
        net.write_text(_shell_netlist(32, 1024, 10))
        lef = _stub_lef(tmp_path, "synth_sram_x")
        with pytest.raises(RuntimeError):
            bh.prepare_pnr_working_copy(
                design_name="blk", netlist_path=str(net), sdc_path=str(tmp_path / "s.sdc"),
                output_dir=str(tmp_path / "out"),
                macro_bindings=[_binding("synth_sram_x", 32, 1024, lef=lef)])

    def test_env_off_swallows_and_std_path_survives(self, tmp_path, monkeypatch):
        # env-off (pre-fix): placement not required -> the shell path is skipped,
        # no exception, the std-cell TCL is produced normally.
        monkeypatch.setattr(mr, "discover_macros", lambda *a, **k: {})
        monkeypatch.setenv("CORESMITH_PNR_MACRO_PLACEMENT", "0")
        def _boom(*a, **k):
            raise RuntimeError("plan_floorplan blew up")
        monkeypatch.setattr(mb, "plan_floorplan", _boom)
        net = tmp_path / "flat.v"
        net.write_text(_shell_netlist(32, 1024, 10))
        tcl = bh.prepare_pnr_working_copy(
            design_name="blk", netlist_path=str(net), sdc_path=str(tmp_path / "s.sdc"),
            output_dir=str(tmp_path / "out"),
            macro_bindings=[_binding("synth_sram_x", 32, 1024)])
        assert Path(tcl).exists()
        assert "set macro_names" not in Path(tcl).read_text()


def _routed_def_no_macro(tmp_path, block="blk"):
    d = tmp_path / "syn" / "output" / block / "pnr"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{block}_routed.def").write_text(
        "DESIGN blk ;\nCOMPONENTS 1 ;\n"
        "- u0 sky130_fd_sc_hd__inv_2 + PLACED ( 0 0 ) N ;\n"
        "END COMPONENTS\nEND DESIGN\n")
    return d


class TestFailurePathMemoryAbsent:
    @pytest.mark.asyncio
    async def test_pnr_failure_converts_to_memory_absent(self, tmp_path, monkeypatch):
        # PnR reports FAILURE and the (partial) DEF has no macro -> the honest
        # memory-absent diagnosis becomes the reported error (not the raw churn).
        monkeypatch.setenv("CORESMITH_PNR_MACRO_PLACEMENT", "1")
        monkeypatch.setattr(mr, "discover_macros", lambda *a, **k: {})
        _routed_def_no_macro(tmp_path)
        net = tmp_path / "flat.v"
        net.write_text("module blk(input clk);\nendmodule\n")

        async def _fake_llm(**kwargs):
            return {"success": False, "error": "routing exploded"}
        monkeypatch.setattr(bg, "_run_llm_eda_step", _fake_llm)

        state = {
            "current_block": {"name": "blk"}, "attempt": 1,
            "project_root": str(tmp_path), "flat_netlist_path": str(net),
            "flat_sdc_path": "", "target_clock_mhz": 50.0, "max_attempts": 3,
            "macro_bindings": [_binding("synth_sram_x", 32, 1024)],
        }
        result = await bg.run_pnr_node(state)
        assert result["timing_result"]["met"] is False
        err = result["previous_error"]
        assert "0 were PLACED" in err and "synth_sram_x" in err

    @pytest.mark.asyncio
    async def test_env_off_keeps_raw_failure(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_PNR_MACRO_PLACEMENT", "0")     # pre-fix
        monkeypatch.setattr(mr, "discover_macros", lambda *a, **k: {})
        _routed_def_no_macro(tmp_path)
        net = tmp_path / "flat.v"
        net.write_text("module blk(input clk);\nendmodule\n")

        async def _fake_llm(**kwargs):
            return {"success": False, "error": "routing exploded"}
        monkeypatch.setattr(bg, "_run_llm_eda_step", _fake_llm)

        state = {
            "current_block": {"name": "blk"}, "attempt": 1,
            "project_root": str(tmp_path), "flat_netlist_path": str(net),
            "flat_sdc_path": "", "target_clock_mhz": 50.0, "max_attempts": 3,
            "macro_bindings": [_binding("synth_sram_x", 32, 1024)],
        }
        result = await bg.run_pnr_node(state)
        assert "routing exploded" in result["previous_error"]
        assert "0 were PLACED" not in result["previous_error"]


# ---------------------------------------------------------------------------
# Real top-module selection (regression fix): a SUB-BLOCK-FIRST netlist must
# link the real integration top, not the first-defined sub-block.
# ---------------------------------------------------------------------------

def _subblock_first_shell_netlist(
    width=32, depth=1024, aw=10, mod=r"\$paramod$abc\cs_mem_macro_shell"
):
    """Generic macro-rewritten shape: a sub-block DEFINED FIRST that reuses a
    shell, the real integration top DEFINED LAST, and the shell def last. The
    shell is instantiated 2x per sub-block, the top instantiates 3 sub-blocks +
    1 direct extra -> 3 shell text-sites that FLATTEN to 3*2 + 1 = 7 leaves."""
    def inst(name):
        return (f"  {mod}  {name} (.addr0(a0), .addr1(a1), .ce0(ce0), .ce1(ce1),"
                " .clk(clk), .rdata0(q0), .rdata1(q1), .wdata0(wd), .we0(we0));\n")
    conn = ("(.clk(clk), .ce0(ce0), .we0(we0), .a0(a0), .wd(wd), .q0(q0),"
            " .ce1(ce1), .a1(a1), .q1(q1))")
    hdr = "(clk, ce0, we0, a0, wd, q0, ce1, a1, q1)"
    decls = (
        "  input clk; input ce0; input we0;\n"
        f"  input [{aw-1}:0] a0; input [{width-1}:0] wd; output [{width-1}:0] q0;\n"
        f"  input ce1; input [{aw-1}:0] a1; output [{width-1}:0] q1;\n"
    )
    return (
        f"module sub_bank {hdr};\n" + decls + inst("u_s0") + inst("u_s1")
        + "endmodule\n"
        f"module chip_top {hdr};\n" + decls
        + f"  sub_bank u_b0 {conn};\n"
        + f"  sub_bank u_b1 {conn};\n"
        + f"  sub_bank u_b2 {conn};\n"
        + inst("u_extra")
        + "endmodule\n"
        f"module {mod} (clk, ce0, we0, addr0, wdata0, rdata0, ce1, addr1, rdata1);\n"
        "  input clk; wire clk;\n  input ce0; wire ce0;\n  input we0; wire we0;\n"
        f"  input [{aw-1}:0] addr0; wire [{aw-1}:0] addr0;\n"
        f"  input [{width-1}:0] wdata0; wire [{width-1}:0] wdata0;\n"
        f"  output [{width-1}:0] rdata0; wire [{width-1}:0] rdata0;\n"
        "  input ce1; wire ce1;\n"
        f"  input [{aw-1}:0] addr1; wire [{aw-1}:0] addr1;\n"
        f"  output [{width-1}:0] rdata1; wire [{width-1}:0] rdata1;\n"
        f"  assign rdata1 = {width}'d0;\n  assign rdata0 = {width}'d0;\nendmodule\n"
    )


class TestTopModuleSelection:
    def test_prepare_sets_real_top_not_subblock(self, tmp_path, monkeypatch):
        # prepare_pnr_working_copy must emit `set design_name <real_top>` for a
        # sub-block-first netlist -- NOT the first-defined sub-block.
        monkeypatch.setattr(mr, "discover_macros", lambda *a, **k: {})
        monkeypatch.setenv("CORESMITH_PNR_MACRO_PLACEMENT", "0")  # isolate Fix 1
        net = tmp_path / "flat.v"
        net.write_text(_subblock_first_shell_netlist())
        tcl = bh.prepare_pnr_working_copy(
            design_name="blk", netlist_path=str(net),
            sdc_path=str(tmp_path / "s.sdc"), output_dir=str(tmp_path / "out"))
        content = Path(tcl).read_text()
        assert 'set design_name "chip_top"' in content
        assert 'set design_name "sub_bank"' not in content
        assert 'set design_name "blk"' not in content

    def test_generate_links_real_top_not_subblock(self, tmp_path):
        net = tmp_path / "flat.v"
        net.write_text(_subblock_first_shell_netlist())
        tcl = bh.generate_pnr_tcl(
            "blk", str(net), str(tmp_path / "s.sdc"), str(tmp_path / "out"))
        content = Path(tcl).read_text()
        assert "link_design chip_top" in content
        assert "link_design sub_bank" not in content

    def test_missing_netlist_falls_back_to_block_name(self, tmp_path):
        # unreadable netlist -> preferred name preserved (pre-fix fallback).
        tcl = bh.generate_pnr_tcl(
            "my_adder", "/no/such/net.v", str(tmp_path / "s.sdc"),
            str(tmp_path / "out"))
        assert "link_design my_adder" in Path(tcl).read_text()


# ---------------------------------------------------------------------------
# Fix 3: die/placement sized for the FLATTENED macro leaf count (not the
# shell-module text-instantiation count).
# ---------------------------------------------------------------------------

class TestFlattenedDiePlan:
    def test_materialize_counts_flattened_leaves(self):
        # 3 shell text-sites flatten to 7 physical leaves.
        net = _subblock_first_shell_netlist(32, 1024, 10)
        _new, placed = mb.materialize_macro_netlist(
            net, [_binding("synth_sram_x", 32, 1024)])
        assert len(placed) == 7

    def test_prepare_plans_die_for_flattened_count(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mr, "discover_macros", lambda *a, **k: {})
        monkeypatch.setenv("CORESMITH_PNR_MACRO_PLACEMENT", "1")
        net = tmp_path / "flat.v"
        net.write_text(_subblock_first_shell_netlist(32, 1024, 10))
        lef = _stub_lef(tmp_path, "synth_sram_x")
        tcl = bh.prepare_pnr_working_copy(
            design_name="blk", netlist_path=str(net),
            sdc_path=str(tmp_path / "s.sdc"), output_dir=str(tmp_path / "out"),
            macro_bindings=[_binding("synth_sram_x", 32, 1024, lef=lef)])
        content = Path(tcl).read_text()
        # macro_place must carry ONE position per flattened leaf (7), not 3.
        positions = re.findall(r"\{[-\d.]+ [-\d.]+ R0\}", content)
        assert len(positions) == 7


# ---------------------------------------------------------------------------
# Fix 2: cell-count guard fires inside run_pnr_node (fragment linked as top).
# ---------------------------------------------------------------------------

def _routed_def_ncells(tmp_path, n, block="blk"):
    d = tmp_path / "syn" / "output" / block / "pnr"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{block}_routed.def"
    comps = "\n".join(
        f"- u{i} sky130_fd_sc_hd__inv_2 + PLACED ( {i} {i} ) N ;"
        for i in range(n))
    p.write_text(
        f"DESIGN {block} ;\nCOMPONENTS {n} ;\n{comps}\nEND COMPONENTS\n"
        "END DESIGN\n")
    return str(p)


class TestCellCountGuardInNode:
    def _state(self, tmp_path, gate_count):
        net = tmp_path / "flat.v"
        net.write_text("module blk(input clk);\nendmodule\n")
        return {
            "current_block": {"name": "blk"}, "attempt": 1,
            "project_root": str(tmp_path), "flat_netlist_path": str(net),
            "flat_sdc_path": "", "target_clock_mhz": 50.0, "max_attempts": 3,
            "macro_bindings": None, "synth_gate_count": gate_count,
        }

    def _wire_success_llm(self, monkeypatch, routed_def):
        monkeypatch.setattr(mr, "discover_macros", lambda *a, **k: {})
        monkeypatch.setattr(bh, "render_layout_image", lambda *a, **k: True)

        async def _fake_llm(**kwargs):
            return {"success": True, "routed_def_path": routed_def,
                    "wns_ns": 0.0}
        monkeypatch.setattr(bg, "_run_llm_eda_step", _fake_llm)

    @pytest.mark.asyncio
    async def test_fragment_linked_as_top_fails(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_PNR_CELLCOUNT_GUARD", "1")   # default ON
        d = _routed_def_ncells(tmp_path, 154)                     # fragment
        self._wire_success_llm(monkeypatch, d)
        result = await bg.run_pnr_node(self._state(tmp_path, 4682))
        assert result["timing_result"]["met"] is False
        assert "Cell-count shortfall" in result["previous_error"]
        assert "154" in result["previous_error"]

    @pytest.mark.asyncio
    async def test_matching_counts_pass(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_PNR_CELLCOUNT_GUARD", "1")
        d = _routed_def_ncells(tmp_path, 220)     # >= 50% of 200 (PnR adds cells)
        self._wire_success_llm(monkeypatch, d)
        result = await bg.run_pnr_node(self._state(tmp_path, 200))
        assert result.get("routed_def_path") == d
        assert "previous_error" not in result

    @pytest.mark.asyncio
    async def test_env_off_keeps_fragment_success(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_PNR_CELLCOUNT_GUARD", "0")   # pre-fix
        d = _routed_def_ncells(tmp_path, 154)
        self._wire_success_llm(monkeypatch, d)
        result = await bg.run_pnr_node(self._state(tmp_path, 4682))
        assert result.get("routed_def_path") == d
        assert "previous_error" not in result


# ===========================================================================
# Tiled-composition materialization: a shell resolved to an N-tile COMPOSITION
# plan (over-provision/width-tile of a clean pre-built macro) must be
# MATERIALIZED into its concrete tiles, PLACED, and covered by the memory-absent
# assertion -- closing the false-pass where a composition plan was logged then
# dropped (the shell stayed unresolved, the memory was physically absent, yet
# the run proceeded to DRC/LVS). Env-gated default-ON; both branches tested.
# FAIRNESS: generic synthetic geometries only -- no benchmark/design names.
# ===========================================================================

def _comp_binding(base="synth_sram_tile", width=64, depth=32, *, mdata=32,
                  mwords=64, tiles_wide=2, tiles_deep=1, mask=8, lef=""):
    """A `macro_bindings` entry for a shell resolved to an N-tile composition:
    top-level fields = the BASE tile macro (DRC/LVS black box + placed-macro
    assertion key on it); the `composition` sub-dict = the tile array shape the
    materializer tiles from."""
    return {
        "name": base, "lef": lef, "gds": "", "lib": "", "spice": "", "verilog": "",
        "kind": "sram", "width": width, "depth": depth, "nport": 2,
        "ports": "1rw1r", "macro_data_bits": mdata, "macro_words": mwords,
        "macro_mask_bits": mask,
        "composition": {
            "tiles_wide": tiles_wide, "tiles_deep": tiles_deep,
            "provisioned_words": tiles_deep * mwords,
            "provisioned_bits": tiles_wide * mdata, "base": base,
        },
    }


def _synth_plan(width=64, depth=32, *, mdata=32, mwords=64, tiles_wide=2,
                tiles_deep=1, base_name="synth_sram_tile"):
    """A (ShellSpec, CompositionPlan) pair, as the shell binder resolves."""
    from orchestrator.langgraph.macro_registry import MacroInfo, ShellSpec
    from orchestrator.langgraph.openram_gen import CompositionPlan
    base = MacroInfo(
        name=base_name, lef="t.lef", gds="t.gds", lib="t.lib", spice="t.sp",
        verilog="t.v", data_bits=mdata, words=mwords, mask_bits=8, ports="1rw1r")
    plan = CompositionPlan(
        words=depth, data_bits=width, tiles_wide=tiles_wide,
        tiles_deep=tiles_deep, base=base)
    sp = ShellSpec(kind="sram", width=width, depth=depth, nport=2)
    return sp, plan


class TestCompositionGate:
    def test_flag_both_branches(self, monkeypatch):
        monkeypatch.delenv("CORESMITH_MACRO_COMPOSE_TILES", raising=False)
        assert sw.macro_compose_tiles_enabled() is True           # default ON
        monkeypatch.setenv("CORESMITH_MACRO_COMPOSE_TILES", "0")
        assert sw.macro_compose_tiles_enabled() is False           # pre-fix


class TestCompositionBuilder:
    def test_width_tile_bit_ranges_derived_from_plan(self):
        # logical 64w x 32d from 2x (32w x 64d): WIDTH tiling + depth over-prov.
        frag = mb.build_macro_composition_instance(
            64, 32, macro_name="synth_sram_tile", macro_data_bits=32,
            macro_words=64, tiles_wide=2, tiles_deep=1, macro_mask_bits=8)
        # two concrete tiles of the BASE macro.
        assert "synth_sram_tile u_tile_r0_c0 (" in frag
        assert "synth_sram_tile u_tile_r0_c1 (" in frag
        # WIDTH tiling: tile0 <- din[31:0], tile1 <- din[63:32] (LVS validates).
        assert ".din0(wdata0[31:0])" in frag
        assert ".din0(wdata0[63:32])" in frag
        # read reassembled MSB-column-first: rdata0[63:32]<-c1, [31:0]<-c0.
        assert "assign rdata0 = { _dout0_c1[31:0], _dout0_c0[31:0] };" in frag
        assert "assign rdata1 = { _dout1_c1[31:0], _dout1_c0[31:0] };" in frag
        # DEPTH over-provision: 5-bit logical addr zero-extended into 6-bit tile.
        assert "{ {1{1'b0}}, addr0 }" in frag
        assert "{ {1{1'b0}}, addr1 }" in frag
        # active-LOW controls via REAL inverters, SHARED across both tiles.
        assert f"{mb._INV_CELL} _u_csb0 (.A(ce0), .Y(_csb0));" in frag
        assert f"{mb._INV_CELL} _u_web0 (.A(we0), .Y(_web0));" in frag
        assert f"{mb._INV_CELL} _u_csb1 (.A(ce1), .Y(_csb1));" in frag
        # exactly one shared control set (not one per tile).
        assert frag.count("_u_csb0 ") == 1 and frag.count("_u_web0 ") == 1
        # 32b / 8b write_size -> 4 mask lanes all-ones (full-word write).
        assert ".wmask0(4'hf)" in frag

    def test_partial_last_width_column_zero_extends(self):
        # logical 48w from 2x 32w tiles: col0 full [31:0], col1 partial [47:32]
        # (16b) zero-extended into the 32b tile din; upper read bits dropped.
        frag = mb.build_macro_composition_instance(
            48, 64, macro_name="synth_sram_tile", macro_data_bits=32,
            macro_words=64, tiles_wide=2, tiles_deep=1, macro_mask_bits=8)
        assert ".din0(wdata0[31:0])" in frag                        # full col0
        assert ".din0({ {16{1'b0}}, wdata0[47:32] })" in frag       # partial col1
        # rdata assembled from used bits only (col1 contributes 16, col0 32).
        assert "assign rdata0 = { _dout0_c1[15:0], _dout0_c0[31:0] };" in frag

    def test_pure_width_tile_no_depth_overprovision(self):
        # logical 64w x 64d from 2x (32w x 64d): exact depth, no addr zext.
        frag = mb.build_macro_composition_instance(
            64, 64, macro_name="synth_sram_tile", macro_data_bits=32,
            macro_words=64, tiles_wide=2, tiles_deep=1, macro_mask_bits=8)
        assert frag.count("synth_sram_tile u_tile_r0_c") == 2
        assert ".addr0(addr0)" in frag                              # no zext
        assert "1'b0}}, addr0" not in frag

    def test_multibank_depth_is_rejected(self):
        # genuine multi-bank (tiles_deep>1) is NOT realized by shared-control
        # tiling -- the builder refuses rather than emit a mis-banked memory.
        with pytest.raises(ValueError):
            mb.build_macro_composition_instance(
                32, 128, macro_name="synth_sram_tile", macro_data_bits=32,
                macro_words=64, tiles_wide=1, tiles_deep=2, macro_mask_bits=8)


class TestCompositionMaterialize:
    def test_shell_becomes_tiles(self):
        net = _shell_netlist(64, 32, 5)
        new, placed = mb.materialize_macro_netlist(net, [_comp_binding()])
        assert "assign rdata0 = 64'd0;" not in new                 # stub gone
        assert new.count("synth_sram_tile u_tile_r0_c") == 2       # 2 tiles
        assert r"\$paramod$abc\cs_mem_macro_shell  u_mem (" in new  # name kept
        # one physical entry per tile so the floorplan sizes for the real count.
        assert len(placed) == 2
        assert {p[0].name for p in placed} == {"synth_sram_tile"}

    def test_flattened_tile_count(self):
        # 3 shell text-sites flatten to 7 shell leaves; each shell = 2 tiles
        # -> 14 physical macro tiles planned.
        net = _subblock_first_shell_netlist(64, 32, 5)
        _new, placed = mb.materialize_macro_netlist(net, [_comp_binding()])
        assert len(placed) == 14

    @pytest.mark.skipif(not _HAVE_YOSYS, reason="yosys absent")
    def test_materialized_netlist_passes_hierarchy_check(self, tmp_path):
        import subprocess
        net = _shell_netlist(64, 32, 5)
        new, _placed = mb.materialize_macro_netlist(net, [_comp_binding()])
        bb = tmp_path / "bb.v"
        bb.write_text(
            "(* blackbox *) module sky130_fd_sc_hd__inv_2"
            "(input A, output Y); endmodule\n"
            "(* blackbox *) module synth_sram_tile(\n"
            "  input clk0, input csb0, input web0, input [3:0] wmask0,\n"
            "  input [5:0] addr0, input [31:0] din0, output [31:0] dout0,\n"
            "  input clk1, input csb1, input [5:0] addr1, output [31:0] dout1);\n"
            "endmodule\n")
        netf = tmp_path / "mat.v"
        netf.write_text(new)
        ys = tmp_path / "s.ys"
        ys.write_text(
            f"read_verilog -sv {bb}\nread_verilog -sv {netf}\n"
            "hierarchy -check -top top\ncheck\nstat\n")
        r = subprocess.run(["yosys", "-s", str(ys)], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-800:]
        # both tiles survive elaboration as real instances.
        assert re.search(r"^\s*2\s+synth_sram_tile\b", r.stdout, re.MULTILINE)


class TestCompositionBindings:
    def test_single_bank_plan_becomes_binding(self):
        sp, plan = _synth_plan()
        bindings, err = bg._composition_plan_bindings([(sp, plan)])
        assert err == ""
        assert len(bindings) == 1
        b = bindings[0]
        assert b["name"] == "synth_sram_tile"                      # base macro
        assert (b["width"], b["depth"]) == (64, 32)                # shell geom
        assert b["macro_data_bits"] == 32 and b["macro_words"] == 64
        assert b["composition"]["tiles_wide"] == 2
        assert b["composition"]["tiles_deep"] == 1

    def test_multibank_plan_is_hard_error(self):
        sp, plan = _synth_plan(width=32, depth=128, tiles_wide=1, tiles_deep=2)
        bindings, err = bg._composition_plan_bindings([(sp, plan)])
        assert bindings == []
        assert "MULTI-BANK" in err and "blocker" in err

    def test_binder_carries_plan_when_enabled(self, tmp_path, monkeypatch):
        # _bind_macro_shells_for_backend feeds a resolved plan into macro_bindings
        # (gated ON). Patch the resolver to return a plan.
        from orchestrator.langgraph.macro_registry import BindResult
        sp, plan = _synth_plan()
        monkeypatch.setattr(mr, "bind_macro_shells",
                            lambda *a, **k: BindResult(resolved=[], plans=[(sp, plan)]))
        monkeypatch.setenv("CORESMITH_MACRO_COMPOSE_TILES", "1")
        net = tmp_path / "flat.v"
        net.write_text(_shell_netlist(64, 32, 5))
        bindings, err = bg._bind_macro_shells_for_backend(str(net))
        assert err == ""
        assert [b["name"] for b in bindings] == ["synth_sram_tile"]
        assert bindings[0].get("composition")

    def test_binder_drops_plan_when_disabled(self, tmp_path, monkeypatch):
        from orchestrator.langgraph.macro_registry import BindResult
        sp, plan = _synth_plan()
        monkeypatch.setattr(mr, "bind_macro_shells",
                            lambda *a, **k: BindResult(resolved=[], plans=[(sp, plan)]))
        monkeypatch.setenv("CORESMITH_MACRO_COMPOSE_TILES", "0")   # pre-fix
        net = tmp_path / "flat.v"
        net.write_text(_shell_netlist(64, 32, 5))
        bindings, err = bg._bind_macro_shells_for_backend(str(net))
        assert err == "" and bindings == []                       # plan dropped


class TestCompositionPnrEmission:
    def test_composed_shell_emits_all_tiles(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mr, "discover_macros", lambda *a, **k: {})
        monkeypatch.setenv("CORESMITH_PNR_MACRO_PLACEMENT", "1")
        net = tmp_path / "flat.v"
        net.write_text(_shell_netlist(64, 32, 5))
        lef = _stub_lef(tmp_path, "synth_sram_tile")
        tcl = bh.prepare_pnr_working_copy(
            design_name="blk", netlist_path=str(net),
            sdc_path=str(tmp_path / "s.sdc"), output_dir=str(tmp_path / "out"),
            macro_bindings=[_comp_binding(lef=lef)])
        content = Path(tcl).read_text()
        # one macro NAME (the shared base master) ...
        assert "set macro_names [list synth_sram_tile]" in content
        # ... but TWO placements (one per tile).
        positions = re.findall(r"\{[-\d.]+ [-\d.]+ R0\}", content)
        assert len(positions) == 2
        assert "blk_macro.v" in content
        mat = (tmp_path / "out" / "blk_macro.v").read_text()
        assert mat.count("synth_sram_tile u_tile_r0_c") == 2


def _def_with_counts(tmp_path, counts):
    """A routed DEF placing `counts[master]` instances of each master."""
    p = tmp_path / "routed.def"
    lines, n, i = [], 0, 0
    for master, k in counts.items():
        for _ in range(k):
            lines.append(f"- u{i} {master} + PLACED ( {i*10} {i*10} ) N ;")
            i += 1
            n += 1
    p.write_text(
        "DESIGN blk ;\n"
        f"COMPONENTS {n} ;\n" + "\n".join(lines) + "\nEND COMPONENTS\nEND DESIGN\n")
    return str(p)


class TestCompositionMemoryAbsent:
    def test_full_tile_set_passes(self, tmp_path):
        d = _def_with_counts(tmp_path, {"synth_sram_tile": 2,
                                        "sky130_fd_sc_hd__inv_2": 1})
        assert bg.memory_absent_pnr_error([_comp_binding()], d) is None

    def test_dropped_plan_caught_even_when_other_macro_placed(self, tmp_path):
        # the false-pass shape: a direct macro is placed (so >=1 macro present),
        # but the composition's tiles are ABSENT -> must still hard-fail.
        d = _def_with_counts(tmp_path, {"synth_sram_coeff": 1,
                                        "sky130_fd_sc_hd__inv_2": 3})
        err = bg.memory_absent_pnr_error(
            [_binding("synth_sram_coeff", 16, 256), _comp_binding()], d)
        assert err is not None
        assert "0 tile(s) were PLACED" in err and "2-tile composition" in err

    def test_partial_tile_set_caught(self, tmp_path):
        d = _def_with_counts(tmp_path, {"synth_sram_tile": 1})     # 1 of 2 tiles
        err = bg.memory_absent_pnr_error([_comp_binding()], d)
        assert err is not None
        assert "1 tile(s) were PLACED" in err

    def test_all_absent_uses_zero_placed_message(self, tmp_path):
        d = _def_with_counts(tmp_path, {"sky130_fd_sc_hd__inv_2": 2})
        err = bg.memory_absent_pnr_error([_comp_binding()], d)
        assert err is not None
        assert "0 were PLACED" in err                              # aggregate msg
