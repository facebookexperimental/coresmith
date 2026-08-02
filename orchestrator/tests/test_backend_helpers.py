# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for the backend helper functions (Tcl generation, report parsing,
subprocess wrappers).

Unit tests run without EDA tools (fast, no Nix).
Integration tests require Nix + Sky130 PDK and are marked with
``@pytest.mark.slow`` and ``@pytest.mark.requires_nix``.
"""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import pytest

from orchestrator.langgraph.backend_helpers import (
    CELL_LEF,
    LIBERTY,
    MAGIC_BIN,
    MAGIC_DRC_UM_PER_INTERNAL_UNIT,
    NETGEN_BIN,
    OPENROAD_BIN,
    PROJECT_ROOT,
    TECH_LEF,
    _count_drc_report_violations,
    _drc_rule_layer,
    _parse_lvs_deltas,
    _parse_magic_drc_count,
    drc_macro_interior_exclude_enabled,
    drc_report_fallback_enabled,
    generate_drc_tcl,
    generate_pnr_tcl,
    generate_rcx_tcl,
    macro_bboxes_from_def,
    parse_drc_report,
    parse_openroad_reports,
    parse_pnr_stdout,
    placed_macro_bboxes,
)

# ═══════════════════════════════════════════════════════════════════════════
# Tcl Generation
# ═══════════════════════════════════════════════════════════════════════════

class TestGeneratePnrTcl:
    def test_creates_file(self, tmp_path):
        tcl = generate_pnr_tcl(
            "test_block", "/fake/netlist.v", "/fake/sdc.sdc", str(tmp_path),
        )
        assert Path(tcl).exists()
        assert Path(tcl).name == "pnr_test_block.tcl"

    def test_contains_block_name(self, tmp_path):
        tcl = generate_pnr_tcl(
            "my_adder", "/fake/netlist.v", "/fake/sdc.sdc", str(tmp_path),
        )
        content = Path(tcl).read_text()
        assert "link_design my_adder" in content
        assert "my_adder_routed.def" in content
        assert "my_adder_pnr.v" in content
        assert "my_adder_pwr.v" in content

    def test_contains_make_tracks(self, tmp_path):
        tcl = generate_pnr_tcl(
            "b", "/fake/n.v", "/fake/s.sdc", str(tmp_path),
        )
        content = Path(tcl).read_text()
        assert "make_tracks li1" in content
        assert "make_tracks met1" in content
        assert "make_tracks met5" in content

    def test_contains_set_wire_rc(self, tmp_path):
        tcl = generate_pnr_tcl(
            "b", "/fake/n.v", "/fake/s.sdc", str(tmp_path),
        )
        content = Path(tcl).read_text()
        assert "set_wire_rc -signal -layer met2" in content
        assert "set_wire_rc -clock  -layer met3" in content

    def test_contains_set_routing_layers(self, tmp_path):
        tcl = generate_pnr_tcl(
            "b", "/fake/n.v", "/fake/s.sdc", str(tmp_path),
        )
        content = Path(tcl).read_text()
        assert "set_routing_layers -signal met1-met4 -clock met3-met4" in content

    def test_ends_with_exit(self, tmp_path):
        tcl = generate_pnr_tcl(
            "b", "/fake/n.v", "/fake/s.sdc", str(tmp_path),
        )
        content = Path(tcl).read_text()
        assert content.strip().endswith("exit")

    def test_custom_utilization(self, tmp_path):
        tcl = generate_pnr_tcl(
            "b", "/fake/n.v", "/fake/s.sdc", str(tmp_path),
            utilization=60,
        )
        content = Path(tcl).read_text()
        assert "-utilization 60" in content


class TestGenerateDrcTcl:
    def test_creates_file(self, tmp_path):
        tcl = generate_drc_tcl("test_block", "/fake/routed.def", str(tmp_path))
        assert Path(tcl).exists()
        assert Path(tcl).name == "drc_test_block.tcl"

    def test_contains_block_name(self, tmp_path):
        tcl = generate_drc_tcl("my_block", "/fake/routed.def", str(tmp_path))
        content = Path(tcl).read_text()
        assert "load my_block" in content
        assert "flatten my_block_flat" in content
        assert "my_block.gds" in content
        assert "my_block.spice" in content

    def test_ends_with_quit(self, tmp_path):
        tcl = generate_drc_tcl("b", "/fake/r.def", str(tmp_path))
        content = Path(tcl).read_text()
        assert "quit -noprompt" in content


class TestGenerateRcxTcl:
    def test_creates_file(self, tmp_path):
        tcl = generate_rcx_tcl(
            "test_block", "/fake/routed.def", "/fake/sdc.sdc", str(tmp_path),
        )
        assert Path(tcl).exists()
        assert Path(tcl).name == "rcx_test_block.tcl"

    def test_contains_via_resistance(self, tmp_path):
        tcl = generate_rcx_tcl(
            "b", "/fake/r.def", "/fake/s.sdc", str(tmp_path),
        )
        content = Path(tcl).read_text()
        assert "findLayer mcon" in content
        assert "setResistance 9.249146" in content
        assert "extract_parasitics -ext_model_file" in content

    def test_contains_write_spef(self, tmp_path):
        tcl = generate_rcx_tcl(
            "b", "/fake/r.def", "/fake/s.sdc", str(tmp_path),
        )
        content = Path(tcl).read_text()
        assert "write_spef" in content


# ═══════════════════════════════════════════════════════════════════════════
# Report Parsers
# ═══════════════════════════════════════════════════════════════════════════

class TestParseOpenroadReports:
    def test_parses_wns(self, tmp_path):
        (tmp_path / "timing_wns.rpt").write_text("wns max 0.00\n")
        m = parse_openroad_reports(str(tmp_path))
        assert m["wns_ns"] == 0.0
        assert m["timing_met"] is True

    def test_parses_negative_wns(self, tmp_path):
        (tmp_path / "timing_wns.rpt").write_text("wns max -1.23\n")
        m = parse_openroad_reports(str(tmp_path))
        assert m["wns_ns"] == -1.23
        assert m["timing_met"] is False

    def test_parses_tns(self, tmp_path):
        (tmp_path / "timing_tns.rpt").write_text("tns max 0.00\n")
        m = parse_openroad_reports(str(tmp_path))
        assert m["tns_ns"] == 0.0

    def test_parses_power(self, tmp_path):
        power_rpt = textwrap.dedent("""\
            Group                  Internal  Switching    Leakage      Total
                                      Power      Power      Power      Power (Watts)
            ----------------------------------------------------------------
            Total                  5.93e-05   1.77e-05   3.80e-10   7.70e-05 100.0%
                                      77.0%      23.0%       0.0%
        """)
        (tmp_path / "power.rpt").write_text(power_rpt)
        m = parse_openroad_reports(str(tmp_path))
        assert abs(m["total_power_mw"] - 0.077) < 0.001
        assert m["dynamic_power_mw"] > 0
        assert m["leakage_power_mw"] < 0.001

    def test_parses_setup_slack(self, tmp_path):
        setup_rpt = textwrap.dedent("""\
            Startpoint: a[0]
                       15.40   slack (MET)
        """)
        (tmp_path / "timing_setup.rpt").write_text(setup_rpt)
        m = parse_openroad_reports(str(tmp_path))
        assert m["setup_slack_ns"] == 15.40

    def test_empty_dir(self, tmp_path):
        m = parse_openroad_reports(str(tmp_path))
        assert m["wns_ns"] == 0.0
        # No timing_wns.rpt -> timing is unmeasured -> None (fail-closed,
        # A-Fix 2g), NOT an implicit pass.
        assert m["timing_met"] is None


class TestParsePnrStdout:
    def test_parses_area_format2(self):
        stdout = "Design area 955 um^2 49% utilization.\n"
        m = parse_pnr_stdout(stdout)
        assert m["design_area_um2"] == 955.0
        assert m["utilization_pct"] == 49.0

    def test_parses_wns_from_stdout(self):
        stdout = "wns max 0.00\ntns max 0.00\n"
        m = parse_pnr_stdout(stdout)
        assert m["wns_ns"] == 0.0
        assert m["tns_ns"] == 0.0

    def test_parses_power_from_stdout(self):
        stdout = "Total  5.93e-05   1.77e-05   3.80e-10   7.70e-05 100.0%\n"
        m = parse_pnr_stdout(stdout)
        assert abs(m["total_power_mw"] - 0.077) < 0.001

    def test_empty_stdout(self):
        m = parse_pnr_stdout("")
        assert m["design_area_um2"] == 0.0


class TestParseDrcReport:
    def test_clean(self, tmp_path):
        rpt = tmp_path / "drc.rpt"
        rpt.write_text("Design: test\nDRC count: 0\n")
        r = parse_drc_report(str(rpt))
        assert r["clean"] is True
        assert r["violation_count"] == 0

    def test_violations(self, tmp_path):
        # parse_drc_report extracts violations from Magic's
        # ``{rule_name} {coords} ...`` syntax -- the rule name is in the
        # first set of braces followed immediately by another `{`.
        rpt = tmp_path / "drc.rpt"
        rpt.write_text(
            "Design: test\n"
            "DRC count: 3\n"
            "{via spacing} {0 0 1 1}\n"
            "{metal width} {2 2 3 3}\n"
            "{enclosure} {4 4 5 5}\n"
        )
        r = parse_drc_report(str(rpt))
        assert r["clean"] is False
        assert r["violation_count"] == 3
        assert len(r["violations"]) == 3

    def test_missing_file(self):
        r = parse_drc_report("/nonexistent/file.rpt")
        assert r["clean"] is False
        assert r["violation_count"] == -1

    def test_native_listall_format_no_macros(self, tmp_path):
        # Magic native `drc listall why <file>` shape: "<cell> <count>",
        # dashed separators, why-string headers, bare "x1 y1 x2 y2" tiles.
        rpt = tmp_path / "drc.rpt"
        rpt.write_text(
            "synth_top 2\n"
            "----------------------------------------\n"
            "Metal4 minimum area < 0.24um^2 (met4.4a)\n"
            "----------------------------------------\n"
            " 25000 25000 25076 25076\n"
            " 80000 80000 80076 80076\n"
        )
        r = parse_drc_report(str(rpt))
        assert r["clean"] is False
        assert r["violation_count"] == 2


# ---------------------------------------------------------------------------
# Signed-off hard-macro interior exclusion (DRC gate)
# ---------------------------------------------------------------------------

# All fixtures are GENERIC synthetic collateral -- no benchmark/design/golden
# names. Coordinate convention (VERIFIED against a real routed report): the
# Magic native report emits tiles in INTERNAL units of 0.005 um each, so a macro
# bbox at [100,150] x [100,150] um occupies internal [20000,30000] on each axis.
_MACRO_LAYERS = frozenset({"met1", "met2", "met3", "met4"})


def _macro_bbox_100_150(tag="generic_macro_a"):
    return placed_macro_bboxes(
        [{"x": 100.0, "y": 100.0, "w": 50.0, "h": 50.0, "orient": "N", "tag": tag}]
    )


def _synthetic_native_report(tmp_path):
    """A native report with: (a) a met4 tile INSIDE the macro, (b) a met4 tile
    OUTSIDE any macro, (c) a met5 tile INSIDE the macro."""
    rpt = tmp_path / "drc.rpt"
    rpt.write_text(
        "synth_top 3\n"
        "----------------------------------------\n"
        "Metal4 minimum area < 0.24um^2 (met4.4a)\n"
        "----------------------------------------\n"
        " 25000 25000 25076 25076\n"   # center ~125um -> INSIDE [100,150]
        " 80000 80000 80076 80076\n"   # center ~400um -> OUTSIDE
        "----------------------------------------\n"
        "Metal5 minimum spacing (met5.2)\n"
        "----------------------------------------\n"
        " 25000 25000 25076 25076\n"   # INSIDE, but met5 is not obstructed
    )
    return rpt


class TestDrcMacroInteriorExclusion:
    def test_default_flag_on(self, monkeypatch):
        monkeypatch.delenv("CORESMITH_DRC_MACRO_INTERIOR_EXCLUDE", raising=False)
        assert drc_macro_interior_exclude_enabled() is True

    def test_flag_off(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_DRC_MACRO_INTERIOR_EXCLUDE", "0")
        assert drc_macro_interior_exclude_enabled() is False

    def test_in_macro_met4_excluded_outside_and_met5_kept(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_DRC_MACRO_INTERIOR_EXCLUDE", "1")
        rpt = _synthetic_native_report(tmp_path)
        r = parse_drc_report(str(rpt), macro_bboxes=_macro_bbox_100_150())
        # (a) the in-macro met4 tile is excluded ...
        assert r["excluded_count"] == 1
        assert r["excluded_detail"] == {"generic_macro_a": 1}
        # ... while (b) the outside-macro met4 tile and (c) the in-macro met5
        # tile both survive -> 2 counted, gate still dirty (honest).
        assert r["violation_count"] == 2
        assert r["clean"] is False

    def test_env_off_counts_everything(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_DRC_MACRO_INTERIOR_EXCLUDE", "0")
        rpt = _synthetic_native_report(tmp_path)
        r = parse_drc_report(str(rpt), macro_bboxes=_macro_bbox_100_150())
        assert r["violation_count"] == 3
        assert "excluded_count" not in r
        assert r["clean"] is False

    def test_no_bboxes_no_exclusion(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_DRC_MACRO_INTERIOR_EXCLUDE", "1")
        rpt = _synthetic_native_report(tmp_path)
        r = parse_drc_report(str(rpt), macro_bboxes=None)
        assert r["violation_count"] == 3
        assert "excluded_count" not in r

    def test_all_in_macro_becomes_clean(self, tmp_path, monkeypatch):
        # Every met4 tile inside the macro -> post-exclusion count 0 -> clean.
        monkeypatch.setenv("CORESMITH_DRC_MACRO_INTERIOR_EXCLUDE", "1")
        rpt = tmp_path / "drc.rpt"
        rpt.write_text(
            "synth_top 2\n"
            "----------------------------------------\n"
            "Metal4 minimum area < 0.24um^2 (met4.4a)\n"
            "----------------------------------------\n"
            " 25000 25000 25076 25076\n"
            " 26000 26000 26076 26076\n"
        )
        r = parse_drc_report(str(rpt), macro_bboxes=_macro_bbox_100_150())
        assert r["excluded_count"] == 2
        assert r["violation_count"] == 0
        assert r["clean"] is True

    def test_unknown_layer_never_excluded(self, tmp_path, monkeypatch):
        # A rule with no identifiable metal layer is fail-closed (kept).
        monkeypatch.setenv("CORESMITH_DRC_MACRO_INTERIOR_EXCLUDE", "1")
        rpt = tmp_path / "drc.rpt"
        rpt.write_text(
            "synth_top 1\n"
            "----------------------------------------\n"
            "Nwell spacing rule\n"
            "----------------------------------------\n"
            " 25000 25000 25076 25076\n"
        )
        r = parse_drc_report(str(rpt), macro_bboxes=_macro_bbox_100_150())
        assert r["violation_count"] == 1
        assert r.get("excluded_count", 0) == 0


class TestDrcCoordUnitConversion:
    def test_constant_is_internal_unit(self):
        # Magic internal unit = 0.005 um (um x 200), not um/1000.
        assert MAGIC_DRC_UM_PER_INTERNAL_UNIT == 0.005

    def test_internal_units_place_tile_at_correct_um(self, tmp_path, monkeypatch):
        # A tile centered at internal-coord 25000 is at 125 um under the correct
        # x0.005 conversion (inside macro [100,150] -> excluded). Under the WRONG
        # um/1000 rule it would be 25 um -- outside [100,150] -> kept. So the
        # tile being excluded proves x0.005 is used.
        monkeypatch.setenv("CORESMITH_DRC_MACRO_INTERIOR_EXCLUDE", "1")
        rpt = tmp_path / "drc.rpt"
        rpt.write_text(
            "synth_top 1\n"
            "----------------------------------------\n"
            "Metal4 minimum area < 0.24um^2 (met4.4a)\n"
            "----------------------------------------\n"
            " 25000 25000 25076 25076\n"
        )
        # Macro at the x0.005 position -> excluded.
        r_correct = parse_drc_report(str(rpt), macro_bboxes=_macro_bbox_100_150())
        assert r_correct["excluded_count"] == 1
        # Macro at the (wrong) um/1000 position [24,26] -> NOT excluded, proving
        # the parser does not use um/1000.
        wrong = placed_macro_bboxes(
            [{"x": 24.0, "y": 24.0, "w": 2.0, "h": 2.0, "orient": "N", "tag": "m"}]
        )
        r_wrong = parse_drc_report(str(rpt), macro_bboxes=wrong)
        assert r_wrong.get("excluded_count", 0) == 0


class TestDrcMicronReport:
    """The `drc listall why` script is LLM-authored per run, so some runs emit
    tiles in already-scaled MICRONS (decimals, via `cif scale out`) instead of
    Magic INTERNAL units (integers). The parser must read BOTH: an integer-only
    tile regex silently dropped every micron tile and returned a FALSE CLEAN
    (count 0), so the macro-interior exclusion never ran. Fixtures are generic
    synthetic collateral -- no design/benchmark names.
    """

    def _micron_report(self, tmp_path):
        """A micron (decimal) report with (a) a met4 tile at the BOTTOM EDGE
        band of a macro placed at [100,150]x[100,150] um -- center 100.19 um,
        just inside the y1=100 boundary; (b) a met4 tile far OUTSIDE; (c) a met5
        tile inside the macro (met5 is never obstructed)."""
        rpt = tmp_path / "drc_um.rpt"
        rpt.write_text(
            "DRC errors for cell synth_top\n"
            "----------------------------------------\n"
            "Metal4 minimum area < 0.24um^2 (met4.4a)\n"
            "----------------------------------------\n"
            " 120.000 100.000 120.380 100.380\n"   # center (120.19,100.19) INSIDE, bottom-edge band
            " 400.000 400.000 400.380 400.380\n"   # center ~400 um OUTSIDE
            "----------------------------------------\n"
            "Metal5 minimum spacing (met5.2)\n"
            "----------------------------------------\n"
            " 120.000 100.000 120.380 100.380\n"   # INSIDE, but met5 not obstructed
        )
        return rpt

    def test_micron_tiles_are_parsed_not_false_clean(self, tmp_path, monkeypatch):
        # With NO bboxes the parser must count all 3 micron tiles -- the old
        # integer-only regex returned 0/clean here (the jpeg false-clean bug).
        monkeypatch.setenv("CORESMITH_DRC_MACRO_INTERIOR_EXCLUDE", "1")
        rpt = self._micron_report(tmp_path)
        r = parse_drc_report(str(rpt), macro_bboxes=None)
        assert r["violation_count"] == 3
        assert r["clean"] is False

    def test_micron_edge_band_excluded_outside_and_met5_kept(self, tmp_path, monkeypatch):
        # Micron path scales by 1.0 (not x0.005): the in-macro met4 edge-band
        # tile is dropped; the outside met4 and the in-macro met5 both survive.
        monkeypatch.setenv("CORESMITH_DRC_MACRO_INTERIOR_EXCLUDE", "1")
        rpt = self._micron_report(tmp_path)
        r = parse_drc_report(str(rpt), macro_bboxes=_macro_bbox_100_150())
        assert r["excluded_count"] == 1
        assert r["excluded_detail"] == {"generic_macro_a": 1}
        assert r["violation_count"] == 2      # outside met4 + in-macro met5
        assert r["clean"] is False

    def test_micron_all_in_macro_becomes_clean(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_DRC_MACRO_INTERIOR_EXCLUDE", "1")
        rpt = tmp_path / "drc_um.rpt"
        rpt.write_text(
            "DRC errors for cell synth_top\n"
            "----------------------------------------\n"
            "Metal4 minimum area < 0.24um^2 (met4.4a)\n"
            "----------------------------------------\n"
            " 120.000 100.000 120.380 100.380\n"
            " 130.000 140.000 130.380 140.380\n"
        )
        r = parse_drc_report(str(rpt), macro_bboxes=_macro_bbox_100_150())
        assert r["excluded_count"] == 2
        assert r["violation_count"] == 0
        assert r["clean"] is True

    def test_micron_env_off_counts_everything(self, tmp_path, monkeypatch):
        # Both-branch: gate OFF -> every micron tile counted, no exclusion.
        monkeypatch.setenv("CORESMITH_DRC_MACRO_INTERIOR_EXCLUDE", "0")
        rpt = self._micron_report(tmp_path)
        r = parse_drc_report(str(rpt), macro_bboxes=_macro_bbox_100_150())
        assert r["violation_count"] == 3
        assert "excluded_count" not in r

    def test_micron_scale_not_confused_with_internal(self, tmp_path, monkeypatch):
        # A micron tile at 120 um must NOT be scaled by 0.005 (which would put it
        # at 0.6 um, outside the macro). Proven by it being excluded from a macro
        # at [100,150] um only under the correct x1.0 micron scale.
        monkeypatch.setenv("CORESMITH_DRC_MACRO_INTERIOR_EXCLUDE", "1")
        rpt = tmp_path / "drc_um.rpt"
        rpt.write_text(
            "DRC errors for cell synth_top\n"
            "----------------------------------------\n"
            "Metal4 minimum area < 0.24um^2 (met4.4a)\n"
            "----------------------------------------\n"
            " 124.810 124.810 125.190 125.190\n"   # center 125 um -> inside [100,150]
        )
        r = parse_drc_report(str(rpt), macro_bboxes=_macro_bbox_100_150())
        assert r["excluded_count"] == 1
        assert r["violation_count"] == 0


class TestPlacedMacroBboxes:
    def test_north_orientation_uses_size(self):
        b = placed_macro_bboxes([{"x": 10.0, "y": 20.0, "w": 30.0, "h": 40.0, "orient": "N"}])
        assert (b[0]["x1"], b[0]["y1"], b[0]["x2"], b[0]["y2"]) == (10.0, 20.0, 40.0, 60.0)
        assert b[0]["layers"] == _MACRO_LAYERS  # default obstructed set

    def test_rotated_orientation_swaps_wh(self):
        # R90 (and E/W/FE/FW) swap w<->h.
        b = placed_macro_bboxes([(10.0, 20.0, "R90", 30.0, 40.0)])
        assert (b[0]["x1"], b[0]["y1"], b[0]["x2"], b[0]["y2"]) == (10.0, 20.0, 50.0, 50.0)

    def test_flipped_orientation_keeps_wh(self):
        # FS / S / MX are mirror/180 -- footprint dims unchanged.
        b = placed_macro_bboxes([(10.0, 20.0, "FS", 30.0, 40.0)])
        assert (b[0]["x2"], b[0]["y2"]) == (40.0, 60.0)

    def test_sequence_and_custom_layers(self):
        b = placed_macro_bboxes([(0.0, 0.0, "N", 5.0, 5.0, {"met2"}, "tagX")])
        assert b[0]["layers"] == frozenset({"met2"})
        assert b[0]["tag"] == "tagX"


class TestMacroBboxesFromDef:
    def _make_lef(self, tmp_path, obs_layers):
        lef = tmp_path / "generic_macro_a.lef"
        obs = "\n".join(f"      LAYER {lyr} ;\n        RECT 0 0 1 1 ;" for lyr in obs_layers)
        lef.write_text(
            "MACRO generic_macro_a\n"
            "  SIZE 50 BY 50 ;\n"
            "  PIN foo\n    USE SIGNAL ;\n  END foo\n"
            f"  OBS\n{obs}\n  END\n"
            "END generic_macro_a\n"
        )
        return lef

    def _registry(self, lef):
        from orchestrator.langgraph.macro_registry import MacroInfo
        return {
            "generic_macro_a": MacroInfo(
                name="generic_macro_a", lef=str(lef),
                width_um=50.0, height_um=50.0,
            )
        }

    def _def(self, tmp_path):
        d = tmp_path / "routed.def"
        d.write_text(
            "VERSION 5.8 ;\n"
            "UNITS DISTANCE MICRONS 1000 ;\n"
            "DIEAREA ( 0 0 ) ( 500000 500000 ) ;\n"
            "COMPONENTS 2 ;\n"
            "    - u_std/cell_1 sky130_fd_sc_hd__inv_2 + PLACED ( 5000 5000 ) N ;\n"
            "    - u_mem/u_macro generic_macro_a + FIXED ( 100000 100000 ) N ;\n"
            "END COMPONENTS\n"
            "END DESIGN\n"
        )
        return d

    def test_reads_placed_macro_bbox_in_um(self, tmp_path):
        lef = self._make_lef(tmp_path, ["met1", "met2", "met3", "met4"])
        d = self._def(tmp_path)
        boxes = macro_bboxes_from_def(str(d), registry=self._registry(lef))
        assert len(boxes) == 1  # the std cell (unknown master) is ignored
        b = boxes[0]
        # 100000 dbu / 1000 = 100 um origin, SIZE 50 -> [100,150]
        assert (b["x1"], b["y1"], b["x2"], b["y2"]) == (100.0, 100.0, 150.0, 150.0)
        assert b["tag"] == "generic_macro_a"
        assert b["layers"] == _MACRO_LAYERS

    def test_obs_met5_is_capped_out(self, tmp_path):
        # Even if the LEF OBS lists met5, exclusion caps at met1-met4 (a real
        # met5-over-macro violation must still be counted).
        lef = self._make_lef(tmp_path, ["met1", "met2", "met3", "met4", "met5"])
        d = self._def(tmp_path)
        boxes = macro_bboxes_from_def(str(d), registry=self._registry(lef))
        assert "met5" not in boxes[0]["layers"]
        assert boxes[0]["layers"] == _MACRO_LAYERS

    def test_missing_def_returns_empty(self, tmp_path):
        assert macro_bboxes_from_def("/nonexistent/x.def", registry={}) == []

    def test_no_registry_returns_empty(self, tmp_path):
        d = self._def(tmp_path)
        assert macro_bboxes_from_def(str(d), registry={}) == []


class TestDrcReportFallback:
    """Magic emits a BLANK ``DRC violations:`` line (and blank ``DRC count:`` in
    the report header) when ``drc listall count`` returns nothing, even though
    ``drc listall why`` still writes every violation rect to magic_drc.rpt. The
    stdout-only parser then returns a FALSE-CLEAN 0 over a report holding
    thousands of real violations (live proof: jpeg std_signoff printed
    ``DRC violations:`` blank while magic_drc.rpt held 26,695 rects). The
    fallback recounts from the report. Fixtures mirror the real
    ``drc listall why`` Tcl-brace format ``{rule} {{x1 y1 x2 y2} ...}``.
    """

    # Blank count header + Tcl-brace rects (the real jpeg magic_drc.rpt shape).
    _BLANK_STDOUT = "Total DRC errors found: 0\nDRC violations: \n"
    _REPORT_3 = (
        "Design: synth_top\n"
        "DRC count: \n"
        "{Local interconnect spacing < 0.17um (li.3)} "
        "{{10961 96611 10977 96627} {10961 96611 10981 96627} "
        "{28515 242879 28529 242899}}\n"
    )
    _REPORT_CLEAN = "Design: synth_top\nDRC count: 0\n\n"

    def test_count_report_violations_brace_form(self):
        assert _count_drc_report_violations(self._REPORT_3) == 3

    def test_count_report_violations_clean(self):
        assert _count_drc_report_violations(self._REPORT_CLEAN) == 0

    def test_count_report_violations_bare_tiles(self):
        rpt = (
            "cellname 2\n----------\nrule why\n----------\n"
            " 10 20 30 40\n 50 60 70 80\n"
        )
        assert _count_drc_report_violations(rpt) == 2

    def test_blank_stdout_falls_back_to_report_count(self, monkeypatch):
        # Gate ON (default): blank stdout + report with 3 rects -> 3, not 0.
        monkeypatch.delenv("CORESMITH_DRC_REPORT_FALLBACK", raising=False)
        assert _parse_magic_drc_count(self._BLANK_STDOUT, self._REPORT_3) == 3

    def test_blank_stdout_no_report_is_legacy_zero(self):
        # No report text -> unchanged legacy behavior (0 from blank count line).
        assert _parse_magic_drc_count(self._BLANK_STDOUT) == 0
        assert _parse_magic_drc_count(self._BLANK_STDOUT, "") == 0

    def test_gate_off_preserves_false_clean(self, monkeypatch):
        # Gate OFF: raw stdout-only count (the legacy false-clean 0) preserved.
        monkeypatch.setenv("CORESMITH_DRC_REPORT_FALLBACK", "0")
        assert _parse_magic_drc_count(self._BLANK_STDOUT, self._REPORT_3) == 0

    def test_clean_report_stays_zero(self, monkeypatch):
        monkeypatch.delenv("CORESMITH_DRC_REPORT_FALLBACK", raising=False)
        assert _parse_magic_drc_count(self._BLANK_STDOUT, self._REPORT_CLEAN) == 0

    def test_explicit_stdout_count_wins(self, monkeypatch):
        # A real numeric stdout count is authoritative; no fallback override.
        monkeypatch.delenv("CORESMITH_DRC_REPORT_FALLBACK", raising=False)
        assert _parse_magic_drc_count("DRC violations: 5\n", self._REPORT_3) == 5

    def test_gate_helper_default_on(self, monkeypatch):
        monkeypatch.delenv("CORESMITH_DRC_REPORT_FALLBACK", raising=False)
        assert drc_report_fallback_enabled() is True

    def test_gate_helper_explicit_off(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_DRC_REPORT_FALLBACK", "0")
        assert drc_report_fallback_enabled() is False


class TestDrcRuleLayer:
    def test_magic_layer_name(self):
        assert _drc_rule_layer("Metal4 minimum area < 0.24um^2 (met4.4a)") == "met4"

    def test_met5(self):
        assert _drc_rule_layer("Metal5 minimum spacing (met5.2)") == "met5"

    def test_human_name_only(self):
        assert _drc_rule_layer("Metal3 width") == "met3"

    def test_no_layer(self):
        assert _drc_rule_layer("Nwell spacing rule") is None
        assert _drc_rule_layer("") is None


class TestParseMagicDrcCount:
    def test_normal_count(self):
        assert _parse_magic_drc_count("DRC violations: 5\n") == 5

    def test_zero_count(self):
        assert _parse_magic_drc_count("DRC violations: 0\n") == 0

    def test_empty_count(self):
        """Magic prints 'DRC violations: ' (no number) when count is 0."""
        assert _parse_magic_drc_count("DRC violations: \n") == 0

    def test_multiple_lines(self):
        stdout = "stuff\nDRC violations: \nmore stuff\nDRC violations: \n"
        assert _parse_magic_drc_count(stdout) == 0

    def test_no_match(self):
        assert _parse_magic_drc_count("no DRC info here\n") == -1


class TestParseLvsDeltas:
    def test_matching(self):
        stdout = "Circuit 1: 117 devices, 50 nets\nCircuit 2: 117 devices, 50 nets\n"
        d, n = _parse_lvs_deltas(stdout)
        assert d == 0
        assert n == 0

    def test_mismatch(self):
        stdout = "Circuit 1: 117 devices, 50 nets\nCircuit 2: 118 devices, 56 nets\n"
        d, n = _parse_lvs_deltas(stdout)
        assert d == 1
        assert n == 6

    def test_no_data(self):
        d, n = _parse_lvs_deltas("nothing here")
        assert d == 0
        assert n == 0


# ═══════════════════════════════════════════════════════════════════════════
# Tool Binary Resolution
# ═══════════════════════════════════════════════════════════════════════════

class TestToolResolution:
    def test_openroad_binary_resolved(self):
        assert OPENROAD_BIN.endswith("openroad-nix.sh")
        assert Path(OPENROAD_BIN).exists()

    def test_magic_binary_resolved(self):
        assert MAGIC_BIN.endswith("magic-nix.sh")
        assert Path(MAGIC_BIN).exists()

    def test_netgen_binary_resolved(self):
        assert NETGEN_BIN.endswith("netgen-nix.sh")
        assert Path(NETGEN_BIN).exists()

    @pytest.mark.requires_nix
    def test_pdk_files_exist(self):
        # PDK files (Sky130) are provisioned by the Nix-based dev environment;
        # CI runners that don't have Nix on PATH won't have them either.
        assert TECH_LEF.exists(), f"Tech LEF not found: {TECH_LEF}"
        assert CELL_LEF.exists(), f"Cell LEF not found: {CELL_LEF}"
        assert LIBERTY.exists(), f"Liberty not found: {LIBERTY}"


# ═══════════════════════════════════════════════════════════════════════════
# Integration Tests (require Nix + PDK -- slow)
# ═══════════════════════════════════════════════════════════════════════════

_NETLIST = PROJECT_ROOT / "syn" / "output" / "adder_16bit" / "adder_16bit_netlist.v"
_SDC = PROJECT_ROOT / "syn" / "output" / "adder_16bit" / "adder_16bit.sdc"
_HAS_NETLIST = _NETLIST.exists() and _SDC.exists()
_HAS_NIX = shutil.which("nix") is not None

requires_nix = pytest.mark.skipif(
    not _HAS_NIX, reason="Nix not installed"
)
requires_netlist = pytest.mark.skipif(
    not _HAS_NETLIST, reason="adder_16bit netlist not found (run synthesis first)"
)


@requires_nix
@requires_netlist
class TestPnRIntegration:
    """Run real OpenROAD PnR on adder_16bit. Slow (~15s)."""

    @pytest.mark.slow
    def test_run_pnr_flow(self, tmp_path):
        from orchestrator.langgraph.backend_helpers import run_pnr_flow

        out_dir = str(tmp_path / "pnr")
        result = run_pnr_flow(
            "adder_16bit", str(_NETLIST), str(_SDC), out_dir,
        )

        assert result["success"] is True
        assert result["timing_met"] is True
        assert result["design_area_um2"] > 0
        assert result["total_power_mw"] > 0
        assert result["route_drc_violations"] == 0
        assert Path(result["routed_def_path"]).exists()
        assert Path(result["pnr_verilog_path"]).exists()
        assert Path(result["pwr_verilog_path"]).exists()
        assert result.get("log_path")


@requires_nix
@requires_netlist
class TestDrcIntegration:
    """Run real Magic DRC on adder_16bit. Slow (~5s)."""

    @pytest.mark.slow
    def test_run_drc_flow(self, tmp_path):
        from orchestrator.langgraph.backend_helpers import run_drc_flow, run_pnr_flow

        out_dir = str(tmp_path / "pnr")
        pnr = run_pnr_flow(
            "adder_16bit", str(_NETLIST), str(_SDC), out_dir,
        )
        assert pnr["success"]

        drc = run_drc_flow(
            "adder_16bit", pnr["routed_def_path"], out_dir,
        )

        assert drc["clean"] is True
        assert drc["violation_count"] == 0
        assert Path(drc["gds_path"]).exists()
        assert Path(drc["spice_path"]).exists()

        # No stray .ext files in project root
        ext_files = list(PROJECT_ROOT.glob("*.ext"))
        assert len(ext_files) == 0, f"Stray .ext files: {ext_files}"


@requires_nix
@requires_netlist
class TestLvsIntegration:
    """Run real Netgen LVS on adder_16bit. Slow (~20s total)."""

    @pytest.mark.slow
    def test_run_lvs_flow(self, tmp_path):
        from orchestrator.langgraph.backend_helpers import (
            run_drc_flow,
            run_lvs_flow,
            run_pnr_flow,
        )

        out_dir = str(tmp_path / "pnr")
        pnr = run_pnr_flow(
            "adder_16bit", str(_NETLIST), str(_SDC), out_dir,
        )
        assert pnr["success"]

        drc = run_drc_flow(
            "adder_16bit", pnr["routed_def_path"], out_dir,
        )
        assert drc["clean"]

        lvs = run_lvs_flow(
            "adder_16bit", drc["spice_path"],
            pnr["pwr_verilog_path"], out_dir,
        )

        assert lvs["match"] is True
        # Expected tap cell delta
        assert lvs["device_delta"] <= 2
        assert Path(lvs["report_path"]).exists()


# ---------------------------------------------------------------------------
# Synthesis attempt-history retention (run3-followups #3)
# ---------------------------------------------------------------------------
#
# The flat-synth driver retries Yosys inside its own LLM step. A live run logged
# "Synthesis succeeded on attempt 2" and attempt 1's failure reason was retained
# NOWHERE -- not in attempt_history, not in previous_error, not in
# synth_result.json. A driver that heals itself in silence teaches nobody: the
# same script defect recurs every run and the only trace is a discarded
# transcript.

class TestSynthAttemptHistory:
    def _collect(self, **kw):
        from orchestrator.langgraph.backend_helpers import (
            collect_synth_attempt_history,
        )
        kw.setdefault("now", "2026-07-31T00:00:00")
        return collect_synth_attempt_history(kw.pop("result", {}), **kw)

    def test_clean_first_attempt_records_nothing(self, tmp_path):
        assert self._collect(
            result={"success": True},
            output_dir=str(tmp_path),
            llm_reply="Synthesis succeeded.",
        ) == []

    def test_driver_reported_failures_are_retained_on_a_SUCCESS(self, tmp_path):
        h = self._collect(
            result={"success": True, "attempt_history": [
                {"attempt": 1, "error_summary": "ERROR: Module `cs_sram_1rw' "
                                                "is not part of the design"},
            ]},
            output_dir=str(tmp_path),
            llm_reply="Synthesis succeeded on attempt 2.",
        )
        assert [e["attempt"] for e in h] == [1]
        assert "cs_sram_1rw" in h[0]["error_summary"]
        assert h[0]["source"] == "driver_reported"
        assert h[0]["unrecorded"] is False

    def test_yosys_logs_on_disk_are_harvested_without_the_driver(self, tmp_path):
        """Deterministic evidence: the driver need not be honest about its own
        retries for the failure reason to survive."""
        (tmp_path / "synth_attempt1.log").write_text(
            "1. Executing Verilog frontend\n"
            "ERROR: syntax error, unexpected TOK_ID at line 12\n")
        h = self._collect(result={"success": True}, output_dir=str(tmp_path),
                          llm_reply="done")
        assert len(h) == 1
        assert h[0]["source"] == "yosys_log"
        assert "syntax error" in h[0]["error_summary"]
        assert "synth_attempt1.log" in h[0]["error_summary"]

    def test_a_claimed_retry_with_no_evidence_is_recorded_as_NOT_RETAINED(
            self, tmp_path):
        """THE defect. The driver says it took two attempts and supplies nothing
        about the first. The gap becomes a record instead of a silence."""
        h = self._collect(result={"success": True}, output_dir=str(tmp_path),
                          llm_reply="Synthesis succeeded on attempt 2.")
        assert len(h) == 1
        assert h[0]["attempt"] == 1
        assert h[0]["unrecorded"] is True
        assert "not retained" in h[0]["error_summary"]

    def test_real_evidence_wins_over_the_not_retained_placeholder(self, tmp_path):
        (tmp_path / "a.log").write_text("ERROR: cannot open liberty file\n")
        h = self._collect(result={"success": True}, output_dir=str(tmp_path),
                          llm_reply="succeeded on attempt 3")
        assert [e["attempt"] for e in h] == [1, 2]
        assert h[0]["unrecorded"] is False and "liberty" in h[0]["error_summary"]
        assert h[1]["unrecorded"] is True

    def test_node_level_prior_failure_is_folded_in(self, tmp_path):
        h = self._collect(result={"success": True}, output_dir=str(tmp_path),
                          prior_error="hierarchy -check failed on top",
                          node_attempt=2, llm_reply="ok")
        assert [e["source"] for e in h] == ["node_previous_error"]
        assert h[0]["attempt"] == 1

    def test_collector_never_raises_on_garbage(self, tmp_path):
        assert self._collect(result={"attempt_history": "not-a-list"},
                             output_dir="/nonexistent/dir",
                             llm_reply=None) == []

    def test_persist_merges_into_the_synth_result_artifact(self, tmp_path):
        import json as _json

        from orchestrator.langgraph.backend_helpers import (
            persist_synth_attempt_history,
        )
        p = tmp_path / "synth_result.json"
        p.write_text(_json.dumps({"success": True, "gate_count": 42}))
        hist = [{"attempt": 1, "error_summary": "ERROR: boom",
                 "source": "yosys_log", "timestamp": "t", "unrecorded": False}]
        assert persist_synth_attempt_history(str(p), hist) is True
        data = _json.loads(p.read_text())
        assert data["gate_count"] == 42                 # existing keys survive
        assert data["attempt_history"] == hist
        assert data["attempt_failures"] == 1
        assert data["attempt_history_unrecorded"] == 0

    def test_describe_names_every_attempt(self):
        from orchestrator.langgraph.backend_helpers import (
            describe_synth_attempt_history,
        )
        assert describe_synth_attempt_history([]) == ""
        txt = describe_synth_attempt_history([
            {"attempt": 1, "error_summary": "boom", "source": "yosys_log",
             "timestamp": "t", "unrecorded": False},
            {"attempt": 2, "error_summary": "(not retained) ...",
             "source": "unrecorded", "timestamp": "t", "unrecorded": True},
        ])
        assert "attempt 1" in txt and "attempt 2 [NOT RETAINED]" in txt
