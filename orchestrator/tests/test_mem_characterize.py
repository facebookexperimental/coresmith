# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the PDK memory-characterization agent (mem_characterize).

Hermetic: the parsing helpers are fed captured tool output, and the model /
recommendation logic is fitted on synthetic-but-physically-shaped MemPoints, so
NO yosys / OpenROAD / PDK is required to run these. (The real end-to-end sweep
is exercised separately by the standalone `sweep` entrypoint.)
"""
from __future__ import annotations

import pytest

from orchestrator.langgraph import mem_characterize as mc
from orchestrator.langgraph.mem_characterize import MemPoint

# --------------------------------------------------------------------------
# Parsing: yosys stat table + STA arrival
# --------------------------------------------------------------------------

_YOSYS_STAT = """
=== mem_char_top ===

   Number of wires:               1234
   Chip area for module '\\mem_char_top': 311267.280000
     of which used for sequential elements: 246956.851200 (79.34%)

     8224 2.47E+05   sky130_fd_sc_hd__edfxtp_1
      250  938.4     sky130_fd_sc_hd__nand2_1
     1014 3806.15    sky130_fd_sc_hd__nor2_1
      316 2.77E+03   sky130_fd_sc_hd__o221ai_1
"""


class TestParsing:
    def test_stat_table_cells_and_ff(self):
        total, ff = mc._parse_stat_table(_YOSYS_STAT)
        # 8224 + 250 + 1014 + 316 = 9804 total cells; 8224 are FFs
        assert total == 9804
        assert ff == 8224

    def test_area_regex(self):
        m = mc._YOSYS_AREA_RE.search(_YOSYS_STAT)
        assert m and abs(float(m.group(1)) - 311267.28) < 1e-3

    def test_arrival_parse(self):
        rpt = (
            "  Delay    Time   Description\n"
            "   5.02    5.02 ^ addr[1] (in)\n"
            "   3.00    8.02 v _4375_/X (mux4)\n"
            "           8.98   data arrival time\n"
        )
        m = mc._ARRIVAL_RE.search(rpt)
        assert m and abs(float(m.group(1)) - 8.98) < 1e-6

    def test_lef_area(self, tmp_path):
        lef = tmp_path / "m.lef"
        lef.write_text("MACRO m\n   SIZE 479.78 BY 397.5 ;\nEND m\n")
        area = mc._lef_area_um2(str(lef))
        assert area and abs(area - 479.78 * 397.5) < 1e-3

    def test_lib_access_time(self, tmp_path):
        lib = tmp_path / "m.lib"
        lib.write_text(
            'pin(dout0){ timing(){ related_pin:"clk0";\n'
            '  cell_rise(t){ values("0.339, 0.368, 0.484",\n'
            '                       "0.339, 0.368, 0.484"); }\n'
            '  cell_fall(t){ values("0.300, 0.300, 0.300"); } } }\n'
        )
        acc = mc._lib_access_time_ns(str(lib))
        assert acc and abs(acc - 0.484) < 1e-6


# --------------------------------------------------------------------------
# RTL emission: behavioral wrapper, comb-vs-registered read
# --------------------------------------------------------------------------

class TestEmit:
    def test_flop_comb_has_comb_read_mux(self):
        v = mc._emit_flop_top("1rw", 32, 256, comb_read=True)
        # comb read = combinational assign from the shadow array (the N:1 mux)
        assert "assign rdata = shadow[addr];" in v
        assert "cs_sram_1rw " in v

    def test_registered_uses_wrapper_registered_read(self):
        v = mc._emit_flop_top("1rw", 32, 256, comb_read=False)
        # registered read: no combinational shadow mux to the output
        assert "shadow" not in v
        assert ".rdata(rdata)" in v

    def test_1rw1r_ports(self):
        v = mc._emit_flop_top("1rw1r", 16, 64, comb_read=False)
        assert "cs_sram_1rw1r" in v
        assert "rdata0" in v and "rdata1" in v


# --------------------------------------------------------------------------
# Synthetic table: physically-shaped MemPoints for model/rule/predict tests
# --------------------------------------------------------------------------

def _synthetic_table() -> list[MemPoint]:
    """A small table whose Fmax falls with depth (flops) and is high+flat for
    macros -- the real physics, so monotonicity/recommendation tests are valid
    without invoking EDA tools."""
    rows: list[MemPoint] = []
    for ports in ("1rw", "1rw1r"):
        for w in (8, 16, 32, 64):
            for d in (16, 64, 256, 1024, 4096):
                # flop comb-read Fmax ~ 1000 / (1.0 + 0.03*depth) ns model
                delay = 1.0 + 0.03 * d
                fmax = 1000.0 / delay
                area = 600.0 * w * d / 1000.0 + 500.0
                rows.append(MemPoint(ports=ports, width=w, depth=d, impl="flop",
                                     area_um2=area, fmax_mhz=fmax, ff=w * d,
                                     routability_risk=mc._routability_risk(
                                         "flop", d, fmax)))
                rows.append(MemPoint(ports=ports, width=w, depth=d,
                                     impl="registered_flop",
                                     area_um2=area * 1.05,
                                     fmax_mhz=1000.0 / (0.5 + 0.025 * d),
                                     ff=w * d))
                # macro: feasible only for depth>=256 (deep enough); fast+flat
                feasible = d >= 256
                rows.append(MemPoint(
                    ports=ports, width=w, depth=d, impl="macro",
                    area_um2=(180000.0 if feasible else None),
                    fmax_mhz=(2000.0 if feasible else None),
                    macro_feasible=feasible,
                    macro_impl=("exact" if feasible else ""),
                    routability_risk="low"))
    return rows


class TestModel:
    def test_fit_predicts_and_is_monotone_in_depth(self):
        model = mc.fit_model(_synthetic_table())
        assert model.kind in ("gbt", "knn")
        # deeper flop => lower predicted Fmax
        f_shallow = model.predict_fmax("1rw", 16, 16, "flop")
        f_deep = model.predict_fmax("1rw", 16, 4096, "flop")
        assert f_shallow is not None and f_deep is not None
        assert f_shallow > f_deep

    def test_area_grows_with_bits(self):
        model = mc.fit_model(_synthetic_table())
        a_small = model.predict_area("1rw", 8, 16, "flop")
        a_big = model.predict_area("1rw", 64, 4096, "flop")
        assert a_small is not None and a_big is not None
        assert a_big > a_small


class TestRules:
    def test_deep_recommends_macro(self):
        r = mc.recommend_impl_rule(16, 1024)
        assert r["recommended_impl"] == "macro"

    def test_shallow_stays_flops_but_registered(self):
        r = mc.recommend_impl_rule(16, 16)
        assert r["recommended_impl"] == "registered_flop"

    def test_wide_shallow_registered_or_reshape(self):
        # width 640, depth 8: wide + shallow
        r = mc.recommend_impl_rule(640, 8)
        assert r["recommended_impl"] == "registered_flop"
        assert "shallow" in r["reason"] or "register" in r["reason"]

    def test_deep_narrow_real_geom_flagged_macro(self):
        # The run's marquee failure shape: narrow (8 wide) but DEEP (640/10240).
        # depth > D_crit must steer off flops regardless of the small bit count.
        for depth in (640, 10240):
            r = mc.recommend_impl_rule(8, depth)
            assert r["recommended_impl"] == "macro", (8, depth, r)


class TestPredictApi:
    def test_deep_flop_flagged_macro(self):
        model = mc.fit_model(_synthetic_table())
        res = mc.predict_mem(16, 1024, "1rw1r", target_mhz=100.0, model=model)
        # a deep memory cannot meet 100 MHz as flops -> macro (it's feasible)
        assert res["recommended_impl"] == "macro"
        assert res["macro_feasible"] is True

    def test_small_shallow_can_stay_flops(self):
        model = mc.fit_model(_synthetic_table())
        res = mc.predict_mem(16, 16, "1rw", target_mhz=100.0, model=model)
        # shallow meets target easily; never a bare comb 'flop'
        assert res["recommended_impl"] in ("registered_flop", "flop")
        assert res["recommended_impl"] != "flop"  # promoted to registered

    def test_wide_shallow_no_macro_reshape(self):
        # build a table where a wide-shallow geom has macro_feasible False and
        # flops can't hit a very high target -> reshape
        rows = _synthetic_table()
        rows.append(MemPoint(ports="1rw1r", width=640, depth=8, impl="macro",
                             macro_feasible=False))
        # flop @ 640x8: shallow so fast; force an unreachable target instead
        model = mc.fit_model(rows)
        res = mc.predict_mem(640, 8, "1rw1r", target_mhz=5000.0, model=model)
        assert res["recommended_impl"] == "reshape"
        assert res["macro_feasible"] is False


class TestRoutabilityRisk:
    def test_macro_low_deep_flop_high(self):
        assert mc._routability_risk("macro", 4096, 2000.0) == "low"
        assert mc._routability_risk("flop", 1024, 30.0) == "high"
        assert mc._routability_risk("flop", 16, 200.0) == "low"


class TestCacheRoundtrip:
    def test_save_load(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mc, "CACHE_DIR", tmp_path)
        rows = _synthetic_table()[:5]
        mc.save_table(rows)
        loaded = mc.load_table()
        assert len(loaded) == 5
        assert loaded[0].width == rows[0].width
        assert loaded[0].impl == rows[0].impl


# --------------------------------------------------------------------------
# [mem-model-fix] Applicability domain: is_in_grid over the REAL cache rows
# --------------------------------------------------------------------------

class TestApplicabilityDomain:
    def test_grid_bounds_over_measured_rows(self):
        b = mc.grid_bounds(_synthetic_table())
        # widths 8..64, depths 16..4096; biggest capacity is the 64x4096 corner
        assert b == (8, 64, 16, 4096, 64 * 4096)

    def test_grid_bounds_none_when_no_measured_row(self):
        assert mc.grid_bounds([]) is None
        # a row with neither area nor Fmax does not bound anything
        assert mc.grid_bounds(
            [MemPoint(ports="1rw", width=8, depth=8, impl="flop")]) is None

    def test_in_grid_true_inside_and_on_corner(self):
        t = _synthetic_table()
        assert mc.is_in_grid(16, 256, t) is True
        assert mc.is_in_grid(64, 4096, t) is True   # exact corner

    def test_out_of_grid_by_depth(self):
        assert mc.is_in_grid(16, 100_000, _synthetic_table()) is False

    def test_out_of_grid_by_width(self):
        assert mc.is_in_grid(256, 256, _synthetic_table()) is False

    def test_out_of_grid_by_bit_capacity_corner(self):
        # each dimension individually in-range, but the PRODUCT exceeds anything
        # ever characterized -> out of grid (the bit-cap clause).
        t = [MemPoint(ports="1rw", width=64, depth=16, impl="flop",
                      area_um2=1000.0, fmax_mhz=500.0),
             MemPoint(ports="1rw", width=8, depth=4096, impl="flop",
                      area_um2=2000.0, fmax_mhz=10.0)]
        # bounds: w[8,64], d[16,4096], max_bits = max(1024, 32768) = 32768
        assert mc.is_in_grid(64, 16, t) is True
        assert mc.is_in_grid(8, 4096, t) is True
        assert mc.is_in_grid(64, 4096, t) is False   # 262144 bits > 32768

    def test_empty_table_is_not_in_grid(self):
        assert mc.is_in_grid(8, 8, []) is False


# --------------------------------------------------------------------------
# [mem-model-fix] Deterministic-first macro routing: LEF banking authoritative
# --------------------------------------------------------------------------

class TestDeterministicFirstMacro:
    def test_macro_feasible_out_of_grid_uses_lef_not_regressor(self, monkeypatch):
        # geometry absent from the table -> live resolve; its LEF area/Fmax are
        # AUTHORITATIVE, the (saturated) regressor must not override them.
        model = mc.fit_model(_synthetic_table())
        lef_area, lef_fmax = 46_000_000.0, 1600.0
        monkeypatch.setattr(mc, "_resolve_macro",
            lambda ports, w, d: MemPoint(
                ports=ports, width=w, depth=d, impl="macro", macro_feasible=True,
                macro_impl="compose", area_um2=lef_area, fmax_mhz=lef_fmax))
        res = mc.predict_mem(8, 100_000, "1rw1r", target_mhz=100.0, model=model)
        cm = res["candidates"]["macro"]
        assert cm["source"] == "lef_exact"
        assert cm["area_um2"] == pytest.approx(lef_area)
        assert cm["fmax_mhz"] == pytest.approx(lef_fmax)
        # the LEF number is NOT whatever the regressor would have said
        assert cm["area_um2"] != pytest.approx(
            model.predict_area("1rw1r", 8, 100_000, "macro"))
        assert res["recommended_impl"] == "macro"
        assert res["estimate_source"] == "lef_exact"
        assert res["in_grid"] is False

    def test_macro_with_cached_row_uses_measured_no_live_resolve(self, monkeypatch):
        # a cached (in-grid) geometry must use its MEASURED row, never a live
        # resolve (byte-identical to pre-fix behavior).
        model = mc.fit_model(_synthetic_table())
        monkeypatch.setattr(mc, "_resolve_macro",
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("_resolve_macro must not run for a cached row")))
        res = mc.predict_mem(16, 1024, "1rw", target_mhz=100.0, model=model)
        cm = res["candidates"]["macro"]
        assert cm["source"] == "pdk_measured"
        assert cm["area_um2"] == pytest.approx(180000.0)   # from _synthetic_table
        assert cm["fmax_mhz"] == pytest.approx(2000.0)


# --------------------------------------------------------------------------
# [mem-model-fix] Out-of-grid analytic extrapolation (flop / registered_flop)
# --------------------------------------------------------------------------

def _no_macro(monkeypatch):
    monkeypatch.setattr(mc, "_resolve_macro",
        lambda ports, w, d: MemPoint(ports=ports, width=w, depth=d,
                                     impl="macro", macro_feasible=False))


class TestOutOfGridExtrapolation:
    def test_flop_area_is_edge_per_bit_times_bits(self, monkeypatch):
        model = mc.fit_model(_synthetic_table())
        _no_macro(monkeypatch)
        w, d = 13, 100_000     # deeper + oddball width -> out of grid, no macro
        res = mc.predict_mem(w, d, "1rw", target_mhz=100.0, model=model)
        assert res["in_grid"] is False
        c = res["candidates"]["flop"]
        assert c["source"] == "analytic_extrapolation"
        expected = mc._per_bit_area_cost(model._rows, "flop") * (w * d)
        assert c["area_um2"] == pytest.approx(expected)
        assert c["area_um2"] > 0

    def test_fmax_clamped_to_worst_in_grid(self, monkeypatch):
        model = mc.fit_model(_synthetic_table())
        _no_macro(monkeypatch)
        res = mc.predict_mem(13, 100_000, "1rw", target_mhz=100.0, model=model)
        for impl in ("flop", "registered_flop"):
            worst = mc._worst_in_grid_fmax(model._rows, impl)
            c = res["candidates"][impl]
            assert c["fmax_mhz"] == pytest.approx(worst)
            # invariant: out-of-grid Fmax <= the slowest measured in-grid Fmax
            in_grid = [r.fmax_mhz for r in model._rows
                       if r.impl == impl and r.fmax_mhz]
            assert c["fmax_mhz"] <= min(in_grid) + 1e-9

    def test_estimate_source_is_extrapolation_when_no_macro(self, monkeypatch):
        model = mc.fit_model(_synthetic_table())
        _no_macro(monkeypatch)
        res = mc.predict_mem(13, 100_000, "1rw", target_mhz=100.0, model=model)
        assert res["estimate_source"] == "analytic_extrapolation"
        assert res["macro_feasible"] is False

    def test_per_bit_cost_falls_back_when_impl_absent(self, monkeypatch):
        # a table with no flop rows -> the analytic fallback per-bit cost
        monkeypatch.setenv("CORESMITH_FLOP_UM2_PER_BIT", "25.0")
        rows = [r for r in _synthetic_table() if r.impl == "macro"]
        assert mc._per_bit_area_cost(rows, "flop") == pytest.approx(25.0)


# --------------------------------------------------------------------------
# [mem-model-fix] Regression pins: in-grid predictions unchanged (byte-identical)
# --------------------------------------------------------------------------

class TestInGridRegressionPins:
    def test_flop_regflop_in_grid_are_regressor_verbatim(self):
        model = mc.fit_model(_synthetic_table())
        res = mc.predict_mem(16, 256, "1rw", target_mhz=100.0, model=model)
        assert res["in_grid"] is True
        for impl in ("flop", "registered_flop"):
            c = res["candidates"][impl]
            assert c["source"] == "regressor_in_grid"
            assert c["area_um2"] == model.predict_area("1rw", 16, 256, impl)
            assert c["fmax_mhz"] == model.predict_fmax("1rw", 16, 256, impl)

    def test_macro_cached_row_is_measured_verbatim(self):
        model = mc.fit_model(_synthetic_table())
        res = mc.predict_mem(32, 256, "1rw", target_mhz=100.0, model=model)
        c = res["candidates"]["macro"]
        assert c["source"] == "pdk_measured"
        assert c["area_um2"] == pytest.approx(180000.0)
        assert c["fmax_mhz"] == pytest.approx(2000.0)

    def test_recommendation_unchanged_for_in_grid_deep_flop(self):
        # the pre-fix recommendation contract still holds in-grid
        model = mc.fit_model(_synthetic_table())
        res = mc.predict_mem(16, 1024, "1rw1r", target_mhz=100.0, model=model)
        assert res["recommended_impl"] == "macro"
        assert res["macro_feasible"] is True
