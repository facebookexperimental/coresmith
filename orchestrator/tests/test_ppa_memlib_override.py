# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""PR#14: per-block STA/depth probes resolve engine memory-wrapper instances
(mem_lib_sources_for_rtl) + evaluate_ppa defers BUDGET dims on a chip-lead
uarch_feasibility override. Pure; no LLM/EDA."""
from __future__ import annotations

from orchestrator.langgraph.ppa_check import (
    _maxfanout_synth_script,
    evaluate_ppa,
    mem_lib_sources_for_rtl,
    ppa_honor_feas_override_enabled,
)


class TestMemLibSources:
    def _lib(self, tmp_path, monkeypatch):
        import orchestrator.langgraph.sram_wrapper as sw
        lib = tmp_path / "cs_sram.v"
        lib.write_text("module cs_fpmem; endmodule\n")
        monkeypatch.setattr(sw, "wrapper_lib_path", lambda: str(lib))
        return str(lib)

    def test_cs_instance_pulls_lib(self, tmp_path, monkeypatch):
        lib = self._lib(tmp_path, monkeypatch)
        rtl = "module b(input clk);\ncs_fpmem #(.W(8)) u_m (.clk(clk));\nendmodule"
        assert mem_lib_sources_for_rtl(rtl) == [lib]

    def test_all_wrapper_families_match(self, tmp_path, monkeypatch):
        lib = self._lib(tmp_path, monkeypatch)
        for inst in ("cs_sram_1rw", "cs_sram_1rw1r", "cs_fpmem", "cs_rom_1r"):
            rtl = f"module b;\n{inst} u ();\nendmodule"
            assert mem_lib_sources_for_rtl(rtl) == [lib], inst

    def test_no_reference_no_lib(self, tmp_path, monkeypatch):
        self._lib(tmp_path, monkeypatch)
        assert mem_lib_sources_for_rtl(
            "module b(input clk); reg [7:0] r; endmodule") == []

    def test_missing_lib_file_returns_empty(self, tmp_path, monkeypatch):
        import orchestrator.langgraph.sram_wrapper as sw
        monkeypatch.setattr(sw, "wrapper_lib_path",
                            lambda: str(tmp_path / "nonexistent.v"))
        assert mem_lib_sources_for_rtl("module b; cs_fpmem u(); endmodule") == []

    def test_empty_rtl(self, tmp_path, monkeypatch):
        self._lib(tmp_path, monkeypatch)
        assert mem_lib_sources_for_rtl("") == []

    def test_synth_script_reads_all_sources(self, tmp_path):
        script = _maxfanout_synth_script(
            ["a.v", "b.v"], "lib.lib", tmp_path / "n.v", "top", False, False)
        assert "read_verilog -sv a.v b.v" in script

    def test_rtl_including_lib_skips_lib(self, tmp_path, monkeypatch):
        # A wrapper that pulls the library via `include (possibly an absolute
        # path to a DIFFERENT engine checkout -- path-dedup can't merge it)
        # must not get the library a second time via extra_sources.
        self._lib(tmp_path, monkeypatch)
        rtl = ('`include "/some/other/engine/orchestrator/langgraph/'
               'rtl_lib/cs_sram.v"\n'
               "module w; cs_fpmem u(); endmodule")
        assert mem_lib_sources_for_rtl(rtl) == []

    def test_rtl_defining_lib_modules_skips_lib(self, tmp_path, monkeypatch):
        # An integration wrapper with the library baked in must NOT pull the
        # library again (yosys module re-definition kills both STA sub-flows).
        self._lib(tmp_path, monkeypatch)
        rtl = ("module cs_mem_macro_shell; endmodule\n"
               "module w; cs_fpmem u(); endmodule")
        assert mem_lib_sources_for_rtl(rtl) == []

    def test_dedup_sources_drops_repeated_lib(self, tmp_path):
        from orchestrator.langgraph.ppa_check import _dedup_sources
        rtl = tmp_path / "w.v"
        rtl.write_text("module w; endmodule")
        lib = tmp_path / "cs_sram.v"
        lib.write_text("module cs_fpmem; endmodule")
        srcs = _dedup_sources(str(rtl), [str(lib), str(lib),
                                         str(tmp_path / "missing.v")])
        assert srcs == [str(rtl), str(lib)]

    def test_dedup_sources_drops_rtl_repeat(self, tmp_path):
        from orchestrator.langgraph.ppa_check import _dedup_sources
        rtl = tmp_path / "w.v"
        rtl.write_text("module w; endmodule")
        assert _dedup_sources(str(rtl), [str(rtl)]) == [str(rtl)]


class TestEvaluatePpaOverride:
    def test_area_over_budget_deferred(self):
        v = evaluate_ppa(actual_ff=100, ff_budget=1000,
                         actual_area_um2=130000.0, area_budget_um2=68000.0,
                         budget_overridden=True)
        assert v.ok is True
        area = [c for c in v.checks if c["metric"] == "chip_area_um2"][0]
        assert area["passed"] is False
        assert area["deferred_by_override"] is True
        assert v.reasons == []

    def test_area_over_budget_still_fails_without_override(self):
        v = evaluate_ppa(actual_ff=100, ff_budget=1000,
                         actual_area_um2=130000.0, area_budget_um2=68000.0)
        assert v.ok is False
        assert any("exceeds budget" in r for r in v.reasons)

    def test_logic_ff_over_budget_deferred(self):
        v = evaluate_ppa(actual_ff=12000, ff_budget=1000,
                         budget_overridden=True)
        assert v.ok is True
        ff = [c for c in v.checks if c["metric"] == "flip_flop_count"][0]
        assert ff["passed"] is False
        assert ff["deferred_by_override"] is True

    def test_hard_ceiling_gates_despite_override(self):
        v = evaluate_ppa(actual_ff=60000, ff_budget=1000,
                         budget_overridden=True)
        assert v.ok is False
        assert any(c["metric"] == "flip_flop_hard_ceiling" and not c["passed"]
                   for c in v.checks)
        # the deferred FF-budget check must not swallow the ceiling reason
        assert any("hard ceiling" in r for r in v.reasons)

    def test_timing_gates_despite_override(self):
        v = evaluate_ppa(actual_ff=100, ff_budget=1000,
                         wns_ns=-6.45, period_ns=20.0,
                         budget_overridden=True)
        assert v.ok is False
        assert any(c["metric"] == "wns_ns" and not c["passed"]
                   for c in v.checks)


class TestHonorOverrideEnv:
    def test_default_on(self, monkeypatch):
        monkeypatch.delenv("CORESMITH_PPA_HONOR_FEAS_OVERRIDE", raising=False)
        assert ppa_honor_feas_override_enabled() is True

    def test_disable(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_PPA_HONOR_FEAS_OVERRIDE", "0")
        assert ppa_honor_feas_override_enabled() is False


class TestDropIncludeProvidedSources:
    def _mk(self, tmp_path, name, text):
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        return str(p)

    def test_included_block_dropped_from_list(self, tmp_path):
        from orchestrator.langgraph.integration_helpers import (
            _drop_include_provided_sources,
        )
        blk = self._mk(tmp_path, "io/frontend.v", "module frontend; endmodule")
        top = self._mk(tmp_path, "top.v",
                       f'`include "{blk}"\nmodule top; frontend u(); endmodule')
        assert _drop_include_provided_sources([top, blk]) == [top]

    def test_basename_match_other_checkout(self, tmp_path):
        from orchestrator.langgraph.integration_helpers import (
            _drop_include_provided_sources,
        )
        lib = self._mk(tmp_path, "worktree/cs_sram.v", "module cs_fpmem; endmodule")
        top = self._mk(tmp_path, "top.v",
                       '`include "/some/old/engine/rtl_lib/cs_sram.v"\n'
                       "module top; endmodule")
        assert _drop_include_provided_sources([top, lib]) == [top]

    def test_relative_include_resolved(self, tmp_path):
        from orchestrator.langgraph.integration_helpers import (
            _drop_include_provided_sources,
        )
        blk = self._mk(tmp_path, "sub/blk.v", "module blk; endmodule")
        top = self._mk(tmp_path, "sub/top.v",
                       '`include "blk.v"\nmodule top; endmodule')
        assert _drop_include_provided_sources([top, blk]) == [top]

    def test_first_entry_never_dropped(self, tmp_path):
        from orchestrator.langgraph.integration_helpers import (
            _drop_include_provided_sources,
        )
        top = self._mk(tmp_path, "top.v", "module top; endmodule")
        other = self._mk(tmp_path, "other.v", f'`include "{top}"\nmodule o; endmodule')
        out = _drop_include_provided_sources([top, other])
        assert top in out and other in out

    def test_no_includes_unchanged(self, tmp_path):
        from orchestrator.langgraph.integration_helpers import (
            _drop_include_provided_sources,
        )
        a = self._mk(tmp_path, "a.v", "module a; endmodule")
        b = self._mk(tmp_path, "b.v", "module b; endmodule")
        assert _drop_include_provided_sources([a, b]) == [a, b]
