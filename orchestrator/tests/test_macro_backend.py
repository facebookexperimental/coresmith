# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for macro_backend (floorplan plan + tool injection snippets) and
openram_gen (fallback resolution). Hermetic -- no EDA tools / PDK needed."""
from __future__ import annotations

from orchestrator.langgraph import macro_backend as mb
from orchestrator.langgraph import openram_gen as og
from orchestrator.langgraph.macro_registry import MacroInfo


def _macro(name="sky130_sram_1kbyte_1rw1r_32x256_8", words=256, bits=32, w=479.78,
           h=397.5, root=None):
    """A MacroInfo. If `root` (tmp dir) given, write stub collateral files so
    `collateral_complete()` is True (needed by find_exact/plan_composition)."""
    paths = {k: f"/pdk/{name}.{k}" for k in ("lef", "gds", "lib", "spice", "verilog")}
    if root is not None:
        for k in paths:
            p = root / f"{name}.{k}"
            p.write_text("stub")
            paths[k] = str(p)
    return MacroInfo(
        name=name, lef=paths["lef"], gds=paths["gds"], lib=paths["lib"],
        spice=paths["spice"], verilog=paths["verilog"], width_um=w, height_um=h,
        words=words, data_bits=bits, bits=words * bits, ports="1rw1r",
        mask_bits=8, power_pin="vccd1", ground_pin="vssd1",
    )


class TestExtractInstances:
    def test_plain_and_escaped_names(self, tmp_path):
        m = _macro()
        net = tmp_path / "net.v"
        net.write_text(
            f"{m.name} u_mem (.clk0(c));\n"
            f"{m.name} \\u_top.u_bank1 (.clk0(c));\n"
        )
        got = mb.extract_macro_instances(str(net), [m])
        insts = sorted(i for _m, i in got)
        assert insts == ["u_mem", "u_top.u_bank1"]  # backslash stripped


class TestFloorplan:
    def test_die_fits_macro_and_places_it(self):
        m = _macro()
        die, core, place = mb.plan_floorplan([(m, "u_mem")], std_cell_area_um2=125_000)
        _, _, urx, ury = (float(x) for x in die.split())
        # die must exceed macro footprint + margins
        assert urx > m.width_um and ury > m.height_um
        assert len(place) == 1
        inst, x, y, orient, mac = place[0]
        assert inst == "u_mem" and orient == "R0" and mac is m
        assert x >= 0 and y >= 0

    def test_two_macros_do_not_overlap(self):
        m1, m2 = _macro(), _macro(name="sky130_sram_2kbyte_1rw1r_32x512_8", words=512, w=683.1)
        die, core, place = mb.plan_floorplan(
            [(m1, "a"), (m2, "b")], std_cell_area_um2=125_000
        )
        assert len(place) == 2
        (_, ax, ay, _, am), (_, bx, by, _, bm) = place
        # bounding boxes disjoint (separated in x OR in y -- row may wrap)
        disjoint_x = bx >= ax + am.width_um or ax >= bx + bm.width_um
        disjoint_y = by >= ay + am.height_um or ay >= by + bm.height_um
        assert disjoint_x or disjoint_y


class TestSnippets:
    def test_synth_injection(self):
        bb, libs, hil = mb.synth_injection([_macro()])
        assert "read_verilog -lib" in bb
        assert "read_liberty -lib" in libs
        assert "hilomap" in hil and "conb_1" in hil

    def test_synth_injection_empty(self):
        assert mb.synth_injection([]) == ("", "", "")

    def test_pnr_header_vars(self):
        m = _macro()
        _, _, place = mb.plan_floorplan([(m, "u_mem")], 125_000)
        hdr = mb.pnr_header_vars([m], "0 0 790 790", "30 30 760 760", place)
        assert "set macro_lefs" in hdr and m.lef in hdr
        assert "set macro_names [list sky130_sram_1kbyte_1rw1r_32x256_8]" in hdr
        assert "set macro_die_area" in hdr
        assert "{vccd1 vssd1}" in hdr  # pg pair

    def test_drc_block_uses_lefview_and_gdsfile(self):
        blk = mb.drc_macro_block([(_macro(), "u_mem")])
        assert "property LEFview true" in blk
        assert "property GDS_FILE" in blk
        assert "lef read" in blk


# ---------------------------------------------------------------------------
# Real top-module detection + flattened leaf counts (PnR-top-selection fix)
# ---------------------------------------------------------------------------

# A generic "sub-block-first" netlist: a leaf that instantiates a macro is
# DEFINED first, a mid-level block reuses it, and the real integration top is
# emitted LAST -- the shape a yosys macro-rewritten netlist has (and the shape
# the old "first module" scan mis-linked). No benchmark/design names.
def _subblock_first_netlist(top="dut_top", macro="synth_sram_a"):
    return (
        # leaf (defined FIRST) -- instantiates a hard macro
        "module leaf_mem (clk, a, q);\n"
        "  input clk; input [3:0] a; output [7:0] q;\n"
        f"  {macro} u_macro (.clk0(clk), .addr0(a), .dout0(q));\n"
        "endmodule\n"
        # mid-level block reused several times, each with two leaves
        "module sub_bank (clk, a, q);\n"
        "  input clk; input [3:0] a; output [7:0] q;\n"
        "  leaf_mem u_m0 (.clk(clk), .a(a), .q(q));\n"
        "  leaf_mem u_m1 (.clk(clk), .a(a), .q(q));\n"
        "endmodule\n"
        # the REAL top (defined LAST, instantiated by nobody)
        f"module {top} (clk, a, q);\n"
        "  input clk; input [3:0] a; output [7:0] q;\n"
        "  sub_bank u_b0 (.clk(clk), .a(a), .q(q));\n"
        "  sub_bank u_b1 (.clk(clk), .a(a), .q(q));\n"
        "  sub_bank u_b2 (.clk(clk), .a(a), .q(q));\n"
        "  leaf_mem u_extra (.clk(clk), .a(a), .q(q));\n"
        "endmodule\n"
    )


class TestModuleGraph:
    def test_parse_defines_and_children(self):
        defined, children = mb.parse_module_graph(_subblock_first_netlist())
        assert defined == ["leaf_mem", "sub_bank", "dut_top"]  # file order
        assert children["leaf_mem"] == ["synth_sram_a"]
        assert children["sub_bank"] == ["leaf_mem", "leaf_mem"]
        assert children["dut_top"] == ["sub_bank", "sub_bank", "sub_bank",
                                       "leaf_mem"]

    def test_ignores_port_and_wire_decls(self):
        # declarations must not be mistaken for instantiations.
        _defined, children = mb.parse_module_graph(_subblock_first_netlist())
        assert "input" not in children["dut_top"]
        assert "output" not in children["dut_top"]

    def test_parse_handles_escaped_paramod_names(self):
        net = (
            r"module \$paramod$abc\leaf (clk); input clk;"
            r" synth_sram_a u (.clk0(clk)); endmodule" "\n"
            r"module wrap (clk); input clk;"
            r" \$paramod$abc\leaf  u_l (.clk(clk)); endmodule" "\n"
        )
        defined, children = mb.parse_module_graph(net)
        assert r"\$paramod$abc\leaf" in defined and "wrap" in defined
        assert children["wrap"] == [r"\$paramod$abc\leaf"]


class TestDetectTopModule:
    def test_subblock_first_returns_real_top_not_leaf(self):
        # the crux: a netlist with leaves FIRST must yield the real top, never
        # the first-defined sub-block.
        net = _subblock_first_netlist(top="dut_top")
        assert mb.detect_top_module(net, "some_slug") == "dut_top"

    def test_slug_mangled_preferred_resolves_real_uninstantiated_top(self):
        # preferred is a PRD-title slug that != the netlist's real module name.
        net = _subblock_first_netlist(top="pipeline_top")
        assert mb.detect_top_module(
            net, "prd___pipeline_top") == "pipeline_top"

    def test_already_correct_preferred_returned_unchanged(self):
        net = _subblock_first_netlist(top="dut_top")
        assert mb.detect_top_module(net, "dut_top") == "dut_top"

    def test_never_returns_instantiated_submodule(self):
        # even if a caller passes a sub-block name, it is instantiated -> never
        # returned as the top.
        net = _subblock_first_netlist(top="dut_top")
        assert mb.detect_top_module(net, "leaf_mem") == "dut_top"

    def test_single_module_design_unchanged(self):
        net = "module solo (clk); input clk; endmodule\n"
        assert mb.detect_top_module(net, "solo") == "solo"

    def test_empty_netlist_falls_back_to_preferred(self):
        assert mb.detect_top_module("", "my_block") == "my_block"

    def test_multiple_tops_picks_best_name_match(self):
        # two uninstantiated modules -> the one matching preferred (normalized).
        net = (
            "module leaf_mem (clk); input clk; endmodule\n"
            "module helper_unused (clk); input clk;"
            " leaf_mem u (.clk(clk)); endmodule\n"
            "module dut_top (clk); input clk;"
            " leaf_mem u (.clk(clk)); endmodule\n"
        )
        # helper_unused and dut_top are both uninstantiated; preferred matches.
        assert mb.detect_top_module(net, "dut_top") == "dut_top"


class TestFlattenedCounts:
    def test_flattens_hierarchical_reuse(self):
        net = _subblock_first_netlist(top="dut_top")
        flat = mb.flattened_type_counts(net, "dut_top")
        # sub_bank x3; each has 2 leaf_mem => 6, + 1 extra = 7 leaf_mem;
        # each leaf_mem has 1 macro => 7 macros.
        assert flat["sub_bank"] == 3
        assert flat["leaf_mem"] == 7
        assert flat["synth_sram_a"] == 7

    def test_unknown_top_returns_empty(self):
        net = _subblock_first_netlist()
        assert mb.flattened_type_counts(net, "no_such_top") == {}


class TestExpandPlacementsToFlattened:
    def test_expands_to_flattened_leaf_count(self):
        net = _subblock_first_netlist(top="dut_top")
        m = _macro(name="synth_sram_a")
        # one TEXT instantiation of the macro (inside leaf_mem) ...
        placed = [(m, "u_macro")]
        out = mb.expand_placements_to_flattened(net, "dut_top", placed)
        # ... expands to the 7 flattened leaves so the plan covers them all.
        assert len(out) == 7
        assert all(mi is m for mi, _ in out)

    def test_no_shrink_below_text_count(self):
        # if flatten can't be computed, never drop below what is present.
        m = _macro(name="synth_sram_a")
        placed = [(m, "u0"), (m, "u1")]
        out = mb.expand_placements_to_flattened("", "x", placed)
        assert len(out) == 2

    def test_empty_placed_is_noop(self):
        assert mb.expand_placements_to_flattened("whatever", "x", []) == []


class TestOpenRAMFallback:
    def _reg(self, tmp_path):
        return {
            "sky130_sram_1kbyte_1rw1r_32x256_8": _macro(root=tmp_path),
            "sky130_sram_2kbyte_1rw1r_32x512_8": _macro(
                name="sky130_sram_2kbyte_1rw1r_32x512_8", words=512, w=683.1,
                root=tmp_path),
        }

    def test_find_exact(self, tmp_path):
        reg = self._reg(tmp_path)
        assert og.find_exact(256, 32, reg).name == "sky130_sram_1kbyte_1rw1r_32x256_8"
        assert og.find_exact(999, 7, reg) is None

    def test_compose_wider(self, tmp_path):
        plan = og.plan_composition(256, 64, self._reg(tmp_path))
        assert plan is not None and plan.tiles_wide == 2 and plan.tiles_deep == 1

    def test_compose_deeper(self, tmp_path):
        plan = og.plan_composition(1024, 32, self._reg(tmp_path))
        assert plan is not None and plan.tiles_deep >= 2

    def test_no_composition_for_odd_geometry(self, tmp_path):
        assert og.plan_composition(300, 17, self._reg(tmp_path)) is None

    def test_over_provision_shallow_depth(self, tmp_path):
        # 32x64: no exact tiling (64 not a multiple of 256/512), but a single
        # 32x256 prebuilt COVERS it with the high address bits tied off.
        reg = self._reg(tmp_path)
        plan = og.plan_over_provisioned(64, 32, reg)
        assert plan is not None
        assert plan.tiles_wide == 1 and plan.tiles_deep == 1
        assert plan.base.name == "sky130_sram_1kbyte_1rw1r_32x256_8"
        assert plan.provisioned_words == 256 and plan.provisioned_bits == 32
        assert plan.over_provisioned is True

    def test_over_provision_wide_word(self, tmp_path):
        # 32x64 with a 64-bit WORD: 2 wide (2x32) x 1 deep (256>=64).
        reg = self._reg(tmp_path)
        plan = og.plan_over_provisioned(32, 64, reg)
        assert plan is not None
        assert plan.tiles_wide == 2 and plan.tiles_deep == 1
        assert plan.provisioned_bits == 64 and plan.provisioned_words == 256
        assert plan.over_provisioned is True

    def test_over_provision_prefers_least_waste(self, tmp_path):
        # 32x200 should pick the 32x256 (waste 56*32) over 32x512 (waste 312*32).
        reg = self._reg(tmp_path)
        plan = og.plan_over_provisioned(200, 32, reg)
        assert plan.base.name == "sky130_sram_1kbyte_1rw1r_32x256_8"

    def test_ensure_over_provisions_before_flopping(self, tmp_path):
        # ensure_macro must reach over-provisioning (not None) when exact +
        # exact-tiling both miss and generation is disabled.
        reg = self._reg(tmp_path)
        import orchestrator.langgraph.openram_gen as ogmod
        orig = ogmod.discover_macros
        ogmod.discover_macros = lambda *a, **k: reg
        try:
            res = ogmod.ensure_macro(64, 32, allow_generate=False)
            assert isinstance(res, ogmod.CompositionPlan)
            assert res.over_provisioned is True
        finally:
            ogmod.discover_macros = orig

    def test_exact_still_beats_over_provision(self, tmp_path):
        # a geometry with an exact prebuilt must NOT over-provision
        reg = self._reg(tmp_path)
        import orchestrator.langgraph.openram_gen as ogmod
        orig = ogmod.discover_macros
        ogmod.discover_macros = lambda *a, **k: reg
        try:
            res = ogmod.ensure_macro(256, 32, allow_generate=False)
            assert res.name == "sky130_sram_1kbyte_1rw1r_32x256_8"
        finally:
            ogmod.discover_macros = orig

    def test_ensure_returns_exact_before_compose(self, tmp_path):
        reg = self._reg(tmp_path)
        import orchestrator.langgraph.openram_gen as ogmod
        orig = ogmod.discover_macros
        ogmod.discover_macros = lambda *a, **k: reg
        try:
            got = og.ensure_macro(256, 32, allow_generate=False)
            assert isinstance(got, MacroInfo) and got.words == 256
            # un-tileable but over-provisionable (300x17 -> 512x32, 3.2x waste
            # < 8x cap): a real macro beats flopping, so NOT None any more.
            op = og.ensure_macro(300, 17, allow_generate=False)
            assert isinstance(op, og.CompositionPlan) and op.over_provisioned
            # a pathologically tiny store (past the waste cap) stays None ->
            # caller re-decides (flops), never places a mostly-empty macro.
            assert og.ensure_macro(2, 1, allow_generate=False) is None
        finally:
            ogmod.discover_macros = orig

    def test_openram_available_requires_runnable_main(self, monkeypatch):
        import orchestrator.langgraph.openram_gen as ogmod
        monkeypatch.delenv("OPENRAM_HOME", raising=False)
        monkeypatch.setattr(ogmod, "_OPENRAM_RUNNABLE", None)
        import importlib.util as _ilu
        # package imports but has NO __main__ -> not runnable -> not available
        real_find = _ilu.find_spec

        def fake_find(name, *a, **k):
            if name == "openram.__main__":
                return None
            return real_find(name, *a, **k)
        monkeypatch.setattr(_ilu, "find_spec", fake_find)
        # ensure `import openram` succeeds in the check by faking it present
        import sys, types
        monkeypatch.setitem(sys.modules, "openram", types.ModuleType("openram"))
        assert ogmod.openram_available() is False
        # now with a runnable __main__ present -> available
        monkeypatch.setattr(ogmod, "_OPENRAM_RUNNABLE", None)

        def fake_find2(name, *a, **k):
            if name == "openram.__main__":
                return object()
            return real_find(name, *a, **k)
        monkeypatch.setattr(_ilu, "find_spec", fake_find2)
        assert ogmod.openram_available() is True

    def test_macro_name_for_is_registry_parseable(self):
        from orchestrator.langgraph.macro_registry import _OPENRAM_NAME_RE
        nm = og.macro_name_for(256, 32, "1rw1r", 8)
        assert _OPENRAM_NAME_RE.match(nm)


# ---------------------------------------------------------------------------
# LVS constant-tie / port-equivalence proof (honest netgen mismatch triage).
# Fixtures are GENERIC (Caravel-standard io_out/io_oeb + SRAM din0/wmask0 pin
# names, an anonymous `chip_top` module) -- no design/benchmark identifiers.
# ---------------------------------------------------------------------------

# A wrapper netlist as yosys write_verilog lowers a constant-tie-off: unused
# GPIO output bits aliased per-bit to a shared representative (constant-0 group
# -> io_oeb[6], constant-1 group -> io_oeb[0], replicated enable -> io_oeb[2]).
_REF_V_TIED = """
module chip_top (io_in, io_out, io_oeb);
  input [37:0] io_in;
  output [37:0] io_out;
  output [37:0] io_oeb;
  assign io_oeb[6] = 1'b0;
  assign io_oeb[0] = 1'b1;
  assign io_oeb[1] = io_oeb[0];
  assign io_oeb[7] = io_oeb[0];
  assign io_oeb[8] = io_oeb[0];
  assign io_oeb[9] = io_oeb[0];
  assign io_oeb[37] = io_oeb[0];
  assign io_oeb[3] = io_oeb[2];
  assign io_oeb[4] = io_oeb[2];
  assign io_oeb[5] = io_oeb[2];
  assign io_out[0] = io_oeb[6];
  assign io_out[1] = io_oeb[6];
  assign io_out[7] = io_oeb[6];
  assign io_out[8] = io_oeb[6];
  assign io_out[9] = io_oeb[6];
endmodule
"""

# A netgen report whose top-level pin matching failed on exactly those tied
# output bits (layout collapsed them to a few representatives).
_REP_TIED_PINS = """
Subcircuit pins:
Circuit 1: chip_top                        |Circuit 2: chip_top
-------------------------------------------|-------------------------------------------
io_out[2]                                  |io_out[2]
io_out[6]                                  |io_out[6]
(no matching pin)                          |io_oeb[37]
(no matching pin)                          |io_oeb[7]
(no matching pin)                          |io_out[7]
(no matching pin)                          |io_out[8]
(no matching pin)                          |io_out[0]
io_oeb[9]                                  |(no matching pin)
io_out[9]                                  |(no matching pin)
-------------------------------------------|-------------------------------------------
Cell pin lists for chip_top and chip_top altered to match.

Final result: Top level cell failed pin matching.
"""

# A NET-mismatch fragment section for over-provisioned SRAM macros: unused
# din0[1:7] tied to `.../zero_` and wmask0[*] to `.../one_` in the reference,
# floating per-pin in the layout extraction.
_REP_MACRO_INPUT = """
Number of nets: 5922 **Mismatch**          |Number of nets: 5889 **Mismatch**
---------------------------------------------------------------------------------------
NET mismatches: Class fragments follow (with fanout counts):
Circuit 1: chip_top                        |Circuit 2: chip_top
Net: u_mem_bank0_din0_3                     |Net: \\u_mem/zero_
  sram_8x1024/din0[3]                       |  sram_8x1024/din0[1]
Net: u_mem_bank1_din0_4                     |(no matching net)
  sram_8x1024/din0[4]                       |
Net: u_mem_wmask_0                          |Net: \\u_mem/one_
  sram_8x1024/wmask0[0] = 1                 |  sram_8x1024/wmask0[1]
---------------------------------------------------------------------------------------
Netlists do not match.
"""

_REP_MATCH = "Final result: Circuits match uniquely.\n"


class TestLvsTieClasses:
    def test_parse_output_tie_classes_constants_and_aliases(self):
        info = mb.parse_output_tie_classes(_REF_V_TIED)
        # every LHS alias bit is a tied bit
        assert "io_oeb[37]" in info["tied_bits"]
        assert "io_out[0]" in info["tied_bits"]
        assert "io_oeb[3]" in info["tied_bits"]
        # constant-rooted groups resolve to constant bits
        assert "io_oeb[37]" in info["const_bits"]   # -> io_oeb[0] = 1'b1
        assert "io_out[9]" in info["const_bits"]     # -> io_oeb[6] = 1'b0
        # a replicated REAL-signal alias is tied but NOT constant
        assert "io_oeb[3]" in info["tied_bits"]
        assert "io_oeb[3]" not in info["const_bits"]

    def test_unmatched_top_pins_both_sides(self):
        pins = set(mb.unmatched_top_pins(_REP_TIED_PINS))
        assert "io_oeb[37]" in pins and "io_out[0]" in pins  # ref side
        assert "io_oeb[9]" in pins and "io_out[9]" in pins    # layout side
        assert "io_out[2]" not in pins and "io_out[6]" not in pins  # matched

    def test_classify_accepts_constant_tied_outputs(self):
        v = mb.classify_lvs_report(_REP_TIED_PINS, _REF_V_TIED)
        assert v["benign"] and v["accept"] and not v["netgen_match"]
        assert v["classes"]["constant_tied_output"] >= 4
        assert v["unresolved"] == []

    def test_classify_accepts_constant_tied_macro_inputs(self):
        v = mb.classify_lvs_report(_REP_MACRO_INPUT, "")
        assert v["benign"] and v["accept"]
        assert v["classes"]["constant_macro_input"] >= 2
        assert v["unresolved"] == []

    def test_classify_passes_through_clean_match(self):
        v = mb.classify_lvs_report(_REP_MATCH, "")
        assert v["netgen_match"] and v["accept"]

    def test_classify_rejects_real_short_not_tied(self):
        # An unmatched top output pin that is NOT constant/aliased in the
        # reference (an independently-driven output shorted in layout) must NOT
        # be accepted -- the honest gate is preserved.
        report = _REP_TIED_PINS.replace("io_out[0]", "io_out[15]")
        v = mb.classify_lvs_report(report, _REF_V_TIED)  # io_out[15] has no tie
        assert not v["benign"] and not v["accept"]
        assert any("io_out[15]" in u for u in v["unresolved"])

    def test_classify_rejects_real_net_short_fragment(self):
        # A net-mismatch fragment on real (non-constant, non-macro-input) nets
        # is a genuine short and stays unresolved.
        report = """
NET mismatches: Class fragments follow (with fanout counts):
Circuit 1: chip_top                        |Circuit 2: chip_top
Net: real_bus_a                             |Net: real_bus_b
  u_core/reg_q[3]                           |  u_core/other_q[5]
---------------------------------------------------------------------------------------
Netlists do not match.
"""
        v = mb.classify_lvs_report(report, "")
        assert not v["benign"] and not v["accept"]

    def test_classify_accepts_representative_and_replication_targets(self):
        # The REPRESENTATIVE bits others alias to (a constant-0 net io_oeb[6], a
        # replicated real-enable net io_oeb[2]) also show as unmatched when the
        # layout renames the collapsed net -- both are single-driver tie targets
        # and must be accepted.
        info = mb.parse_output_tie_classes(_REF_V_TIED)
        assert "io_oeb[2]" in info["alias_targets"]  # replicated real enable
        assert "io_oeb[6]" in info["alias_targets"]  # constant-0 representative
        report = _REP_TIED_PINS.replace(
            "io_oeb[9]                                  |(no matching pin)",
            "io_oeb[2]                                  |(no matching pin)\n"
            "io_oeb[6]                                  |(no matching pin)",
        )
        v = mb.classify_lvs_report(report, _REF_V_TIED)
        assert v["benign"] and v["accept"] and v["unresolved"] == []

    def test_env_gate_default_on(self, monkeypatch):
        monkeypatch.delenv("CORESMITH_LVS_VERIFY_TIES", raising=False)
        assert mb.lvs_verify_ties_enabled() is True
        monkeypatch.setenv("CORESMITH_LVS_VERIFY_TIES", "0")
        assert mb.lvs_verify_ties_enabled() is False
