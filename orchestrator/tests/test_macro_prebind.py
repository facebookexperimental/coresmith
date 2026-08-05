# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Pre-synthesis macro binding: the shell must carry a real memory.

The bug these guard against: cs_mem_macro_shell hard-assigns zero, synthesis
preserves the tie-off, and a gate-sim of the flat netlist diverges on the first
read of real data -- while every upstream RTL gate passes, because RTL uses the
BEHAV arm.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from orchestrator.langgraph.macro_prebind import (
    UNBOUND_SENTINEL,
    PrebindResult,
    emit_bound_shell,
)


@dataclass
class FakeSpec:
    width: int
    depth: int
    nport: int = 2

    def describe(self) -> str:
        return f"sram {self.width}b x {self.depth}"


@dataclass
class FakeMacro:
    name: str
    verilog: str = "/pdk/verilog/m.v"
    mask_bits: int = 0


def _bound(*pairs) -> PrebindResult:
    return PrebindResult(bindings=list(pairs))


class TestActiveLowPolarity:
    """OpenRAM csb0/web0 are active LOW; the shell's ce0/we0 are active high.

    Dropping the inversion gives a memory selected exactly when it should be
    idle -- plausible garbage rather than an obvious failure.
    """

    def test_chip_select_and_write_enable_are_inverted(self):
        v = emit_bound_shell(_bound((FakeSpec(8, 4096), FakeMacro("sram_a"))))
        assert ".csb0(~ce0)" in v
        assert ".csb0(ce0)" not in v, "active-low port driven with active-high signal"
        # A macro with no wmask0 gets the whole-word mask folded into its write
        # enable, so web0 is the INVERTED (we0 AND mask) rather than plain ~we0.
        assert ".web0(~(we0 & (&wmask0)))" in v
        assert ".web0(we0)" not in v

    def test_second_read_port_select_also_inverted(self):
        v = emit_bound_shell(_bound((FakeSpec(8, 4096, nport=2), FakeMacro("sram_a"))))
        assert ".csb1(~ce1)" in v


class TestUnboundIsLoudNotZero:
    def test_fallback_references_an_undefined_module(self):
        """The old behaviour read zeros. Anything that still elaborates is a
        regression, so the fallback must name a module no tool can resolve."""
        v = emit_bound_shell(_bound((FakeSpec(8, 4096), FakeMacro("sram_a"))))
        assert UNBOUND_SENTINEL in v
        assert "u_unbound_geometry" in v

    def test_no_zero_tieoff_anywhere(self):
        v = emit_bound_shell(_bound((FakeSpec(8, 4096), FakeMacro("sram_a"))))
        assert "{WIDTH{1'b0}}" not in v, "reintroduced the zero-read tie-off"

    def test_empty_bindings_still_emit_the_loud_fallback(self):
        """No resolved macro at all must not produce a permissive shell."""
        v = emit_bound_shell(PrebindResult())
        assert UNBOUND_SENTINEL in v
        assert "{WIDTH{1'b0}}" not in v


class TestGeometryDispatch:
    def test_each_geometry_gets_its_own_guarded_arm(self):
        v = emit_bound_shell(
            (_bound((FakeSpec(8, 4096), FakeMacro("sram_8x4096")),
                    (FakeSpec(9, 4096), FakeMacro("sram_9x4096")))))
        assert "WIDTH == 8 && DEPTH == 4096 && NPORT == 2" in v
        assert "WIDTH == 9 && DEPTH == 4096 && NPORT == 2" in v
        assert "sram_8x4096 u_macro" in v
        assert "sram_9x4096 u_macro" in v
        # one `if`, the rest chained -- not independent ifs that could both match
        assert v.count("    if (") == 1
        assert v.count("    else if (") == 1

    def test_single_port_macro_has_no_port1_connections(self):
        v = emit_bound_shell(_bound((FakeSpec(32, 512, nport=1), FakeMacro("sram_1rw"))))
        assert ".clk1(" not in v and ".csb1(" not in v and ".dout1(" not in v
        # the shell still exposes rdata1, so it must be driven from port 0
        assert "assign rdata1 = rdata0;" in v

    def test_write_mask_connected_only_when_the_model_declares_it(self, tmp_path):
        """Connect pins the MODEL has, not pins the registry claims.

        MacroInfo.mask_bits was non-zero for sram_1rw1r_8_4096_8_sky130, whose
        .v has no wmask0 port -- connecting it was a hard elaboration error.
        """
        masked = tmp_path / "masked.v"
        masked.write_text(
            "module m(clk0, csb0, web0, wmask0, addr0, din0, dout0);\n"
            "  input clk0; input csb0; input web0; input [3:0] wmask0;\n"
            "  input [8:0] addr0; input [31:0] din0; output [31:0] dout0;\n"
            "endmodule\n")
        v = emit_bound_shell(_bound(
            (FakeSpec(32, 512, nport=1), FakeMacro("m", str(masked), mask_bits=4))))
        # The SHELL's mask, not a constant. Tying it high was the defect: every
        # partial write silently became a full-word write.
        assert ".wmask0(wmask0)" in v
        assert "1'b1}}" not in v.split(".wmask0")[1][:40], "mask tied to a constant"

        plain = tmp_path / "plain.v"
        plain.write_text(
            "module p(clk0, csb0, web0, addr0, din0, dout0);\n"
            "  input clk0; input csb0; input web0; input [11:0] addr0;\n"
            "  input [7:0] din0; output [7:0] dout0;\n"
            "endmodule\n")
        v2 = emit_bound_shell(_bound(
            (FakeSpec(8, 4096, nport=1), FakeMacro("p", str(plain), mask_bits=1))))
        assert ".wmask0(" not in v2, "connected a pin the model does not declare"
        # ...but the mask still reaches the memory, folded into web0.
        assert ".web0(~(we0 & (&wmask0)))" in v2


def _maskless_env(tmp_path, monkeypatch, width, nb_expected):
    """Wire resolve_prebindings to a maskless macro for the given RTL width."""
    import orchestrator.langgraph.macro_registry as reg
    from orchestrator.langgraph import macro_prebind as mp

    rtl = tmp_path / "blk.v"
    rtl.write_text(
        f"module blk; cs_sram_1rw1r #(.WIDTH({width}), .DEPTH(4096), "
        ".USE_WMASK(1), .WMASK_GRAN(8)) u (); endmodule\n")

    class Spec:
        kind, width_, depth, nport = "sram", width, 4096, 2

        def __init__(self, **kw):
            self.width = kw.get("width", width)
            self.depth = kw.get("depth", 4096)
            self.nport = kw.get("nport", 2)

        def describe(self):
            return f"sram shell {self.width}b x {self.depth}"

    monkeypatch.setattr(mp, "macro_ports",
                        lambda p: {"clk0", "csb0", "web0", "addr0",
                                   "din0", "dout0"})
    monkeypatch.setattr(reg, "detect_macro_shells", lambda t: [])
    monkeypatch.setattr(reg, "discover_macros", lambda: {})
    monkeypatch.setattr(reg, "ShellSpec", Spec, raising=False)
    monkeypatch.setattr(reg, "resolve_shell",
                        lambda s, **kw: FakeMacro("nm", "/p/nm.v", nb_expected))
    return mp, str(rtl)


class TestMaskMismatchIsRefused:
    """A real per-byte mask cannot be expressed by a single write enable.

    Tying wmask high turns a masked write into a full-word write, clobbering
    neighbouring bytes. RTL DV passes (the BEHAV arm honours the mask), so the
    corruption only appears in the macro-backed design.
    """

    def test_multibit_mask_against_maskless_macro_is_refused(self, tmp_path, monkeypatch):
        # WIDTH=32 / GRAN=8 -> 4 mask bits: genuinely per-byte.
        mp, rtl = _maskless_env(tmp_path, monkeypatch, 32, 4)
        res = mp.resolve_prebindings([rtl], allow_generate=False)
        assert res.ok is False
        assert res.unresolved, "maskless macro silently accepted for masked RTL"
        assert any("wmask0" in e for e in res.errors)
        assert any("full-word writes" in e for e in res.errors)

    def test_whole_word_mask_binds_and_the_shell_folds_it(self, tmp_path, monkeypatch):
        """WIDTH=8 / GRAN=8 -> 1 mask bit spanning the word, which is exactly
        what a write enable expresses -- and why OpenRAM omits the port.

        The shell now performs the fold itself, so this is correct by
        construction instead of depending on the RTL author remembering to write
        `we0 = write_fire & wmask` (framebuffer_sram had to be hand-patched to do
        exactly that)."""
        mp, rtl = _maskless_env(tmp_path, monkeypatch, 8, 1)
        res = mp.resolve_prebindings([rtl], allow_generate=False)
        assert res.bindings, "refused a mask that is equivalent to we0"
        assert not res.unresolved and not res.errors
        assert res.ok is True
        assert res.mask_lanes.get((8, 4096, 2)) == 1
        assert any("folded" in w for w in res.warnings)

    def test_multibit_mask_against_a_MASK_CAPABLE_macro_now_BINDS(
        self, tmp_path, monkeypatch
    ):
        """The shell routes the mask, so this is no longer refused -- it works.

        Two of the three memories in exp-raster-macro-20260727 were in exactly
        this state, both driving a genuinely dynamic mask: triangle_store
        (WIDTH=64, 8 lanes) and zbuffer_sram (WIDTH=9, 2 lanes). Both used to
        bind with the mask tied to all-ones and silently lose it; refusing them
        was honest but left the design unable to reach a netlist at all.
        """
        mp, rtl = _maskless_env(tmp_path, monkeypatch, 64, 8)
        # This macro DOES declare wmask0, with a matching 8 lanes.
        monkeypatch.setattr(mp, "macro_ports",
                            lambda p: {"clk0", "csb0", "web0", "wmask0",
                                       "addr0", "din0", "dout0"})
        monkeypatch.setattr(mp, "macro_mask_lanes", lambda p: 8)
        res = mp.resolve_prebindings([rtl], allow_generate=False)
        assert res.ok is True, res.errors
        assert res.bindings and not res.unresolved
        assert res.mask_lanes.get((64, 4096, 2)) == 8
        # And the emitted shell wires it rather than tying it.
        assert ".wmask0(wmask0)" in mp.emit_bound_shell(res)

    def test_lane_count_disagreement_is_refused_not_truncated(
        self, tmp_path, monkeypatch
    ):
        """The old binder replicated `{mask_bits{1'b1}}` and let yosys truncate
        it -- an 8-bit constant into the 2-bit port of
        sram_1rw1r_9_4096_8_sky130. A lane-count disagreement decides which
        BYTES a write touches, so it must refuse."""
        mp, rtl = _maskless_env(tmp_path, monkeypatch, 64, 8)
        monkeypatch.setattr(mp, "macro_ports",
                            lambda p: {"clk0", "csb0", "web0", "wmask0",
                                       "addr0", "din0", "dout0"})
        monkeypatch.setattr(mp, "macro_mask_lanes", lambda p: 2)   # != 8
        res = mp.resolve_prebindings([rtl], allow_generate=False)
        assert res.ok is False and res.unresolved
        joined = " ".join(res.errors)
        assert "8 write-mask lane(s)" in joined and "2 lane(s)" in joined
        assert "truncated" in joined

    def test_unverifiable_lane_count_is_refused(self, tmp_path, monkeypatch):
        """`None` means the width is an expression we will not evaluate. That is
        "cannot verify", which must never become a default."""
        mp, rtl = _maskless_env(tmp_path, monkeypatch, 64, 8)
        monkeypatch.setattr(mp, "macro_ports",
                            lambda p: {"clk0", "csb0", "web0", "wmask0",
                                       "addr0", "din0", "dout0"})
        monkeypatch.setattr(mp, "macro_mask_lanes", lambda p: None)
        res = mp.resolve_prebindings([rtl], allow_generate=False)
        assert res.ok is False and res.unresolved
        assert "unresolvable" in " ".join(res.errors)

    def test_multibit_mask_against_a_MASKLESS_macro_is_still_refused(
        self, tmp_path, monkeypatch
    ):
        """No wiring can carry 4 lanes to a macro with a single whole-word
        write enable. Regenerating the macro is the only fix."""
        mp, rtl = _maskless_env(tmp_path, monkeypatch, 32, 4)
        res = mp.resolve_prebindings([rtl], allow_generate=False)
        assert res.ok is False and res.unresolved
        joined = " ".join(res.errors)
        assert "no wmask0 port at all" in joined
        assert "write_size" in joined


class TestStructure:
    def test_generate_block_is_balanced(self):
        v = emit_bound_shell(_bound((FakeSpec(8, 4096), FakeMacro("sram_a"))))
        # "generate" is a substring of "endgenerate", so count the opener by
        # excluding the closer rather than by naive substring count.
        assert v.count("endgenerate") == 1
        assert v.count("generate") - v.count("endgenerate") == 1
        assert v.count("endmodule") == 1
        assert len([ln for ln in v.split("\n")
                    if ln.startswith("module ")]) == 1

    def test_marked_generated(self):
        v = emit_bound_shell(_bound((FakeSpec(8, 4096), FakeMacro("sram_a"))))
        assert "GENERATED" in v and "do not edit" in v


class TestResultContract:
    def test_ok_is_false_when_anything_is_unresolved(self):
        r = PrebindResult(bindings=[(FakeSpec(8, 4096), FakeMacro("a"))],
                          unresolved=[FakeSpec(9, 4096)])
        assert r.ok is False

    def test_ok_is_false_on_errors_even_with_bindings(self):
        r = PrebindResult(bindings=[(FakeSpec(8, 4096), FakeMacro("a"))],
                          errors=["boom"])
        assert r.ok is False

    def test_ok_true_only_when_clean(self):
        assert PrebindResult(bindings=[(FakeSpec(8, 4096), FakeMacro("a"))]).ok is True

    def test_model_paths_dedupes_and_keeps_order(self):
        r = _bound((FakeSpec(8, 4096), FakeMacro("a", verilog="/p/a.v")),
                   (FakeSpec(9, 4096), FakeMacro("b", verilog="/p/b.v")),
                   (FakeSpec(8, 64), FakeMacro("c", verilog="/p/a.v")))
        assert r.model_paths() == ["/p/a.v", "/p/b.v"]

    def test_model_paths_skips_macros_without_a_model(self):
        r = _bound((FakeSpec(8, 4096), FakeMacro("a", verilog="")))
        assert r.model_paths() == []


class TestSynthSourcePreparation:
    """Reading the original library AND the bound shell would be a duplicate
    definition of cs_mem_macro_shell, so the strip is load-bearing."""

    def test_strip_removes_the_zero_driving_shell_only(self):
        from orchestrator.langgraph.macro_prebind import strip_module
        text = ("module keep_me; endmodule\n"
                "module cs_mem_macro_shell #(parameter W=1)(input a);\n"
                "  assign x = {W{1'b0}};\n"
                "endmodule\n"
                "module also_keep; endmodule\n")
        out, n = strip_module(text, "cs_mem_macro_shell")
        assert n == 1
        assert "cs_mem_macro_shell" not in out
        assert "module keep_me" in out and "module also_keep" in out

    def test_shared_library_file_is_never_modified(self, tmp_path):
        from orchestrator.langgraph.macro_prebind import prepare_synth_sources
        lib = tmp_path / "cs_sram.v"
        original = ("module cs_mem_1rw1r; endmodule\n"
                    "module cs_mem_macro_shell; assign r = 0; endmodule\n")
        lib.write_text(original)
        res = _bound((FakeSpec(8, 4096), FakeMacro("m", "/p/m.v")))

        got = prepare_synth_sources(str(lib), res, tmp_path / "work")

        assert lib.read_text() == original, "mutated the shared wrapper library"
        assert got["wrapper_lib"] != str(lib), "did not substitute a filtered copy"
        filtered = Path(got["wrapper_lib"]).read_text()
        # The name still appears in the generated header comment; what must be
        # gone is the DEFINITION, which is what would collide.
        assert "module cs_mem_macro_shell" not in filtered
        assert "cs_mem_1rw1r" in filtered

    def test_bound_shell_and_models_are_returned(self, tmp_path):
        from orchestrator.langgraph.macro_prebind import prepare_synth_sources
        lib = tmp_path / "cs_sram.v"
        lib.write_text("module cs_mem_macro_shell; endmodule\n")
        res = _bound((FakeSpec(8, 4096), FakeMacro("m8", "/p/a.v")),
                     (FakeSpec(9, 4096), FakeMacro("m9", "/p/b.v")))
        got = prepare_synth_sources(str(lib), res, tmp_path / "w")
        assert Path(got["bound_shell"]).is_file()
        assert "m8 u_macro" in Path(got["bound_shell"]).read_text()
        assert got["models"] == ["/p/a.v", "/p/b.v"]

    def test_no_bindings_leaves_everything_untouched(self, tmp_path):
        """Nothing resolved => do not strip the shell and silently remove the
        only definition, leaving an undefined module."""
        from orchestrator.langgraph.macro_prebind import prepare_synth_sources
        lib = tmp_path / "cs_sram.v"
        lib.write_text("module cs_mem_macro_shell; endmodule\n")
        got = prepare_synth_sources(str(lib), PrebindResult(), tmp_path / "w")
        assert got["wrapper_lib"] == str(lib)
        assert got["bound_shell"] == "" and got["models"] == []


class TestResolveNeedsSources:
    def test_no_readable_sources_is_an_error_not_a_silent_pass(self, tmp_path):
        from orchestrator.langgraph.macro_prebind import resolve_prebindings
        r = resolve_prebindings([str(tmp_path / "missing.v")])
        assert r.ok is False and r.errors


class TestMacroPortsNeverFailsByOmission:
    """A missing name here is read as "the macro has no such pin", so
    ``_ports_for`` leaves it UNCONNECTED and it floats low. That is not a
    degraded answer, it is a wrong one: a dropped ``addr0`` is a memory stuck at
    word zero, and a dropped ``wmask0`` is what suppressed every framebuffer
    write in exp-raster-macro-20260727.
    """

    def _v(self, tmp_path, body):
        f = tmp_path / "m.v"
        f.write_text(body)
        return f

    def test_range_containing_parens(self, tmp_path):
        """THE regression. The old regex was `[^;)]*`, which cannot cross the
        `)` inside `$clog2(`, so `addr0` was never seen."""
        from orchestrator.langgraph.macro_prebind import macro_ports
        f = self._v(tmp_path,
                    "module m(clk0, addr0);\n"
                    "  input clk0;\n"
                    "  input [$clog2(DEPTH)-1:0] addr0;\n"
                    "endmodule\n")
        assert {"clk0", "addr0"} <= macro_ports(f)

    def test_nested_parens_in_a_range(self, tmp_path):
        from orchestrator.langgraph.macro_prebind import macro_ports
        f = self._v(tmp_path,
                    "module m(dout0);\n"
                    "  output [((WIDTH*2)-1):0] dout0;\n"
                    "endmodule\n")
        assert "dout0" in macro_ports(f)

    def test_several_names_in_one_declaration(self, tmp_path):
        from orchestrator.langgraph.macro_prebind import macro_ports
        f = self._v(tmp_path,
                    "module m(a, b, c);\n  input a, b, c;\nendmodule\n")
        assert {"a", "b", "c"} <= macro_ports(f)

    def test_ansi_header(self, tmp_path):
        from orchestrator.langgraph.macro_prebind import macro_ports
        f = self._v(tmp_path,
                    "module m(\n"
                    "  input wire clk0,\n"
                    "  input wire [$clog2(N)-1:0] addr0,\n"
                    "  output reg [7:0] dout0\n"
                    ");\nendmodule\n")
        assert {"clk0", "addr0", "dout0"} <= macro_ports(f)

    def test_type_keywords_are_not_ports(self, tmp_path):
        from orchestrator.langgraph.macro_prebind import macro_ports
        f = self._v(tmp_path,
                    "module m(a);\n  input wire signed [3:0] a;\nendmodule\n")
        got = macro_ports(f)
        assert "a" in got
        assert not ({"wire", "signed", "input"} & got)

    def test_comments_do_not_contribute_ports(self, tmp_path):
        from orchestrator.langgraph.macro_prebind import macro_ports
        f = self._v(tmp_path,
                    "module m(a);\n"
                    "  // input phantom_pin;\n"
                    "  /* inout other_phantom; */\n"
                    "  input a;\n"
                    "endmodule\n")
        got = macro_ports(f)
        assert "a" in got
        assert "phantom_pin" not in got and "other_phantom" not in got

    def test_unreadable_file_is_empty_not_an_exception(self, tmp_path):
        from orchestrator.langgraph.macro_prebind import macro_ports
        assert macro_ports(tmp_path / "absent.v") == set()

    def test_the_real_maskless_macro_still_reports_no_wmask0(self, tmp_path):
        """The scanner must not "fix" the 8x4096 macro into having a mask --
        it genuinely has none, and that is what the binder must see."""
        from orchestrator.langgraph.macro_prebind import macro_ports
        f = self._v(tmp_path,
                    "module sram_1rw1r_8_4096_8_sky130(clk0, csb0, web0, addr0,"
                    " din0, dout0, clk1, csb1, addr1, dout1);\n"
                    "  parameter ADDR_WIDTH = 12;\n"
                    "  input clk0;\n  input csb0;\n  input web0;\n"
                    "  input [ADDR_WIDTH-1:0] addr0;\n"
                    "  input [7:0] din0;\n  output [7:0] dout0;\n"
                    "endmodule\n")
        got = macro_ports(f)
        assert "addr0" in got and "din0" in got
        assert "wmask0" not in got
