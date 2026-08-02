# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""How the per-block yosys synth is INVOKED (cwd + script preamble).

Block-level synthesis is not a toy check -- it is the synthesizability gate and
the input to the block PPA/area numbers -- but it was launched differently from
every other yosys site in the flow, and each difference showed up as a bogus
block-level failure or a bogus block-level timing number:

* it ran in whatever directory the daemon was started in, so a PROJECT-RELATIVE
  ``$readmemh`` / ``INIT_FILE`` path in the RTL (the same path DV resolves
  happily) was unreadable;
* it omitted the ``chparam -set MEM_IMPL "MACRO"`` the flat/backend synth
  applies, so a wrapped memory elaborated BEHAVIORALLY and STA timed a
  thousands-deep read mux -- on the same RTL the backend bound to a macro.
"""
from __future__ import annotations

import shutil

import pytest

from orchestrator.langgraph import pipeline_helpers as ph

_HAS_YOSYS = shutil.which("yosys") is not None


class _StopBeforeYosys(Exception):
    """Raised by the fake subprocess so the call is observed, not executed."""


def _project(tmp_path):
    """A project root with an `inputs/` artifact the RTL reads by relative path."""
    (tmp_path / "inputs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "inputs" / "table.memh").write_text(
        "\n".join(f"{i:02x}" for i in range(16)) + "\n")
    rtl = tmp_path / "rtl"
    rtl.mkdir(parents=True, exist_ok=True)
    f = rtl / "table_reader.v"
    f.write_text(
        "module table_reader(input clk, input [3:0] a, output reg [7:0] q);\n"
        "  reg [7:0] rom [0:15];\n"
        '  initial $readmemh("inputs/table.memh", rom);\n'
        "  always @(posedge clk) q <= rom[a];\n"
        "endmodule\n")
    return f


class TestSynthRunsAtTheProjectRoot:
    """A relative artifact path in the RTL must mean the same thing in synth as
    it does in simulation."""

    def test_yosys_is_launched_with_cwd_at_the_project_root(
            self, tmp_path, monkeypatch):
        monkeypatch.setattr(ph, "PROJECT_ROOT", tmp_path)
        monkeypatch.setenv("CORESMITH_SYNTH_GENERIC", "1")
        rtl = _project(tmp_path)
        seen = {}

        def _fake_run(cmd, **kw):
            seen["cwd"] = kw.get("cwd")
            raise _StopBeforeYosys()

        monkeypatch.setattr(ph.subprocess, "run", _fake_run)
        with pytest.raises(_StopBeforeYosys):
            ph.synthesize_block({"name": "table_reader"}, str(rtl))
        assert seen["cwd"] == str(tmp_path.resolve())

    @pytest.mark.skipif(not _HAS_YOSYS, reason="yosys not installed")
    def test_a_relative_readmemh_resolves(self, tmp_path, monkeypatch):
        """The real thing: yosys must find `inputs/table.memh` from the project
        root. Before the fix it looked in the daemon's cwd and the block failed
        synth with 'Can not open file' while DV read the identical path."""
        monkeypatch.setattr(ph, "PROJECT_ROOT", tmp_path)
        monkeypatch.setenv("CORESMITH_SYNTH_GENERIC", "1")
        rtl = _project(tmp_path)
        out = ph.synthesize_block({"name": "table_reader"}, str(rtl))
        assert out["success"], out.get("log", "")[-2000:]

    @pytest.mark.skipif(not _HAS_YOSYS, reason="yosys not installed")
    def test_a_missing_artifact_still_fails(self, tmp_path, monkeypatch):
        """The fix must not paper over a genuinely absent image -- rooting the
        run is what makes 'not found' mean 'not in the project'."""
        monkeypatch.setattr(ph, "PROJECT_ROOT", tmp_path)
        monkeypatch.setenv("CORESMITH_SYNTH_GENERIC", "1")
        rtl = _project(tmp_path)
        (tmp_path / "inputs" / "table.memh").unlink()
        out = ph.synthesize_block({"name": "table_reader"}, str(rtl))
        assert not out["success"]


# ---------------------------------------------------------------------------
# The block-synth script must select the MACRO impl, exactly as flat synth does
# ---------------------------------------------------------------------------

def _mem_block(tmp_path, name, inst):
    """A block whose only interesting content is one memory-wrapper instance."""
    rtl = tmp_path / "rtl"
    rtl.mkdir(parents=True, exist_ok=True)
    f = rtl / f"{name}.v"
    f.write_text(
        f"module {name}(input clk, input [15:0] a, output [7:0] q);\n"
        f"{inst}\n"
        "endmodule\n")
    return f


def _script_for(tmp_path, monkeypatch, name, inst):
    """The .ys the block-synth builder emits (yosys itself is never run)."""
    monkeypatch.setattr(ph, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("CORESMITH_SYNTH_GENERIC", "1")
    f = _mem_block(tmp_path, name, inst)

    def _fake_run(cmd, **kw):
        raise _StopBeforeYosys()

    monkeypatch.setattr(ph.subprocess, "run", _fake_run)
    with pytest.raises(_StopBeforeYosys):
        ph.synthesize_block({"name": name}, str(f))
    return (tmp_path / "syn" / "output" / name / f"synth_{name}.ys").read_text()


_BIG_ROM = ('  cs_rom_1r #(.WIDTH(8), .DEPTH(3309), '
            '.INIT_FILE("inputs/rom_images/table.memh"))\n'
            '    u_rom (.clk(clk), .ce(1\'b1), .addr(a[11:0]), .rdata(q));')
_SMALL_MEM = ('  cs_mem_1rw #(.WIDTH(8), .DEPTH(16))\n'
              '    u_mem (.clk(clk), .ce(1\'b1), .we(1\'b0), .addr(a[3:0]),\n'
              '           .wdata(8\'d0), .rdata(q));')
_BIG_MEM = ('  cs_mem_1rw1r #(.WIDTH(32), .DEPTH(1024))\n'
            '    u_mem (.clk(clk), .ce0(1\'b1), .we0(1\'b0), .addr0(a[9:0]),\n'
            '           .wdata0(32\'d0), .rdata0(), .ce1(1\'b1),\n'
            '           .addr1(a[9:0]), .rdata1(q));')
_FLOP_TIER = ('  cs_fpmem_1rw #(.WIDTH(8), .DEPTH(4096))\n'
              '    u_fp (.clk(clk), .ce(1\'b1), .we(1\'b0), .addr(a[11:0]),\n'
              '          .wdata(8\'d0), .rdata(q));')


class TestBlockSynthSelectsTheMacroImpl:
    def test_a_macro_eligible_rom_gets_the_chparam(self, tmp_path, monkeypatch):
        script = _script_for(tmp_path, monkeypatch, "rom_block", _BIG_ROM)
        assert 'chparam -set MEM_IMPL "MACRO"' in script
        assert "cs_rom_1r" in script.split("hierarchy")[0]

    def test_a_macro_eligible_cs_mem_gets_the_chparam(self, tmp_path, monkeypatch):
        script = _script_for(tmp_path, monkeypatch, "mem_block", _BIG_MEM)
        assert 'chparam -set MEM_IMPL "MACRO"' in script

    def test_the_unified_primitive_pulls_in_the_wrapper_library(
            self, tmp_path, monkeypatch):
        """A block instantiating cs_mem_* DIRECTLY got no wrapper library read
        at all (uses_wrapper did not know the family), so the module was
        unresolved at `hierarchy -check` -- in synth, lint and sim alike."""
        from orchestrator.langgraph import sram_wrapper as sw
        assert sw.uses_wrapper("cs_mem_1rw1r #(.WIDTH(32), .DEPTH(1024)) u (")
        assert sw.uses_wrapper("cs_mem_1rw u_m (.clk(clk));")
        script = _script_for(tmp_path, monkeypatch, "mem_block", _BIG_MEM)
        assert "cs_sram.v" in script

    def test_a_sub_threshold_memory_does_not(self, tmp_path, monkeypatch):
        """16 words / 128 bits is below both triggers -- flops are correct
        there, and forcing a shell would hide a real (small) memory."""
        script = _script_for(tmp_path, monkeypatch, "small_block", _SMALL_MEM)
        assert "chparam" not in script

    def test_the_blessed_flop_tier_is_never_forced_to_a_macro(
            self, tmp_path, monkeypatch):
        """cs_fpmem_* hard-passes MEM_IMPL("FLOP"); it is excluded from the
        chparam module set, so it must not even trigger the directive."""
        script = _script_for(tmp_path, monkeypatch, "fp_block", _FLOP_TIER)
        assert "chparam" not in script

    def test_the_chparam_runs_before_hierarchy(self, tmp_path, monkeypatch):
        """Setting the parameter after `hierarchy` derives the modules is a
        no-op on the already-baked $paramod copies."""
        script = _script_for(tmp_path, monkeypatch, "rom_block", _BIG_ROM)
        assert script.index("chparam") < script.index("hierarchy -top")

    def test_the_env_gate_turns_it_off(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_SRAM_MACRO", "0")
        script = _script_for(tmp_path, monkeypatch, "rom_block", _BIG_ROM)
        assert "chparam" not in script

    def test_a_block_with_no_memory_is_unchanged(self, tmp_path, monkeypatch):
        script = _script_for(tmp_path, monkeypatch, "plain",
                             "  assign q = a[7:0];")
        assert "chparam" not in script
        assert "cs_sram" not in script


class TestTheThresholdHasOneSource:
    """The synth that AVOIDS a flop memory and the post-synth gate that
    REJECTS one must not be able to disagree about which geometries qualify."""

    def test_the_block_directive_uses_the_gate_s_predicate(self):
        from orchestrator.langgraph import sram_wrapper as sw
        assert sw.macro_impl_eligible(8, 3309)     # deep
        assert sw.macro_impl_eligible(32, 128)     # 4096 bits
        assert not sw.macro_impl_eligible(8, 16)
        assert not sw.macro_impl_eligible(0, 4096)

    def test_the_gate_and_the_directive_agree_geometry_by_geometry(self):
        from orchestrator.langgraph import sram_wrapper as sw
        for w, d in [(8, 16), (8, 255), (8, 256), (32, 62), (32, 63),
                     (8, 3309), (64, 1024)]:
            gate_flags = not sw.gate_memory_as_flops([(w, d)])[0]
            assert gate_flags is sw.macro_impl_eligible(w, d), (w, d)

    def test_an_env_override_moves_both_together(self, monkeypatch):
        from orchestrator.langgraph import sram_wrapper as sw
        monkeypatch.setenv("CORESMITH_SRAM_MIN_BITS", "64")
        monkeypatch.setenv("CORESMITH_SRAM_MACRO_DEPTH", "8192")
        assert sw.macro_impl_eligible(8, 16)                 # 128 bits >= 64
        assert not sw.gate_memory_as_flops([(8, 16)])[0]

    def test_only_the_macro_backed_families_are_counted(self):
        from orchestrator.langgraph import sram_wrapper as sw
        text = ("cs_fpmem_1rw #(.WIDTH(8), .DEPTH(4096)) u_a (.clk(c));\n"
                "cs_rom_1r #(.WIDTH(8), .DEPTH(3309)) u_b (.clk(c));\n"
                "cs_mem_1rw1r #(.WIDTH(32), .DEPTH(1024)) u_c (.clk(c));\n")
        assert sorted(sw.macro_wrapper_instances(text)) == [(8, 3309), (32, 1024)]
