"""Generic dedup of duplicate module definitions across integration sources.

Blocks often bundle the same shared macro (e.g. a Sky130 SRAM behavioural
model); the chip-level sim then sees the module declared multiple times and
Verilator MODDUP-aborts elaboration. `_dedup_module_sources` keeps the first
definition of each module name and strips the rest.
"""
from __future__ import annotations

import re

from orchestrator.langgraph.integration_helpers import _dedup_module_sources

_SRAM = "module sram_macro(\n  input clk\n);\n  reg [7:0] mem;\nendmodule"


def _defs(paths, name):
    return sum(
        len(re.findall(rf"^\s*module\s+{name}\b", open(p).read(), re.M))
        for p in paths
    )


def test_dedup_keeps_one_shared_macro(tmp_path):
    a = tmp_path / "a.v"
    b = tmp_path / "b.v"
    c = tmp_path / "c.v"
    a.write_text("module foo(\n);\nendmodule\n" + _SRAM + "\n")
    b.write_text("module bar(\n);\nendmodule\n" + _SRAM + "\n")
    c.write_text(_SRAM + "\nmodule baz(\n);\nendmodule\n")
    out = _dedup_module_sources([str(a), str(b), str(c)], tmp_path)
    # Exactly one sram_macro definition survives across all sources.
    assert _defs(out, "sram_macro") == 1
    # Unique modules are all preserved.
    for m in ("foo", "bar", "baz"):
        assert _defs(out, m) == 1


def test_no_duplicates_passes_through_unchanged(tmp_path):
    a = tmp_path / "a.v"
    a.write_text("module only(\n);\nendmodule\n")
    out = _dedup_module_sources([str(a)], tmp_path)
    assert out == [str(a)]  # unchanged path when nothing to dedup


# Regression: an `ifdef SYNTHESIS` blackbox / `else` behavioural pair declares
# the SAME module name twice WITHIN one file. The deduper is not preprocessor-
# aware, so it must NOT strip the second (behavioural) occurrence -- doing so
# deleted the only definition visible under simulation (SYNTHESIS undefined) and
# broke Verilator elaboration ("Cannot find module ..."). Only cross-file
# duplicates may be stripped.
_IFDEF_MACRO = (
    "`ifdef SYNTHESIS\n"
    "(* blackbox *)\n"
    "module sram_macro(\n  input clk\n);\nendmodule\n"
    "`else\n"
    "module sram_macro(\n  input clk\n);\n  reg [7:0] mem;\nendmodule\n"
    "`endif\n"
)


def test_synthesis_blackbox_first_does_not_strip_real_def(tmp_path):
    # C23: a pad wrapper carries a SYNTHESIS-ONLY blackbox of a child module
    # (`ifdef SYNTHESIS ... endif) and comes FIRST in the source list. Under
    # the Verilator lint/sim (SYNTHESIS undefined) that blackbox is inactive,
    # so it must NOT become cross-file "seen" and strip the REAL child
    # definition in a later file -- which produced %Error-MODMISSING for every
    # child. The real def must survive; the (inactive) blackbox may stay.
    wrap = tmp_path / "wrapper.v"
    wrap.write_text(
        "`ifdef SYNTHESIS\n"
        "module aes_round_core (input clk); endmodule\n"
        "`endif\n"
        "module wrapper (input clk); endmodule\n")
    child = tmp_path / "aes_round_core.v"
    child.write_text(
        "module aes_round_core (input clk, output [7:0] q);\n"
        "  assign q = 8'd0;\nendmodule\n")
    out = _dedup_module_sources([str(wrap), str(child)], tmp_path,
                                lib_module_names=set())
    child_txt = open(out[1]).read()
    assert "assign q = 8'd0;" in child_txt   # real def kept
    assert "[coresmith dedup] removed" not in child_txt


def test_intra_file_ifdef_else_pair_is_preserved(tmp_path):
    a = tmp_path / "block_packer.v"
    a.write_text("module block_packer(\n);\nendmodule\n" + _IFDEF_MACRO)
    out = _dedup_module_sources([str(a)], tmp_path)
    # No cross-file duplicate exists, so the file must pass through untouched and
    # BOTH ifdef/else variants of sram_macro must survive verbatim.
    assert out == [str(a)]
    assert _defs(out, "sram_macro") == 2
    # The behavioural body (the `else` branch) must not be stripped.
    assert "reg [7:0] mem;" in open(out[0]).read()
    assert "[coresmith dedup] removed" not in open(out[0]).read()


def test_ifdef_pair_kept_but_later_file_dup_stripped(tmp_path):
    # File 1 has the ifdef/else pair (both kept); a LATER file re-bundles the
    # same macro -> that cross-file copy is the real MODDUP hazard and IS
    # stripped, while file 1's in-file pair stays intact.
    a = tmp_path / "block_packer.v"
    b = tmp_path / "other_block.v"
    a.write_text("module block_packer(\n);\nendmodule\n" + _IFDEF_MACRO)
    b.write_text(_SRAM.replace("sram_macro", "sram_macro") + "\nmodule other(\n);\nendmodule\n")
    out = _dedup_module_sources([str(a), str(b)], tmp_path)
    # File 1 untouched (in-file pair preserved); file 2's duplicate stripped.
    assert _defs([out[0]], "sram_macro") == 2
    assert _defs([out[1]], "sram_macro") == 0
    assert _defs(out, "other") == 1
