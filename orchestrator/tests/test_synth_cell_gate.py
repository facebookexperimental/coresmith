# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the cell-explosion synthesizability guard (fix #2).

The memory-preserving FF probe stops at `proc` and judges flip-flops only, so a
combinational-LUT explosion (entropy coding tables, per-mode-replicated prediction) is
invisible to it. The cell-count probe materializes the gate cloud via generic
techmap and fails on a techmap timeout OR a cell count past the ceiling --
running even under CORESMITH_SKIP_SYNTH. All deterministic; the live-yosys probe
is skipped when yosys is absent.
"""
from __future__ import annotations

import shutil

import pytest

from orchestrator.langgraph import ppa_check as pc


def test_count_cells_uses_last_stat_block():
    txt = "Number of cells: 12\n... mid-flow ...\nNumber of cells:  1,234,567\n"
    assert pc.count_cells_from_stat(txt) == 1234567


def test_count_cells_none_when_absent():
    assert pc.count_cells_from_stat("no stat section here") is None


def test_cell_gate_default_on_disable_off(monkeypatch):
    monkeypatch.delenv("CORESMITH_SYNTH_CELL_GATE", raising=False)
    assert pc.synth_cell_gate_enabled() is True
    monkeypatch.setenv("CORESMITH_SYNTH_CELL_GATE", "0")
    assert pc.synth_cell_gate_enabled() is False


def test_max_cell_ceiling_default_and_override(monkeypatch):
    monkeypatch.delenv("CORESMITH_MAX_CELLS", raising=False)
    assert pc.max_cell_ceiling() == 750000
    monkeypatch.setenv("CORESMITH_MAX_CELLS", "1000")
    assert pc.max_cell_ceiling() == 1000


def test_probe_none_when_rtl_absent():
    assert pc.probe_synth_cellcount("/no/such/file.v", "foo") is None


@pytest.mark.skipif(not shutil.which("yosys"), reason="yosys not available")
def test_probe_small_module_counts_cells(tmp_path):
    v = tmp_path / "tiny.v"
    v.write_text(
        "module tiny(input clk, input [7:0] a, input [7:0] b,\n"
        "            output reg [8:0] s);\n"
        "  always @(posedge clk) s <= a + b;\n"
        "endmodule\n"
    )
    r = pc.probe_synth_cellcount(str(v), "tiny")
    assert r is not None
    assert r["elaborated"] is True
    assert r["cell_count"] is not None and 0 < r["cell_count"] < 750000


@pytest.mark.skipif(not shutil.which("yosys"), reason="yosys not available")
def test_probe_multi_counts_cells(tmp_path):
    a = tmp_path / "a.v"
    a.write_text("module a(input x, output y); assign y = ~x; endmodule\n")
    top = tmp_path / "top.v"
    top.write_text(
        "module top(input x, output y); a u(.x(x), .y(y)); endmodule\n"
    )
    r = pc.probe_synth_cellcount_multi([str(top), str(a)], "top")
    assert r is not None and r["elaborated"] is True
    assert r["cell_count"] is not None


def test_chip_top_synth_ok_no_top_never_blocks():
    from orchestrator.langgraph.pipeline_graph import _chip_top_synth_ok
    ok, reason = _chip_top_synth_ok("", "top", "/no/such/top.v", {})
    assert ok is True and reason == ""


@pytest.mark.skipif(not shutil.which("yosys"), reason="yosys not available")
def test_chip_top_synth_ok_pass_ceiling_and_disabled(tmp_path, monkeypatch):
    from orchestrator.langgraph.pipeline_graph import _chip_top_synth_ok
    top = tmp_path / "top.v"
    top.write_text(
        "module top(input clk, input [7:0] a, output reg [7:0] y);\n"
        "  always @(posedge clk) y <= a + 1;\nendmodule\n"
    )
    monkeypatch.delenv("CORESMITH_MAX_CELLS", raising=False)
    monkeypatch.delenv("CORESMITH_SYNTH_CELL_GATE", raising=False)
    ok, reason = _chip_top_synth_ok("", "top", str(top), {})
    assert ok is True and reason == ""
    # an absurdly tight ceiling -> the integrated top fails the cell gate
    monkeypatch.setenv("CORESMITH_MAX_CELLS", "1")
    ok2, reason2 = _chip_top_synth_ok("", "top", str(top), {})
    assert ok2 is False and "ceiling" in reason2
    # gate disabled -> never blocks
    monkeypatch.setenv("CORESMITH_SYNTH_CELL_GATE", "0")
    ok3, _ = _chip_top_synth_ok("", "top", str(top), {})
    assert ok3 is True


# --- fix #4: combinational-depth gate (scheduler made enforcing) -------------

def test_logic_depth_parse_last_match():
    txt = "Longest topological path in m (length=12):\n... (length=35):"
    assert pc.count_logic_depth_from_ltp(txt) == 35


def test_logic_depth_parse_none():
    assert pc.count_logic_depth_from_ltp("no topological path here") is None


def test_logic_depth_gate_default_on_disable(monkeypatch):
    monkeypatch.delenv("CORESMITH_LOGIC_DEPTH_GATE", raising=False)
    assert pc.logic_depth_gate_enabled() is True
    monkeypatch.setenv("CORESMITH_LOGIC_DEPTH_GATE", "0")
    assert pc.logic_depth_gate_enabled() is False


def test_max_logic_depth_default_and_override(monkeypatch):
    monkeypatch.delenv("CORESMITH_MAX_LOGIC_DEPTH", raising=False)
    assert pc.max_logic_depth() == 500
    monkeypatch.setenv("CORESMITH_MAX_LOGIC_DEPTH", "10")
    assert pc.max_logic_depth() == 10


@pytest.mark.skipif(not shutil.which("yosys"), reason="yosys not available")
def test_probe_logic_depth_measures_chain(tmp_path):
    v = tmp_path / "deep.v"
    v.write_text(
        "module deep(input clk, input [7:0] a, output reg [7:0] y);\n"
        "  wire [7:0] t1=a+1,t2=t1+1,t3=t2+1,t4=t3+1,t5=t4+1,t6=t5+1;\n"
        "  always @(posedge clk) y <= t6;\nendmodule\n"
    )
    r = pc.probe_logic_depth(str(v), "deep")
    assert r is not None and r["elaborated"] is True
    assert r["logic_depth"] is not None and r["logic_depth"] > 0


@pytest.mark.skipif(not shutil.which("yosys"), reason="yosys not available")
def test_probe_readmemh_resolves_from_project_root_cwd(tmp_path):
    """C27: a project-root-relative $readmemh INIT path fails the probe when
    yosys runs from an unrelated cwd, and passes with cwd=project_root.

    Faithful to the real failure: the chip_top gate STAGES deduped source
    copies into a temp dir, so yosys's source-file-relative $readmemh fallback
    cannot save it either -- only cwd=project_root resolves 'inputs/...'.
    """
    proj = tmp_path / "proj"
    (proj / "inputs").mkdir(parents=True)
    (proj / "inputs" / "rom.memh").write_text("00\n01\n02\n03\n")
    stage = tmp_path / "chiptop_synth_stage"   # mimics the dedup staging dir
    stage.mkdir()
    v = stage / "romtop.v"
    v.write_text(
        "module romtop(input clk, input [1:0] addr, output reg [7:0] q);\n"
        "  reg [7:0] mem [0:3];\n"
        "  initial $readmemh(\"inputs/rom.memh\", mem);\n"
        "  always @(posedge clk) q <= mem[addr];\n"
        "endmodule\n"
    )
    # From an unrelated cwd (and a staged source copy far from inputs/) the
    # relative path cannot resolve -> yosys errors.
    bad = pc.probe_synth_cellcount(str(v), "romtop", cwd=str(stage))
    assert bad is not None and bad["elaborated"] is False
    assert "readmem" in bad["reason"].lower() or "Can not open" in bad["reason"]
    # From the project root it resolves and techmaps.
    good = pc.probe_synth_cellcount(str(v), "romtop", cwd=str(proj))
    assert good is not None and good["elaborated"] is True
    assert good["cell_count"] is not None

    multi = pc.probe_synth_cellcount_multi([str(v)], "romtop", cwd=str(proj))
    assert multi is not None and multi["elaborated"] is True


@pytest.mark.skipif(not shutil.which("yosys"), reason="yosys not available")
def test_delivered_abi_top_probed_and_drift_recorded(tmp_path, monkeypatch):
    """F3: a delivered rtl/chip_top.v outside the manifest is co-elaborated,
    published in the canonical filelist, probed as a second top, and a
    canonical_top_drift defect is recorded. A drifted (broken) delivered top
    fails the gate even when the assembled manifest top is fine."""
    from orchestrator.langgraph.pipeline_graph import (
        _chip_top_synth_ok,
        read_carried_forward_defects,
    )
    monkeypatch.delenv("CORESMITH_SYNTH_CELL_GATE", raising=False)
    monkeypatch.setenv("CORESMITH_CHIP_TOP_MIN_CELLS", "0")
    proj = tmp_path / "proj"
    (proj / "rtl").mkdir(parents=True)
    blk = proj / "rtl" / "blk.v"
    blk.write_text(
        "module blk(input clk, input [7:0] a, output reg [7:0] y);\n"
        "  always @(posedge clk) y <= a ^ 8'h5a;\nendmodule\n")
    top = proj / "rtl" / "assembled_top.v"
    top.write_text(
        "module assembled_top(input clk, input [7:0] a, output [7:0] y);\n"
        "  blk u_b(.clk(clk), .a(a), .y(y));\nendmodule\n")
    # delivered ABI top: a NOVEL wrapper module, not in the manifest
    (proj / "rtl" / "chip_top.v").write_text(
        "module ppab_dut(input clk, input [7:0] a, output [7:0] y);\n"
        "  blk u_b(.clk(clk), .a(a), .y(y));\nendmodule\n")

    ok, reason = _chip_top_synth_ok(
        str(proj), "assembled_top", str(top), {"blk": str(blk)})
    assert ok is True, reason
    flist = (proj / ".coresmith" / "chip_top_sources.f").read_text()
    assert "chip_top.v" in flist
    defects = read_carried_forward_defects(str(proj))
    assert any(d.get("kind") == "canonical_top_drift" for d in defects)

    # now DRIFT the delivered top (references a module that no longer exists)
    (proj / "rtl" / "chip_top.v").write_text(
        "module ppab_dut(input clk, input [7:0] a, output [7:0] y);\n"
        "  blk_renamed u_b(.clk(clk), .a(a), .y(y));\nendmodule\n")
    ok2, reason2 = _chip_top_synth_ok(
        str(proj), "assembled_top", str(top), {"blk": str(blk)})
    assert ok2 is False
    assert "delivered ABI top" in reason2


def test_delivered_top_with_module_collision_records_unchecked(tmp_path, monkeypatch):
    """F3: a delivered top that REDEFINES manifest modules cannot be
    co-elaborated -- the gate stays judgeable on the manifest and records the
    drift as UNCHECKED instead of MODDUP-failing."""
    from orchestrator.langgraph.pipeline_graph import (
        _chip_top_synth_ok,
        read_carried_forward_defects,
    )
    if not shutil.which("yosys"):
        pytest.skip("yosys not available")
    monkeypatch.delenv("CORESMITH_SYNTH_CELL_GATE", raising=False)
    monkeypatch.setenv("CORESMITH_CHIP_TOP_MIN_CELLS", "0")
    proj = tmp_path / "proj"
    (proj / "rtl").mkdir(parents=True)
    top = proj / "rtl" / "assembled_top.v"
    top.write_text(
        "module assembled_top(input clk, input [7:0] a, output reg [7:0] y);\n"
        "  always @(posedge clk) y <= a + 1;\nendmodule\n")
    # delivered file redefines assembled_top itself -> collision
    (proj / "rtl" / "chip_top.v").write_text(
        "module assembled_top(input clk, input [7:0] a, output reg [7:0] y);\n"
        "  always @(posedge clk) y <= a + 2;\nendmodule\n")
    ok, reason = _chip_top_synth_ok(str(proj), "assembled_top", str(top), {})
    assert ok is True, reason
    defects = read_carried_forward_defects(str(proj))
    drift = [d for d in defects if d.get("kind") == "canonical_top_drift"]
    assert drift and "UNCHECKED" in drift[0].get("detail", "")
