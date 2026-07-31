# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The chip_top gate-sim must be WIRED, and its verdict must BITE.

These test the plumbing, not the verdict logic (that is test_chip_top_gate_sim).
They exist because the gate shipped in a state where it could never run on a
real design and could never block one, while its unit tests passed:

  * ``_run_chip_top_gate_sim`` read ``state["top_rtl_path"]``, which no part of
    the backend flow populates -- ``BackendState`` has no such field,
    ``start_backend``'s initial_state omits it, and ``init_design_node`` does
    not return it. Its test passed only because the test injected the value.
    On every real run the gate reported ``not_run``. The one real verdict ever
    produced came from a hand-written driver that read the source list out of
    the integration DV's Makefile.
  * ``run_simulation`` built ``VERILOG_SOURCES`` from ONE path, so even a
    correct single path could not elaborate an assembled chip.
  * Nothing read ``chip_gate_sim_ok``. A FAIL was recorded in state and the
    flow proceeded to P&R anyway.
"""
from __future__ import annotations

from pathlib import Path

import orchestrator.harness.gate_sim as gs
from orchestrator.langgraph.backend_graph import (
    _run_chip_top_gate_sim,
    route_after_flat_synth,
)
from orchestrator.langgraph.integration_helpers import chip_rtl_sources


# ---------------------------------------------------------------------------
# The reference source list
# ---------------------------------------------------------------------------

class TestChipRtlSources:
    """One definition, shared with integration/validation DV. If the gate's
    reference elaborates a different source set than the DV that passed, the
    comparison is against a different design and the verdict is meaningless."""

    def test_top_comes_first(self, tmp_path):
        top = tmp_path / "chip_top.v"
        top.write_text("module chip_top(); endmodule\n")
        blocks = {}
        for n in ("b1", "b2", "b3"):
            f = tmp_path / f"{n}.v"
            f.write_text(f"module {n}(); endmodule\n")
            blocks[n] = str(f)
        srcs = chip_rtl_sources(str(top), blocks)
        # Callers resolve the Verilator TOPLEVEL from the first entry.
        assert srcs[0] == str(top)
        assert set(srcs) == {str(top)} | set(blocks.values())

    def test_every_block_is_included(self, tmp_path):
        """The whole point: an assembled top is not one file."""
        top = tmp_path / "chip_top.v"
        top.write_text("module chip_top(); endmodule\n")
        blocks = {}
        for n in ("alpha", "beta", "gamma", "delta", "epsilon"):
            f = tmp_path / f"{n}.v"
            f.write_text(f"module {n}(); endmodule\n")
            blocks[n] = str(f)
        assert len(chip_rtl_sources(str(top), blocks)) == 6

    def test_nonexistent_block_is_skipped_not_fatal(self, tmp_path):
        top = tmp_path / "chip_top.v"
        top.write_text("module chip_top(); endmodule\n")
        srcs = chip_rtl_sources(str(top), {"ghost": str(tmp_path / "absent.v")})
        assert srcs == [str(top)]

    def test_top_is_not_duplicated_when_also_listed_as_a_block(self, tmp_path):
        """Verilator MODDUP-aborts on a duplicated module before any
        transaction runs, so a duplicate is a hard build failure."""
        top = tmp_path / "chip_top.v"
        top.write_text("module chip_top(); endmodule\n")
        srcs = chip_rtl_sources(str(top), {"chip_top": str(top)})
        assert srcs.count(str(top)) == 1

    def test_sram_wrapper_lib_is_added_when_a_BLOCK_uses_it(self, tmp_path):
        """The cs_sram instantiation lives in a LEAF block, not the top. A
        builder that only inspected the top would miss it and the chip-level
        build would die on 'Cannot find module cs_sram_1rw1r'."""
        top = tmp_path / "chip_top.v"
        top.write_text("module chip_top(); u_blk b(); endmodule\n")
        blk = tmp_path / "blk.v"
        blk.write_text(
            "module blk();\n"
            "  cs_sram_1rw1r #(.WIDTH(8), .DEPTH(4096)) u_mem (.clk(clk));\n"
            "endmodule\n"
        )
        srcs = chip_rtl_sources(str(top), {"blk": str(blk)})
        assert any("cs_sram" in s for s in srcs), srcs


# ---------------------------------------------------------------------------
# The gate is actually fed
# ---------------------------------------------------------------------------

def _run_dir(tmp_path):
    (tmp_path / "sim_build" / "integration").mkdir(parents=True)
    (tmp_path / "sim_build" / "integration" / "test_my_chip.py").write_text("#tb\n")
    top = tmp_path / "rtl" / "integration" / "my_chip.v"
    top.parent.mkdir(parents=True)
    top.write_text("module my_chip(input clk); endmodule\n")
    blk = tmp_path / "rtl" / "blk.v"
    blk.write_text("module blk(); endmodule\n")
    return tmp_path, top, blk


class TestTheGateIsFedARealSourceList:
    def test_sources_come_from_the_backend_state_the_flow_populates(
        self, tmp_path, monkeypatch
    ):
        """``integration_top_path`` + ``block_rtl_paths`` are what
        ``init_design_node`` really returns. Reading a field the flow never
        sets is how this gate self-disabled on every real run."""
        monkeypatch.setenv(gs.GATE_SIM_ENV, "1")
        root, top, blk = _run_dir(tmp_path)
        seen = {}

        def fake_check(*, block, netlist_path, rtl_path, tb_path):
            seen["rtl_path"] = rtl_path
            return gs.GateSimResult(ran=True, ok=True, status=gs.STATUS_PASS,
                                    reason="ok", cycles_compared=10,
                                    output_bits_compared=100)

        monkeypatch.setattr(gs, "check_gate_sim", fake_check)
        ok, status, _ = _run_chip_top_gate_sim({
            "project_root": str(root),
            "design_name": "my_chip",
            "integration_top_path": str(top),
            "block_rtl_paths": {"blk": str(blk)},
        }, "net.v")

        assert ok is True and status == gs.STATUS_PASS
        srcs = seen["rtl_path"]
        assert not isinstance(srcs, str), "an assembled chip is not one file"
        assert str(top) in srcs and str(blk) in srcs
        assert srcs[0] == str(top)

    def test_no_assembled_top_anywhere_is_not_run_never_a_pass(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv(gs.GATE_SIM_ENV, "1")
        root, _top, _blk = _run_dir(tmp_path)
        ok, status, reason = _run_chip_top_gate_sim({
            "project_root": str(root),
            "design_name": "my_chip",
        }, "net.v")
        assert ok is None and status == gs.STATUS_NOT_RUN
        assert reason                            # absence is always explained


# ---------------------------------------------------------------------------
# The verdict bites
# ---------------------------------------------------------------------------

class TestTheVerdictBlocksPnR:
    def test_fail_routes_to_diagnose_not_pnr(self, tmp_path):
        """The flat netlist provably does not reproduce the verified RTL.
        Hardening it would spend hours of P&R on a design that does not work."""
        net = tmp_path / "net.v"
        net.write_text("module chip_top(); endmodule\n")
        assert route_after_flat_synth({
            "flat_netlist_path": str(net), "chip_gate_sim_ok": False,
        }) == "diagnose"

    def test_pass_proceeds(self, tmp_path):
        net = tmp_path / "net.v"
        net.write_text("module chip_top(); endmodule\n")
        assert route_after_flat_synth({
            "flat_netlist_path": str(net), "chip_gate_sim_ok": True,
        }) == "run_pnr"

    def test_not_applicable_does_not_block(self, tmp_path):
        """``None`` means the gate did not APPLY (disabled, no integration TB,
        no toolchain). That is not a verdict, and a gate that cannot run must
        not wall off the flow -- it is already logged with a reason."""
        net = tmp_path / "net.v"
        net.write_text("module chip_top(); endmodule\n")
        assert route_after_flat_synth({
            "flat_netlist_path": str(net), "chip_gate_sim_ok": None,
        }) == "run_pnr"
        assert route_after_flat_synth({
            "flat_netlist_path": str(net),
        }) == "run_pnr"

    def test_missing_netlist_still_diagnoses(self, tmp_path):
        assert route_after_flat_synth({
            "flat_netlist_path": str(tmp_path / "absent.v"),
            "chip_gate_sim_ok": True,
        }) == "diagnose"


class TestModdupHazard:
    """A deterministically-assembled Caravel top and the pad-adapter BLOCK it
    was assembled from both declare ``module user_project_wrapper``. Two
    compilation units defining one module is a Verilator MODDUP abort at
    elaboration -- before a single transaction runs -- so the reference source
    list handed to a simulator has to be deduped or the gate reports a build
    failure that says nothing about the netlist."""

    def _caravel_layout(self, tmp_path):
        pad = tmp_path / "rtl" / "user_project_wrapper.v"
        pad.parent.mkdir(parents=True)
        pad.write_text("module user_project_wrapper (input wire clk);\n"
                       "endmodule\n")
        top = tmp_path / "rtl" / "integration" / "user_project_wrapper.v"
        top.parent.mkdir(parents=True)
        top.write_text("module user_project_wrapper (input wire clk);\n"
                       "  user_project_wrapper_pads u_pads ();\n"
                       "endmodule\n")
        return top, pad

    def test_duplicate_module_is_removed_when_a_dedup_dir_is_given(self, tmp_path):
        top, pad = self._caravel_layout(tmp_path)
        srcs = chip_rtl_sources(
            str(top), {"user_project_wrapper_io": str(pad)},
            dedup_dir=tmp_path / "scratch")
        decls = 0
        for s in srcs:
            decls += Path(s).read_text().count("module user_project_wrapper ")
        assert decls == 1, f"{decls} definitions survived: {srcs}"

    def test_raw_paths_when_no_dedup_dir_is_given(self, tmp_path):
        """Callers that only want paths (linting, reporting) get them back
        unchanged -- deduping writes files, which a pure query must not do."""
        top, pad = self._caravel_layout(tmp_path)
        srcs = chip_rtl_sources(str(top), {"user_project_wrapper_io": str(pad)})
        assert srcs == [str(top), str(pad)]
