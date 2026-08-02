# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""RTL ports must match the signals the interface contract declares.

Calibrated against all 8 blocks of exp-raster-validate-20260730: with the rule
implemented here, 3 blocks conform, 1 is exempt (Caravel pad boundary), and 4
deviate -- and every one of those 4 is a genuine deviation that would make a
contract edge unresolvable at integration.

An earlier, stricter version demanded `<channel>_<field>` universally and failed
6 of 8, mostly because the CHECK was wrong: it wanted `edge_event_sck_rise` and
`csn_sync_csn_sync`. Running it against real RTL before wiring it in is what
caught that.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchestrator.langgraph.contract_conformance import check_block, declared_ports


def _project(tmp_path, edges):
    (tmp_path / ".coresmith").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".coresmith" / "interface_contracts.json").write_text(
        json.dumps({"contracts": edges}))
    return tmp_path


def _edge(producer, consumer, chan, fields=(), sideband=()):
    return {
        "edge_id": f"{producer}__{chan}__to__{consumer}__{chan}",
        "producer_block": producer, "producer_port": chan,
        "consumer_block": consumer, "consumer_port": chan,
        "fields": [{"name": f} for f in fields],
        "sideband_signals": list(sideband),
    }


def _rtl(tmp_path, name, ports):
    f = tmp_path / f"{name}.v"
    f.write_text(f"module {name} (\n  "
                 + ",\n  ".join(f"input wire {p}" for p in ports)
                 + "\n);\nendmodule\n")
    return f


class TestBothNamingStylesAreAccepted:
    """Blocks legitimately use either form. The prefixed one disambiguates a
    generic field name; the bare one is used when the name already stands
    alone (`sck_rise`, `qspi_csn`)."""

    @pytest.mark.parametrize("ports", [
        ["framebuffer_read_rdata", "framebuffer_read_read_enable"],   # prefixed
        ["rdata", "read_enable"],                                     # bare
    ])
    def test_conforming_block(self, tmp_path, ports):
        root = _project(tmp_path, [
            _edge("aperture", "fb", "framebuffer_read",
                  fields=["rdata"], sideband=["read_enable"])])
        r = check_block(root, "fb", _rtl(tmp_path, "fb", ports))
        assert r.ok, (r.missing, r.undeclared, r.ambiguous)
        assert r.checked_edges == 1


class TestTheRealDeviations:
    def test_collapsed_duplicate_token_is_caught(self, tmp_path):
        """THE systematic defect: channel `host_write` + signal `write_enable`
        emitted as `host_write_enable`. Neither the prefixed nor the bare form,
        so the contract edge cannot be resolved by name."""
        root = _project(tmp_path, [
            _edge("aperture", "store", "host_write",
                  fields=["wdata"], sideband=["write_enable"])])
        r = check_block(root, "aperture",
                        _rtl(tmp_path, "aperture",
                             ["host_write_wdata", "host_write_enable"]))
        assert not r.ok
        assert ("host_write", "host_write_write_enable") in r.missing
        assert "host_write_enable" in r.undeclared

    def test_direction_suffix_is_caught(self, tmp_path):
        """`raster_read_fault_i` -- real, from raster_scan_pipeline."""
        root = _project(tmp_path, [
            _edge("zb", "pipe", "raster_read", fields=["rdata"],
                  sideband=["fault"])])
        r = check_block(root, "pipe",
                        _rtl(tmp_path, "pipe",
                             ["raster_read_rdata", "raster_read_fault_i"]))
        assert not r.ok
        assert "raster_read_fault_i" in r.undeclared

    def test_missing_signal_is_caught(self, tmp_path):
        root = _project(tmp_path, [
            _edge("a", "b", "ch", fields=["data"], sideband=["valid"])])
        r = check_block(root, "b", _rtl(tmp_path, "b", ["ch_data"]))
        assert ("ch", "ch_valid") in r.missing


class TestAmbiguityIsADeviation:
    """The stitcher cannot resolve these either, which is the whole point."""

    def test_both_forms_present(self, tmp_path):
        root = _project(tmp_path, [_edge("a", "b", "ch", fields=["data"])])
        r = check_block(root, "b", _rtl(tmp_path, "b", ["ch_data", "data"]))
        assert not r.ok and r.ambiguous

    def test_one_bare_name_claimed_by_two_channels(self, tmp_path):
        root = _project(tmp_path, [
            _edge("a", "b", "rd_one", fields=["req_addr"]),
            _edge("a", "b", "rd_two", fields=["req_addr"]),
        ])
        r = check_block(root, "b", _rtl(tmp_path, "b", ["req_addr"]))
        assert not r.ok
        assert any("claimed by both" in w for _c, w in r.ambiguous)


class TestPadBoundaryIsNotABlanketExemption:
    """The mandated pin NAMES are exempt. The block is not.

    A blanket exemption hid the largest defect in the raster design: the pad
    block was generated as a complete chip top that instantiates the other
    blocks internally, with the channel signals as internal wires and NO inward
    ports at all. The architecture specifies a pin ADAPTER with ports; the RTL
    produced a competing top, which nothing can wire.
    """

    def test_mandated_pins_are_never_undeclared(self, tmp_path):
        root = _project(tmp_path, [
            _edge("pads", "core", "ch", fields=["data"])])
        r = check_block(root, "pads",
                        _rtl(tmp_path, "pads",
                             ["io_in", "io_out", "io_oeb", "ch_data"]))
        assert r.ok, (r.missing, r.undeclared)
        assert r.locked_boundary
        assert not r.undeclared, "flagged an externally-mandated pin"

    def test_a_pad_block_still_owes_its_inward_channel_ports(self, tmp_path):
        """THE defect. Carrying io_* does not excuse a block from exposing the
        signals its contract says it produces."""
        root = _project(tmp_path, [
            _edge("pads", "core", "qspi_async_pins",
                  fields=["qspi_csn", "qspi_sck"])])
        r = check_block(root, "pads",
                        _rtl(tmp_path, "pads", ["io_in", "io_out", "io_oeb"]))
        assert not r.ok
        assert r.locked_boundary
        assert ("qspi_async_pins", "qspi_async_pins_qspi_csn") in r.missing


class TestPortsAreReadFromOneModuleOnly:
    def test_stub_modules_later_in_the_file_do_not_count(self, tmp_path):
        """Generated files carry stub declarations of child modules after the
        real one. Unioning their ports gave the pad block a FALSE PASS."""
        f = tmp_path / "pads.v"
        f.write_text(
            "module pads (\n  input wire io_in,\n  output wire io_out,\n"
            "  output wire io_oeb\n);\nendmodule\n\n"
            "module child (\n  input wire ch_data\n);\nendmodule\n")
        root = _project(tmp_path, [_edge("pads", "core", "ch", fields=["data"])])
        r = check_block(root, "pads", f)
        assert not r.ok, "read ports from a stub module lower in the file"
        assert ("ch", "ch_data") in r.missing


class TestFeedbackIsActionable:
    def test_it_states_the_exact_required_name(self, tmp_path):
        """A vague "match the contract" changes nothing -- contract_lookup
        already injects exactly that instruction and the generator collapsed the
        token anyway."""
        root = _project(tmp_path, [
            _edge("aperture", "store", "host_write", sideband=["write_enable"])])
        r = check_block(root, "aperture",
                        _rtl(tmp_path, "aperture", ["host_write_enable"]))
        fb = r.as_feedback()
        assert "host_write_write_enable" in fb
        assert "do not shorten a repeated word" in fb


class TestPortParsing:
    def test_ansi_and_non_ansi_and_comments(self):
        assert {"a", "b", "c"} <= declared_ports(
            "module m(a, b);\n  // input phantom;\n  input wire [3:0] a;\n"
            "  output b;\n  inout c;\nendmodule\n")
        assert "phantom" not in declared_ports("module m(a);\n// input phantom;\n"
                                               "input a;\nendmodule\n")

    def test_missing_file_is_reported_not_raised(self, tmp_path):
        root = _project(tmp_path, [])
        r = check_block(root, "gone", tmp_path / "absent.v")
        assert not r.exempt and r.reason


class TestALeafMustNotAssembleTheDesign:
    """A block that instantiates its siblings is a competing chip top.

    `user_project_wrapper_io` was generated as a 264-line module instantiating
    all seven other blocks and exposing only the Caravel boundary. Nothing can
    wire that -- the assembler has no inward ports to connect, so it refuses,
    the LLM fallback drops blocks, and integration produces no chip_top at all.
    Every downstream gate (flat synth, chip gate-sim) is unreachable behind it.

    The mandated module name is the likely cause: told its module must be named
    `user_project_wrapper`, a generator writes the Caravel wrapper, which by
    convention instantiates the user's design. The name induces the error, so
    the structural property is what to check, not the prompt.
    """

    def _design(self, tmp_path, body):
        f = tmp_path / "pads.v"
        f.write_text("module pads (\n  input wire io_in,\n  output wire io_out,\n"
                     "  output wire io_oeb\n);\n" + body + "\nendmodule\n")
        return f

    def test_instantiating_a_sibling_is_a_deviation(self, tmp_path):
        root = _project(tmp_path, [])
        f = self._design(tmp_path, "  qspi_cdc_frontend u_front ();")
        r = check_block(root, "pads", f, siblings=["pads", "qspi_cdc_frontend"])
        assert not r.ok
        assert r.instantiates == ["qspi_cdc_frontend"]
        assert "It is a leaf, not the chip top" in r.as_feedback()

    def test_a_clean_leaf_passes(self, tmp_path):
        root = _project(tmp_path, [])
        r = check_block(root, "pads",
                        self._design(tmp_path, "  assign io_out = io_in;"),
                        siblings=["pads", "qspi_cdc_frontend"])
        assert r.ok and not r.instantiates

    def test_an_alias_wrapper_after_the_block_is_not_counted(self, tmp_path):
        """A file may legitimately carry a thin alias module after the block --
        the real one does. Only the BLOCK's own module body is examined."""
        f = tmp_path / "pads.v"
        f.write_text(
            "module pads (\n  input wire io_in\n);\n  assign x = io_in;\n"
            "endmodule\n\n"
            "module pads_alias (\n  input wire io_in\n);\n"
            "  pads u (.io_in(io_in));\nendmodule\n")
        root = _project(tmp_path, [])
        r = check_block(root, "pads", f, siblings=["pads", "pads_alias"])
        assert not r.instantiates

    def test_siblings_default_to_empty_so_existing_callers_are_unaffected(
        self, tmp_path
    ):
        root = _project(tmp_path, [])
        r = check_block(root, "pads",
                        self._design(tmp_path, "  qspi_cdc_frontend u ();"))
        assert not r.instantiates


class TestDeterministicPortRepair:
    """The generator will not fix these, measured rather than assumed.

    All four deviating blocks were regenerated with fresh uarch specs and fresh
    RTL, handed the exact required port name and the explicit "do not shorten a
    repeated word" rule -- which `contract_lookup` already carried -- and all
    four came back with the SAME deviations. So the repair belongs in a pass over
    the emitted RTL, which is safe here only because the checker re-verifies the
    result afterwards.
    """

    def test_collapsed_token_is_repaired(self, tmp_path):
        from orchestrator.langgraph.contract_conformance import repair_block_ports
        root = _project(tmp_path, [
            _edge("ap", "store", "host_write",
                  fields=["wdata"], sideband=["write_enable"])])
        f = _rtl(tmp_path, "ap", ["host_write_wdata", "host_write_enable"])
        out = repair_block_ports(root, "ap", f, apply=True)
        assert out["renames"] == {"host_write_enable": "host_write_write_enable"}
        assert out["conforms"] is True
        assert "host_write_write_enable" in f.read_text()

    def test_the_channel_disambiguates_a_shared_signal_name(self, tmp_path):
        """`host_read_enable` and `framebuffer_read_enable` both end in
        `read_enable`. Matching on trailing tokens ties and repairs neither;
        matching within the channel resolves both."""
        from orchestrator.langgraph.contract_conformance import plan_port_repairs
        root = _project(tmp_path, [
            _edge("ap", "s", "host_read", sideband=["read_enable"]),
            _edge("ap", "f", "framebuffer_read", sideband=["read_enable"]),
        ])
        f = _rtl(tmp_path, "ap", ["host_read_enable", "framebuffer_read_enable"])
        plan = plan_port_repairs(check_block(root, "ap", f))
        assert plan == {
            "host_read_enable": "host_read_read_enable",
            "framebuffer_read_enable": "framebuffer_read_read_enable",
        }

    def test_a_different_vocabulary_is_repaired_when_unambiguous(self, tmp_path):
        """`qspi_req_addr` for contract channel `qspi_aperture`'s `req_addr`.
        It wears no channel prefix, so tiers 1 and 2 cannot see it -- but with
        exactly one unaccounted candidate the answer is not a guess."""
        from orchestrator.langgraph.contract_conformance import plan_port_repairs
        root = _project(tmp_path, [
            _edge("eng", "ap", "qspi_aperture", sideband=["req_addr"])])
        f = _rtl(tmp_path, "ap", ["qspi_req_addr"])
        assert plan_port_repairs(check_block(root, "ap", f)) == {
            "qspi_req_addr": "qspi_aperture_req_addr"}

    def test_ports_bound_to_another_channel_are_excluded(self, tmp_path):
        """THE case that makes tier 3 safe, taken from the real block.

        control_status_aperture declares three ports ending in `_req_addr`:
        framebuffer_read_req_addr, host_read_req_addr and qspi_req_addr. Two are
        already the accepted implementation of their own channel's req_addr, so
        only one is a candidate. Without that exclusion this is a coin flip
        between three channels."""
        from orchestrator.langgraph.contract_conformance import plan_port_repairs
        root = _project(tmp_path, [
            _edge("a", "ap", "framebuffer_read", sideband=["req_addr"]),
            _edge("b", "ap", "host_read", sideband=["req_addr"]),
            _edge("c", "ap", "qspi_aperture", sideband=["req_addr"]),
        ])
        f = _rtl(tmp_path, "ap", ["framebuffer_read_req_addr",
                                  "host_read_req_addr", "qspi_req_addr"])
        assert plan_port_repairs(check_block(root, "ap", f)) == {
            "qspi_req_addr": "qspi_aperture_req_addr"}

    def test_two_unaccounted_candidates_are_refused(self, tmp_path):
        """No unique answer means no repair. Renaming the wrong wire
        cross-wires a channel, which RTL cannot show you."""
        from orchestrator.langgraph.contract_conformance import plan_port_repairs
        root = _project(tmp_path, [
            _edge("eng", "ap", "ch", sideband=["req_addr"])])
        f = _rtl(tmp_path, "ap", ["alpha_req_addr", "beta_req_addr"])
        assert plan_port_repairs(check_block(root, "ap", f)) == {}

    def test_nothing_is_applied_without_apply(self, tmp_path):
        from orchestrator.langgraph.contract_conformance import repair_block_ports
        root = _project(tmp_path, [
            _edge("ap", "s", "host_write", sideband=["write_enable"])])
        f = _rtl(tmp_path, "ap", ["host_write_enable"])
        before = f.read_text()
        out = repair_block_ports(root, "ap", f)
        assert out["renames"] and f.read_text() == before

    def test_a_backup_is_left_behind(self, tmp_path):
        from orchestrator.langgraph.contract_conformance import repair_block_ports
        root = _project(tmp_path, [
            _edge("ap", "s", "host_write", sideband=["write_enable"])])
        f = _rtl(tmp_path, "ap", ["host_write_enable"])
        repair_block_ports(root, "ap", f, apply=True)
        assert Path(str(f) + ".pre_portrepair").exists()

    def test_the_result_is_re_checked_not_assumed(self, tmp_path):
        """conforms must come from a fresh check after the edit, so a repair
        that does not actually fix the block cannot report success."""
        from orchestrator.langgraph.contract_conformance import repair_block_ports
        root = _project(tmp_path, [
            _edge("ap", "s", "ch", fields=["data"], sideband=["valid"])])
        f = _rtl(tmp_path, "ap", ["ch_dat"])          # nothing repairable
        out = repair_block_ports(root, "ap", f, apply=True)
        assert out["conforms"] is False


# ---------------------------------------------------------------------------
# The stage, and its wiring into the per-block RTL flow
# ---------------------------------------------------------------------------

class TestTheStage:
    """``run_conformance_stage`` is what the block flow calls: check, repair
    what is provable, re-check, and carry the testbench along."""

    def test_a_deviation_is_repaired_and_the_channel_reported(self, tmp_path):
        from orchestrator.langgraph.contract_conformance import (
            run_conformance_stage,
        )
        root = _project(tmp_path, [
            _edge("ap", "s", "host_write", sideband=["write_enable"])])
        f = _rtl(tmp_path, "ap", ["host_write_enable"])
        rec = run_conformance_stage(root, "ap", f)
        assert rec["ran"] and rec["ok"]
        assert rec["renames"] == {"host_write_enable": "host_write_write_enable"}
        assert rec["rename_channels"]["host_write_enable"] == "host_write"
        assert rec["before_missing"] == 1 and rec["after_missing"] == 0
        assert "host_write_write_enable" in f.read_text()

    def test_no_contract_edge_means_not_run_not_pass(self, tmp_path):
        """A block no edge names has no evidence either way. The stage must
        say so instead of manufacturing a green verdict."""
        from orchestrator.langgraph.contract_conformance import (
            run_conformance_stage,
        )
        root = _project(tmp_path, [_edge("x", "y", "ch", fields=["data"])])
        rec = run_conformance_stage(root, "ap", _rtl(tmp_path, "ap", ["clk"]))
        assert rec["ran"] is False and rec["checked_edges"] == 0
        assert "no interface contract edge" in rec["reason"]

    def test_an_unrepairable_deviation_fails_with_the_exact_name(self, tmp_path):
        from orchestrator.langgraph.contract_conformance import (
            run_conformance_stage,
        )
        root = _project(tmp_path, [
            _edge("ap", "s", "ch", fields=["data"], sideband=["valid"])])
        rec = run_conformance_stage(root, "ap", _rtl(tmp_path, "ap", ["ch_dat"]))
        assert rec["ran"] and rec["ok"] is False
        assert "ch_data" in rec["feedback"] and "ch_valid" in rec["feedback"]
        assert rec["deviations"]

    def test_testbench_dut_references_follow_the_rename(self, tmp_path):
        from orchestrator.langgraph.contract_conformance import (
            run_conformance_stage,
        )
        root = _project(tmp_path, [
            _edge("ap", "s", "host_write", sideband=["write_enable"])])
        f = _rtl(tmp_path, "ap", ["host_write_enable"])
        tb = tmp_path / "test_ap.py"
        tb.write_text("async def t(dut):\n"
                      "    dut.host_write_enable.value = 1\n"
                      "    return int(dut.host_write_enable.value)\n")
        rec = run_conformance_stage(root, "ap", f, tb_path=str(tb))
        assert "dut.host_write_write_enable" in tb.read_text()
        assert "dut.host_write_enable" not in tb.read_text()
        assert rec["tb"]["changed"] and not rec["tb"]["needs_regen"]
        assert Path(str(tb) + ".pre_portrepair").exists()

    def test_a_name_string_reference_asks_for_regeneration(self, tmp_path):
        """Generated testbenches really do drive ``getattr(dut, field)`` over a
        tuple of port-name strings -- and key their MODEL stimulus with the same
        strings. Rewriting those blind would corrupt the model side, so the
        stage asks for a regeneration instead of guessing."""
        from orchestrator.langgraph.contract_conformance import (
            run_conformance_stage,
        )
        root = _project(tmp_path, [
            _edge("ap", "s", "host_write", sideband=["write_enable"])])
        f = _rtl(tmp_path, "ap", ["host_write_enable"])
        tb = tmp_path / "test_ap.py"
        tb.write_text('FIELDS = ("host_write_enable",)\n'
                      'def t(dut):\n'
                      '    for k in FIELDS:\n'
                      '        getattr(dut, k).value = 0\n')
        rec = run_conformance_stage(root, "ap", f, tb_path=str(tb))
        assert rec["tb"]["needs_regen"] is True
        assert rec["tb"]["residual"] == ["host_write_enable"]

    def test_an_unreadable_rtl_is_not_a_verdict(self, tmp_path):
        from orchestrator.langgraph.contract_conformance import (
            run_conformance_stage,
        )
        root = _project(tmp_path, [
            _edge("ap", "s", "ch", fields=["data"])])
        rec = run_conformance_stage(root, "ap", tmp_path / "absent.v")
        assert rec["ran"] is False and rec["ok"] is True and rec["reason"]


class TestWiredIntoTheBlockFlow:
    """The check existed and had tests; nothing CALLED it. These assert the
    PRODUCTION node -- ``generate_testbench_node`` -- runs it."""

    @staticmethod
    def _state(tmp_path, rtl, tb):
        return {
            "current_block": {"name": "ap", "testbench": str(
                Path(tb).relative_to(tmp_path))},
            "project_root": str(tmp_path),
            "attempt": 1,
            "rtl_path": str(rtl),
            "tb_path": str(tb),
            "force_regen_tb": False,
            "preserve_testbench": True,
        }

    @pytest.mark.asyncio
    async def test_the_node_repairs_before_it_simulates(self, tmp_path):
        from unittest.mock import patch

        from orchestrator.langgraph.pipeline_graph import generate_testbench_node
        root = _project(tmp_path, [
            _edge("ap", "s", "host_write", sideband=["write_enable"])])
        rtl = _rtl(tmp_path, "ap", ["host_write_enable"])
        tb = tmp_path / "tb" / "test_ap.py"
        tb.parent.mkdir(parents=True)
        tb.write_text("def t(dut):\n    dut.host_write_enable.value = 1\n")
        seen = {}

        def _sim(block, rtl_path, tb_path, attempt):
            # The sim that runs is the one AFTER the repair -- that is the
            # whole reason the stage sits here and not after the testbench.
            seen["rtl"] = Path(rtl_path).read_text()
            seen["tb"] = Path(tb_path).read_text()
            return {"passed": True, "log": "ok", "returncode": 0,
                    "tests_passed": 1, "tests_total": 1}

        with patch("orchestrator.langgraph.pipeline_graph.run_simulation", _sim):
            out = await generate_testbench_node(
                self._state(tmp_path, rtl, tb))
        assert out["sim_passed"] is True
        assert out["conformance_renames"] == {
            "host_write_enable": "host_write_write_enable"}
        assert "host_write_write_enable" in seen["rtl"]
        assert "dut.host_write_write_enable" in seen["tb"]
        rec = json.loads((root / ".coresmith" / "blocks" / "ap"
                          / "contract_conformance.json").read_text())
        assert rec["renames"] and rec["ok"]

    @pytest.mark.asyncio
    async def test_a_deviating_block_never_reaches_sim(self, tmp_path):
        from unittest.mock import patch

        from orchestrator.langgraph.pipeline_graph import generate_testbench_node
        root = _project(tmp_path, [
            _edge("ap", "s", "ch", fields=["data"], sideband=["valid"])])
        rtl = _rtl(tmp_path, "ap", ["ch_dat"])
        tb = tmp_path / "tb" / "test_ap.py"
        tb.parent.mkdir(parents=True)
        tb.write_text("# tb\n")
        calls = []

        with patch("orchestrator.langgraph.pipeline_graph.run_simulation",
                   lambda *a, **k: calls.append(a) or {"passed": True}):
            out = await generate_testbench_node(
                self._state(tmp_path, rtl, tb))
        assert out["sim_passed"] is False and not calls
        prev = (root / ".coresmith" / "blocks" / "ap"
                / "previous_error.txt").read_text()
        assert "CONTRACT-CONFORMANCE" in prev and "ch_data" in prev

    @pytest.mark.asyncio
    async def test_the_env_gate_turns_it_off(self, tmp_path, monkeypatch):
        from unittest.mock import patch

        from orchestrator.langgraph.pipeline_graph import generate_testbench_node
        monkeypatch.setenv("CORESMITH_CONTRACT_CONFORMANCE_GATE", "0")
        root = _project(tmp_path, [
            _edge("ap", "s", "host_write", sideband=["write_enable"])])
        rtl = _rtl(tmp_path, "ap", ["host_write_enable"])
        tb = tmp_path / "tb" / "test_ap.py"
        tb.parent.mkdir(parents=True)
        tb.write_text("# tb\n")
        with patch("orchestrator.langgraph.pipeline_graph.run_simulation",
                   lambda *a, **k: {"passed": True, "log": "", "returncode": 0}):
            out = await generate_testbench_node(
                self._state(tmp_path, rtl, tb))
        assert out["sim_passed"] is True
        assert "host_write_enable" in rtl.read_text()
        assert "host_write_write_enable" not in rtl.read_text()
        assert not (root / ".coresmith" / "blocks" / "ap"
                    / "contract_conformance.json").exists()

    @pytest.mark.asyncio
    async def test_the_second_failure_parks_instead_of_looping(self, tmp_path):
        from unittest.mock import patch

        from orchestrator.langgraph.pipeline_graph import generate_testbench_node
        root = _project(tmp_path, [
            _edge("ap", "s", "ch", fields=["data"], sideband=["valid"])])
        rtl = _rtl(tmp_path, "ap", ["ch_dat"])
        tb = tmp_path / "tb" / "test_ap.py"
        tb.parent.mkdir(parents=True)
        tb.write_text("# tb\n")
        parked = []

        with patch("orchestrator.langgraph.pipeline_graph.run_simulation",
                   lambda *a, **k: {"passed": True}), \
             patch("orchestrator.langgraph.pipeline_graph.interrupt",
                   lambda payload: parked.append(payload) or {}):
            await generate_testbench_node(self._state(tmp_path, rtl, tb))
            assert not parked          # first failure: retry, do not park
            await generate_testbench_node(self._state(tmp_path, rtl, tb))

        assert len(parked) == 1
        assert parked[0]["type"] == "contract_conformance_unrepairable"
        assert "ch_data" in parked[0]["expected_ports"]
        # the counter resets after a park, so a later genuine change gets a
        # fresh pair of tries rather than parking on every entry
        assert not (root / ".coresmith" / "blocks" / "ap"
                    / "_conformance_failures.txt").exists()


# ---------------------------------------------------------------------------
# Slash-bearing contract text -> a legal identifier, never a `/` in a port name
# ---------------------------------------------------------------------------
#
# The schema template the interface-definition agent is handed literally offers
# the slash form:
#
#     "producer_port": "<m_axis_<name> or m_<name>_srdy/m_<name>_data>"
#
# so real contracts carry it. Every fixture below is the SHAPE of a real edge
# from a 12-block codec whose contract used all three enumeration styles plus a
# two-name sideband alias; only the design-specific words are renamed. The
# derivation concatenated these verbatim, produced port names containing `/`
# (illegal unescaped Verilog), rewrote six blocks' RTL to them, and could never
# converge -- and it double-prefixed a signal that already carried the channel
# (`s_chan_s_chan_addr`).

def _edge2(producer, consumer, pport, cport, fields=(), sideband=()):
    """An edge whose two ends spell the channel differently (the real case)."""
    return {
        "edge_id": f"{producer}__to__{consumer}",
        "producer_block": producer, "producer_port": pport,
        "consumer_block": consumer, "consumer_port": cport,
        "fields": [{"name": f} for f in fields],
        "sideband_signals": [{"name": s} for s in sideband],
    }


#: The addressed-read edge: first segment fully qualified, the rest bare.
_READ_EDGE = _edge2(
    "transform_engine", "sample_memory",
    "m_store_read_req/addr", "s_store_read_req/addr",
    fields=["sample_u8"], sideband=["req", "address", "rvalid", "fault"])

#: The elastic payload edge: both segments fully qualified.
_ELASTIC_EDGE = _edge2(
    "transform_engine", "quantizer",
    "m_result_out_srdy/m_result_out_data",
    "s_result_out_drdy/s_result_out_data",
    fields=["idx", "value_s16"], sideband=["srdy", "drdy"])

#: The control edge: qualified head, bare tail.
_CONTROL_EDGE = _edge2(
    "control_wrapper", "transform_engine",
    "m_job_control_valid/data", "s_job_control_valid/data",
    fields=["job_index", "job_start"], sideband=["valid"])

#: The chip-boundary stream: a SHORT common prefix, plus two sidebands each
#: declared under TWO names (generic role / mandated spelling) and one that
#: already carries the producer's channel prefix.
_STREAM_EDGE = _edge2(
    "control_wrapper", "ingest_addressor",
    "in_valid/in_data/in_last",
    "s_source_srdy/s_source_data/s_source_last",
    fields=["required_zero", "payload_byte"],
    sideband=["srdy/in_valid", "drdy/in_ready", "in_last"])

_SLASH_EDGES = [_READ_EDGE, _ELASTIC_EDGE, _CONTROL_EDGE, _STREAM_EDGE]


class TestChannelBaseParsesTheEnumeration:
    @pytest.mark.parametrize("raw,expect", [
        ("m_store_read_req/addr", "m_store_read"),
        ("s_store_read_req/addr", "s_store_read"),
        ("m_result_out_srdy/m_result_out_data", "m_result_out"),
        ("s_result_out_drdy/s_result_out_data", "s_result_out"),
        ("m_job_control_valid/data", "m_job_control"),
        ("in_valid/in_data/in_last", "in"),
        ("s_source_srdy/s_source_data/s_source_last", "s_source"),
        ("m_axis_frame", "m_axis_frame"),          # the bare form is untouched
        ("", ""),
    ])
    def test_the_channel_is_the_common_token_prefix(self, raw, expect):
        from orchestrator.langgraph.contract_conformance import channel_base
        assert channel_base(raw) == expect


class TestNoDerivedPortEverContainsASlash:
    """The blocking defect: `s_store_read_req/addr_rvalid` is not an
    identifier, so the repair pass rewrote RTL to a name no tool can parse and
    the loop could not converge."""

    def test_every_row_of_every_real_shape_is_a_legal_identifier(self, tmp_path):
        from orchestrator.langgraph.contract_conformance import (
            contract_port_rows,
            is_legal_identifier,
        )
        root = _project(tmp_path, _SLASH_EDGES)
        blocks = {b for e in _SLASH_EDGES
                  for b in (e["producer_block"], e["consumer_block"])}
        seen = 0
        for block in blocks:
            for r in contract_port_rows(str(root), block):
                seen += 1
                assert "/" not in r["port"], r
                assert is_legal_identifier(r["port"]), r
                assert "/" not in r["channel"], r
        assert seen >= 20, "the fixture must actually exercise the derivation"

    def test_the_gate_never_demands_a_name_with_a_slash(self, tmp_path):
        root = _project(tmp_path, _SLASH_EDGES)
        r = check_block(root, "sample_memory",
                        _rtl(tmp_path, "sample_memory", ["clk"]))
        assert r.missing
        for _chan, port in r.missing:
            assert "/" not in port, port

    def test_the_expected_names_are_the_contract_s_own_port_list(self, tmp_path):
        """`s_store_read_req/addr` enumerates two of the channel's real ports,
        so the derived set must contain exactly those spellings."""
        root = _project(tmp_path, [_READ_EDGE])
        want = {p for _c, p in
                check_block(root, "sample_memory",
                            _rtl(tmp_path, "sample_memory", ["clk"])).missing}
        assert want == {
            "s_store_read_sample_u8", "s_store_read_req",
            "s_store_read_address", "s_store_read_rvalid",
            "s_store_read_fault",
        }

    def test_rtl_spelled_the_contract_s_way_conforms(self, tmp_path):
        root = _project(tmp_path, [_ELASTIC_EDGE])
        ports = ["m_result_out_idx", "m_result_out_value_s16",
                 "m_result_out_srdy", "m_result_out_drdy"]
        r = check_block(root, "transform_engine",
                        _rtl(tmp_path, "transform_engine", ports))
        assert r.ok, (r.missing, r.undeclared, r.ambiguous)


class TestTheChannelPrefixIsIdempotent:
    """`s_chan_s_chan_addr_field`: a signal that already carries the channel
    was prefixed a second time. The port_naming skill's rule is
    `<channel>_<field>`, and applying it twice must be the same as once."""

    def test_a_signal_already_carrying_the_channel_is_not_prefixed_again(self):
        from orchestrator.langgraph.contract_conformance import canonical_port
        assert canonical_port("in", "in_last") == ("in_last", "in_last")
        assert canonical_port("s_read_req", "s_read_req_addr") == (
            "s_read_req_addr", "s_read_req_addr")

    def test_a_doubled_TOKEN_is_still_required(self):
        """Different thing, and the skill says it is CORRECT: `write_enable`
        does not start with `data_write_`, so it is still prefixed."""
        from orchestrator.langgraph.contract_conformance import canonical_port
        assert canonical_port("data_write", "write_enable")[0] == \
            "data_write_write_enable"

    def test_no_port_repeats_its_whole_channel(self, tmp_path):
        from orchestrator.langgraph.contract_conformance import contract_port_rows
        root = _project(tmp_path, _SLASH_EDGES)
        for block in ("control_wrapper", "ingest_addressor"):
            for r in contract_port_rows(str(root), block):
                assert not r["port"].startswith(r["channel"] + "_"
                                                + r["channel"] + "_"), r

    def test_the_prefixed_form_is_the_only_form_then(self, tmp_path):
        """`in_last` on channel `in` has ONE acceptable spelling, so a block
        exposing exactly it is conforming -- not 'ambiguous with itself'."""
        root = _project(tmp_path, [_STREAM_EDGE])
        ports = ["in_required_zero", "in_payload_byte", "in_valid",
                 "in_ready", "in_last"]
        r = check_block(root, "control_wrapper",
                        _rtl(tmp_path, "control_wrapper", ports))
        assert r.ok, (r.missing, r.undeclared, r.ambiguous)


class TestATwoNameSidebandResolvesToOneName:
    """`srdy/in_valid` declares ONE wire under the generic handshake role and
    the spelling this edge mandates. Only one of them can be a port."""

    def test_the_end_that_already_spells_it_keeps_its_spelling(self):
        from orchestrator.langgraph.contract_conformance import canonical_port
        # producer channel `in`: the contract's own port list says `in_valid`
        assert canonical_port("in", "srdy/in_valid")[0] == "in_valid"
        assert canonical_port("in", "drdy/in_ready")[0] == "in_ready"

    def test_the_other_end_takes_the_generic_role(self):
        from orchestrator.langgraph.contract_conformance import canonical_port
        # consumer channel `s_source`: its port list says `s_source_srdy`
        assert canonical_port("s_source", "srdy/in_valid")[0] == "s_source_srdy"
        assert canonical_port("s_source", "drdy/in_ready")[0] == "s_source_drdy"

    def test_both_ends_of_the_real_edge_derive_their_own_spelling(self, tmp_path):
        from orchestrator.langgraph.contract_conformance import contract_port_rows
        root = _project(tmp_path, [_STREAM_EDGE])
        prod = {r["port"] for r in contract_port_rows(str(root), "control_wrapper")}
        cons = {r["port"] for r in contract_port_rows(str(root), "ingest_addressor")}
        assert {"in_valid", "in_ready", "in_last"} <= prod
        assert {"s_source_srdy", "s_source_drdy"} <= cons
        assert not any("/" in p for p in prod | cons)


class TestAnUnreducibleNameIsDroppedLoudly:
    """A contract that cannot be reduced to an identifier is a CONTRACT defect.
    Emitting the name anyway rewrites the RTL to garbage; the only safe action
    is to drop the signal and say so."""

    def test_an_illegal_channel_drops_the_edge_and_logs(self, tmp_path, caplog):
        import logging as _logging

        from orchestrator.langgraph.contract_conformance import (
            _ILLEGAL_REPORTED,
            contract_port_rows,
        )
        _ILLEGAL_REPORTED.clear()
        root = _project(tmp_path, [
            _edge2("a", "b", "m bad-chan!", "m bad-chan!", fields=["data"])])
        with caplog.at_level(_logging.ERROR):
            rows = contract_port_rows(str(root), "b")
        assert rows == []
        assert "not a legal Verilog identifier" in caplog.text

    def test_an_illegal_signal_drops_only_that_signal(self, tmp_path, caplog):
        import logging as _logging

        from orchestrator.langgraph.contract_conformance import (
            _ILLEGAL_REPORTED,
            contract_port_rows,
        )
        _ILLEGAL_REPORTED.clear()
        root = _project(tmp_path, [
            _edge2("a", "b", "m_ch", "s_ch", fields=["data", "addr[3:0]"])])
        with caplog.at_level(_logging.ERROR):
            ports = {r["port"] for r in contract_port_rows(str(root), "b")}
        assert ports == {"s_ch_data"}
        assert "addr[3:0]" in caplog.text

    def test_a_repair_to_an_illegal_name_is_refused(self):
        from orchestrator.langgraph.contract_conformance import (
            ConformanceResult,
            plan_port_repairs,
        )
        res = ConformanceResult(block="b")
        res.missing = [("s_ch", "s_ch_req/addr")]
        res.undeclared = ["s_ch_reqaddr"]
        assert plan_port_repairs(res) == {}
