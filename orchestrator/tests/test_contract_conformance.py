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
