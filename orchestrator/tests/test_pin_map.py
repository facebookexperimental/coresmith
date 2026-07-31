# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The pin assignment is data, and every pad bit gets exactly one driver.

Calibrated against the raster PRD, whose GPIO requirement reads:

    "GPIO mapping is locked: io[0]=qspi_csn input, io[1]=qspi_sck input,
     io[5:2]=bidirectional qspi_io[3:0], and io[6]=irq output. io_oeb[0]=
     io_oeb[1]=1, io_oeb[6]=0, and io_oeb[5:2]=0 only during QSPI read-data
     drive phases; otherwise io_oeb[5:2]=1. All other io_oeb bits are 1 and all
     unused io_out bits are 0."

Every clause of that sentence is asserted below against generated Verilog.
"""
from __future__ import annotations

import json

from orchestrator.architecture.pin_map import (
    emit_pin_routing,
    load_pin_map,
    parse_pin_map,
)

RASTER = {
    "bus_width": 38,
    "entries": [
        {"signal": "qspi_csn", "dir": "in", "msb": 0, "lsb": 0},
        {"signal": "qspi_sck", "dir": "in", "msb": 1, "lsb": 1},
        {"signal": "qspi_io_in", "dir": "in", "msb": 5, "lsb": 2},
        {"signal": "qspi_io_out", "dir": "out", "msb": 5, "lsb": 2,
         "oe": "qspi_drive_en"},
        {"signal": "irq_level", "dir": "out", "msb": 6, "lsb": 6},
    ],
}


def _emit(raw=RASTER):
    pm = parse_pin_map(raw)
    assert pm.ok, pm.errors
    decls, assigns = emit_pin_routing(pm)
    return "\n".join(decls), "\n".join(assigns)


class TestTheRasterMappingIsReproducedExactly:
    def test_inputs_are_sliced_off_io_in(self):
        d, _ = _emit()
        assert "wire qspi_csn = io_in[0];" in d
        assert "wire qspi_sck = io_in[1];" in d
        assert "wire [3:0] qspi_io_in = io_in[5:2];" in d

    def test_outputs_drive_io_out(self):
        _, a = _emit()
        assert "assign io_out[5:2] = qspi_io_out;" in a
        assert "assign io_out[6] = irq_level;" in a

    def test_output_enable_is_inverted_because_oeb_is_active_low(self):
        """The single most reversible mistake on this boundary: drive the pad
        exactly when the design wanted to be silent."""
        _, a = _emit()
        assert "assign io_oeb[5:2] = {4{~qspi_drive_en}};" in a

    def test_an_output_with_no_enable_drives_permanently(self):
        """PRD: io_oeb[6]=0."""
        _, a = _emit()
        assert "assign io_oeb[6] = 1'b0;" in a

    def test_unused_out_bits_are_zero_and_unused_oeb_bits_are_one(self):
        """PRD: 'All other io_oeb bits are 1 and all unused io_out bits are 0.'
        Includes io_oeb[0]=io_oeb[1]=1, which the host owns."""
        _, a = _emit()
        assert "assign io_oeb[1:0] = 2'b11;" in a
        assert "assign io_out[1:0] = 2'b00;" in a
        assert "assign io_out[37:7] = 31'b" + "0" * 31 + ";" in a
        assert "assign io_oeb[37:7] = 31'b" + "1" * 31 + ";" in a

    def test_every_bit_of_every_output_bus_has_exactly_one_driver(self):
        """A floating pad bit is a real defect; a doubly-driven one is a short.
        Counting drivers is cheaper than trusting the emitter."""
        import re
        _, a = _emit()
        for bus in ("io_out", "io_oeb"):
            covered = []
            for m in re.finditer(rf"assign {bus}\[(\d+)(?::(\d+))?\]", a):
                hi = int(m.group(1))
                lo = int(m.group(2)) if m.group(2) else hi
                covered.extend(range(lo, hi + 1))
            assert sorted(covered) == list(range(38)), f"{bus} driver coverage"
            assert len(covered) == len(set(covered)), f"{bus} double-driven"


class TestRefusalsRatherThanGuesses:
    def test_two_signals_on_one_bit_is_refused(self):
        pm = parse_pin_map({"bus_width": 38, "entries": [
            {"signal": "a", "dir": "in", "msb": 3, "lsb": 0},
            {"signal": "b", "dir": "in", "msb": 2, "lsb": 2}]})
        assert not pm.ok
        assert any("claimed by both" in e for e in pm.errors)

    def test_input_and_output_may_share_a_bit_index(self):
        """io_in[5:2] and io_out[5:2] are different physical directions of the
        same bidirectional pad -- that is the normal case, not a collision."""
        pm = parse_pin_map({"bus_width": 38, "entries": [
            {"signal": "d_in", "dir": "in", "msb": 5, "lsb": 2},
            {"signal": "d_out", "dir": "out", "msb": 5, "lsb": 2}]})
        assert pm.ok, pm.errors

    def test_out_of_range_bits_are_refused(self):
        pm = parse_pin_map({"bus_width": 8, "entries": [
            {"signal": "a", "dir": "in", "msb": 9, "lsb": 9}]})
        assert not pm.ok and any("outside" in e for e in pm.errors)

    def test_a_duplicated_signal_name_is_refused(self):
        pm = parse_pin_map({"bus_width": 38, "entries": [
            {"signal": "a", "dir": "in", "msb": 0, "lsb": 0},
            {"signal": "a", "dir": "out", "msb": 7, "lsb": 7}]})
        assert not pm.ok and any("mapped 2 times" in e for e in pm.errors)

    def test_an_enable_on_an_input_is_refused(self):
        pm = parse_pin_map({"bus_width": 38, "entries": [
            {"signal": "a", "dir": "in", "msb": 0, "lsb": 0, "oe": "en"}]})
        assert not pm.ok and any("only meaningful" in e for e in pm.errors)

    def test_a_bad_direction_is_refused(self):
        pm = parse_pin_map({"bus_width": 38, "entries": [
            {"signal": "a", "dir": "inout", "msb": 0, "lsb": 0}]})
        assert not pm.ok and any("dir must be" in e for e in pm.errors)


class TestLoading:
    def test_absent_pin_map_is_none_not_an_error(self, tmp_path):
        """A design that is not on a fixed pinout has no pin map, and that is
        not a failure -- the caller falls back to its previous behaviour."""
        (tmp_path / ".coresmith").mkdir()
        (tmp_path / ".coresmith" / "prd_spec.json").write_text(
            json.dumps({"prd": {"functional_requirements": []}}))
        assert load_pin_map(tmp_path) is None

    def test_missing_file_is_none(self, tmp_path):
        assert load_pin_map(tmp_path) is None

    def test_structured_map_is_loaded(self, tmp_path):
        (tmp_path / ".coresmith").mkdir()
        (tmp_path / ".coresmith" / "prd_spec.json").write_text(
            json.dumps({"prd": {"pin_map": RASTER}}))
        pm = load_pin_map(tmp_path)
        assert pm and pm.ok and len(pm.entries) == 5

    def test_prose_is_NOT_parsed(self, tmp_path):
        """Reading the sentence would put a guess back on the production path,
        which is the thing this replaces. Prose alone means no pin map."""
        (tmp_path / ".coresmith").mkdir()
        (tmp_path / ".coresmith" / "prd_spec.json").write_text(json.dumps({"prd": {
            "functional_requirements": [
                "GPIO mapping is locked: io[0]=qspi_csn input, "
                "io[1]=qspi_sck input, io[5:2]=bidirectional qspi_io[3:0]."]}}))
        assert load_pin_map(tmp_path) is None
