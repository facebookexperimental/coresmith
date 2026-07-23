# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the deterministic LVS benign-pin reconciliation in
``orchestrator/langgraph/backend_helpers.py``.

netgen prints "Top level cell failed pin matching." for EVERY openframe-wrapped
block -- the shared-constant GPIO bus (io_in/io_out/io_oeb) is structural
symmetry it reports as unmatched/"shorted" top-level pins, even for a correct
layout. ``reconcile_lvs_match`` (gate ``CORESMITH_LVS_BENIGN_CLASSIFY``, default
ON) upgrades that raw FAIL to a match ONLY when the report proves the failure is
limited to a benign pin set, and MUST keep failing on a real short / device
delta / device-class inequality.

The fixtures below are SYNTHETIC but mirror the REAL netgen report formats
pulled read-only from E6 signoff artifacts:
  - table format (raster ``..._lvs.rpt``, a design the engine recorded
    ``lvs match:true`` that still prints "failed pin matching"): a
    ``Subcircuit pins:`` table whose unmatched rows are all io_in/io_out/io_oeb,
    closing with "Device classes <top> and <top> are equivalent." +
    "Final result: Top level cell failed pin matching.". Net counts collapse;
    device counts do not.
  - "are shorted in cell" format (fft ``netgen_fft.log`` / jpeg
    ``netgen_jpeg.log``): "Circuit 1 contains N devices, Circuit 2 contains M
    devices. *** MISMATCH ***" + "Pins io_oeb[42] and io_oeb[43] are shorted in
    cell <top> (1)" lines + "Final result:\\nTop level cell failed pin matching."

Hermetic -- no EDA tools / PDK needed.
"""
from __future__ import annotations

import pytest

from orchestrator.langgraph.backend_helpers import (
    classify_netgen_lvs_benign,
    reconcile_lvs_match,
    lvs_benign_classify_enabled,
)

_GATE = "CORESMITH_LVS_BENIGN_CLASSIFY"
TOP = "chip_top"


# ---------------------------------------------------------------------------
# Fixtures (mirror the real netgen report formats)
# ---------------------------------------------------------------------------

# Raster-style .rpt table: unmatched pins are ALL openframe GPIO bus bits
# (io_in/io_out/io_oeb); VGND/VPWR present and matched; top-cell device classes
# equivalent; only net counts collapse. This is the benign case.
_BENIGN_TABLE = """\
Subcircuit summary:
Circuit 1: sky130_fd_sc_hd__decap_3        |Circuit 2: sky130_fd_sc_hd__decap_3
Netlists match uniquely.

Subcircuit pins:
Circuit 1: chip_top                        |Circuit 2: chip_top
-------------------------------------------|-------------------------------------------
io_out[8]                                  |io_out[8]
io_out[7]                                  |io_out[7]
VGND                                       |VGND
VPWR                                       |VPWR
io_in[0]                                   |io_in[0]
(no matching pin)                          |io_out[24]
(no matching pin)                          |io_out[23]
(no matching pin)                          |io_out[0]
io_in[1]                                   |(no matching pin)
io_oeb[1]                                  |(no matching pin)
io_oeb[9]                                  |(no matching pin)
io_out[1]                                  |(no matching pin)
io_out[9]                                  |(no matching pin)
-------------------------------------------|-------------------------------------------
Cell pin lists for chip_top and chip_top altered to match.
Device classes chip_top and chip_top are equivalent.

Final result: Top level cell failed pin matching.
"""

# fft/jpeg-style .log: benign io_oeb/io_out short bus + EQUAL device counts (only
# nets collapse). Benign.
_BENIGN_SHORTED = """\
  Class: sky130_fd_sc_hd__nand2_1 instances: 1745
  Class: sky130_fd_sc_hd__tapvpwrvgnd_1 instances:   1
Circuit contains 26491 nets, and 37 disconnected pins.

Circuit 1 contains 26156 devices, Circuit 2 contains 26156 devices.
Circuit 1 contains 26565 nets,    Circuit 2 contains 26487 nets. *** MISMATCH ***

Pins io_oeb[42] and io_oeb[43] are shorted in cell chip_top (1)
Pins io_oeb[41] and io_oeb[43] are shorted in cell chip_top (1)
Pins io_out[39] and io_oeb[8] are shorted in cell chip_top (1)
Pins io_out[1] and io_out[43] are shorted in cell chip_top (1)
Pins io_out[0] and io_out[43] are shorted in cell chip_top (1)

Cell pin lists for chip_top and chip_top altered to match.
Device classes chip_top and chip_top are equivalent.

Final result:
Top level cell failed pin matching.
"""

# Clean pass.
_CLEAN_MATCH = "Final result: Circuits match uniquely.\n"

# REAL SHORT: an internal signal net shorted in the layout. `net241` and
# `data_out[3]` are NOT in the benign set -> must NOT be masked.
_REAL_INTERNAL_SHORT = """\
Subcircuit pins:
Circuit 1: chip_top                        |Circuit 2: chip_top
-------------------------------------------|-------------------------------------------
io_out[8]                                  |io_out[8]
(no matching pin)                          |io_out[24]
io_oeb[9]                                  |(no matching pin)
-------------------------------------------|-------------------------------------------
Device classes chip_top and chip_top are equivalent.

Pins net241 and data_out[3] are shorted in cell chip_top (1)

Final result: Top level cell failed pin matching.
"""

# DEVICE-COUNT DELTA: benign io_oeb short bus, but a real device delta
# (fft: 26156 vs 26159). Must NOT be masked.
_DEVICE_COUNT_DELTA = """\
Circuit 1 contains 26156 devices, Circuit 2 contains 26159 devices. *** MISMATCH ***
Circuit 1 contains 26565 nets,    Circuit 2 contains 26487 nets. *** MISMATCH ***

Pins io_oeb[42] and io_oeb[43] are shorted in cell chip_top (1)
Pins io_out[0] and io_out[43] are shorted in cell chip_top (1)

Device classes chip_top and chip_top are equivalent.

Final result:
Top level cell failed pin matching.
"""

# DEVICE CLASSES NOT EQUIVALENT: benign io_* pins, but a device class diverged.
# Must NOT be masked.
_DEVCLASS_NOT_EQUIV = """\
Subcircuit pins:
Circuit 1: chip_top                        |Circuit 2: chip_top
-------------------------------------------|-------------------------------------------
(no matching pin)                          |io_out[24]
io_oeb[9]                                  |(no matching pin)
-------------------------------------------|-------------------------------------------
Device classes sky130_fd_sc_hd__foo and sky130_fd_sc_hd__bar are not equivalent.

Final result: Top level cell failed pin matching.
"""

# BENIGN PHYSICAL-ONLY DEVICE DELTA (the fft/jpeg case): stdout form with the
# paired "Contents of circuit 1/2" per-class blocks. Circuit 2 (the reference
# _pwr.v) carries 3 extra zero-transistor filler cells (1 tap + 2 fill) that
# Magic's ext2spice legitimately drops -> a device delta of 3 that is FULLY
# physical-only. Mirrors netgen_fft.log's real class-block structure.
_DEVDELTA_PHYSICAL = """\
Contents of circuit 1:  Circuit: 'chip_top'
  Class: sky130_fd_sc_hd__nand2_1 instances: 1745
  Class: sky130_fd_sc_hd__dfxtp_1 instances: 1065
  Class: sky130_fd_sc_hd__decap_12 instances:   1
Contents of circuit 2:  Circuit: 'chip_top'
  Class: sky130_fd_sc_hd__nand2_1 instances: 1745
  Class: sky130_fd_sc_hd__dfxtp_1 instances: 1065
  Class: sky130_fd_sc_hd__decap_12 instances:   1
  Class: sky130_fd_sc_hd__fill_1 instances:   1
  Class: sky130_fd_sc_hd__fill_2 instances:   1
  Class: sky130_fd_sc_hd__tapvpwrvgnd_1 instances:   1

Circuit 1 contains 2811 devices, Circuit 2 contains 2814 devices. *** MISMATCH ***
Circuit 1 contains 3000 nets,    Circuit 2 contains 2950 nets. *** MISMATCH ***

Pins io_oeb[42] and io_oeb[43] are shorted in cell chip_top (1)
Pins io_out[0] and io_out[43] are shorted in cell chip_top (1)

Final result:
Top level cell failed pin matching.
"""

# REAL LOGIC-CELL DEVICE DELTA: circuit 2 has one extra dfxtp_1 (a sequential
# cell) in addition to a filler. A missing/extra logic cell is a real
# structural divergence and must NOT be masked.
_DEVDELTA_LOGIC = """\
Contents of circuit 1:  Circuit: 'chip_top'
  Class: sky130_fd_sc_hd__dfxtp_1 instances: 1065
Contents of circuit 2:  Circuit: 'chip_top'
  Class: sky130_fd_sc_hd__dfxtp_1 instances: 1066
  Class: sky130_fd_sc_hd__fill_1 instances:   1

Circuit 1 contains 1065 devices, Circuit 2 contains 1067 devices. *** MISMATCH ***

Pins io_oeb[42] and io_oeb[43] are shorted in cell chip_top (1)

Final result:
Top level cell failed pin matching.
"""


# ---------------------------------------------------------------------------
# Pure classifier
# ---------------------------------------------------------------------------

class TestClassifier:
    def test_benign_table_accepted(self):
        v = classify_netgen_lvs_benign(_BENIGN_TABLE, TOP)
        assert v["benign"] is True
        assert v["non_benign_pins"] == []
        assert v["device_classes_equivalent"] and v["device_counts_equal"]
        # 8 unmatched GPIO bus bits reconciled.
        assert len(v["reconciled_pins"]) == 8
        assert set(v["reconciled_pins"]) == {
            "io_out[24]", "io_out[23]", "io_out[0]", "io_in[1]",
            "io_oeb[1]", "io_oeb[9]", "io_out[1]", "io_out[9]",
        }

    def test_benign_shorted_accepted(self):
        v = classify_netgen_lvs_benign(_BENIGN_SHORTED, TOP)
        assert v["benign"] is True
        assert v["non_benign_pins"] == []
        # net-count *** MISMATCH *** on a NET line is NOT disqualifying.
        assert v["device_counts_equal"] is True
        assert len(v["reconciled_pins"]) == 10  # 5 shorted pairs

    def test_clean_match_passthrough(self):
        v = classify_netgen_lvs_benign(_CLEAN_MATCH, TOP)
        assert v["netgen_match"] is True

    def test_reference_declared_tie_bit_is_benign(self):
        # A non-GPIO, non-power output bit that the reference proves is a
        # constant-tie is reconciled via parse_output_tie_classes.
        report = _BENIGN_TABLE.replace("io_out[0]", "dbg_out[3]")
        ref_v = "assign dbg_out[3] = 1'b0;\n"
        v = classify_netgen_lvs_benign(report, TOP, reference_verilog_text=ref_v)
        assert v["benign"] is True
        assert "dbg_out[3]" in v["reconciled_pins"]

    # --- real-mismatch safety: these must NEVER be reconciled ---

    def test_real_internal_short_rejected(self):
        v = classify_netgen_lvs_benign(_REAL_INTERNAL_SHORT, TOP)
        assert v["benign"] is False
        assert "net241" in v["non_benign_pins"]
        assert "data_out[3]" in v["non_benign_pins"]

    def test_device_count_delta_unprovable_rejected(self):
        # Delta present but no per-class blocks to prove it is physical-only ->
        # fail-closed.
        v = classify_netgen_lvs_benign(_DEVICE_COUNT_DELTA, TOP)
        assert v["benign"] is False
        assert v["device_counts_equal"] is False
        assert v["device_delta_benign"] is False

    def test_physical_only_device_delta_accepted(self):
        # THE fft/jpeg CASE: a device delta of 3 that is ENTIRELY tap+fill.
        v = classify_netgen_lvs_benign(_DEVDELTA_PHYSICAL, TOP)
        assert v["benign"] is True
        assert v["device_counts_equal"] is False   # counts are NOT strictly equal
        assert v["device_ok"] is True              # but the delta is benign
        assert v["device_delta"] == 3
        assert v["device_delta_benign"] is True
        assert v["device_delta_nonphysical"] == []
        assert {c for c, _ in v["device_delta_physical"]} == {
            "sky130_fd_sc_hd__fill_1", "sky130_fd_sc_hd__fill_2",
            "sky130_fd_sc_hd__tapvpwrvgnd_1",
        }

    def test_logic_cell_device_delta_rejected(self):
        # A real sequential-cell (dfxtp_1) delta must NOT be masked.
        v = classify_netgen_lvs_benign(_DEVDELTA_LOGIC, TOP)
        assert v["benign"] is False
        assert v["device_delta_benign"] is False
        assert any(c == "sky130_fd_sc_hd__dfxtp_1"
                   for c, _ in v["device_delta_nonphysical"])

    def test_device_classes_not_equivalent_rejected(self):
        v = classify_netgen_lvs_benign(_DEVCLASS_NOT_EQUIV, TOP)
        assert v["benign"] is False
        assert v["device_classes_equivalent"] is False

    def test_no_unmatched_pins_not_reconciled(self):
        # Device classes equivalent but nothing unmatched -> nothing to
        # reconcile -> fail-closed.
        text = "Device classes chip_top and chip_top are equivalent.\n"
        v = classify_netgen_lvs_benign(text, TOP)
        assert v["benign"] is False
        assert v["reconciled_pins"] == []


# ---------------------------------------------------------------------------
# Gate-aware reconciliation (both env branches)
# ---------------------------------------------------------------------------

class TestReconcileGate:
    def test_gate_on_benign_upgrades_to_match(self, monkeypatch):
        monkeypatch.delenv(_GATE, raising=False)  # default ON
        info = reconcile_lvs_match(False, _BENIGN_TABLE, top_cell=TOP)
        assert info["lvs_raw_match"] is False
        assert info["lvs_match"] is True
        assert info["benign_reconciled_pins"] == 8
        assert len(info["benign_reconciled_pin_names"]) == 8

    def test_gate_on_shorted_bus_upgrades_to_match(self, monkeypatch):
        monkeypatch.delenv(_GATE, raising=False)
        info = reconcile_lvs_match(False, _BENIGN_SHORTED, top_cell=TOP)
        assert info["lvs_match"] is True
        assert info["benign_reconciled_pins"] == 10

    def test_gate_on_real_internal_short_stays_fail(self, monkeypatch):
        # THE CRITICAL SAFETY CASE: a real internal-net short is NOT masked.
        monkeypatch.delenv(_GATE, raising=False)
        info = reconcile_lvs_match(False, _REAL_INTERNAL_SHORT, top_cell=TOP)
        assert info["lvs_raw_match"] is False
        assert info["lvs_match"] is False
        assert info["benign_reconciled_pins"] == 0

    def test_gate_on_device_count_delta_stays_fail(self, monkeypatch):
        monkeypatch.delenv(_GATE, raising=False)
        info = reconcile_lvs_match(False, _DEVICE_COUNT_DELTA, top_cell=TOP)
        assert info["lvs_match"] is False

    def test_gate_on_physical_only_delta_upgrades_to_match(self, monkeypatch):
        # fft/jpeg: a 3-cell tap/fill device delta is reconciled to a match.
        monkeypatch.delenv(_GATE, raising=False)
        info = reconcile_lvs_match(False, _DEVDELTA_PHYSICAL, top_cell=TOP)
        assert info["lvs_match"] is True
        assert info["benign_reconciled_pins"] == 4  # 2 shorted pairs

    def test_gate_on_logic_cell_delta_stays_fail(self, monkeypatch):
        monkeypatch.delenv(_GATE, raising=False)
        info = reconcile_lvs_match(False, _DEVDELTA_LOGIC, top_cell=TOP)
        assert info["lvs_match"] is False

    def test_gate_on_devclass_not_equiv_stays_fail(self, monkeypatch):
        monkeypatch.delenv(_GATE, raising=False)
        info = reconcile_lvs_match(False, _DEVCLASS_NOT_EQUIV, top_cell=TOP)
        assert info["lvs_match"] is False

    def test_gate_off_preserves_raw_fail(self, monkeypatch):
        # Legacy behavior: "failed pin matching" -> match stays False, even on
        # the benign fixture. No reconciliation performed.
        monkeypatch.setenv(_GATE, "0")
        info = reconcile_lvs_match(False, _BENIGN_TABLE, top_cell=TOP)
        assert info["lvs_raw_match"] is False
        assert info["lvs_match"] is False
        assert info["benign_reconciled_pins"] == 0

    def test_raw_match_true_is_passthrough(self, monkeypatch):
        # A real netgen "match uniquely" is never touched, gate on or off.
        for val in ("1", "0"):
            monkeypatch.setenv(_GATE, val)
            info = reconcile_lvs_match(True, _CLEAN_MATCH, top_cell=TOP)
            assert info["lvs_match"] is True
            assert info["lvs_raw_match"] is True


class TestGateHelper:
    def test_default_on(self, monkeypatch):
        monkeypatch.delenv(_GATE, raising=False)
        assert lvs_benign_classify_enabled() is True

    def test_explicit_off(self, monkeypatch):
        monkeypatch.setenv(_GATE, "0")
        assert lvs_benign_classify_enabled() is False

    def test_explicit_on(self, monkeypatch):
        monkeypatch.setenv(_GATE, "1")
        assert lvs_benign_classify_enabled() is True
