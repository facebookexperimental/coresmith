# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""A pin map retires the pad-adapter block BEFORE it is ever generated.

Calibrated against the raster run that died on this block twice. Its PRD
declares a structured ``pin_map``; its architecture declares an eighth block
carrying the locked Caravel boundary; its interface contract says that block
translates six signals -- and the pin map routes all six (five mapped signals
plus one output-enable). The generator produced a full chip top instead of a
leaf adapter 6 times out of 6, and the conformance gate correctly refused the
shape and parked the run. The block is the defect, not the RTL: a pin map
replaces it, so the flow must not ask for it.

Everything here is name-free on purpose. The adapter is identified by the
LOCKED BOUNDARY it carries (io_in/io_out/io_oeb) or by an ``rtl_target`` whose
module is the Caravel top -- the same two pieces of evidence
``detect_wrapper_block`` and ``check_block``'s ``locked_boundary`` already use.
"""
from __future__ import annotations

import json

import pytest

from orchestrator.architecture import pin_map_retire as pmr
from orchestrator.architecture.pin_map import parse_pin_map

PIN_MAP = {
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

PAD = "pad_boundary_block"        # deliberately NOT user_project_wrapper*
CORE = "protocol_engine"

BLOCK_QUEUE = [
    {"name": PAD, "tier": 1, "rtl_target": "rtl/user_project_wrapper.v"},
    {"name": CORE, "tier": 2, "rtl_target": "rtl/protocol_engine.v"},
]

BLOCK_DIAGRAM = {
    "blocks": [
        {
            "name": PAD, "tier": 1,
            "rtl_target": "rtl/user_project_wrapper.v",
            "interfaces": {
                "caravel_harness": {"wb_clk_i": 1, "wb_rst_i": 1,
                                    "io_in": 38, "io_out": 38, "io_oeb": 38},
                "qspi_async_pins": {"qspi_csn": 1, "qspi_sck": 1,
                                    "qspi_io_in": 4},
                "qspi_drive": {"qspi_io_out": 4, "qspi_drive_en": 1},
                "irq_level": {"width": 1},
            },
        },
        {"name": CORE, "tier": 2, "rtl_target": "rtl/protocol_engine.v"},
    ],
}

CONTRACTS = {
    "contracts": [
        {"edge_id": "e1", "producer_block": PAD, "producer_port":
            "qspi_async_pins", "consumer_block": CORE,
         "consumer_port": "qspi_async_pins",
         "fields": [{"name": "qspi_csn"}, {"name": "qspi_sck"},
                    {"name": "qspi_io_in"}], "sideband_signals": []},
        {"edge_id": "e2", "producer_block": CORE, "producer_port": "qspi_drive",
         "consumer_block": PAD, "consumer_port": "qspi_drive",
         "fields": [{"name": "qspi_io_out"}, {"name": "qspi_drive_en"}],
         "sideband_signals": []},
        {"edge_id": "e3", "producer_block": CORE, "producer_port": "irq_level",
         "consumer_block": PAD, "consumer_port": "irq_level",
         "fields": [{"name": "irq_level"}], "sideband_signals": []},
    ]
}


def _seed(tmp_path, *, pin_map=PIN_MAP, contracts=CONTRACTS,
          diagram=BLOCK_DIAGRAM):
    cs = tmp_path / ".coresmith"
    cs.mkdir(parents=True, exist_ok=True)
    prd = {"prd": {"summary": "x"}}
    if pin_map is not None:
        prd["prd"]["pin_map"] = pin_map
    (cs / "prd_spec.json").write_text(json.dumps(prd))
    if contracts is not None:
        (cs / "interface_contracts.json").write_text(json.dumps(contracts))
    if diagram is not None:
        (cs / "block_diagram.json").write_text(json.dumps(diagram))
    return tmp_path


@pytest.fixture(autouse=True)
def _default_on(monkeypatch):
    monkeypatch.delenv("CORESMITH_PINMAP_RETIRES_ADAPTER", raising=False)


# --------------------------------------------------------------------------- #
# Identification is structural, never by name
# --------------------------------------------------------------------------- #

class TestTheAdapterIsFoundByEvidence:
    def test_the_locked_boundary_in_the_architecture_entry_identifies_it(
            self, tmp_path):
        _seed(tmp_path)
        # rtl_target stripped, so the ONLY evidence left is the interfaces map.
        queue = [{"name": PAD, "tier": 1}, {"name": CORE, "tier": 2}]
        assert pmr.pad_adapter_blocks(tmp_path, queue) == [PAD]

    def test_an_rtl_target_naming_the_caravel_top_identifies_it(self, tmp_path):
        # No block_diagram at all: rtl_target is the remaining evidence, which
        # is exactly the contract-locked-module-name case.
        _seed(tmp_path, diagram=None)
        assert pmr.pad_adapter_blocks(tmp_path, BLOCK_QUEUE) == [PAD]

    def test_a_design_with_no_pad_boundary_has_no_adapter(self, tmp_path):
        _seed(tmp_path, diagram={"blocks": [{"name": CORE, "tier": 1}]})
        assert pmr.pad_adapter_blocks(
            tmp_path, [{"name": CORE, "tier": 1}]) == []

    def test_the_block_NAME_is_never_the_evidence(self, tmp_path):
        """A block called ``user_project_wrapper_io`` with no locked boundary
        and no Caravel rtl_target is NOT the adapter."""
        _seed(tmp_path, diagram={"blocks": [
            {"name": "user_project_wrapper_io", "tier": 1,
             "interfaces": {"data": {"tdata": 8}}}]})
        assert pmr.pad_adapter_blocks(
            tmp_path,
            [{"name": "user_project_wrapper_io", "tier": 1}]) == []


# --------------------------------------------------------------------------- #
# Coverage decides
# --------------------------------------------------------------------------- #

class TestCoverageDecides:
    def test_full_coverage_retires_the_block(self, tmp_path):
        _seed(tmp_path)
        plan = pmr.plan_retirement(tmp_path, BLOCK_QUEUE)
        assert plan.retire is True and plan.park is False
        assert plan.block == PAD
        assert plan.reason == pmr.RETIRE_REASON
        assert sorted(plan.covered) == [
            "irq_level", "qspi_csn", "qspi_drive_en", "qspi_io_in",
            "qspi_io_out", "qspi_sck"]
        assert plan.uncovered == []

    def test_an_output_enable_counts_as_coverage(self, tmp_path):
        """``qspi_drive_en`` is not a mapped SIGNAL -- it is the ``oe`` of one.
        The top inverts it into io_oeb, so it is routed, so it is covered."""
        _seed(tmp_path)
        pm = parse_pin_map(PIN_MAP)
        assert "qspi_drive_en" in pmr.pin_map_signal_names(pm)

    def test_partial_coverage_PARKS_and_never_retires(self, tmp_path):
        """A boundary where the top routes some pads and a block routes the
        rest is a contradiction -- both would drive the same bus."""
        partial = {"bus_width": 38, "entries": [
            e for e in PIN_MAP["entries"] if e["signal"] != "irq_level"]}
        _seed(tmp_path, pin_map=partial)
        plan = pmr.plan_retirement(tmp_path, BLOCK_QUEUE)
        assert plan.retire is False
        assert plan.park is True
        assert plan.uncovered == ["irq_level"]
        assert "irq_level" in plan.message
        assert "contradiction" in plan.message
        # and the queue is untouched
        assert len(pmr.apply_retirement(BLOCK_QUEUE, plan)) == 2

    def test_zero_coverage_neither_retires_nor_parks(self, tmp_path):
        """This pin map is not about this block; refusing the run would be a
        new failure mode with no evidence behind it."""
        other = {"bus_width": 38, "entries": [
            {"signal": "uart_rx", "dir": "in", "msb": 0, "lsb": 0}]}
        _seed(tmp_path, pin_map=other)
        plan = pmr.plan_retirement(tmp_path, BLOCK_QUEUE)
        assert plan.retire is False and plan.park is False
        assert "covers NONE" in plan.reason

    def test_no_contract_edge_means_no_retirement(self, tmp_path):
        _seed(tmp_path, contracts={"contracts": []})
        plan = pmr.plan_retirement(tmp_path, BLOCK_QUEUE)
        assert plan.retire is False and plan.park is False
        assert "no interface-contract edge" in plan.reason

    def test_an_invalid_pin_map_retires_nothing(self, tmp_path):
        bad = {"bus_width": 38, "entries": [
            {"signal": "a", "dir": "in", "msb": 0, "lsb": 0},
            {"signal": "b", "dir": "in", "msb": 0, "lsb": 0}]}   # bit 0 twice
        _seed(tmp_path, pin_map=bad)
        plan = pmr.plan_retirement(tmp_path, BLOCK_QUEUE)
        assert plan.retire is False and plan.park is False
        assert "INVALID" in plan.reason


# --------------------------------------------------------------------------- #
# Non-pin-map designs are untouched
# --------------------------------------------------------------------------- #

class TestDesignsWithoutAPinMapAreUntouched:
    def test_no_pin_map_no_plan(self, tmp_path):
        _seed(tmp_path, pin_map=None)
        plan = pmr.plan_retirement(tmp_path, BLOCK_QUEUE)
        assert plan.retire is False and plan.park is False
        assert plan.block == ""
        assert pmr.apply_retirement(BLOCK_QUEUE, plan) == BLOCK_QUEUE

    def test_a_bare_project_root_is_not_an_error(self, tmp_path):
        plan = pmr.plan_retirement(tmp_path, BLOCK_QUEUE)
        assert plan.retire is False and plan.park is False


# --------------------------------------------------------------------------- #
# Opt-out
# --------------------------------------------------------------------------- #

class TestTheOptOut:
    def test_env_zero_restores_todays_behaviour(self, tmp_path, monkeypatch):
        _seed(tmp_path)
        monkeypatch.setenv("CORESMITH_PINMAP_RETIRES_ADAPTER", "0")
        assert pmr.retirement_enabled() is False
        from orchestrator.harness.blocks import load_block_queue
        (tmp_path / ".coresmith" / "block_specs.json").write_text(
            json.dumps(BLOCK_QUEUE))
        names = [b["name"] for b in load_block_queue(tmp_path)]
        assert names == [PAD, CORE]

    def test_default_on(self, tmp_path):
        assert pmr.retirement_enabled() is True

    def test_the_harness_queue_agrees_with_the_pipeline(self, tmp_path):
        """Sibling lists, `coresmith verify` and the MCP views must not report
        a block the pipeline is not running."""
        _seed(tmp_path)
        (tmp_path / ".coresmith" / "block_specs.json").write_text(
            json.dumps(BLOCK_QUEUE))
        from orchestrator.harness.blocks import block_names, load_block_queue
        assert [b["name"] for b in load_block_queue(tmp_path)] == [CORE]
        assert block_names(tmp_path) == [CORE]


# --------------------------------------------------------------------------- #
# Artifacts
# --------------------------------------------------------------------------- #

class TestTheReasonSurvivesTheRun:
    def test_a_record_lands_in_the_block_dir_and_the_carried_artifact(
            self, tmp_path):
        _seed(tmp_path)
        plan = pmr.plan_retirement(tmp_path, BLOCK_QUEUE)
        rec = pmr.record_retirement(tmp_path, plan)
        assert rec["block"] == PAD and rec["reason"] == pmr.RETIRE_REASON
        assert rec["skipped"] is True
        note = (tmp_path / ".coresmith" / "blocks" / PAD
                / pmr.BLOCK_NOTE_NAME).read_text()
        assert "RETIRED" in note and "irq_level" in note
        assert "CORESMITH_PINMAP_RETIRES_ADAPTER=0" in note
        rows = pmr.read_retired_blocks(tmp_path)
        assert [r["block"] for r in rows] == [PAD]

    def test_recording_twice_does_not_duplicate(self, tmp_path):
        _seed(tmp_path)
        plan = pmr.plan_retirement(tmp_path, BLOCK_QUEUE)
        pmr.record_retirement(tmp_path, plan)
        pmr.record_retirement(tmp_path, plan)
        assert len(pmr.read_retired_blocks(tmp_path)) == 1

    def test_the_final_report_explains_the_short_block_count(self, tmp_path):
        from orchestrator.langgraph.final_report import (
            build_final_report,
            render_markdown,
        )
        _seed(tmp_path)
        plan = pmr.plan_retirement(tmp_path, BLOCK_QUEUE)
        rec = pmr.record_retirement(tmp_path, plan)
        state = {"block_queue": [{"name": CORE}], "retired_blocks": [rec],
                 "target_clock_mhz": 50.0}
        report = build_final_report(state, str(tmp_path))
        assert report["signoff"]["retired_block_count"] == 1
        assert report["retired_blocks"][0]["block"] == PAD
        md = render_markdown(report)
        assert "Retired blocks" in md and PAD in md


# --------------------------------------------------------------------------- #
# apply_retirement
# --------------------------------------------------------------------------- #

class TestApplyRetirement:
    def test_it_removes_exactly_the_retired_block(self, tmp_path):
        _seed(tmp_path)
        plan = pmr.plan_retirement(tmp_path, BLOCK_QUEUE)
        out = pmr.apply_retirement(BLOCK_QUEUE, plan)
        assert [b["name"] for b in out] == [CORE]

    def test_it_is_idempotent(self, tmp_path):
        _seed(tmp_path)
        plan = pmr.plan_retirement(tmp_path, BLOCK_QUEUE)
        once = pmr.apply_retirement(BLOCK_QUEUE, plan)
        assert pmr.apply_retirement(once, plan) == once
