# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Measured timing is required even for large inferred memories."""
import pytest

from orchestrator.langgraph import pipeline_graph as pg
from orchestrator.langgraph import ppa_check as pc


@pytest.mark.parametrize('wns', [4.0, -2.0, None])
def test_mapped_rom_always_reaches_timing(tmp_path, monkeypatch, wns):
    rtl = tmp_path / 'core.v'
    rtl.write_text('module core(input clk); endmodule')
    monkeypatch.setattr(pc, 'probe_synth_generic', lambda *a, **k: {
        'elaborated': True, 'logic_ff': 3, 'mem_bits': 22528})
    monkeypatch.setattr(pc, 'probe_memory_flops', lambda *a, **k: {
        'elaborated': True, 'memories': [(11, 2048)]})
    monkeypatch.setattr(pc, 'synth_cell_gate_enabled', lambda: False)
    monkeypatch.setattr(pc, 'logic_depth_gate_enabled', lambda: False)
    monkeypatch.setattr(pc, 'sta_maxfanout_enabled', lambda: False)
    seen = []
    def sta(*args, **kwargs):
        seen.append(args)
        return {'wns_ns': wns}
    monkeypatch.setattr(pc, 'run_pre_layout_sta', sta)
    ok, reasons, meta = pg._evaluate_ppa_gate(str(tmp_path), 'core', str(rtl), {
        'netlist_path': str(rtl), 'ff_count': 3, 'chip_area_um2': 20},
        require_gate_flag=False)
    assert len(seen) == 1
    assert meta['timing_required'] is True
    assert meta['wns_ns'] == wns
    if wns is None:
        assert meta['timing_unmeasured'] is True
        assert ok is None
    else:
        assert pg._timing_ok_from_ppa_meta(meta) is (wns >= 0)


def test_required_timing_cannot_finish_as_pass(tmp_path):
    import asyncio
    state = {'project_root': str(tmp_path), 'current_block': {'name': 'core'},
             'attempt': 1, 'sim_passed': True, 'synth_success': True,
             'timing_required': True, 'timing_ok': None}
    result = asyncio.run(pg.block_done_node(state))
    assert not result['completed_blocks'][0]['success']


def test_required_timing_parks_even_when_area_was_measured(tmp_path, monkeypatch):
    monkeypatch.setenv('CORESMITH_PROFILE', 'strict')
    assert pg._ppa_should_park_tooling_missing(str(tmp_path), False,
        {'timing_required': True, 'timing_unmeasured': True})


def test_unspecified_ff_limit_does_not_invent_a_memory_requirement():
    verdict = pc.evaluate_ppa(actual_ff=196674, ff_budget=None, wns_ns=5)
    assert verdict.ok and not verdict.reasons
    assert not any(c["metric"] == "flip_flop_hard_ceiling" for c in verdict.checks)
    assert not pc.evaluate_ppa(actual_ff=196674, ff_budget=None,
                               hard_ff_ceiling=50000, wns_ns=5).ok


def test_failed_buffering_is_a_tool_failure_not_an_rtl_verdict(tmp_path, monkeypatch):
    rtl, lib, sdc = (tmp_path / name for name in ["core.v", "cells.lib", "core.sdc"])
    rtl.write_text("module core(input clk); endmodule")
    lib.write_text("fixture")
    sdc.write_text("create_clock -period 15.625 [get_ports clk]")
    monkeypatch.setattr(pc, "sta_maxfanout_enabled", lambda: True)
    monkeypatch.setattr(pc, "run_pre_layout_sta", lambda *a, **k: {"wns_ns": -50000})
    monkeypatch.setattr(pc, "run_maxfanout_buffered_sta", lambda *a, **k: {
        "sta_ok": True, "wns_ns": -50000, "repair_status": "failed",
        "detail": "mapped-netlist repair timed out", "buffered_wns_ns": None})
    ok, reasons, meta = pg._evaluate_ppa_gate(str(tmp_path), "core", str(rtl), {
        "netlist_path": str(rtl), "sdc_path": str(sdc), "liberty_path": str(lib),
        "ff_count": 196674, "chip_area_um2": 7207159}, require_gate_flag=False)
    assert ok is None
    assert meta["timing_unmeasured"] and "timed out" in meta["sta_error"]
    assert not (tmp_path / ".coresmith/blocks/core/previous_error.txt").exists()
