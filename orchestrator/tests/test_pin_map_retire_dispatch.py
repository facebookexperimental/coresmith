# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The retirement happens on the PRODUCTION dispatch path, not in a helper.

``init_tier_node`` is the graph's START node and every re-entry point, and
``fan_out_tier`` is the conditional edge that leaves it -- so a block dropped
from the queue there is never ``Send()``-ed, never microarchitected, never
generated and never gated. These tests drive those two functions, plus the two
completeness gates that size ``expected`` off ``block_queue``, so the whole
chain is exercised the way the daemon runs it.

Seeded from the raster run that parked twice on this block.
"""
from __future__ import annotations

import json

import pytest

from orchestrator.langgraph import pipeline_graph
from orchestrator.tests.test_pin_map_retire import (
    BLOCK_DIAGRAM,
    BLOCK_QUEUE,
    CONTRACTS,
    CORE,
    PAD,
    PIN_MAP,
)


def _seed(tmp_path, pin_map=PIN_MAP):
    cs = tmp_path / ".coresmith"
    cs.mkdir(parents=True, exist_ok=True)
    prd = {"prd": {"summary": "x"}}
    if pin_map is not None:
        prd["prd"]["pin_map"] = pin_map
    (cs / "prd_spec.json").write_text(json.dumps(prd))
    (cs / "interface_contracts.json").write_text(json.dumps(CONTRACTS))
    (cs / "block_diagram.json").write_text(json.dumps(BLOCK_DIAGRAM))
    return tmp_path


def _state(tmp_path, queue=None):
    return {
        "project_root": str(tmp_path),
        "target_clock_mhz": 50.0,
        "max_attempts": 3,
        "block_queue": list(queue if queue is not None else BLOCK_QUEUE),
        "tier_list": [],
        "current_tier_index": 0,
        "completed_blocks": [],
    }


@pytest.fixture(autouse=True)
def _default_on(monkeypatch):
    monkeypatch.delenv("CORESMITH_PINMAP_RETIRES_ADAPTER", raising=False)


class TestInitTierRetiresBeforeAnyFanOut:
    @pytest.mark.asyncio
    async def test_the_queue_shrinks_and_the_reason_is_published(self, tmp_path):
        _seed(tmp_path)
        out = await pipeline_graph.init_tier_node(_state(tmp_path))
        assert [b["name"] for b in out["block_queue"]] == [CORE]
        assert out["retired_blocks"][0]["block"] == PAD
        assert out["retired_blocks"][0]["reason"] == "retired_by_pin_map"
        # tier_list is computed from the REDUCED queue, so the adapter's tier
        # disappears with it rather than fanning out an empty tier.
        assert out["tier_list"] == [2]

    @pytest.mark.asyncio
    async def test_fan_out_never_sends_the_retired_block(self, tmp_path):
        _seed(tmp_path)
        st = _state(tmp_path)
        out = await pipeline_graph.init_tier_node(st)
        st.update(out)                       # what LangGraph merges before the edge
        sends = pipeline_graph.fan_out_tier(st)
        assert [s.arg["current_block"]["name"] for s in sends] == [CORE]

    @pytest.mark.asyncio
    async def test_a_design_without_a_pin_map_is_untouched(self, tmp_path):
        _seed(tmp_path, pin_map=None)
        out = await pipeline_graph.init_tier_node(_state(tmp_path))
        assert "block_queue" not in out      # nothing rewritten
        assert "retired_blocks" not in out
        assert out["tier_list"] == [1, 2]

    @pytest.mark.asyncio
    async def test_the_opt_out_restores_todays_behaviour(
            self, tmp_path, monkeypatch):
        _seed(tmp_path)
        monkeypatch.setenv("CORESMITH_PINMAP_RETIRES_ADAPTER", "0")
        out = await pipeline_graph.init_tier_node(_state(tmp_path))
        assert "block_queue" not in out
        assert "retired_blocks" not in out
        assert out["tier_list"] == [1, 2]

    @pytest.mark.asyncio
    async def test_re_entry_is_idempotent(self, tmp_path):
        """Every tier advance re-enters init_tier. The second pass must not
        re-log, re-record or re-shrink anything."""
        _seed(tmp_path)
        st = _state(tmp_path)
        st.update(await pipeline_graph.init_tier_node(st))
        again = await pipeline_graph.init_tier_node(st)
        assert [b["name"] for b in again.get("block_queue", st["block_queue"])] \
            == [CORE]
        from orchestrator.architecture import pin_map_retire as pmr
        assert len(pmr.read_retired_blocks(tmp_path)) == 1


class TestPartialCoverageParks:
    @pytest.mark.asyncio
    async def test_it_raises_an_interrupt_with_the_uncovered_signals(
            self, tmp_path, monkeypatch):
        partial = {"bus_width": 38, "entries": [
            e for e in PIN_MAP["entries"] if e["signal"] != "irq_level"]}
        _seed(tmp_path, pin_map=partial)
        seen = {}

        def _fake_interrupt(payload):
            seen.update(payload)
            return {"action": "keep_block"}

        monkeypatch.setattr(pipeline_graph, "interrupt", _fake_interrupt)
        out = await pipeline_graph.init_tier_node(_state(tmp_path))
        assert seen["type"] == "pin_map_partial_coverage"
        assert seen["block"] == PAD
        assert seen["uncovered_signals"] == ["irq_level"]
        assert "retry" in seen["supported_actions"]
        # keep_block => the queue is NOT reduced (today's behaviour).
        assert "block_queue" not in out

    @pytest.mark.asyncio
    async def test_override_retires_anyway_and_says_so(
            self, tmp_path, monkeypatch):
        partial = {"bus_width": 38, "entries": [
            e for e in PIN_MAP["entries"] if e["signal"] != "irq_level"]}
        _seed(tmp_path, pin_map=partial)
        monkeypatch.setattr(pipeline_graph, "interrupt",
                            lambda _p: {"action": "override"})
        out = await pipeline_graph.init_tier_node(_state(tmp_path))
        assert [b["name"] for b in out["block_queue"]] == [CORE]
        assert "override" in out["retired_blocks"][0]["explanation"]

    @pytest.mark.asyncio
    async def test_retry_replans_and_retires_once_the_map_is_fixed(
            self, tmp_path, monkeypatch):
        partial = {"bus_width": 38, "entries": [
            e for e in PIN_MAP["entries"] if e["signal"] != "irq_level"]}
        _seed(tmp_path, pin_map=partial)

        def _fix_then_retry(_payload):
            # the operator amends prd.pin_map, then resumes `retry`
            _seed(tmp_path, pin_map=PIN_MAP)
            return {"action": "retry"}

        monkeypatch.setattr(pipeline_graph, "interrupt", _fix_then_retry)
        out = await pipeline_graph.init_tier_node(_state(tmp_path))
        assert [b["name"] for b in out["block_queue"]] == [CORE]

    @pytest.mark.asyncio
    async def test_an_unresolving_driver_is_bounded_not_infinite(
            self, tmp_path, monkeypatch):
        """In production each re-park SUSPENDS the graph, so this bounds an
        outer agent that keeps sending `retry` without fixing anything (and
        stops a plain-return interrupt double from spinning)."""
        partial = {"bus_width": 38, "entries": [
            e for e in PIN_MAP["entries"] if e["signal"] != "irq_level"]}
        _seed(tmp_path, pin_map=partial)
        monkeypatch.setattr(pipeline_graph, "interrupt",
                            lambda _p: {"action": "retry"})
        with pytest.raises(RuntimeError, match="partially covers"):
            await pipeline_graph.init_tier_node(_state(tmp_path))


class TestTheCompletenessGatesAgree:
    """`expected` on both gates is sized off ``block_queue``, so a retired
    block is not missing -- it is not expected."""

    @pytest.mark.asyncio
    async def test_pipeline_complete_passes_with_the_reduced_queue(
            self, tmp_path, monkeypatch):
        _seed(tmp_path)
        st = _state(tmp_path)
        st.update(await pipeline_graph.init_tier_node(st))
        st["completed_blocks"] = [
            {"name": CORE, "success": True, "attempts": 1, "phase": "rtl"}]
        st["pipeline_phase"] = "rtl"

        def _boom(_payload):
            raise AssertionError("pipeline_incomplete fired on a retired block")

        monkeypatch.setattr(pipeline_graph, "interrupt", _boom)
        out = await pipeline_graph.pipeline_complete_node(st)
        assert out["frontend_complete"] is True

    def test_missing_from_ignores_a_skipped_block(self):
        """The other half: ``_eligible_blocks`` already excludes skipped
        blocks, so a block marked skipped is never reported unresolved."""
        from orchestrator.langgraph.integration_helpers import missing_from
        completed = [{"name": CORE}, {"name": PAD, "skipped": True}]
        assert missing_from({CORE: "/x.v"}, completed) == []


# --------------------------------------------------------------------------- #
# Assembly when the adapter NEVER EXISTED
# --------------------------------------------------------------------------- #

_ENGINE_V = """\
module protocol_engine (
  input  wire wb_clk_i,
  input  wire wb_rst_i,
  input  wire qspi_csn,
  input  wire qspi_sck,
  input  wire [3:0] qspi_io_in,
  output wire [3:0] qspi_io_out,
  output wire qspi_drive_en,
  output wire irq_level
);
  assign qspi_io_out  = qspi_io_in;
  assign qspi_drive_en = ~qspi_csn;
  assign irq_level     = qspi_sck;
endmodule
"""


class TestTheAssemblerNeedsNoAdapterPresent:
    """The existing ``dropped_adapter`` path drops a block that EXISTS. After a
    retirement the block never existed, ``detect_wrapper_block`` returns None
    and the pin map is the sole enable condition -- that combination must still
    assemble a boundary-routed top."""

    def _assemble(self, tmp_path):
        from orchestrator.architecture.pin_map import parse_pin_map
        from orchestrator.langgraph.integration_helpers import (
            detect_wrapper_block,
            generate_caravel_wrapper_top,
            parse_verilog_ports,
        )
        p = tmp_path / "protocol_engine.v"
        p.write_text(_ENGINE_V)
        modules = {CORE: parse_verilog_ports(str(p))}
        assert detect_wrapper_block(modules) is None   # the enable condition
        # The contract edges STILL name the retired block on one end.
        edges = [
            {"producer_block": PAD, "consumer_block": CORE,
             "producer_port": "qspi_async_pins",
             "consumer_port": "qspi_async_pins",
             "fields": [{"name": "qspi_csn"}, {"name": "qspi_sck"},
                        {"name": "qspi_io_in"}],
             "sideband_signals": [], "edge_id": "e1"},
            {"producer_block": CORE, "consumer_block": PAD,
             "producer_port": "qspi_drive", "consumer_port": "qspi_drive",
             "fields": [{"name": "qspi_io_out"}, {"name": "qspi_drive_en"}],
             "sideband_signals": [], "edge_id": "e2"},
        ]
        return generate_caravel_wrapper_top(
            modules, edges, {CORE: str(p)}, str(tmp_path / "out"),
            None, parse_pin_map(PIN_MAP))

    def test_it_assembles_with_no_wrapper_block_and_no_hazards(self, tmp_path):
        asm = self._assemble(tmp_path)
        assert asm["wiring_errors"] == [], asm["wiring_errors"]
        assert asm["module_name"] == "user_project_wrapper"
        assert asm["block_count"] == 1
        assert asm["instantiated"] == [CORE]
        # Nothing was dropped -- there was nothing there to drop.
        assert asm["dropped_adapter"] == ""

    def test_edges_naming_the_retired_block_are_not_hazards(self, tmp_path):
        """Assembly used to drop these edges explicitly with the adapter. When
        the adapter never existed they must simply not resolve to anything --
        and specifically must NOT be reported as unresolvable NAMED edges."""
        asm = self._assemble(tmp_path)
        assert not any(PAD in e for e in asm["wiring_errors"])

    def test_the_boundary_is_routed_by_the_top(self, tmp_path):
        v = self._assemble(tmp_path)["verilog"]
        assert "wire qspi_csn = io_in[0];" in v
        assert "wire [3:0] qspi_io_in = io_in[5:2];" in v
        assert "assign io_out[5:2] = qspi_io_out;" in v
        assert "assign io_oeb[5:2] = {4{~qspi_drive_en}};" in v
        assert "assign io_out[6] = irq_level;" in v
        # and the surviving block's ports bind to those pin-map signals
        assert ".qspi_csn(qspi_csn)" in v
        assert ".qspi_io_out(qspi_io_out)" in v
        assert ".irq_level(irq_level)" in v

    def test_the_missing_instantiation_postcondition_is_satisfied(
            self, tmp_path):
        """integration_check's postcondition: every block in ``modules`` is
        instantiated, minus a deliberately dropped adapter. A retired block is
        not in ``modules`` at all, so it cannot be reported missing."""
        asm = self._assemble(tmp_path)
        dropped = asm.get("dropped_adapter") or ""
        modules_seen = {CORE}
        missing = [b for b in modules_seen
                   if b != dropped and f"u_{b} (" not in asm["verilog"]]
        assert missing == []
