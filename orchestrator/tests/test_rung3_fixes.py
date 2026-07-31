# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""rung3-fixes-1 -- engine defects found live by the rung-3 video_codec run.

Hermetic (no LLM, no EDA except an optional real-yosys probe elsewhere):

 * Defect 1 -- the deterministic integration checker must NOT manufacture a
   phantom error for a PORT-LESS connection (interface name only). It resolves
   the port by the interface name when it can, else SKIPS informationally.
 * Defect 2 -- integration_check's retry/fix_rtl path must RE-PARK (a final
   integration_failure interrupt) instead of letting route_after_integration
   silently END the graph with errors outstanding.
 * Minor 4 -- the SKIP_SYNTH probe path must WRITE the ppa_report.json its
   scoreboard row references.
"""
from __future__ import annotations

import json

import pytest

from orchestrator.langgraph import pipeline_graph
from orchestrator.langgraph.integration_helpers import (
    VerilogModule,
    VerilogPort,
    _find_port_fuzzy,
    check_integration_compatibility,
)


def _mk_module(name, ports):
    return VerilogModule(
        name=name,
        ports=[VerilogPort(name=n, direction=d, width=w,
                           msb=max(0, w - 1), lsb=0)
               for (n, d, w) in ports],
    )


# ===========================================================================
# Defect 1 -- port-less connection: match-by-interface, else skip-informational
# ===========================================================================

class TestPortlessConnectionCompat:
    def _modules(self, *, src_w=8, dst_w=8):
        return {
            "a": _mk_module("blk_a", [("clk", "input", 1),
                                      ("data_out", "output", src_w)]),
            "b": _mk_module("blk_b", [("clk", "input", 1),
                                      ("data_in", "input", dst_w)]),
        }

    def test_find_port_fuzzy_resolves_portless_by_interface(self):
        # port_name == "" (no attribution) resolves via the connection name.
        mod = self._modules()["a"]
        p = _find_port_fuzzy(mod, "", "data")
        assert p is not None and p.name == "data_out"

    def test_find_port_fuzzy_named_lookup_unchanged(self):
        # A real port name still resolves exactly; a genuinely-absent named
        # port still returns None (no connection-name fallback when port_name
        # already produced key terms).
        mod = self._modules()["a"]
        assert _find_port_fuzzy(mod, "data_out", "iface").name == "data_out"
        assert _find_port_fuzzy(mod, "totally_absent", "iface") is None

    def test_portless_matching_interface_is_checked_not_errored(self):
        # Interface name "data" resolves data_out/data_in; matching widths ->
        # zero findings (the check RAN, found nothing wrong).
        conns = [{"from_block": "a", "to_block": "b", "interface": "data"}]
        out = check_integration_compatibility(conns, self._modules(src_w=8, dst_w=8))
        assert [m for m in out if m.severity == "error"] == []
        assert [m for m in out if m.issue_type == "unresolved_portless_connection"] == []

    def test_portless_matching_interface_still_catches_width_mismatch(self):
        # Proves the check actually RAN on the resolved ports: 8b -> 16b is a
        # real width_mismatch error even though the connection was port-less.
        conns = [{"from_block": "a", "to_block": "b", "interface": "data"}]
        out = check_integration_compatibility(conns, self._modules(src_w=8, dst_w=16))
        errs = [m for m in out if m.severity == "error"]
        assert any(m.issue_type == "width_mismatch" for m in errs)

    def test_portless_no_match_is_skipped_informational_zero_errors(self):
        # Interface name matches NO port -> SKIP with an info note, never a
        # manufactured "Source port '' not found" error.
        conns = [{"from_block": "a", "to_block": "b", "interface": "zzz_unmatched"}]
        out = check_integration_compatibility(conns, self._modules())
        assert [m for m in out if m.severity == "error"] == []
        info = [m for m in out if m.issue_type == "unresolved_portless_connection"]
        assert len(info) == 1
        assert info[0].severity == "info"
        # And critically: no empty-string port name leaked into a description.
        assert "port ''" not in info[0].description

    def test_named_missing_port_is_still_an_error(self):
        # Regression guard: a NON-empty from_port that is genuinely absent (no
        # exact/variant/substring fuzzy match) stays a hard missing_port error
        # -- the skip only applies to PORT-LESS conns. "xctrl_sig" shares no
        # >2-char token with clk/data_out, so the fuzzy matcher can't resolve it.
        conns = [{
            "from_block": "a", "to_block": "b",
            "from_port": "xctrl_sig", "to_port": "data_in",
            "interface": "data",
        }]
        out = check_integration_compatibility(conns, self._modules())
        errs = [m for m in out if m.severity == "error"]
        assert any(m.issue_type == "missing_port" for m in errs)

    def test_portless_destination_no_match_skips(self):
        # Source resolves (data_out), destination is port-less and unmatched
        # ("ctrl" matches no port on b) -> info skip, zero errors.
        mods = {
            "a": _mk_module("blk_a", [("ctrl", "output", 8)]),
            "b": _mk_module("blk_b", [("data_in", "input", 8)]),
        }
        conns = [{"from_block": "a", "to_block": "b", "interface": "ctrl"}]
        out = check_integration_compatibility(conns, mods)
        assert [m for m in out if m.severity == "error"] == []
        assert any(m.issue_type == "unresolved_portless_connection" for m in out)


# ===========================================================================
# Defect 2 -- integration_check re-parks retry/fix_rtl (no silent END)
# ===========================================================================

class _Interrupts:
    """Stateful fake ``interrupt`` returning a scripted response sequence."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, payload):
        self.calls.append(dict(payload))
        if self._responses:
            return self._responses.pop(0)
        return {"action": "abort"}


def _wire_mismatch_design(monkeypatch, *, src_width=8, dst_width=16):
    """A lint-clean 2-block design whose deterministic checker finds a width
    mismatch (error_count >= 1, lint_clean True -> accept offered)."""
    connections = [{
        "from_block": "a", "to_block": "b",
        "from_port": "out_data", "to_port": "in_data",
        "interface": "axis", "data_width": 8,
    }]
    modules = {
        "a": _mk_module("blk_a", [("out_data", "output", src_width)]),
        "b": _mk_module("blk_b", [("in_data", "input", dst_width)]),
    }

    monkeypatch.setattr(
        pipeline_graph, "load_architecture_connections",
        lambda pr: (connections, "twoblk"),
    )
    monkeypatch.setattr(
        pipeline_graph, "discover_block_rtl",
        lambda pr, passed: {"a": "/nope/a.v", "b": "/nope/b.v"},
    )
    monkeypatch.setattr(
        pipeline_graph, "parse_verilog_ports",
        lambda path, module=None: (modules["a"] if path.endswith("a.v")
                                   else modules["b"]),
    )
    monkeypatch.setattr(
        pipeline_graph, "lint_top_level",
        lambda top, blocks, name: {"clean": True, "errors": "", "log_path": ""},
    )

    from orchestrator.langchain.agents import integration_lead

    class _FakeAgent:
        async def integrate(self, **kw):
            return {
                "mismatches": [],
                "module_name": "twoblk_top",
                "rtl_path": "/nope/twoblk_top.v",
                "verilog": "module twoblk_top(); blk_a x(); blk_b y(); endmodule",
                "wire_count": 1,
                "notes": "",
                "skipped_connections": [],
            }

    monkeypatch.setattr(integration_lead, "IntegrationLeadAgent", _FakeAgent)
    monkeypatch.setattr(
        integration_lead, "assert_blocks_instantiated", lambda *a, **k: "")
    monkeypatch.setattr(
        integration_lead, "assert_no_memory_primitive_defined", lambda *a, **k: "")

    return {
        "project_root": "/tmp/does-not-matter",
        "pipeline_phase": "rtl",
        "completed_blocks": [
            {"name": "a", "success": True, "phase": "rtl"},
            {"name": "b", "success": True, "phase": "rtl"},
        ],
        "block_queue": [{"name": "a"}, {"name": "b"}],
    }


class TestIntegrationRetryReparks:
    @pytest.mark.asyncio
    async def test_retry_reparks_not_silent_end(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_DETERMINISTIC_INTEGRATION_CHECK", "1")
        state = _wire_mismatch_design(monkeypatch)
        fake = _Interrupts([{"action": "retry"}, {"action": "abort"}])
        monkeypatch.setattr(pipeline_graph, "interrupt", fake)

        result = await pipeline_graph.integration_check_node(state)
        ir = result["integration_result"]

        # The retry did NOT return to a silent END: a SECOND (re-park)
        # interrupt fired, carrying the outstanding errors + an accept/abort
        # choice + a loud error_message.
        assert len(fake.calls) == 2, "expected a re-park interrupt after retry"
        repark = fake.calls[1]
        assert repark["type"] == "integration_failure"
        assert repark["error_count"] >= 1
        assert "accept" in repark["supported_actions"]
        assert "abort" in repark["supported_actions"]
        assert repark.get("error_message")
        assert repark.get("repark_round") == 1
        assert ir.get("repark_rounds") == 1
        assert ir.get("error_message")
        # Explicit abort at the re-park -> aborted (abort semantics preserved).
        assert ir.get("aborted") is True
        assert pipeline_graph.route_after_integration(result) == "__end__"

    @pytest.mark.asyncio
    async def test_accept_at_repark_terminates_loop_to_dv(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_DETERMINISTIC_INTEGRATION_CHECK", "1")
        state = _wire_mismatch_design(monkeypatch)
        fake = _Interrupts([{"action": "retry"}, {"action": "accept"}])
        monkeypatch.setattr(pipeline_graph, "interrupt", fake)

        result = await pipeline_graph.integration_check_node(state)
        ir = result["integration_result"]

        assert len(fake.calls) == 2
        assert ir.get("accepted_by_user") is True
        assert ir.get("aborted") is not True
        # accept advances to DV (model_integration is the flag-off no-op pass).
        assert pipeline_graph.route_after_integration(result) != "__end__"

    @pytest.mark.asyncio
    async def test_first_abort_unchanged_no_repark(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_DETERMINISTIC_INTEGRATION_CHECK", "1")
        state = _wire_mismatch_design(monkeypatch)
        fake = _Interrupts([{"action": "abort"}])
        monkeypatch.setattr(pipeline_graph, "interrupt", fake)

        result = await pipeline_graph.integration_check_node(state)
        ir = result["integration_result"]

        # A direct abort still terminates in ONE interrupt -- no re-park.
        assert len(fake.calls) == 1
        assert ir.get("aborted") is True
        assert ir.get("repark_rounds") is None
        assert pipeline_graph.route_after_integration(result) == "__end__"

    @pytest.mark.asyncio
    async def test_repark_cap_fails_closed_to_loud_abort(self, monkeypatch):
        # A driver (or a plain-return interrupt double) that NEVER resolves --
        # always fix_rtl -- must be bounded by the re-park cap and fail closed
        # to a LOUD terminal abort (never a silent END, never an infinite loop).
        monkeypatch.setenv("CORESMITH_DETERMINISTIC_INTEGRATION_CHECK", "1")
        state = _wire_mismatch_design(monkeypatch)
        calls = {"n": 0}

        def always_retry(payload):
            calls["n"] += 1
            return {"action": "fix_rtl", "rtl_fix_description": "x"}

        monkeypatch.setattr(pipeline_graph, "interrupt", always_retry)

        result = await pipeline_graph.integration_check_node(state)
        ir = result["integration_result"]

        assert calls["n"] <= pipeline_graph._INTEGRATION_REPARK_CAP + 1
        assert ir.get("aborted") is True
        assert ir.get("error_message")
        assert pipeline_graph.route_after_integration(result) == "__end__"

    @pytest.mark.asyncio
    async def test_accept_first_pass_still_advances(self, monkeypatch):
        # A first-pass accept (operator waives the mismatch immediately) still
        # advances to DV without ever re-parking.
        monkeypatch.setenv("CORESMITH_DETERMINISTIC_INTEGRATION_CHECK", "1")
        state = _wire_mismatch_design(monkeypatch)
        fake = _Interrupts([{"action": "accept"}])
        monkeypatch.setattr(pipeline_graph, "interrupt", fake)

        result = await pipeline_graph.integration_check_node(state)
        assert len(fake.calls) == 1
        assert result["integration_result"].get("accepted_by_user") is True
        assert pipeline_graph.route_after_integration(result) != "__end__"


# ===========================================================================
# Minor 4 -- SKIP_SYNTH writes the ppa_report.json its row references
# ===========================================================================

class TestSkipSynthWritesReport:
    @pytest.mark.asyncio
    async def test_skip_synth_pass_writes_report_at_recorded_path(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("CORESMITH_SKIP_SYNTH", "1")
        meta = {
            "ff": 128, "cells": 50, "mem_bits": 0, "area_um2": None,
            "elaborated": True, "budget_ff": 200, "budget_area_um2": None,
        }
        monkeypatch.setattr(
            pipeline_graph, "_evaluate_ppa_gate",
            lambda *a, **k: (True, [], dict(meta)),
        )
        state = {
            "current_block": {"name": "blk"},
            "project_root": str(tmp_path),
            "attempt": 1,
            "rtl_path": "",
            "pipeline_phase": "rtl",
        }
        result = await pipeline_graph.synthesize_node(state)
        assert result["synth_success"] is True

        report = tmp_path / ".coresmith" / "blocks" / "blk" / "ppa_report.json"
        assert report.exists(), "SKIP_SYNTH must write the ppa_report it records"
        data = json.loads(report.read_text())
        assert data["probe"] == "skip_synth"
        assert data["ppa_ok"] is True
        assert data["ff"] == 128
        assert data["budget_ff"] == 200
