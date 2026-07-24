# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""A-Fix 3(a): deterministic integration-compatibility checker wired into
integration_check_node.

Hermetic -- no LLM, no EDA. The Integration Lead agent + disk I/O helpers are
monkeypatched; the deterministic ``check_integration_compatibility`` runs for
real and its width/direction/missing-port findings must merge into the
existing ``integration_failure`` interrupt.
"""

from __future__ import annotations

import pytest

from orchestrator.langgraph import pipeline_graph
from orchestrator.langgraph.integration_helpers import (
    VerilogModule,
    VerilogPort,
)

# ---------------------------------------------------------------------------
# Pure helpers: env gate + merge
# ---------------------------------------------------------------------------

def test_gate_default_on(monkeypatch):
    monkeypatch.delenv("CORESMITH_DETERMINISTIC_INTEGRATION_CHECK", raising=False)
    assert pipeline_graph._deterministic_integration_check_enabled() is True


def test_gate_can_be_disabled(monkeypatch):
    monkeypatch.setenv("CORESMITH_DETERMINISTIC_INTEGRATION_CHECK", "0")
    assert pipeline_graph._deterministic_integration_check_enabled() is False


def test_merge_dedups_and_error_severity_wins():
    llm = [
        {"from_block": "a", "to_block": "b",
         "issue_type": "width_mismatch", "severity": "warning",
         "description": "llm noticed"},
    ]
    det = [
        # same identity as the LLM entry -> merged, severity upgraded to error
        {"from_block": "a", "to_block": "b",
         "issue_type": "width_mismatch", "severity": "error",
         "description": "deterministic"},
        # new identity -> appended, tagged deterministic
        {"from_block": "c", "to_block": "d",
         "issue_type": "missing_port", "severity": "error",
         "description": "port missing"},
    ]
    merged = pipeline_graph._merge_mismatches(llm, det)
    assert len(merged) == 2
    ab = next(m for m in merged if m["to_block"] == "b")
    assert ab["severity"] == "error"          # deterministic error wins
    assert ab["deterministic"] is True
    assert ab["description"] == "llm noticed"  # LLM entry preserved, not clobbered
    cd = next(m for m in merged if m["to_block"] == "d")
    assert cd["deterministic"] is True


def test_merge_tolerates_non_dicts():
    merged = pipeline_graph._merge_mismatches([None, "x"], [{"from_block": "a"}])
    assert len(merged) == 1
    assert merged[0]["deterministic"] is True


# ---------------------------------------------------------------------------
# Node wiring
# ---------------------------------------------------------------------------

def _mk_module(name, ports):
    return VerilogModule(
        name=name,
        ports=[VerilogPort(name=n, direction=d, width=w,
                           msb=max(0, w - 1), lsb=0)
               for (n, d, w) in ports],
    )


def _wire_two_block_design(monkeypatch, *, src_width, dst_width):
    """Patch the node's disk/agent seams for a 2-block design with a
    producer->consumer connection of the given port widths."""
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
        lambda path: modules["a"] if path.endswith("a.v") else modules["b"],
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


@pytest.mark.asyncio
async def test_deterministic_width_mismatch_flows_into_interrupt(monkeypatch):
    monkeypatch.setenv("CORESMITH_DETERMINISTIC_INTEGRATION_CHECK", "1")
    state = _wire_two_block_design(monkeypatch, src_width=8, dst_width=16)

    captured = {}

    def fake_interrupt(payload):
        captured.update(payload)
        return {"action": "accept"}

    monkeypatch.setattr(pipeline_graph, "interrupt", fake_interrupt)

    result = await pipeline_graph.integration_check_node(state)

    # The agent found NO mismatches; the deterministic checker found the
    # 8b -> 16b width mismatch and it flowed into the integration_failure
    # interrupt as an error.
    assert captured.get("type") == "integration_failure"
    assert captured.get("error_count", 0) >= 1
    det = [
        m for m in captured.get("mismatches", [])
        if m.get("issue_type") == "width_mismatch"
    ]
    assert det, f"expected deterministic width mismatch, got {captured.get('mismatches')}"
    assert det[0]["severity"] == "error"
    assert det[0].get("deterministic") is True
    # `accept` stayed the operator override (lint clean).
    assert "accept" in captured.get("supported_actions", [])
    assert result["integration_result"]["accepted_by_user"] is True


@pytest.mark.asyncio
async def test_gate_off_skips_deterministic_check(monkeypatch):
    monkeypatch.setenv("CORESMITH_DETERMINISTIC_INTEGRATION_CHECK", "0")
    state = _wire_two_block_design(monkeypatch, src_width=8, dst_width=16)

    fired = {"interrupt": False}

    def fake_interrupt(payload):
        fired["interrupt"] = True
        return {"action": "accept"}

    monkeypatch.setattr(pipeline_graph, "interrupt", fake_interrupt)

    result = await pipeline_graph.integration_check_node(state)

    # With the deterministic check disabled and the agent reporting no
    # mismatches on a lint-clean top, the node passes without interrupting.
    assert fired["interrupt"] is False
    ir = result["integration_result"]
    assert ir.get("error_count", 0) == 0
    assert ir.get("accepted_by_user") is None or ir.get("accepted_by_user") is False


@pytest.mark.asyncio
async def test_matching_widths_no_false_positive(monkeypatch):
    monkeypatch.setenv("CORESMITH_DETERMINISTIC_INTEGRATION_CHECK", "1")
    state = _wire_two_block_design(monkeypatch, src_width=8, dst_width=8)

    fired = {"interrupt": False}

    def fake_interrupt(payload):
        fired["interrupt"] = True
        return {"action": "accept"}

    monkeypatch.setattr(pipeline_graph, "interrupt", fake_interrupt)

    result = await pipeline_graph.integration_check_node(state)

    # Ports agree -> deterministic checker finds nothing -> no interrupt.
    assert fired["interrupt"] is False
    assert result["integration_result"].get("error_count", 0) == 0


# ---------------------------------------------------------------------------
# dv-hardening-28: direction-aware fuzzy port match on multi-interface blocks
# ---------------------------------------------------------------------------

def _multi_iface_block(name: str) -> VerilogModule:
    # A block with BOTH an s_axis_* input interface and an m_axis_* output
    # interface sharing the 'framed_byte' label -- the shape that tripped the
    # direction-blind substring match on armD + aes.
    return VerilogModule(name=name, ports=[
        VerilogPort("clk", "input"), VerilogPort("rst_n", "input"),
        VerilogPort("s_axis_framed_byte_tdata", "input", 8),
        VerilogPort("s_axis_framed_byte_tvalid", "input"),
        VerilogPort("s_axis_framed_byte_tready", "output"),
        VerilogPort("m_axis_framed_byte_tdata", "output", 8),
        VerilogPort("m_axis_framed_byte_tvalid", "output"),
        VerilogPort("m_axis_framed_byte_tready", "input"),
    ])


def test_multi_interface_no_false_direction_error():
    from orchestrator.langgraph.integration_helpers import (
        check_integration_compatibility,
    )
    producer = _multi_iface_block("framer")
    consumer = _multi_iface_block("egress")
    # producer's OUTPUT interface -> consumer's INPUT interface, by label only.
    conns = [{
        "from_block": "framer", "to_block": "egress",
        "from_port": "", "to_port": "", "interface": "framed_byte",
        "data_width": 8,
    }]
    ms = check_integration_compatibility(
        conns, {"framer": producer, "egress": consumer})
    dir_errs = [m for m in ms if m.issue_type == "direction_error"]
    assert not dir_errs, (
        "producer-output lookup must resolve m_axis (output), not s_axis "
        f"(input); got {[m.description for m in dir_errs]}")


def test_fuzzy_prefers_requested_direction():
    from orchestrator.langgraph.integration_helpers import _find_port_fuzzy
    blk = _multi_iface_block("b")
    out_p = _find_port_fuzzy(blk, "framed_byte", "framed_byte",
                             prefer_direction="output")
    in_p = _find_port_fuzzy(blk, "framed_byte", "framed_byte",
                            prefer_direction="input")
    assert out_p is not None and out_p.direction == "output"
    assert in_p is not None and in_p.direction == "input"
