# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""``integration_check_node`` must refuse to assemble a chip missing a block.

Root cause behind the raster run's missing GPIO boundary: ``discover_block_rtl``
resolved a block only from its result ``rtl_path`` or the ``<block_name>.v``
filename convention, so the ONE block whose file is not named after it -- the
Caravel pad adapter, whose module name is locked to ``user_project_wrapper`` by
an interface contract and whose spec carries
``rtl_target='rtl/user_project_wrapper.v'`` -- resolved to nothing and was
dropped from the returned dict with no error. 7 of 8 blocks reached ``modules``,
``detect_wrapper_block`` returned None, the default-ON deterministic Caravel
assembler never ran, and the assembled top carried no io_in/io_out/io_oeb at
all while integration DV reported PASS on a co-tuned BFM.

Discovery is fixed in ``integration_helpers``; this pins the CALLER: an eligible
block whose RTL cannot be located is an assembly blocker that parks the existing
``integration_failure`` interrupt BEFORE the Integration Lead call, with
``override``/``abort`` and a single fail-closed re-check on ``retry`` -- the same
shape as the contract-staleness preflight beside it.
"""

from __future__ import annotations

import pytest

from orchestrator.langgraph import pipeline_graph
from orchestrator.langgraph.integration_helpers import VerilogModule, VerilogPort


def _state(tmp_path, *, write_missing_rtl: bool = False):
    """Two passed blocks; `pad_adapter` resolves only via its spec rtl_target."""
    (tmp_path / "rtl").mkdir(parents=True, exist_ok=True)
    found = tmp_path / "rtl" / "core.v"
    found.write_text("module core(input clk); endmodule\n")
    if write_missing_rtl:
        # the pad adapter's REAL file: named after its locked module, not the block
        (tmp_path / "rtl" / "user_project_wrapper.v").write_text(
            "module user_project_wrapper(input clk); endmodule\n"
        )
    return {
        "project_root": str(tmp_path),
        "completed_blocks": [
            {"name": "core", "success": True, "rtl_path": str(found)},
            {"name": "pad_adapter", "success": True},
        ],
        "block_queue": [{"name": "core"}, {"name": "pad_adapter"}],
    }


def _seams(monkeypatch, *, rtl_target: str = ""):
    monkeypatch.setattr(
        pipeline_graph, "load_architecture_connections", lambda pr: ([{"a": 1}], "chip"))
    monkeypatch.setattr(
        pipeline_graph, "parse_verilog_ports",
        lambda p: VerilogModule(name="core", ports=[VerilogPort("clk", "input")]))
    monkeypatch.setattr(pipeline_graph, "write_graph_event", lambda *a, **k: None)

    # discover_block_rtl / unresolved_block_rtl read the block spec for
    # rtl_target; stub the spec loader seam by patching the helpers' own reader.
    import orchestrator.langgraph.integration_helpers as ih

    real_discover = ih.discover_block_rtl

    def _discover(project_root, completed_blocks):
        blocks = []
        for b in completed_blocks:
            b = dict(b)
            if b.get("name") == "pad_adapter" and rtl_target:
                b["rtl_target"] = rtl_target
            blocks.append(b)
        return real_discover(project_root, blocks)

    monkeypatch.setattr(ih, "discover_block_rtl", _discover)
    monkeypatch.setattr(pipeline_graph, "discover_block_rtl", _discover)
    return ih


def test_block_rtl_complete_gate_default_on(monkeypatch):
    monkeypatch.delenv("CORESMITH_BLOCK_RTL_COMPLETE_GATE", raising=False)
    assert pipeline_graph._block_rtl_complete_gate_enabled() is True
    monkeypatch.setenv("CORESMITH_BLOCK_RTL_COMPLETE_GATE", "0")
    assert pipeline_graph._block_rtl_complete_gate_enabled() is False


@pytest.mark.asyncio
async def test_missing_block_rtl_parks_before_assembly(monkeypatch, tmp_path):
    monkeypatch.delenv("CORESMITH_BLOCK_RTL_COMPLETE_GATE", raising=False)
    _seams(monkeypatch)
    captured = {}
    monkeypatch.setattr(
        pipeline_graph, "interrupt",
        lambda p: (captured.update(p), {"action": "abort"})[1])

    result = await pipeline_graph.integration_check_node(_state(tmp_path))

    assert captured.get("error_kind") == "unresolved_block_rtl"
    assert captured.get("missing_block_rtl") == ["pad_adapter"]
    assert "override" in captured.get("supported_actions", [])
    ir = result["integration_result"]
    assert ir["aborted"] is True
    assert ir["missing_blocks"] == ["pad_adapter"]
    # no top-level RTL was written -- the gate parks BEFORE assembly
    assert not (tmp_path / "rtl" / "integration").exists()


@pytest.mark.asyncio
async def test_retry_that_does_not_resolve_fails_closed(monkeypatch, tmp_path):
    monkeypatch.delenv("CORESMITH_BLOCK_RTL_COMPLETE_GATE", raising=False)
    _seams(monkeypatch)
    monkeypatch.setattr(pipeline_graph, "interrupt", lambda p: {"action": "retry"})

    result = await pipeline_graph.integration_check_node(_state(tmp_path))
    ir = result["integration_result"]
    assert ir["aborted"] is True
    assert "unresolved after retry" in ir["reason"]


@pytest.mark.asyncio
async def test_rtl_target_resolution_clears_the_gate(monkeypatch, tmp_path):
    """The fixed discovery path: rtl_target resolves the locked pad adapter."""
    monkeypatch.delenv("CORESMITH_BLOCK_RTL_COMPLETE_GATE", raising=False)
    _seams(monkeypatch, rtl_target="rtl/user_project_wrapper.v")
    monkeypatch.setattr(
        pipeline_graph, "interrupt",
        lambda p: (_ for _ in ()).throw(AssertionError("gate must not park")))
    # stop the node right after the gate so no LLM/assembly runs
    monkeypatch.setattr(
        pipeline_graph, "parse_verilog_ports",
        lambda p: VerilogModule(name="", ports=[]))

    result = await pipeline_graph.integration_check_node(
        _state(tmp_path, write_missing_rtl=True))
    # both blocks resolved -> the gate passed; the node then stops on the
    # unparsable-RTL guard rather than on a missing block
    assert result["integration_result"]["reason"] == "No block RTL could be parsed"


@pytest.mark.asyncio
async def test_total_absence_of_rtl_is_left_to_the_existing_skip(
    monkeypatch, tmp_path
):
    """Nothing resolved at all -> no chip to assemble; the existing skip reports it.

    The gate is about a chip assembled AROUND a missing block, not about an empty
    discovery (which the "No block RTL could be parsed" return already surfaces).
    """
    monkeypatch.delenv("CORESMITH_BLOCK_RTL_COMPLETE_GATE", raising=False)
    monkeypatch.setattr(
        pipeline_graph, "load_architecture_connections", lambda pr: ([{"a": 1}], "chip"))
    monkeypatch.setattr(pipeline_graph, "write_graph_event", lambda *a, **k: None)
    monkeypatch.setattr(
        pipeline_graph, "interrupt",
        lambda p: (_ for _ in ()).throw(AssertionError("gate must not park")))
    state = {
        "project_root": str(tmp_path),
        "completed_blocks": [
            {"name": "core", "success": True},
            {"name": "pad_adapter", "success": True},
        ],
        "block_queue": [{"name": "core"}, {"name": "pad_adapter"}],
    }
    result = await pipeline_graph.integration_check_node(state)
    assert result["integration_result"]["reason"] == "No block RTL could be parsed"


@pytest.mark.asyncio
async def test_gate_off_restores_drop_and_continue(monkeypatch, tmp_path):
    monkeypatch.setenv("CORESMITH_BLOCK_RTL_COMPLETE_GATE", "0")
    _seams(monkeypatch)
    monkeypatch.setattr(
        pipeline_graph, "interrupt",
        lambda p: (_ for _ in ()).throw(AssertionError("gate must not park")))
    monkeypatch.setattr(
        pipeline_graph, "parse_verilog_ports",
        lambda p: VerilogModule(name="", ports=[]))

    result = await pipeline_graph.integration_check_node(_state(tmp_path))
    # unchanged legacy behavior: the missing block is simply absent
    assert result["integration_result"]["reason"] == "No block RTL could be parsed"


@pytest.mark.asyncio
async def test_override_assembles_without_the_block(monkeypatch, tmp_path):
    monkeypatch.delenv("CORESMITH_BLOCK_RTL_COMPLETE_GATE", raising=False)
    _seams(monkeypatch)
    monkeypatch.setattr(pipeline_graph, "interrupt", lambda p: {"action": "override"})
    monkeypatch.setattr(
        pipeline_graph, "parse_verilog_ports",
        lambda p: VerilogModule(name="", ports=[]))

    result = await pipeline_graph.integration_check_node(_state(tmp_path))
    assert result["integration_result"]["reason"] == "No block RTL could be parsed"
