# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""A4: the orchestrator assembles the shell top before tier 0 and after every
tier, recording snapshots; the final top is the same deterministic assembly."""
from __future__ import annotations

import asyncio
from pathlib import Path

from orchestrator.langgraph import pipeline_graph as pg
from orchestrator.state_store.project_db import open_project

_FX = Path(__file__).parent / "fixtures" / "vip2"
_EDGE = {"edge_id": "req__m_q__to__rsp__s_q", "producer_block": "req", "producer_port": "m_q",
         "consumer_block": "rsp", "consumer_port": "s_q", "handshake_protocol": "req_resp",
         "data_width_bits": 8, "fields": [{"name": "addr", "width": 8, "msb": 7, "lsb": 0}],
         "sideband_signals": [{"name": "req_valid", "width": 1, "direction": "producer->consumer"},
                              {"name": "req_gnt", "width": 1, "direction": "consumer->producer"},
                              {"name": "rsp_valid", "width": 1, "direction": "consumer->producer"},
                              {"name": "rdata", "width": 8, "direction": "consumer->producer"}],
         "timing": {"req_to_rsp_cycles": {"exact": 1}}}
_QUEUE = [{"name": "req", "tier": 1, "rtl_target": "rtl/req.v"},
          {"name": "rsp", "tier": 1, "rtl_target": "rtl/rsp.v"}]


def _project(tmp_path, chassis=None):
    (tmp_path / "inputs").mkdir(exist_ok=True)
    (tmp_path / "inputs" / "task.yaml").write_text(f"chassis: {chassis}\n" if chassis else "top: chip_top\n")
    db = open_project(tmp_path)
    db.import_contracts({"contracts": [_EDGE]})
    return db


def _state(tmp_path):
    return {"project_root": str(tmp_path), "block_queue": _QUEUE, "tier_list": [1], "current_tier_index": 0}


class TestShellNodes:
    def test_init_assembles_all_stubs_and_snapshots(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_SHELL_INTEGRATION", "1")
        monkeypatch.delenv("CORESMITH_TOP_MODULE", raising=False)
        db = _project(tmp_path)
        out = asyncio.run(pg.shell_integration_init_node(_state(tmp_path)))
        snap = out["shell_snapshot"]
        assert snap["top"] == "chip_top" and sorted(snap["stub_blocks"]) == ["req", "rsp"]
        assert snap["wiring_errors"] == [] and snap["wires"] == 5
        assert (tmp_path / ".coresmith" / "shell" / "chip_top.v").exists()
        latest = db.latest_integration_snapshot()
        assert latest["tier"] == "init" and latest["stub_blocks"] == snap["stub_blocks"]
        assert (tmp_path / ".coresmith" / "integration_snapshot.json").exists()

    def test_update_uses_real_rtl_for_published_blocks(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_SHELL_INTEGRATION", "1")
        monkeypatch.delenv("CORESMITH_TOP_MODULE", raising=False)
        db = _project(tmp_path)
        (tmp_path / "rtl").mkdir()
        (tmp_path / "rtl" / "rsp.v").write_text(
            (_FX / "responder.v").read_text().replace("module responder", "module rsp"))
        db.set_result("rsp", "best", {"sim_passed": True, "done": True})
        out = asyncio.run(pg.shell_integration_update_node(_state(tmp_path)))
        snap = out["shell_snapshot"]
        assert snap["real_blocks"] == ["rsp"] and snap["stub_blocks"] == ["req"]
        assert "integration_contract_failures" not in out
        # a block whose RTL drifted from the contract is attributed
        (tmp_path / "rtl" / "rsp.v").write_text(
            (tmp_path / "rtl" / "rsp.v").read_text().replace("s_q_rsp_valid", "s_q_resp_valid"))
        out = asyncio.run(pg.shell_integration_update_node(_state(tmp_path)))
        fails = out["integration_contract_failures"]
        assert fails and fails[0]["block"] == "rsp" and fails[0]["category"] == "INTEGRATION_CONTRACT"

    def test_caravel_chassis_and_flag_off_skip(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_SHELL_INTEGRATION", "1")
        _project(tmp_path, chassis="caravel")
        assert asyncio.run(pg.shell_integration_init_node(_state(tmp_path))) == {}
        monkeypatch.setenv("CORESMITH_SHELL_INTEGRATION", "0")
        _project(tmp_path)
        assert asyncio.run(pg.shell_integration_init_node(_state(tmp_path))) == {}

    def test_graph_wires_the_nodes(self):
        g = pg.build_pipeline_graph() if hasattr(pg, "build_pipeline_graph") else None
        if g is None:
            return
        assert "shell_init" in g.nodes and "shell_update" in g.nodes

    def test_all_real_assembly_for_the_final_top(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CORESMITH_TOP_MODULE", raising=False)
        _project(tmp_path)
        (tmp_path / "rtl").mkdir()
        for b in ("req", "rsp"):
            (tmp_path / "rtl" / f"{b}.v").write_text(
                (_FX / "responder.v").read_text().replace("module responder", f"module {b}")
                if b == "rsp" else
                "module req(input wire clk, input wire rst_n, output wire [7:0] m_q_addr, "
                "output wire m_q_req_valid, input wire m_q_req_gnt, input wire m_q_rsp_valid, "
                "input wire [7:0] m_q_rdata);\n  assign m_q_addr = 0; assign m_q_req_valid = 0;\nendmodule\n")
        asm, elab, snap = pg._shell_assemble(str(tmp_path), _QUEUE, tier="final", all_real=True)
        assert asm.stubs == [] and asm.wiring_errors == [] and snap["real_blocks"] == ["req", "rsp"]


def test_engine_primitive_modmissing_does_not_requeue_blocks(tmp_path, monkeypatch):
    """The live-run symptom: an engine-lib MODMISSING in a published block's
    file became an INTEGRATION_CONTRACT failure and re-ran the tier."""
    from orchestrator.langgraph import shell_integration as SI
    monkeypatch.setenv("CORESMITH_SHELL_INTEGRATION", "1")
    monkeypatch.delenv("CORESMITH_TOP_MODULE", raising=False)
    db = _project(tmp_path)
    (tmp_path / "rtl").mkdir()
    (tmp_path / "rtl" / "rsp.v").write_text(
        (_FX / "responder.v").read_text().replace("module responder", "module rsp"))
    db.set_result("rsp", "best", {"sim_passed": True, "done": True})
    err = (f"%Error-MODMISSING: {tmp_path}/rtl/rsp.v:12:3: "
           "Cannot find file containing module: 'cs_sram_1rw1r'")
    monkeypatch.setattr(SI, "elaborate", lambda asm, **kw: {"ran": True, "ok": False, "errors": [err]})
    out = asyncio.run(pg.shell_integration_update_node(_state(tmp_path)))
    assert "integration_contract_failures" not in out
    assert out["shell_snapshot"]["elaborated"] is False    # still reported, just not blamed


def test_primitive_fabric_rtl_joins_the_shell(tmp_path, monkeypatch):
    """A materialized fabric lives at rtl/interconnect/cs_fabric_<name>.v with a
    non-ANSI header; the shell used to look for rtl/<block>.v and stub it."""
    monkeypatch.setenv("CORESMITH_SHELL_INTEGRATION", "1")
    monkeypatch.delenv("CORESMITH_TOP_MODULE", raising=False)
    db = _project(tmp_path)
    (tmp_path / "rtl" / "interconnect").mkdir(parents=True)
    (tmp_path / "rtl" / "interconnect" / "cs_fabric_x.v").write_text(
        "module cs_fabric_x(clk, rst_n, s_q_addr, s_q_req_valid, s_q_req_gnt, s_q_rsp_valid, s_q_rdata);\n"
        "  input clk;\n  input rst_n;\n  input [7:0] s_q_addr;\n  input s_q_req_valid;\n"
        "  output s_q_req_gnt;\n  output s_q_rsp_valid;\n  output [7:0] s_q_rdata;\n"
        "  assign s_q_req_gnt = 1'b1;\n  assign s_q_rsp_valid = s_q_req_valid;\n  assign s_q_rdata = s_q_addr;\n"
        "endmodule\n")
    db.set_result("rsp", "best", {"sim_passed": True, "done": True})
    queue = [_QUEUE[0], {"name": "rsp", "tier": 0, "kind": "primitive", "rtl_target": "", "fabric": {"name": "x"}}]
    assert pg._block_rtl_target(str(tmp_path), queue[1]) == tmp_path / "rtl" / "interconnect" / "cs_fabric_x.v"
    asm, elab, snap = pg._shell_assemble(str(tmp_path), queue, tier=0)
    assert snap["real_blocks"] == ["rsp"] and asm.wiring_errors == [] and asm.wires == 5
    assert "cs_fabric_x u_rsp (" in asm.verilog
    assert elab.get("ok") is not False, elab


_FABRIC_X = ("module cs_fabric_x(clk, rst_n, s_q_addr, s_q_req_valid, s_q_req_gnt, s_q_rsp_valid, s_q_rdata);\n"
             "  input clk;\n  input rst_n;\n  input [7:0] s_q_addr;\n  input s_q_req_valid;\n"
             "  output s_q_req_gnt;\n  output s_q_rsp_valid;\n  output [7:0] s_q_rdata;\n"
             "  assign s_q_req_gnt = 1'b1;\n  assign s_q_rsp_valid = s_q_req_valid;\n  assign s_q_rdata = s_q_addr;\n"
             "endmodule\n")
# the checkpointed queue of the live run: the fabric spec carries no rtl_target
_PRIM_QUEUE = [{"name": "req", "tier": 1, "rtl_target": ""},
               {"name": "rsp", "tier": 0, "kind": "primitive", "rtl_target": "", "fabric": {"name": "x"}}]


def test_legacy_discovery_resolves_the_primitive_like_the_shell(tmp_path):
    """Q25: the Integration Check resolved by rtl_target / <block>.v only, so the
    fabric at rtl/interconnect/cs_fabric_<name>.v was 'NO RTL FOUND' and
    integration ended fail-closed, while the shell had it as real RTL."""
    from orchestrator.langgraph.integration_helpers import discover_block_rtl, merge_block_specs, missing_from
    (tmp_path / "rtl" / "interconnect").mkdir(parents=True)
    fabric = tmp_path / "rtl" / "interconnect" / "cs_fabric_x.v"
    fabric.write_text(_FABRIC_X)
    (tmp_path / "rtl" / "req.v").write_text("module req(input wire clk);\nendmodule\n")
    passed = merge_block_specs([{"name": "req", "success": True}, {"name": "rsp", "success": True}], _PRIM_QUEUE)
    rtl = discover_block_rtl(str(tmp_path), passed)
    assert rtl == {"req": str(tmp_path / "rtl" / "req.v"), "rsp": str(fabric)}
    assert missing_from(rtl, passed) == []
    assert rtl["rsp"] == str(pg._block_rtl_target(str(tmp_path), _PRIM_QUEUE[1]))
    fabric.unlink()      # not materialized: still unresolved (fail-closed), never a guess
    assert missing_from(discover_block_rtl(str(tmp_path), passed), passed) == ["rsp"]


def test_integration_check_adopts_the_shell_with_primitive_and_top_named_block(tmp_path, monkeypatch):
    """Q25, node level: the deterministic final top is the shell. A block named
    like the declared top is instantiated as its renamed <top>_core copy and the
    fabric as its generated module; adoption must list those files (not the
    block's own `module <top>`) and expect those module names."""
    import json
    import shutil

    import pytest
    if not (shutil.which("verilator") and shutil.which("yosys")):
        pytest.skip("verilator/yosys not on PATH")
    monkeypatch.setenv("CORESMITH_DETERMINISTIC_TOP", "1")
    monkeypatch.setenv("CORESMITH_TOP_MODULE", "req")      # the pad-adapter shape: block req is the boundary
    _project(tmp_path)
    (tmp_path / "rtl" / "interconnect").mkdir(parents=True)
    fabric = tmp_path / "rtl" / "interconnect" / "cs_fabric_x.v"
    fabric.write_text(_FABRIC_X)
    (tmp_path / "rtl" / "req.v").write_text(
        "module req(input wire clk, input wire rst_n, output wire [7:0] m_q_addr, "
        "output wire m_q_req_valid, input wire m_q_req_gnt, input wire m_q_rsp_valid, "
        "input wire [7:0] m_q_rdata);\n  assign m_q_addr = 0; assign m_q_req_valid = 0;\nendmodule\n")
    state = {"project_root": str(tmp_path), "block_queue": _PRIM_QUEUE,
             "completed_blocks": [{"name": "req", "success": True}, {"name": "rsp", "success": True}]}
    res = asyncio.run(pg._prepare_integration_check(state))["integration_result"]
    assert res.get("deterministic_top_assembled"), res
    assert res["top_module"] == "req" and res["design_name"] == "req"
    core = tmp_path / ".coresmith" / "shell" / "req_core.v"
    assert res["block_rtl_paths"] == {"req": str(core), "rsp": str(fabric)}
    rec = json.loads((tmp_path / ".coresmith" / "candidate.json").read_text())
    assert rec["top_module"] == "req" and {"req_core", "cs_fabric_x"} <= set(rec["elaborated_cells"])
    assert str((tmp_path / "rtl" / "req.v").resolve()) not in rec["sources"]
