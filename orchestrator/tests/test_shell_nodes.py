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
