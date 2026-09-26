# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""A2-4: block DV consumes the generated interface VIPs -- the TB must import
them, the simulation build carries their SVA binds, and a revised contract
re-specs the block."""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from orchestrator.langgraph import pipeline_graph as pg
from orchestrator.langgraph import vip_lib as V
from orchestrator.state_store.project_db import open_project

_EDGE = {"edge_id": "req__m_q__to__rsp__s_q", "producer_block": "req", "producer_port": "m_q",
         "consumer_block": "rsp", "consumer_port": "s_q", "handshake_protocol": "req_resp",
         "data_width_bits": 8, "fields": [{"name": "addr", "width": 8, "msb": 7, "lsb": 0}],
         "sideband_signals": [{"name": "req_valid", "width": 1},
                              {"name": "rsp_valid", "width": 1, "direction": "consumer->producer"},
                              {"name": "rdata", "width": 8, "direction": "consumer->producer"}],
         "timing": {"req_to_rsp_cycles": {"exact": 1}}}


def _project(tmp_path):
    (tmp_path / "inputs").mkdir(exist_ok=True)
    (tmp_path / "inputs/task.yaml").write_text("chassis: none\n")
    db = open_project(tmp_path)
    db.import_contracts({"contracts": [_EDGE]})
    V.write_all_vips(tmp_path, [_EDGE], contract_version=db.contracts_version())
    return db


def _rtl(tmp_path, ports):
    f = tmp_path / "rsp.v"
    f.write_text("module rsp (\n  " + ",\n  ".join(f"input wire {p}" for p in ports) + "\n);\nendmodule\n")
    return f


def _state(tmp_path, rtl, tb):
    return {"current_block": {"name": "rsp", "testbench": str(Path(tb).relative_to(tmp_path))},
            "project_root": str(tmp_path), "attempt": 1, "rtl_path": str(rtl), "tb_path": str(tb),
            "force_regen_tb": False, "preserve_testbench": True}


class TestContractSlice:
    def test_slice_carries_edges_and_timing_summary(self, tmp_path):
        _project(tmp_path)
        path = pg._write_contract_slice(str(tmp_path), "rsp")
        data = json.loads(Path(path).read_text())
        assert data["block"] == "rsp" and data["edges"][0]["edge_id"] == _EDGE["edge_id"]
        assert "exactly 1 cycle" in data["edges"][0]["timing_summary"]


class TestTbLintGate:
    @pytest.mark.asyncio
    async def test_tb_without_the_vip_import_is_rejected_before_sim(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_INTERFACE_VIP", "1")
        monkeypatch.setenv("CORESMITH_CONTRACT_CONFORMANCE_GATE", "0")
        monkeypatch.setenv("CORESMITH_CONTRACT_PORT_GATE", "0")
        _project(tmp_path)
        rtl = _rtl(tmp_path, ["clk", "rst_n", "s_q_addr", "s_q_req_valid", "s_q_rsp_valid", "s_q_rdata"])
        tb = tmp_path / "tb" / "test_rsp.py"
        tb.parent.mkdir(parents=True)
        tb.write_text("# hand-modelled neighbour\nimport cocotb\n")
        calls = []
        with patch("orchestrator.langgraph.pipeline_graph.run_simulation",
                   lambda *a, **k: calls.append(a) or {"passed": True}):
            out = await pg.generate_testbench_node(_state(tmp_path, rtl, tb))
        assert out["sim_passed"] is False and out["phase"] == "tb" and not calls
        prev = (tmp_path / ".coresmith/blocks/rsp/previous_error.txt").read_text()
        assert "INTERFACE-VIP LINT" in prev and "vip.req__m_q__to__rsp__s_q" in prev
        # the diagnose fast path turns it into a TB regeneration, no LLM call
        st = {**_state(tmp_path, rtl, tb), "phase": "tb"}
        assert (await pg.diagnose_node(st))["debug_action"] == "retry_tb"

    @pytest.mark.asyncio
    async def test_tb_that_imports_the_vip_reaches_sim(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_INTERFACE_VIP", "1")
        monkeypatch.setenv("CORESMITH_CONTRACT_CONFORMANCE_GATE", "0")
        monkeypatch.setenv("CORESMITH_CONTRACT_PORT_GATE", "0")
        _project(tmp_path)
        rtl = _rtl(tmp_path, ["clk", "rst_n", "s_q_addr", "s_q_req_valid", "s_q_rsp_valid", "s_q_rdata"])
        tb = tmp_path / "tb" / "test_rsp.py"
        tb.parent.mkdir(parents=True)
        tb.write_text("from vip.req__m_q__to__rsp__s_q import Driver, Monitor, assertions, SIDES\n")
        from orchestrator.state_store.trust import capture_run_baseline
        capture_run_baseline(tmp_path)
        calls = []
        with patch("orchestrator.langgraph.pipeline_graph.run_simulation",
                   lambda *a, **k: calls.append(a) or {"passed": True}):
            out = await pg.generate_testbench_node(_state(tmp_path, rtl, tb))
        assert calls and out["sim_passed"] is True

    def test_lint_is_silent_without_vips_or_when_disabled(self, tmp_path, monkeypatch):
        assert pg._vip_tb_lint(str(tmp_path), "rsp", tmp_path / "nope.py") == []
        _project(tmp_path)
        tb = tmp_path / "t.py"
        tb.write_text("# no import\n")
        monkeypatch.setenv("CORESMITH_INTERFACE_VIP", "1")
        assert pg._vip_tb_lint(str(tmp_path), "rsp", tb)
        monkeypatch.setenv("CORESMITH_INTERFACE_VIP", "0")
        assert pg._vip_tb_lint(str(tmp_path), "rsp", tb) == []


class TestSimulationBuild:
    def test_makefile_carries_sva_binds_and_pythonpath(self, tmp_path, monkeypatch):
        import orchestrator.langgraph.pipeline_helpers as ph
        monkeypatch.setattr(ph, "PROJECT_ROOT", tmp_path)
        monkeypatch.setenv("CORESMITH_VIP_SVA_BIND", "1")
        _project(tmp_path)
        rtl = tmp_path / "rsp.v"
        rtl.write_text("module rsp(input clk); endmodule\n")
        tb = tmp_path / "test_rsp.py"
        tb.write_text("# tb\n")
        monkeypatch.setattr(ph, "apply_build_fingerprint", lambda *a, **k: None)
        monkeypatch.setattr(ph, "_normalize_cocotb_timing_keywords", lambda *a, **k: None)
        monkeypatch.setattr(ph, "create_golden_model_wrapper", lambda *a, **k: None)
        captured = {}

        class _Stop(Exception):
            pass

        def fake_popen(cmd, *a, **k):
            captured["makefile"] = open(os.path.join(cmd[2], "Makefile")).read()
            captured["env"] = k.get("env") or {}
            raise _Stop()
        monkeypatch.setattr(ph.subprocess, "Popen", fake_popen)
        with pytest.raises(_Stop):
            ph.run_simulation({"name": "rsp"}, str(rtl), str(tb), project_root=str(tmp_path))
        mk = captured["makefile"]
        assert "req__m_q__to__rsp__s_q__consumer_sva.sv" in mk and "--assert" in mk
        assert "__producer_sva.sv" not in mk               # only this block's side
        assert str(tmp_path / ".coresmith") in captured["env"]["PYTHONPATH"]
        monkeypatch.setenv("CORESMITH_VIP_SVA_BIND", "0")
        with pytest.raises(_Stop):
            ph.run_simulation({"name": "rsp"}, str(rtl), str(tb), project_root=str(tmp_path))
        assert "--assert" not in captured["makefile"]


class TestStaleSpecRefresh:
    def test_fan_out_forces_respec_of_stale_blocks(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pg, "_stale_specs",
                            lambda pr, names: [{"block": "rsp", "reason": "revised"}])
        monkeypatch.setattr(pg, "_uarch_single_context_enabled", lambda: True)
        state = {"project_root": str(tmp_path), "target_clock_mhz": 50.0, "max_attempts": 3,
                 "block_queue": [{"name": "req", "tier": 1}, {"name": "rsp", "tier": 1}],
                 "tier_list": [1], "current_tier_index": 0}
        sends = pg.fan_out_tier(state)
        reuse = {s.arg["current_block"]["name"]: s.arg["reuse_spec"] for s in sends}
        assert reuse == {"req": True, "rsp": False}

    def test_stale_specs_reads_the_db(self, tmp_path):
        db = open_project(tmp_path)
        db.import_block_diagram({"blocks": [{"name": "rsp", "tier": 1}], "connections": []})
        db.import_contracts({"contracts": [_EDGE]})
        db.stamp_block_spec("rsp")
        assert pg._stale_specs(str(tmp_path), ["rsp"]) == []
        e2 = dict(_EDGE)
        e2["timing"] = {"req_to_rsp_cycles": {"exact": 2}}
        db.import_contracts({"contracts": [e2]})
        assert [d["block"] for d in pg._stale_specs(str(tmp_path), ["rsp"])] == ["rsp"]
