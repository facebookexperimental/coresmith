# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""B1: a primitive block is materialized (generated) instead of authored, and
the bus families flow through contracts, conformance and VIPs."""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from orchestrator.architecture.specialists import contract_timing as ct
from orchestrator.langgraph import contract_conformance as cc
from orchestrator.langgraph import pipeline_graph as pg
from orchestrator.langgraph import vip_lib as V
from orchestrator.state_store.project_db import open_project

_FAB = {"name": "soc", "masters": [{"name": "hart0"}],
        "slaves": [{"name": "ram", "protocol": "axi4", "base": 0x80000000, "size": 0x1000000}]}
_BLOCK = {"name": "fabric", "kind": "primitive", "primitive": "cs_fabric", "tier": 0, "fabric": _FAB,
          "rtl_target": "rtl/interconnect/cs_fabric_soc.v", "testbench": "tb/cocotb/test_cs_fabric_soc.py"}


class TestQueue:
    def test_block_specs_carry_the_primitive_fields(self, tmp_path):
        db = open_project(tmp_path)
        db.import_block_diagram({"blocks": [_BLOCK, {"name": "hart0", "tier": 3}], "connections": []})
        specs = {b["name"]: b for b in db.block_specs()}
        assert specs["fabric"]["kind"] == "primitive" and specs["fabric"]["fabric"]["name"] == "soc"
        assert "kind" not in specs["hart0"]

    def test_fan_out_preserves_the_generated_testbench(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pg, "_stale_specs", lambda pr, names: [])
        monkeypatch.setattr(pg, "_uarch_single_context_enabled", lambda: False)
        state = {"project_root": str(tmp_path), "target_clock_mhz": 50.0, "max_attempts": 3,
                 "block_queue": [_BLOCK, {"name": "hart0", "tier": 0}], "tier_list": [0], "current_tier_index": 0}
        sends = {s.arg["current_block"]["name"]: s.arg for s in pg.fan_out_tier(state)}
        assert sends["fabric"]["preserve_testbench"] is True and sends["hart0"]["preserve_testbench"] is False


class TestMaterialize:
    def test_routing(self):
        g = pg.build_block_subgraph()
        assert "materialize_primitive" in g.nodes
        assert pg.route_after_init({"current_block": _BLOCK}) == "materialize_primitive"
        assert pg.route_after_init({"current_block": {"name": "x"}}) == "generate_uarch_spec"
        assert pg.route_after_materialize({}) == "generate_testbench"
        assert pg.route_after_materialize({"primitive_failed": True}) == "block_done"

    def test_node_writes_rtl_tb_and_spec(self, tmp_path, monkeypatch):
        import orchestrator.fabric as fabric

        def fake_generate(spec, out_dir, tb_dir=None, **kw):
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{spec.module_name}.v").write_text(f"module {spec.module_name}(); endmodule\n")
            tbd = Path(tb_dir or out_dir)
            tbd.mkdir(parents=True, exist_ok=True)
            (tbd / f"test_{spec.module_name}.py").write_text("# generated tb\n")
            return SimpleNamespace(module=spec.module_name, rtl_path=str(out / f"{spec.module_name}.v"),
                                   tb_path=str(tbd / f"test_{spec.module_name}.py"), cached=False,
                                   ports=[{"name": "clk", "dir": "input", "width": 1}])
        monkeypatch.setattr(fabric, "generate_fabric", fake_generate)
        db = open_project(tmp_path)
        db.import_block_diagram({"blocks": [_BLOCK], "connections": []})
        out = asyncio.run(pg.materialize_primitive_node({"project_root": str(tmp_path), "current_block": _BLOCK,
                                                         "attempt": 1}))
        assert out["lint_clean"] and out["preserve_testbench"] and out["uarch_approved"]
        assert (tmp_path / "rtl/interconnect/cs_fabric_soc.v").read_text().startswith("module cs_fabric_soc")
        assert (tmp_path / "tb/cocotb/test_cs_fabric_soc.py").exists()
        md = (tmp_path / "arch/uarch_specs/fabric.md").read_text()
        assert "GENERATED primitive" in md and "INV-FABRIC-DECODE-001" in md and "0x80000000" in md
        with db._conn() as conn:
            row = conn.execute("SELECT spec_contract_version FROM blocks WHERE name='fabric'").fetchone()
        assert row is not None and row["spec_contract_version"] is not None

    def test_invalid_spec_escalates(self, tmp_path):
        bad = {**_BLOCK, "fabric": {**_FAB, "slaves": []}}
        out = asyncio.run(pg.materialize_primitive_node({"project_root": str(tmp_path), "current_block": bad,
                                                         "attempt": 1}))
        assert out["primitive_failed"] and out["debug_action"] == "escalate"
        assert "at least one slave" in (tmp_path / ".coresmith/blocks/fabric/previous_error.txt").read_text()

    def test_primitive_skips_assertion_stage_and_vip_lint(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_ASSERTION_STAGE", "1")
        out = asyncio.run(pg.assertion_check_node({"project_root": str(tmp_path), "current_block": _BLOCK,
                                                   "rtl_path": str(tmp_path / "x.v")}))
        assert out == {"assertion_ok": True}


class TestBusFamilies:
    _EDGE = {"edge_id": "hart0__m_axi__to__fabric__s_hart0", "producer_block": "hart0", "producer_port": "m_axi",
             "consumer_block": "fabric", "consumer_port": "s_hart0", "handshake_protocol": "axi4",
             "data_width_bits": 64, "bus_params": {"addr_width": 32, "id_width": 4}}

    def test_signal_specs_expand_the_amba_channel_set(self):
        names = {s["name"]: s for s in cc.signal_specs(self._EDGE)}
        assert {"awvalid", "awready", "wdata", "wstrb", "bresp", "rlast", "arid"} <= set(names)
        assert names["wdata"]["width"] == "64" and names["wstrb"]["width"] == "8" and names["awid"]["width"] == "4"
        assert names["awready"]["dir"] == "consumer->producer"
        rows = cc.signal_specs({**self._EDGE, "handshake_protocol": "apb", "consumer_port": "s_apb"})
        assert {"psel", "penable", "pready", "prdata", "pslverr"} <= {r["name"] for r in rows}

    def test_bus_timing_defaults_and_vip(self):
        e = dict(self._EDGE)
        ct.normalize_timing(e)
        assert e["timing"]["valid_hold_until_ready"] is True and e["timing"]["ordering"] == "in_order"
        src = V.render_vip_module(e)
        compile(src, "v", "exec")
        assert "AxiMaster" in src and "HANDSHAKES = " in src and "class ApbMaster" in src
        ec = V.EdgeContract.from_contract(e)
        assert ec.side("consumer").port("awvalid") == "s_hart0_awvalid"

    def test_interface_definition_accepts_bus_edges(self):
        from orchestrator.architecture.specialists.interface_definition import _validate_contracts
        c = [{**self._EDGE, "fields": [], "timing": {}}]
        v, _ = _validate_contracts({"contracts": c}, [{"from": "hart0", "to": "fabric"}])
        assert v["contract_violations"] == []
