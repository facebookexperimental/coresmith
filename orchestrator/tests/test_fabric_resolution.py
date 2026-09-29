# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""B1: the architecture graph makes the SoC bus a generated primitive."""
from __future__ import annotations

import asyncio
import json

from orchestrator.architecture.specialists import fabric_resolution as FR
from orchestrator.langgraph import architecture_graph as ag
from orchestrator.state_store.project_db import open_project

_FABRIC = {"name": "soc", "data_width": 32, "addr_width": 32,
           "masters": [{"name": "hart0", "id_width": 4}, {"name": "dma", "id_width": 4}],
           "slaves": [{"name": "ram", "protocol": "axi4", "base": "0x80000000", "size": "0x1000000"},
                      {"name": "uart", "protocol": "apb", "base": "0x10000000", "size": "0x1000"}]}


def _declared():
    return {"blocks": [{"name": "hart0", "tier": 3}, {"name": "dma", "tier": 2},
                       {"name": "ram", "tier": 1}, {"name": "uart", "tier": 1},
                       {"name": "fabric", "kind": "primitive", "primitive": "cs_fabric", "fabric": _FABRIC}],
            "connections": [{"from": "hart0", "to": "fabric", "handshake_protocol": "req_resp"},
                            {"from": "dma", "to": "fabric", "handshake_protocol": "req_resp"},
                            {"from": "fabric", "to": "ram", "handshake_protocol": "mem_write"},
                            {"from": "fabric", "to": "uart", "handshake_protocol": "req_resp"},
                            {"from": "hart0", "to": "dma", "handshake_protocol": "valid_only"}]}


def _hand_written():
    return {"blocks": [{"name": "hart0"}, {"name": "gpu"},
                       {"name": "axi_arbiter", "description": "muxes the harts and the GPU onto the AXI port"},
                       {"name": "ram"}],
            "connections": [{"from": "hart0", "to": "axi_arbiter", "handshake_protocol": "req_resp"},
                            {"from": "gpu", "to": "axi_arbiter", "handshake_protocol": "req_resp"},
                            {"from": "axi_arbiter", "to": "ram", "handshake_protocol": "req_resp"}]}


class TestResolve:
    def test_declared_fabric_is_normalised_and_edges_typed(self):
        r = FR.resolve(_declared())
        assert r["fabrics"] == ["fabric"] and not r["errors"] and not r["ambiguous"]
        d = r["diagram"]
        fab = next(b for b in d["blocks"] if b["name"] == "fabric")
        assert fab["tier"] == 0 and fab["rtl_target"] == "rtl/interconnect/cs_fabric_soc.v"
        assert fab["golden_exempt"] is True and fab["fabric"]["slaves"][1]["base"] == 0x10000000
        edges = {(c["from"], c["to"]): c for c in d["connections"]}
        assert edges[("hart0", "fabric")]["handshake_protocol"] == "axi4"
        assert edges[("hart0", "fabric")]["to_port"] == "s_hart0"
        assert edges[("fabric", "uart")]["handshake_protocol"] == "apb"
        assert edges[("fabric", "uart")]["from_port"] == "m_uart" and edges[("fabric", "uart")]["to_port"] == "s_apb"
        assert edges[("fabric", "ram")]["handshake_protocol"] == "axi4"
        assert edges[("hart0", "dma")]["handshake_protocol"] == "valid_only"   # untouched

    def test_invalid_spec_is_an_error(self):
        doc = _declared()
        doc["blocks"][-1]["fabric"] = {**_FABRIC, "slaves": [{"name": "ram", "base": "0x1000", "size": "0x3000"}]}
        r = FR.resolve(doc)
        assert r["errors"] and "power of two" in r["errors"][0]["errors"][0]

    def test_hand_written_interconnect_is_ambiguous_with_a_proposal(self):
        r = FR.resolve(_hand_written())
        assert not r["fabrics"] and len(r["ambiguous"]) == 1
        a = r["ambiguous"][0]
        assert a["block"] == "axi_arbiter"
        assert [m["name"] for m in a["proposed_spec"]["masters"]] == ["gpu", "hart0"]
        assert a["proposed_spec"]["slaves"][0]["name"] == "ram" and a["proposed_spec"]["slaves"][0]["base"] is None

    def test_shared_target_without_any_fabric_is_ambiguous(self):
        doc = {"blocks": [{"name": "a"}, {"name": "b"}, {"name": "mem"}],
               "connections": [{"from": "a", "to": "mem", "handshake_protocol": "mem_write"},
                               {"from": "b", "to": "mem", "handshake_protocol": "req_resp"}]}
        r = FR.resolve(doc)
        assert r["ambiguous"][0]["block"] == "mem" and "2 initiators" in r["ambiguous"][0]["reason"]

    def test_plain_streaming_design_is_untouched(self):
        doc = {"blocks": [{"name": "a"}, {"name": "b"}],
               "connections": [{"from": "a", "to": "b", "handshake_protocol": "axi_stream"}]}
        r = FR.resolve(doc)
        assert not r["ambiguous"] and not r["fabrics"] and r["diagram"]["connections"] == doc["connections"]


class TestNode:
    def test_graph_and_routing(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_FABRIC_RESOLUTION", "1")
        g = ag.build_architecture_graph()
        assert "Fabric Resolution" in g.nodes
        assert ag._post_diagram_gate_target() == "Fabric Resolution"
        assert ag.review_diagram({"block_diagram": {"blocks": [{"name": "a"}], "questions": []}}) == "Fabric Resolution"
        monkeypatch.setenv("CORESMITH_FABRIC_RESOLUTION", "0")
        assert ag._post_diagram_gate_target() != "Fabric Resolution"

    def test_declared_fabric_rewrites_state_and_db(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_FABRIC_RESOLUTION", "1")
        monkeypatch.delenv("CORESMITH_ENABLE_CHIP_LEAD", raising=False)
        db = open_project(tmp_path)
        out = asyncio.run(ag.fabric_resolution_node({"project_root": str(tmp_path), "round": 1,
                                                     "block_diagram": _declared()}))
        assert out["fabric_resolution"]["fabrics"] == ["fabric"]
        specs = {b["name"]: b for b in db.block_specs()}
        assert specs["fabric"]["kind"] == "primitive" and specs["fabric"]["fabric"]["name"] == "soc"
        assert specs["fabric"]["tier"] == "0" or specs["fabric"]["tier"] == 0

    def test_ambiguous_parks_and_accepts_a_spec(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_FABRIC_RESOLUTION", "1")
        monkeypatch.delenv("CORESMITH_ENABLE_CHIP_LEAD", raising=False)
        parks = []

        def fake_interrupt(payload):
            parks.append(payload)
            spec = dict(payload["ambiguous"][0]["proposed_spec"])
            spec["slaves"] = [{"name": "ram", "protocol": "axi4", "base": "0x80000000", "size": "0x1000000"}]
            return {"action": "accept", "feedback": json.dumps({"block": "axi_arbiter", "fabric": spec})}
        monkeypatch.setattr(ag, "interrupt", fake_interrupt)
        out = asyncio.run(ag.fabric_resolution_node({"project_root": str(tmp_path), "round": 1,
                                                     "block_diagram": _hand_written()}))
        assert parks and parks[0]["type"] == "fabric_ambiguous"
        assert out["fabric_resolution"]["fabrics"] == ["axi_arbiter"]
        fab = next(b for b in out["block_diagram"]["blocks"] if b["name"] == "axi_arbiter")
        assert fab["kind"] == "primitive" and {m["name"] for m in fab["fabric"]["masters"]} == {"gpu", "hart0"}
        edges = {(c["from"], c["to"]): c["handshake_protocol"] for c in out["block_diagram"]["connections"]}
        assert edges[("hart0", "axi_arbiter")] == "axi4" and edges[("axi_arbiter", "ram")] == "axi4"

    def test_skip_keeps_the_diagram(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_FABRIC_RESOLUTION", "1")
        monkeypatch.delenv("CORESMITH_ENABLE_CHIP_LEAD", raising=False)
        monkeypatch.setattr(ag, "interrupt", lambda payload: {"action": "skip"})
        out = asyncio.run(ag.fabric_resolution_node({"project_root": str(tmp_path), "round": 1,
                                                     "block_diagram": _hand_written()}))
        assert out["fabric_resolution"]["unresolved"] == ["axi_arbiter"]
        assert ag.route_after_fabric_resolution(out) != "Abort"
