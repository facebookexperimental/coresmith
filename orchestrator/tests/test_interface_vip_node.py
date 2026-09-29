# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""A2: the architecture graph renders the interface VIPs right after the
contracts freeze, from the project database."""
from __future__ import annotations

import asyncio
import json

from orchestrator.langgraph import architecture_graph as ag
from orchestrator.state_store.project_db import open_project

_EDGE = {"edge_id": "a__m_q__to__b__s_q", "producer_block": "a", "producer_port": "m_q",
         "consumer_block": "b", "consumer_port": "s_q", "handshake_protocol": "req_resp",
         "data_width_bits": 8, "fields": [{"name": "addr", "width": 8, "msb": 7, "lsb": 0}],
         "sideband_signals": [{"name": "req_valid"}, {"name": "rsp_valid"}, {"name": "rdata", "width": 8}],
         "timing": {"req_to_rsp_cycles": {"exact": 2}}}


def _state(tmp_path):
    return {"project_root": str(tmp_path), "round": 1,
            "block_diagram": {"blocks": [{"name": "a"}, {"name": "b"}]},
            "interface_contracts": {"contracts": [_EDGE]}}


def test_graph_wires_the_node():
    g = ag.build_architecture_graph()
    assert "Interface VIP" in g.nodes
    assert ag.route_after_interface_definition.__edge_labels__["Interface VIP"] == "OK"
    assert ag.route_after_interface_definition({"constraint_result": None}) == "Interface VIP"


def test_node_renders_from_the_database(tmp_path, monkeypatch):
    monkeypatch.setenv("CORESMITH_INTERFACE_VIP", "1")
    db = open_project(tmp_path)
    db.import_contracts({"contracts": [_EDGE]})
    out = asyncio.run(ag.interface_vip_node(_state(tmp_path)))
    assert out["interface_vip_index"]["edges"] == 1 and out["interface_vip_index"]["errors"] == {}
    assert out["interface_vip_index"]["contract_version"] == db.contracts_version()
    py = tmp_path / ".coresmith" / "vip" / "a__m_q__to__b__s_q.py"
    assert py.exists() and "$past(s_q_req_valid, 2) |-> s_q_rsp_valid" in (tmp_path / ".coresmith" / "vip" / "a__m_q__to__b__s_q__consumer_sva.sv").read_text()
    idx = json.loads((tmp_path / ".coresmith" / "vip_index.json").read_text())
    assert idx["edges"]["a__m_q__to__b__s_q"]["consumer"] == "b"


def test_node_falls_back_to_state_and_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("CORESMITH_INTERFACE_VIP", "1")
    out = asyncio.run(ag.interface_vip_node(_state(tmp_path)))
    assert out["interface_vip_index"]["edges"] == 1
    monkeypatch.setenv("CORESMITH_INTERFACE_VIP", "0")
    assert asyncio.run(ag.interface_vip_node(_state(tmp_path))) == {}
