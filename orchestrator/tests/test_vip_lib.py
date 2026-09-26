# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""A2: one deterministic VIP per contract edge, in the conformance gate's
port spelling, asserting the contract's timing."""
from __future__ import annotations

import json

from orchestrator.langgraph import vip_lib as V


def _req_resp():
    return {"edge_id": "l1d__m_coh_req__to__cc__s_l1d_req", "producer_block": "l1d",
            "producer_port": "m_coh_req", "consumer_block": "cc", "consumer_port": "s_l1d_req",
            "handshake_protocol": "req_resp", "data_width_bits": 29,
            "fields": [{"name": "type", "msb": 28, "lsb": 26, "width": 3},
                       {"name": "line_addr", "msb": 25, "lsb": 0, "width": 26}],
            "sideband_signals": [{"name": "req_valid", "width": 1, "direction": "producer->consumer"},
                                 {"name": "req_gnt", "width": 1, "direction": "consumer->producer"},
                                 {"name": "rsp_valid", "width": 1, "direction": "consumer->producer"},
                                 {"name": "rsp_data", "width": 64, "direction": "consumer->producer"}],
            "timing": {"req_to_rsp_cycles": {"exact": 1}}}


def _stream():
    return {"edge_id": "a__m_axis_px__to__b__s_axis_px", "producer_block": "a",
            "producer_port": "m_axis_px", "consumer_block": "b", "consumer_port": "s_axis_px",
            "handshake_protocol": "axi_stream", "data_width_bits": 16,
            "fields": [{"name": "tdata", "width": 16, "msb": 15, "lsb": 0}],
            "sideband_signals": [{"name": "tlast"}], "timing": {"valid_to_ready_max_stall": 4}}


def _valid_only():
    return {"edge_id": "cp__m_start__to__tre__s_start", "producer_block": "cp",
            "producer_port": "m_start", "consumer_block": "tre", "consumer_port": "s_start",
            "handshake_protocol": "valid_only", "data_width_bits": 8,
            "fields": [{"name": "tile", "width": 8, "msb": 7, "lsb": 0}]}


class TestContractModel:
    def test_req_resp_roles_and_ports(self):
        ec = V.EdgeContract.from_contract(_req_resp())
        assert (ec.valid_signal, ec.ready_signal, ec.response_valid_signal) == ("req_valid", "req_gnt", "rsp_valid")
        assert [s.name for s in ec.payload_signals] == ["type", "line_addr"]
        assert [s.name for s in ec.response_signals] == ["rsp_data"]
        c = ec.side("consumer")
        assert c.block == "cc" and c.channel == "s_l1d_req"
        assert c.port("line_addr") == "s_l1d_req_line_addr" and c.port("rsp_valid") == "s_l1d_req_rsp_valid"
        p = ec.side("producer")
        assert p.port("req_gnt") == "m_coh_req_req_gnt"
        assert ec.timing["req_to_rsp_cycles"] == {"min": 1, "max": 1, "exact": 1}

    def test_stream_defaults_are_filled(self):
        ec = V.EdgeContract.from_contract(_stream())
        assert ec.timing["valid_hold_until_ready"] is True
        assert ec.last_signal == "tlast" and ec.ready_signal == "tready"
        assert [s.name for s in ec.payload_signals] == ["tdata"]

    def test_fingerprint_tracks_timing_and_signals(self):
        a = V.EdgeContract.from_contract(_req_resp()).fingerprint()
        e = _req_resp()
        e["timing"] = {"req_to_rsp_cycles": {"exact": 2}}
        assert V.EdgeContract.from_contract(e).fingerprint() != a
        assert V.EdgeContract.from_contract(_req_resp()).fingerprint() == a


class TestCodegen:
    def test_modules_compile_and_are_deterministic(self):
        for edge in (_req_resp(), _stream(), _valid_only()):
            src = V.render_vip_module(edge)
            assert src == V.render_vip_module(edge)
            compile(src, "vip", "exec")
            assert "class Driver" in src and "class Monitor" in src and "async def assertions" in src
            assert "GENERATED" in src

    def test_module_constants_carry_the_contract(self):
        src = V.render_vip_module(_req_resp())
        ns: dict = {}
        # execute only the constant prelude (before the first class)
        exec(src.split("class VIPError")[0], ns)
        assert ns["TIMING"]["req_to_rsp_cycles"] == {"min": 1, "max": 1, "exact": 1}
        assert ns["SIDES"]["consumer"]["ports"]["type"] == "s_l1d_req_type"
        assert ns["PAYLOAD"] == ["type", "line_addr"] and ns["RESPONSE"] == ["rsp_data"]
        assert ns["VALID"] == "req_valid" and ns["READY"] == "req_gnt"

    def test_sva_bind_emits_the_timing_rules(self):
        sv = V.render_sva_bind(_req_resp(), "coherence_controller", "consumer")
        assert "bind coherence_controller" in sv
        assert "|-> ##1 s_l1d_req_rsp_valid" in sv and "a_reset_idle" in sv
        sv = V.render_sva_bind(_stream(), "b", "consumer")
        assert "##[0:4] s_axis_px_tready" in sv and "a_hold" in sv
        e = _req_resp()
        e["timing"] = {"req_to_rsp_cycles": {"min": 1, "max": 3}}
        assert "##[1:3]" in V.render_sva_bind(e, "cc", "consumer")
        sv = V.render_sva_bind(_valid_only(), "tre", "consumer")
        assert "a_reset_idle" in sv and "a_latency" not in sv


class TestFilesAndIndex:
    def test_write_all_and_lookup(self, tmp_path):
        idx = V.write_all_vips(tmp_path, [_req_resp(), _stream(), _valid_only()], contract_version=3)
        assert set(idx["edges"]) == {_req_resp()["edge_id"], _stream()["edge_id"], _valid_only()["edge_id"]}
        assert (tmp_path / ".coresmith" / "vip" / "l1d__m_coh_req__to__cc__s_l1d_req.py").exists()
        assert (tmp_path / ".coresmith" / "vip" / "__init__.py").exists()
        assert (tmp_path / ".coresmith" / "vip_index.json").exists()
        rows = V.vips_for_block(tmp_path, "cc")
        assert len(rows) == 1 and rows[0]["role"] == "consumer" and rows[0]["channel"] == "s_l1d_req"
        assert rows[0]["sva"].endswith("__consumer_sva.sv")
        assert {r["role"] for r in V.vips_for_block(tmp_path, "l1d")} == {"producer"}
        assert V.vips_for_block(tmp_path, "nobody") == []
        assert json.loads((tmp_path / ".coresmith" / "vip_index.json").read_text())["contract_version"] == 3

    def test_bad_edge_does_not_lose_the_rest(self, tmp_path):
        idx = V.write_all_vips(tmp_path, [_stream(), {"edge_id": "broken", "fields": 5}])
        assert "error" in idx["edges"]["broken"] and _stream()["edge_id"] in idx["edges"]


class TestLint:
    def test_tb_must_import_every_required_vip(self):
        req = [{"edge_id": "e1", "module": "e1_mod", "role": "consumer"},
               {"edge_id": "e2", "module": "e2_mod", "role": "producer"}]
        tb = "from vip.e1_mod import Driver, Monitor\n"
        probs = V.lint_tb_imports(tb, req)
        assert len(probs) == 1 and "e2" in probs[0] and "vip.e2_mod" in probs[0]
        assert V.lint_tb_imports(tb + "import vip.e2_mod as e2\n", req) == []
        assert V.lint_tb_imports('m = vip_load("e2_mod")\n' + tb, req) == []
        assert V.lint_tb_imports("", []) == []
