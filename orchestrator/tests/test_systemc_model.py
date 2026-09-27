# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""B2: block models that follow the skeleton conventions assemble into a
SoC model that builds and runs (real SystemC when available)."""
from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.systemc_model import (
    build,
    channel_binding,
    detect,
    render_block_skeleton,
    render_soc_model,
    render_soc_top_header,
    smoke,
    write_build,
)

_E1 = {"edge_id": "req__m_q__to__rsp__s_q", "producer_block": "req", "producer_port": "m_q",
       "consumer_block": "rsp", "consumer_port": "s_q", "handshake_protocol": "req_resp",
       "data_width_bits": 8, "timing": {"req_to_rsp_cycles": {"exact": 1}}}
_E2 = {"edge_id": "rsp__m_axis_o__to__sink__s_axis_o", "producer_block": "rsp", "producer_port": "m_axis_o",
       "consumer_block": "sink", "consumer_port": "s_axis_o", "handshake_protocol": "axi_stream",
       "data_width_bits": 8}
_E3 = {"edge_id": "sink__m_led__to__pads__s_led", "producer_block": "sink", "producer_port": "m_led",
       "consumer_block": "pads", "consumer_port": "s_led", "handshake_protocol": "static", "data_width_bits": 1}
_EDGES = [_E1, _E2, _E3]
_BLOCKS = ["req", "rsp", "sink", "pads"]

_REQ = '''#include "req_model.h"
req_model::req_model(sc_core::sc_module_name n) : sc_module(n), m_q("m_q"), reads(0) { SC_THREAD(run); sensitive << clk.pos(); }
void req_model::reset() { reads = 0; }
void req_model::dump_state(std::ostream& os) const { os << "reads=" << reads << " last=" << (int)last << "\\n"; }
void req_model::run() {
  while (!rst_n.read()) wait();
  for (int i = 0; i < 4; ++i) {
    wait();
    uint8_t d = 0; sc_core::sc_time delay = sc_core::SC_ZERO_TIME;
    cs_transact(m_q, false, i, &d, 1, delay);
    last = d; ++reads;
  }
}
'''
_RSP = '''#include "rsp_model.h"
rsp_model::rsp_model(sc_core::sc_module_name n) : sc_module(n), s_q("s_q"), served(0) {
  s_q.register_b_transport(this, &rsp_model::b_transport_s_q); SC_THREAD(run); sensitive << clk.pos(); }
void rsp_model::reset() { served = 0; }
void rsp_model::dump_state(std::ostream& os) const { os << "served=" << served << "\\n"; }
void rsp_model::b_transport_s_q(tlm::tlm_generic_payload& t, sc_core::sc_time& delay) {
  uint8_t v = (uint8_t)(t.get_address() + 1); std::memcpy(t.get_data_ptr(), &v, 1);
  t.set_response_status(tlm::TLM_OK_RESPONSE); delay += cs_clock_period(); ++served;
  m_axis_o.write((cs_beat_t)v);
}
void rsp_model::run() { while (true) wait(); }
'''
_SINK = '''#include "sink_model.h"
sink_model::sink_model(sc_core::sc_module_name n) : sc_module(n), beats(0) { SC_THREAD(run); sensitive << clk.pos(); }
void sink_model::reset() { beats = 0; }
void sink_model::dump_state(std::ostream& os) const { os << "beats=" << beats << "\\n"; }
void sink_model::run() { while (true) { cs_beat_t b = s_axis_o.read(); ++beats; m_led.write(b & 1); } }
'''
_PADS = '''#include "pads_model.h"
pads_model::pads_model(sc_core::sc_module_name n) : sc_module(n) { SC_THREAD(run); sensitive << clk.pos(); }
void pads_model::reset() {}
void pads_model::dump_state(std::ostream& os) const { os << "led=" << s_led.read() << "\\n"; }
void pads_model::run() { while (true) wait(); }
'''


def _extra_members(name):
    return {"req": "  unsigned reads; uint8_t last = 0;\n", "rsp": "  unsigned served;\n",
            "sink": "  unsigned beats;\n", "pads": ""}[name]


class TestConventions:
    def test_channel_binding_per_family(self):
        b = channel_binding(_E1)
        assert b["kind"] == "socket" and b["producer_member"] == "m_q" and b["consumer_member"] == "s_q"
        assert b["latency_cycles"] == 1
        assert channel_binding(_E2)["kind"] == "fifo" and channel_binding(_E3)["kind"] == "signal"

    def test_skeleton_lists_every_bound_member(self):
        h = render_block_skeleton("rsp", _EDGES)
        assert "simple_target_socket<rsp_model> s_q" in h and "sc_fifo_out<cs_beat_t> m_axis_o" in h
        assert "b_transport_s_q" in h and "latency 1 cycle" in h
        assert "sc_in<cs_word_t> s_led" in render_block_skeleton("pads", _EDGES)

    def test_soc_model_binds_every_edge(self):
        src = render_soc_top_header(_BLOCKS, _EDGES, top_name="tiny")
        assert "SC_MODULE(soc_model_top)" in src and "u_req.m_q.bind(u_rsp.s_q)" in src
        assert "sc_fifo<cs_beat_t> f_0" in src and "u_sink.s_axis_o(f_0)" in src
        assert "sc_signal<cs_word_t> s_0" in src and "u_pads.s_led(s_0)" in src
        assert "reset_all()" in src and "dump_all(std::ostream& os)" in src
        drv = render_soc_model(_BLOCKS, _EDGES, top_name="tiny")
        assert '#include "soc_model_top.h"' in drv and "SOC_MODEL_SMOKE_OK" in drv

    def test_static_fan_out_shares_one_signal(self):
        # An sc_out binds exactly once: two consumers of the same static
        # producer port must share the signal (E109 on the first SoC run).
        fan = {**_E3, "edge_id": "sink__m_led__to__req__s_led", "consumer_block": "req", "consumer_port": "s_led"}
        src = render_soc_top_header(_BLOCKS, _EDGES + [fan], top_name="tiny")
        assert src.count("u_sink.m_led(") == 1
        assert "u_pads.s_led(s_0)" in src and "u_req.s_led(s_0)" in src and "s_1" not in src


@pytest.mark.slow
@pytest.mark.skipif(not detect()["ok"], reason="SystemC toolchain not available")
def test_hand_written_models_build_and_run(tmp_path):
    md = write_build(tmp_path, _BLOCKS, _EDGES, top_name="tiny")
    for b, body in (("req", _REQ), ("rsp", _RSP), ("sink", _SINK), ("pads", _PADS)):
        h = render_block_skeleton(b, _EDGES).replace("  void run();", _extra_members(b) + "  void run();")
        (md / f"{b}_model.h").write_text(h)
        (md / f"{b}_model.cpp").write_text(body)
    res = build(md)
    assert res["ok"], res["log"][-3000:]
    run = smoke(md, ns=500)
    assert run["ok"], run["log"][-3000:]
    assert "reads=4" in run["log"] and "served=4" in run["log"] and "beats=4" in run["log"]
    assert Path(md / "smoke.log").exists()


@pytest.mark.slow
@pytest.mark.skipif(not detect()["ok"], reason="SystemC toolchain not available")
def test_generated_fabric_router_routes_and_decerrs(tmp_path):
    from orchestrator.fabric import FabricMaster, FabricSlave, FabricSpec
    from orchestrator.systemc_model.fabric_model import render_fabric_model
    spec = FabricSpec(name="soc", masters=[FabricMaster("hart0")],
                      slaves=[FabricSlave("ram", "axi4", 0x8000_0000, 0x1000)])
    edges = [{"edge_id": "hart0__m_axi__to__fabric__s_hart0", "producer_block": "hart0", "producer_port": "m_axi",
              "consumer_block": "fabric", "consumer_port": "s_hart0", "handshake_protocol": "axi4", "data_width_bits": 32},
             {"edge_id": "fabric__m_ram__to__ram__s_axi", "producer_block": "fabric", "producer_port": "m_ram",
              "consumer_block": "ram", "consumer_port": "s_axi", "handshake_protocol": "axi4", "data_width_bits": 32}]
    blocks = ["hart0", "fabric", "ram"]
    md = write_build(tmp_path, blocks, edges, top_name="fab")
    h, c = render_fabric_model("fabric", spec, edges)
    assert "simple_target_socket_tagged_optional" in h and "simple_initiator_socket_optional" in h
    assert "s_cs_tester" in h and "s_cs_tester.register_b_transport" in c
    assert ".size() == 0" in c
    (md / "fabric_model.h").write_text(h)
    (md / "fabric_model.cpp").write_text(c)
    (md / "hart0_model.h").write_text(render_block_skeleton("hart0", edges).replace(
        "  void run();", "  unsigned ok = 0, bad = 0;\n  void run();"))
    (md / "hart0_model.cpp").write_text('''#include "hart0_model.h"
hart0_model::hart0_model(sc_core::sc_module_name n) : sc_module(n), m_axi("m_axi") { SC_THREAD(run); sensitive << clk.pos(); }
void hart0_model::reset() { ok = bad = 0; }
void hart0_model::dump_state(std::ostream& os) const { os << "ok=" << ok << " bad=" << bad << "\\n"; }
void hart0_model::run() { while (!rst_n.read()) wait(); wait();
  uint8_t v[4] = {1,2,3,4}; sc_core::sc_time d = sc_core::SC_ZERO_TIME;
  if (cs_transact(m_axi, true, 0x80000010ULL, v, 4, d) == tlm::TLM_OK_RESPONSE) ++ok; else ++bad;
  if (cs_transact(m_axi, false, 0x80000010ULL, v, 4, d) == tlm::TLM_OK_RESPONSE && v[0] == 1) ++ok; else ++bad;
  if (cs_transact(m_axi, false, 0x10000000ULL, v, 4, d) == tlm::TLM_ADDRESS_ERROR_RESPONSE) ++ok; else ++bad; }
''')
    (md / "ram_model.h").write_text(render_block_skeleton("ram", edges).replace(
        "  void run();", "  cs_mem mem;\n  void run();"))
    (md / "ram_model.cpp").write_text('''#include "ram_model.h"
ram_model::ram_model(sc_core::sc_module_name n) : sc_module(n), s_axi("s_axi") {
  s_axi.register_b_transport(this, &ram_model::b_transport_s_axi); SC_THREAD(run); sensitive << clk.pos(); }
void ram_model::reset() {}
void ram_model::dump_state(std::ostream& os) const { os << "bytes=" << mem.bytes.size() << "\\n"; }
void ram_model::b_transport_s_axi(tlm::tlm_generic_payload& t, sc_core::sc_time& d) {
  if (t.is_write()) mem.write(t.get_address(), t.get_data_ptr(), t.get_data_length());
  else mem.read(t.get_address(), t.get_data_ptr(), t.get_data_length());
  t.set_response_status(tlm::TLM_OK_RESPONSE); d += cs_clock_period(); }
void ram_model::run() { while (true) wait(); }
''')
    res = build(md)
    assert res["ok"], res["log"][-3000:]
    run = smoke(md, ns=300)
    assert run["ok"] and "ok=3 bad=0" in run["log"] and "routed=2" in run["log"] and "decerr=1" in run["log"], run["log"][-2000:]
