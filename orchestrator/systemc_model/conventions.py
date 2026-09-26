# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""How a contract edge maps to SystemC TLM constructs (loosely timed).

Every block model is ``SC_MODULE(<block>_model)`` and exposes, per contract
channel it owns, a member named after the channel base (``m_q``, ``s_axi``,
``m_axis_px``) of the type below. The assembler binds producer to consumer
by these names, so a model that follows the skeleton is wireable by
construction.

  family                     producer member                      consumer member
  req_resp / mem_write /     tlm_utils::simple_initiator_socket   tlm_utils::simple_target_socket
  valid_only / axi4 /        <T>                                  <T>   (b_transport)
  axi_lite / apb
  axi_stream / srdy_drdy     sc_core::sc_fifo_out<cs_beat_t>      sc_core::sc_fifo_in<cs_beat_t>
  static                     sc_core::sc_out<cs_word_t>           sc_core::sc_in<cs_word_t>

``cs_beat_t`` / ``cs_word_t`` are ``uint64_t`` payloads (wider buses carry
several words; the generic payload's data pointer carries bytes). Latency
comes from the contract's ``timing`` object: a target adds
``req_to_rsp_cycles.exact`` (or ``min``) x clock period to the ``b_transport``
delay; streams sleep one clock per beat.
"""
from __future__ import annotations

import re

from orchestrator.langgraph.contract_conformance import channel_base

SOCKET_FAMILIES = ("req_resp", "mem_write", "valid_only", "axi4", "axi_lite", "apb")
FIFO_FAMILIES = ("axi_stream", "srdy_drdy")
STATIC_FAMILIES = ("static",)


def _ident(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", s or "")


def model_name(block: str) -> str:
    return f"{_ident(block)}_model"


def channel_binding(edge: dict) -> dict:
    """The names/types both ends of ``edge`` must expose."""
    fam = str(edge.get("handshake_protocol") or "valid_only").lower()
    pchan = _ident(channel_base(str(edge.get("producer_port") or "")) or "out")
    cchan = _ident(channel_base(str(edge.get("consumer_port") or "")) or "in")
    t = edge.get("timing") if isinstance(edge.get("timing"), dict) else {}
    lat = (t.get("req_to_rsp_cycles") or {}) if isinstance(t, dict) else {}
    latency = lat.get("exact") if lat.get("exact") is not None else (lat.get("min") or 0)
    if fam in FIFO_FAMILIES:
        kind = "fifo"
        ptype, ctype = "sc_core::sc_fifo_out<cs_beat_t>", "sc_core::sc_fifo_in<cs_beat_t>"
    elif fam in STATIC_FAMILIES:
        kind = "signal"
        ptype, ctype = "sc_core::sc_out<cs_word_t>", "sc_core::sc_in<cs_word_t>"
    else:
        kind = "socket"
        ptype, ctype = ("tlm_utils::simple_initiator_socket<{P}>", "tlm_utils::simple_target_socket<{C}>")
    return {"edge_id": str(edge.get("edge_id") or f"{edge.get('producer_block')}__to__{edge.get('consumer_block')}"),
            "family": fam, "kind": kind, "producer": str(edge.get("producer_block") or ""),
            "consumer": str(edge.get("consumer_block") or ""), "producer_member": pchan,
            "consumer_member": cchan, "producer_type": ptype, "consumer_type": ctype,
            "latency_cycles": int(latency or 0), "data_width": int(edge.get("data_width_bits") or 0)}


def block_bindings(edges: list[dict], block: str) -> list[dict]:
    out = []
    for e in edges:
        b = channel_binding(e)
        if b["producer"] == block:
            out.append({**b, "role": "producer"})
        if b["consumer"] == block:
            out.append({**b, "role": "consumer"})
    return out


def render_block_skeleton(block: str, edges: list[dict], *, clock_period_ns: float = 10.0) -> str:
    """The header the model agent completes: every member the assembler binds,
    the b_transport hooks registered, latency constants from the contracts."""
    mn = model_name(block)
    binds = block_bindings(edges, block)
    L = [f"// {mn}.h -- SystemC TLM-2.0 LT model of block '{block}' (B2). GENERATED skeleton:",
         "// keep every member below (the SoC assembler binds them by name); implement the",
         "// behaviour in the .cpp. Loosely timed: b_transport adds the contract latency.",
         "#pragma once", "#include \"cs_model_common.h\"", "",
         f"SC_MODULE({mn}) {{"]
    seen = set()
    for b in binds:
        member = b["producer_member"] if b["role"] == "producer" else b["consumer_member"]
        if member in seen:
            continue
        seen.add(member)
        ty = (b["producer_type"] if b["role"] == "producer" else b["consumer_type"]).format(P=mn, C=mn)
        L.append(f"  {ty} {member};   // edge {b['edge_id']} ({b['family']}, {b['role']})")
    L.append("  sc_core::sc_in<bool> clk;")
    L.append("  sc_core::sc_in<bool> rst_n;")
    L.append("")
    L.append(f"  SC_CTOR({mn});")
    L.append("  void reset();")
    L.append("  void dump_state(std::ostream& os) const;")
    for b in binds:
        if b["role"] == "consumer" and b["kind"] == "socket":
            L.append(f"  void b_transport_{b['consumer_member']}(tlm::tlm_generic_payload& trans, "
                     f"sc_core::sc_time& delay);   // latency {b['latency_cycles']} cycle(s)")
    L.append("  void run();   // SC_THREAD: the block's behaviour")
    L.append("};")
    L.append("")
    L.append(f"// Constructor template (put in {mn}.cpp):")
    L.append(f"// {mn}::{mn}(sc_core::sc_module_name n) : sc_module(n)")
    inits = [f"{(b['producer_member'] if b['role']=='producer' else b['consumer_member'])}(\"{(b['producer_member'] if b['role']=='producer' else b['consumer_member'])}\")"
             for b in binds if b["kind"] == "socket"]
    if inits:
        L.append("//   , " + ", ".join(dict.fromkeys(inits)))
    L.append("// {")
    for b in binds:
        if b["role"] == "consumer" and b["kind"] == "socket":
            L.append(f"//   {b['consumer_member']}.register_b_transport(this, &{mn}::b_transport_{b['consumer_member']});")
    L.append("//   SC_THREAD(run); sensitive << clk.pos();")
    L.append("// }")
    L.append(f"// Clock period: {clock_period_ns} ns (cs_clock_period()).")
    return "\n".join(L) + "\n"


COMMON_HEADER = '''// cs_model_common.h -- shared by every generated SystemC block model (B2).
#pragma once
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <systemc>
#include <tlm>
#include <tlm_utils/simple_initiator_socket.h>
#include <tlm_utils/simple_target_socket.h>

typedef uint64_t cs_beat_t;   // one stream beat (payload packed LSB-first; wider beats split)
typedef uint64_t cs_word_t;   // one static bundle / bus word

inline sc_core::sc_time cs_clock_period() {
    static double ns = []() {
        const char* e = std::getenv("CS_CLOCK_PERIOD_NS");
        return e ? std::atof(e) : 10.0;
    }();
    return sc_core::sc_time(ns, sc_core::SC_NS);
}

// A minimal helper for models that keep byte-addressed state.
struct cs_mem {
    std::map<uint64_t, uint8_t> bytes;
    void write(uint64_t addr, const uint8_t* p, unsigned n) { for (unsigned i = 0; i < n; ++i) bytes[addr + i] = p[i]; }
    void read(uint64_t addr, uint8_t* p, unsigned n) const {
        for (unsigned i = 0; i < n; ++i) { auto it = bytes.find(addr + i); p[i] = it == bytes.end() ? 0 : it->second; }
    }
};

// Issue one generic-payload transaction on an initiator socket (LT).
template <class SOCK>
inline tlm::tlm_response_status cs_transact(SOCK& sock, bool write, uint64_t addr, uint8_t* data, unsigned len,
                                            sc_core::sc_time& delay) {
    tlm::tlm_generic_payload t;
    t.set_command(write ? tlm::TLM_WRITE_COMMAND : tlm::TLM_READ_COMMAND);
    t.set_address(addr);
    t.set_data_ptr(data);
    t.set_data_length(len);
    t.set_streaming_width(len);
    t.set_byte_enable_ptr(nullptr);
    t.set_dmi_allowed(false);
    t.set_response_status(tlm::TLM_INCOMPLETE_RESPONSE);
    sock->b_transport(t, delay);
    return t.get_response_status();
}
'''
