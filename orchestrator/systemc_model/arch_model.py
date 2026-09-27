# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The executable SAD: an abstract SystemC performance model of the SoC
(architect sitting, step 2).

Before decomposition, before any fabric is chosen, the architect describes
the chip as *components* (initiators, targets, memories with a service
latency and an address window) and the *links* between them (latency,
bytes/cycle). This module generates a loosely-timed SystemC model of that
description -- generic initiators driven by a scenario the FRD harness
writes, targets with service latency, an abstract fabric that accounts bytes,
transactions, queueing and outstanding depth per link -- builds and runs it,
and turns the measured link table into a ``FabricSpec`` (masters, slaves,
data width, outstanding depth), so the fabric is an OUTPUT of analysis.

Spec (``model/arch/arch_model.json``)::

    {"name": "soc", "clock_mhz": 64, "addr_width": 32,
     "components": [
       {"name": "cpu", "kind": "initiator", "instances": 2, "protocol": "axi4",
        "energy_pj_per_txn": 20},
       {"name": "gpu", "kind": "initiator"},
       {"name": "ram", "kind": "target", "base": "0x80000000", "size": "0x8000000",
        "latency_cycles": 8, "bytes_per_cycle": 8, "protocol": "axi4"},
       {"name": "uart", "kind": "target", "base": "0x10000000", "size": "0x1000",
        "latency_cycles": 2, "protocol": "apb"}],
     "fabric": {"latency_cycles": 2, "bytes_per_cycle": 8},
     "links": [{"from": "gpu", "to": "ram", "bytes_per_cycle": 16}]}

Component kinds: ``initiator`` (issues transactions; the scenario body is
``top.u_<name>[i].body = [&](cs_arch_initiator& me){...}``), ``target``
(memory-mapped, latency + bandwidth), ``both`` (a DMA-like block: target
window plus an initiator port).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


def _int(v, default=0) -> int:
    if v is None or v == "":
        return default
    if isinstance(v, str) and v.strip().lower().startswith("0x"):
        return int(v, 16)
    return int(v)


@dataclass
class ArchComponent:
    name: str
    kind: str = "target"                 # initiator | target | both
    instances: int = 1
    protocol: str = "axi4"               # axi4 | axi_lite | apb (target side)
    base: int = 0
    size: int = 0
    latency_cycles: int = 1              # target service latency
    bytes_per_cycle: int = 8             # target bandwidth
    energy_pj_per_txn: float = 0.0
    energy_pj_per_byte: float = 0.0
    static_mw: float = 0.0

    @property
    def is_initiator(self) -> bool:
        return self.kind in ("initiator", "both")

    @property
    def is_target(self) -> bool:
        return self.kind in ("target", "both")


@dataclass
class ArchSpec:
    name: str
    clock_mhz: float = 100.0
    addr_width: int = 32
    components: list[ArchComponent] = field(default_factory=list)
    fabric_latency_cycles: int = 2
    fabric_bytes_per_cycle: int = 8
    links: list[dict] = field(default_factory=list)   # {from, to, latency_cycles?, bytes_per_cycle?}

    @classmethod
    def from_json(cls, d: dict) -> "ArchSpec":
        comps = []
        for c in d.get("components") or []:
            comps.append(ArchComponent(
                name=str(c.get("name") or ""), kind=str(c.get("kind") or "target"),
                instances=_int(c.get("instances"), 1), protocol=str(c.get("protocol") or "axi4"),
                base=_int(c.get("base"), 0), size=_int(c.get("size"), 0),
                latency_cycles=_int(c.get("latency_cycles"), 1), bytes_per_cycle=_int(c.get("bytes_per_cycle"), 8),
                energy_pj_per_txn=float(c.get("energy_pj_per_txn") or 0), energy_pj_per_byte=float(c.get("energy_pj_per_byte") or 0),
                static_mw=float(c.get("static_mw") or 0)))
        fab = d.get("fabric") or {}
        return cls(name=str(d.get("name") or "soc"), clock_mhz=float(d.get("clock_mhz") or 100),
                   addr_width=_int(d.get("addr_width"), 32), components=comps,
                   fabric_latency_cycles=_int(fab.get("latency_cycles"), 2),
                   fabric_bytes_per_cycle=_int(fab.get("bytes_per_cycle"), 8),
                   links=list(d.get("links") or []))

    def validate(self) -> list[str]:
        errs = []
        if not _NAME.match(self.name):
            errs.append(f"name {self.name!r} must be snake_case")
        names = [c.name for c in self.components]
        if len(set(names)) != len(names):
            errs.append("duplicate component names")
        for c in self.components:
            if not _NAME.match(c.name):
                errs.append(f"component {c.name!r} must be snake_case")
            if c.kind not in ("initiator", "target", "both"):
                errs.append(f"{c.name}: kind must be initiator|target|both")
            if c.is_target:
                if c.size <= 0 or c.size & (c.size - 1):
                    errs.append(f"{c.name}: target size must be a power of two (> 0)")
                if c.base % max(1, c.size):
                    errs.append(f"{c.name}: base 0x{c.base:X} not aligned to size 0x{c.size:X}")
                if c.protocol not in ("axi4", "axi_lite", "apb"):
                    errs.append(f"{c.name}: protocol must be axi4|axi_lite|apb")
            if c.instances < 1 or c.instances > 64:
                errs.append(f"{c.name}: instances out of range")
        targets = [c for c in self.components if c.is_target]
        for i, a in enumerate(targets):
            for b in targets[i + 1:]:
                if a.base < b.base + b.size and b.base < a.base + a.size:
                    errs.append(f"targets {a.name} and {b.name} overlap")
        if not any(c.is_initiator for c in self.components):
            errs.append("no initiator component")
        if not targets:
            errs.append("no target component")
        known = set(names)
        for l in self.links:
            if l.get("from") not in known or l.get("to") not in known:
                errs.append(f"link {l.get('from')}->{l.get('to')}: unknown component")
        return errs

    def link_params(self, src: str, dst: str) -> tuple[int, int]:
        for l in self.links:
            if l.get("from") == src and l.get("to") == dst:
                return _int(l.get("latency_cycles"), self.fabric_latency_cycles), _int(l.get("bytes_per_cycle"), self.fabric_bytes_per_cycle)
        return self.fabric_latency_cycles, self.fabric_bytes_per_cycle


COMMON_HEADER = r'''// cs_arch_common.h -- GENERATED abstract SystemC performance model primitives (executable SAD).
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <systemc>
#include <tlm>
#include <tlm_utils/simple_initiator_socket.h>
#include <tlm_utils/simple_target_socket.h>

inline sc_core::sc_time cs_clock_period() {
    static double ns = []() { const char* e = std::getenv("CS_CLOCK_PERIOD_NS"); return e ? std::atof(e) : 10.0; }();
    return sc_core::sc_time(ns, sc_core::SC_NS);
}
inline uint64_t cs_cycles(const sc_core::sc_time& t) { return (uint64_t)(t / cs_clock_period() + 1e-9); }

struct cs_link_stats {
    uint64_t txns = 0, bytes = 0, reads = 0, writes = 0, busy_cycles = 0, wait_cycles = 0;
    uint64_t max_outstanding = 0, outstanding = 0, decerr = 0;
};

// A target: address window, service latency, bandwidth, byte memory.
struct cs_arch_target : sc_core::sc_module {
    tlm_utils::simple_target_socket<cs_arch_target> s;
    uint64_t base, size; unsigned latency, bpc; sc_core::sc_time next_free;
    std::map<uint64_t, uint8_t> mem; cs_link_stats st;
    cs_arch_target(sc_core::sc_module_name n, uint64_t base_, uint64_t size_, unsigned lat, unsigned bpc_)
        : sc_module(n), s("s"), base(base_), size(size_), latency(lat), bpc(bpc_ ? bpc_ : 1), next_free(sc_core::SC_ZERO_TIME) {
        s.register_b_transport(this, &cs_arch_target::b_transport);
    }
    void b_transport(tlm::tlm_generic_payload& t, sc_core::sc_time& d) {
        uint64_t a = t.get_address(); unsigned n = t.get_data_length(); uint8_t* p = t.get_data_ptr();
        if (a < base || a + n > base + size) { t.set_response_status(tlm::TLM_ADDRESS_ERROR_RESPONSE); ++st.decerr; return; }
        sc_core::sc_time now = sc_core::sc_time_stamp() + d;
        sc_core::sc_time start = now > next_free ? now : next_free;
        st.wait_cycles += cs_cycles(start - now);
        unsigned cyc = latency + (n + bpc - 1) / bpc;
        next_free = start + cyc * cs_clock_period();
        d = next_free - sc_core::sc_time_stamp();
        st.busy_cycles += cyc; ++st.txns; st.bytes += n;
        if (t.is_write()) { ++st.writes; for (unsigned i = 0; i < n; ++i) mem[a + i] = p[i]; }
        else { ++st.reads; for (unsigned i = 0; i < n; ++i) { auto it = mem.find(a + i); p[i] = it == mem.end() ? 0 : it->second; } }
        t.set_response_status(tlm::TLM_OK_RESPONSE);
    }
    size_t load_bin(const std::string& path, uint64_t at) {
        FILE* f = std::fopen(path.c_str(), "rb"); if (!f) return 0; uint8_t buf[4096]; size_t n, tot = 0;
        while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) { for (size_t i = 0; i < n; ++i) mem[at + tot + i] = buf[i]; tot += n; }
        std::fclose(f); return tot;
    }
};

// The abstract fabric: routes by address, accounts every (initiator, target) link.
struct cs_arch_fabric : sc_core::sc_module {
    struct link { unsigned latency, bpc; sc_core::sc_time next_free; cs_link_stats st; };
    std::vector<tlm_utils::simple_target_socket_tagged<cs_arch_fabric>*> s;   // one per initiator
    std::vector<tlm_utils::simple_initiator_socket<cs_arch_fabric>*> m;       // one per target
    std::vector<std::string> inames, tnames; std::vector<uint64_t> tbase, tsize;
    std::map<std::pair<int,int>, link> links; unsigned def_lat, def_bpc; uint64_t decerr = 0;
    cs_arch_fabric(sc_core::sc_module_name n, unsigned lat, unsigned bpc) : sc_module(n), def_lat(lat), def_bpc(bpc ? bpc : 1) {}
    int add_initiator(const std::string& name) {
        int i = (int)s.size(); auto* p = new tlm_utils::simple_target_socket_tagged<cs_arch_fabric>(("s_" + name).c_str());
        p->register_b_transport(this, &cs_arch_fabric::b_transport, i); s.push_back(p); inames.push_back(name); return i;
    }
    int add_target(const std::string& name, uint64_t base, uint64_t size) {
        int j = (int)m.size(); m.push_back(new tlm_utils::simple_initiator_socket<cs_arch_fabric>(("m_" + name).c_str()));
        tnames.push_back(name); tbase.push_back(base); tsize.push_back(size); return j;
    }
    void set_link(int i, int j, unsigned lat, unsigned bpc) { links[{i, j}] = link{lat, bpc ? bpc : 1, sc_core::SC_ZERO_TIME, {}}; }
    link& L(int i, int j) { auto it = links.find({i, j}); if (it == links.end()) { set_link(i, j, def_lat, def_bpc); it = links.find({i, j}); } return it->second; }
    void b_transport(int i, tlm::tlm_generic_payload& t, sc_core::sc_time& d) {
        uint64_t a = t.get_address(); int j = -1;
        for (size_t k = 0; k < m.size(); ++k) if (a >= tbase[k] && a < tbase[k] + tsize[k]) { j = (int)k; break; }
        if (j < 0) { ++decerr; t.set_response_status(tlm::TLM_ADDRESS_ERROR_RESPONSE); return; }
        link& l = L(i, j); unsigned n = t.get_data_length();
        sc_core::sc_time now = sc_core::sc_time_stamp() + d;
        sc_core::sc_time start = now > l.next_free ? now : l.next_free;
        l.st.wait_cycles += cs_cycles(start - now);
        unsigned cyc = l.latency + (n + l.bpc - 1) / l.bpc;
        l.next_free = start + cyc * cs_clock_period();
        d = l.next_free - sc_core::sc_time_stamp();
        l.st.busy_cycles += cyc; ++l.st.txns; l.st.bytes += n; if (t.is_write()) ++l.st.writes; else ++l.st.reads;
        if (++l.st.outstanding > l.st.max_outstanding) l.st.max_outstanding = l.st.outstanding;
        (*m[j])->b_transport(t, d);
        --l.st.outstanding;
    }
    void dump(std::ostream& os, double clock_mhz, uint64_t total_cycles) const {
        os << "  \"links\": [";
        bool first = true;
        for (auto& kv : links) {
            const link& l = kv.second; if (!first) os << ","; first = false;
            os << "\n    {\"from\": \"" << inames[kv.first.first] << "\", \"to\": \"" << tnames[kv.first.second] << "\", \"txns\": " << l.st.txns
               << ", \"bytes\": " << l.st.bytes << ", \"reads\": " << l.st.reads << ", \"writes\": " << l.st.writes
               << ", \"busy_cycles\": " << l.st.busy_cycles << ", \"wait_cycles\": " << l.st.wait_cycles
               << ", \"max_outstanding\": " << l.st.max_outstanding
               << ", \"bytes_per_cycle\": " << (total_cycles ? (double)l.st.bytes / total_cycles : 0.0)
               << ", \"utilization\": " << (total_cycles ? (double)l.st.busy_cycles / total_cycles : 0.0) << "}";
        }
        os << "\n  ],\n  \"decerr\": " << decerr << ",\n";
        (void)clock_mhz;
    }
};

// An initiator whose behaviour is a scenario body the harness assigns.
struct cs_arch_initiator : sc_core::sc_module {
    tlm_utils::simple_initiator_socket<cs_arch_initiator> m;
    std::function<void(cs_arch_initiator&)> body;
    cs_link_stats st; uint64_t busy_cycles = 0, idle_cycles = 0; bool done = false;
    sc_core::sc_in<bool> clk; sc_core::sc_in<bool> rst_n;
    SC_HAS_PROCESS(cs_arch_initiator);
    cs_arch_initiator(sc_core::sc_module_name n) : sc_module(n), m("m"), clk("clk"), rst_n("rst_n") { SC_THREAD(run); }
    // One transaction; returns the response status. ``delay`` accounts the round trip.
    tlm::tlm_response_status issue(bool write, uint64_t addr, uint8_t* data, unsigned len) {
        tlm::tlm_generic_payload t; sc_core::sc_time d = sc_core::SC_ZERO_TIME;
        t.set_command(write ? tlm::TLM_WRITE_COMMAND : tlm::TLM_READ_COMMAND); t.set_address(addr); t.set_data_ptr(data);
        t.set_data_length(len); t.set_streaming_width(len); t.set_byte_enable_ptr(nullptr); t.set_dmi_allowed(false);
        t.set_response_status(tlm::TLM_INCOMPLETE_RESPONSE);
        m->b_transport(t, d); ++st.txns; st.bytes += len; if (write) ++st.writes; else ++st.reads;
        uint64_t c = cs_cycles(d); busy_cycles += c; sc_core::wait(d);
        return t.get_response_status();
    }
    uint64_t rd64(uint64_t a) { uint8_t b[8] = {0}; issue(false, a, b, 8); uint64_t v = 0; for (int i = 7; i >= 0; --i) v = (v << 8) | b[i]; return v; }
    void wr64(uint64_t a, uint64_t v) { uint8_t b[8]; for (int i = 0; i < 8; ++i) { b[i] = v & 0xff; v >>= 8; } issue(true, a, b, 8); }
    void compute(uint64_t cycles) { busy_cycles += cycles; sc_core::wait(cycles * cs_clock_period()); }   // local work
    void idle(uint64_t cycles) { idle_cycles += cycles; sc_core::wait(cycles * cs_clock_period()); }
    void run() { while (!rst_n.read()) sc_core::wait(clk.posedge_event()); if (body) body(*this); done = true; }
};
'''


def render_arch_top(spec: ArchSpec) -> str:
    L = [f"// arch_model_top.h -- GENERATED executable SAD of '{spec.name}'. Do not edit.", "#pragma once",
         '#include "cs_arch_common.h"', "", "SC_MODULE(arch_model_top) {",
         "  sc_core::sc_in<bool> clk; sc_core::sc_in<bool> rst_n;", "  cs_arch_fabric fabric;"]
    inits = [c for c in spec.components if c.is_initiator]
    tgts = [c for c in spec.components if c.is_target]
    for c in inits:
        L.append(f"  std::vector<cs_arch_initiator*> u_{c.name};   // {c.instances} instance(s)")
    for c in tgts:
        L.append(f"  cs_arch_target u_{c.name}{'_t' if c.is_initiator else ''};")
    L += ["  std::map<std::string, int> iidx, tidx;", "",
          f"  SC_CTOR(arch_model_top) : clk(\"clk\"), rst_n(\"rst_n\"), fabric(\"fabric\", {spec.fabric_latency_cycles}, {spec.fabric_bytes_per_cycle})"]
    for c in tgts:
        L.append(f"    , u_{c.name}{'_t' if c.is_initiator else ''}(\"{c.name}\", 0x{c.base:X}ULL, 0x{c.size:X}ULL, {c.latency_cycles}, {c.bytes_per_cycle})")
    L.append("  {")
    for c in tgts:
        member = f"u_{c.name}{'_t' if c.is_initiator else ''}"
        L.append(f"    tidx[\"{c.name}\"] = fabric.add_target(\"{c.name}\", 0x{c.base:X}ULL, 0x{c.size:X}ULL); fabric.m.back()->bind({member}.s);")
    for c in inits:
        L.append(f"    for (int i = 0; i < {c.instances}; ++i) {{")
        L.append(f"      auto* u = new cs_arch_initiator((std::string(\"{c.name}\") + std::to_string(i)).c_str());")
        L.append(f"      u->clk(clk); u->rst_n(rst_n); int k = fabric.add_initiator(\"{c.name}\" + std::string(i ? std::to_string(i) : \"\"));")
        L.append(f"      u->m.bind(*fabric.s.back()); iidx[\"{c.name}\" + std::to_string(i)] = k; u_{c.name}.push_back(u);")
        L.append("    }")
    for l in spec.links:
        lat, bpc = spec.link_params(str(l.get("from")), str(l.get("to")))
        L.append(f"    for (auto& kv : iidx) if (kv.first.rfind(\"{l.get('from')}\", 0) == 0) fabric.set_link(kv.second, tidx[\"{l.get('to')}\"], {lat}, {bpc});")
    L += ["  }", "", "  bool all_done() const {"]
    for c in inits:
        L.append(f"    for (auto* u : u_{c.name}) if (!u->done) return false;")
    L += ["    return true;", "  }", "",
          "  void stats_json(std::ostream& os, uint64_t total_cycles) const {",
          f"    os << \"{{\\n  \\\"model\\\": \\\"{spec.name}\\\", \\\"clock_mhz\\\": {spec.clock_mhz}, \\\"total_cycles\\\": \" << total_cycles << \",\\n\";",
          f"    fabric.dump(os, {spec.clock_mhz}, total_cycles);",
          "    os << \"  \\\"initiators\\\": [\"; bool f = true;"]
    for c in inits:
        L.append(f"    for (size_t i = 0; i < u_{c.name}.size(); ++i) {{ auto* u = u_{c.name}[i]; if (!f) os << \",\"; f = false;")
        L.append(f"      os << \"\\n    {{\\\"name\\\": \\\"{c.name}\" << i << \"\\\", \\\"txns\\\": \" << u->st.txns << \", \\\"bytes\\\": \" << u->st.bytes"
                 f" << \", \\\"busy_cycles\\\": \" << u->busy_cycles << \", \\\"idle_cycles\\\": \" << u->idle_cycles"
                 f" << \", \\\"energy_pj\\\": \" << (u->st.txns * {c.energy_pj_per_txn} + u->st.bytes * {c.energy_pj_per_byte}) << \", \\\"done\\\": \" << (u->done ? \"true\" : \"false\") << \"}}\"; }}")
    L.append("    os << \"\\n  ],\\n  \\\"targets\\\": [\"; f = true;")
    for c in tgts:
        member = f"u_{c.name}{'_t' if c.is_initiator else ''}"
        L.append(f"    if (!f) os << \",\"; f = false; os << \"\\n    {{\\\"name\\\": \\\"{c.name}\\\", \\\"txns\\\": \" << {member}.st.txns << \", \\\"bytes\\\": \" << {member}.st.bytes"
                 f" << \", \\\"busy_cycles\\\": \" << {member}.st.busy_cycles << \", \\\"wait_cycles\\\": \" << {member}.st.wait_cycles"
                 f" << \", \\\"decerr\\\": \" << {member}.st.decerr << \", \\\"utilization\\\": \" << (total_cycles ? (double){member}.st.busy_cycles / total_cycles : 0.0)"
                 f" << \", \\\"energy_pj\\\": \" << ({member}.st.txns * {c.energy_pj_per_txn} + {member}.st.bytes * {c.energy_pj_per_byte}) << \"}}\";")
    static_mw = sum(c.static_mw for c in spec.components)
    L += ["    double e = 0;"]
    for c in inits:
        L.append(f"    for (auto* u : u_{c.name}) e += u->st.txns * {c.energy_pj_per_txn} + u->st.bytes * {c.energy_pj_per_byte};")
    for c in tgts:
        member = f"u_{c.name}{'_t' if c.is_initiator else ''}"
        L.append(f"    e += {member}.st.txns * {c.energy_pj_per_txn} + {member}.st.bytes * {c.energy_pj_per_byte};")
    L += [f"    double secs = total_cycles / ({spec.clock_mhz} * 1e6);",
          f"    os << \"\\n  ],\\n  \\\"energy_pj\\\": \" << e << \", \\\"dynamic_mw\\\": \" << (secs > 0 ? e * 1e-12 / secs * 1e3 : 0.0) << \", \\\"static_mw\\\": {static_mw} \\n}}\" << std::endl;",
          "  }", "};", ""]
    return "\n".join(L)


ARCH_DRIVER = r'''// arch_model.cpp -- GENERATED smoke driver of the executable SAD. Do not edit.
#include "arch_model_top.h"
#include <fstream>
using namespace sc_core;
int sc_main(int argc, char** argv) {
    double run_ns = 10000; std::string stats = "stats.json";
    for (int i = 1; i + 1 < argc; ++i) { if (std::string(argv[i]) == "--ns") run_ns = std::atof(argv[i + 1]); if (std::string(argv[i]) == "--stats") stats = argv[i + 1]; }
    sc_clock clk("clk", cs_clock_period()); sc_signal<bool> rst_n("rst_n");
    arch_model_top top("top"); top.clk(clk); top.rst_n(rst_n);
    // smoke scenario: every initiator touches every target once
    for (auto& kv : top.iidx) (void)kv;
    rst_n.write(false); sc_start(3 * cs_clock_period()); rst_n.write(true);
    sc_start(sc_time(run_ns, SC_NS));
    uint64_t cyc = cs_cycles(sc_time_stamp());
    std::ofstream f(stats); top.stats_json(f, cyc); f.close();
    std::cout << "ARCH_STATS_FILE " << stats << std::endl << "ARCH_MODEL_OK" << std::endl;
    return 0;
}
'''

MAKEFILE = '''# GENERATED: executable SAD (abstract SystemC performance model).
SYSTEMC_HOME ?= {systemc_home}
CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wno-unused-parameter
INC := -I.
LIB := -lsystemc
ifneq ($(strip $(SYSTEMC_HOME)),)
INC += -I$(SYSTEMC_HOME)/include
LIB := -L$(SYSTEMC_HOME)/lib -L$(SYSTEMC_HOME)/lib-linux64 -Wl,-rpath,$(SYSTEMC_HOME)/lib -lsystemc
endif
MODEL_SRCS :=
FRD_SRCS := $(wildcard frd_eval/*.cpp)
arch_model: arch_model.cpp $(wildcard *.h)
\t$(CXX) $(CXXFLAGS) $(INC) arch_model.cpp $(LIB) -o $@
frd_eval/frd_eval: $(FRD_SRCS) $(wildcard *.h) $(wildcard frd_eval/*.h)
\t$(CXX) $(CXXFLAGS) $(INC) -Ifrd_eval $(FRD_SRCS) $(LIB) -o $@
clean:
\trm -f arch_model frd_eval/frd_eval stats.json
.PHONY: clean
'''


def arch_dir(project_root) -> Path:
    return Path(project_root) / "model" / "arch"


def load_spec(project_root) -> ArchSpec:
    p = arch_dir(project_root) / "arch_model.json"
    return ArchSpec.from_json(json.loads(p.read_text()))


def write_build(project_root, spec: ArchSpec, *, systemc_home: str = "") -> Path:
    md = arch_dir(project_root)
    md.mkdir(parents=True, exist_ok=True)
    (md / "frd_eval").mkdir(exist_ok=True)
    (md / "cs_arch_common.h").write_text(COMMON_HEADER)
    (md / "arch_model_top.h").write_text(render_arch_top(spec))
    (md / "arch_model.cpp").write_text(ARCH_DRIVER)
    (md / "Makefile").write_text(MAKEFILE.format(systemc_home=systemc_home))
    return md


def build(md, *, timeout_s: int = 600) -> dict:
    from .toolchain import systemc_home
    env = dict(os.environ)
    if systemc_home():
        env["SYSTEMC_HOME"] = systemc_home()
    try:
        p = subprocess.run(["make", "-s", "arch_model"], cwd=md, capture_output=True, text=True, timeout=timeout_s, env=env)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "log": str(exc)}
    (Path(md) / "build.log").write_text(p.stdout + p.stderr)
    return {"ok": p.returncode == 0 and (Path(md) / "arch_model").exists(), "log": (p.stdout + p.stderr)[-6000:]}


def run(md, *, ns: int = 10000, timeout_s: int = 600) -> dict:
    exe = Path(md) / "arch_model"
    if not exe.exists():
        return {"ok": False, "log": "arch_model not built", "stats": None}
    try:
        p = subprocess.run([str(exe), "--ns", str(ns), "--stats", "stats.json"], cwd=md, capture_output=True, text=True, timeout=timeout_s)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "log": str(exc), "stats": None}
    out = p.stdout + p.stderr
    (Path(md) / "run.log").write_text(out)
    stats = None
    sp = Path(md) / "stats.json"
    if sp.exists():
        try:
            stats = json.loads(sp.read_text())
        except ValueError:
            stats = None
    return {"ok": p.returncode == 0 and "ARCH_MODEL_OK" in out and stats is not None, "log": out[-4000:], "stats": stats}


def read_stats(md) -> dict | None:
    sp = Path(md) / "stats.json"
    if not sp.exists():
        return None
    try:
        return json.loads(sp.read_text())
    except ValueError:
        return None


def _pow2_at_least(v: int, lo: int, hi: int) -> int:
    x = lo
    while x < v and x < hi:
        x *= 2
    return min(x, hi)


def derive_fabric(spec: ArchSpec, stats: dict, *, name: str | None = None, min_data_width: int = 32,
                  max_data_width: int = 128, headroom: float = 2.0) -> dict:
    """The measured link table -> FabricSpec JSON. Masters = every initiator
    instance that issued traffic (+ the ones that issued none, kept so the
    port exists), slaves = every target with its window and protocol, data
    width = the smallest power of two that carries the busiest link's
    bytes/cycle with ``headroom``, outstanding depth = max observed (>= 2)."""
    links = stats.get("links") or []
    busiest = max((float(l.get("bytes_per_cycle") or 0) for l in links), default=0.0)
    width_bits = _pow2_at_least(int(busiest * headroom * 8 + 0.999), min_data_width, max_data_width)
    max_out = max((int(l.get("max_outstanding") or 0) for l in links), default=1)
    masters = []
    for c in spec.components:
        if c.is_initiator:
            for i in range(c.instances):
                masters.append({"name": f"{c.name}{i if c.instances > 1 else ''}", "protocol": "axi4", "id_width": 4,
                                "max_outstanding": max(2, min(16, max_out))})
    # a 'both' component's window is its register slave: <name>_regs (its initiator keeps the name)
    slaves = [{"name": c.name + ("_regs" if c.is_initiator else ""), "protocol": c.protocol,
               "base": f"0x{c.base:X}", "size": f"0x{c.size:X}"} for c in spec.components if c.is_target]
    per_link = [{"from": l.get("from"), "to": l.get("to"), "bytes_per_cycle": l.get("bytes_per_cycle"),
                 "utilization": l.get("utilization"), "max_outstanding": l.get("max_outstanding")} for l in links]
    return {"name": name or spec.name, "masters": masters, "slaves": slaves, "data_width": width_bits,
            "addr_width": spec.addr_width, "max_outstanding": max(2, min(16, max_out)),
            "derived_from": {"busiest_link_bytes_per_cycle": busiest, "headroom": headroom, "total_cycles": stats.get("total_cycles"),
                             "links": per_link}}
