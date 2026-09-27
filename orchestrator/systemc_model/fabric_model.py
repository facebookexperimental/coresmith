# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The SystemC model of a generated fabric primitive is itself generated:
a TLM router that decodes the address map and forwards each transaction
from its ``s_<master>`` target socket to the ``m_<slave>`` initiator socket
(DECERR when unmapped). No LLM involved."""
from __future__ import annotations

from orchestrator.fabric.spec import FabricSpec

from .conventions import model_name


def render_fabric_model(block: str, spec: FabricSpec, edges: list[dict]) -> tuple[str, str]:
    """``(header, cpp)`` for the fabric block; members follow the contract
    edges (``s_<master>`` targets, ``m_<slave>`` initiators)."""
    mn = model_name(block)
    masters = [m.name for m in spec.masters]
    slaves = spec.slaves
    H = ["// GENERATED SystemC TLM-2.0 LT router for fabric primitive '%s' (B2)." % block,
         "#pragma once", '#include "cs_model_common.h"', "", f"SC_MODULE({mn}) {{"]
    # Optional sockets: a fabric port the block diagram declares but no
    # contract edge reaches (e.g. a second hart's I/O port modelled once)
    # stays unbound instead of failing elaboration (SystemC E109).
    for m in masters:
        H.append(f"  tlm_utils::simple_target_socket_tagged_optional<{mn}> s_{m};")
    for s in slaves:
        H.append(f"  tlm_utils::simple_initiator_socket_optional<{mn}> m_{s.name};")
    H += ["  sc_core::sc_in<bool> clk;", "  sc_core::sc_in<bool> rst_n;", "  unsigned long long routed = 0, decerr = 0;",
          f"  SC_CTOR({mn});", "  void reset();", "  void dump_state(std::ostream& os) const;",
          "  void b_transport(int id, tlm::tlm_generic_payload& t, sc_core::sc_time& d);", "};"]
    C = [f'#include "{mn}.h"', f"{mn}::{mn}(sc_core::sc_module_name n) : sc_module(n)"]
    inits = [f's_{m}("s_{m}")' for m in masters] + [f'm_{s.name}("m_{s.name}")' for s in slaves]
    C.append("  , " + ", ".join(inits))
    C.append("{")
    for i, m in enumerate(masters):
        C.append(f"  s_{m}.register_b_transport(this, &{mn}::b_transport, {i});")
    C.append("}")
    C.append(f"void {mn}::reset() {{ routed = 0; decerr = 0; }}")
    C.append(f'void {mn}::dump_state(std::ostream& os) const {{ os << "routed=" << routed << "\\n" << "decerr=" << decerr << "\\n"; }}')
    C.append(f"void {mn}::b_transport(int id, tlm::tlm_generic_payload& t, sc_core::sc_time& d) {{")
    C.append("  (void)id; uint64_t a = t.get_address(); d += cs_clock_period();")
    for s in slaves:
        C.append(f"  if (a >= 0x{s.base:X}ULL && a < 0x{s.base + s.size:X}ULL) {{")
        C.append(f"    if (m_{s.name}.size() == 0) {{ ++decerr; t.set_response_status(tlm::TLM_ADDRESS_ERROR_RESPONSE); return; }}")
        C.append(f"    ++routed; m_{s.name}->b_transport(t, d); return; }}")
    C.append("  ++decerr; t.set_response_status(tlm::TLM_ADDRESS_ERROR_RESPONSE);")
    C.append("}")
    return "\n".join(H) + "\n", "\n".join(C) + "\n"
