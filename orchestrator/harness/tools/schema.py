# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""``coresmith schema <kind>``: the document shapes ``register`` expects, as
the architect's reference -- so it never has to read engine source to learn
them (the first smoke sitting did exactly that, then wandered into a sibling
project's artifacts)."""
from __future__ import annotations

SCHEMAS: dict[str, str] = {
    "prd": '''PRD -- register `arch/prd_spec.json` (structured) next to `arch/prd_spec.md` (prose).
{
  "prd": {
    "title": "...", "revision": "1.0", "summary": "...",
    "target_technology": {"pdk": "...", "process_nm": 130, "rationale": "..."},
    "speed_and_feeds": {"target_clock_mhz": 64, "input_data_rate_mbps": 0, "output_data_rate_mbps": 0,
                        "latency_budget_us": 0, "throughput_requirements": "..."},
    "area_budget": {"max_gate_count": 0, "max_die_area_mm2": 0, "notes": "..."},
    "power_budget": {"total_power_mw": 0, "power_domains": ["core"], "leakage_budget_mw": 0},
    "dataflow": "...",
    "parameters": [{"name": "num_harts", "role": "dimension", "min": 1, "max": 2, "unit": "harts", "boundary_values": [1, 2]}],
    "functional_requirements": [
      "FR-TOP-1: <requirement text> [HARD, <source ref>]",          # id-in-string form, or
      {"id": "FR-CPU-2", "requirement": "...", "priority": "must_have", "acceptance": "..."}
    ],
    "validation_kpis": [{"id": "KPI-ISA-1", "metric": "...", "threshold": "...", "test_method": "...", "source": "..."}],
    "constraints": ["CON-1: ...", "..."],
    "open_items": ["Q-1: <a question the requirements do not decide>"],   # each becomes a must-answer question
    "golden_model_available": false,
    "risk_flags": ["NO_GOLDEN_REFERENCE_MODEL"]
  }
}
Rules: every functional requirement and KPI has an id (FR-<AREA>-<n>, KPI-<AREA>-<n>); a functional
requirement without an id is refused. Cite the source requirement in brackets.''',

    "frd": '''FRD -- register `arch/frd_spec.md` (markdown). Every requirement is a block:

## <Section: Performance Requirements | Interface Requirements | Semantic Invariants | Timing Requirements | Physical Design Requirements | MPW Submission Acceptance Criteria | ...>

1. **ID**: PERF-001
   - **Requirement**: <text> [HARD, KPI-FPS-1]          # cite the PRD KPI / FR ids it derives from
   - **Acceptance criteria**: <measurable, with the tool/command and threshold>
   - **Priority**: must_have | should_have
   - **Model check**: <how the executable architecture model observes it, or:
                      not model-testable -- <reason: physical design / STA / DRC / cycle-exact>>

Id prefixes: PERF, IFACE, INV (semantic invariants -- the ONLY invariant namespace; blocks cite them),
TIME, PHYS, MPW, TEST. `## Mission-Scale Acceptance Test (MANDATORY)` names `inputs/acceptance_stimulus.py`.
The stage machine refuses: KPIs no FRD item derives from, items without acceptance/priority, must-have
items without a Model check, open must-answer questions.''',

    "ers": '''ERS -- register `arch/ers_spec.json`.
{
  "ers": {
    "title": "...", "summary": "...",
    "functional_requirements": ["FR-TOP-1 [HARD, KPI-ISA-1]: ...", "..."],
    "system_invariants": [{"id": "INV-001", "description": "...", "affected_blocks": ["rv_l1d", "coh_bus"],
                           "required_state": ["..."], "verification": "..."}],
    "validation_dv_requirements": [{"id": "VAL-001", "requirement": "[HARD] ...", "measurable_kpi": "...",
                                    "threshold": "...", "test_method": "...", "covers": ["KPI-ISA-1", "FR-CPU-2", "INV-005"]}],
    "per_block_requirements": [{"block_name": "rv_exec", "requirements": ["ERS-rv_exec-1: ... (PERF-006)", "..."],
                                "interface_protocol": "...", "estimated_gates": 12000}],
    "open_items": ["C-ERS-1: <correction at source, with the item id>"]
  }
}
`covers` and bracketed ids become links (covers / derives_from); `affected_blocks` -> owned_by.''',

    "block_diagram": '''Block diagram -- register `arch/block_diagram.json`.
{
  "blocks": [
    {"name": "rv_core", "tier": 1, "subsystem": "cpu", "cluster": "cpu", "instances": 2,
     "description": "...", "estimated_gates": 120000, "flip_flop_budget": 9000,
     "owns": ["PERF-006", "INV-003"],                      # FRD items this block is responsible for
     "interfaces": {"m_l1i_req": {"handshake": "req_resp", "req": 64, "resp": 69}, "...": {}},
     "golden_slice": "..."},
    {"name": "soc_fabric", "tier": 0, "kind": "primitive", "primitive": "cs_fabric",
     "fabric": <the FabricSpec JSON from `coresmith fabric derive` (.coresmith/fabric_spec.json)>}
  ],
  "connections": [
    {"edge_id": "rv_core0__m_fabric__to__soc_fabric__s_cpu0",
     "producer_block": "rv_core", "producer_instance": 0, "producer_port": "m_fabric",
     "consumer_block": "soc_fabric", "consumer_port": "s_cpu0",
     "handshake_protocol": "axi4 | axi_lite | apb | req_resp | srdy_drdy | axi_stream | valid_only | mem_write | static",
     "data_width_bits": 64, "semantic_contract": "..."}
  ],
  "system_invariants": ["INV-001", "INV-003"],             # cite FRD ids; never a new namespace
  "global_output_contract": ["..."], "questions": []
}
Rules: a block with instances > 1 needs producer_instance / consumer_instance on every connection that
touches it; a req/resp channel is ONE connection (never two reversed ones); every fabric master port
must be reached by a connection; widths agree with the contracts. Every must-have FRD item (except
PHYS/MPW) must be owned by some block. `cluster` groups blocks for the cluster workers.''',

    "contracts": '''Interface contracts -- register `arch/interface_contracts.json`.
{"design_summary": "...", "contracts": [
  {"edge_id": "rv_core0__m_fabric__to__soc_fabric__s_cpu0",
   "producer_block": "rv_core", "producer_port": "m_fabric", "consumer_block": "soc_fabric", "consumer_port": "s_cpu0",
   "handshake_protocol": "axi4", "data_width_bits": 64,
   "bus_params": {"addr_width": 32, "id_width": 4, "user_width": 1},        # bus families only; NO generic 'data' field
   "fields": [{"name": "opcode", "width": 4, "msb": 3, "lsb": 0, "encoding": "..."}],   # non-bus families: bit-exact payload
   "sideband_signals": [{"name": "last", "width": 1}],
   "timing": {"req_to_rsp_cycles": {"min": 2, "max": 2, "exact": 2}, "valid_to_ready_max_stall": 8,
              "ordering": "in_order", "burst": {"last_signal": "last", "max_beats": 16},
              "reset_idle_cycles": 4, "valid_hold_until_ready": true},
   "flow_control_policy": "...", "semantic_contract": "..."}
]}
One contract per diagram connection; instance-expanded edge ids; the diagram's widths must match.''',

    "abi": '''HW/SW ABI -- register `arch/hw_sw_abi.md` (markdown, >= 200 chars). Must contain: the memory map
(every window with base/size/owner), every register map (offset, name, bits, reset, access, side
effects), interrupt map, the GPU ISA / command and job formats, boot sequence and image layout, and a
dated changelog. Software starts against this document; it is frozen at the `interfaces` stage.''',

    "uarch": '''uArch spec -- register `arch/uarch_specs/<block>.md` with `--block <block>`. Sections required:
`## 2` Interface, `## 3` Microarchitecture, `### 4a` Cross-Block Semantic Invariants (INV-<BLOCK>-nnn lines,
each citing the FRD INV-nnn it refines), `## 5` Reset, `### 6a` Output Timing Contract, `## 9` Verilog
Interface Stub. `### 6.1` must cite the PERF-nnn ids it meets with the model evidence; "MEETS" without a
PERF id is flagged.''',

    "arch_model": '''Executable architecture model -- `model/arch/arch_model.json` (then `coresmith model build|run|eval --arch`).
{"name": "soc", "clock_mhz": 64, "addr_width": 32,
 "components": [
   {"name": "cpu", "kind": "initiator", "instances": 2, "energy_pj_per_txn": 20},
   {"name": "gpu", "kind": "both", "base": "0x30000000", "size": "0x20000", "latency_cycles": 2},
   {"name": "ram", "kind": "target", "base": "0x80000000", "size": "0x8000000", "latency_cycles": 8,
    "bytes_per_cycle": 8, "protocol": "axi4", "energy_pj_per_byte": 1.0, "static_mw": 5},
   {"name": "uart", "kind": "target", "base": "0x10000000", "size": "0x1000", "latency_cycles": 2, "protocol": "apb"}],
 "fabric": {"latency_cycles": 2, "bytes_per_cycle": 8},
 "links": [{"from": "gpu", "to": "ram", "bytes_per_cycle": 16, "latency_cycles": 3}]}
kinds: initiator | target | both (a target window plus an initiator port). Sizes are powers of two,
bases aligned. The FRD harness (model eval) drives `top.u_<name>[i]->body` scenarios from MEASURED
workload numbers and prints one verdict per FRD item.''',
}

SCHEMAS["harness"] = "FRD evaluation harness: written by the engine's agent under model/arch/frd_eval/ (model eval --arch) or model/frd_eval/."
SCHEMAS["sad"] = "SAD -- register `arch/sad_spec.md` (markdown): system decisions, memory map table, component list with the numbers the arch model uses."


def schema(kind: str) -> str:
    return SCHEMAS.get(kind, "unknown kind; kinds: " + ", ".join(sorted(SCHEMAS)))
