# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""A-Fix 3(c): interface-contract structural violations are BLOCKING.

``_validate_contracts`` now emits structured ``contract_violations``; the
``interface_definition_node`` surfaces them as a structural constraint_result
and ``route_after_interface_definition`` diverts the run to Escalate
Constraints (replacing the hard Interface Definition -> Memory Map edge).

Hermetic -- the LLM specialist is monkeypatched.
"""

from __future__ import annotations

import asyncio

from orchestrator.architecture.specialists.interface_definition import (
    _max_interface_width,
    _validate_contracts,
)
from orchestrator.langgraph import architecture_graph as ag


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _types(violations):
    return sorted({v["type"] for v in violations})


# ---------------------------------------------------------------------------
# Env gate + width bound
# ---------------------------------------------------------------------------

def test_gate_default_on(monkeypatch):
    monkeypatch.delenv("CORESMITH_INTERFACE_CONTRACT_GATE", raising=False)
    assert ag._interface_contract_gate_enabled() is True


def test_gate_can_be_disabled(monkeypatch):
    monkeypatch.setenv("CORESMITH_INTERFACE_CONTRACT_GATE", "0")
    assert ag._interface_contract_gate_enabled() is False


def test_max_interface_width_default_and_override(monkeypatch):
    monkeypatch.delenv("CORESMITH_MAX_INTERFACE_WIDTH", raising=False)
    assert _max_interface_width() == 1024
    monkeypatch.setenv("CORESMITH_MAX_INTERFACE_WIDTH", "64")
    assert _max_interface_width() == 64


# ---------------------------------------------------------------------------
# _validate_contracts blocking classes
# ---------------------------------------------------------------------------

def test_empty_contracts_no_blocking_violation():
    v, notes = _validate_contracts({"contracts": []}, [{"from": "a", "to": "b"}])
    assert v["contract_violations"] == []
    assert any("no interface contracts" in n for n in notes)


def test_missing_contract_partial_coverage_blocks():
    contracts = [{"producer_block": "a", "consumer_block": "b",
                  "data_width_bits": 8,
                  "fields": [{"name": "d", "width": 8, "msb": 7, "lsb": 0}]}]
    v, _ = _validate_contracts(
        {"contracts": contracts},
        [{"from": "a", "to": "b"}, {"from": "b", "to": "c"}],
    )
    assert "missing_contract" in _types(v["contract_violations"])


def test_width_field_sum_mismatch_blocks():
    contracts = [{"producer_block": "a", "consumer_block": "b",
                  "data_width_bits": 32,
                  "fields": [{"name": "f", "width": 16, "msb": 15, "lsb": 0}]}]
    v, _ = _validate_contracts({"contracts": contracts}, [{"from": "a", "to": "b"}])
    assert "width_field_sum_mismatch" in _types(v["contract_violations"])


def test_field_overlap_blocks():
    contracts = [{"producer_block": "a", "consumer_block": "b",
                  "data_width_bits": 16,
                  "fields": [{"name": "x", "width": 8, "msb": 7, "lsb": 0},
                             {"name": "y", "width": 8, "msb": 10, "lsb": 3}]}]
    v, _ = _validate_contracts({"contracts": contracts}, [{"from": "a", "to": "b"}])
    assert "field_overlap" in _types(v["contract_violations"])


def test_cycle_edge_and_fifo_depth_block():
    contracts = [
        {"producer_block": "p", "consumer_block": "q", "data_width_bits": 8,
         "fields": [{"name": "d", "width": 8, "msb": 7, "lsb": 0}],
         "flow_control_policy": {"semantics": "free_running"}},
        {"producer_block": "q", "consumer_block": "p", "data_width_bits": 8,
         "fields": [{"name": "d", "width": 8, "msb": 7, "lsb": 0}],
         "flow_control_policy": {"semantics": "elastic_fifo",
                                 "feedback_cycle": True,
                                 "min_buffer_depth_beats": 1}},
    ]
    v, _ = _validate_contracts(
        {"contracts": contracts},
        [{"from": "p", "to": "q"}, {"from": "q", "to": "p"}],
    )
    t = _types(v["contract_violations"])
    assert "cycle_edge_semantics" in t
    assert "fifo_depth" in t


def test_over_wide_bus_blocks_unless_serialized(monkeypatch):
    monkeypatch.setenv("CORESMITH_MAX_INTERFACE_WIDTH", "1024")
    wide = [{"producer_block": "a", "consumer_block": "b", "data_width_bits": 7000}]
    v, _ = _validate_contracts({"contracts": wide}, [{"from": "a", "to": "b"}])
    assert "over_wide_bus" in _types(v["contract_violations"])
    # serialized escape
    wide_ser = [{"producer_block": "a", "consumer_block": "b",
                 "data_width_bits": 7000, "serialized": True}]
    v2, _ = _validate_contracts({"contracts": wide_ser}, [{"from": "a", "to": "b"}])
    assert "over_wide_bus" not in _types(v2["contract_violations"])


# ---------------------------------------------------------------------------
# Node + router wiring
# ---------------------------------------------------------------------------

def _patch_specialist(monkeypatch, contract_violations):
    import orchestrator.architecture.specialists.interface_definition as ifd_mod

    async def fake_analyze(**kw):
        return {"result": {"contracts": [{"producer_block": "a",
                                          "consumer_block": "b"}],
                           "contract_violations": list(contract_violations),
                           "open_questions": []},
                "questions": []}

    monkeypatch.setattr(ifd_mod, "analyze_interface_definition", fake_analyze)


def _state(tmp_path):
    return {"project_root": str(tmp_path), "round": 1,
            "block_diagram": {"blocks": [{"name": "a"}, {"name": "b"}],
                              "connections": [{"from": "a", "to": "b"}]},
            "requirements": ""}


def test_node_violations_route_to_escalate(monkeypatch, tmp_path):
    monkeypatch.setenv("CORESMITH_INTERFACE_CONTRACT_GATE", "1")
    viol = [{"edge": "a->b", "type": "over_wide_bus", "category": "structural",
             "severity": "error", "violation": "too wide",
             "source": "interface_definition"}]
    _patch_specialist(monkeypatch, viol)

    out = _run(ag.interface_definition_node(_state(tmp_path)))
    cr = out["constraint_result"]
    assert cr["has_structural"] is True
    assert cr["source"] == "interface_definition"
    assert len(cr["violations"]) == 1
    assert ag.route_after_interface_definition(out) == "Escalate Constraints"


def test_node_clean_routes_to_memory_map(monkeypatch, tmp_path):
    monkeypatch.setenv("CORESMITH_INTERFACE_CONTRACT_GATE", "1")
    _patch_specialist(monkeypatch, [])

    out = _run(ag.interface_definition_node(_state(tmp_path)))
    # Clean re-emit ALWAYS writes an explicit clean interface-definition verdict
    # (so a stale structural result from a prior round cannot re-divert forever).
    cr = out["constraint_result"]
    assert cr["all_pass"] is True
    assert cr["has_structural"] is False
    assert cr["violations"] == []
    assert cr["source"] == "interface_definition"
    # router reads the clean constraint_result -> Memory Map
    state_after = dict(_state(tmp_path))
    state_after.update(out)
    assert ag.route_after_interface_definition(state_after) == "Memory Map"


def test_node_gate_off_never_blocks(monkeypatch, tmp_path):
    monkeypatch.setenv("CORESMITH_INTERFACE_CONTRACT_GATE", "0")
    viol = [{"edge": "a->b", "type": "over_wide_bus", "category": "structural",
             "severity": "error", "violation": "too wide",
             "source": "interface_definition"}]
    _patch_specialist(monkeypatch, viol)

    out = _run(ag.interface_definition_node(_state(tmp_path)))
    # gate off -> violations ignored, no structural constraint_result
    assert "constraint_result" not in out
    assert ag.route_after_interface_definition(out) == "Memory Map"


def test_router_ignores_stale_constraint_result(monkeypatch):
    monkeypatch.setenv("CORESMITH_INTERFACE_CONTRACT_GATE", "1")
    # a structural constraint_result from a PRIOR real constraint check
    stale = {"constraint_result": {"has_structural": True,
                                   "source": "constraint_check",
                                   "violations": [{"violation": "x"}]}}
    assert ag.route_after_interface_definition(stale) == "Memory Map"


# ---------------------------------------------------------------------------
# Non-streaming protocol families (interface-protocol-families fix)
# ---------------------------------------------------------------------------

def _cycle_edge(proto, semantics="", depth=None):
    fc = {"semantics": semantics} if semantics else {}
    if depth is not None:
        fc["min_buffer_depth_beats"] = depth
    return {"producer_block": "mac", "consumer_block": "mem",
            "handshake_protocol": proto, "data_width_bits": 8,
            "fields": [{"name": "d", "width": 8, "msb": 7, "lsb": 0}],
            "flow_control_policy": fc}


def _both_dirs(proto):
    # mac<->mem form a 2-cycle in the graph (write path + read path)
    return [
        {**_cycle_edge(proto), "producer_block": "mac", "consumer_block": "mem"},
        {**_cycle_edge(proto), "producer_block": "mem", "consumer_block": "mac"},
    ]


def test_mem_write_on_cycle_not_forced_into_fifo():
    # a memory write/read pair closes a graph cycle but is fixed-latency and
    # always-accepted -- must NOT be flagged cycle_edge_semantics / fifo_depth.
    edges = [{"from": "mac", "to": "mem"}, {"from": "mem", "to": "mac"}]
    v, _ = _validate_contracts({"contracts": _both_dirs("mem_write")}, edges)
    t = _types(v["contract_violations"])
    assert "cycle_edge_semantics" not in t
    assert "fifo_depth" not in t


def test_req_resp_and_static_and_valid_only_exempt_from_cycle_fifo():
    edges = [{"from": "mac", "to": "mem"}, {"from": "mem", "to": "mac"}]
    for proto in ("req_resp", "static", "valid_only"):
        v, _ = _validate_contracts({"contracts": _both_dirs(proto)}, edges)
        assert "cycle_edge_semantics" not in _types(v["contract_violations"]), proto


def test_streaming_on_cycle_still_requires_fifo():
    # regression guard: srdy_drdy (streaming) on a cycle without elastic_fifo
    # is STILL a violation (the video_codec deadlock class the check exists for).
    edges = [{"from": "mac", "to": "mem"}, {"from": "mem", "to": "mac"}]
    v, _ = _validate_contracts(
        {"contracts": _both_dirs("srdy_drdy")}, edges)  # semantics unset
    assert "cycle_edge_semantics" in _types(v["contract_violations"])


def test_new_families_accepted_no_crash():
    # each new family with a coherent single edge produces no structural error
    for proto in ("req_resp", "mem_write", "valid_only", "static"):
        c = [{"producer_block": "a", "consumer_block": "b",
              "handshake_protocol": proto, "data_width_bits": 8,
              "fields": [{"name": "d", "width": 8, "msb": 7, "lsb": 0}]}]
        v, _ = _validate_contracts({"contracts": c}, [{"from": "a", "to": "b"}])
        assert v["contract_violations"] == [], (proto, v["contract_violations"])


# ---------------------------------------------------------------------------
# End-to-end interface-family SELECTION + deterministic coherence gate
# (interface-family-propagation fix)
#
# FAIRNESS: this is a GENERIC synthetic decomposition -- a compute datapath
# writes results into an on-chip SRAM subsystem, plus a host start/config
# strobe. No benchmark / exercise names, no golden filenames, no exercise
# params. The block-diagram edges declare their authoritative
# handshake_protocol; the interface specialist might still PROPOSE a streaming
# family + FIFO from invented port spelling. We assert on the families the
# propagation SELECTS, not merely that hand-labeled correct contracts don't
# crash.
# ---------------------------------------------------------------------------

from orchestrator.architecture.specialists.interface_definition import (  # noqa: E402
    _propagate_edge_families,
)
from orchestrator.architecture.constraints import (  # noqa: E402
    _check_interface_family_coherence,
)


def _generic_family_diagram():
    return {
        "blocks": [
            {"name": "compute_core"},
            {"name": "result_store"},
            {"name": "host_ctrl"},
        ],
        "connections": [
            # compute datapath -> on-chip SRAM subsystem: a direct write.
            {"from": "compute_core", "to": "result_store",
             "handshake_protocol": "mem_write", "data_width": 32},
            # host start / config strobe into the compute datapath.
            {"from": "host_ctrl", "to": "compute_core",
             "handshake_protocol": "valid_only", "data_width": 16},
        ],
    }


def _mislabeled_specialist_contracts():
    """What an over-eager specialist might FREEZE: it invented ready/FIFO on the
    write edge and a ready on the strobe -- both families are WRONG (srdy_drdy),
    exactly the live-run failure this fix targets."""
    return {
        "contracts": [
            {
                "producer_block": "compute_core",
                "consumer_block": "result_store",
                "handshake_protocol": "srdy_drdy",
                "data_width_bits": 32,
                "fields": [{"name": "wdata", "width": 32, "msb": 31, "lsb": 0}],
                "sideband_signals": [
                    {"name": "wr_addr", "purpose": "write address"},
                    {"name": "wr_ready", "purpose": "invented backpressure"},
                ],
                "flow_control_policy": {
                    "semantics": "elastic_fifo",
                    "min_buffer_depth_beats": 8,
                    "consumer_can_stall": True,
                    "feedback_cycle": True,
                },
            },
            {
                "producer_block": "host_ctrl",
                "consumer_block": "compute_core",
                "handshake_protocol": "srdy_drdy",
                "data_width_bits": 16,
                "fields": [{"name": "cfg", "width": 16, "msb": 15, "lsb": 0}],
                "sideband_signals": [
                    {"name": "start", "purpose": "start strobe"},
                    {"name": "cfg_drdy", "purpose": "invented ready"},
                ],
                "flow_control_policy": {
                    "semantics": "skid",
                    "min_buffer_depth_beats": 1,
                },
            },
        ]
    }


def _has_ready_signal(contract):
    return any(
        ("ready" in str(s.get("name", "")).lower())
        or ("drdy" in str(s.get("name", "")).lower())
        for s in contract.get("sideband_signals", []) or []
    )


def test_family_selection_forces_mem_write_and_valid_only(monkeypatch):
    # SELECTION assertion: the propagation must CHOOSE mem_write (write edge)
    # and valid_only (strobe) from the block-diagram intent, overriding the
    # specialist's invented srdy_drdy labels.
    monkeypatch.delenv("CORESMITH_INTERFACE_FAMILY_PROPAGATION", raising=False)
    diagram = _generic_family_diagram()
    out, notes = _propagate_edge_families(
        _mislabeled_specialist_contracts(), diagram["connections"]
    )
    by_pair = {
        (c["producer_block"], c["consumer_block"]): c for c in out["contracts"]
    }
    write = by_pair[("compute_core", "result_store")]
    strobe = by_pair[("host_ctrl", "compute_core")]

    assert write["handshake_protocol"] == "mem_write"
    assert strobe["handshake_protocol"] == "valid_only"
    # srdy_drdy count for these edges is 0.
    assert sum(
        1 for c in out["contracts"] if c["handshake_protocol"] == "srdy_drdy"
    ) == 0
    assert notes, "propagation should record what it changed"

    # Neither memory-write nor strobe contract carries a ready, an elastic_fifo,
    # or a FIFO depth.
    for c in (write, strobe):
        fc = c.get("flow_control_policy") or {}
        assert fc.get("semantics") != "elastic_fifo"
        assert int(fc.get("min_buffer_depth_beats") or 0) == 0
        assert not _has_ready_signal(c)

    # The start/config contract still carries the start strobe, with no ready.
    strobe_names = [
        str(s.get("name", "")).lower() for s in strobe.get("sideband_signals", [])
    ]
    assert any("start" in n for n in strobe_names)
    assert not _has_ready_signal(strobe)

    # And the deterministic coherence gate now PASSES the propagated contracts.
    assert _check_interface_family_coherence(diagram, out) == []


def test_family_propagation_gate_off_is_noop(monkeypatch):
    monkeypatch.setenv("CORESMITH_INTERFACE_FAMILY_PROPAGATION", "0")
    diagram = _generic_family_diagram()
    out, notes = _propagate_edge_families(
        _mislabeled_specialist_contracts(), diagram["connections"]
    )
    # gate off -> families untouched (specialist invention trusted verbatim).
    assert sorted(c["handshake_protocol"] for c in out["contracts"]) == [
        "srdy_drdy", "srdy_drdy",
    ]
    assert notes == []


def test_mislabeled_write_edge_rejected_by_coherence(monkeypatch):
    # A deliberately mislabeled write edge (srdy_drdy + FIFO on a write-only-
    # memory edge) is REJECTED deterministically -- regardless of any LLM
    # verdict -- because the block-diagram intent is mem_write.
    monkeypatch.delenv("CORESMITH_INTERFACE_FAMILY_GATE", raising=False)
    diagram = _generic_family_diagram()
    v = _check_interface_family_coherence(
        diagram, _mislabeled_specialist_contracts()
    )
    assert v, "expected deterministic rejection of the mislabeled write edge"
    assert all(x["check"] == "inter_block_payload_protocol_coherence" for x in v)
    assert all(
        x["category"] == "structural" and x["severity"] == "error" for x in v
    )
    assert any("compute_core->result_store" in x["violation"] for x in v)


def test_mem_write_contract_with_stray_ready_rejected(monkeypatch):
    # Even when the diagram edge omits handshake_protocol, a contract that
    # LABELS itself mem_write but carries a stray ready is rejected.
    monkeypatch.delenv("CORESMITH_INTERFACE_FAMILY_GATE", raising=False)
    diagram = {"connections": [{"from": "a", "to": "b"}]}  # no declared family
    contracts = {"contracts": [{
        "producer_block": "a", "consumer_block": "b",
        "handshake_protocol": "mem_write", "data_width_bits": 8,
        "fields": [{"name": "wdata", "width": 8, "msb": 7, "lsb": 0}],
        "sideband_signals": [{"name": "wr_ready"}],
    }]}
    v = _check_interface_family_coherence(diagram, contracts)
    assert any("a->b" in x["violation"] for x in v)


def test_family_coherence_gate_off_is_noop(monkeypatch):
    monkeypatch.setenv("CORESMITH_INTERFACE_FAMILY_GATE", "0")
    diagram = _generic_family_diagram()
    assert _check_interface_family_coherence(
        diagram, _mislabeled_specialist_contracts()
    ) == []


def test_family_coherence_no_contracts_noop():
    # No contracts -> no-op (does not perturb existing constraint-check tests).
    assert _check_interface_family_coherence(_generic_family_diagram(), {}) == []


# ---------------------------------------------------------------------------
# flow_control_policy derives MONOTONICALLY from the FINAL handshake_protocol
# family (no-backpressure fix).
#
# A no-backpressure family (mem_write / valid_only / static) is always-accepted,
# so its flow_control_policy MUST be free_running / no-feedback / depth-0 --
# even when the LLM proposed request_response + feedback_cycle=true because a
# SEPARATE later completion/done event exists elsewhere in the graph. The
# neutralization must fire off the contract's FINAL family, so it still fires
# when the family label came only from the CONTRACT (the block-diagram edge
# omitted handshake_protocol or its (producer,consumer) pair wasn't in the
# intent map). Genuine streaming edges (srdy_drdy / axi_stream) are UNTOUCHED.
# ---------------------------------------------------------------------------


def _mixed_family_contracts_with_completion_mislabel():
    """A valid_only strobe and a static bundle the specialist WRONGLY froze as
    request_response + feedback_cycle=true (it inferred a response because a
    separate done/completion strobe exists), a mem_write also mislabeled
    request_response, a req_resp read (legitimately request_response), and a
    genuine srdy_drdy stream on a cycle carrying an elastic_fifo."""
    return {
        "contracts": [
            {  # valid_only strobe, mislabeled request_response
                "producer_block": "ctrl", "consumer_block": "core",
                "handshake_protocol": "valid_only", "data_width_bits": 16,
                "fields": [{"name": "cmd", "width": 16, "msb": 15, "lsb": 0}],
                "flow_control_policy": {
                    "semantics": "request_response",
                    "feedback_cycle": True,
                    "consumer_can_stall": True,
                    "min_buffer_depth_beats": 4,
                },
            },
            {  # static bundle, mislabeled request_response
                "producer_block": "core", "consumer_block": "ctrl",
                "handshake_protocol": "static", "data_width_bits": 8,
                "fields": [{"name": "status", "width": 8, "msb": 7, "lsb": 0}],
                "flow_control_policy": {
                    "semantics": "request_response",
                    "feedback_cycle": True,
                },
            },
            {  # mem_write, mislabeled request_response
                "producer_block": "core", "consumer_block": "store",
                "handshake_protocol": "mem_write", "data_width_bits": 32,
                "fields": [{"name": "wdata", "width": 32, "msb": 31, "lsb": 0}],
                "flow_control_policy": {
                    "semantics": "request_response",
                    "feedback_cycle": True,
                    "producer_can_stall": True,
                },
            },
            {  # req_resp read -- request_response is CORRECT, must be preserved
                "producer_block": "core", "consumer_block": "store",
                "handshake_protocol": "req_resp", "data_width_bits": 32,
                "fields": [{"name": "rdata", "width": 32, "msb": 31, "lsb": 0}],
                "flow_control_policy": {"semantics": "request_response"},
            },
            {  # genuine streaming edge on a cycle -- elastic_fifo is REQUIRED
                "producer_block": "sa", "consumer_block": "sb",
                "handshake_protocol": "srdy_drdy", "data_width_bits": 8,
                "fields": [{"name": "s", "width": 8, "msb": 7, "lsb": 0}],
                "flow_control_policy": {
                    "semantics": "elastic_fifo",
                    "min_buffer_depth_beats": 4,
                    "feedback_cycle": True,
                    "consumer_can_stall": True,
                },
            },
        ]
    }


def _fc(contract):
    return contract.get("flow_control_policy") or {}


def test_flow_policy_monotonic_from_final_family(monkeypatch):
    # Default-ON. The block diagram OMITS handshake_protocol on every edge, so
    # the families come ONLY from the contracts -- the diagram-intent-only guard
    # would skip them. Neutralization must still fire off the final family.
    monkeypatch.delenv("CORESMITH_INTERFACE_FAMILY_PROPAGATION", raising=False)
    diagram = {"connections": [
        {"from": "ctrl", "to": "core"},
        {"from": "core", "to": "ctrl"},
        {"from": "core", "to": "store"},
        {"from": "sa", "to": "sb"},
    ]}
    out, notes = _propagate_edge_families(
        _mixed_family_contracts_with_completion_mislabel(), diagram["connections"]
    )
    by_fam = {}
    for c in out["contracts"]:
        by_fam.setdefault(c["handshake_protocol"], []).append(c)

    # Every no-backpressure family ends free_running / no-feedback / depth-0 /
    # no-stall -- regardless of the completion-event-driven request_response the
    # LLM proposed and regardless of the diagram omitting the family.
    for fam in ("valid_only", "static", "mem_write"):
        for c in by_fam[fam]:
            fc = _fc(c)
            assert fc["semantics"] == "free_running", (fam, fc)
            assert fc.get("feedback_cycle") is False, (fam, fc)
            assert int(fc.get("min_buffer_depth_beats") or 0) == 0, (fam, fc)
            assert not fc.get("consumer_can_stall"), (fam, fc)
            assert not fc.get("producer_can_stall"), (fam, fc)

    # req_resp is UNTOUCHED -- request/response is reserved for it.
    assert _fc(by_fam["req_resp"][0])["semantics"] == "request_response"

    # A genuine streaming edge is UNTOUCHED: its elastic_fifo + feedback_cycle
    # + depth survive (streaming backpressure is legitimate).
    sfc = _fc(by_fam["srdy_drdy"][0])
    assert sfc["semantics"] == "elastic_fifo"
    assert sfc.get("feedback_cycle") is True
    assert int(sfc.get("min_buffer_depth_beats") or 0) == 4
    assert notes, "propagation should record what it neutralized"


def test_flow_policy_monotonic_gate_off_is_noop(monkeypatch):
    # Gate OFF -> the LLM's raw (wrong) policy is trusted verbatim; no rewrite.
    monkeypatch.setenv("CORESMITH_INTERFACE_FAMILY_PROPAGATION", "0")
    diagram = {"connections": [
        {"from": "ctrl", "to": "core"},
        {"from": "core", "to": "ctrl"},
        {"from": "core", "to": "store"},
        {"from": "sa", "to": "sb"},
    ]}
    out, notes = _propagate_edge_families(
        _mixed_family_contracts_with_completion_mislabel(), diagram["connections"]
    )
    assert notes == []
    for c in out["contracts"]:
        if c["handshake_protocol"] in ("valid_only", "static", "mem_write"):
            assert _fc(c)["semantics"] == "request_response"


# ---------------------------------------------------------------------------
# Regression: a 20-edge design with the same family mix as the live escalation
# (6 static + 7 valid_only + 3 mem_write + 4 req_resp) produces ZERO
# protocol-coherence violations AFTER propagation. GENERIC synthetic names --
# no benchmark / exercise / block names from any real design.
# ---------------------------------------------------------------------------

def _twenty_edge_family_mix():
    families = (
        ["mem_write"] * 3 + ["req_resp"] * 4 + ["valid_only"] * 7 + ["static"] * 6
    )
    conns: list[dict] = []
    contracts: list[dict] = []
    for i, fam in enumerate(families):
        p, c = f"prod_{i:02d}", f"cons_{i:02d}"
        conns.append(
            {"from": p, "to": c, "handshake_protocol": fam, "data_width": 8}
        )
        if fam == "mem_write":
            fc = {"semantics": "free_running", "feedback_cycle": False,
                  "min_buffer_depth_beats": 0}
        elif fam == "req_resp":
            fc = {"semantics": "request_response", "feedback_cycle": False}
        else:  # valid_only / static: the completion-event-driven mislabel
            fc = {"semantics": "request_response", "feedback_cycle": True,
                  "consumer_can_stall": True}
        contracts.append({
            "producer_block": p, "consumer_block": c,
            "handshake_protocol": fam, "data_width_bits": 8,
            "fields": [{"name": "d", "width": 8, "msb": 7, "lsb": 0}],
            "flow_control_policy": fc,
        })
    return {"blocks": [], "connections": conns}, {"contracts": contracts}


def test_twenty_edge_family_mix_zero_coherence_violations(monkeypatch):
    monkeypatch.delenv("CORESMITH_INTERFACE_FAMILY_PROPAGATION", raising=False)
    monkeypatch.delenv("CORESMITH_INTERFACE_FAMILY_GATE", raising=False)
    diagram, raw = _twenty_edge_family_mix()

    # The RAW LLM contracts fail the honest gate: 6 static + 7 valid_only edges
    # carry request_response -> 13 protocol-coherence errors.
    before = _check_interface_family_coherence(diagram, raw)
    assert len(before) == 13, [v["violation"] for v in before]

    # After the generator neutralizes monotonically from the final family, the
    # gate passes cleanly -- the 13-error class is GONE.
    out, _notes = _propagate_edge_families(raw, diagram["connections"])
    after = _check_interface_family_coherence(diagram, out)
    assert after == [], [v["violation"] for v in after]

    # req_resp policy is preserved on all 4 read edges.
    reqresp = [
        c for c in out["contracts"] if c["handshake_protocol"] == "req_resp"
    ]
    assert len(reqresp) == 4
    assert all(
        (c.get("flow_control_policy") or {}).get("semantics") == "request_response"
        for c in reqresp
    )


# ---------------------------------------------------------------------------
# PARALLEL EDGES between one block pair + addressed-read-with-response family.
#
# The block-diagram family propagation (and the coherence gate) used to key the
# edge intent by (producer_block, consumer_block) ALONE, so two PARALLEL edges
# between the same pair -- e.g. an addressed 1R1W SRAM read (req_resp) and a
# direct write (mem_write) -- overwrote each other last-wins. The read then
# inherited the write's mem_write family and got free_running-ified while still
# carrying its rdata/rvalid/fault response channel, which the honest gate
# correctly rejected (2 errors) and blocked the run. Port-aware keying keeps
# each parallel edge's own family; response-preservation is defense-in-depth.
# GENERIC synthetic names -- no benchmark / exercise / real block names.
# ---------------------------------------------------------------------------


def _parallel_pair_diagram():
    """Two PARALLEL edges between the same block pair: an addressed 1R1W SRAM
    read (req_resp) and a direct write (mem_write)."""
    return {
        "blocks": [{"name": "datapath_core"}, {"name": "scratch_sram"}],
        "connections": [
            {"from": "datapath_core", "to": "scratch_sram",
             "interface": "line_read", "handshake_protocol": "req_resp",
             "data_width": 24},
            {"from": "datapath_core", "to": "scratch_sram",
             "interface": "line_write", "handshake_protocol": "mem_write",
             "data_width": 20},
        ],
    }


def _parallel_pair_contracts():
    """What the specialist froze BEFORE the fix: the parallel-edge collision
    collapsed the read onto the write's mem_write family, but the read still
    carries a full response channel (addr/ren + rdata/rvalid/fault, no ready)."""
    return {"contracts": [
        {
            "edge_id": "datapath_core__line_read__to__scratch_sram__line_read",
            "producer_block": "datapath_core", "producer_port": "line_read",
            "consumer_block": "scratch_sram", "consumer_port": "line_read",
            # DELIBERATELY WRONG: mislabeled mem_write via the collision.
            "handshake_protocol": "mem_write", "data_width_bits": 24,
            "fields": [
                {"name": "addr", "width": 12, "msb": 23, "lsb": 12},
                {"name": "ren", "width": 1, "msb": 11, "lsb": 11},
                {"name": "rdata", "width": 9, "msb": 10, "lsb": 2},
                {"name": "rvalid", "width": 1, "msb": 1, "lsb": 1},
                {"name": "fault", "width": 1, "msb": 0, "lsb": 0},
            ],
            "flow_control_policy": {"semantics": "free_running",
                                    "min_buffer_depth_beats": 0},
        },
        {
            "edge_id": "datapath_core__line_write__to__scratch_sram__line_write",
            "producer_block": "datapath_core", "producer_port": "line_write",
            "consumer_block": "scratch_sram", "consumer_port": "line_write",
            "handshake_protocol": "mem_write", "data_width_bits": 20,
            "fields": [
                {"name": "addr", "width": 12, "msb": 19, "lsb": 8},
                {"name": "wdata", "width": 8, "msb": 7, "lsb": 0},
            ],
            "flow_control_policy": {"semantics": "free_running",
                                    "min_buffer_depth_beats": 0},
        },
    ]}


def test_parallel_edges_keep_distinct_families(monkeypatch):
    # Port-aware anchoring: the read edge resolves to its OWN declared req_resp
    # instead of inheriting the sibling write edge's mem_write.
    monkeypatch.delenv("CORESMITH_INTERFACE_FAMILY_PROPAGATION", raising=False)
    monkeypatch.delenv("CORESMITH_INTERFACE_FAMILY_GATE", raising=False)
    diagram = _parallel_pair_diagram()
    out, notes = _propagate_edge_families(
        _parallel_pair_contracts(), diagram["connections"]
    )
    by_port = {c["producer_port"]: c for c in out["contracts"]}
    read = by_port["line_read"]
    write = by_port["line_write"]

    # The addressed read-with-response is req_resp -- NOT mem_write.
    assert read["handshake_protocol"] == "req_resp"
    assert read["flow_control_policy"]["semantics"] == "request_response"
    # Its response channel survives.
    assert {"rdata", "rvalid"} <= {f["name"] for f in read["fields"]}

    # The true write edge is UNAFFECTED (no regression to the flow-policy fix).
    assert write["handshake_protocol"] == "mem_write"
    assert write["flow_control_policy"]["semantics"] == "free_running"
    assert int(write["flow_control_policy"].get("min_buffer_depth_beats") or 0) == 0

    # The honest coherence gate now returns ZERO errors for this pair.
    assert _check_interface_family_coherence(diagram, out) == []
    assert notes, "propagation should record the req_resp derivation"


def test_parallel_read_write_propagation_gate_off_is_noop(monkeypatch):
    # PROPAGATION gate OFF -> the mislabeled mem_write read label is left
    # verbatim (pre-fix behavior); no req_resp derivation, no notes.
    monkeypatch.setenv("CORESMITH_INTERFACE_FAMILY_PROPAGATION", "0")
    diagram = _parallel_pair_diagram()
    out, notes = _propagate_edge_families(
        _parallel_pair_contracts(), diagram["connections"]
    )
    assert notes == []
    by_port = {c["producer_port"]: c for c in out["contracts"]}
    assert by_port["line_read"]["handshake_protocol"] == "mem_write"
    assert by_port["line_read"]["flow_control_policy"]["semantics"] == "free_running"


def test_read_with_response_preserved_when_intent_missing(monkeypatch):
    # Diagram declares NO family on either edge -> nothing to anchor to. The
    # response channel ALONE must force the read to req_resp; a plain write
    # (no response field) is left mem_write. Defense-in-depth for a mislabeled
    # read whose diagram intent is missing/ambiguous.
    monkeypatch.delenv("CORESMITH_INTERFACE_FAMILY_PROPAGATION", raising=False)
    monkeypatch.delenv("CORESMITH_INTERFACE_FAMILY_GATE", raising=False)
    diagram = {"connections": [
        {"from": "core", "to": "mem", "interface": "rd"},
        {"from": "core", "to": "mem", "interface": "wr"},
    ]}
    contracts = {"contracts": [
        {"producer_block": "core", "producer_port": "rd",
         "consumer_block": "mem", "consumer_port": "rd",
         "handshake_protocol": "mem_write", "data_width_bits": 21,
         "fields": [
             {"name": "addr", "width": 11, "msb": 20, "lsb": 10},
             {"name": "rdata", "width": 8, "msb": 9, "lsb": 2},
             {"name": "rvalid", "width": 1, "msb": 1, "lsb": 1},
             {"name": "fault", "width": 1, "msb": 0, "lsb": 0},
         ],
         "flow_control_policy": {"semantics": "free_running"}},
        {"producer_block": "core", "producer_port": "wr",
         "consumer_block": "mem", "consumer_port": "wr",
         "handshake_protocol": "mem_write", "data_width_bits": 16,
         "fields": [
             {"name": "addr", "width": 8, "msb": 15, "lsb": 8},
             {"name": "wdata", "width": 8, "msb": 7, "lsb": 0},
         ],
         "flow_control_policy": {"semantics": "free_running"}},
    ]}
    out, _notes = _propagate_edge_families(contracts, diagram["connections"])
    by_port = {c["producer_port"]: c for c in out["contracts"]}
    assert by_port["rd"]["handshake_protocol"] == "req_resp"
    assert by_port["rd"]["flow_control_policy"]["semantics"] == "request_response"
    # A real write (no response field) is NOT upgraded.
    assert by_port["wr"]["handshake_protocol"] == "mem_write"
    assert by_port["wr"]["flow_control_policy"]["semantics"] == "free_running"
    assert _check_interface_family_coherence(diagram, out) == []


def test_mem_write_contract_with_response_rejected(monkeypatch):
    # A hand-constructed mem_write contract that carries a response channel
    # (rdata/rvalid) is REJECTED deterministically by the honest gate.
    monkeypatch.delenv("CORESMITH_INTERFACE_FAMILY_GATE", raising=False)
    diagram = {"connections": [
        {"from": "a", "to": "b", "interface": "rd",
         "handshake_protocol": "mem_write"},
    ]}
    contracts = {"contracts": [{
        "producer_block": "a", "producer_port": "rd",
        "consumer_block": "b", "consumer_port": "rd",
        "handshake_protocol": "mem_write", "data_width_bits": 10,
        "fields": [
            {"name": "addr", "width": 8, "msb": 9, "lsb": 2},
            {"name": "rdata", "width": 1, "msb": 1, "lsb": 1},
            {"name": "rvalid", "width": 1, "msb": 0, "lsb": 0},
        ],
    }]}
    v = _check_interface_family_coherence(diagram, contracts)
    assert v, "a mem_write contract carrying a response channel must be rejected"
    assert any("a->b" in x["violation"] for x in v)
    assert all(
        x["check"] == "inter_block_payload_protocol_coherence" for x in v
    )
    assert any("response" in x["violation"].lower() for x in v)


def test_mem_write_with_response_gate_off_is_noop(monkeypatch):
    monkeypatch.setenv("CORESMITH_INTERFACE_FAMILY_GATE", "0")
    diagram = {"connections": [
        {"from": "a", "to": "b", "interface": "rd",
         "handshake_protocol": "mem_write"},
    ]}
    contracts = {"contracts": [{
        "producer_block": "a", "producer_port": "rd",
        "consumer_block": "b", "consumer_port": "rd",
        "handshake_protocol": "mem_write", "data_width_bits": 2,
        "fields": [
            {"name": "rdata", "width": 1, "msb": 1, "lsb": 1},
            {"name": "rvalid", "width": 1, "msb": 0, "lsb": 0},
        ],
    }]}
    assert _check_interface_family_coherence(diagram, contracts) == []


# ---------------------------------------------------------------------------
# Retry routing: an interface-contract escalation retries by regenerating the
# CONTRACTS (Interface Definition), not the block diagram.
# ---------------------------------------------------------------------------

def _ifd_escalation_state():
    return {"constraint_result": {
        "source": "interface_definition",
        "has_structural": True,
        "violations": [{"type": "over_wide_bus", "category": "structural",
                        "violation": "x"}],
    }}


def test_retry_gate_default_on():
    import os
    os.environ.pop("CORESMITH_INTERFACE_RETRY_ROUTING", None)
    assert ag._interface_retry_to_definition_enabled() is True


def test_retry_gate_can_be_disabled(monkeypatch):
    monkeypatch.setenv("CORESMITH_INTERFACE_RETRY_ROUTING", "0")
    assert ag._interface_retry_to_definition_enabled() is False


def test_interface_escalation_retry_routes_to_interface_definition(monkeypatch):
    monkeypatch.delenv("CORESMITH_INTERFACE_RETRY_ROUTING", raising=False)
    assert (
        ag._escalation_retry_target(_ifd_escalation_state())
        == "Interface Definition"
    )


def test_interface_escalation_retry_gate_off_falls_back_to_block_diagram(monkeypatch):
    monkeypatch.setenv("CORESMITH_INTERFACE_RETRY_ROUTING", "0")
    assert ag._escalation_retry_target(_ifd_escalation_state()) == "Block Diagram"


def test_constraint_check_structural_escalation_stays_block_diagram(monkeypatch):
    # A real diagram/topology defect (constraint-check-sourced, no
    # interface_definition source) must still regenerate the block diagram.
    monkeypatch.delenv("CORESMITH_INTERFACE_RETRY_ROUTING", raising=False)
    state = {"constraint_result": {
        "has_structural": True,
        "violations": [{"category": "structural", "violation": "topology broken"}],
    }}
    assert ag._escalation_retry_target(state) == "Block Diagram"
