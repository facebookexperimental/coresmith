# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""A2: the contract ``timing`` object is normalised per family and gated."""
from __future__ import annotations

from orchestrator.architecture.specialists import contract_timing as ct


def _c(fam, timing=None, sideband=()):
    c = {"edge_id": "a__x__to__b__x", "handshake_protocol": fam,
         "sideband_signals": [{"name": s} for s in sideband]}
    if timing is not None:
        c["timing"] = timing
    return c


def test_req_resp_exact_expands_to_min_max():
    c = _c("req_resp", {"req_to_rsp_cycles": {"exact": 1}})
    ct.normalize_timing(c)
    assert c["timing"]["req_to_rsp_cycles"] == {"min": 1, "max": 1, "exact": 1}
    assert c["timing"]["ordering"] == "in_order"
    assert c["timing"]["valid_to_ready_max_stall"] is None
    assert ct.timing_violations(c) == []
    assert "exactly 1 cycle" in ct.timing_summary(c)


def test_req_resp_scalar_and_bounded_forms():
    c = _c("req_resp", {"req_to_rsp_cycles": 3})
    ct.normalize_timing(c)
    assert c["timing"]["req_to_rsp_cycles"] == {"min": 3, "max": 3, "exact": 3}
    c = _c("req_resp", {"req_to_rsp_cycles": {"min": 2, "max": 1}})
    _, notes = ct.normalize_timing(c)
    assert c["timing"]["req_to_rsp_cycles"]["max"] == 2 and notes
    c = _c("req_resp", {"req_to_rsp_cycles": {"max": 4}})
    ct.normalize_timing(c)
    assert c["timing"]["req_to_rsp_cycles"] == {"min": 0, "max": 4, "exact": None}


def test_streaming_defaults_and_burst_from_sideband():
    c = _c("axi_stream", sideband=("tlast", "tuser"))
    ct.normalize_timing(c)
    t = c["timing"]
    assert t["valid_hold_until_ready"] is True and t["burst"]["last_signal"] == "tlast"
    assert t["req_to_rsp_cycles"] is None and t["reset_idle_cycles"] == 1
    c = _c("srdy_drdy", {"valid_to_ready_max_stall": "8", "req_to_rsp_cycles": 2})
    _, notes = ct.normalize_timing(c)
    assert c["timing"]["valid_to_ready_max_stall"] == 8
    assert c["timing"]["req_to_rsp_cycles"] is None and notes


def test_always_accepted_and_static_strip_stall_fields():
    for fam in ("mem_write", "valid_only"):
        c = _c(fam, {"valid_to_ready_max_stall": 3, "valid_hold_until_ready": True})
        _, notes = ct.normalize_timing(c)
        assert c["timing"]["valid_to_ready_max_stall"] is None
        assert c["timing"]["valid_hold_until_ready"] is False and notes
    c = _c("static", {"ordering": "in_order", "burst": {"last_signal": "x"}})
    ct.normalize_timing(c)
    assert c["timing"]["ordering"] == "n/a" and c["timing"]["reset_idle_cycles"] == 0
    assert c["timing"]["burst"] == {"last_signal": None, "max_beats": None}


def test_missing_latency_is_gated(monkeypatch):
    c = _c("req_resp")
    ct.normalize_timing(c)
    monkeypatch.setenv("CORESMITH_CONTRACT_TIMING_GATE", "1")
    assert [v["type"] for v in ct.timing_violations(c)] == ["missing_timing"]
    monkeypatch.setenv("CORESMITH_CONTRACT_TIMING_GATE", "0")
    assert ct.timing_violations(c) == []
    assert ct.timing_violations(_c("valid_only")) == []


def test_normalize_is_idempotent():
    c = _c("req_resp", {"req_to_rsp_cycles": {"min": 1, "max": 2}})
    ct.normalize_timing(c)
    first = dict(c["timing"])
    changed, _ = ct.normalize_timing(c)
    assert not changed and c["timing"] == first
