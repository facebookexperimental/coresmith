# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the performance roofline -- the throughput floor.

Covers: the machine-readable ``perf`` block parser, the modulo-scheduling
roofline math (RecMII/ResMII/II, peak cyc/op, binding constraint), the FRD
PERF-NNN requirement check + the self-imposed budget when no cap is declared,
per-block emission, and fail-open behaviour on a missing/absent spec.
"""
from __future__ import annotations

import json

import pytest

from orchestrator.langgraph import perf_roofline as pr


# Deterministic delay model for tests: ns scales with width so tests never need
# a real characterized PDK.  mul is the fat op; everything else is thin.
def _dfn(op, width, pdk=None):
    if op == "mul":
        return width / 4.0       # mul16 -> 4.0 ns
    return width / 100.0         # add16 -> 0.16 ns ... tiny, always fits


@pytest.fixture(autouse=True)
def _fast_delay(monkeypatch):
    monkeypatch.setattr(pr, "_predict_delay", _dfn)


PERF_SPEC_SRC = """
Some uArch prose describing the datapath.

```perf
{ "op_unit": "block", "target_clock_mhz": 100, "iterations": 64,
  "pipeline_chain": [["mul", 16], ["add", 16]],
  "resources": [{"name": "mac_mul", "op": "mul", "width": 16, "instances": 1,
                 "uses_per_iter": 64},
                {"name": "mac_add", "op": "add", "width": 16, "instances": 1,
                 "uses_per_iter": 64}],
  "rec_cycles": [{"name": "acc", "ops": [["add", 16]], "distance": 1}],
  "drain_cyc": 2, "io_framing_cyc": 4,
  "perf_req_id": "PERF-001", "perf_req_cyc_per_op": 100,
  "declared_cyc_per_op": 512 }
```
More prose.
"""


def test_parse_perf_spec():
    spec = pr.parse_perf_spec(PERF_SPEC_SRC, block_name="demo")
    assert spec is not None
    assert spec.iterations == 64
    assert spec.op_unit == "block"
    assert spec.perf_req_id == "PERF-001"
    assert spec.perf_req_cyc_per_op == 100
    assert spec.declared_cyc_per_op == 512
    assert len(spec.pipeline_chain) == 2
    assert len(spec.resources) == 2
    assert len(spec.rec_cycles) == 1


def test_parse_missing_block_is_none():
    assert pr.parse_perf_spec("no perf block here", "b") is None
    assert pr.parse_perf_spec("```perf\nnot json\n```", "b") is None


def test_resmii_binds_when_resource_serialized():
    # 64 uses / 1 instance -> ResMII 64; the accumulator recurrence is II=1.
    spec = pr.parse_perf_spec(PERF_SPEC_SRC, block_name="demo")
    m = pr.analyze(spec)
    assert m["ResMII"] == 64
    assert m["RecMII"] == 1
    assert m["II_min"] == 64
    # peak = iterations*II + fill + drain + io = 64*64 + (D-1) + 2 + 4
    assert m["cyc_per_op_peak"] == 64 * 64 + (m["pipeline_depth"] - 1) + 2 + 4
    assert m["binding_constraint"]["type"] == "ResMII"
    assert m["binding_constraint"]["name"] == "mac_mul"


def test_widening_instances_lowers_ii():
    # Same spec but 64 multiplier + adder instances -> ResMII 1 -> II=1.
    src = PERF_SPEC_SRC.replace('"instances": 1', '"instances": 64')
    spec = pr.parse_perf_spec(src, block_name="demo")
    m = pr.analyze(spec)
    assert m["ResMII"] == 1
    assert m["II_min"] == 1
    assert m["cyc_per_op_peak"] < 64 * 64  # far fewer cycles when pipelined


def test_meets_throughput_req_false_when_declared_over_cap():
    spec = pr.parse_perf_spec(PERF_SPEC_SRC, block_name="demo")
    m = pr.analyze(spec)
    # declared 512 vs cap 100 -> misses
    assert m["meets_throughput_req"] is False
    assert m["perf_req_source"] == "frd"
    assert any("MISSES" in n for n in m["sanity_notes"])


def _crypto_delay(op, width, pdk=None):
    """Realistic-ish sky130 delays for the AES round op vocabulary (ns).

    lut8 ~= 1.36, gfmul8 ~= 1.44, xortree32 ~= 0.74 -> a ~3.5 ns round cone that
    fits a 20 ns clock in one stage but not a 3 ns clock.
    """
    return {"lut": 0.17, "gfmul": 0.18, "xortree": 0.023}.get(op, 0.02) * width


def _crypto_round_spec(T_ns):
    # SubBytes(S-box lut) -> MixColumns(GF-mul + column xor) round recurrence.
    cone = [pr.Op("lut", 8), pr.Op("gfmul", 8), pr.Op("xortree", 32)]
    return pr.DataflowSpec(
        name="aes_round", T_ns=T_ns, margin_ns=round(0.1 * T_ns, 4),
        iterations=11, pipeline_chain=list(cone),
        rec_cycles=[pr.RecCycle("round", list(cone), distance=1)],
    )


def test_crypto_cone_ii_couples_to_period(monkeypatch):
    """engine-v31 step 2: the round recurrence latency is Fmax-COUPLED.

    The SAME crypto op cone gives II=1 at a 20 ns clock (cone fits one period)
    and II>=2 at a 3 ns clock (cone spans >1 stage) -- with realistic LUT/GF/xor
    delays, not a prose 1-cycle latency. This is the property that keeps a
    declared 11-cycle II=1 AES plan timing-feasible by construction.
    """
    monkeypatch.setattr(pr, "_predict_delay", _crypto_delay)

    m20 = pr.analyze(_crypto_round_spec(20.0))
    assert m20["II_min"] == 1, m20["recmii_terms"]
    assert m20["RecMII"] == 1
    # the round cone fits inside one stage budget at 20 ns
    assert 0 < m20["round_cone_delay_ns"] <= m20["stage_budget_ns"]
    assert m20["recmii_terms"][0]["priced"] is True

    m3 = pr.analyze(_crypto_round_spec(3.0))
    assert m3["II_min"] >= 2, m3["recmii_terms"]
    assert m3["RecMII"] >= 2
    # the same cone no longer fits a single 3 ns stage
    assert m3["round_cone_delay_ns"] > m3["stage_budget_ns"]


def test_unpriced_recurrence_is_flagged(monkeypatch):
    """A recurrence whose op cone the PDK cannot price is surfaced as NOT
    Fmax-coupled (priced=False + sanity note) rather than silently trusted."""
    def _partial(op, width, pdk=None):
        return None if op == "mystery" else width / 100.0
    monkeypatch.setattr(pr, "_predict_delay", _partial)
    spec = pr.DataflowSpec(
        name="b", T_ns=10.0, margin_ns=1.0, iterations=4,
        pipeline_chain=[pr.Op("add", 8)],
        rec_cycles=[pr.RecCycle("r", [pr.Op("mystery", 8)], distance=1)],
    )
    m = pr.analyze(spec)
    assert m["recmii_terms"][0]["priced"] is False
    assert any("not fully PDK-priced" in n for n in m["sanity_notes"])


def test_self_imposed_budget_when_no_cap():
    # Drop the FRD cap: the engine self-imposes peak * derate.
    src = PERF_SPEC_SRC.replace(', "perf_req_cyc_per_op": 100', "")
    spec = pr.parse_perf_spec(src, block_name="demo")
    m = pr.analyze(spec)
    assert m["perf_req_source"] == "self_imposed"
    assert m["perf_req_cyc_per_op"] == m["self_imposed_budget_cyc_per_op"]
    # "no cap" is still a target: budget = ceil(peak * derate)
    import math
    assert m["perf_req_cyc_per_op"] == math.ceil(m["cyc_per_op_peak"] * m["derate"])


def test_emit_perf_model(tmp_path):
    spec_dir = tmp_path / "arch" / "uarch_specs"
    spec_dir.mkdir(parents=True)
    (spec_dir / "demo.md").write_text(PERF_SPEC_SRC)
    out = pr.emit_perf_model(tmp_path, "demo", target_clock_mhz=100.0)
    assert out is not None and out.exists()
    doc = json.loads(out.read_text())
    assert doc["block"] == "demo"
    assert doc["meets_throughput_req"] is False
    # round-trip read
    assert pr.read_perf_model(tmp_path, "demo")["cyc_per_op_peak"] == doc["cyc_per_op_peak"]


def test_emit_fail_open_on_missing_spec(tmp_path):
    # No uarch spec on disk -> nothing to emit, no raise.
    assert pr.emit_perf_model(tmp_path, "nonexistent") is None
    assert pr.read_perf_model(tmp_path, "nonexistent") is None


def test_roofline_enabled_env(monkeypatch):
    monkeypatch.delenv("CORESMITH_PERF_ROOFLINE", raising=False)
    assert pr.roofline_enabled() is False
    monkeypatch.setenv("CORESMITH_PERF_ROOFLINE", "1")
    assert pr.roofline_enabled() is True
