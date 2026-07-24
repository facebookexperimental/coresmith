# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the quantitative fidelity tier of the composition gate
(microarchitecture restructure, step 2).

Distinguishes a within-budget derate (PASS, recorded in the derate ledger) from
a below-budget break (FAIL) and from an above-escalate-floor derate (PASS but
flagged for chip-lead sign-off) -- which the legacy binary tiers cannot. All
deterministic, no LLM / no gate sim.
"""
from __future__ import annotations

import json

import orchestrator.architecture.fidelity as fz


def _metric_file(tmp_path, body="return float(observed)"):
    p = tmp_path / "metric.py"
    p.write_text(f"def fidelity(expected, observed):\n    {body}\n", encoding="utf-8")
    return str(p)


def test_gate_off_by_default(monkeypatch):
    monkeypatch.delenv("CORESMITH_FIDELITY_GATE", raising=False)
    assert fz.fidelity_gate_enabled() is False
    monkeypatch.setenv("CORESMITH_FIDELITY_GATE", "1")
    assert fz.fidelity_gate_enabled() is True


def test_no_metric_returns_none(tmp_path, monkeypatch):
    monkeypatch.delenv("CORESMITH_FIDELITY_METRIC", raising=False)
    assert fz.compute_fidelity_derate(str(tmp_path), b"x", b"y") is None


def test_within_budget_passes_and_escalates(tmp_path, monkeypatch):
    monkeypatch.setenv("CORESMITH_FIDELITY_METRIC", _metric_file(tmp_path))
    monkeypatch.setenv(
        "CORESMITH_FIDELITY_BUDGET",
        json.dumps({"floor": 40, "escalate_floor": 42, "ideal": 44, "metric": "psnr_db"}),
    )
    d = fz.compute_fidelity_derate(str(tmp_path), b"ref", 41.0)
    assert d["within_budget"] is True
    assert d["escalate"] is True  # 41 passes floor 40 but is below escalate_floor 42
    assert d["measured"] == 41.0
    assert d["metric"] == "psnr_db"
    assert round(d["derate_pct"], 1) == round(100 * (44 - 41) / 44, 1)


def test_below_floor_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("CORESMITH_FIDELITY_METRIC", _metric_file(tmp_path))
    monkeypatch.setenv("CORESMITH_FIDELITY_BUDGET", json.dumps({"floor": 40}))
    d = fz.compute_fidelity_derate(str(tmp_path), b"ref", 39.0)
    assert d["within_budget"] is False


def test_clean_pass_no_escalate(tmp_path, monkeypatch):
    monkeypatch.setenv("CORESMITH_FIDELITY_METRIC", _metric_file(tmp_path))
    monkeypatch.setenv(
        "CORESMITH_FIDELITY_BUDGET", json.dumps({"floor": 40, "escalate_floor": 42})
    )
    d = fz.compute_fidelity_derate(str(tmp_path), b"ref", 43.0)
    assert d["within_budget"] is True and d["escalate"] is False


def test_lower_is_better_error_metric(tmp_path, monkeypatch):
    monkeypatch.setenv("CORESMITH_FIDELITY_METRIC", _metric_file(tmp_path))
    monkeypatch.setenv(
        "CORESMITH_FIDELITY_BUDGET", json.dumps({"floor": 10, "direction": "lower"})
    )
    assert fz.compute_fidelity_derate(str(tmp_path), b"ref", 5.0)["within_budget"] is True
    assert fz.compute_fidelity_derate(str(tmp_path), b"ref", 15.0)["within_budget"] is False


def test_budget_from_json_path(tmp_path, monkeypatch):
    bp = tmp_path / "budget.json"
    bp.write_text(json.dumps({"floor": 40}), encoding="utf-8")
    monkeypatch.setenv("CORESMITH_FIDELITY_BUDGET", str(bp))
    b = fz.resolve_fidelity_budget(str(tmp_path))
    assert b["floor"] == 40.0 and b["direction"] == "higher"


def test_metric_exception_is_worst_case(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "CORESMITH_FIDELITY_METRIC",
        _metric_file(tmp_path, body="raise ValueError('boom')"),
    )
    monkeypatch.setenv("CORESMITH_FIDELITY_BUDGET", json.dumps({"floor": 40}))
    d = fz.compute_fidelity_derate(str(tmp_path), b"ref", b"obs")
    assert d["within_budget"] is False  # higher-is-better -> -inf < floor


def test_ledger_records_and_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("CORESMITH_FIDELITY_METRIC", _metric_file(tmp_path))
    monkeypatch.setenv(
        "CORESMITH_FIDELITY_BUDGET",
        json.dumps({"floor": 40, "escalate_floor": 42, "ideal": 44}),
    )
    fid = fz.compute_fidelity_derate(str(tmp_path), b"ref", 41.0)
    fz.write_derate_ledger(str(tmp_path), fid, byte_exact=False)
    led = tmp_path / ".coresmith" / "derate_ledger.json"
    doc = json.loads(led.read_text())
    assert doc["integrated_within_budget"] is True
    assert doc["integrated_escalate"] is True
    assert doc["entries"][0]["measured"] == 41.0
    fz.write_derate_ledger(str(tmp_path), fid, byte_exact=False)  # re-run
    doc2 = json.loads(led.read_text())
    assert len(doc2["entries"]) == 1  # replaced, not appended


def test_byte_exact_ledger_is_clean(tmp_path):
    fz.write_derate_ledger(str(tmp_path), None, byte_exact=True)
    doc = json.loads((tmp_path / ".coresmith" / "derate_ledger.json").read_text())
    assert doc["integrated_within_budget"] is True
    assert doc["integrated_escalate"] is False


def test_gate_tier_appends_violation_below_budget(tmp_path, monkeypatch):
    import orchestrator.architecture.model_integration as mi

    monkeypatch.setenv(
        "CORESMITH_FIDELITY_METRIC", _metric_file(tmp_path, body="return float(len(observed))")
    )
    # below budget: len(observed)=3 < floor 100 -> FAIL (violation appended)
    monkeypatch.setenv("CORESMITH_FIDELITY_BUDGET", json.dumps({"floor": 100}))
    v: list[dict] = []
    mi._run_fidelity_tier(str(tmp_path), b"reference_long_expected_stream", b"abc", v)
    assert len(v) == 1 and v[0]["criterion"] == "fidelity_below_budget"
    # armC live (twice driver-flagged): a below-budget fidelity score is a
    # CONTENT failure -- block_math (targeted re-spec), never 'contract'
    # (which broadcast-re-fanned / ENDed the run on a quant-table bug).
    assert v[0]["gap_class"] == "block_math"

    # within budget: floor 1 -> len 3 >= 1 -> PASS (no violation)
    monkeypatch.setenv("CORESMITH_FIDELITY_BUDGET", json.dumps({"floor": 1}))
    v2: list[dict] = []
    mi._run_fidelity_tier(str(tmp_path), b"reference_long_expected_stream", b"abc", v2)
    assert v2 == []
    assert (tmp_path / ".coresmith" / "derate_ledger.json").exists()


# --- step 3: derate escalation / sign-off -------------------------------------

def test_read_escalation_then_signed_off(tmp_path, monkeypatch):
    monkeypatch.setenv("CORESMITH_FIDELITY_METRIC", _metric_file(tmp_path))
    monkeypatch.setenv(
        "CORESMITH_FIDELITY_BUDGET",
        json.dumps({"floor": 40, "escalate_floor": 42, "ideal": 44}),
    )
    fid = fz.compute_fidelity_derate(str(tmp_path), b"ref", 41.0)  # within budget, escalate
    fz.write_derate_ledger(str(tmp_path), fid, byte_exact=False)
    esc = fz.read_derate_escalation(str(tmp_path))
    assert esc is not None and esc["block"] == "_integrated" and esc["escalate"] is True
    fz.mark_derate_signed_off(str(tmp_path))
    assert fz.read_derate_escalation(str(tmp_path)) is None  # not re-prompted


def test_no_escalation_when_clean(tmp_path, monkeypatch):
    monkeypatch.setenv("CORESMITH_FIDELITY_METRIC", _metric_file(tmp_path))
    monkeypatch.setenv(
        "CORESMITH_FIDELITY_BUDGET", json.dumps({"floor": 40, "escalate_floor": 42})
    )
    fid = fz.compute_fidelity_derate(str(tmp_path), b"ref", 43.0)  # clean pass
    fz.write_derate_ledger(str(tmp_path), fid, byte_exact=False)
    assert fz.read_derate_escalation(str(tmp_path)) is None


def test_no_escalation_without_ledger(tmp_path):
    assert fz.read_derate_escalation(str(tmp_path)) is None


def test_byte_exact_never_escalates(tmp_path):
    fz.write_derate_ledger(str(tmp_path), None, byte_exact=True)
    assert fz.read_derate_escalation(str(tmp_path)) is None
