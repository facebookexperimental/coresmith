# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the v3 measured-throughput gate.

Covers: the pure gate math (21 vs 11 -> FAIL, 11.5 vs 11 -> PASS at 1.1), the
artifact parser, the fail-closed missing-artifact path, the not-applicable
paths (no declared cyc/op, gate disabled), the squeeze-loop decision + round
bounds, and the chip-level measured record.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchestrator.langgraph import throughput_gate as tg


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _write_perf_model(root: Path, block: str, **fields) -> None:
    bd = root / ".coresmith" / "blocks" / block
    bd.mkdir(parents=True, exist_ok=True)
    model = {"block": block, "op_unit": "block"}
    model.update(fields)
    (bd / "perf_model.json").write_text(json.dumps(model))


def _write_artifact(sim_dir: Path, measured, n_ops=8, key="measured_cyc_per_op"):
    sim_dir.mkdir(parents=True, exist_ok=True)
    (sim_dir / tg.BLOCK_ARTIFACT).write_text(
        json.dumps({key: measured, "n_ops": n_ops}))


@pytest.fixture(autouse=True)
def _gate_on(monkeypatch):
    # Default is ON, but be explicit so a hostile ambient env can't flip it.
    monkeypatch.setenv("CORESMITH_MEASURED_THROUGHPUT_GATE", "1")
    monkeypatch.setenv("CORESMITH_THROUGHPUT_SQUEEZE", "1")


# --------------------------------------------------------------------------- #
# pure gate math
# --------------------------------------------------------------------------- #
def test_gate_math_fail_21_vs_11():
    gm = tg.gate_math(21, 11)
    assert gm["passed"] is False
    assert gm["threshold_cyc_per_op"] == pytest.approx(12.1)
    assert gm["ratio"] == pytest.approx(21 / 11, rel=1e-3)


def test_gate_math_pass_at_1p1_boundary():
    # 11.5 <= 11 * 1.1 = 12.1 -> PASS
    assert tg.gate_math(11.5, 11)["passed"] is True
    # exactly on the ceiling passes
    assert tg.gate_math(12.1, 11)["passed"] is True
    # a hair over fails
    assert tg.gate_math(12.2, 11)["passed"] is False


def test_gate_math_nonpositive_declared_passes():
    assert tg.gate_math(99, 0)["passed"] is True


def test_threshold_for():
    assert tg.threshold_for(10) == pytest.approx(11.0)


# --------------------------------------------------------------------------- #
# artifact parsing
# --------------------------------------------------------------------------- #
def test_artifact_parse_primary_key(tmp_path):
    _write_artifact(tmp_path, 21.0, n_ops=16)
    art = tg.read_throughput_artifact(tmp_path)
    assert art == {"measured_cyc_per_op": 21.0, "n_ops": 16}


def test_artifact_parse_alt_key(tmp_path):
    _write_artifact(tmp_path, 7.5, key="cyc_per_op")
    art = tg.read_throughput_artifact(tmp_path)
    assert art["measured_cyc_per_op"] == 7.5


def test_artifact_missing_returns_none(tmp_path):
    assert tg.read_throughput_artifact(tmp_path) is None


def test_artifact_nested_rglob(tmp_path):
    nested = tmp_path / "sub" / "deep"
    _write_artifact(nested, 3.0)
    assert tg.read_throughput_artifact(tmp_path)["measured_cyc_per_op"] == 3.0


# --------------------------------------------------------------------------- #
# block gate: end-to-end verdicts
# --------------------------------------------------------------------------- #
def test_block_gate_fail_measured_over_declared(tmp_path):
    _write_perf_model(tmp_path, "aes_ks", declared_cyc_per_op=11,
                      cyc_per_op_peak=11,
                      binding_constraint={"type": "ResMII", "name": "sbox",
                                          "detail": "16 uses/1 inst",
                                          "lever": "add lanes"})
    sim = tmp_path / "sim"
    _write_artifact(sim, 21.0)
    rec = tg.evaluate_block_throughput(str(tmp_path), "aes_ks", sim)
    assert rec["applicable"] is True
    assert rec["passed"] is False
    assert rec["artifact_missing"] is False
    assert rec["measured_cyc_per_op"] == 21.0
    assert rec["declared_cyc_per_op"] == 11
    assert rec["threshold_cyc_per_op"] == pytest.approx(12.1)
    assert "MEASURED-THROUGHPUT GATE" in rec["report"]
    # deficit report names the RTL-perf class + the two banned patterns
    assert "REQUEST/RESPONSE" in rec["report"]


def test_block_gate_pass_within_ceiling(tmp_path):
    _write_perf_model(tmp_path, "b", declared_cyc_per_op=11)
    sim = tmp_path / "sim"
    _write_artifact(sim, 11.5)
    rec = tg.evaluate_block_throughput(str(tmp_path), "b", sim)
    assert rec["applicable"] is True
    assert rec["passed"] is True
    assert rec["ratio"] == pytest.approx(11.5 / 11, rel=1e-3)
    assert rec["report"] == ""


def test_block_gate_missing_artifact_fails_closed(tmp_path):
    _write_perf_model(tmp_path, "b", declared_cyc_per_op=11)
    sim = tmp_path / "sim"
    sim.mkdir()
    rec = tg.evaluate_block_throughput(str(tmp_path), "b", sim)
    assert rec["applicable"] is True
    assert rec["passed"] is False
    assert rec["artifact_missing"] is True
    assert rec["measured_cyc_per_op"] is None
    assert "test_throughput_measure" in rec["report"]


def test_block_gate_no_declared_is_na(tmp_path):
    # perf_model exists but declares no cyc/op -> not applicable, never demotes
    _write_perf_model(tmp_path, "b", cyc_per_op_peak=5)
    sim = tmp_path / "sim"
    _write_artifact(sim, 99.0)
    rec = tg.evaluate_block_throughput(str(tmp_path), "b", sim)
    assert rec["applicable"] is False
    assert rec["passed"] is None
    assert "no machine-readable" in rec["reason"] or "no §6.1" in rec["reason"]


def test_block_gate_no_perf_model_is_na(tmp_path):
    sim = tmp_path / "sim"
    _write_artifact(sim, 99.0)
    rec = tg.evaluate_block_throughput(str(tmp_path), "missing", sim)
    assert rec["applicable"] is False
    assert rec["passed"] is None


def test_block_gate_disabled_is_na(tmp_path, monkeypatch):
    monkeypatch.setenv("CORESMITH_MEASURED_THROUGHPUT_GATE", "0")
    _write_perf_model(tmp_path, "b", declared_cyc_per_op=11)
    sim = tmp_path / "sim"
    _write_artifact(sim, 99.0)
    rec = tg.evaluate_block_throughput(str(tmp_path), "b", sim)
    assert rec["applicable"] is False
    assert "disabled" in rec["reason"]


def test_gate_enabled_default_when_unset(monkeypatch):
    monkeypatch.delenv("CORESMITH_MEASURED_THROUGHPUT_GATE", raising=False)
    assert tg.measured_throughput_gate_enabled() is True


# --------------------------------------------------------------------------- #
# squeeze loop
# --------------------------------------------------------------------------- #
def test_squeeze_needed_when_above_peak_ceiling(tmp_path):
    _write_perf_model(tmp_path, "b", declared_cyc_per_op=15, cyc_per_op_peak=10)
    # measured 15 > peak 10 * 1.1 = 11 -> squeeze warranted
    s = tg.squeeze_needed(str(tmp_path), "b", 15.0)
    assert s is not None
    assert s["peak_cyc_per_op"] == 10
    assert s["threshold_cyc_per_op"] == pytest.approx(11.0)


def test_squeeze_not_needed_within_peak_ceiling(tmp_path):
    _write_perf_model(tmp_path, "b", declared_cyc_per_op=11, cyc_per_op_peak=10)
    assert tg.squeeze_needed(str(tmp_path), "b", 10.5) is None


def test_squeeze_not_needed_without_measured(tmp_path):
    _write_perf_model(tmp_path, "b", cyc_per_op_peak=10)
    assert tg.squeeze_needed(str(tmp_path), "b", None) is None


def test_squeeze_max_rounds_bounds(monkeypatch):
    monkeypatch.delenv("CORESMITH_SQUEEZE_MAX_ROUNDS", raising=False)
    assert tg.squeeze_max_rounds() == 2
    monkeypatch.setenv("CORESMITH_SQUEEZE_MAX_ROUNDS", "99")
    assert tg.squeeze_max_rounds() == 5   # clamped
    monkeypatch.setenv("CORESMITH_SQUEEZE_MAX_ROUNDS", "-3")
    assert tg.squeeze_max_rounds() == 0   # clamped
    monkeypatch.setenv("CORESMITH_SQUEEZE_MAX_ROUNDS", "garbage")
    assert tg.squeeze_max_rounds() == 2   # default on parse error


def test_squeeze_enabled_default(monkeypatch):
    monkeypatch.delenv("CORESMITH_THROUGHPUT_SQUEEZE", raising=False)
    assert tg.throughput_squeeze_enabled() is True


def test_format_squeeze_request_names_numbers_and_lever():
    need = {"measured_cyc_per_op": 30.0, "peak_cyc_per_op": 10.0,
            "threshold_cyc_per_op": 11.0,
            "binding": "ResMII sbox: 16 uses/1 inst; lever: add lanes"}
    req = tg.format_squeeze_request("aes_ks", need)
    assert "30.0" in req and "10.0" in req and "11.0" in req
    assert "aes_ks" in req
    assert "BYTE-EXACT" in req
    assert "add lanes" in req  # binding lever surfaced
    # objective forbids regressing area/Fmax
    assert "Fmax" in req


# --------------------------------------------------------------------------- #
# chip-level measured throughput
# --------------------------------------------------------------------------- #
def test_chip_measure_records_without_budget(tmp_path):
    sim = tmp_path / "isim"
    sim.mkdir()
    (sim / tg.CHIP_ARTIFACT).write_text(
        json.dumps({"measured_cyc_per_op": 37.0, "n_ops": 1}))
    rec = tg.evaluate_chip_throughput(str(tmp_path), sim)
    assert rec["measured_cyc_per_op_chip"] == 37.0
    assert rec["applicable"] is False  # no budget resolvable -> measure only
    assert rec["budget_source"] == "none"


def test_chip_gate_fails_over_budget(tmp_path):
    sim = tmp_path / "isim"
    sim.mkdir()
    (sim / tg.CHIP_ARTIFACT).write_text(
        json.dumps({"measured_cyc_per_op": 37.0, "n_ops": 1}))
    rec = tg.evaluate_chip_throughput(
        str(tmp_path), sim, state={"chip_cyc_per_op_budget": 21})
    assert rec["applicable"] is True
    assert rec["passed"] is False
    assert rec["budget_cyc_per_op"] == 21
    assert rec["threshold_cyc_per_op"] == pytest.approx(23.1)
    assert rec["budget_source"] == "state"


def test_chip_budget_from_persisted_model(tmp_path):
    (tmp_path / ".coresmith").mkdir(parents=True)
    (tmp_path / ".coresmith" / "chip_perf_model.json").write_text(
        json.dumps({"declared_cyc_per_op": 21, "notes": "aes"}))
    budget, source, _ = tg.chip_throughput_budget(str(tmp_path))
    assert budget == 21
    assert "chip_perf_model" in source


def test_chip_measure_no_artifact_is_na(tmp_path):
    sim = tmp_path / "isim"
    sim.mkdir()
    rec = tg.evaluate_chip_throughput(str(tmp_path), sim)
    assert rec["measured_cyc_per_op_chip"] is None
    assert rec["applicable"] is False


def test_persist_and_read_chip_throughput(tmp_path):
    rec = {"scope": "chip", "measured_cyc_per_op_chip": 30.0}
    tg.persist_chip_throughput(str(tmp_path), rec)
    got = tg.read_chip_throughput(str(tmp_path))
    assert got["measured_cyc_per_op_chip"] == 30.0


# --------------------------------------------------------------------------- #
# engine-v31 step 4: throughput-artifact FRESHNESS (ignore stale artifacts)
# --------------------------------------------------------------------------- #
def test_stale_artifact_ignored_when_older_than_rtl(tmp_path):
    import os
    import time
    art = tmp_path / tg.BLOCK_ARTIFACT
    art.write_text(json.dumps({"measured_cyc_per_op": 11.0, "n_ops": 16}))
    rtl = tmp_path / "blk.v"
    rtl.write_text("module blk; endmodule")
    now = time.time()
    os.utime(art, (now - 100, now - 100))   # artifact is 100s OLDER than RTL
    os.utime(rtl, (now, now))
    # RTL changed after the artifact was written -> stale -> ignored (re-measure)
    assert tg.read_throughput_artifact(tmp_path, rtl) is None
    # back-compat: with no rtl_path, the artifact still reads
    got = tg.read_throughput_artifact(tmp_path)
    assert got is not None and got["measured_cyc_per_op"] == 11.0


def test_fresh_artifact_kept_when_newer_than_rtl(tmp_path):
    import os
    import time
    rtl = tmp_path / "blk.v"
    rtl.write_text("module blk; endmodule")
    art = tmp_path / tg.BLOCK_ARTIFACT
    art.write_text(json.dumps({"measured_cyc_per_op": 12.0, "n_ops": 8}))
    now = time.time()
    os.utime(rtl, (now - 100, now - 100))   # RTL older than the fresh artifact
    os.utime(art, (now, now))
    got = tg.read_throughput_artifact(tmp_path, rtl)
    assert got is not None and got["measured_cyc_per_op"] == 12.0


def test_stale_artifact_forces_measure_in_block_gate(tmp_path, monkeypatch):
    # A stale artifact must make evaluate_block_throughput fail CLOSED
    # (artifact_missing) so the TB re-measures -- not silently pass on the old
    # number.
    import os
    import time
    monkeypatch.setenv("CORESMITH_MEASURED_THROUGHPUT_GATE", "1")
    monkeypatch.setattr(tg, "declared_cyc_per_op", lambda pr, b: 21.0)
    art = tmp_path / tg.BLOCK_ARTIFACT
    art.write_text(json.dumps({"measured_cyc_per_op": 11.0, "n_ops": 16}))
    rtl = tmp_path / "blk.v"
    rtl.write_text("module blk; endmodule")
    now = time.time()
    os.utime(art, (now - 100, now - 100))
    os.utime(rtl, (now, now))
    rec = tg.evaluate_block_throughput(str(tmp_path), "blk", tmp_path, rtl)
    assert rec["applicable"] is True
    assert rec["passed"] is False
    assert rec["artifact_missing"] is True
