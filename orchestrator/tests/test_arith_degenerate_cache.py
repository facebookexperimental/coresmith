# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""A degenerate arithmetic characterization (STA absent) must NOT be cached.

When OpenROAD/STA isn't on the box every synth+STA point fails, fit_delay_model
returns an empty model, and the old code cached it as success -- so
is_characterized() returned True while per-op pricing was silently dead. These
pin: degenerate detection, is_characterized rejecting it, ensure_characterized
not writing it, and a clock resolver that reads the run's real target clock.
"""
from __future__ import annotations

import json

import pytest

from orchestrator.langgraph import arith_characterize as ac
from orchestrator.langgraph import latency_audit as la


def test_degenerate_detection():
    assert ac._model_is_degenerate(None) is True
    assert ac._model_is_degenerate({}) is True
    assert ac._model_is_degenerate({"model": {}}) is True
    assert ac._model_is_degenerate({"model": {"add": {"n": 0}}}) is True
    assert ac._model_is_degenerate({"model": {"add": {"a": 0.1, "n": 4}}}) is False


def test_is_characterized_rejects_degenerate(tmp_path, monkeypatch):
    monkeypatch.setattr(ac, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(ac, "_cache_path", lambda pdk=None: tmp_path / "arith_x.json")
    # a degenerate cache file exists -> must NOT count as characterized
    (tmp_path / "arith_x.json").write_text(json.dumps({"model": {}}))
    assert ac.is_characterized() is False
    # a real model -> characterized
    (tmp_path / "arith_x.json").write_text(
        json.dumps({"model": {"add": {"a": 0.1, "b": 0.0, "c": 0.0, "n": 4}}})
    )
    assert ac.is_characterized() is True


def test_ensure_characterized_does_not_cache_degenerate(tmp_path, monkeypatch):
    cache = tmp_path / "arith_x.json"
    monkeypatch.setattr(ac, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(ac, "_cache_path", lambda pdk=None: cache)
    # force the sweep to return all-failed points (STA unavailable)
    monkeypatch.setattr(
        ac, "characterize_ops",
        lambda widths=None: [ac.OpPoint("add", 16, None, None, "STA failed")],
    )
    doc = ac.ensure_characterized()
    assert doc["degenerate"] is True
    assert not cache.exists(), "a degenerate model must NOT be cached as success"
    assert ac.is_characterized() is False


def test_ensure_characterized_caches_a_real_model(tmp_path, monkeypatch):
    cache = tmp_path / "arith_x.json"
    monkeypatch.setattr(ac, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(ac, "_cache_path", lambda pdk=None: cache)
    monkeypatch.setattr(
        ac, "characterize_ops",
        lambda widths=None: [
            ac.OpPoint("add", 8, 1.0, 10.0, ""),
            ac.OpPoint("add", 16, 2.0, 20.0, ""),
            ac.OpPoint("add", 32, 4.0, 40.0, ""),
        ],
    )
    doc = ac.ensure_characterized()
    assert doc["degenerate"] is False
    assert cache.exists()
    assert ac.is_characterized() is True


def test_resolve_target_clock_from_ers(tmp_path):
    arch = tmp_path / "arch"
    arch.mkdir()
    (arch / "ers_spec.md").write_text("- **Target clock:** 100.0 MHz\n")
    assert la.resolve_target_clock_mhz(tmp_path) == 100.0


def test_resolve_target_clock_from_constraints_json(tmp_path):
    cs = tmp_path / ".coresmith"
    cs.mkdir()
    (cs / "constraints.json").write_text(json.dumps({"target_clock_mhz": 200.0}))
    assert la.resolve_target_clock_mhz(tmp_path) == 200.0


def test_resolve_target_clock_default(tmp_path):
    assert la.resolve_target_clock_mhz(tmp_path) == 50.0
    assert la.resolve_target_clock_mhz(tmp_path, default=125.0) == 125.0


def test_stage_map_fragment_resolves_clock(tmp_path):
    bm = tmp_path / "arch" / "block_models"
    bm.mkdir(parents=True)
    (bm / "blk.py").write_text(
        'STAGE_BUDGET=[{"name":"s","latency_cycles":1,"iters":1,"ops":["add16"]}]\n'
        'DECLARED_LATENCY_CYCLES=1\n'
    )
    (tmp_path / "arch" / "ers_spec.md").write_text("Target clock: 100 MHz\n")
    frag = la.stage_map_fragment(tmp_path, "blk")
    # 100 MHz -> 10 ns period must appear in the rendered map header
    assert "10.0 ns" in frag
