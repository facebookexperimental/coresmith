"""WP-25: MAX-GEOMETRY requires executed owner cases even for deterministic testbenches."""
from __future__ import annotations

import json

from orchestrator.langgraph import pipeline_graph as pg


def _project(tmp_path):
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs/task.yaml").write_text(json.dumps({"max_geometry_cases": {
        "owner_max": {"fft_points": 256, "qspi_byte_address": 16777215},
    }}))
    cs = tmp_path / ".coresmith"
    cs.mkdir(parents=True, exist_ok=True)
    (cs / "ers_spec.json").write_text(json.dumps({"ers": {"parameters": [
        {"name": "fft_points", "role": "dimension", "max": 256},
        {"name": "qspi_byte_address", "role": "dimension", "max": 16777215}]}}))
    tb = tmp_path / "tb.py"
    tb.write_text("# MAXGEO_CASE: name=rand cfg0=0 in_bytes=1024 out_bytes=1024\n")
    return tmp_path, tb


def test_deterministic_bfm_scope_is_unexecuted(tmp_path, monkeypatch):
    monkeypatch.delenv("CORESMITH_MAXGEO_GATE", raising=False)
    root, tb = _project(tmp_path)
    v = pg._maxgeo_gate_verdict(str(root), str(tb), {"deterministic_bfm": True, "contract": {"bus": "qspi"}})
    assert v is not None and v["verdict"] == "unknown"
    assert set(v["uncovered_dims"]) == {"fft_points", "qspi_byte_address"}


def test_llm_testbench_still_fails_hard(tmp_path, monkeypatch):
    monkeypatch.delenv("CORESMITH_MAXGEO_GATE", raising=False)
    root, tb = _project(tmp_path)
    for record in (None, {"deterministic_bfm": False}, {"deterministic_bfm": True}):
        v = pg._maxgeo_gate_verdict(str(root), str(tb), record)
        assert v is not None and not v.get("advisory") and "reason" in v, record
