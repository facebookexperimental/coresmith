# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for routing the structured diagnosis into previous_error.txt.

The regen agent reads previous_error.txt, NOT diagnosis.json -- so historically
the actionable fix the diagnose node produced never reached the fixer, which
re-attacked the same wall blind (the codec RD-core burned ~12h this way). These
cover _compose_actionable_error + _route_diagnosis_to_previous_error and the
CORESMITH_ROUTE_DIAGNOSIS env gate (both branches).
"""
from __future__ import annotations

from orchestrator.langgraph.pipeline_graph import (
    _compose_actionable_error,
    _route_diagnosis_to_previous_error,
)

DIAG = {
    "category": "UNSYNTHESIZABLE",
    "confidence": 0.92,
    "diagnosis": "Wide flat-packed reg top_y_line_q[5119:0] sliced by a runtime "
                 "index -> barrel-shifter cloud, yosys proc cannot elaborate.",
    "suggested_fix": "Replace top_y_line_q with a cs_fpmem addressed memory "
                     "(registered 1-cycle read).",
    "constraints": [
        {"description": "convert line buffer to cs_fpmem",
         "code_snippet": "cs_fpmem_1rw #(.W(8),.D(640)) top_y (...);",
         "file": "rtl/intra_rd_encode_core.v"},
    ],
}


def test_compose_leads_with_fix_then_log_tail():
    msg = _compose_actionable_error(DIAG, raw_log="X" * 9000)
    # the actionable content comes FIRST, raw log is demoted to context
    assert msg.index("SUGGESTED FIX") < msg.index("raw tool log")
    assert "cs_fpmem" in msg and "top_y_line_q" in msg
    assert "code_snippet" not in msg  # rendered, not raw json
    assert "cs_fpmem_1rw #(.W(8),.D(640))" in msg
    assert len(msg) <= 5000


def test_compose_handles_no_constraints_and_no_log():
    msg = _compose_actionable_error(
        {"category": "X", "diagnosis": "d", "suggested_fix": "do the thing"},
        raw_log="",
    )
    assert "do the thing" in msg and "raw tool log" not in msg


def test_route_writes_actionable_when_enabled(tmp_path, monkeypatch):
    monkeypatch.delenv("CORESMITH_ROUTE_DIAGNOSIS", raising=False)
    pe = tmp_path / "previous_error.txt"
    pe.write_text("raw yosys log tail PROC_DLATCH ...")  # the useless old content
    routed = _route_diagnosis_to_previous_error(tmp_path, DIAG, pe.read_text())
    assert routed is True
    body = pe.read_text()
    assert "SUGGESTED FIX" in body and "cs_fpmem" in body


def test_route_disabled_by_env_restores_raw_log_only(tmp_path, monkeypatch):
    monkeypatch.setenv("CORESMITH_ROUTE_DIAGNOSIS", "0")
    pe = tmp_path / "previous_error.txt"
    pe.write_text("raw log")
    routed = _route_diagnosis_to_previous_error(tmp_path, DIAG, pe.read_text())
    assert routed is False
    assert pe.read_text() == "raw log"  # untouched


def test_route_skips_when_no_structured_fix(tmp_path, monkeypatch):
    monkeypatch.delenv("CORESMITH_ROUTE_DIAGNOSIS", raising=False)
    pe = tmp_path / "previous_error.txt"
    pe.write_text("raw log")
    # an infra/fast diagnosis with no suggested_fix + no constraints -> skip
    routed = _route_diagnosis_to_previous_error(
        tmp_path, {"category": "INFRASTRUCTURE_ERROR", "suggested_fix": "",
                   "constraints": []}, pe.read_text())
    assert routed is False
    assert pe.read_text() == "raw log"
