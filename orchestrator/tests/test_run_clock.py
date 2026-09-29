# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The run's target clock: operator env override > run state > task.yaml > 50 MHz."""
from __future__ import annotations

from orchestrator.langgraph.pipeline_helpers import resolve_run_clock_mhz


def test_resolution_order(tmp_path, monkeypatch):
    monkeypatch.delenv("CORESMITH_TARGET_CLOCK_MHZ", raising=False)
    assert resolve_run_clock_mhz(None, tmp_path) == 50.0
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs" / "task.yaml").write_text("top: soc_top\ntarget_clock_mhz: 64   # MHz\n")
    assert resolve_run_clock_mhz(None, tmp_path) == 64.0
    assert resolve_run_clock_mhz(50.0, tmp_path) == 50.0        # the checkpointed run value wins over task.yaml
    monkeypatch.setenv("CORESMITH_TARGET_CLOCK_MHZ", "64")
    assert resolve_run_clock_mhz(50.0, tmp_path) == 64.0        # ... and the operator override over both
    monkeypatch.setenv("CORESMITH_TARGET_CLOCK_MHZ", "bogus")
    assert resolve_run_clock_mhz(None, tmp_path) == 64.0


def test_malformed_task_yaml_falls_back(tmp_path, monkeypatch):
    monkeypatch.delenv("CORESMITH_TARGET_CLOCK_MHZ", raising=False)
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs" / "task.yaml").write_text("target_clock_mhz: [oops\n")
    assert resolve_run_clock_mhz(None, tmp_path) == 50.0
