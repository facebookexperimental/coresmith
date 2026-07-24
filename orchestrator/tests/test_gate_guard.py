# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the fail-closed gate guard (A-Fix 2)."""

from __future__ import annotations

import pytest

from orchestrator.langgraph.gate_guard import (
    GateResult,
    gate_error_violation,
    gate_fail_open_enabled,
    gate_guard,
)

pytestmark = pytest.mark.failclosed


class TestGateGuard:
    def test_success_passes_and_preserves_value(self):
        r = gate_guard("g", lambda: ["a", "b"])
        assert isinstance(r, GateResult)
        assert r.passed is True
        assert r.skipped is False
        assert r.errored is False
        assert r.value == ["a", "b"]

    def test_success_forwards_args_and_kwargs(self):
        r = gate_guard("g", lambda a, b=0: a + b, 2, b=3)
        assert r.passed is True
        assert r.value == 5

    def test_exception_fails_closed(self):
        def _boom():
            raise RuntimeError("environment broke")

        r = gate_guard("model_integration", _boom)
        assert r.passed is False
        assert r.skipped is False  # an ERROR is not an honest skip
        assert r.errored is True
        assert "RuntimeError" in r.reason
        assert "environment broke" in r.error  # traceback tail

    def test_classify_maps_return_to_passed(self):
        # An empty violations list means "pass"; a non-empty one means "fail".
        ok = gate_guard("g", list, classify=lambda v: len(v) == 0)
        bad = gate_guard("g", lambda: ["x"], classify=lambda v: len(v) == 0)
        assert ok.passed is True
        assert bad.passed is False
        assert bad.errored is False  # classify=False is not an error

    def test_fail_open_escape_hatch_tolerates_exception(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_GATE_FAIL_OPEN", "1")

        def _boom():
            raise ValueError("boom")

        r = gate_guard("g", _boom)
        assert r.passed is True
        assert r.skipped is True
        assert r.errored is True  # the error is still RECORDED, just tolerated

    def test_fail_open_default_off(self, monkeypatch):
        monkeypatch.delenv("CORESMITH_GATE_FAIL_OPEN", raising=False)
        assert gate_fail_open_enabled() is False

    def test_fail_open_env_on(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_GATE_FAIL_OPEN", "1")
        assert gate_fail_open_enabled() is True


class TestGateErrorViolation:
    def test_shape(self):
        v = gate_error_violation("BoomError: kaboom", "traceback tail here")
        assert v["criterion"] == "gate_error"
        assert v["gap_class"] == "block_math"
        assert "kaboom" in v["observed"]
        assert "NOT a pass" in v["suggested_fix"]
        assert "traceback tail here" in v["suggested_fix"]

    def test_custom_gap_class(self):
        v = gate_error_violation("x", gap_class="contract")
        assert v["gap_class"] == "contract"
