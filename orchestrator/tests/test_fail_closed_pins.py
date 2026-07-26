# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Fail-closed pins (Package C, C4).

Each test encodes a fail-closed behavior that A-Fix 2 (Package A, plan commit 5)
delivers. They were xfail(strict=False) until the fix landed; commit 5 REMOVED
the xfail markers, turning them into permanent regression guards. If a future
change regresses any of these back to fail-open, these tests fail hard.

Keep each pin narrow and API-level.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.failclosed


def test_gate_guard_module_fails_closed_on_exception():
    """A-Fix 2: gate_guard(name, fn) turns a raised gate into passed=False."""
    from orchestrator.langgraph.gate_guard import gate_guard

    def _boom():
        raise RuntimeError("gate environment broke")

    result = gate_guard("model_integration", _boom)
    assert result.passed is False
    assert result.skipped is False  # an ERROR is not an honest skip
    assert result.error  # traceback tail recorded


def test_gate_guard_fail_open_escape_hatch():
    """CORESMITH_GATE_FAIL_OPEN=1 is the single global rollback knob."""
    import os

    from orchestrator.langgraph.gate_guard import gate_guard

    def _boom():
        raise RuntimeError("boom")

    old = os.environ.get("CORESMITH_GATE_FAIL_OPEN")
    os.environ["CORESMITH_GATE_FAIL_OPEN"] = "1"
    try:
        result = gate_guard("g", _boom)
        # Under the escape hatch a gate error does NOT fail the block.
        assert result.passed is True or result.skipped is True
    finally:
        if old is None:
            os.environ.pop("CORESMITH_GATE_FAIL_OPEN", None)
        else:
            os.environ["CORESMITH_GATE_FAIL_OPEN"] = old


def test_ppa_verdict_records_unmeasured():
    """A-Fix 2e: PpaVerdict grows an ``unmeasured`` list for missing budget/measurement."""
    from orchestrator.langgraph.ppa_check import PpaVerdict

    v = PpaVerdict(ok=True)
    assert hasattr(v, "unmeasured")
    assert isinstance(v.unmeasured, list)


def test_equiv_skip_distinguishes_harness_error():
    """A-Fix 2c: _skip can mark a harness error (retry-then-fail) vs an honest skip."""
    from orchestrator.langgraph.rtl_model_equiv import _skip

    r = _skip("verilator missing", harness_error=True)
    assert r["skipped"] is True
    assert r.get("harness_error") is True


def test_backend_timing_met_defaults_none_when_unmeasured(tmp_path):
    """A-Fix 2g: parse_openroad_reports leaves timing_met None until a WNS is read."""
    from orchestrator.langgraph.backend_helpers import parse_openroad_reports

    metrics = parse_openroad_reports(str(tmp_path))  # no timing_wns.rpt present
    assert metrics["timing_met"] is None
