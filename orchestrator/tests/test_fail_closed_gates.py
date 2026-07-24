# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""A-Fix 2 fail-closed site behaviors (the deterministic, in-process pieces).

These cover the site conversions that are unit-testable without driving a heavy
graph or EDA toolchain:

- (c) equiv honest-skip vs harness-error split (rtl_model_equiv._skip + the
      timeout_scale retry parameter)
- (e) PPA ``unmeasured`` recording
- (g) backend timing_met None when unmeasured (also pinned in
      test_fail_closed_pins / test_backend_helpers)
- (h) CONDITIONAL_PASS is met only with slack/waiver/opt-in
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.failclosed


# --- (c) equiv skip taxonomy -------------------------------------------------

class TestEquivSkipTaxonomy:
    def test_honest_skip_is_not_a_harness_error(self):
        from orchestrator.langgraph.rtl_model_equiv import _skip

        r = _skip("interface is not single AXI-Stream")
        assert r["skipped"] is True
        assert r["passed"] is False
        assert "harness_error" not in r  # honest skip stays non-blocking

    def test_harness_error_skip_is_flagged(self):
        from orchestrator.langgraph.rtl_model_equiv import _skip

        r = _skip("build could not run", harness_error=True)
        assert r["skipped"] is True
        assert r.get("harness_error") is True

    def test_check_equiv_exposes_timeout_scale(self):
        # The caller retries a harness error once with timeout_scale=2.0.
        from orchestrator.langgraph.rtl_model_equiv import (
            check_rtl_model_equivalence,
        )

        sig = inspect.signature(check_rtl_model_equivalence)
        assert "timeout_scale" in sig.parameters
        assert sig.parameters["timeout_scale"].default == 1.0


# --- (e) PPA unmeasured ------------------------------------------------------

class TestPpaUnmeasured:
    def test_missing_budget_records_ff_unmeasured(self):
        from orchestrator.langgraph.ppa_check import evaluate_ppa

        v = evaluate_ppa(actual_ff=1500, ff_budget=None)
        assert v.ok is True  # ok-semantics unchanged: cannot-judge still passes
        ff = [u for u in v.unmeasured if u["metric"] == "flip_flop_count"]
        assert ff and ff[0]["have_budget"] is False
        assert ff[0]["have_actual"] is True

    def test_missing_measurement_records_ff_unmeasured(self):
        from orchestrator.langgraph.ppa_check import evaluate_ppa

        v = evaluate_ppa(actual_ff=None, ff_budget=1200)
        ff = [u for u in v.unmeasured if u["metric"] == "flip_flop_count"]
        assert ff and ff[0]["have_actual"] is False

    def test_partial_area_records_area_unmeasured(self):
        from orchestrator.langgraph.ppa_check import evaluate_ppa

        # budget supplied, no measured area -> area is unmeasured
        v = evaluate_ppa(actual_ff=100, ff_budget=1200,
                         area_budget_um2=300_000.0, actual_area_um2=None)
        area = [u for u in v.unmeasured if u["metric"] == "chip_area_um2"]
        assert area and area[0]["have_actual"] is False

    def test_no_area_at_all_is_not_recorded(self):
        from orchestrator.langgraph.ppa_check import evaluate_ppa

        # neither budget nor measurement -> area simply not in scope, no noise
        v = evaluate_ppa(actual_ff=100, ff_budget=1200)
        assert not any(u["metric"] == "chip_area_um2" for u in v.unmeasured)


# --- (Section 3a) STA ran-but-no-timing must FAIL CLOSED ---------------------

class TestStaFailClosed:
    def test_sta_error_fails_closed_by_default(self, monkeypatch):
        from orchestrator.langgraph.ppa_check import evaluate_ppa

        monkeypatch.delenv("CORESMITH_PPA_TIMING_ADVISORY", raising=False)
        # STA was attempted (netlist present) but produced no parseable timing.
        v = evaluate_ppa(actual_ff=100, ff_budget=1200, wns_ns=None,
                         sta_error="OpenSTA rc=1: cannot link netlist")
        assert v.ok is False
        wns = [c for c in v.checks if c["metric"] == "wns_ns"]
        assert wns and wns[0]["passed"] is False
        assert wns[0]["unmeasured_timing"] is True
        assert any("no parseable timing" in r for r in v.reasons)

    def test_sta_absent_stays_unmeasured_not_failed(self):
        from orchestrator.langgraph.ppa_check import evaluate_ppa

        # No sta_error -> STA was not attemptable (tool/PDK absent). Legit skip:
        # timing dimension simply not judged, block not failed on that axis.
        v = evaluate_ppa(actual_ff=100, ff_budget=1200, wns_ns=None)
        assert v.ok is True
        assert not any(c["metric"] == "wns_ns" for c in v.checks)

    def test_sta_error_advisory_records_but_passes(self, monkeypatch):
        from orchestrator.langgraph.ppa_check import evaluate_ppa

        monkeypatch.setenv("CORESMITH_PPA_TIMING_ADVISORY", "1")
        v = evaluate_ppa(actual_ff=100, ff_budget=1200, wns_ns=None,
                         sta_error="timeout")
        # advisory mode is the deliberate opt-out: recorded, not blocking
        wns = [c for c in v.checks if c["metric"] == "wns_ns"]
        assert wns and wns[0]["advisory"] is True and wns[0]["passed"] is True
        assert v.ok is True


# --- (g) backend timing_met None ---------------------------------------------

class TestTimingMetUnmeasured:
    def test_empty_dir_timing_met_none(self, tmp_path):
        from orchestrator.langgraph.backend_helpers import parse_openroad_reports

        m = parse_openroad_reports(str(tmp_path))
        assert m["timing_met"] is None

    def test_negative_wns_timing_met_false(self, tmp_path):
        from orchestrator.langgraph.backend_helpers import parse_openroad_reports

        (tmp_path / "timing_wns.rpt").write_text("wns max -2.0\n")
        m = parse_openroad_reports(str(tmp_path))
        assert m["timing_met"] is False


# --- (h) CONDITIONAL_PASS gating ---------------------------------------------

class TestConditionalPassAllowed:
    def test_negative_slack_no_waiver_rejected(self, monkeypatch):
        monkeypatch.delenv("CORESMITH_ALLOW_CONDITIONAL_PASS", raising=False)
        monkeypatch.delenv("CORESMITH_GATE_FAIL_OPEN", raising=False)
        from orchestrator.langgraph.backend_graph import _conditional_pass_allowed

        assert _conditional_pass_allowed(-1.5, waiver_exists=False) is False

    def test_non_negative_slack_allowed(self, monkeypatch):
        monkeypatch.delenv("CORESMITH_ALLOW_CONDITIONAL_PASS", raising=False)
        monkeypatch.delenv("CORESMITH_GATE_FAIL_OPEN", raising=False)
        from orchestrator.langgraph.backend_graph import _conditional_pass_allowed

        assert _conditional_pass_allowed(0.0, waiver_exists=False) is True
        assert _conditional_pass_allowed(3.2, waiver_exists=False) is True

    def test_waiver_allows_negative_slack(self, monkeypatch):
        monkeypatch.delenv("CORESMITH_ALLOW_CONDITIONAL_PASS", raising=False)
        monkeypatch.delenv("CORESMITH_GATE_FAIL_OPEN", raising=False)
        from orchestrator.langgraph.backend_graph import _conditional_pass_allowed

        assert _conditional_pass_allowed(-1.5, waiver_exists=True) is True

    def test_env_opt_in_allows_negative_slack(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_ALLOW_CONDITIONAL_PASS", "1")
        monkeypatch.delenv("CORESMITH_GATE_FAIL_OPEN", raising=False)
        from orchestrator.langgraph.backend_graph import _conditional_pass_allowed

        assert _conditional_pass_allowed(-1.5, waiver_exists=False) is True

    def test_global_fail_open_allows_negative_slack(self, monkeypatch):
        monkeypatch.delenv("CORESMITH_ALLOW_CONDITIONAL_PASS", raising=False)
        monkeypatch.setenv("CORESMITH_GATE_FAIL_OPEN", "1")
        from orchestrator.langgraph.backend_graph import _conditional_pass_allowed

        assert _conditional_pass_allowed(-1.5, waiver_exists=False) is True


# --- A-Fix 2f: PPA gate 3-tuple + tooling-missing park ----------------------

class TestPpaToolingMissingPark:
    def _meta_missing(self):
        return {"tooling_missing": True}

    def test_evaluate_ppa_gate_returns_3tuple_when_off(self, tmp_path, monkeypatch):
        # Gate disabled -> (None, [], {}) 3-tuple (never a 2-tuple anymore).
        monkeypatch.delenv("CORESMITH_PPA_GATE", raising=False)
        from orchestrator.langgraph.pipeline_graph import _evaluate_ppa_gate

        rtl = tmp_path / "b.v"
        rtl.write_text("module b(); endmodule\n")
        res = _evaluate_ppa_gate(str(tmp_path), "b", str(rtl), None)
        assert res == (None, [], {})

    def test_park_when_strict_tooling_missing_unwaived(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_PROFILE", "strict")
        monkeypatch.delenv("CORESMITH_GATE_FAIL_OPEN", raising=False)
        from orchestrator.langgraph.pipeline_graph import (
            _ppa_should_park_tooling_missing,
        )

        assert _ppa_should_park_tooling_missing(
            str(tmp_path), None, self._meta_missing()) is True

    def test_no_park_under_legacy(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_PROFILE", "legacy")
        from orchestrator.langgraph.pipeline_graph import (
            _ppa_should_park_tooling_missing,
        )

        assert _ppa_should_park_tooling_missing(
            str(tmp_path), None, self._meta_missing()) is False

    def test_no_park_when_gate_judged(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_PROFILE", "strict")
        from orchestrator.langgraph.pipeline_graph import (
            _ppa_should_park_tooling_missing,
        )

        # ppa_ok not None -> the gate DID judge -> not unmeasurable.
        assert _ppa_should_park_tooling_missing(
            str(tmp_path), True, self._meta_missing()) is False
        assert _ppa_should_park_tooling_missing(
            str(tmp_path), False, self._meta_missing()) is False

    def test_no_park_when_not_tooling_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_PROFILE", "strict")
        from orchestrator.langgraph.pipeline_graph import (
            _ppa_should_park_tooling_missing,
        )

        assert _ppa_should_park_tooling_missing(str(tmp_path), None, {}) is False

    def test_no_park_under_fail_open(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_PROFILE", "strict")
        monkeypatch.setenv("CORESMITH_GATE_FAIL_OPEN", "1")
        from orchestrator.langgraph.pipeline_graph import (
            _ppa_should_park_tooling_missing,
        )

        assert _ppa_should_park_tooling_missing(
            str(tmp_path), None, self._meta_missing()) is False

    def test_waiver_roundtrip_suppresses_park(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_PROFILE", "strict")
        monkeypatch.delenv("CORESMITH_GATE_FAIL_OPEN", raising=False)
        from orchestrator.langgraph.pipeline_graph import (
            _ppa_should_park_tooling_missing,
            _ppa_tooling_waived,
            _record_ppa_tooling_waiver,
        )

        assert _ppa_tooling_waived(str(tmp_path)) is False
        assert _ppa_should_park_tooling_missing(
            str(tmp_path), None, self._meta_missing()) is True
        _record_ppa_tooling_waiver(str(tmp_path), "blkA")
        assert _ppa_tooling_waived(str(tmp_path)) is True
        # Once waived, the gate no longer parks this run.
        assert _ppa_should_park_tooling_missing(
            str(tmp_path), None, self._meta_missing()) is False
