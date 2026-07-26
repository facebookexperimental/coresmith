# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the nightly-canary decision functions (Package C, C6).

The canary drives the daemon headless; its two safety-critical decisions must be
correct regardless of a live LLM/EDA box:

- A FAILED integration_dv / validation_dv interrupt must STOP the run (abort/skip)
  and never auto-``approve`` -- approving a failed DV would ship an unverified
  chip. (Those interrupts only fire on failure; their supported_actions never
  include ``approve``.)
- A daemon HTTP 409 (transient poll-loop race) must be retried with exponential
  backoff, then error out -- not silently treated as success.

These are pure functions with injectable runner/sleeper, so they need no daemon.
"""
from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def _canary():
    path = _REPO / "scripts" / "nightly_canary.py"
    spec = importlib.util.spec_from_file_location("nightly_canary", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The real payloads the pipeline raises (from pipeline_graph.integration_dv_node
# / validation_dv_node): note NO "approve" in supported_actions and NO "failed"
# key -- reaching the interrupt at all means DV failed.
_INTEGRATION_DV_FAIL = {
    "type": "integration_dv_failure",
    "design_name": "adder8",
    "supported_actions": ["retry", "fix_rtl", "fix_tb", "abort"],
}
_VALIDATION_DV_FAIL = {
    "type": "validation_dv_failure",
    "design_name": "adder8",
    "supported_actions": ["retry", "fix_rtl", "fix_tb", "abort"],
}
_SPEC_REVIEW = {
    "type": "uarch_spec_review",
    "block_name": "adder",
    "supported_actions": ["approve", "revise"],
}
_ASK_HUMAN = {
    "type": "block_failure_ask_human",
    "block_name": "adder",
    "supported_actions": ["retry", "skip", "fix_rtl", "fix_tb", "abort"],
}


def _cp(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(
        args=["resume"], returncode=returncode, stdout=stdout, stderr=stderr,
    )


class TestDvFailureNeverApproved:
    def test_integration_dv_failure_detected(self):
        c = _canary()
        assert c._is_dv_failure(_INTEGRATION_DV_FAIL) is True
        assert c._is_dv_failure(_VALIDATION_DV_FAIL) is True

    def test_spec_review_not_a_dv_failure(self):
        c = _canary()
        assert c._is_dv_failure(_SPEC_REVIEW) is False
        assert c._is_dv_failure(_ASK_HUMAN) is False

    def test_failed_dv_picks_abort_never_approve(self):
        c = _canary()
        for payload in (_INTEGRATION_DV_FAIL, _VALIDATION_DV_FAIL):
            action = c._pick_action(payload, retries_seen=0, max_retries=3)
            assert action["action"] != "approve"
            assert action["action"] in payload["supported_actions"]
            assert action["action"] == "abort"

    def test_failed_dv_falls_back_to_skip_when_no_abort(self):
        c = _canary()
        payload = {"type": "integration_dv_failure",
                   "supported_actions": ["retry", "skip", "fix_rtl"]}
        action = c._pick_action(payload, 0, 3)
        assert action["action"] == "skip"

    def test_spec_review_still_approves(self):
        c = _canary()
        assert c._pick_action(_SPEC_REVIEW, 0, 3)["action"] == "approve"

    def test_ask_human_retries_then_skips(self):
        c = _canary()
        assert c._pick_action(_ASK_HUMAN, 0, 3)["action"] == "retry"
        assert c._pick_action(_ASK_HUMAN, 5, 3)["action"] == "skip"


class TestConflictRetryBackoff:
    def test_409_retries_with_backoff_then_errors(self):
        c = _canary()
        calls = {"n": 0}
        slept: list[float] = []

        def runner():
            calls["n"] += 1
            return _cp(returncode=1, stderr="error: POST http://x/run/resume -> 409\nno pending interrupt")

        ok = c._resume("/pr", {"action": "abort"}, max_attempts=4,
                       base_backoff_s=1.0, sleeper=slept.append, runner=runner)
        assert ok is False              # persistent 409 -> unrecoverable error
        assert calls["n"] == 4          # tried max_attempts times
        # Exponential backoff between the (max_attempts - 1) retries.
        assert slept == [1.0, 2.0, 4.0]

    def test_409_then_success_returns_ok(self):
        c = _canary()
        seq = [
            _cp(returncode=1, stderr="-> 409 pipeline still running; nothing to resume"),
            _cp(returncode=0, stdout='{"resumed": true}'),
        ]
        slept: list[float] = []

        def runner():
            return seq.pop(0)

        ok = c._resume("/pr", {"action": "approve"}, max_attempts=4,
                       sleeper=slept.append, runner=runner)
        assert ok is True
        assert slept == [1.0]           # one backoff before the successful retry

    def test_non_conflict_error_fails_immediately(self):
        c = _canary()
        calls = {"n": 0}

        def runner():
            calls["n"] += 1
            return _cp(returncode=1, stderr="-> 400 action 'approve' not supported; allowed: ['abort']")

        ok = c._resume("/pr", {"action": "approve"}, max_attempts=4,
                       sleeper=lambda _s: None, runner=runner)
        assert ok is False
        assert calls["n"] == 1          # a 400 is not retried

    def test_success_returns_true_no_retry(self):
        c = _canary()
        ok = c._resume("/pr", {"action": "approve"},
                       runner=lambda: _cp(returncode=0, stdout="{}"))
        assert ok is True


# Contract-audit classifications the pipeline embeds in a DV-failure payload.
_AUDIT_PROCESS = {"category": "DV_PROCESS_ERROR", "confidence": 0.96,
                  "local_fix_possible": True, "recommended_action": "fix_tb"}
_AUDIT_PROCESS_LOWCONF = {"category": "DV_PROCESS_ERROR", "confidence": 0.5}
# rung2 defect 5: TESTBENCH_BUG is now a retry-once class (the engine regenerates
# the TB and often passes). Live case: conf 0.86, fix_tb, local_fix_possible.
_AUDIT_TB_BUG = {"category": "TESTBENCH_BUG", "confidence": 0.86,
                 "local_fix_possible": True, "recommended_action": "fix_tb"}
_AUDIT_TB_NO_LOCALFIX = {"category": "TESTBENCH_BUG", "confidence": 0.95,
                         "local_fix_possible": False}
_AUDIT_TB_LOWCONF = {"category": "TESTBENCH_BUG", "confidence": 0.84,
                     "local_fix_possible": True}
# A genuine functional mismatch is NEVER retried (stops immediately).
_AUDIT_FUNCTIONAL = {"category": "LOCAL_RTL_BUG", "confidence": 0.95,
                     "local_fix_possible": True}


def _dv_fail(audit):
    return {"type": "integration_dv_failure", "design_name": "adder8",
            "supported_actions": ["retry", "fix_rtl", "fix_tb", "abort"],
            "contract_audit": audit}


class TestProcessErrorClassification:
    def test_high_conf_process_error_is_retryable(self):
        c = _canary()
        assert c._is_retryable_dv_failure(_AUDIT_PROCESS) is True

    def test_testbench_bug_is_retryable(self):
        # rung2 defect 5: TESTBENCH_BUG conf 0.86 + local_fix_possible -> retry.
        c = _canary()
        assert c._is_retryable_dv_failure(_AUDIT_TB_BUG) is True

    def test_testbench_bug_without_local_fix_not_retryable(self):
        c = _canary()
        assert c._is_retryable_dv_failure(_AUDIT_TB_NO_LOCALFIX) is False

    def test_testbench_bug_below_confidence_not_retryable(self):
        # 0.84 < 0.85 threshold.
        c = _canary()
        assert c._is_retryable_dv_failure(_AUDIT_TB_LOWCONF) is False

    def test_confidence_boundary_085_is_retryable(self):
        c = _canary()
        audit = {"category": "TESTBENCH_BUG", "confidence": 0.85,
                 "local_fix_possible": True}
        assert c._is_retryable_dv_failure(audit) is True

    def test_low_conf_process_error_not_retryable(self):
        c = _canary()
        assert c._is_retryable_dv_failure(_AUDIT_PROCESS_LOWCONF) is False

    def test_functional_never_retryable(self):
        c = _canary()
        assert c._is_retryable_dv_failure(_AUDIT_FUNCTIONAL) is False
        assert c._is_retryable_dv_failure({}) is False

    def test_backcompat_alias_points_at_new_predicate(self):
        c = _canary()
        assert c._is_process_error is c._is_retryable_dv_failure

    def test_load_audit_prefers_embedded(self):
        c = _canary()
        assert c._load_contract_audit(_dv_fail(_AUDIT_PROCESS), "/pr") == _AUDIT_PROCESS

    def test_load_audit_disk_fallback(self, tmp_path):
        import json
        c = _canary()
        ad = tmp_path / ".coresmith" / "contract_audit"
        ad.mkdir(parents=True)
        (ad / "integration_dv_contract_audit.json").write_text(json.dumps(_AUDIT_PROCESS))
        payload = {"type": "integration_dv_failure",
                   "supported_actions": ["retry", "abort"]}
        assert c._load_contract_audit(payload, str(tmp_path)) == _AUDIT_PROCESS


class TestCanaryProcessErrorRetry:
    def _drive(self, monkeypatch, tmp_path, states):
        c = _canary()
        it = iter(states)

        def fake_state(_pr):
            try:
                return next(it)
            except StopIteration:
                return {"status": "done", "pipeline_done": False}

        resumes: list[str] = []

        def fake_resume(_pr, action, **_kw):
            resumes.append(action["action"])
            return True

        monkeypatch.setattr(c, "_cli", lambda *a, **k: _cp(0, "{}"))
        monkeypatch.setattr(c, "_state", fake_state)
        monkeypatch.setattr(c, "_resume", fake_resume)
        monkeypatch.setattr(c.time, "sleep", lambda *_a, **_k: None)
        monkeypatch.setattr(c.subprocess, "run", lambda *a, **k: _cp(0, ""))
        rc = c.run_canary(project_root=str(tmp_path), blocks_file="/b.yaml",
                          timeout_s=100.0, poll_s=0.0, max_retries=3)
        return rc, resumes

    @staticmethod
    def _as_interrupt(payload):
        return {"status": "interrupted", "interrupts": [{"payload": payload}]}

    def test_process_error_retries_once_then_stops(self, monkeypatch, tmp_path):
        # DV process/infra flake twice: retry ONCE, then budget-exhausted -> stop+FAIL.
        p = self._as_interrupt(_dv_fail(_AUDIT_PROCESS))
        rc, resumes = self._drive(monkeypatch, tmp_path, [p, p])
        assert resumes == ["retry", "abort"]     # exactly one retry, then abort
        assert rc == 1                            # DV failed -> FAIL verdict

    def test_testbench_bug_retries_once_then_stops(self, monkeypatch, tmp_path):
        # rung2 defect 5 live case: high-conf TESTBENCH_BUG -> retry once.
        p = self._as_interrupt(_dv_fail(_AUDIT_TB_BUG))
        rc, resumes = self._drive(monkeypatch, tmp_path, [p, p])
        assert resumes == ["retry", "abort"]
        assert rc == 1

    def test_shared_budget_across_classes(self, monkeypatch, tmp_path):
        # A process flake THEN a testbench bug share the SINGLE per-run budget:
        # exactly one retry total, then stop.
        p1 = self._as_interrupt(_dv_fail(_AUDIT_PROCESS))
        p2 = self._as_interrupt(_dv_fail(_AUDIT_TB_BUG))
        rc, resumes = self._drive(monkeypatch, tmp_path, [p1, p2])
        assert resumes.count("retry") == 1
        assert resumes[-1] == "abort"
        assert rc == 1

    def test_functional_failure_stops_immediately(self, monkeypatch, tmp_path):
        p = self._as_interrupt(_dv_fail(_AUDIT_FUNCTIONAL))
        rc, resumes = self._drive(monkeypatch, tmp_path, [p])
        assert resumes == ["abort"]               # no retry for a functional bug
        assert "retry" not in resumes
        assert rc == 1

    def test_testbench_bug_without_local_fix_stops_immediately(self, monkeypatch, tmp_path):
        p = self._as_interrupt(_dv_fail(_AUDIT_TB_NO_LOCALFIX))
        rc, resumes = self._drive(monkeypatch, tmp_path, [p])
        assert resumes == ["abort"]
        assert "retry" not in resumes
        assert rc == 1

    def test_retry_budget_not_exceeded(self, monkeypatch, tmp_path):
        # Three consecutive process-error failures: still only ONE retry.
        p = self._as_interrupt(_dv_fail(_AUDIT_PROCESS))
        rc, resumes = self._drive(monkeypatch, tmp_path, [p, p, p])
        assert resumes.count("retry") == 1
        assert resumes[-1] == "abort"
        assert rc == 1

    def test_low_confidence_process_error_stops_immediately(self, monkeypatch, tmp_path):
        p = self._as_interrupt(_dv_fail(_AUDIT_PROCESS_LOWCONF))
        rc, resumes = self._drive(monkeypatch, tmp_path, [p])
        assert resumes == ["abort"]               # low confidence -> no retry
        assert rc == 1
