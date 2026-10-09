# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The native-session loop (the cluster workers' base class): bounded by its
invocation count, stops to wait for answers to must-answer questions, resumes
from that state, keeps the runner's stderr for outcome classification, books
cumulative session cost as per-invocation deltas -- and has no no-progress
counter: an unchanged blocker list is never a reason to stop."""
import json

import pytest

from orchestrator.architect import ArchitectSession
from orchestrator.architect.cluster import ClusterSession
from orchestrator.state_store.project_db import open_project

_BLOCKED = {"stage": "interfaces", "blocked_by": [{"code": "SHELL_NO_BOUNDARY", "text": "t", "ids": [], "count": 0}]}


class _Script:
    """A stubbed session: ``stages`` is what ``stage_status`` returns before the
    first invocation and after each one; ``costs`` the cumulative cost each
    invocation reports; ``do`` runs "the model" inside invocation i."""

    def __init__(self, stages, costs=None, do=None, sid="sess-1", ok=True, extra=None):
        self.stages = list(stages)
        self.costs = list(costs or [])
        self.do = do or {}
        self.sid = sid
        self.ok = ok
        self.extra = extra or {}
        self.sittings = 0
        self.prompts = []

    def install(self, sess, monkeypatch):
        idx = {"i": 0}

        def stage_status():
            i = min(idx["i"], len(self.stages) - 1)
            return json.loads(json.dumps(self.stages[i]))

        def sit(prompt, *, resume="", index=1, **k):
            self.sittings += 1
            self.prompts.append(prompt)
            if self.sittings in self.do:
                self.do[self.sittings]()
            idx["i"] += 1
            cost = self.costs[self.sittings - 1] if self.sittings - 1 < len(self.costs) else None
            return {"ok": self.ok, "rc": 0 if self.ok else -9, "session_id": self.sid, "text": "done",
                    "cost_usd": cost, "turns": 2, "elapsed_s": 1.0, "transcript": "", **self.extra}
        monkeypatch.setattr(sess, "stage_status", stage_status)
        monkeypatch.setattr(sess, "sit", sit)
        monkeypatch.setattr(sess, "run_status", lambda: "(status)")
        return self


@pytest.fixture
def sess(tmp_path, monkeypatch):
    for k in ("CORESMITH_ARCHITECT_NO_PROGRESS_SITTINGS", "CORESMITH_ROLE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))
    open_project(tmp_path)
    return ArchitectSession(tmp_path, claude_path="false", max_sittings=12)


def test_unchanged_blockers_never_stop_the_session(sess, monkeypatch):
    """No no-progress counter: the loop runs to its invocation bound (the
    legacy env var changes nothing)."""
    monkeypatch.setenv("CORESMITH_ARCHITECT_NO_PROGRESS_SITTINGS", "2")
    sess.max_sittings = 5
    sc = _Script([_BLOCKED] * 20).install(sess, monkeypatch)
    st = sess.run()
    assert st["state"] == "blocked" and st["stop_reason"] == "max_sittings" and sc.sittings == 5
    assert "no_progress" not in json.dumps(st)


def test_done_stage_ends_the_session(sess, monkeypatch):
    sc = _Script([_BLOCKED, {"stage": "blocks", "blocked_by": []}]).install(sess, monkeypatch)
    st = sess.run()
    assert st["state"] == "done" and sc.sittings == 1 and st["sittings"] == 1


def test_open_must_answer_question_stops_waiting_for_answers_then_resumes(sess, monkeypatch, tmp_path):
    db = open_project(tmp_path)

    def ask():
        db.add_question("please declare the chip top", must_answer=True, asked_by="worker:cpu")
    sc = _Script([_BLOCKED] * 20, do={1: ask}).install(sess, monkeypatch)
    st = sess.run()
    assert st["state"] == "waiting_for_answers" and sc.sittings == 1
    q = st["waiting_questions"][0]
    assert q["id"] == 1 and q["answer"] == 'coresmith question answer Q1 --ruling "<your ruling>"'
    db.answer_question(1, "chip_top: clk, rst_n, uart_rx, uart_tx")
    done = {"stage": "blocks", "blocked_by": []}
    sc2 = _Script([_BLOCKED, done]).install(sess, monkeypatch)
    st = sess.run()
    assert st["state"] == "done" and sc2.sittings == 1 and st["sittings"] == 2
    assert sess.status()["resumed_from"] == "waiting_for_answers"


def test_only_open_questions_blocking_waits(sess, monkeypatch):
    qs = {"stage": "requirements", "blocked_by": [{"code": "OPEN_QUESTIONS", "text": "t", "ids": ["Q3"], "count": 1}]}
    monkeypatch.setattr(sess, "open_questions", lambda: [])
    sc = _Script([_BLOCKED, qs]).install(sess, monkeypatch)
    st = sess.run()
    assert st["state"] == "waiting_for_answers" and sc.sittings == 1


def test_runner_failure_is_classified_from_its_diagnostics(sess, monkeypatch, tmp_path):
    """A killed invocation is reported from the runner's stderr tail, not from
    the model's last message, and recorded as an agent failure."""
    sc = _Script([_BLOCKED] * 3, ok=False, extra={"stderr_tail": "[invocation timed out after 60s]",
                                                   "text": "Let me diagnose by running refine again:"}).install(sess, monkeypatch)
    st = sess.run()
    assert st["state"] == "tool_failed" and sc.sittings == 1
    assert st["stop_reason"] == "[invocation timed out after 60s]" and "diagnose" not in st["stop_reason"]
    assert open_project(tmp_path).get_flag("agent_failure")["kind"] == "tool_failed"


def test_sit_keeps_the_stderr_tail_on_the_result(tmp_path, monkeypatch):
    monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))
    s = ClusterSession(tmp_path, "cpu", ["mcu"], claude_path="false")

    class _Runner:
        name, model = "claude", "m"

        def run(self, prompt, **k):
            return {"ok": False, "rc": 1, "session_id": "", "text": "model text", "cost_usd": None,
                    "cost_cumulative": True, "turns": 0, "stderr": "boom: provider refused\n"}
    monkeypatch.setattr(ClusterSession, "runner", property(lambda self: _Runner()))
    res = s.sit("p", index=1)
    assert res["stderr_tail"] == "boom: provider refused\n" and "stderr" not in res
    assert (tmp_path / ".coresmith" / "clusters" / "cpu" / "stderr-1.log").read_text() == "boom: provider refused\n"


def test_sitting_env_sets_no_legacy_architect_flag(sess):
    env = sess.sitting_env({"CORESMITH_ACTOR": "worker:x"})
    assert "CORESMITH_ARCHITECT_SITTING" not in env and env["CORESMITH_ACTOR"] == "worker:x"
    assert env["CORESMITH_PROJECT_ROOT"] == str(sess.root)


def test_cost_is_the_delta_of_the_cumulative_session_cost(sess, monkeypatch):
    sess.max_sittings = 3
    _Script([_BLOCKED] * 10, costs=[8.07, 8.61, 12.42]).install(sess, monkeypatch)
    st = sess.run()
    assert st["cost_usd"] == pytest.approx(12.42) and st["session_cost_usd"] == pytest.approx(12.42)
    assert st["last"]["cost_usd"] == pytest.approx(12.42 - 8.61) and st["last"]["session_cost_usd"] == 12.42
    # a later run continues from the cumulative figure
    sess.max_sittings = 1
    _Script([_BLOCKED] * 10, costs=[13.0]).install(sess, monkeypatch)
    assert sess.run()["cost_usd"] == pytest.approx(13.0)


def test_legacy_summed_status_is_corrected(sess, monkeypatch):
    """A status.json written by the summing runner."""
    sess.status_path.write_text(json.dumps({"state": "blocked", "session_id": "sess-1", "sittings": 24,
                                            "cost_usd": 250.98, "last": {"cost_usd": 12.42}}))
    sess.max_sittings = 1
    _Script([_BLOCKED] * 10, costs=[12.9]).install(sess, monkeypatch)
    assert sess.run()["cost_usd"] == pytest.approx(12.9)


def test_no_architect_verb_and_no_engine_owned_architect_module():
    """The engine neither exposes `coresmith architect ...` nor keeps the
    on-call module: the Architect is the coding agent outside the engine."""
    import argparse
    import importlib

    from orchestrator.harness import cli
    ap = argparse.ArgumentParser(prog="coresmith")
    sub = ap.add_subparsers(dest="cmd")
    cli.register_subcommands(sub)
    assert "architect" not in sub.choices and "model" in sub.choices and "harness" in sub.choices
    assert not hasattr(cli, "cmd_architect")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("orchestrator.architect.oncall")
    from orchestrator.architect import session as s
    for name in ("on_call", "oncall_prompt", "build_system_prompt", "no_progress_limit", "blocker_key"):
        assert not hasattr(s, name) and not hasattr(ArchitectSession, name), name
