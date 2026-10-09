# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The explicit authoring verbs fail truthfully through the REAL adapter path:
a provider process that writes the file and prints a plausible answer but
exits non-zero, or ends its turn with a failure event at exit 0, makes
``model author`` / ``harness author`` exit 3 -- the file left on disk is a
stale/partial artifact, the partial response and the provider's diagnostics
are preserved in ``llm_calls.jsonl`` and the turn log, and nothing is passed
off as a successful authoring call. (No mocked generator: the real
``SystemCModelGenerator`` / ``FRDEvalGenerator`` drive ``ClaudeLLM`` against a
fake ``codex`` binary.)"""
from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from orchestrator.harness.tools import integrate as it
from orchestrator.tests.test_model_commands import _BLOCKS, _inproc, _project


@pytest.fixture
def fake_codex(tmp_path, monkeypatch):
    monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))
    monkeypatch.delenv("CORESMITH_LLM_LOG_ROOT", raising=False)
    monkeypatch.delenv("CORESMITH_TELEMETRY_ROOT", raising=False)
    monkeypatch.setenv("CORESMITH_LLM_PROVIDER", "codex")
    monkeypatch.setenv("CORESMITH_CODEX_ISOLATE_WORKDIR", "0")
    monkeypatch.delenv("CORESMITH_LLM_SLOTS", raising=False)
    monkeypatch.delenv("CORESMITH_CODEX_RESUME", raising=False)
    import orchestrator.systemc_model as scm
    monkeypatch.setattr(scm, "detect", lambda: {"ok": True, "reason": "", "cxx": "g++", "systemc_home": ""})

    def make(body: str) -> Path:
        p = tmp_path / "fake-codex.sh"
        p.write_text("#!/bin/sh\ncat >/dev/null\n" + body)
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
        monkeypatch.setenv("CODEX_CLI_PATH", str(p))
        return p
    return make


def _records(root: Path, name: str) -> list[dict]:
    p = root / ".coresmith" / name
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _ev(obj: dict) -> str:
    return "printf '%s\\n' '" + json.dumps(obj) + "'\n"


def test_model_author_exits_3_when_the_provider_writes_the_file_then_exits_nonzero(tmp_path, fake_codex):
    db = _project(tmp_path)
    cpp = tmp_path / "model" / "req_model.cpp"
    cpp.parent.mkdir(exist_ok=True)
    answer = json.dumps({"files_written": ["model/req_model.cpp"], "notes": "implemented req"})
    fake_codex(f"printf '// partial: the provider died here\\n' > {cpp}\n"
               + _ev({"type": "thread.started", "thread_id": "th-a"})
               + _ev({"type": "item.completed", "item": {"id": "m1", "type": "agent_message", "text": answer}})
               + "echo 'codex: sandbox killed the session' >&2\nexit 2\n")
    res = it.model_author(db, tmp_path, ["req"])
    assert res["ok"] is False and res["written"] == [] and res["not_written"] == ["req"]
    err = res["provider_errors"]["req"]
    assert err.startswith("[ClaudeLLM error: codex CLI exited with code 2") and "sandbox killed" in err
    assert "implemented req" in err                              # the partial answer follows the envelope
    assert cpp.read_text().startswith("// partial")               # the file the provider left is not deleted ...
    rec = _records(tmp_path, "llm_calls.jsonl")[-1]
    assert rec["error"].startswith("codex CLI exited with code 2") and "implemented req" in rec["response"]
    assert rec["run_name"] == "SystemC Model [Req]"
    assert [t["event"]["type"] for t in _records(tmp_path, "codex_turns.jsonl")] == ["thread.started", "item.completed"]
    rc, out = _inproc(tmp_path, ["model", "author", "--block", "req"])   # ... and the CLI says exit 3
    assert rc == 3 and out["ok"] is False and "exited with code 2" in out["blocks"]["req"]["provider_error"]
    assert out["blocks"]["req"]["written"] is False
    build = it.model_build(db, tmp_path)
    assert "req" not in build["missing_models"] and build["blocks"]["req"] == "existing"   # a check sees the file as-is


def test_harness_author_exits_3_on_an_exit_zero_turn_failed_with_the_harness_on_disk(tmp_path, fake_codex):
    db = _project(tmp_path)
    it.model_build(db, tmp_path)
    hdir = tmp_path / "model" / "frd_eval"
    hdir.mkdir(exist_ok=True)
    (hdir / "frd_eval.cpp").write_text("// stale harness from an earlier attempt\n")
    fake_codex(_ev({"type": "thread.started", "thread_id": "th-b"})
               + _ev({"type": "item.completed", "item": {"id": "m1", "type": "agent_message",
                                                          "text": "I started rewriting frd_eval.cpp"}})
               + _ev({"type": "turn.failed", "error": {"message": "stream disconnected before completion"}})
               + "exit 0\n")
    res = it.harness_author(db, tmp_path)
    assert res["ok"] is False and res["written"] is False
    assert res["provider_error"].startswith("[ClaudeLLM error: codex CLI reported: stream disconnected")
    assert "I started rewriting" in res["provider_error"]
    assert (hdir / "frd_eval.cpp").read_text().startswith("// stale")
    rec = _records(tmp_path, "llm_calls.jsonl")[-1]
    assert rec["error"].startswith("codex CLI reported: stream disconnected") and rec["usage"]["session_id"] == "th-b"
    rc, out = _inproc(tmp_path, ["harness", "author"])
    assert rc == 3 and out["ok"] is False and "stream disconnected" in out["provider_error"]
    assert db.checks(kind="model_eval") == []


def test_a_provider_that_succeeds_is_still_a_success_through_the_same_path(tmp_path, fake_codex):
    db = _project(tmp_path)
    cpp = tmp_path / "model" / "rsp_model.cpp"
    cpp.parent.mkdir(exist_ok=True)
    answer = json.dumps({"files_written": ["model/rsp_model.cpp"], "notes": "ok"})
    fake_codex(f"printf '// written by the provider\\n' > {cpp}\n"
               + _ev({"type": "thread.started", "thread_id": "th-c"})
               + _ev({"type": "error", "message": "stream disconnected; retrying"})      # interim, recovered
               + _ev({"type": "item.completed", "item": {"id": "m1", "type": "agent_message", "text": answer}})
               + _ev({"type": "turn.completed", "usage": {"input_tokens": 9, "output_tokens": 3}})
               + "exit 0\n")
    rc, out = _inproc(tmp_path, ["model", "author", "--block", "rsp"])
    assert rc == 0 and out["written"] == ["rsp"] and out["provider_errors"] == {}
    rec = _records(tmp_path, "llm_calls.jsonl")[-1]
    assert rec["error"] == "" and rec["usage"]["provider_notices"] == ["stream disconnected; retrying"]
    assert sorted(it.model_build(db, tmp_path)["missing_models"]) == sorted(b for b in _BLOCKS if b != "rsp")
