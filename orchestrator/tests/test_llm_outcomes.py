# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Helper-call outcomes are truthful and the trace survives the call: an empty
response, a non-zero provider exit or a terminal failure event (exit 0) makes
the call FAIL -- the caller gets the recognizable error envelope with the
partial output after it, ``llm_calls.jsonl`` records the error and the partial
response, an event is written -- while a recoverable interim notice followed
by real output stays a success. The error envelope returned to callers is
announced as ``llm_error``; provider turns are streamed to the turn log as they
arrive with the call's identity (one write per record), so a call killed by its
caller keeps what it produced."""
from __future__ import annotations

import asyncio
import json
import os
import stat
import time
from pathlib import Path

import pytest

from orchestrator.langchain.agents import coresmith_llm as cl
from orchestrator.langchain.agents.coresmith_llm import ClaudeLLM


def _fake_codex(tmp_path, body: str) -> str:
    p = tmp_path / "fake-codex.sh"
    p.write_text("#!/bin/sh\ncat >/dev/null\n" + body)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


def _ev(obj: dict) -> str:
    """One shell line printing the JSON event."""
    return "printf '%s\\n' '" + json.dumps(obj) + "'\n"


@pytest.fixture
def codex(tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(root))
    monkeypatch.delenv("CORESMITH_LLM_LOG_ROOT", raising=False)
    monkeypatch.delenv("CORESMITH_TELEMETRY_ROOT", raising=False)
    monkeypatch.setenv("CORESMITH_LLM_PROVIDER", "codex")
    monkeypatch.setenv("CORESMITH_CODEX_ISOLATE_WORKDIR", "0")
    monkeypatch.delenv("CORESMITH_LLM_SLOTS", raising=False)
    monkeypatch.delenv("CORESMITH_CODEX_RESUME", raising=False)

    def make(body: str, timeout: int = 30):
        monkeypatch.setenv("CODEX_CLI_PATH", _fake_codex(tmp_path, body))
        model = ClaudeLLM(model="gpt-5.6-sol", timeout=timeout)
        model._POLL_INTERVAL_S = 0.2
        return model, root
    return make


def _records(root: Path, name: str) -> list[dict]:
    p = root / ".coresmith" / name
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


_MSG = '{"type":"item.completed","item":{"id":"i1","type":"agent_message","text":"hello from the model"}}'


def test_empty_response_is_recorded_as_an_error(codex):
    model, root = codex('echo "Error: failed to initialize app-server client: Read-only file system" >&2\nexit 1\n')
    out = asyncio.run(model.call(system="s", prompt="p", run_name="SystemC Model [X]"))
    assert cl.is_llm_error_response(out) and "Read-only file system" in out
    rec = _records(root, "llm_calls.jsonl")[-1]
    assert rec["error"] and "Read-only file system" in rec["error"] and rec["run_name"] == "SystemC Model [X]"
    assert rec["response"] == out and rec["timed_out"] is False and isinstance(rec["call_index"], int)
    events = [e["event"] for e in _records(root, "pipeline_events.jsonl")]
    assert "llm_empty_response" in events and "llm_error" in events and "llm_end" in events
    err = [e for e in _records(root, "pipeline_events.jsonl") if e["event"] == "llm_error"][-1]
    assert "Read-only file system" in err["error"] and err["run_name"] == "SystemC Model [X]"


def test_nonzero_exit_with_output_fails_the_call_and_keeps_the_partial_output(codex):
    model, root = codex(f"echo '{_MSG}'\necho 'warning: sandbox denied' >&2\nexit 2\n")
    out = asyncio.run(model.call(system="s", prompt="p", run_name="Generate Verilog [Y]"))
    assert cl.is_llm_error_response(out)                      # the caller sees a failure ...
    assert out.startswith("[ClaudeLLM error: codex CLI exited with code 2: warning: sandbox denied]")
    assert out.endswith("\nhello from the model")              # ... with the partial output after the envelope
    rec = _records(root, "llm_calls.jsonl")[-1]
    assert rec["error"].startswith("codex CLI exited with code 2") and "sandbox denied" in rec["error"]
    assert rec["response"] == out and "hello from the model" in rec["response"]
    events = _records(root, "pipeline_events.jsonl")
    nz = [e for e in events if e["event"] == "llm_nonzero_exit"]
    assert nz and nz[-1]["exit_code"] == 2 and nz[-1]["output_chars"] == len("hello from the model")
    assert [e["event"] for e in events if e["event"] in ("llm_error", "llm_end")][-2:] == ["llm_error", "llm_end"]
    turns = _records(root, "codex_turns.jsonl")
    assert [t["event"]["item"]["text"] for t in turns] == ["hello from the model"]   # the raw trajectory kept it


def test_exit_zero_turn_failed_after_partial_text_fails_the_call(codex):
    model, root = codex(
        _ev({"type": "thread.started", "thread_id": "th-7"})
        + _ev({"type": "item.completed", "item": {"id": "m1", "type": "agent_message", "text": "half written"}})
        + _ev({"type": "turn.failed", "error": {"message": "stream disconnected before completion"}})
        + "exit 0\n")
    out = asyncio.run(model.call(system="s", prompt="p", run_name="SystemC Model [Z]"))
    assert cl.is_llm_error_response(out) and "stream disconnected before completion" in out
    assert out.endswith("\nhalf written")
    rec = _records(root, "llm_calls.jsonl")[-1]
    assert rec["error"].startswith("codex CLI reported: stream disconnected") and "half written" in rec["response"]
    assert rec["usage"]["provider_error"].startswith("stream disconnected") and rec["usage"]["session_id"] == "th-7"
    events = _records(root, "pipeline_events.jsonl")
    assert any(e["event"] == "llm_result_error" and e["exit_code"] == 0 for e in events)
    assert [t["event"]["type"] for t in _records(root, "codex_turns.jsonl")] == ["thread.started", "item.completed", "turn.failed"]


def test_exit_zero_interim_error_then_real_success_is_a_success(codex):
    model, root = codex(
        _ev({"type": "thread.started", "thread_id": "th-8"})
        + _ev({"type": "error", "message": "stream disconnected; retrying"})
        + _ev({"type": "item.completed", "item": {"id": "m1", "type": "agent_message", "text": "the whole answer"}})
        + _ev({"type": "turn.completed", "usage": {"input_tokens": 11, "output_tokens": 4}})
        + "exit 0\n")
    out = asyncio.run(model.call(system="s", prompt="p", run_name="SystemC Model [W]"))
    assert out == "the whole answer" and not cl.is_llm_error_response(out)
    rec = _records(root, "llm_calls.jsonl")[-1]
    assert rec["error"] == "" and rec["usage"]["provider_notices"] == ["stream disconnected; retrying"]
    assert rec["usage"]["input_tokens"] == 11
    events = [e["event"] for e in _records(root, "pipeline_events.jsonl")]
    assert "llm_error" not in events and "llm_result_error" not in events


def test_claude_result_is_error_fails_the_call_and_keeps_the_partial_text(tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(root))
    monkeypatch.delenv("CORESMITH_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("CORESMITH_LLM_SLOTS", raising=False)
    fake = tmp_path / "fake-claude.sh"
    fake.write_text(
        "#!/bin/sh\ncat >/dev/null\n"
        + _ev({"type": "assistant", "message": {"content": [{"type": "text", "text": "module m;"}]}})
        + _ev({"type": "result", "subtype": "error_during_execution", "is_error": True,
               "result": "API Error: 529 overloaded", "usage": {"input_tokens": 3}, "num_turns": 1})
        + "exit 0\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("CLAUDE_CLI_PATH", str(fake))
    model = ClaudeLLM(model="opus-5", timeout=30)
    model._POLL_INTERVAL_S = 0.2
    out = asyncio.run(model.call(system="s", prompt="p", run_name="Generate Verilog [Q]"))
    assert cl.is_llm_error_response(out) and "529 overloaded" in out and out.endswith("\nmodule m;")
    rec = _records(root, "llm_calls.jsonl")[-1]
    assert rec["provider"] == "claude_cli" and rec["error"].startswith("claude CLI result failed (error_during_execution)")
    assert "module m;" in rec["response"] and rec["usage"]["result_error"] == "API Error: 529 overloaded"
    assert any(e["event"] == "llm_result_error" for e in _records(root, "pipeline_events.jsonl"))


def test_turn_records_never_interleave_under_concurrent_appends(tmp_path):
    """Eight writers, forty 64 KiB records each, one file: every line parses
    and every record is intact (one write(2) per record on O_APPEND)."""
    from concurrent.futures import ThreadPoolExecutor
    path = tmp_path / ".coresmith" / "codex_turns.jsonl"
    payload = "x" * (64 * 1024)

    def _writer(w: int) -> int:
        n = 0
        for i in range(40):
            line = json.dumps({"type": "item.completed", "item": {"w": w, "i": i, "text": payload}})
            n += int(cl._append_turn(path, {"pid": w, "call_index": w, "run_name": f"w{w}"}, line))
        return n
    with ThreadPoolExecutor(max_workers=8) as ex:
        assert sum(ex.map(_writer, range(8))) == 320
    lines = path.read_text().splitlines()
    assert len(lines) == 320
    seen = set()
    for ln in lines:
        rec = json.loads(ln)                                   # no interleaved fragments
        assert rec["event"]["item"]["text"] == payload and rec["pid"] == rec["event"]["item"]["w"]
        seen.add((rec["event"]["item"]["w"], rec["event"]["item"]["i"]))
    assert len(seen) == 320


def test_turns_are_streamed_with_the_call_identity_and_survive_a_kill(codex):
    model, root = codex(
        'printf \'%s\\n\' \'{"type":"thread.started","thread_id":"th-1"}\'\n'
        'printf \'%s\\n\' \'{"type":"item.completed","item":{"id":"c1","type":"command_execution","command":"ls","exit_code":0}}\'\n'
        'sleep 60\n', timeout=3)
    t0 = time.time()
    out = asyncio.run(model.call(system="s", prompt="p", run_name="SystemC Model [Enc Entropy]"))
    assert time.time() - t0 < 30 and cl.is_llm_error_response(out) and "timed out" in out
    rec = _records(root, "llm_calls.jsonl")[-1]
    assert rec["timed_out"] is True and rec["error"].startswith("codex CLI timed out")
    turns = _records(root, "codex_turns.jsonl")
    assert [t["event"]["type"] for t in turns] == ["thread.started", "item.completed"]
    for t in turns:
        assert t["call_index"] == rec["call_index"] and t["run_name"] == "SystemC Model [Enc Entropy]"
        assert t["pid"] and t["wall_start"] and t["process_scope"]
    start = [e for e in _records(root, "pipeline_events.jsonl") if e["event"] == "llm_call_start"][-1]
    assert start["pid"] == turns[0]["pid"] and start["call_index"] == rec["call_index"]
    assert start["process_scope"] == turns[0]["process_scope"]


def test_turns_are_on_disk_before_the_call_returns(codex, tmp_path):
    """The reader thread persists each event as it arrives (a caller that kills
    the CLI mid-call, e.g. `timeout 870 coresmith model author`, keeps them)."""
    marker = tmp_path / "seen"
    model, root = codex(
        'printf \'%s\\n\' \'{"type":"thread.started","thread_id":"th-2"}\'\n'
        f'while [ ! -e {marker} ]; do sleep 0.1; done\n'
        f"echo '{_MSG}'\n", timeout=30)
    import threading

    def _release():
        deadline = time.time() + 20
        while time.time() < deadline:
            if _records(root, "codex_turns.jsonl"):
                marker.write_text("go")
                return
            time.sleep(0.1)
    th = threading.Thread(target=_release, daemon=True)
    th.start()
    out = asyncio.run(model.call(system="s", prompt="p", run_name="r"))
    th.join(5)
    assert out == "hello from the model" and marker.exists()
    assert [t["event"]["type"] for t in _records(root, "codex_turns.jsonl")] == ["thread.started", "item.completed"]


def test_error_envelope_is_announced_as_llm_error(tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(root))
    monkeypatch.delenv("CORESMITH_LLM_PROVIDER", raising=False)
    monkeypatch.setattr(cl, "_find_claude_binary", lambda: "/bin/true")
    model = ClaudeLLM(model="opus-5", timeout=5)
    monkeypatch.setattr(model, "_generate_via_cli", lambda *a, **k: "[ClaudeLLM error: claude CLI stalled after 5s]")
    out = asyncio.run(model.call(system="s", prompt="p", run_name="Stalled"))
    assert cl.is_llm_error_response(out)
    events = _records(root, "pipeline_events.jsonl")
    assert [e["event"] for e in events if e["event"] in ("llm_error", "llm_end")] == ["llm_error", "llm_end"]
    assert "stalled" in [e for e in events if e["event"] == "llm_error"][0]["error"]
    monkeypatch.setattr(model, "_generate_via_cli", lambda *a, **k: "module m; endmodule")
    asyncio.run(model.call(system="s", prompt="p", run_name="Fine"))
    tail = [e["event"] for e in _records(root, "pipeline_events.jsonl")][-2:]
    assert tail == ["llm_start", "llm_end"] or tail[-1] == "llm_end"
    assert "error" not in [e for e in _records(root, "pipeline_events.jsonl") if e["event"] == "llm_end"][-1]


def test_call_log_root_follows_the_project_root(codex):
    model, root = codex(f"echo '{_MSG}'\n")
    asyncio.run(model.call(system="s", prompt="p", run_name="r"))
    assert (root / ".coresmith" / "llm_calls.jsonl").exists()
    assert os.environ["CORESMITH_PROJECT_ROOT"] == str(root)
