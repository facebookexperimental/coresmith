# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Native sessions on OpenCode (non-Claude models: GLM / Kimi via OpenRouter,
Muse Spark): runner selection and the invocation / cluster-worker paths against a
fake ``opencode`` binary."""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from orchestrator.architect import ArchitectSession
from orchestrator.architect.cluster import ClusterSession
from orchestrator.architect.runners import (
    OPENCODE_AGENT,
    ClaudeRunner,
    OpenCodeRunner,
    opencode_config,
    resolve_model,
    resolve_runner,
)
from orchestrator.langchain.agents import coresmith_llm

ROOT = Path(__file__).resolve().parents[2]
CS = str(ROOT / "bin" / "coresmith")

_ENV_KEYS = ("CORESMITH_ARCHITECT_PROVIDER", "CORESMITH_LLM_PROVIDER", "CORESMITH_ARCHITECT_MODEL",
             "CORESMITH_OPENCODE_MODEL", "CORESMITH_MODEL", "CORESMITH_COORDINATOR_MODEL",
             "CORESMITH_OPENCODE_ENDPOINT", "CORESMITH_OPENCODE_VARIANT", "OPENCODE_CONFIG_CONTENT",
             "META_MODEL_API_KEY", "CORESMITH_OPENCODE_MAX_RETRIES")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising=False)


# A fake opencode: records argv / role / actor / PATH / the inline config, emits
# `opencode run --format json` events (step_start / tool_use / text /
# step_finish with a per-step cost) for the session given by --session (or a
# new one), and runs $FAKE_OC_DO in the cwd so "the model" can use the CLI.
# $FAKE_OC_FAIL_ONCE=<file>: the first call emits a 502 error event and exits 1.
_FAKE = r"""#!/usr/bin/env bash
prompt=$(cat)
echo "$@ ROLE=$CORESMITH_ROLE ACTOR=$CORESMITH_ACTOR CWD=$(pwd)" >> "$FAKE_LOG"
printf '%s' "$OPENCODE_CONFIG_CONTENT" > "$FAKE_LOG.config"
printf '%s' "$PATH" > "$FAKE_LOG.path"
printf '%s\n' "$prompt" >> "$FAKE_LOG.prompts"
sid="ses_fake1"
prev=""
for a in "$@"; do [ "$prev" = "--session" ] && sid="$a"; prev="$a"; done
if [ -n "$FAKE_OC_FAIL_ONCE" ] && [ ! -e "$FAKE_OC_FAIL_ONCE" ]; then
  touch "$FAKE_OC_FAIL_ONCE"
  echo '{"type":"step_start","sessionID":"'"$sid"'","part":{"type":"step-start","sessionID":"'"$sid"'"}}'
  echo '{"type":"error","sessionID":"'"$sid"'","error":{"name":"UnknownError","data":{"message":"{\"code\":502,\"message\":\"Network connection lost.\",\"metadata\":{\"error_type\":\"provider_unavailable\"}}"}}}'
  exit 1
fi
[ -n "$FAKE_OC_SLEEP" ] && sleep "$FAKE_OC_SLEEP"
echo '{"type":"step_start","sessionID":"'"$sid"'","part":{"type":"step-start","sessionID":"'"$sid"'"}}'
echo '{"type":"text","sessionID":"'"$sid"'","part":{"type":"text","text":"checking the stage"}}'
if [ -n "$FAKE_OC_DO" ]; then bash -c "$FAKE_OC_DO" >&2 || true; fi
echo '{"type":"tool_use","sessionID":"'"$sid"'","part":{"type":"tool","tool":"bash","state":{"status":"completed","input":{"command":"coresmith status --json"},"output":"{}"}}}'
echo '{"type":"step_finish","sessionID":"'"$sid"'","part":{"type":"step-finish","reason":"tool-calls","tokens":{"total":1100,"input":1000,"output":100,"reasoning":10,"cache":{"read":0,"write":0}},"cost":'"${FAKE_OC_COST:-0.1}"'}}'
echo '{"type":"step_start","sessionID":"'"$sid"'","part":{"type":"step-start","sessionID":"'"$sid"'"}}'
echo '{"type":"text","sessionID":"'"$sid"'","part":{"type":"text","text":"stage is requirements"}}'
echo '{"type":"step_finish","sessionID":"'"$sid"'","part":{"type":"step-finish","reason":"stop","tokens":{"total":1300,"input":1200,"output":100,"reasoning":0,"cache":{"read":800,"write":0}},"cost":'"${FAKE_OC_COST:-0.1}"'}}'
"""


def _fake(tmp_path, monkeypatch) -> tuple[str, Path]:
    p = tmp_path / "fake_opencode.sh"
    p.write_text(_FAKE)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "oc.log"
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.delenv("FAKE_OC_DO", raising=False)
    return str(p), log


# ---------------------------------------------------------------------------
# Runner selection
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("arch,llm,expect", [
    (None, None, "claude"),
    (None, "claude", "claude"),
    (None, "codex", "codex"),
    (None, "opencode", "opencode"),
    (None, "openrouter", "opencode"),
    ("opencode", None, "opencode"),
    ("opencode", "claude", "opencode"),
    ("claude", "opencode", "claude"),

])
def test_runner_selection_from_env(monkeypatch, arch, llm, expect):
    if arch:
        monkeypatch.setenv("CORESMITH_ARCHITECT_PROVIDER", arch)
    if llm:
        monkeypatch.setenv("CORESMITH_LLM_PROVIDER", llm)
    assert resolve_runner() == expect


def test_unknown_architect_provider_is_refused(monkeypatch):
    monkeypatch.setenv("CORESMITH_ARCHITECT_PROVIDER", "gpt")
    with pytest.raises(ValueError, match="unsupported architect runner"):
        resolve_runner()


def test_model_resolution_per_runner(monkeypatch):
    assert resolve_model("claude") == "claude-opus-5-5"
    monkeypatch.setenv("CORESMITH_MODEL", "opus-5")
    assert resolve_model("claude") == "opus-5"
    assert resolve_model("opencode") == "openrouter/moonshotai/kimi-k3"      # the engine's tier alias map
    monkeypatch.setenv("CORESMITH_OPENCODE_MODEL", "openrouter/z-ai/glm-5.3")
    assert resolve_model("opencode") == "openrouter/z-ai/glm-5.3"
    monkeypatch.setenv("CORESMITH_ARCHITECT_MODEL", "openrouter/~z-ai/glm-latest")
    assert resolve_model("opencode") == "openrouter/~z-ai/glm-latest"
    assert resolve_model("opencode", "opencode/muse-spark-1.3-contributor-free") == "opencode/muse-spark-1.3-contributor-free"
    for k in ("CORESMITH_ARCHITECT_MODEL", "CORESMITH_OPENCODE_MODEL", "CORESMITH_MODEL"):
        monkeypatch.delenv(k)
    monkeypatch.setenv("CORESMITH_OPENCODE_ENDPOINT", "muse")
    assert resolve_model("opencode") == coresmith_llm.DEFAULT_MUSE_SPARK_MODEL


def test_session_picks_the_runner_and_model(tmp_path, monkeypatch):
    assert ArchitectSession(tmp_path).runner_name == "claude"
    assert isinstance(ArchitectSession(tmp_path, claude_path="/bin/false").runner, ClaudeRunner)
    monkeypatch.setenv("CORESMITH_LLM_PROVIDER", "opencode")
    monkeypatch.setenv("CORESMITH_OPENCODE_MODEL", "openrouter/~z-ai/glm-latest")
    s = ArchitectSession(tmp_path, opencode_path="/bin/false")
    assert s.runner_name == "opencode" and s.model == "openrouter/~z-ai/glm-latest"
    assert isinstance(s.runner, OpenCodeRunner) and s.runner.binary == "/bin/false"


def test_a_recorded_session_keeps_its_runner(tmp_path, monkeypatch):
    """The daemon's environment need not match the shell that ran ``architect
    start``: a session is resumed by the CLI that created it."""
    d = tmp_path / ".coresmith" / "architect"
    d.mkdir(parents=True)
    (d / "status.json").write_text(json.dumps({"session_id": "ses_x", "runner": "opencode",
                                               "model": "openrouter/~z-ai/glm-latest"}))
    s = ArchitectSession(tmp_path)
    assert s.runner_name == "opencode" and s.model == "openrouter/~z-ai/glm-latest"
    monkeypatch.setenv("CORESMITH_ARCHITECT_PROVIDER", "claude")       # an explicit choice still wins
    assert ArchitectSession(tmp_path).runner_name == "claude"


# ---------------------------------------------------------------------------
# The runner against the fake binary
# ---------------------------------------------------------------------------
def test_start_and_resume_through_the_fake(tmp_path, monkeypatch):
    fake, log = _fake(tmp_path, monkeypatch)
    r = OpenCodeRunner(fake, "openrouter/~z-ai/glm-latest")
    kw = {"transcript": tmp_path / "t1.jsonl", "cwd": tmp_path, "env": dict(os.environ), "max_turns": 9,
          "timeout_s": 60}
    sid, text, cost, turns = r.start("hello", "SYSTEM PROMPT", **kw)
    assert sid == "ses_fake1" and text == "stage is requirements" and turns == 2
    assert cost == pytest.approx(0.2)          # the invocation's own cost: the sum of its steps
    assert (tmp_path / "system.md").read_text() == "SYSTEM PROMPT"
    argv = log.read_text().splitlines()
    assert "--session" not in argv[0]
    for flag in ("--pure run --format json --thinking --auto", "--model openrouter/~z-ai/glm-latest",
                 f"--agent {OPENCODE_AGENT}", f"--dir {tmp_path}"):
        assert flag in argv[0], flag
    cfg = json.loads(Path(f"{log}.config").read_text())
    assert cfg["instructions"] == [str(tmp_path / "system.md")]
    agent = cfg["agent"][OPENCODE_AGENT]
    assert agent["steps"] == 9 and agent["permission"]["*"] == "allow" and agent["permission"]["question"] == "deny"
    sid2, text2, cost2, _ = r.resume(sid, "continue", **dict(kw, transcript=tmp_path / "t2.jsonl"))
    assert sid2 == "ses_fake1" and "--session ses_fake1" in log.read_text().splitlines()[1]
    assert text2 == "stage is requirements" and cost2 == pytest.approx(0.2)


def test_operator_inline_config_is_kept_and_muse_registered(tmp_path):
    base = json.dumps({"instructions": ["mine.md"], "provider": {"x": {}}})
    cfg = json.loads(opencode_config(tmp_path / "system.md", 5, "openrouter/z-ai/glm-5.3", base))
    assert cfg["instructions"] == ["mine.md", str(tmp_path / "system.md")] and "x" in cfg["provider"]
    cfg = json.loads(opencode_config(tmp_path / "system.md", 5, coresmith_llm.DEFAULT_MUSE_SPARK_MODEL))
    prov = cfg["provider"][coresmith_llm.MUSE_SPARK_PROVIDER_ID]
    assert prov["options"]["apiKey"] == "{env:META_MODEL_API_KEY}"


def test_muse_endpoint_without_a_key_fails_fast(tmp_path, monkeypatch):
    fake, log = _fake(tmp_path, monkeypatch)
    monkeypatch.setenv("CORESMITH_ARCHITECT_PROVIDER", "opencode")
    monkeypatch.setenv("CORESMITH_OPENCODE_ENDPOINT", "muse")
    s = ArchitectSession(tmp_path, opencode_path=fake)
    res = s.sit("p", index=1)
    assert not res["ok"] and "META_MODEL_API_KEY" in res["error"] and not log.exists()
    monkeypatch.setenv("META_MODEL_API_KEY", "k")
    assert ArchitectSession(tmp_path, opencode_path=fake).sit("p", index=2)["ok"]
    assert coresmith_llm.MUSE_SPARK_PROVIDER_ID in json.loads(Path(f"{log}.config").read_text())["provider"]


def test_a_dropped_stream_is_resumed_once(tmp_path, monkeypatch):
    fake, log = _fake(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_OC_FAIL_ONCE", str(tmp_path / "failed-once"))
    monkeypatch.setattr("orchestrator.architect.runners.time.sleep", lambda s: None)
    res = OpenCodeRunner(fake, "openrouter/~z-ai/glm-latest").run(
        "p", system_file=tmp_path / "s.md", resume="", transcript=tmp_path / "t.jsonl", cwd=tmp_path,
        env=dict(os.environ), max_turns=5, timeout_s=120)
    assert res["ok"] and res["attempts"] == 2 and res["session_id"] == "ses_fake1"
    calls = log.read_text().splitlines()
    assert len(calls) == 2 and "--session ses_fake1" in calls[1]
    assert "provider_unavailable" in (tmp_path / "t.jsonl").read_text()     # both attempts are in the transcript


def test_the_time_limit_kills_the_sitting(tmp_path, monkeypatch):
    fake, _ = _fake(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_OC_SLEEP", "30")
    res = OpenCodeRunner(fake, "m/x").run("p", system_file=tmp_path / "s.md", resume="", transcript=tmp_path / "t.jsonl",
                                          cwd=tmp_path, env=dict(os.environ), max_turns=5, timeout_s=2)
    assert res["rc"] == 124 and not res["ok"] and "timed out" in res["stderr"]


# ---------------------------------------------------------------------------
# Invocation / cluster worker
# ---------------------------------------------------------------------------
def test_sit_writes_the_same_files_and_sets_path(tmp_path, monkeypatch):
    fake, log = _fake(tmp_path, monkeypatch)
    monkeypatch.setenv("CORESMITH_ARCHITECT_PROVIDER", "opencode")
    monkeypatch.setenv("CORESMITH_ARCHITECT_MODEL", "openrouter/~z-ai/glm-latest")
    s = ClusterSession(tmp_path, "cpu", ["mcu"], opencode_path=fake, max_turns=11, coresmith_bin=CS)
    res = s.sit("do the blocks", index=1)
    assert res["ok"] and res["runner"] == "opencode" and res["model"] == "openrouter/~z-ai/glm-latest"
    assert res["tokens"]["total_tokens"] == 2400 and res["turns"] == 2
    assert "stderr_tail" in res and "stderr" not in res
    adir = tmp_path / ".coresmith" / "clusters" / "cpu"
    for f in ("prompt-1.md", "transcript-1.jsonl", "stderr-1.log", "system.md"):
        assert (adir / f).exists(), f
    assert "coresmith block-done" in (adir / "system.md").read_text()
    assert Path(f"{log}.path").read_text().split(os.pathsep)[0] == str(ROOT / "bin")
    assert f"CWD={tmp_path}" in log.read_text() and "--agent coresmith-sitting" in log.read_text()


def test_zero_cost_records_tokens_and_a_note(tmp_path, monkeypatch):
    fake, _ = _fake(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_OC_COST", "0")
    res = ArchitectSession(tmp_path, runner="opencode", opencode_path=fake).sit("p", index=1)
    assert res["cost_usd"] == 0 and res["tokens"]["total_tokens"] == 2400 and "no cost" in res["cost_note"]


def test_cluster_worker_on_opencode_carries_its_actor(tmp_path, monkeypatch):
    fake, log = _fake(tmp_path, monkeypatch)
    monkeypatch.setenv("CORESMITH_LLM_PROVIDER", "opencode")
    s = ClusterSession(tmp_path, "cpu", ["mcu"], opencode_path=fake, coresmith_bin=CS)
    res = s.sit("work", index=1)
    assert res["ok"] and res["runner"] == "opencode" and "ACTOR=worker:cpu" in log.read_text()
    cdir = tmp_path / ".coresmith" / "clusters" / "cpu"
    assert (cdir / "transcript-1.jsonl").exists() and (cdir / "system.md").exists()
    cfg = json.loads(Path(f"{log}.config").read_text())
    assert cfg["instructions"] == [str(cdir / "system.md")]


def test_unknown_engine_provider_is_refused(monkeypatch):
    monkeypatch.setenv("CORESMITH_LLM_PROVIDER", "bogus-provider")
    with pytest.raises(ValueError, match="Unsupported CORESMITH_LLM_PROVIDER"):
        resolve_runner()
