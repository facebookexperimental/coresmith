# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for the ClaudeLLM provider abstraction.

Tests:
- Model name mapping (short names -> CLI model IDs)
- Provider detection (always claude_cli)
- Process registry (register, unregister, kill)
- Popen watchdog (stall detection, timeout, heartbeat)
- --permission-mode auto flag inclusion (headless run)
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from orchestrator.langchain.agents import coresmith_llm
from orchestrator.langchain.agents.coresmith_llm import (
    _CLI_MODEL_MAP,
    _CODEX_MODEL_MAP,
    _OPENCODE_MODEL_MAP,
    _RESUME_FLAGS_CACHE,
    DEFAULT_CODEX_MODEL,
    DEFAULT_OPENCODE_MODEL,
    DEFAULT_MODEL,
    ClaudeLLM,
    _active_processes,
    _active_processes_lock,
    _call_site_context,
    _codex_resume_supported_flags,
    _detect_provider,
    _llm_breakers,
    _llm_breakers_lock,
    _log_llm_call,
    _parse_codex_json,
    _parse_opencode_json,
    _register_process,
    _resolve_model,
    _unregister_process,
    kill_active_cli_processes,
)


def _reset_breakers():
    """Clear the LLM circuit-breaker registry so a prior test's failures
    don't trip ``call()``'s pre-flight check in a fresh test."""
    with _llm_breakers_lock:
        _llm_breakers.clear()


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch):
    monkeypatch.delenv("CORESMITH_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("CORESMITH_CODEX_MODEL", raising=False)
    monkeypatch.delenv("CORESMITH_OPENCODE_MODEL", raising=False)
    monkeypatch.delenv("OPENCODE_CONFIG_CONTENT", raising=False)
    # Keep the JSONL-log path tests hermetic: _llm_log_root() prefers these
    # over CORESMITH_PROJECT_ROOT, so an ambient value (e.g. a CI/log-capture
    # harness) would redirect writes away from the per-test tmp_path.
    monkeypatch.delenv("CORESMITH_LLM_LOG_ROOT", raising=False)
    monkeypatch.delenv("CORESMITH_TELEMETRY_ROOT", raising=False)


class TestModelNameMapping:
    def test_opus_48_maps_correctly(self):
        assert _resolve_model("opus-4.8") == "claude-opus-4-8"

    def test_opus_47_legacy_alias_maps_to_current_opus(self):
        # Legacy alias kept for back-compat with older configs.
        assert _resolve_model("opus-4.7") == "claude-opus-4-8"

    def test_opus_46_legacy_alias_maps_to_current_opus(self):
        # Legacy alias kept for back-compat with older configs.
        assert _resolve_model("opus-4.6") == "claude-opus-4-8"

    def test_sonnet_46_maps_correctly(self):
        assert _resolve_model("sonnet-4.6") == "claude-sonnet-4-6"

    def test_haiku_45_maps_correctly(self):
        assert _resolve_model("haiku-4.5") == "claude-haiku-4-5-20251001"

    def test_unknown_model_passes_through(self):
        assert _resolve_model("custom-model-123") == "custom-model-123"

    def test_all_cli_models_have_mappings(self):
        expected_shorts = ["opus-4.8", "opus-4.7", "opus-4.6", "sonnet-4.6", "sonnet-4.5", "haiku-4.5", "haiku-3.5"]
        for short in expected_shorts:
            assert short in _CLI_MODEL_MAP, f"Missing CLI mapping: {short}"

    def test_default_model_constant(self):
        assert DEFAULT_MODEL in _CLI_MODEL_MAP

    def test_empty_model_falls_back_to_default(self, monkeypatch):
        monkeypatch.delenv("CORESMITH_MODEL", raising=False)
        assert _resolve_model("") == _CLI_MODEL_MAP[DEFAULT_MODEL]

    def test_coresmith_model_env_overrides_passed_value(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_MODEL", "haiku-4.5")
        assert _resolve_model("opus-4.7") == "claude-haiku-4-5-20251001"

    def test_coresmith_model_env_with_full_id_passes_through(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_MODEL", "claude-some-future-model-99")
        assert _resolve_model("opus-4.7") == "claude-some-future-model-99"

    def test_empty_coresmith_model_does_not_override(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_MODEL", "")
        assert _resolve_model("opus-4.7") == "claude-opus-4-8"

    def test_codex_opus_maps_to_gpt_56(self):
        assert _resolve_model("opus-4.8", "codex_cli") == "gpt-5.6"

    def test_codex_default_model_constant(self):
        assert DEFAULT_CODEX_MODEL == "gpt-5.6"
        assert _CODEX_MODEL_MAP["opus-4.8"] == DEFAULT_CODEX_MODEL

    def test_coresmith_codex_model_env_overrides_passed_value(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_CODEX_MODEL", "gpt-5.5")
        assert _resolve_model("sonnet-4.6", "codex_cli") == "gpt-5.5"

    def test_opencode_all_tiers_target_hosted_kimi_k3(self):
        expected = "openrouter/moonshotai/kimi-k3"
        assert DEFAULT_OPENCODE_MODEL == expected
        assert set(_OPENCODE_MODEL_MAP.values()) == {expected}
        assert _resolve_model("opus-4.8", "opencode_cli") == expected
        assert _resolve_model("sonnet-4.6", "opencode_cli") == expected

    def test_opencode_model_env_overrides_passed_value(self, monkeypatch):
        custom = "openrouter/moonshotai/kimi-k3:exacto"
        monkeypatch.setenv("CORESMITH_OPENCODE_MODEL", custom)
        assert _resolve_model("opus-4.8", "opencode_cli") == custom


class TestProviderDetection:
    def test_defaults_to_claude_cli(self, monkeypatch):
        monkeypatch.delenv("CORESMITH_LLM_PROVIDER", raising=False)
        assert _detect_provider() == "claude_cli"

    def test_codex_provider_env(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_LLM_PROVIDER", "codex")
        assert _detect_provider() == "codex_cli"

    @pytest.mark.parametrize("alias", ["opencode", "opencode_cli", "openrouter"])
    def test_opencode_provider_aliases(self, monkeypatch, alias):
        monkeypatch.setenv("CORESMITH_LLM_PROVIDER", alias)
        assert _detect_provider() == "opencode_cli"


class TestOpenCodeJsonParsing:
    def test_parse_text_and_usage(self):
        stdout = (
            '{"type":"step_start","part":{"type":"step-start"}}\n'
            '{"type":"text","part":{"type":"text","text":"ready"}}\n'
            '{"type":"step_finish","part":{"type":"step-finish","cost":0.25,'
            '"tokens":{"total":8237,"input":6430,"output":4,"reasoning":11,'
            '"cache":{"write":0,"read":1792}}}}\n'
        )
        text, usage = _parse_opencode_json(stdout)
        assert text == "ready"
        assert usage == {
            "input_tokens": 6430,
            "output_tokens": 4,
            "total_tokens": 8237,
            "cache_read_input_tokens": 1792,
            "cache_creation_input_tokens": 0,
            "reasoning_output_tokens": 11,
            "total_cost_usd": 0.25,
        }


class TestOpenCodeInvocation:
    @patch(
        "orchestrator.langchain.agents.coresmith_llm._find_opencode_binary",
        return_value="/usr/bin/opencode",
    )
    def test_kimi_k3_command_stdin_and_tool_deny_config(
        self, _mock_find, monkeypatch, tmp_path,
    ):
        monkeypatch.setenv("CORESMITH_LLM_PROVIDER", "openrouter")
        monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))
        monkeypatch.setenv("OPENCODE_CONFIG_CONTENT", '{"theme":"system"}')
        model = ClaudeLLM(model="opus-4.8", timeout=10, disable_tools=True)

        with patch.object(model, "_run_cli_with_watchdog") as watchdog:
            watchdog.return_value = ("ready", "", 0, 1.0, False, False, {})
            output = model._generate_via_cli("system", "hello")

        assert output == "ready"
        cmd = watchdog.call_args.args[0]
        assert cmd == [
            "/usr/bin/opencode", "--pure", "run", "--format", "json",
            "--model", "openrouter/moonshotai/kimi-k3",
            "--dir", str(tmp_path), "--auto",
        ]
        assert "<system>\nsystem\n</system>" in watchdog.call_args.args[1]
        assert "<user>\nhello\n</user>" in watchdog.call_args.args[1]
        config = json.loads(watchdog.call_args.kwargs["process_env"]["OPENCODE_CONFIG_CONTENT"])
        assert config == {"theme": "system", "permission": "deny"}


class TestCodexJsonParsing:
    def test_parse_codex_json_agent_message_and_usage(self):
        stdout = (
            '{"type":"thread.started","thread_id":"t"}\n'
            '{"type":"item.completed","item":{"type":"agent_message","text":"hello"}}\n'
            '{"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":2}}\n'
        )
        text, usage = _parse_codex_json(stdout)
        assert text == "hello"
        # session_id surfaced from the thread.started event alongside token usage.
        assert usage == {"input_tokens": 10, "output_tokens": 2, "session_id": "t"}

    def test_parse_codex_json_extracts_session_id(self):
        stdout = (
            '{"type":"thread.started","thread_id":"sess-abc-123"}\n'
            '{"type":"item.completed","item":{"type":"agent_message","text":"hi"}}\n'
        )
        text, usage = _parse_codex_json(stdout)
        assert text == "hi"
        assert usage.get("session_id") == "sess-abc-123"

    def test_parse_codex_json_no_thread_started_has_no_session_id(self):
        stdout = (
            '{"type":"item.completed","item":{"type":"agent_message","text":"hi"}}\n'
            '{"type":"turn.completed","usage":{"input_tokens":1}}\n'
        )
        _, usage = _parse_codex_json(stdout)
        assert "session_id" not in usage


class TestCodexSessionResume:
    """The codex argv gains ``exec resume <id>`` only when the flag is on."""

    @pytest.fixture(autouse=True)
    def _hermetic_resume_probe(self, monkeypatch):
        # Keep these argv-shape tests hermetic + fast: force the capability
        # probe to "unknown" (None) so the resume argv is the legacy/unfiltered
        # form and no real `codex exec resume --help` subprocess is spawned.
        monkeypatch.setattr(
            coresmith_llm, "_codex_resume_supported_flags", lambda path: None,
        )

    @patch("orchestrator.langchain.agents.coresmith_llm._find_codex_binary")
    def test_resume_argv_when_flag_on(self, mock_find, monkeypatch):
        monkeypatch.setenv("CORESMITH_LLM_PROVIDER", "codex")
        monkeypatch.setenv("CORESMITH_CODEX_RESUME", "1")
        monkeypatch.delenv("CORESMITH_CODEX_MODEL", raising=False)
        monkeypatch.delenv("CORESMITH_MODEL", raising=False)
        mock_find.return_value = "/usr/bin/codex"

        model = ClaudeLLM(model="opus-4.7", timeout=10)
        with patch.object(model, "_run_cli_with_watchdog") as mock_watchdog:
            mock_watchdog.return_value = ("out", "", 0, 1.0, False, False, {})
            model._generate_via_cli("sys", "hi", "sess-xyz")

            cmd = mock_watchdog.call_args[0][0]
            assert cmd[:2] == ["/usr/bin/codex", "exec"]
            # `resume <id>` is inserted right after `exec`.
            assert cmd[2] == "resume"
            assert cmd[3] == "sess-xyz"
            # the normal flags + trailing stdin sentinel are still present.
            assert "--json" in cmd
            assert cmd[-1] == "-"

    @patch("orchestrator.langchain.agents.coresmith_llm._find_codex_binary")
    def test_no_resume_when_flag_off(self, mock_find, monkeypatch):
        monkeypatch.setenv("CORESMITH_LLM_PROVIDER", "codex")
        monkeypatch.delenv("CORESMITH_CODEX_RESUME", raising=False)
        monkeypatch.delenv("CORESMITH_CODEX_MODEL", raising=False)
        monkeypatch.delenv("CORESMITH_MODEL", raising=False)
        mock_find.return_value = "/usr/bin/codex"

        model = ClaudeLLM(model="opus-4.7", timeout=10)
        with patch.object(model, "_run_cli_with_watchdog") as mock_watchdog:
            mock_watchdog.return_value = ("out", "", 0, 1.0, False, False, {})
            # a session id is passed, but the flag is OFF -> plain exec.
            model._generate_via_cli("sys", "hi", "sess-xyz")

            cmd = mock_watchdog.call_args[0][0]
            assert cmd[:2] == ["/usr/bin/codex", "exec"]
            assert "resume" not in cmd
            assert "sess-xyz" not in cmd

    @patch("orchestrator.langchain.agents.coresmith_llm._find_codex_binary")
    def test_no_resume_when_no_session_id(self, mock_find, monkeypatch):
        monkeypatch.setenv("CORESMITH_LLM_PROVIDER", "codex")
        monkeypatch.setenv("CORESMITH_CODEX_RESUME", "1")
        mock_find.return_value = "/usr/bin/codex"

        model = ClaudeLLM(model="opus-4.7", timeout=10)
        with patch.object(model, "_run_cli_with_watchdog") as mock_watchdog:
            mock_watchdog.return_value = ("out", "", 0, 1.0, False, False, {})
            model._generate_via_cli("sys", "hi", None)
            cmd = mock_watchdog.call_args[0][0]
            assert "resume" not in cmd

    @patch("orchestrator.langchain.agents.coresmith_llm._find_codex_binary")
    def test_resume_failure_falls_back_to_fresh_exec(self, mock_find, monkeypatch):
        monkeypatch.setenv("CORESMITH_LLM_PROVIDER", "codex")
        monkeypatch.setenv("CORESMITH_CODEX_RESUME", "1")
        mock_find.return_value = "/usr/bin/codex"

        model = ClaudeLLM(model="opus-4.7", timeout=10)
        # first call (resume) fails with rc!=0 / empty; second (fresh) succeeds.
        with patch.object(model, "_run_cli_with_watchdog") as mock_watchdog:
            mock_watchdog.side_effect = [
                ("", "no such session", 1, 1.0, False, False, {}),
                ("recovered output", "", 0, 1.0, False, False, {"session_id": "new"}),
            ]
            out = model._generate_via_cli("sys", "hi", "gone-session")

        assert out == "recovered output"
        assert mock_watchdog.call_count == 2
        first_cmd = mock_watchdog.call_args_list[0][0][0]
        second_cmd = mock_watchdog.call_args_list[1][0][0]
        assert first_cmd[2] == "resume"          # attempt 1 tried to resume
        assert "resume" not in second_cmd        # attempt 2 is a fresh exec
        assert model.last_session_id == "new"    # new session captured


class TestLLMTelemetry:
    def test_log_llm_call_backdates_otel_span(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))
        recorded = {}

        class FakeSpan:
            def set_attribute(self, *args):
                pass

            def set_status(self, *args):
                pass

            def end(self, end_time=None):
                recorded["end_time"] = end_time

        class FakeTracer:
            def start_span(self, name, attributes=None, start_time=None):
                recorded["name"] = name
                recorded["attributes"] = attributes or {}
                recorded["start_time"] = start_time
                return FakeSpan()

        start_ns = time.time_ns() - 2_500_000_000
        with patch(
            "orchestrator.langchain.agents.coresmith_llm._get_llm_tracer",
            return_value=FakeTracer(),
        ):
            _log_llm_call(
                model="gpt-5.5",
                provider="codex_cli",
                system_prompt="system",
                user_prompt="prompt",
                response="response",
                duration_s=2.5,
                timeout=60,
                start_ts_ns=start_ns,
            )

        assert recorded["name"] == "LLM gpt-5.5 (codex_cli)"
        assert recorded["start_time"] == start_ns
        assert recorded["end_time"] is not None
        assert recorded["end_time"] - recorded["start_time"] >= 2_400_000_000
        assert recorded["attributes"]["llm.duration_s"] == 2.5


class TestProcessRegistry:
    """Test the active subprocess registry (Fix #11)."""

    def setup_method(self):
        with _active_processes_lock:
            _active_processes.clear()

    def teardown_method(self):
        with _active_processes_lock:
            _active_processes.clear()

    def test_register_and_unregister(self):
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = None
        mock_proc.pid = 12345

        _register_process(mock_proc)
        tid = threading.get_ident()
        with _active_processes_lock:
            assert tid in _active_processes
            assert _active_processes[tid] is mock_proc

        _unregister_process()
        with _active_processes_lock:
            assert tid not in _active_processes

    def test_kill_active_processes(self):
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = None  # still running
        mock_proc.pid = 99999

        _register_process(mock_proc)

        killed = kill_active_cli_processes()
        assert killed == 1
        mock_proc.kill.assert_called_once()

        with _active_processes_lock:
            assert len(_active_processes) == 0

    def test_kill_skips_already_exited(self):
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = 0  # already exited
        mock_proc.pid = 11111

        _register_process(mock_proc)

        killed = kill_active_cli_processes()
        assert killed == 0
        mock_proc.kill.assert_not_called()

    def test_unregister_idempotent(self):
        _unregister_process()
        _unregister_process()


class TestCommandConstruction:
    """Verify the CLI command runs headlessly via --permission-mode auto."""

    @patch("orchestrator.langchain.agents.coresmith_llm._find_claude_binary")
    def test_permission_mode_auto_in_cmd(self, mock_find):
        mock_find.return_value = "/usr/bin/claude"

        model = ClaudeLLM(model="opus-4.6", timeout=10)

        with patch.object(model, "_run_cli_with_watchdog") as mock_watchdog:
            mock_watchdog.return_value = ("test output", "", 0, 1.0, False, False, {})
            model._generate_via_cli("system prompt", "hello")

            call_args = mock_watchdog.call_args
            cmd = call_args[0][0]  # first positional arg is cmd
            assert "--permission-mode" in cmd
            assert cmd[cmd.index("--permission-mode") + 1] == "auto"
            assert "--dangerously-skip-permissions" not in cmd

    @patch("orchestrator.langchain.agents.coresmith_llm._find_claude_binary")
    def test_print_mode_flags(self, mock_find):
        mock_find.return_value = "/usr/bin/claude"

        model = ClaudeLLM(model="opus-4.6", timeout=10)

        with patch.object(model, "_run_cli_with_watchdog") as mock_watchdog:
            mock_watchdog.return_value = ("test output", "", 0, 1.0, False, False, {})
            model._generate_via_cli("system prompt", "hello")

            cmd = mock_watchdog.call_args[0][0]
            assert "-p" in cmd
            assert "--output-format" in cmd
            # stream-json gives us per-call usage + cost via the `result` event
            assert "stream-json" in cmd
            # CLI requires --verbose alongside stream-json under --print
            assert "--verbose" in cmd

    @patch("orchestrator.langchain.agents.coresmith_llm._find_codex_binary")
    def test_codex_exec_flags_and_gpt_56_model(self, mock_find, monkeypatch):
        monkeypatch.setenv("CORESMITH_LLM_PROVIDER", "codex")
        monkeypatch.delenv("CORESMITH_CODEX_MODEL", raising=False)
        monkeypatch.delenv("CORESMITH_MODEL", raising=False)
        mock_find.return_value = "/usr/bin/codex"

        model = ClaudeLLM(model="opus-4.8", timeout=10)

        with patch.object(model, "_run_cli_with_watchdog") as mock_watchdog:
            mock_watchdog.return_value = ("test output", "", 0, 1.0, False, False, {})
            model._generate_via_cli("system prompt", "hello")

            cmd = mock_watchdog.call_args[0][0]
            assert cmd[:2] == ["/usr/bin/codex", "exec"]
            assert "--json" in cmd
            assert "-m" in cmd
            assert cmd[cmd.index("-m") + 1] == "gpt-5.6"
            assert "--dangerously-bypass-approvals-and-sandbox" in cmd


class TestWatchdogBehaviour:
    """Test stall detection and timeout in _run_cli_with_watchdog."""

    @patch("orchestrator.langchain.agents.coresmith_llm._find_claude_binary")
    def test_timeout_returns_partial_output(self, mock_find):
        """When the hard timeout fires, partial output should be captured."""
        mock_find.return_value = "/usr/bin/echo"

        model = ClaudeLLM(model="opus-4.6", timeout=3)

        result = model._generate_via_cli("system prompt", "hello")
        assert isinstance(result, str)

    @patch("orchestrator.langchain.agents.coresmith_llm._find_claude_binary")
    def test_stall_detection_with_short_threshold(self, mock_find):
        """A process that produces no output should be killed by stall detection."""
        mock_find.return_value = "/bin/sleep"

        model = ClaudeLLM(model="opus-4.6", timeout=600)
        model._STALL_THRESHOLD_S = 3
        model._POLL_INTERVAL_S = 0.5

        cmd = ["/bin/sleep", "600"]
        t0 = time.monotonic()

        stdout, stderr, rc, elapsed, timed_out, stalled, usage = (
            model._run_cli_with_watchdog(cmd, "", "/tmp", "test", t0)
        )

        assert stalled is True
        assert timed_out is False
        assert elapsed < 30  # should be killed well before timeout
        # No `result` event in a stalled stream → usage dict is empty.
        assert usage == {}

    @patch("orchestrator.langchain.agents.coresmith_llm._find_claude_binary")
    def test_grandchild_holding_pipe_does_not_stall_exit(self, mock_find, tmp_path):
        """A CLI that emits its response then leaves a grandchild (a spawned
        sim) holding the stdout pipe open must NOT stall the call until the
        hard-timeout deadline. The post-response process-group reap kills the
        grandchild so the reader threads see EOF and the call returns promptly.
        """
        # Stub CLI: print the final result event, spawn a `sleep 60` grandchild
        # that inherits stdout (holding the pipe write-end open), then exit.
        stub = tmp_path / "stub_cli.sh"
        stub.write_text(
            "#!/bin/sh\n"
            "printf '{\"type\":\"result\",\"result\":\"DONE\"}\\n'\n"
            "sleep 60 &\n"
            "exit 0\n"
        )
        stub.chmod(0o755)
        mock_find.return_value = str(stub)

        # Large hard timeout so ONLY the reap logic (not the timeout) can end
        # the call; a regression would block on the pipe for ~timeout seconds.
        model = ClaudeLLM(model="opus-4.6", timeout=600)
        model._POLL_INTERVAL_S = 0.2
        model._REAP_GRACE_S = 5.0

        t0 = time.monotonic()
        (response_text, stderr, rc, elapsed, timed_out, stalled, usage) = (
            model._run_cli_with_watchdog(
                [str(stub)], "prompt", str(tmp_path), "test", t0
            )
        )
        wall = time.monotonic() - t0

        # The response completed cleanly...
        assert timed_out is False
        assert "DONE" in response_text
        # ...and the call returned within a few seconds of completion, NOT after
        # the 600s hard timeout (regression) -- the grandchild was reaped.
        assert wall < 15, f"call stalled {wall:.1f}s after response (grandchild not reaped)"


# ═══════════════════════════════════════════════════════════════════════════
# A-Fix 6: codex `exec resume` flag capability probe
# ═══════════════════════════════════════════════════════════════════════════
class TestCodexResumeFlagProbe:
    def setup_method(self):
        _RESUME_FLAGS_CACHE.clear()

    def teardown_method(self):
        _RESUME_FLAGS_CACHE.clear()

    def test_probe_extracts_long_and_short_flags(self):
        help_text = (
            "Resume a previous session\n\n"
            "Usage: codex exec resume [OPTIONS] [SESSION_ID]\n\n"
            "Options:\n"
            "      --json            Print events as JSON\n"
            "      --skip-git-repo-check\n"
            "  -C, --cd <DIR>        Working directory\n"
            "  -c <KEY=VALUE>        Config override\n"
            "  -m, --model <MODEL>   Model to use\n"
        )
        proc = MagicMock(stdout=help_text, stderr="")
        with patch("orchestrator.langchain.agents.coresmith_llm.subprocess.run", return_value=proc) as mrun:
            flags = _codex_resume_supported_flags("/usr/bin/codex")
        assert flags is not None
        assert "--json" in flags
        assert "-C" in flags and "--cd" in flags
        assert "-m" in flags and "--model" in flags
        assert "-c" in flags
        # The newer CLI in the measured bug does NOT accept --sandbox on resume.
        assert "--sandbox" not in flags
        mrun.assert_called_once()

    def test_probe_is_cached_per_binary(self):
        proc = MagicMock(stdout="  --json\n", stderr="")
        with patch("orchestrator.langchain.agents.coresmith_llm.subprocess.run", return_value=proc) as mrun:
            _codex_resume_supported_flags("/usr/bin/codex")
            _codex_resume_supported_flags("/usr/bin/codex")
        assert mrun.call_count == 1  # second call served from cache

    def test_probe_failure_returns_none(self):
        with patch(
            "orchestrator.langchain.agents.coresmith_llm.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            assert _codex_resume_supported_flags("/nope/codex") is None

    def test_probe_no_flags_returns_none(self):
        proc = MagicMock(stdout="no options here", stderr="")
        with patch("orchestrator.langchain.agents.coresmith_llm.subprocess.run", return_value=proc):
            assert _codex_resume_supported_flags("/usr/bin/codex") is None

    def test_empty_path_returns_none_without_subprocess(self):
        with patch("orchestrator.langchain.agents.coresmith_llm.subprocess.run") as mrun:
            assert _codex_resume_supported_flags("") is None
            mrun.assert_not_called()


class TestBuildCodexCmdFlagFiltering:
    """`_build_codex_cmd` drops unsupported flags ONLY on the resume argv."""

    def test_fresh_argv_never_filtered_byte_identical(self, monkeypatch):
        monkeypatch.delenv("CORESMITH_CODEX_RESUME", raising=False)
        expected = [
            "/x/codex", "exec",
            "--json",
            "--dangerously-bypass-approvals-and-sandbox",
            "--sandbox", "workspace-write",
            "--skip-git-repo-check",
            "-C", "/wd",
            "-c", "model_reasoning_effort=high",
            "-m", "gpt-5.5",
            "-",
        ]
        # A supported_flags set that would drop everything must NOT affect fresh.
        cmd = ClaudeLLM._build_codex_cmd(
            "/x/codex", "gpt-5.5", "/wd", "workspace-write",
            None, supported_flags=frozenset(),
        )
        assert cmd == expected

    def test_resume_legacy_argv_when_probe_none(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_CODEX_RESUME", "1")
        cmd = ClaudeLLM._build_codex_cmd(
            "/x/codex", "gpt-5.5", "/wd", "workspace-write",
            "sess", supported_flags=None,
        )
        assert cmd[2:4] == ["resume", "sess"]
        assert "--sandbox" in cmd  # None -> no filtering

    def test_resume_drops_unsupported_flags(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_CODEX_RESUME", "1")
        supported = frozenset({"--json", "-C", "-m"})
        cmd = ClaudeLLM._build_codex_cmd(
            "/x/codex", "gpt-5.5", "/wd", "workspace-write",
            "sess", supported_flags=supported,
        )
        assert cmd[2:4] == ["resume", "sess"]
        assert "--sandbox" not in cmd
        assert "workspace-write" not in cmd  # dropped flag's value gone too
        assert "--dangerously-bypass-approvals-and-sandbox" not in cmd
        assert "--json" in cmd
        assert cmd[cmd.index("-m") + 1] == "gpt-5.5"
        assert cmd[cmd.index("-C") + 1] == "/wd"
        assert cmd[-1] == "-"

    def test_resume_flags_dropped_helper(self):
        dropped = ClaudeLLM._resume_flags_dropped(frozenset({"--json", "-C", "-m"}))
        assert "--sandbox" in dropped
        assert "--dangerously-bypass-approvals-and-sandbox" in dropped
        assert "--json" not in dropped


class TestReasoningEffortTiering:
    """Per-stage codex reasoning effort: the architecture specialists (PRD/SAD/
    FRD/uarch) pass ``reasoning_effort=arch_reasoning_effort()`` (default
    xhigh); everything else stays at the global default (high)."""

    def _effort_of(self, cmd: list[str]) -> str:
        vals = [t for t in cmd if t.startswith("model_reasoning_effort=")]
        assert len(vals) == 1
        return vals[0].split("=", 1)[1]

    def test_default_is_high(self, monkeypatch):
        monkeypatch.delenv("CORESMITH_CODEX_REASONING_EFFORT", raising=False)
        cmd = ClaudeLLM._build_codex_cmd(
            "/x/codex", "gpt-5.5", "/wd", "workspace-write", None)
        assert self._effort_of(cmd) == "high"

    def test_instance_override_wins(self, monkeypatch):
        monkeypatch.delenv("CORESMITH_CODEX_REASONING_EFFORT", raising=False)
        cmd = ClaudeLLM._build_codex_cmd(
            "/x/codex", "gpt-5.5", "/wd", "workspace-write", None,
            reasoning_effort="xhigh")
        assert self._effort_of(cmd) == "xhigh"

    def test_override_beats_global_env(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_CODEX_REASONING_EFFORT", "medium")
        cmd = ClaudeLLM._build_codex_cmd(
            "/x/codex", "gpt-5.5", "/wd", "workspace-write", None,
            reasoning_effort="xhigh")
        assert self._effort_of(cmd) == "xhigh"
        # ...and without the override the global env still applies.
        cmd = ClaudeLLM._build_codex_cmd(
            "/x/codex", "gpt-5.5", "/wd", "workspace-write", None)
        assert self._effort_of(cmd) == "medium"

    def test_arch_reasoning_effort_default_and_env(self, monkeypatch):
        from orchestrator.langchain.agents.coresmith_llm import arch_reasoning_effort
        monkeypatch.delenv("CORESMITH_CODEX_REASONING_EFFORT_ARCH",
                           raising=False)
        assert arch_reasoning_effort() == "xhigh"
        monkeypatch.setenv("CORESMITH_CODEX_REASONING_EFFORT_ARCH", "high")
        assert arch_reasoning_effort() == "high"


class TestCodexResumeReprobe:
    """On `unexpected argument`, re-probe once and retry the resume before
    giving up on the session (fresh exec)."""

    @patch("orchestrator.langchain.agents.coresmith_llm._find_codex_binary")
    def test_unexpected_argument_reprobes_then_retries_resume(self, mock_find, monkeypatch):
        monkeypatch.setenv("CORESMITH_LLM_PROVIDER", "codex")
        monkeypatch.setenv("CORESMITH_CODEX_RESUME", "1")
        monkeypatch.delenv("CORESMITH_CODEX_MODEL", raising=False)
        monkeypatch.delenv("CORESMITH_MODEL", raising=False)
        mock_find.return_value = "/usr/bin/codex"

        # 1st probe: CLI (wrongly, per stale cache) advertises --sandbox.
        # 2nd probe (after invalidate): no --sandbox.
        probes = iter([
            frozenset({
                "--json", "--dangerously-bypass-approvals-and-sandbox",
                "--sandbox", "--skip-git-repo-check", "-C", "-c", "-m",
            }),
            frozenset({
                "--json", "--dangerously-bypass-approvals-and-sandbox",
                "--skip-git-repo-check", "-C", "-c", "-m",
            }),
        ])
        monkeypatch.setattr(
            coresmith_llm, "_codex_resume_supported_flags", lambda p: next(probes),
        )

        model = ClaudeLLM(model="opus-4.7", timeout=10)
        with patch.object(model, "_run_cli_with_watchdog") as mw:
            mw.side_effect = [
                ("", "error: unexpected argument '--sandbox'", 2, 1.0, False, False, {}),
                ("ok after reprobe", "", 0, 1.0, False, False, {"session_id": "s2"}),
            ]
            out = model._generate_via_cli("sys", "hi", "sess-1")

        assert out == "ok after reprobe"
        assert mw.call_count == 2
        first_cmd = mw.call_args_list[0][0][0]
        second_cmd = mw.call_args_list[1][0][0]
        assert "--sandbox" in first_cmd            # the failing argv
        assert second_cmd[2] == "resume"           # retried as a RESUME, not fresh
        assert "--sandbox" not in second_cmd       # filtered after re-probe
        assert model.last_session_id == "s2"

    @patch("orchestrator.langchain.agents.coresmith_llm._find_codex_binary")
    def test_unexpected_argument_then_falls_back_to_fresh(self, mock_find, monkeypatch):
        monkeypatch.setenv("CORESMITH_LLM_PROVIDER", "codex")
        monkeypatch.setenv("CORESMITH_CODEX_RESUME", "1")
        monkeypatch.delenv("CORESMITH_CODEX_MODEL", raising=False)
        monkeypatch.delenv("CORESMITH_MODEL", raising=False)
        mock_find.return_value = "/usr/bin/codex"
        monkeypatch.setattr(
            coresmith_llm, "_codex_resume_supported_flags",
            lambda p: frozenset({"--json", "-C", "-c", "-m"}),
        )

        model = ClaudeLLM(model="opus-4.7", timeout=10)
        with patch.object(model, "_run_cli_with_watchdog") as mw:
            mw.side_effect = [
                ("", "error: unexpected argument '--sandbox'", 2, 1.0, False, False, {}),
                ("", "still broken", 1, 1.0, False, False, {}),
                ("fresh ok", "", 0, 1.0, False, False, {"session_id": "s3"}),
            ]
            out = model._generate_via_cli("sys", "hi", "sess-1")

        assert out == "fresh ok"
        assert mw.call_count == 3
        third_cmd = mw.call_args_list[2][0][0]
        assert "resume" not in third_cmd           # final attempt is a fresh exec


class TestCodexResumeCwd:
    """[plan-8b] When the flag filter drops -C on a resume argv, the intended
    isolated workdir must be handed to the Popen via cwd= so relative-path
    writes in the resumed turn don't land in the caller's cwd."""

    def test_launch_cwd_helper(self):
        wd = "/run/dir"
        # -C present -> codex switches itself; launch cwd irrelevant (None).
        assert ClaudeLLM._codex_launch_cwd(["codex", "exec", "-C", wd, "-"], wd) is None
        assert ClaudeLLM._codex_launch_cwd(["codex", "exec", "--cd", wd, "-"], wd) is None
        # -C dropped -> hand the workdir to Popen(cwd=...).
        assert (
            ClaudeLLM._codex_launch_cwd(["codex", "exec", "resume", "s", "--json", "-"], wd)
            == wd
        )

    @patch("orchestrator.langchain.agents.coresmith_llm._find_codex_binary")
    def test_resume_drops_C_passes_cwd_to_popen(self, mock_find, monkeypatch, tmp_path):
        monkeypatch.setenv("CORESMITH_LLM_PROVIDER", "codex")
        monkeypatch.setenv("CORESMITH_CODEX_RESUME", "1")
        # Keep the workdir predictable: no per-call codex-call-* temp dir.
        monkeypatch.setenv("CORESMITH_CODEX_ISOLATE_WORKDIR", "0")
        monkeypatch.setenv("CORESMITH_CODEX_WORKDIR", str(tmp_path))
        monkeypatch.delenv("CORESMITH_CODEX_MODEL", raising=False)
        monkeypatch.delenv("CORESMITH_MODEL", raising=False)
        mock_find.return_value = "/usr/bin/codex"
        # The CLI advertises every flag EXCEPT -C on `exec resume`.
        monkeypatch.setattr(
            coresmith_llm, "_codex_resume_supported_flags",
            lambda p: frozenset({
                "--json", "--dangerously-bypass-approvals-and-sandbox",
                "--sandbox", "--skip-git-repo-check", "-c", "-m",
            }),
        )

        model = ClaudeLLM(model="opus-4.7", timeout=10)
        with patch.object(model, "_run_cli_with_watchdog") as mw:
            mw.return_value = ("out", "", 0, 1.0, False, False, {})
            model._generate_via_cli("sys", "hi", "sess-xyz")

        cmd = mw.call_args[0][0]
        assert cmd[2] == "resume"
        assert "-C" not in cmd                       # filter dropped it
        # ...and the workdir is restored via Popen cwd=.
        assert mw.call_args.kwargs.get("cwd") == str(tmp_path)

    @patch("orchestrator.langchain.agents.coresmith_llm._find_codex_binary")
    def test_resume_keeps_C_leaves_cwd_none(self, mock_find, monkeypatch, tmp_path):
        monkeypatch.setenv("CORESMITH_LLM_PROVIDER", "codex")
        monkeypatch.setenv("CORESMITH_CODEX_RESUME", "1")
        monkeypatch.setenv("CORESMITH_CODEX_ISOLATE_WORKDIR", "0")
        monkeypatch.setenv("CORESMITH_CODEX_WORKDIR", str(tmp_path))
        monkeypatch.delenv("CORESMITH_CODEX_MODEL", raising=False)
        monkeypatch.delenv("CORESMITH_MODEL", raising=False)
        mock_find.return_value = "/usr/bin/codex"
        # -C IS supported -> codex switches itself; launch cwd stays None.
        monkeypatch.setattr(
            coresmith_llm, "_codex_resume_supported_flags",
            lambda p: frozenset({
                "--json", "--dangerously-bypass-approvals-and-sandbox",
                "--sandbox", "--skip-git-repo-check", "-C", "-c", "-m",
            }),
        )

        model = ClaudeLLM(model="opus-4.7", timeout=10)
        with patch.object(model, "_run_cli_with_watchdog") as mw:
            mw.return_value = ("out", "", 0, 1.0, False, False, {})
            model._generate_via_cli("sys", "hi", "sess-xyz")

        cmd = mw.call_args[0][0]
        assert "-C" in cmd
        assert mw.call_args.kwargs.get("cwd") is None


# ═══════════════════════════════════════════════════════════════════════════
# C2: testing-provider registry (fault/replay)
# ═══════════════════════════════════════════════════════════════════════════
class TestTestingProviderRegistry:
    def test_detect_provider_accepts_fault_and_replay(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_LLM_PROVIDER", "fault")
        assert _detect_provider() == "fault"
        monkeypatch.setenv("CORESMITH_LLM_PROVIDER", "replay")
        assert _detect_provider() == "replay"

    def test_detect_provider_rejects_unknown(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_LLM_PROVIDER", "bogus")
        with pytest.raises(ValueError):
            _detect_provider()

    def test_testing_provider_needs_no_binary(self, monkeypatch):
        # Construction must NOT try to locate a claude/codex binary.
        monkeypatch.setenv("CORESMITH_LLM_PROVIDER", "fault")
        with patch(
            "orchestrator.langchain.agents.coresmith_llm._find_codex_binary",
            side_effect=AssertionError("should not probe for a binary"),
        ), patch(
            "orchestrator.langchain.agents.coresmith_llm._find_claude_binary",
            side_effect=AssertionError("should not probe for a binary"),
        ):
            model = ClaudeLLM(model="opus-4.7", timeout=5)
        assert model._provider == "fault"

    def test_generate_via_cli_dispatches_to_testing_backend(self, monkeypatch):
        import sys
        import types

        seen = {}

        class FakeBackend:
            def generate(self, llm, system, prompt, resume):
                seen["args"] = (system, prompt, resume)
                seen["llm_is_instance"] = isinstance(llm, ClaudeLLM)
                return "FAULT_OUTPUT"

        fake_mod = types.ModuleType("orchestrator.testing.fault_provider")
        fake_mod.get_backend = lambda: FakeBackend()
        monkeypatch.setitem(sys.modules, "orchestrator.testing.fault_provider", fake_mod)

        monkeypatch.setenv("CORESMITH_LLM_PROVIDER", "fault")
        model = ClaudeLLM(model="opus-4.7", timeout=5)
        out = model._generate_via_cli("SYS", "USR", "rid-1")
        assert out == "FAULT_OUTPUT"
        assert seen["args"] == ("SYS", "USR", "rid-1")
        assert seen["llm_is_instance"] is True

    def test_missing_testing_module_raises_clear_error(self, monkeypatch):
        import sys

        # Ensure the real (absent) module is not importable.
        monkeypatch.setitem(sys.modules, "orchestrator.testing.replay_provider", None)
        monkeypatch.setenv("CORESMITH_LLM_PROVIDER", "replay")
        model = ClaudeLLM(model="opus-4.7", timeout=5)
        with pytest.raises((RuntimeError, ImportError)):
            model._generate_via_cli("s", "p", None)


# ═══════════════════════════════════════════════════════════════════════════
# C2: call-site attribution (run_name / call_index / graph) in llm_calls.jsonl
# ═══════════════════════════════════════════════════════════════════════════
class TestCallSiteAttribution:
    def test_log_llm_call_reads_call_site_context(self, tmp_path, monkeypatch):
        import json as _json

        monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))
        token = _call_site_context.set(
            {"run_name": "generate_rtl:blk", "call_index": 42, "graph": "pipeline"}
        )
        try:
            _log_llm_call(
                model="gpt-5.5",
                provider="codex_cli",
                system_prompt="s",
                user_prompt="p",
                response="r",
                duration_s=1.0,
                timeout=60,
            )
        finally:
            _call_site_context.reset(token)

        rec = _json.loads(
            (tmp_path / ".coresmith" / "llm_calls.jsonl").read_text().strip()
        )
        assert rec["run_name"] == "generate_rtl:blk"
        assert rec["call_index"] == 42
        assert rec["graph"] == "pipeline"

    def test_log_llm_call_absent_context_defaults_empty(self, tmp_path, monkeypatch):
        import json as _json

        monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))
        # No context set (direct log call).
        _call_site_context.set(None)
        _log_llm_call(
            model="m", provider="claude_cli", system_prompt="", user_prompt="",
            response="x", duration_s=0.1, timeout=1,
        )
        rec = _json.loads(
            (tmp_path / ".coresmith" / "llm_calls.jsonl").read_text().strip()
        )
        assert rec["run_name"] == ""
        assert rec["call_index"] is None

    @patch("orchestrator.langchain.agents.coresmith_llm._find_claude_binary")
    def test_call_propagates_context_into_executor(self, mock_find, tmp_path, monkeypatch):
        """End-to-end: call() sets the ContextVar and it survives the
        run_in_executor hop so _log_llm_call (in the worker thread) records it."""
        import asyncio
        import json as _json

        _reset_breakers()
        monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))
        monkeypatch.delenv("CORESMITH_LLM_PROVIDER", raising=False)
        mock_find.return_value = "/usr/bin/claude"

        model = ClaudeLLM(model="opus-4.7", timeout=10)
        with patch.object(model, "_run_cli_with_watchdog") as mw:
            mw.return_value = ("hello world", "", 0, 1.0, False, False, {})
            out = asyncio.run(
                model.call(system="s", prompt="p", run_name="generate_rtl:adder8")
            )
        assert out == "hello world"

        lines = (tmp_path / ".coresmith" / "llm_calls.jsonl").read_text().strip().splitlines()
        rec = _json.loads(lines[-1])
        assert rec["run_name"] == "generate_rtl:adder8"
        assert isinstance(rec["call_index"], int) and rec["call_index"] >= 1

    @patch("orchestrator.langchain.agents.coresmith_llm._find_claude_binary")
    def test_call_indices_are_monotonic(self, mock_find, tmp_path, monkeypatch):
        import asyncio
        import json as _json

        _reset_breakers()
        monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))
        monkeypatch.delenv("CORESMITH_LLM_PROVIDER", raising=False)
        mock_find.return_value = "/usr/bin/claude"

        model = ClaudeLLM(model="opus-4.7", timeout=10)
        with patch.object(model, "_run_cli_with_watchdog") as mw:
            mw.return_value = ("x", "", 0, 1.0, False, False, {})
            asyncio.run(model.call(system="s", prompt="p", run_name="a"))
            asyncio.run(model.call(system="s", prompt="p", run_name="b"))

        lines = (tmp_path / ".coresmith" / "llm_calls.jsonl").read_text().strip().splitlines()
        idxs = [_json.loads(line)["call_index"] for line in lines[-2:]]
        assert idxs[1] > idxs[0]


class TestFailedCallLogging:
    """call()'s except branch now records failed calls (raised exceptions)."""

    @patch("orchestrator.langchain.agents.coresmith_llm._find_claude_binary")
    def test_raised_exception_is_logged(self, mock_find, tmp_path, monkeypatch):
        import asyncio
        import json as _json

        _reset_breakers()
        monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))
        monkeypatch.delenv("CORESMITH_LLM_PROVIDER", raising=False)
        mock_find.return_value = "/usr/bin/claude"

        model = ClaudeLLM(model="opus-4.7", timeout=10)
        with patch.object(
            model, "_run_cli_with_watchdog", side_effect=RuntimeError("provider boom"),
        ), pytest.raises(RuntimeError):
            asyncio.run(model.call(system="s", prompt="p", run_name="rn"))

        log = tmp_path / ".coresmith" / "llm_calls.jsonl"
        rec = _json.loads(log.read_text().strip().splitlines()[-1])
        assert "provider boom" in rec["error"]
        assert rec["response"] == ""
        assert rec["run_name"] == "rn"
