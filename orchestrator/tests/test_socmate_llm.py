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

import subprocess
import threading
import time
from unittest.mock import MagicMock, patch

import pytest


from orchestrator.langchain.agents.socmate_llm import (
    ClaudeLLM,
    DEFAULT_CODEX_MODEL,
    DEFAULT_MODEL,
    _CODEX_MODEL_MAP,
    _CLI_MODEL_MAP,
    _detect_provider,
    _log_llm_call,
    _parse_codex_json,
    _resolve_model,
    _register_process,
    _unregister_process,
    _active_processes,
    _active_processes_lock,
    kill_active_cli_processes,
)


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch):
    monkeypatch.delenv("SOCMATE_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("SOCMATE_CODEX_MODEL", raising=False)


class TestModelNameMapping:
    def test_opus_47_maps_correctly(self):
        assert _resolve_model("opus-4.7") == "claude-opus-4-7"

    def test_opus_46_legacy_alias_maps_to_current_opus(self):
        # Legacy alias kept for back-compat with older configs.
        assert _resolve_model("opus-4.6") == "claude-opus-4-7"

    def test_sonnet_46_maps_correctly(self):
        assert _resolve_model("sonnet-4.6") == "claude-sonnet-4-6"

    def test_haiku_45_maps_correctly(self):
        assert _resolve_model("haiku-4.5") == "claude-haiku-4-5-20251001"

    def test_unknown_model_passes_through(self):
        assert _resolve_model("custom-model-123") == "custom-model-123"

    def test_all_cli_models_have_mappings(self):
        expected_shorts = ["opus-4.7", "opus-4.6", "sonnet-4.6", "sonnet-4.5", "haiku-4.5", "haiku-3.5"]
        for short in expected_shorts:
            assert short in _CLI_MODEL_MAP, f"Missing CLI mapping: {short}"

    def test_default_model_constant(self):
        assert DEFAULT_MODEL in _CLI_MODEL_MAP

    def test_empty_model_falls_back_to_default(self, monkeypatch):
        monkeypatch.delenv("SOCMATE_MODEL", raising=False)
        assert _resolve_model("") == _CLI_MODEL_MAP[DEFAULT_MODEL]

    def test_socmate_model_env_overrides_passed_value(self, monkeypatch):
        monkeypatch.setenv("SOCMATE_MODEL", "haiku-4.5")
        assert _resolve_model("opus-4.7") == "claude-haiku-4-5-20251001"

    def test_socmate_model_env_with_full_id_passes_through(self, monkeypatch):
        monkeypatch.setenv("SOCMATE_MODEL", "claude-some-future-model-99")
        assert _resolve_model("opus-4.7") == "claude-some-future-model-99"

    def test_empty_socmate_model_does_not_override(self, monkeypatch):
        monkeypatch.setenv("SOCMATE_MODEL", "")
        assert _resolve_model("opus-4.7") == "claude-opus-4-7"

    def test_codex_opus_maps_to_gpt_55(self):
        assert _resolve_model("opus-4.7", "codex_cli") == "gpt-5.5"

    def test_codex_default_model_constant(self):
        assert DEFAULT_CODEX_MODEL == "gpt-5.5"
        assert _CODEX_MODEL_MAP["opus-4.7"] == DEFAULT_CODEX_MODEL

    def test_socmate_codex_model_env_overrides_passed_value(self, monkeypatch):
        monkeypatch.setenv("SOCMATE_CODEX_MODEL", "gpt-5.5")
        assert _resolve_model("sonnet-4.6", "codex_cli") == "gpt-5.5"


class TestProviderDetection:
    def test_defaults_to_claude_cli(self, monkeypatch):
        monkeypatch.delenv("SOCMATE_LLM_PROVIDER", raising=False)
        assert _detect_provider() == "claude_cli"

    def test_codex_provider_env(self, monkeypatch):
        monkeypatch.setenv("SOCMATE_LLM_PROVIDER", "codex")
        assert _detect_provider() == "codex_cli"


class TestCodexJsonParsing:
    def test_parse_codex_json_agent_message_and_usage(self):
        stdout = (
            '{"type":"thread.started","thread_id":"t"}\n'
            '{"type":"item.completed","item":{"type":"agent_message","text":"hello"}}\n'
            '{"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":2}}\n'
        )
        text, usage = _parse_codex_json(stdout)
        assert text == "hello"
        assert usage == {"input_tokens": 10, "output_tokens": 2}


class TestLLMTelemetry:
    def test_log_llm_call_backdates_otel_span(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SOCMATE_PROJECT_ROOT", str(tmp_path))
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
            "orchestrator.langchain.agents.socmate_llm._get_llm_tracer",
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

    @patch("orchestrator.langchain.agents.socmate_llm._find_claude_binary")
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

    @patch("orchestrator.langchain.agents.socmate_llm._find_claude_binary")
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

    @patch("orchestrator.langchain.agents.socmate_llm._find_codex_binary")
    def test_codex_exec_flags_and_gpt_55_model(self, mock_find, monkeypatch):
        monkeypatch.setenv("SOCMATE_LLM_PROVIDER", "codex")
        monkeypatch.delenv("SOCMATE_CODEX_MODEL", raising=False)
        monkeypatch.delenv("SOCMATE_MODEL", raising=False)
        mock_find.return_value = "/usr/bin/codex"

        model = ClaudeLLM(model="opus-4.7", timeout=10)

        with patch.object(model, "_run_cli_with_watchdog") as mock_watchdog:
            mock_watchdog.return_value = ("test output", "", 0, 1.0, False, False, {})
            model._generate_via_cli("system prompt", "hello")

            cmd = mock_watchdog.call_args[0][0]
            assert cmd[:2] == ["/usr/bin/codex", "exec"]
            assert "--json" in cmd
            assert "-m" in cmd
            assert cmd[cmd.index("-m") + 1] == "gpt-5.5"
            assert "--dangerously-bypass-approvals-and-sandbox" in cmd


class TestWatchdogBehaviour:
    """Test stall detection and timeout in _run_cli_with_watchdog."""

    @patch("orchestrator.langchain.agents.socmate_llm._find_claude_binary")
    def test_timeout_returns_partial_output(self, mock_find):
        """When the hard timeout fires, partial output should be captured."""
        mock_find.return_value = "/usr/bin/echo"

        model = ClaudeLLM(model="opus-4.6", timeout=3)

        result = model._generate_via_cli("system prompt", "hello")
        assert isinstance(result, str)

    @patch("orchestrator.langchain.agents.socmate_llm._find_claude_binary")
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
