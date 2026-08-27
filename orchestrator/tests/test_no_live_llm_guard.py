# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The suite must never shell out to a real provider CLI unless it says so.

``_detect_provider()`` defaults to ``claude_cli``, so any node reaching a real
``ClaudeLLM.call()`` launches the actual Claude CLI. Graph tests carrying no
``live_llm`` marker used to do that in PR CI, which is what made the job take
hours instead of minutes. ``conftest._no_live_llm_cli`` closes that door; these
tests keep it closed.
"""

from __future__ import annotations

import os
import subprocess

import pytest

from orchestrator.langchain.agents import coresmith_llm
from orchestrator.tests.conftest import (
    _PROVIDER_BINARY_FINDERS,
    live_llm_calls_allowed,
)

_STUB_NAME = "no-live-llm"


class TestGuardIsOnByDefault:
    """Default branch: CORESMITH_ALLOW_LIVE_LLM_IN_TESTS unset."""

    @pytest.mark.parametrize("finder", _PROVIDER_BINARY_FINDERS)
    def test_every_resolver_returns_the_stub(self, finder):
        path = getattr(coresmith_llm, finder)()
        assert _STUB_NAME in path, f"{finder} should resolve to the stub"
        assert os.path.isfile(path) and os.access(path, os.X_OK), (
            f"{finder} must resolve to an executable so import-time "
            "ClaudeLLM() construction still succeeds"
        )

    def test_invoking_the_stub_fails_fast(self):
        """The stub must fail, not silently succeed and look like a real reply."""
        proc = subprocess.run(
            [coresmith_llm._find_claude_binary(), "-p", "hello"],
            capture_output=True, text=True, timeout=30,
        )
        assert proc.returncode != 0
        assert "live_llm" in proc.stderr

    def test_a_default_llm_resolves_to_the_stub(self):
        """The guard has to survive the real construction path, not just the resolver."""
        llm = coresmith_llm.ClaudeLLM()
        assert _STUB_NAME in llm.claude_path

    def test_helper_reports_guard_active(self):
        assert live_llm_calls_allowed() is False


class TestGuardYieldsToTheTest:
    """The fixture must not outrank a test that pins its own binary."""

    def test_env_var_still_wins(self, monkeypatch):
        monkeypatch.setenv("CLAUDE_CLI_PATH", "/usr/bin/claude")
        assert coresmith_llm.ClaudeLLM().claude_path == "/usr/bin/claude"

    def test_a_test_level_resolver_patch_still_wins(self, monkeypatch):
        monkeypatch.setattr(
            coresmith_llm, "_find_claude_binary", lambda: "/usr/bin/claude")
        assert coresmith_llm.ClaudeLLM().claude_path == "/usr/bin/claude"


class TestEscapeHatch:
    """Gated branch: CORESMITH_ALLOW_LIVE_LLM_IN_TESTS=1 restores old behavior."""

    def test_helper_reports_guard_disabled(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_ALLOW_LIVE_LLM_IN_TESTS", "1")
        assert live_llm_calls_allowed() is True

    def test_only_exactly_one_enables_it(self, monkeypatch):
        for value in ("", "0", "true", "yes"):
            monkeypatch.setenv("CORESMITH_ALLOW_LIVE_LLM_IN_TESTS", value)
            assert live_llm_calls_allowed() is False


@pytest.mark.live_llm
def test_live_llm_marked_tests_keep_the_real_cli():
    """A test that opts in is left alone -- the marker is the opt-in."""
    assert _STUB_NAME not in coresmith_llm._find_claude_binary()
