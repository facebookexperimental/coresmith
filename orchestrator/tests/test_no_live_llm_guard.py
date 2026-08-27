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

from orchestrator.tests.conftest import _PROVIDER_CLI_ENV, live_llm_calls_allowed


class TestGuardIsOnByDefault:
    """Default branch: CORESMITH_ALLOW_LIVE_LLM_IN_TESTS unset."""

    @pytest.mark.parametrize("var", _PROVIDER_CLI_ENV)
    def test_every_provider_cli_points_at_the_stub(self, var):
        path = os.environ.get(var, "")
        assert path, f"{var} should be pointed at the stub"
        assert os.path.isfile(path) and os.access(path, os.X_OK), (
            f"{var}={path!r} must resolve to an executable so import-time "
            "ClaudeLLM() construction still succeeds"
        )

    def test_invoking_the_stub_fails_fast(self):
        """The stub must fail, not silently succeed and look like a real reply."""
        proc = subprocess.run(
            [os.environ["CLAUDE_CLI_PATH"], "-p", "hello"],
            capture_output=True, text=True, timeout=30,
        )
        assert proc.returncode != 0
        assert "live_llm" in proc.stderr

    def test_helper_reports_guard_active(self):
        assert live_llm_calls_allowed() is False


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
    stub_marker = "no-live-llm"
    assert stub_marker not in os.environ.get("CLAUDE_CLI_PATH", "")
