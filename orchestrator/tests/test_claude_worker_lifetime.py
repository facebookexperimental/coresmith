# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Claude workers are bounded calls, not sessions with later wakeups."""
import json
import os
import sys
import time
from unittest.mock import patch

import pytest

from orchestrator.langchain.agents.coresmith_llm import ClaudeLLM


@pytest.mark.parametrize("provider", ["claude_cli", "codex_cli"])
def test_background_policy_is_private_and_claude_only(tmp_path, monkeypatch, provider):
    monkeypatch.setenv("CORESMITH_LLM_PROVIDER", "claude")
    keys = ("CLAUDE_CODE_DISABLE_BACKGROUND_TASKS", "CLAUDE_CODE_DISABLE_CRON")
    for key in keys:
        monkeypatch.setenv(key, "0")
    record = tmp_path / "child-env.json"
    stub = tmp_path / "stub.py"
    stub.write_text(
        "import json, os\n"
        f"open({str(record)!r}, 'w').write(json.dumps({{k:os.environ[k] for k in {keys!r}}}))\n"
        "print(json.dumps({'type':'result','result':'DONE'}), flush=True)\n")
    model = ClaudeLLM(model="opus-5.5", timeout=10, claude_path="/bin/true")
    model._provider = provider
    model._POLL_INTERVAL_S = 0.1
    model._run_cli_with_watchdog([sys.executable, str(stub)], "", str(tmp_path),
                                 "test", time.monotonic())
    assert json.loads(record.read_text()) == dict.fromkeys(keys, "1" if provider == "claude_cli" else "0")
    assert all(os.environ[key] == "0" for key in keys)


@pytest.mark.parametrize("disable_tools", [False, True])
def test_scheduled_tools_are_unavailable(tmp_path, monkeypatch, disable_tools):
    monkeypatch.setenv("CORESMITH_LLM_PROVIDER", "claude")
    monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))
    model = ClaudeLLM(model="opus-5.5", timeout=10, claude_path="/bin/true",
                     disable_tools=disable_tools)
    with patch.object(model, "_run_cli_with_watchdog",
                      return_value=("DONE", "", 0, .1, False, False, {})) as launch:
        model._generate_via_claude_cli("system", "prompt")
    cmd = launch.call_args.args[0]
    assert cmd.count("--disallowedTools") == 1
    denied = set(cmd[cmd.index("--disallowedTools") + 1].split(","))
    assert {"Monitor", "ScheduleWakeup"} <= denied
    assert ("Bash" in denied) is disable_tools
