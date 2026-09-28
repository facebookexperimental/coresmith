# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Failure receipts belong to the failed call, even across PID reuse."""
import json
from pathlib import Path

from orchestrator.langchain.agents.coresmith_llm import _worker_failure_evidence
from orchestrator.langgraph.pipeline_graph import _worker_receipts_from_failure


def test_claude_turn_limit_returns_once_with_usage(tmp_path, monkeypatch):
    from orchestrator.langchain.agents import coresmith_llm as llm
    monkeypatch.setenv('CLAUDE_CLI_PATH', '/fixture/claude')
    monkeypatch.setenv('CORESMITH_LLM_PROVIDER', 'claude_cli')
    monkeypatch.setattr(llm, '_llm_log_root', lambda: str(tmp_path))
    calls, logs = [], []
    def run(*args, **kwargs):
        calls.append(args)
        return ('partial work', '', 0, 12, False, False,
                {'result_subtype': 'error_max_turns', 'output_tokens': 500})
    monkeypatch.setattr(llm.ClaudeLLM, '_run_cli_with_watchdog', run)
    monkeypatch.setattr(llm, '_log_llm_call', lambda **kwargs: logs.append(kwargs))
    worker = llm.ClaudeLLM(max_turns=1)
    result = worker._generate_via_claude_cli('', 'write RTL')
    assert len(calls) == len(logs) == 1
    assert 'generation budget exhausted' in result
    assert logs[0]['usage']['output_tokens'] == 500


def test_receipts_do_not_overwrite_on_pid_reuse_or_inventory_sibling_files(tmp_path):
    sibling = tmp_path / "sim_build" / "other_block" / "dump.vcd"
    sibling.parent.mkdir(parents=True)
    sibling.write_text("unrelated waveform")
    command = 'make -C sim_build/failed_block TRACE_FILE=partial.vcd'
    first = _worker_failure_evidence(str(tmp_path), 123, command, "last cycle 37", "timeout", "call-a")
    second = _worker_failure_evidence(str(tmp_path), 123, "other command", "", "timeout", "call-b")
    assert first != second
    data = json.loads(Path(first).read_text())
    assert data["call_id"] == "call-a"
    assert data["stderr_tail"] == "last cycle 37"
    assert Path(data["trajectory_path"]).read_text() == command
    assert str(sibling) not in Path(first).read_text()
    assert "other command" not in Path(first).read_text()
    # The next audit gets only the receipt in its own failure, irrespective of
    # which sibling finished last. An ordinary sim failure gets no worker file.
    assert _worker_receipts_from_failure(str(tmp_path), f"Error: timeout\nWorker failure evidence: {first}\npartial answer") == [first]
    assert _worker_receipts_from_failure(str(tmp_path), "sim failed") == []
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    assert _worker_receipts_from_failure(str(tmp_path), f"Worker failure evidence: {outside}") == []
