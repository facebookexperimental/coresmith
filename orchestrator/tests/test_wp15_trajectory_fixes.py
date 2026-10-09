"""WP-15: fixes derived from the Arm F / E2 trajectory audits."""
from __future__ import annotations

import json

from orchestrator.langchain.agents.coresmith_llm import _parse_codex_json


def test_codex_error_event_becomes_llm_error_marker():
    out = "\n".join([
        json.dumps({"type": "thread.started", "thread_id": "t1"}),
        json.dumps({"type": "turn.started"}),
        json.dumps({"type": "error", "message": "You've hit your usage limit. Visit ..."}),
    ])
    text, usage = _parse_codex_json(out)
    assert text.startswith("[ClaudeLLM error: codex CLI reported: You've hit your usage limit")
    assert usage["session_id"] == "t1"


def test_codex_turn_failed_event_becomes_llm_error_marker():
    out = json.dumps({"type": "turn.failed", "error": {"message": "You've hit your usage limit."}})
    text, _ = _parse_codex_json(out)
    assert text.startswith("[ClaudeLLM error: codex CLI reported: You've hit your usage limit")


def test_codex_agent_message_wins_over_error_event():
    """A transport notice FOLLOWED by real output is recoverable: the text is
    the result and the notice is kept as diagnostics."""
    out = "\n".join([
        json.dumps({"type": "error", "message": "transient"}),
        json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "done"}}),
    ])
    text, usage = _parse_codex_json(out)
    assert text == "done" and usage["provider_notices"] == ["transient"] and "provider_error" not in usage


def test_codex_interim_error_then_completed_turn_is_a_success():
    out = "\n".join([
        json.dumps({"type": "thread.started", "thread_id": "t9"}),
        json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "first"}}),
        json.dumps({"type": "error", "message": "stream disconnected; retrying"}),
        json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "final"}}),
        json.dumps({"type": "turn.completed", "usage": {"input_tokens": 5, "output_tokens": 2}}),
    ])
    text, usage = _parse_codex_json(out)
    assert text == "final" and usage["input_tokens"] == 5 and usage["session_id"] == "t9"
    assert usage["provider_notices"] == ["stream disconnected; retrying"] and "provider_error" not in usage


def test_codex_turn_failed_after_partial_text_is_a_failure_with_the_partial_kept():
    """``turn.failed`` is terminal (exit 0): the partial agent text is not a
    result -- it follows the envelope so is_llm_error_response() matches."""
    from orchestrator.langchain.agents.coresmith_llm import is_llm_error_response
    out = "\n".join([
        json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "half of the file"}}),
        json.dumps({"type": "turn.failed", "error": {"message": "stream disconnected before completion"}}),
    ])
    text, usage = _parse_codex_json(out)
    assert is_llm_error_response(text) and "stream disconnected before completion" in text
    assert text.endswith("\nhalf of the file")
    assert usage["provider_error"] == "stream disconnected before completion"


def test_codex_trailing_error_event_is_terminal():
    out = "\n".join([
        json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "partial"}}),
        json.dumps({"type": "error", "message": "You've hit your usage limit."}),
    ])
    text, usage = _parse_codex_json(out)
    assert text.startswith("[ClaudeLLM error: codex CLI reported: You've hit your usage limit.]")
    assert text.endswith("\npartial") and usage["provider_error"].startswith("You've hit")


def test_claude_result_is_error_is_not_a_success():
    from orchestrator.langchain.agents.coresmith_llm import _parse_stream_json
    out = "\n".join([
        json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "partial prose"}]}}),
        json.dumps({"type": "result", "subtype": "error_during_execution", "is_error": True,
                    "result": "API Error: 529 overloaded", "usage": {"input_tokens": 3}, "num_turns": 2}),
    ])
    text, usage = _parse_stream_json(out)
    assert text == "partial prose"                       # the assistant text, not the diagnostic
    assert usage["result_error"] == "API Error: 529 overloaded" and usage["result_subtype"] == "error_during_execution"
    assert usage["input_tokens"] == 3 and usage["num_turns"] == 2
    ok = "\n".join([
        json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": "fine", "usage": {}}),
    ])
    text, usage = _parse_stream_json(ok)
    assert text == "fine" and "result_error" not in usage


def test_infra_markers_include_usage_limit():
    import inspect

    from orchestrator.langgraph import pipeline_graph as pg
    src = inspect.getsource(pg.diagnose_node)
    assert "usage limit" in src and "[ClaudeLLM error:" in src


def test_squeeze_is_gone():
    from orchestrator.langgraph import pipeline_graph as pg
    assert not hasattr(pg, "_maybe_squeeze_throughput")


def test_stage_project_inputs_symlinks_rom_images(tmp_path):
    from orchestrator.langgraph.integration_helpers import _stage_project_inputs
    (tmp_path / "inputs" / "rom_images").mkdir(parents=True)
    (tmp_path / "inputs" / "rom_images" / "q.memh").write_text("00\n")
    sim = tmp_path / "sim_build" / "integration"
    sim.mkdir(parents=True)
    _stage_project_inputs(sim, tmp_path)
    assert (sim / "inputs" / "rom_images" / "q.memh").read_text() == "00\n"
    _stage_project_inputs(sim, tmp_path)  # idempotent
    assert (sim / "inputs").is_symlink()


def test_stage_project_inputs_without_inputs_dir_is_noop(tmp_path):
    from orchestrator.langgraph.integration_helpers import _stage_project_inputs
    sim = tmp_path / "sim_build" / "integration"
    sim.mkdir(parents=True)
    _stage_project_inputs(sim, tmp_path)
    assert not (sim / "inputs").exists()


def test_acceptance_stimulus_falls_back_to_model_stimulus(tmp_path, monkeypatch):
    from orchestrator.architecture.reference_oracle import _acceptance_stimulus_path
    monkeypatch.delenv("CORESMITH_ACCEPTANCE_STIMULUS", raising=False)
    monkeypatch.delenv("CORESMITH_MODEL_STIMULUS", raising=False)
    assert _acceptance_stimulus_path(str(tmp_path)) == ""
    (tmp_path / "inputs").mkdir()
    ms = tmp_path / "inputs" / "model_stimulus.py"
    ms.write_text("stimulus = b'x'\n")
    assert _acceptance_stimulus_path(str(tmp_path)) == str(ms)
    acc = tmp_path / "inputs" / "acceptance_stimulus.py"
    acc.write_text("stimulus = b'y'\n")
    assert _acceptance_stimulus_path(str(tmp_path)) == str(acc)
