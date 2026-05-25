# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Comprehensive tests for the webview server and its integration with
the webview frontend.

Tests are organized into three focus areas:
  1. Timeline event recording accuracy
  2. Live LLM status streaming
  3. LLM log display when timeline segments are clicked

Each area tests both the server-side Python functions (unit) and the
HTTP API endpoints (integration), using fixture-based temporary project
directories with synthetic event data.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import sys
import time
from http.server import HTTPServer
from pathlib import Path
from threading import Thread

import pytest


# ---------------------------------------------------------------------------
# Helpers — import serve.py from the vscode-ext directory (hyphenated name
# prevents normal Python package import).
# ---------------------------------------------------------------------------

_SERVE_PATH = (
    Path(__file__).resolve().parents[1] / "vscode-ext" / "serve.py"
)

# The vscode-ext/serve.py webview server has been removed from the orchestrator
# (replaced by a separate frontend stack). All tests in this file exercise that
# server, so skip the entire module when serve.py is absent rather than letting
# every test ERROR at fixture setup. Re-enable if/when the webview server returns.
if not _SERVE_PATH.exists():
    pytest.skip(
        "orchestrator/vscode-ext/serve.py removed; webview server tests obsolete",
        allow_module_level=True,
    )


def _import_serve(project_root: Path):
    """Import serve.py via importlib from its file path, with PROJECT_ROOT
    pointed at the given directory."""
    os.environ["CORESMITH_PROJECT_ROOT"] = str(project_root)
    spec = importlib.util.spec_from_file_location("serve", str(_SERVE_PATH))
    mod = importlib.util.module_from_spec(spec)
    # Prevent re-registration in sys.modules causing cross-test pollution
    sys.modules.pop("serve", None)
    spec.loader.exec_module(mod)
    mod.PROJECT_ROOT = project_root
    return mod


@pytest.fixture(autouse=True)
def _isolate_project_root(tmp_path, monkeypatch):
    """Ensure every test gets its own project root and .coresmith directory."""
    coresmith_dir = tmp_path / ".coresmith"
    coresmith_dir.mkdir()
    monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))


@pytest.fixture
def serve_module(tmp_path, monkeypatch):
    """Import serve.py with PROJECT_ROOT pointed at the temp directory."""
    monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))
    spec = importlib.util.spec_from_file_location(
        "serve_under_test", str(_SERVE_PATH)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Override PROJECT_ROOT to the temp directory
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)
    return mod


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def write_events(tmp_path: Path, events: list[dict]):
    """Write a list of event dicts to pipeline_events.jsonl."""
    log_path = tmp_path / ".coresmith" / "pipeline_events.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")


def write_llm_calls(tmp_path: Path, calls: list[dict]):
    """Write LLM call records to llm_calls.jsonl."""
    log_path = tmp_path / ".coresmith" / "llm_calls.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        for call in calls:
            f.write(json.dumps(call) + "\n")


def write_live_stream(tmp_path: Path, pid: int, data: dict):
    """Write a live stream JSON file for a subprocess."""
    live_dir = tmp_path / ".coresmith" / "live_streams"
    live_dir.mkdir(parents=True, exist_ok=True)
    stream_file = live_dir / f"{pid}.json"
    stream_file.write_text(json.dumps(data))


def create_traces_db(tmp_path: Path, spans: list[dict]):
    """Create a minimal traces.db SQLite database with spans.

    Column names match the reader.py query: status_msg (not status_message).
    """
    db_path = tmp_path / ".coresmith" / "traces.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS spans (
            span_id TEXT PRIMARY KEY,
            trace_id TEXT,
            parent_id TEXT,
            node_name TEXT,
            name TEXT,
            status_code INTEGER DEFAULT 0,
            status_msg TEXT DEFAULT '',
            start_ns INTEGER,
            end_ns INTEGER,
            attributes TEXT DEFAULT '{}',
            events TEXT DEFAULT '[]',
            resource TEXT DEFAULT '{}'
        )
    """)
    for span in spans:
        conn.execute("""
            INSERT INTO spans (span_id, trace_id, parent_id, node_name,
                               name, status_code, status_msg,
                               start_ns, end_ns, attributes, events, resource)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            span.get("span_id", ""),
            span.get("trace_id", "trace-1"),
            span.get("parent_id"),
            span.get("node_name", ""),
            span.get("name", ""),
            span.get("status_code", 1),
            span.get("status_message", ""),
            span.get("start_ns", 0),
            span.get("end_ns", 0),
            json.dumps(span.get("attributes", {})),
            json.dumps(span.get("events", [])),
            json.dumps(span.get("resource", {})),
        ))
    conn.commit()
    conn.close()


# ═══════════════════════════════════════════════════════════════════════════
# AREA 1: Timeline Event Recording Accuracy
# ═══════════════════════════════════════════════════════════════════════════

class TestTimelineEventAccuracy:
    """Verify that timeline events from pipeline_events.jsonl are parsed
    into accurate Gantt segments with correct timing, status, and grouping."""

    # --- 1.1 Basic enter/exit pairing ---

    def test_single_node_enter_exit_creates_segment(self, tmp_path, serve_module):
        """A matched enter/exit pair produces one segment with correct timing."""
        t0 = 1700000000.0
        events = [
            {"ts": t0, "iso": "2024-01-01T00:00:00", "event": "graph_node_enter",
             "node": "Generate RTL", "block": "scrambler", "graph": "frontend"},
            {"ts": t0 + 10.0, "iso": "2024-01-01T00:00:10", "event": "graph_node_exit",
             "node": "Generate RTL", "block": "scrambler", "graph": "frontend",
             "clean": True, "chars": 500},
        ]
        write_events(tmp_path, events)

        result = serve_module.get_timeline_data("frontend")

        assert len(result["blocks"]) == 1
        block = result["blocks"][0]
        assert block["name"] == "scrambler"
        assert block["graph"] == "frontend"
        assert len(block["attempts"]) == 1
        assert len(block["attempts"][0]["segments"]) == 1

        seg = block["attempts"][0]["segments"][0]
        assert seg["node"] == "Generate RTL"
        assert seg["start_ts"] == t0
        assert seg["end_ts"] == t0 + 10.0
        assert seg["duration_s"] == 10.0
        assert seg["status"] == "done"
        assert seg["clean"] is True

    # --- 1.2 Multiple nodes in sequence ---

    def test_sequential_nodes_create_separate_segments(self, tmp_path, serve_module):
        """Multiple enter/exit pairs for the same block create separate segments."""
        t0 = 1700000000.0
        events = [
            {"ts": t0, "event": "graph_node_enter", "node": "Generate RTL",
             "block": "scrambler", "graph": "frontend"},
            {"ts": t0 + 5, "event": "graph_node_exit", "node": "Generate RTL",
             "block": "scrambler", "graph": "frontend", "clean": True},
            {"ts": t0 + 5, "event": "graph_node_enter", "node": "Lint Check",
             "block": "scrambler", "graph": "frontend"},
            {"ts": t0 + 8, "event": "graph_node_exit", "node": "Lint Check",
             "block": "scrambler", "graph": "frontend", "clean": True},
        ]
        write_events(tmp_path, events)

        result = serve_module.get_timeline_data("frontend")
        segs = result["blocks"][0]["attempts"][0]["segments"]
        assert len(segs) == 2
        assert segs[0]["node"] == "Generate RTL"
        assert segs[1]["node"] == "Lint Check"
        assert segs[0]["duration_s"] == 5.0
        assert segs[1]["duration_s"] == 3.0

    # --- 1.3 Failed node status ---

    def test_failed_node_has_failed_status(self, tmp_path, serve_module):
        """When exit event has clean=False, segment status should be 'failed'."""
        t0 = 1700000000.0
        events = [
            {"ts": t0, "event": "graph_node_enter", "node": "Lint Check",
             "block": "scrambler", "graph": "frontend"},
            {"ts": t0 + 3, "event": "graph_node_exit", "node": "Lint Check",
             "block": "scrambler", "graph": "frontend", "clean": False,
             "violations": 5},
        ]
        write_events(tmp_path, events)

        result = serve_module.get_timeline_data("frontend")
        seg = result["blocks"][0]["attempts"][0]["segments"][0]
        assert seg["status"] == "failed"
        assert seg["clean"] is False
        assert seg["violations"] == 5

    # --- 1.4 Running (open) segments ---

    def test_open_segment_marked_as_running(self, tmp_path, serve_module):
        """An enter without a matching exit should appear as a running segment."""
        t0 = 1700000000.0
        events = [
            {"ts": t0, "event": "graph_node_enter", "node": "Generate RTL",
             "block": "scrambler", "graph": "frontend"},
        ]
        write_events(tmp_path, events)

        result = serve_module.get_timeline_data("frontend")
        seg = result["blocks"][0]["attempts"][0]["segments"][0]
        assert seg["status"] == "running"
        assert seg["end_ts"] is None
        assert seg["duration_s"] is None

    # --- 1.5 HITL nodes marked as waiting ---

    def test_hitl_node_marked_as_waiting(self, tmp_path, serve_module):
        """Nodes like 'Review Uarch Spec' should be 'waiting', not 'running'."""
        t0 = 1700000000.0
        events = [
            {"ts": t0, "event": "graph_node_enter", "node": "Review Uarch Spec",
             "block": "scrambler", "graph": "frontend"},
        ]
        write_events(tmp_path, events)

        result = serve_module.get_timeline_data("frontend")
        seg = result["blocks"][0]["attempts"][0]["segments"][0]
        assert seg["status"] == "waiting"

    # --- 1.6 Multi-attempt grouping ---

    def test_multiple_attempts_grouped_correctly(self, tmp_path, serve_module):
        """Events with different attempt numbers are grouped into separate attempts."""
        t0 = 1700000000.0
        events = [
            {"ts": t0, "event": "block_start", "block": "scrambler",
             "graph": "frontend", "attempt": 1},
            {"ts": t0, "event": "graph_node_enter", "node": "Generate RTL",
             "block": "scrambler", "graph": "frontend", "attempt": 1},
            {"ts": t0 + 5, "event": "graph_node_exit", "node": "Generate RTL",
             "block": "scrambler", "graph": "frontend", "attempt": 1, "clean": False},
            {"ts": t0 + 6, "event": "block_start", "block": "scrambler",
             "graph": "frontend", "attempt": 2},
            {"ts": t0 + 6, "event": "graph_node_enter", "node": "Generate RTL",
             "block": "scrambler", "graph": "frontend", "attempt": 2},
            {"ts": t0 + 12, "event": "graph_node_exit", "node": "Generate RTL",
             "block": "scrambler", "graph": "frontend", "attempt": 2, "clean": True},
        ]
        write_events(tmp_path, events)

        result = serve_module.get_timeline_data("frontend")
        block = result["blocks"][0]
        assert len(block["attempts"]) == 2
        assert block["attempts"][0]["attempt"] == 1
        assert block["attempts"][1]["attempt"] == 2

    # --- 1.7 Multi-block isolation ---

    def test_multiple_blocks_appear_as_separate_rows(self, tmp_path, serve_module):
        """Events for different blocks should create separate block entries."""
        t0 = 1700000000.0
        events = [
            {"ts": t0, "event": "graph_node_enter", "node": "Generate RTL",
             "block": "scrambler", "graph": "frontend"},
            {"ts": t0 + 1, "event": "graph_node_enter", "node": "Generate RTL",
             "block": "viterbi_decoder", "graph": "frontend"},
            {"ts": t0 + 5, "event": "graph_node_exit", "node": "Generate RTL",
             "block": "scrambler", "graph": "frontend", "clean": True},
            {"ts": t0 + 8, "event": "graph_node_exit", "node": "Generate RTL",
             "block": "viterbi_decoder", "graph": "frontend", "clean": True},
        ]
        write_events(tmp_path, events)

        result = serve_module.get_timeline_data("frontend")
        block_names = [b["name"] for b in result["blocks"]]
        assert "scrambler" in block_names
        assert "viterbi_decoder" in block_names

    # --- 1.8 Graph filtering ---

    def test_graph_filter_isolates_events(self, tmp_path, serve_module):
        """Filtering by graph name only returns events for that graph."""
        t0 = 1700000000.0
        events = [
            {"ts": t0, "event": "graph_node_enter", "node": "Block Diagram",
             "graph": "architecture"},
            {"ts": t0 + 5, "event": "graph_node_exit", "node": "Block Diagram",
             "graph": "architecture"},
            {"ts": t0 + 10, "event": "graph_node_enter", "node": "Generate RTL",
             "block": "scrambler", "graph": "frontend"},
            {"ts": t0 + 15, "event": "graph_node_exit", "node": "Generate RTL",
             "block": "scrambler", "graph": "frontend", "clean": True},
        ]
        write_events(tmp_path, events)

        arch_result = serve_module.get_timeline_data("architecture")
        assert len(arch_result["blocks"]) == 1
        assert arch_result["blocks"][0]["graph"] == "architecture"

        front_result = serve_module.get_timeline_data("frontend")
        assert len(front_result["blocks"]) == 1
        assert front_result["blocks"][0]["name"] == "scrambler"

    # --- 1.9 Empty events file ---

    def test_empty_events_returns_empty_blocks(self, tmp_path, serve_module):
        """No events file or empty file returns empty block list."""
        result = serve_module.get_timeline_data("")
        assert result["blocks"] == []
        assert result["pipeline_start"] is None

    # --- 1.10 Pipeline start/end timestamps ---

    def test_pipeline_start_end_derived_from_events(self, tmp_path, serve_module):
        """pipeline_start and pipeline_end bracket all events."""
        t0 = 1700000000.0
        events = [
            {"ts": t0, "event": "graph_node_enter", "node": "A", "block": "b1",
             "graph": "frontend"},
            {"ts": t0 + 100, "event": "graph_node_exit", "node": "A", "block": "b1",
             "graph": "frontend"},
        ]
        write_events(tmp_path, events)

        result = serve_module.get_timeline_data("frontend")
        assert result["pipeline_start"] == t0
        assert result["pipeline_end"] == t0 + 100

    # --- 1.11 Exit metadata captured ---

    def test_exit_metadata_captured_in_segment(self, tmp_path, serve_module):
        """Rich metadata from exit events (gate_count, elapsed_s, category)
        should be captured in the segment."""
        t0 = 1700000000.0
        events = [
            {"ts": t0, "event": "graph_node_enter", "node": "Synthesize",
             "block": "scrambler", "graph": "frontend"},
            {"ts": t0 + 30, "event": "graph_node_exit", "node": "Synthesize",
             "block": "scrambler", "graph": "frontend", "success": True,
             "gate_count": 1500, "elapsed_s": 30.0, "category": "LOGIC_ERROR"},
        ]
        write_events(tmp_path, events)

        result = serve_module.get_timeline_data("frontend")
        seg = result["blocks"][0]["attempts"][0]["segments"][0]
        assert seg["gate_count"] == 1500
        assert seg["elapsed_s"] == 30.0
        assert seg["category"] == "LOGIC_ERROR"

    # --- 1.12 Duplicate enter events ignored ---

    def test_duplicate_enter_keeps_original_start_time(self, tmp_path, serve_module):
        """When LangGraph re-executes a node on interrupt resume, a second
        enter event fires. The original start time should be preserved."""
        t0 = 1700000000.0
        events = [
            {"ts": t0, "event": "graph_node_enter", "node": "Review Uarch Spec",
             "block": "scrambler", "graph": "frontend"},
            {"ts": t0 + 30, "event": "graph_node_enter", "node": "Review Uarch Spec",
             "block": "scrambler", "graph": "frontend"},
            {"ts": t0 + 35, "event": "graph_node_exit", "node": "Review Uarch Spec",
             "block": "scrambler", "graph": "frontend"},
        ]
        write_events(tmp_path, events)

        result = serve_module.get_timeline_data("frontend")
        seg = result["blocks"][0]["attempts"][0]["segments"][0]
        assert seg["start_ts"] == t0  # NOT t0 + 30

    # --- 1.13 Completion status ---

    def test_advance_block_marks_block_done(self, tmp_path, serve_module):
        """The 'Advance Block' node exit with success=True marks the block done."""
        t0 = 1700000000.0
        events = [
            {"ts": t0, "event": "graph_node_enter", "node": "Advance Block",
             "block": "scrambler", "graph": "frontend"},
            {"ts": t0 + 1, "event": "graph_node_exit", "node": "Advance Block",
             "block": "scrambler", "graph": "frontend", "success": True},
        ]
        write_events(tmp_path, events)

        result = serve_module.get_timeline_data("frontend")
        assert result["blocks"][0]["status"] == "done"

    # --- 1.14 Active and waiting counts ---

    def test_waiting_for_human_and_active_counts(self, tmp_path, serve_module):
        """Open segments at HITL nodes count as waiting, others as active."""
        # Use a recent timestamp; serve.py treats runs older than 60s as
        # historical and zeroes out the waiting/active counters since the
        # process is no longer live.
        import time as _t
        t0 = _t.time()
        events = [
            {"ts": t0, "event": "graph_node_enter", "node": "Review Uarch Spec",
             "block": "scrambler", "graph": "frontend"},
            {"ts": t0 + 1, "event": "graph_node_enter", "node": "Generate RTL",
             "block": "viterbi", "graph": "frontend"},
        ]
        write_events(tmp_path, events)

        result = serve_module.get_timeline_data("frontend")
        assert result["waiting_for_human"] == 1
        assert result["active_blocks"] == 1

    # --- 1.15 Architecture events use graph as block ---

    def test_architecture_events_use_graph_as_block(self, tmp_path, serve_module):
        """Architecture events have no 'block' field; the graph name is used."""
        t0 = 1700000000.0
        events = [
            {"ts": t0, "event": "graph_node_enter", "node": "Block Diagram",
             "graph": "architecture", "round": 1},
            {"ts": t0 + 10, "event": "graph_node_exit", "node": "Block Diagram",
             "graph": "architecture", "round": 1},
        ]
        write_events(tmp_path, events)

        result = serve_module.get_timeline_data("architecture")
        assert result["blocks"][0]["name"] == "Architecture"

    # --- 1.16 Malformed JSONL lines skipped ---

    def test_malformed_jsonl_lines_skipped(self, tmp_path, serve_module):
        """Corrupt lines in the JSONL should not crash timeline parsing."""
        t0 = 1700000000.0
        log_path = tmp_path / ".coresmith" / "pipeline_events.jsonl"
        with open(log_path, "w") as f:
            f.write("NOT VALID JSON\n")
            f.write(json.dumps({"ts": t0, "event": "graph_node_enter",
                                "node": "A", "block": "b", "graph": "frontend"}) + "\n")
            f.write("{truncated\n")
            f.write(json.dumps({"ts": t0 + 5, "event": "graph_node_exit",
                                "node": "A", "block": "b", "graph": "frontend"}) + "\n")

        result = serve_module.get_timeline_data("frontend")
        assert len(result["blocks"]) == 1
        assert result["blocks"][0]["attempts"][0]["segments"][0]["node"] == "A"

    # --- 1.17 Node metadata captured ---

    def test_node_metadata_keys_captured(self, tmp_path, serve_module):
        """Metadata keys like node_count, edge_count, path are captured."""
        t0 = 1700000000.0
        events = [
            {"ts": t0, "event": "graph_node_enter", "node": "Create Documentation",
             "graph": "architecture"},
            {"ts": t0 + 5, "event": "graph_node_exit", "node": "Create Documentation",
             "graph": "architecture", "node_count": 8, "edge_count": 12,
             "path": "/tmp/docs/block_diagram.json"},
        ]
        write_events(tmp_path, events)

        result = serve_module.get_timeline_data("architecture")
        seg = result["blocks"][0]["attempts"][0]["segments"][0]
        assert seg["metadata"]["node_count"] == 8
        assert seg["metadata"]["edge_count"] == 12
        assert seg["metadata"]["path"] == "/tmp/docs/block_diagram.json"


# ═══════════════════════════════════════════════════════════════════════════
# AREA 2: Live LLM Status Streaming
# ═══════════════════════════════════════════════════════════════════════════

class TestLiveLLMStreaming:
    """Verify that live LLM call data — both completed calls from
    llm_calls.jsonl and in-progress streams from live_streams/ — are
    correctly correlated with node time windows and returned."""

    # --- 2.1 Completed LLM calls matched by timestamp window ---

    def test_completed_llm_call_within_node_window(self, tmp_path, serve_module):
        """An LLM call whose timestamp falls within a node's enter/exit
        window should be returned."""
        t0 = 1700000000.0
        events = [
            {"ts": t0, "event": "graph_node_enter", "node": "Generate RTL",
             "block": "scrambler"},
            {"ts": t0 + 30, "event": "graph_node_exit", "node": "Generate RTL",
             "block": "scrambler"},
        ]
        llm_calls = [
            {"ts": t0 + 5, "model": "claude-sonnet-4-20250514", "duration_s": 8.5,
             "system_prompt": "You are an RTL generator.",
             "user_prompt": "Generate scrambler module.",
             "response": "module scrambler (...);"},
        ]
        write_events(tmp_path, events)
        write_llm_calls(tmp_path, llm_calls)

        result = serve_module.get_live_calls("Generate RTL")
        assert len(result) >= 1
        spans = result[0]["spans"][0]["children"]
        assert len(spans) == 1
        assert spans[0]["name"] == "LLM claude-sonnet-4-20250514"
        assert spans[0]["status"] == "ok"
        assert "RTL generator" in spans[0]["attributes"]["input.value"]
        assert "module scrambler" in spans[0]["attributes"]["output.value"]

    # --- 2.2 LLM call outside node window excluded ---

    def test_llm_call_outside_window_excluded(self, tmp_path, serve_module):
        """An LLM call outside the node's time window should not be included."""
        t0 = 1700000000.0
        events = [
            {"ts": t0, "event": "graph_node_enter", "node": "Generate RTL",
             "block": "scrambler"},
            {"ts": t0 + 10, "event": "graph_node_exit", "node": "Generate RTL",
             "block": "scrambler"},
        ]
        llm_calls = [
            {"ts": t0 + 50, "model": "claude-sonnet-4-20250514", "duration_s": 3.0,
             "system_prompt": "...", "user_prompt": "...", "response": "..."},
        ]
        write_events(tmp_path, events)
        write_llm_calls(tmp_path, llm_calls)

        result = serve_module.get_live_calls("Generate RTL")
        if result:
            children = result[0]["spans"][0].get("children", [])
            assert len(children) == 0

    # --- 2.3 Open (still running) node window ---

    def test_open_node_window_includes_calls(self, tmp_path, serve_module):
        """When a node has entered but not exited, LLM calls after the
        enter timestamp should be included."""
        t0 = 1700000000.0
        events = [
            {"ts": t0, "event": "graph_node_enter", "node": "Generate RTL",
             "block": "scrambler"},
        ]
        llm_calls = [
            {"ts": t0 + 5, "model": "claude-sonnet-4-20250514", "duration_s": 10.0,
             "system_prompt": "sys", "user_prompt": "usr", "response": "resp"},
        ]
        write_events(tmp_path, events)
        write_llm_calls(tmp_path, llm_calls)

        result = serve_module.get_live_calls("Generate RTL")
        assert len(result) >= 1
        children = result[0]["spans"][0]["children"]
        assert len(children) == 1

    # --- 2.4 Streaming live_streams/ file detected ---

    def test_active_streaming_file_detected(self, tmp_path, serve_module):
        """An active (non-done) streaming file within the node's time window
        should appear with status='streaming'."""
        t0 = 1700000000.0
        events = [
            {"ts": t0, "event": "graph_node_enter", "node": "Generate RTL",
             "block": "scrambler"},
        ]
        write_events(tmp_path, events)
        # llm_calls.jsonl must exist (even if empty) for get_live_calls to
        # proceed to the streaming file check
        write_llm_calls(tmp_path, [])
        write_live_stream(tmp_path, pid=12345, data={
            "pid": 12345,
            "model": "claude-sonnet-4-20250514",
            "started_ts": t0 + 2,
            "elapsed_s": 5.0,
            "partial_stdout": "module scrambler(\n  input clk,\n",
            "stdout_bytes": 35,
            "done": False,
        })

        result = serve_module.get_live_calls("Generate RTL")
        assert len(result) >= 1
        children = result[0]["spans"][0]["children"]
        streaming = [c for c in children if c["status"] == "streaming"]
        assert len(streaming) == 1
        assert streaming[0]["attributes"]["streaming"] is True
        assert "module scrambler" in streaming[0]["attributes"]["output.value"]

    # --- 2.5 Done streaming file still visible within 2-minute window ---

    def test_done_streaming_file_within_window(self, tmp_path, serve_module):
        """A done streaming file younger than 120s should still be returned."""
        t0 = 1700000000.0
        now = time.time()
        events = [
            {"ts": t0, "event": "graph_node_enter", "node": "Generate RTL",
             "block": "scrambler"},
        ]
        write_events(tmp_path, events)
        write_llm_calls(tmp_path, [])
        write_live_stream(tmp_path, pid=99999, data={
            "pid": 99999,
            "model": "claude-sonnet-4-20250514",
            "started_ts": t0 + 1,
            "elapsed_s": 15.0,
            "partial_stdout": "complete output",
            "done": True,
            "done_ts": now,  # Just completed
        })

        result = serve_module.get_live_calls("Generate RTL")
        assert len(result) >= 1
        children = result[0]["spans"][0]["children"]
        done_streams = [c for c in children if c["status"] == "ok"
                        and "stream_" in c.get("span_id", "")]
        assert len(done_streams) == 1

    # --- 2.6 Stale done streaming file cleaned up ---

    def test_stale_done_streaming_file_cleaned_up(self, tmp_path, serve_module):
        """A done streaming file older than 120s should be deleted and not returned."""
        t0 = 1700000000.0
        events = [
            {"ts": t0, "event": "graph_node_enter", "node": "Generate RTL",
             "block": "scrambler"},
        ]
        write_events(tmp_path, events)
        write_llm_calls(tmp_path, [])
        old_done_ts = time.time() - 300  # 5 minutes ago
        write_live_stream(tmp_path, pid=11111, data={
            "pid": 11111,
            "model": "claude-sonnet-4-20250514",
            "started_ts": t0 + 1,
            "elapsed_s": 10.0,
            "partial_stdout": "old output",
            "done": True,
            "done_ts": old_done_ts,
        })

        stream_file = tmp_path / ".coresmith" / "live_streams" / "11111.json"
        assert stream_file.exists()

        serve_module.get_live_calls("Generate RTL")

        # File should have been deleted
        assert not stream_file.exists()

    # --- 2.7 Multiple blocks grouped correctly ---

    def test_live_calls_grouped_by_block(self, tmp_path, serve_module):
        """LLM calls from different blocks within the same node should be
        grouped into separate attempt entries."""
        t0 = 1700000000.0
        events = [
            {"ts": t0, "event": "graph_node_enter", "node": "Generate RTL",
             "block": "scrambler", "attempt": 1},
            {"ts": t0 + 20, "event": "graph_node_exit", "node": "Generate RTL",
             "block": "scrambler"},
            {"ts": t0 + 25, "event": "graph_node_enter", "node": "Generate RTL",
             "block": "viterbi", "attempt": 1},
            {"ts": t0 + 50, "event": "graph_node_exit", "node": "Generate RTL",
             "block": "viterbi"},
        ]
        llm_calls = [
            {"ts": t0 + 5, "model": "claude-sonnet-4-20250514", "duration_s": 5.0,
             "system_prompt": "sys", "user_prompt": "scrambler", "response": "r1"},
            {"ts": t0 + 30, "model": "claude-sonnet-4-20250514", "duration_s": 5.0,
             "system_prompt": "sys", "user_prompt": "viterbi", "response": "r2"},
        ]
        write_events(tmp_path, events)
        write_llm_calls(tmp_path, llm_calls)

        result = serve_module.get_live_calls("Generate RTL")
        assert len(result) == 2
        block_names = [r["spans"][0]["attributes"]["block_name"] for r in result]
        assert "scrambler" in block_names
        assert "viterbi" in block_names

    # --- 2.7b Per-attempt grouping for same block ---

    def test_live_calls_grouped_per_attempt(self, tmp_path, serve_module):
        """Multiple attempts of the same block should produce separate
        entries with correct attempt numbers (not merged into one)."""
        t0 = 1700000000.0
        events = [
            {"ts": t0, "event": "graph_node_enter", "node": "Generate RTL",
             "block": "scrambler", "attempt": 1},
            {"ts": t0 + 20, "event": "graph_node_exit", "node": "Generate RTL",
             "block": "scrambler"},
            {"ts": t0 + 25, "event": "graph_node_enter", "node": "Generate RTL",
             "block": "scrambler", "attempt": 2},
            {"ts": t0 + 50, "event": "graph_node_exit", "node": "Generate RTL",
             "block": "scrambler"},
        ]
        llm_calls = [
            {"ts": t0 + 5, "model": "claude-sonnet-4-20250514", "duration_s": 10.0,
             "system_prompt": "sys", "user_prompt": "attempt 1",
             "response": "module v1;"},
            {"ts": t0 + 30, "model": "claude-sonnet-4-20250514", "duration_s": 12.0,
             "system_prompt": "sys", "user_prompt": "attempt 2",
             "response": "module v2;"},
        ]
        write_events(tmp_path, events)
        write_llm_calls(tmp_path, llm_calls)

        result = serve_module.get_live_calls("Generate RTL")
        assert len(result) == 2, f"Expected 2 per-attempt groups, got {len(result)}"
        attempts = [r["attempt"] for r in result]
        assert 1 in attempts
        assert 2 in attempts

        # Each attempt should have exactly 1 LLM call
        for entry in result:
            children = entry["spans"][0]["children"]
            assert len(children) == 1

        # Verify the responses are in the right groups
        for entry in result:
            child = entry["spans"][0]["children"][0]
            if entry["attempt"] == 1:
                assert "module v1" in child["attributes"]["output.value"]
            else:
                assert "module v2" in child["attributes"]["output.value"]

    # --- 2.7c Attempt number preserved from event data ---

    def test_live_calls_attempt_number_from_event(self, tmp_path, serve_module):
        """The attempt number in the result should come from the event's
        attempt field, not be a sequential counter."""
        t0 = 1700000000.0
        events = [
            {"ts": t0, "event": "graph_node_enter", "node": "Generate RTL",
             "block": "scrambler", "attempt": 3},
            {"ts": t0 + 20, "event": "graph_node_exit", "node": "Generate RTL",
             "block": "scrambler"},
        ]
        llm_calls = [
            {"ts": t0 + 5, "model": "claude-sonnet-4-20250514", "duration_s": 8.0,
             "system_prompt": "sys", "user_prompt": "usr", "response": "resp"},
        ]
        write_events(tmp_path, events)
        write_llm_calls(tmp_path, llm_calls)

        result = serve_module.get_live_calls("Generate RTL")
        assert len(result) == 1
        assert result[0]["attempt"] == 3

    # --- 2.7d Architecture nodes with same round get unique attempts ---

    def test_duplicate_round_numbers_get_unique_attempts(self, tmp_path, serve_module):
        """Architecture events that share the same round number should
        get unique attempt numbers to avoid duplicate tab keys."""
        t0 = 1700000000.0
        events = [
            {"ts": t0, "event": "graph_node_enter", "node": "Block Diagram",
             "graph": "architecture", "round": 1},
            {"ts": t0 + 100, "event": "graph_node_exit", "node": "Block Diagram",
             "graph": "architecture"},
            {"ts": t0 + 200, "event": "graph_node_enter", "node": "Block Diagram",
             "graph": "architecture", "round": 1},
            {"ts": t0 + 400, "event": "graph_node_exit", "node": "Block Diagram",
             "graph": "architecture"},
        ]
        llm_calls = [
            {"ts": t0 + 50, "model": "claude-opus-4-6-20250610", "duration_s": 30.0,
             "system_prompt": "sys", "user_prompt": "first", "response": "r1"},
            {"ts": t0 + 300, "model": "claude-opus-4-6-20250610", "duration_s": 40.0,
             "system_prompt": "sys", "user_prompt": "second", "response": "r2"},
        ]
        write_events(tmp_path, events)
        write_llm_calls(tmp_path, llm_calls)

        result = serve_module.get_live_calls("Block Diagram")
        assert len(result) == 2
        attempts = [r["attempt"] for r in result]
        # Attempts must be unique
        assert len(set(attempts)) == 2, f"Duplicate attempts: {attempts}"

    # --- 2.8 has_streaming flag set correctly ---

    def test_has_streaming_flag(self, tmp_path, serve_module):
        """Result entries containing streaming calls should have has_streaming=True."""
        t0 = 1700000000.0
        events = [
            {"ts": t0, "event": "graph_node_enter", "node": "Generate RTL",
             "block": "scrambler"},
        ]
        write_events(tmp_path, events)
        write_llm_calls(tmp_path, [])
        write_live_stream(tmp_path, pid=55555, data={
            "pid": 55555, "model": "claude-sonnet-4-20250514",
            "started_ts": t0 + 1, "elapsed_s": 3.0,
            "partial_stdout": "partial", "done": False,
        })

        result = serve_module.get_live_calls("Generate RTL")
        assert result[0]["has_streaming"] is True
        assert result[0]["live"] is True

    # --- 2.9 No events file returns empty ---

    def test_no_events_returns_empty(self, tmp_path, serve_module):
        """When there is no events file, get_live_calls returns empty list."""
        result = serve_module.get_live_calls("Generate RTL")
        assert result == []

    # --- 2.10 Error LLM calls marked correctly ---

    def test_error_llm_call_status(self, tmp_path, serve_module):
        """LLM calls with error flag should have status='error'."""
        t0 = 1700000000.0
        events = [
            {"ts": t0, "event": "graph_node_enter", "node": "Generate RTL",
             "block": "scrambler"},
            {"ts": t0 + 10, "event": "graph_node_exit", "node": "Generate RTL",
             "block": "scrambler"},
        ]
        llm_calls = [
            {"ts": t0 + 3, "model": "claude-sonnet-4-20250514", "duration_s": 2.0,
             "error": "Rate limit exceeded",
             "system_prompt": "sys", "user_prompt": "usr", "response": ""},
        ]
        write_events(tmp_path, events)
        write_llm_calls(tmp_path, llm_calls)

        result = serve_module.get_live_calls("Generate RTL")
        children = result[0]["spans"][0]["children"]
        assert children[0]["status"] == "error"

    # --- 2.11 Input.value truncation at 32k ---

    def test_input_value_truncated(self, tmp_path, serve_module):
        """System prompt + user prompt are truncated to 32000 chars."""
        t0 = 1700000000.0
        events = [
            {"ts": t0, "event": "graph_node_enter", "node": "Generate RTL",
             "block": "scrambler"},
            {"ts": t0 + 10, "event": "graph_node_exit", "node": "Generate RTL",
             "block": "scrambler"},
        ]
        long_prompt = "x" * 40000
        llm_calls = [
            {"ts": t0 + 2, "model": "claude-sonnet-4-20250514", "duration_s": 5.0,
             "system_prompt": long_prompt, "user_prompt": "short",
             "response": "resp"},
        ]
        write_events(tmp_path, events)
        write_llm_calls(tmp_path, llm_calls)

        result = serve_module.get_live_calls("Generate RTL")
        children = result[0]["spans"][0]["children"]
        assert len(children[0]["attributes"]["input.value"]) <= 32000


# ═══════════════════════════════════════════════════════════════════════════
# AREA 3: LLM Logs Displayed on Timeline Click
# ═══════════════════════════════════════════════════════════════════════════

class TestLLMLogDisplay:
    """Verify that when a timeline segment is clicked, the correct LLM call
    data is fetched and structured for the DetailPanel to render.

    This covers the traces endpoint, live calls endpoint, and the data
    format that the webview's extractLLMCalls function expects."""

    # --- 3.1 OTel traces returned for completed nodes ---

    def test_otel_traces_returned_for_completed_node(self, tmp_path, serve_module):
        """When OTel spans exist for a node, /api/traces/{node} returns them
        in the expected attempt-grouped format."""
        start_ns = 1700000000_000_000_000
        end_ns = start_ns + 10_000_000_000  # 10s later
        spans = [
            {"span_id": "root-1", "trace_id": "t1", "parent_id": None,
             "node_name": "Generate RTL", "name": "Generate RTL [scrambler] attempt 1",
             "status_code": 1, "start_ns": start_ns, "end_ns": end_ns,
             "attributes": {"block_name": "scrambler"}},
            {"span_id": "llm-1", "trace_id": "t1", "parent_id": "root-1",
             "node_name": "Generate RTL", "name": "ChatModel claude-sonnet-4-20250514",
             "status_code": 1, "start_ns": start_ns + 1_000_000_000,
             "end_ns": start_ns + 8_000_000_000,
             "attributes": {
                 "openinference.span.kind": "LLM",
                 "llm.model_name": "claude-sonnet-4-20250514",
                 "input.value": "Generate a scrambler module",
                 "output.value": "module scrambler(...);\nendmodule",
             }},
        ]
        create_traces_db(tmp_path, spans)

        from orchestrator.telemetry.reader import get_node_traces
        db_path = str(tmp_path / ".coresmith" / "traces.db")
        result = get_node_traces(db_path, "Generate RTL")

        assert len(result) >= 1
        # Should have root span with LLM child
        root_spans = result[0]["spans"]
        assert len(root_spans) >= 1
        llm_children = root_spans[0].get("children", [])
        assert len(llm_children) >= 1
        assert llm_children[0]["attributes"]["openinference.span.kind"] == "LLM"

    # --- 3.2 Live calls format matches trace format ---

    def test_live_calls_format_matches_trace_format(self, tmp_path, serve_module):
        """The live calls response should have the same structure as OTel
        traces so the webview can render both uniformly."""
        t0 = 1700000000.0
        events = [
            {"ts": t0, "event": "graph_node_enter", "node": "Generate RTL",
             "block": "scrambler"},
            {"ts": t0 + 20, "event": "graph_node_exit", "node": "Generate RTL",
             "block": "scrambler"},
        ]
        llm_calls = [
            {"ts": t0 + 2, "model": "claude-sonnet-4-20250514", "duration_s": 8.0,
             "system_prompt": "You are RTL gen.", "user_prompt": "Make it.",
             "response": "module x; endmodule"},
        ]
        write_events(tmp_path, events)
        write_llm_calls(tmp_path, llm_calls)

        result = serve_module.get_live_calls("Generate RTL")

        # Verify expected structure
        entry = result[0]
        assert "attempt" in entry
        assert "spans" in entry
        assert "live" in entry

        root_span = entry["spans"][0]
        assert "span_id" in root_span
        assert "name" in root_span
        assert "children" in root_span

        child = root_span["children"][0]
        assert "span_id" in child
        assert "name" in child
        assert "status" in child
        assert "duration_ms" in child
        assert "attributes" in child
        assert "input.value" in child["attributes"]
        assert "output.value" in child["attributes"]
        assert "llm.model_name" in child["attributes"]

    # --- 3.3 System and user prompts concatenated in input.value ---

    def test_input_value_contains_system_and_user_prompt(self, tmp_path, serve_module):
        """input.value should concatenate system_prompt + separator + user_prompt."""
        t0 = 1700000000.0
        events = [
            {"ts": t0, "event": "graph_node_enter", "node": "Generate RTL",
             "block": "scrambler"},
            {"ts": t0 + 10, "event": "graph_node_exit", "node": "Generate RTL",
             "block": "scrambler"},
        ]
        llm_calls = [
            {"ts": t0 + 2, "model": "claude-sonnet-4-20250514", "duration_s": 5.0,
             "system_prompt": "SYSTEM_CONTENT_HERE",
             "user_prompt": "USER_CONTENT_HERE",
             "response": "RESPONSE"},
        ]
        write_events(tmp_path, events)
        write_llm_calls(tmp_path, llm_calls)

        result = serve_module.get_live_calls("Generate RTL")
        child = result[0]["spans"][0]["children"][0]
        input_val = child["attributes"]["input.value"]
        assert "SYSTEM_CONTENT_HERE" in input_val
        assert "USER_CONTENT_HERE" in input_val
        assert "---" in input_val  # separator

    # --- 3.4 Response in output.value ---

    def test_output_value_contains_response(self, tmp_path, serve_module):
        """output.value should contain the LLM response text."""
        t0 = 1700000000.0
        events = [
            {"ts": t0, "event": "graph_node_enter", "node": "Generate RTL",
             "block": "scrambler"},
            {"ts": t0 + 10, "event": "graph_node_exit", "node": "Generate RTL",
             "block": "scrambler"},
        ]
        llm_calls = [
            {"ts": t0 + 2, "model": "claude-sonnet-4-20250514", "duration_s": 5.0,
             "system_prompt": "sys", "user_prompt": "usr",
             "response": "module scrambler(input clk); endmodule"},
        ]
        write_events(tmp_path, events)
        write_llm_calls(tmp_path, llm_calls)

        result = serve_module.get_live_calls("Generate RTL")
        child = result[0]["spans"][0]["children"][0]
        assert "module scrambler" in child["attributes"]["output.value"]

    # --- 3.5 Streaming call output.value has partial stdout ---

    def test_streaming_call_has_partial_output(self, tmp_path, serve_module):
        """Active streaming calls should include partial_stdout in output.value."""
        t0 = 1700000000.0
        events = [
            {"ts": t0, "event": "graph_node_enter", "node": "Generate RTL",
             "block": "scrambler"},
        ]
        write_events(tmp_path, events)
        write_llm_calls(tmp_path, [])
        write_live_stream(tmp_path, pid=77777, data={
            "pid": 77777, "model": "claude-sonnet-4-20250514",
            "started_ts": t0 + 1, "elapsed_s": 4.0,
            "partial_stdout": "module scrambler(\n  input clk,\n  input rst_n,",
            "done": False,
        })

        result = serve_module.get_live_calls("Generate RTL")
        streaming_calls = [c for c in result[0]["spans"][0]["children"]
                           if c["status"] == "streaming"]
        assert len(streaming_calls) == 1
        assert "module scrambler" in streaming_calls[0]["attributes"]["output.value"]

    # --- 3.6 Duration calculated correctly ---

    def test_duration_ms_calculated(self, tmp_path, serve_module):
        """duration_ms should be duration_s * 1000 from the LLM call record."""
        t0 = 1700000000.0
        events = [
            {"ts": t0, "event": "graph_node_enter", "node": "Generate RTL",
             "block": "scrambler"},
            {"ts": t0 + 20, "event": "graph_node_exit", "node": "Generate RTL",
             "block": "scrambler"},
        ]
        llm_calls = [
            {"ts": t0 + 2, "model": "claude-sonnet-4-20250514", "duration_s": 12.345,
             "system_prompt": "sys", "user_prompt": "usr", "response": "resp"},
        ]
        write_events(tmp_path, events)
        write_llm_calls(tmp_path, llm_calls)

        result = serve_module.get_live_calls("Generate RTL")
        child = result[0]["spans"][0]["children"][0]
        assert child["duration_ms"] == 12345.0

    # --- 3.7 block_name attribute set correctly ---

    def test_block_name_attribute_set(self, tmp_path, serve_module):
        """Each LLM call span should carry the correct block_name attribute."""
        t0 = 1700000000.0
        events = [
            {"ts": t0, "event": "graph_node_enter", "node": "Generate RTL",
             "block": "scrambler"},
            {"ts": t0 + 10, "event": "graph_node_exit", "node": "Generate RTL",
             "block": "scrambler"},
        ]
        llm_calls = [
            {"ts": t0 + 2, "model": "claude-sonnet-4-20250514", "duration_s": 5.0,
             "system_prompt": "s", "user_prompt": "u", "response": "r"},
        ]
        write_events(tmp_path, events)
        write_llm_calls(tmp_path, llm_calls)

        result = serve_module.get_live_calls("Generate RTL")
        child = result[0]["spans"][0]["children"][0]
        assert child["attributes"]["block_name"] == "scrambler"

    # --- 3.8 No LLM calls returns empty ---

    def test_no_llm_calls_returns_empty(self, tmp_path, serve_module):
        """Node with events but no LLM calls returns empty list."""
        t0 = 1700000000.0
        events = [
            {"ts": t0, "event": "graph_node_enter", "node": "Lint Check",
             "block": "scrambler"},
            {"ts": t0 + 3, "event": "graph_node_exit", "node": "Lint Check",
             "block": "scrambler"},
        ]
        write_events(tmp_path, events)
        # No llm_calls.jsonl
        result = serve_module.get_live_calls("Lint Check")
        assert result == []

    # --- 3.9 Unknown node returns empty ---

    def test_unknown_node_returns_empty(self, tmp_path, serve_module):
        """Querying for a node that doesn't exist returns empty list."""
        t0 = 1700000000.0
        events = [
            {"ts": t0, "event": "graph_node_enter", "node": "Generate RTL",
             "block": "scrambler"},
        ]
        write_events(tmp_path, events)

        result = serve_module.get_live_calls("Nonexistent Node")
        assert result == []


# ═══════════════════════════════════════════════════════════════════════════
# HTTP API Integration Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestHTTPEndpoints:
    """Integration tests that start the actual HTTP server and verify
    responses from the timeline, live calls, and traces endpoints."""

    @pytest.fixture
    def server(self, tmp_path, serve_module):
        """Start the webview HTTP server on a random port."""
        httpd = HTTPServer(("127.0.0.1", 0), serve_module.WebviewHandler)
        port = httpd.server_address[1]
        thread = Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        # Small delay to let the server fully start
        time.sleep(0.1)
        yield f"http://127.0.0.1:{port}"
        httpd.shutdown()

    def _get_json(self, url: str) -> dict:
        """GET a URL and parse JSON response, bypassing any system proxy."""
        import http.client
        from urllib.parse import urlparse
        parsed = urlparse(url)
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5)
        conn.request("GET", parsed.path + ("?" + parsed.query if parsed.query else ""))
        resp = conn.getresponse()
        body = resp.read().decode()
        if resp.status >= 400:
            from urllib.error import HTTPError
            raise HTTPError(url, resp.status, resp.reason, resp.headers, None)
        return json.loads(body)

    # --- HTTP: Timeline endpoint ---

    def test_timeline_endpoint_returns_json(self, tmp_path, server, serve_module):
        """GET /api/timeline returns valid JSON with expected keys."""
        t0 = 1700000000.0
        write_events(tmp_path, [
            {"ts": t0, "event": "graph_node_enter", "node": "A",
             "block": "b", "graph": "frontend"},
            {"ts": t0 + 5, "event": "graph_node_exit", "node": "A",
             "block": "b", "graph": "frontend"},
        ])

        data = self._get_json(f"{server}/api/timeline")
        assert "blocks" in data
        assert "pipeline_start" in data
        assert "pipeline_end" in data
        assert len(data["blocks"]) == 1

    def test_timeline_endpoint_with_graph_filter(self, tmp_path, server, serve_module):
        """GET /api/timeline?graph=architecture filters correctly."""
        t0 = 1700000000.0
        write_events(tmp_path, [
            {"ts": t0, "event": "graph_node_enter", "node": "Block Diagram",
             "graph": "architecture"},
            {"ts": t0 + 5, "event": "graph_node_exit", "node": "Block Diagram",
             "graph": "architecture"},
            {"ts": t0 + 10, "event": "graph_node_enter", "node": "Generate RTL",
             "block": "scrambler", "graph": "frontend"},
        ])

        data = self._get_json(f"{server}/api/timeline?graph=architecture")
        assert len(data["blocks"]) == 1
        assert data["blocks"][0]["graph"] == "architecture"

    # --- HTTP: Live calls endpoint ---

    def test_live_calls_endpoint(self, tmp_path, server, serve_module):
        """GET /api/live_calls/{node} returns LLM call data."""
        t0 = 1700000000.0
        write_events(tmp_path, [
            {"ts": t0, "event": "graph_node_enter", "node": "Generate RTL",
             "block": "scrambler"},
            {"ts": t0 + 20, "event": "graph_node_exit", "node": "Generate RTL",
             "block": "scrambler"},
        ])
        write_llm_calls(tmp_path, [
            {"ts": t0 + 5, "model": "claude-sonnet-4-20250514", "duration_s": 8.0,
             "system_prompt": "sys", "user_prompt": "usr", "response": "resp"},
        ])

        data = self._get_json(f"{server}/api/live_calls/Generate%20RTL")
        assert len(data) >= 1
        assert data[0]["spans"][0]["children"][0]["name"] == "LLM claude-sonnet-4-20250514"

    def test_live_calls_endpoint_missing_node(self, tmp_path, server, serve_module):
        """GET /api/live_calls/ without a node name returns 400."""
        from urllib.error import HTTPError
        with pytest.raises(HTTPError) as exc_info:
            self._get_json(f"{server}/api/live_calls/")
        assert exc_info.value.code == 400

    # --- HTTP: Traces endpoint ---

    def test_traces_endpoint(self, tmp_path, server, serve_module):
        """GET /api/traces/{node} returns OTel span data."""
        start_ns = 1700000000_000_000_000
        end_ns = start_ns + 5_000_000_000
        create_traces_db(tmp_path, [
            {"span_id": "s1", "trace_id": "t1", "parent_id": None,
             "node_name": "Lint Check", "name": "Lint Check [scrambler]",
             "status_code": 1, "start_ns": start_ns, "end_ns": end_ns,
             "attributes": {}},
        ])

        data = self._get_json(f"{server}/api/traces/Lint%20Check")
        assert isinstance(data, list)
        if data:
            assert "spans" in data[0]

    # --- HTTP: Status endpoint ---

    def test_status_endpoint(self, tmp_path, server, serve_module):
        """GET /api/status returns per-node execution status."""
        t0 = 1700000000.0
        write_events(tmp_path, [
            {"ts": t0, "event": "graph_node_enter", "node": "Generate RTL",
             "block": "scrambler", "graph": "frontend"},
        ])

        data = self._get_json(f"{server}/api/status?graph=frontend")
        assert "Generate RTL" in data
        assert data["Generate RTL"] == "running"

    def test_status_done_after_exit(self, tmp_path, server, serve_module):
        """After exit event, status should be 'done'."""
        t0 = 1700000000.0
        write_events(tmp_path, [
            {"ts": t0, "event": "graph_node_enter", "node": "A",
             "block": "b", "graph": "frontend"},
            {"ts": t0 + 5, "event": "graph_node_exit", "node": "A",
             "block": "b", "graph": "frontend"},
        ])

        data = self._get_json(f"{server}/api/status?graph=frontend")
        assert data["A"] == "done"

    def test_status_failed_on_error(self, tmp_path, server, serve_module):
        """Exit with error flag sets status to 'failed'."""
        t0 = 1700000000.0
        write_events(tmp_path, [
            {"ts": t0, "event": "graph_node_enter", "node": "Lint Check",
             "block": "b", "graph": "frontend"},
            {"ts": t0 + 3, "event": "graph_node_exit", "node": "Lint Check",
             "block": "b", "graph": "frontend", "error": "lint failed"},
        ])

        data = self._get_json(f"{server}/api/status?graph=frontend")
        assert data["Lint Check"] == "failed"

    # --- HTTP: Interrupts endpoint ---

    def test_interrupts_endpoint(self, tmp_path, server, serve_module):
        """GET /api/interrupts returns HITL interrupt data."""
        t0 = 1700000000.0
        write_events(tmp_path, [
            {"ts": t0, "event": "graph_node_enter", "node": "Review Uarch Spec",
             "block": "scrambler", "graph": "frontend"},
        ])

        data = self._get_json(f"{server}/api/interrupts")
        assert "interrupts" in data
        assert "count" in data
        assert data["count"] >= 1
        assert data["interrupts"][0]["node"] == "Review Uarch Spec"


# ═══════════════════════════════════════════════════════════════════════════
# Execution Status Accuracy
# ═══════════════════════════════════════════════════════════════════════════

class TestExecutionStatus:
    """Verify get_execution_status() accurately derives per-node status."""

    def test_enter_sets_running(self, tmp_path, serve_module):
        write_events(tmp_path, [
            {"ts": 1.0, "event": "graph_node_enter", "node": "Generate RTL",
             "block": "s", "graph": "frontend"},
        ])
        status = serve_module.get_execution_status("frontend")
        assert status["Generate RTL"] == "running"

    def test_exit_sets_done(self, tmp_path, serve_module):
        write_events(tmp_path, [
            {"ts": 1.0, "event": "graph_node_enter", "node": "A", "block": "s"},
            {"ts": 2.0, "event": "graph_node_exit", "node": "A", "block": "s"},
        ])
        status = serve_module.get_execution_status()
        assert status["A"] == "done"

    def test_exit_with_error_sets_failed(self, tmp_path, serve_module):
        write_events(tmp_path, [
            {"ts": 1.0, "event": "graph_node_enter", "node": "X", "block": "s"},
            {"ts": 2.0, "event": "graph_node_exit", "node": "X", "block": "s",
             "error": "boom"},
        ])
        status = serve_module.get_execution_status()
        assert status["X"] == "failed"

    def test_graph_filter_excludes_other_graphs(self, tmp_path, serve_module):
        write_events(tmp_path, [
            {"ts": 1.0, "event": "graph_node_enter", "node": "A",
             "graph": "architecture"},
            {"ts": 2.0, "event": "graph_node_enter", "node": "B",
             "graph": "frontend", "block": "s"},
        ])
        status = serve_module.get_execution_status("frontend")
        assert "A" not in status
        assert "B" in status

    def test_empty_file_returns_empty(self, tmp_path, serve_module):
        status = serve_module.get_execution_status()
        assert status == {}


# ═══════════════════════════════════════════════════════════════════════════
# Architecture Escalation Tracking
# ═══════════════════════════════════════════════════════════════════════════

class TestArchitectureEscalations:
    """Verify that architecture escalation events are tracked correctly
    in the interrupts endpoint, which the DetailPanel uses for HITL nodes."""

    def test_ers_escalation_waiting(self, tmp_path, serve_module):
        """An open Escalate PRD node should appear as a waiting interrupt."""
        t0 = 1700000000.0
        write_events(tmp_path, [
            {"ts": t0, "event": "graph_node_enter", "node": "Escalate PRD",
             "graph": "architecture", "round": 1,
             "questions": [{"id": "q1", "question": "Target throughput?",
                            "category": "performance"}],
             "question_count": 1},
        ])

        result = serve_module.get_pending_interrupts()
        esc = [i for i in result["interrupts"]
               if i["type"] == "architecture_escalation"]
        assert len(esc) == 1
        assert esc[0]["status"] == "waiting"
        assert esc[0]["phase"] == "prd"
        assert len(esc[0]["questions"]) == 1

    def test_ers_escalation_completed(self, tmp_path, serve_module):
        """A completed Escalate PRD should have status=completed with response."""
        t0 = 1700000000.0
        write_events(tmp_path, [
            {"ts": t0, "event": "graph_node_enter", "node": "Escalate PRD",
             "graph": "architecture", "round": 1,
             "questions": [{"id": "q1", "question": "Target throughput?"}],
             "question_count": 1},
            {"ts": t0 + 60, "event": "graph_node_exit", "node": "Escalate PRD",
             "graph": "architecture", "action": "continue",
             "has_answers": True, "answer_count": 1,
             "answer_keys": ["q1"]},
        ])

        result = serve_module.get_pending_interrupts()
        esc = [i for i in result["interrupts"]
               if i["type"] == "architecture_escalation"]
        assert len(esc) == 1
        assert esc[0]["status"] == "completed"
        assert esc[0]["response"]["has_answers"] is True
        assert esc[0]["response"]["action"] == "continue"
