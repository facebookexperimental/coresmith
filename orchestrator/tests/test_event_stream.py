# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for the pipeline event stream (JSONL logging).

Tests:
- write_graph_event creates JSONL file with valid records
- Events have required fields (ts, iso, event, node)
- read_events reads back with optional timestamp filtering
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from orchestrator.langgraph.event_stream import (
    write_graph_event,
    read_events,
    aggregate_failure_categories,
)


@pytest.fixture
def event_project(tmp_path):
    """Temporary project root for event stream tests."""
    coresmith_dir = tmp_path / ".coresmith"
    coresmith_dir.mkdir()
    return str(tmp_path)


class TestWriteGraphEvent:
    def test_creates_jsonl_file(self, event_project):
        write_graph_event(event_project, "Test Node", "graph_node_enter", {"key": "value"})

        log_path = Path(event_project) / ".coresmith" / "pipeline_events.jsonl"
        assert log_path.exists()

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 1

        record = json.loads(lines[0])
        assert record["event"] == "graph_node_enter"
        assert record["node"] == "Test Node"
        assert record["key"] == "value"

    def test_event_has_required_fields(self, event_project):
        write_graph_event(event_project, "Block Diagram", "graph_node_exit", {"round": 1})

        log_path = Path(event_project) / ".coresmith" / "pipeline_events.jsonl"
        record = json.loads(log_path.read_text().strip())

        assert "ts" in record
        assert "iso" in record
        assert "event" in record
        assert "node" in record
        assert record["status"] == "ok"
        assert isinstance(record["ts"], float)

    def test_graph_node_exit_derives_uniform_status(self, event_project):
        write_graph_event(event_project, "Pass Node", "graph_node_exit", {"passed": True})
        write_graph_event(event_project, "Fail Node", "graph_node_exit", {"passed": False})
        write_graph_event(event_project, "Skip Node", "graph_node_exit", {"skipped": True})
        write_graph_event(event_project, "Abort Node", "graph_node_exit", {"pipeline_aborted": True})

        events = read_events(event_project)
        assert [event["status"] for event in events] == [
            "ok",
            "error",
            "skipped",
            "interrupted",
        ]

    def test_multiple_events_append(self, event_project):
        write_graph_event(event_project, "A", "graph_node_enter")
        write_graph_event(event_project, "B", "graph_node_exit")
        write_graph_event(event_project, "C", "graph_node_enter")

        log_path = Path(event_project) / ".coresmith" / "pipeline_events.jsonl"
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 3

        nodes = [json.loads(line)["node"] for line in lines]
        assert nodes == ["A", "B", "C"]

    def test_events_without_data(self, event_project):
        write_graph_event(event_project, "Node", "graph_node_enter")

        log_path = Path(event_project) / ".coresmith" / "pipeline_events.jsonl"
        record = json.loads(log_path.read_text().strip())
        assert record["node"] == "Node"


class TestReadEvents:
    def test_reads_all_events(self, event_project):
        write_graph_event(event_project, "A", "graph_node_enter")
        write_graph_event(event_project, "B", "graph_node_exit")

        events = read_events(event_project)
        assert len(events) == 2
        assert events[0]["node"] == "A"
        assert events[1]["node"] == "B"

    def test_reads_empty_file(self, event_project):
        events = read_events(event_project)
        assert events == []

    def test_timestamp_filtering(self, event_project):
        write_graph_event(event_project, "old", "graph_node_enter")
        cutoff = time.time()
        time.sleep(0.05)
        write_graph_event(event_project, "new", "graph_node_exit")

        events = read_events(event_project, after_ts=cutoff)
        assert len(events) == 1
        assert events[0]["node"] == "new"


# ---------------------------------------------------------------------------
# Failure aggregation helper
# ---------------------------------------------------------------------------

class TestAggregateFailureCategories:
    """Tests for the shared aggregate_failure_categories() helper.

    This helper scans graph_node_exit events for failure categories and
    produces a summary used by both the observer and get_pipeline_state().
    """

    def test_empty_events_returns_zeros(self):
        result = aggregate_failure_categories([])
        assert result["total_failures"] == 0
        assert result["failure_categories"] == {}
        assert result["per_block_failures"] == {}
        assert result["systematic_patterns"] == []
        assert result["avg_retries"] == 0.0

    def test_single_block_failure(self):
        events = [
            {"event": "graph_node_exit", "block": "scrambler", "category": "LOGIC_ERROR"},
        ]
        result = aggregate_failure_categories(events)
        assert result["total_failures"] == 1
        assert result["failure_categories"] == {"LOGIC_ERROR": 1}
        assert result["per_block_failures"] == {"scrambler": ["LOGIC_ERROR"]}
        assert result["systematic_patterns"] == []

    def test_multiple_categories_across_blocks(self):
        events = [
            {"event": "graph_node_exit", "block": "scrambler", "category": "LOGIC_ERROR"},
            {"event": "graph_node_exit", "block": "encoder", "category": "INTERFACE_MISMATCH"},
            {"event": "graph_node_exit", "block": "encoder", "category": "LOGIC_ERROR"},
        ]
        result = aggregate_failure_categories(events)
        assert result["total_failures"] == 3
        assert result["failure_categories"]["LOGIC_ERROR"] == 2
        assert result["failure_categories"]["INTERFACE_MISMATCH"] == 1
        assert result["per_block_failures"]["scrambler"] == ["LOGIC_ERROR"]
        assert result["per_block_failures"]["encoder"] == ["INTERFACE_MISMATCH", "LOGIC_ERROR"]

    def test_systematic_pattern_three_blocks(self):
        """A category affecting 3+ distinct blocks is a systematic pattern."""
        events = [
            {"event": "graph_node_exit", "block": "a", "category": "LOGIC_ERROR"},
            {"event": "graph_node_exit", "block": "b", "category": "LOGIC_ERROR"},
            {"event": "graph_node_exit", "block": "c", "category": "LOGIC_ERROR"},
        ]
        result = aggregate_failure_categories(events)
        assert "LOGIC_ERROR" in result["systematic_patterns"]

    def test_two_blocks_not_systematic(self):
        """2 blocks is below the threshold for systematic patterns."""
        events = [
            {"event": "graph_node_exit", "block": "a", "category": "LOGIC_ERROR"},
            {"event": "graph_node_exit", "block": "b", "category": "LOGIC_ERROR"},
        ]
        result = aggregate_failure_categories(events)
        assert result["systematic_patterns"] == []

    def test_same_block_multiple_times_not_systematic(self):
        """Same block failing 3 times with same category is NOT systematic."""
        events = [
            {"event": "graph_node_exit", "block": "a", "category": "LOGIC_ERROR"},
            {"event": "graph_node_exit", "block": "a", "category": "LOGIC_ERROR"},
            {"event": "graph_node_exit", "block": "a", "category": "LOGIC_ERROR"},
        ]
        result = aggregate_failure_categories(events)
        assert result["systematic_patterns"] == []
        assert result["total_failures"] == 3

    def test_avg_retries_from_block_starts(self):
        events = [
            {"event": "block_start", "block": "a", "attempt": 3},
            {"event": "block_start", "block": "b", "attempt": 1},
        ]
        result = aggregate_failure_categories(events)
        assert result["avg_retries"] == 2.0

    def test_block_start_updates_to_latest_attempt(self):
        """Multiple block_start events for same block: last attempt wins."""
        events = [
            {"event": "block_start", "block": "a", "attempt": 1},
            {"event": "block_start", "block": "a", "attempt": 3},
        ]
        result = aggregate_failure_categories(events)
        assert result["avg_retries"] == 3.0

    def test_ignores_non_failure_events(self):
        """Events without a category field are not counted as failures."""
        events = [
            {"event": "graph_node_enter", "block": "scrambler"},
            {"event": "graph_node_exit", "block": "scrambler", "clean": True},
            {"event": "graph_node_exit", "block": "scrambler", "passed": True},
        ]
        result = aggregate_failure_categories(events)
        assert result["total_failures"] == 0

    def test_ignores_empty_category(self):
        """Events with category='' are not counted."""
        events = [
            {"event": "graph_node_exit", "block": "scrambler", "category": ""},
        ]
        result = aggregate_failure_categories(events)
        assert result["total_failures"] == 0

    def test_ignores_events_without_block(self):
        """Events with no block field are skipped."""
        events = [
            {"event": "graph_node_exit", "category": "LOGIC_ERROR"},
        ]
        result = aggregate_failure_categories(events)
        assert result["total_failures"] == 0
