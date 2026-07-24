# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
SQLite-backed OpenTelemetry SpanExporter.

Writes spans to a local SQLite database in WAL mode, which supports
concurrent writes from multiple processes safely.  Each span's
``node_name`` is extracted from its display name (stripping block-name
brackets and attempt/iteration suffixes) so the webview can query
traces by graph-node identity.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from collections.abc import Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

# Matches the bracketed or parenthesized block name and any trailing suffix.
# Architecture graph nodes use parentheses (e.g. "Gather Requirements (PRD)"),
# pipeline nodes use brackets (e.g. "Generate RTL [scrambler]").
# e.g. "Generate RTL [Scrambler] - Retry #2" -> "Generate RTL"
#      "Gather Requirements (PRD)"           -> "Gather Requirements"
#      "Block Diagram - Iteration #3"        -> "Block Diagram"
#      "Constraint Check"                    -> "Constraint Check"
_NODE_NAME_RE = re.compile(r"^(.+?)(?:\s*[\[(].*|$)")
_SUFFIX_RE = re.compile(r"\s*-\s*(?:Retry|Iteration)\s*#\d+.*$")
_ATTEMPT_SUFFIX_RE = re.compile(r"\s+attempt\s+\d+$", re.IGNORECASE)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS spans (
    span_id     TEXT PRIMARY KEY,
    trace_id    TEXT NOT NULL,
    parent_id   TEXT,
    node_name   TEXT,
    name        TEXT NOT NULL,
    status_code INTEGER DEFAULT 0,
    status_msg  TEXT,
    start_ns    INTEGER NOT NULL,
    end_ns      INTEGER,
    attributes  TEXT,
    events      TEXT,
    resource    TEXT
);
CREATE INDEX IF NOT EXISTS idx_spans_node_name ON spans(node_name);
CREATE INDEX IF NOT EXISTS idx_spans_start_ns  ON spans(start_ns);
CREATE INDEX IF NOT EXISTS idx_spans_trace_id  ON spans(trace_id);
CREATE INDEX IF NOT EXISTS idx_spans_parent_id ON spans(parent_id);
"""


def extract_node_name(span_name: str) -> str:
    """Extract the canonical graph-node name from an OTel span name.

    Examples::

        >>> extract_node_name("Generate RTL [Scrambler] - Retry #2")
        'Generate RTL'
        >>> extract_node_name("Gather Requirements (PRD)")
        'Gather Requirements'
        >>> extract_node_name("Block Diagram - Iteration #3")
        'Block Diagram'
        >>> extract_node_name("Constraint Check")
        'Constraint Check'
        >>> extract_node_name("Simulate [scrambler]")
        'Simulate'
        >>> extract_node_name("LLM claude-opus-4-6 (claude_cli)")
        'LLM claude-opus-4-6'
    """
    name = span_name.strip()
    # Strip " [BlockName]..." first
    m = _NODE_NAME_RE.match(name)
    if m:
        name = m.group(1).strip()
    # Strip " - Retry #N" / " - Iteration #N" suffixes
    name = _SUFFIX_RE.sub("", name)
    # Strip " attempt N" suffix (pipeline graph style)
    name = _ATTEMPT_SUFFIX_RE.sub("", name)
    return name.strip()


def _span_to_row(span: ReadableSpan) -> tuple:
    """Convert a ReadableSpan to a SQLite row tuple."""
    ctx = span.get_span_context()
    parent = span.parent
    parent_id = format(parent.span_id, "032x") if parent else None

    attrs = dict(span.attributes) if span.attributes else {}
    events = [
        {
            "name": e.name,
            "timestamp_ns": e.timestamp,
            "attributes": dict(e.attributes) if e.attributes else {},
        }
        for e in (span.events or [])
    ]
    resource = dict(span.resource.attributes) if span.resource else {}

    return (
        format(ctx.span_id, "032x"),
        format(ctx.trace_id, "032x"),
        parent_id,
        extract_node_name(span.name),
        span.name,
        span.status.status_code.value if span.status else 0,
        span.status.description if span.status else None,
        span.start_time,
        span.end_time,
        json.dumps(attrs, default=str),
        json.dumps(events, default=str),
        json.dumps(resource, default=str),
    )


class SqliteSpanExporter(SpanExporter):
    """Export OTel spans to a local SQLite database.

    Uses WAL journal mode and a busy timeout of 5 seconds so multiple
    processes can write concurrently without blocking each other.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._ensure_schema()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                self._db_path,
                isolation_level="DEFERRED",
                check_same_thread=False,
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
        return self._conn

    def _ensure_schema(self) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.executescript(_SCHEMA)
            conn.commit()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if not spans:
            return SpanExportResult.SUCCESS
        rows = [_span_to_row(s) for s in spans]
        with self._lock:
            try:
                conn = self._get_conn()
                conn.executemany(
                    "INSERT OR REPLACE INTO spans "
                    "(span_id, trace_id, parent_id, node_name, name, "
                    " status_code, status_msg, start_ns, end_ns, "
                    " attributes, events, resource) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )
                conn.commit()
                return SpanExportResult.SUCCESS
            except Exception:
                return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        with self._lock:
            if self._conn is not None:
                self._conn.commit()
        return True
