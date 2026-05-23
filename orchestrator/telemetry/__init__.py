# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Lightweight OpenTelemetry bootstrap for the coresmith ASIC pipeline.

Call ``init_telemetry(project_root)`` once per process before any graph
code runs.  Spans are exported to ``.coresmith/traces.db`` via the
:class:`SqliteSpanExporter`.
"""

from __future__ import annotations

import os
import threading

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from .exporter import SqliteSpanExporter

_init_lock = threading.Lock()
_initialized = False


def init_telemetry(project_root: str, service_name: str = "coresmith-asic") -> None:
    """Initialise the global OTel TracerProvider with a SQLite exporter.

    Safe to call multiple times -- subsequent calls are no-ops.

    Args:
        project_root: Project root directory.  The trace database is
            created at ``<project_root>/.coresmith/traces.db``.
        service_name: OTel service name resource attribute.
    """
    global _initialized
    if _initialized:
        return

    with _init_lock:
        if _initialized:
            return

        db_dir = os.path.join(project_root, ".coresmith")
        os.makedirs(db_dir, exist_ok=True)
        db_path = os.path.join(db_dir, "traces.db")

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        exporter = SqliteSpanExporter(db_path)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        _initialized = True
