# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Measurements a testbench records for bounded requirement items.

A PERF/TIME item carries a metric and bounds (``--min`` / ``--max``); the
cocotb verdict alone never decides it. The testbench measures the quantity and
records it::

    from orchestrator.harness.measure import record
    record("PERF-003", 107759, unit="fps", test="test_throughput_b2b")

which appends one JSON line ``{"item", "value", "unit", "test", "module",
"ts"}`` to the file named by ``$CORESMITH_MEASUREMENTS`` or, when that is
unset, to ``./measurements.jsonl`` in the simulation's working directory. The
engine's block simulations set ``$CORESMITH_MEASUREMENTS`` to the canonical
``<root>/sim_build/<block>/measurements.jsonl`` (fresh for every run) that a
module build and ``coresmith block-done`` read: the item's bounds decide
pass/fail there. ``module`` is the cocotb test module that recorded the value
(the nearest ``test_*`` module on the call stack): a module build only counts
measurements recorded by the acceptance testbench's own bound test.

Stdlib only and cocotb-free, so a testbench (or any script) can import it.
"""
from __future__ import annotations

import inspect
import json
import os
import time
from pathlib import Path

ENV = "CORESMITH_MEASUREMENTS"
FILENAME = "measurements.jsonl"


def measurements_path() -> Path:
    """Where :func:`record` appends (``$CORESMITH_MEASUREMENTS`` or ``./measurements.jsonl``)."""
    return Path(os.environ.get(ENV) or FILENAME)


def _recording_module() -> str:
    """The cocotb test module on the call stack (the nearest ``test_*``
    module), else the immediate caller's module."""
    frame = inspect.currentframe()
    caller = ""
    try:
        f = frame.f_back.f_back if frame and frame.f_back else None
        while f is not None:
            name = str(f.f_globals.get("__name__") or "")
            if not caller:
                caller = name
            if name.split(".")[-1].startswith("test_"):
                return name.split(".")[-1]
            f = f.f_back
    finally:
        del frame
    return caller


def record(item: str, value, unit: str = "", test: str = "") -> dict:
    """Append one measurement of ``item`` (e.g. ``PERF-003``) by the cocotb
    test ``test``; returns the row."""
    row = {"item": str(item).strip(), "value": float(value), "unit": str(unit or ""), "test": str(test or ""),
           "module": _recording_module(), "ts": time.time()}
    p = measurements_path()
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    return row


def read(path) -> list[dict]:
    """Every well-formed measurement row in ``path`` in file order (missing file -> [])."""
    out = []
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            row = json.loads(ln)
            if not isinstance(row, dict) or not row.get("item"):
                continue
            row["value"] = float(row["value"])
        except (ValueError, TypeError, KeyError):
            continue
        out.append(row)
    return out
