# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Standardized EDA stubs (Package C).

The existing suite patches ``pipeline_graph.{lint_rtl,run_simulation,
synthesize_block}`` ad hoc in every test. ``stub_eda`` centralizes that so the
fault-injection property tests exercise graph ROUTING without Verilator/Yosys,
which would be slow (and a fork-storm risk on a 4-core box).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# Default passing results (shapes mirror the real helpers).
LINT_CLEAN = {"clean": True, "warnings": ""}
LINT_FAIL = {"clean": False, "errors": "syntax error"}
SIM_PASS = {"passed": True, "log": "all tests passed", "returncode": 0}
SIM_FAIL = {"passed": False, "log": "FAIL", "returncode": 1}
SYNTH_OK = {
    "success": True, "gate_count": 1500, "netlist_path": "/tmp/net.v",
    "sdc_path": "/tmp/f.sdc", "log": "",
}
# Equivalence "honest skip" -> non-blocking (never false-passes).
EQUIV_SKIP = {"passed": False, "skipped": True, "reason": "stubbed", "checked_vectors": 0}
EQUIV_PASS = {"passed": True, "skipped": False, "reason": "stubbed pass", "checked_vectors": 64}


def _as_callable(value: Any, default: dict) -> Callable[..., dict]:
    if value is None:
        value = default
    if callable(value):
        return value

    def _fn(*_a, **_kw):
        return value

    return _fn


def stub_eda(
    monkeypatch,
    *,
    lint: Any = None,
    sim: Any = None,
    synth: Any = None,
    equiv: Any = None,
) -> None:
    """Patch the EDA helpers on the graph module(s).

    Each of ``lint/sim/synth/equiv`` may be a dict (fixed return) or a callable
    (``(*args, **kwargs) -> dict``). Unset -> passing default (equiv -> skip).
    """
    from orchestrator.langgraph import pipeline_graph as pg
    from orchestrator.langgraph import rtl_model_equiv as rme

    monkeypatch.setattr(pg, "lint_rtl", _as_callable(lint, LINT_CLEAN))
    monkeypatch.setattr(pg, "run_simulation", _as_callable(sim, SIM_PASS))
    monkeypatch.setattr(pg, "synthesize_block", _as_callable(synth, SYNTH_OK))
    # Equiv is imported lazily inside the node from rtl_model_equiv, so patch the
    # source module (patching the pipeline_graph namespace would miss it).
    monkeypatch.setattr(
        rme, "check_rtl_model_equivalence", _as_callable(equiv, EQUIV_SKIP)
    )
