# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for fix_synth_errors verify-the-fix-landed guardrail.

The fixer used to return True whenever the LLM call didn't raise -- even when a
failed apply_patch left the RTL untouched, which then burned another ~600s on a
guaranteed-identical synth timeout (the codec RD-core did this repeatedly).
"""
from __future__ import annotations

import asyncio

import pytest

import orchestrator.langgraph.pipeline_helpers as ph

_FLAT = "module b; reg [1023:0] x_q; wire w = x_q[i +: 8]; endmodule\n"


def _run_fix(rtl_path):
    return asyncio.run(ph.fix_synth_errors("b", str(rtl_path), str(rtl_path) + ".log"))


def test_returns_none_when_file_unchanged(tmp_path, monkeypatch):
    rtl = tmp_path / "b.v"
    rtl.write_text(_FLAT)
    import orchestrator.langchain.agents.coresmith_llm as cl

    class _NoChange:
        def __init__(self, *a, **k):
            pass

        async def call(self, **k):  # LLM "succeeds" but touches nothing
            return ""

    monkeypatch.setattr(cl, "ClaudeLLM", _NoChange)
    assert _run_fix(rtl) is None        # <- the guardrail
    assert rtl.read_text() == _FLAT     # untouched


def test_returns_true_when_file_changed(tmp_path, monkeypatch):
    rtl = tmp_path / "b.v"
    rtl.write_text(_FLAT)
    import orchestrator.langchain.agents.coresmith_llm as cl

    class _DoChange:
        def __init__(self, *a, **k):
            pass

        async def call(self, **k):
            rtl.write_text("module b; reg [7:0] mem [0:127]; endmodule\n")

    monkeypatch.setattr(cl, "ClaudeLLM", _DoChange)
    assert _run_fix(rtl) is True


def test_verify_disabled_by_env(tmp_path, monkeypatch):
    rtl = tmp_path / "b.v"
    rtl.write_text(_FLAT)
    monkeypatch.setenv("CORESMITH_VERIFY_FIX_LANDED", "0")
    import orchestrator.langchain.agents.coresmith_llm as cl

    class _NoChange:
        def __init__(self, *a, **k):
            pass

        async def call(self, **k):
            return ""

    monkeypatch.setattr(cl, "ClaudeLLM", _NoChange)
    assert _run_fix(rtl) is True        # legacy behavior restorable
