# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Fault taxonomy tests (Package C, C3).

Each FaultClass produces the documented behavior, exercised where possible
through the REAL guard (generate_rtl's postcondition), else at the backend.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from orchestrator.testing import fault_provider as fp
from orchestrator.testing.faults import (
    PROVIDER_EXCEPTION_MESSAGE,
    FaultClass,
    FaultSchedule,
    FaultSpec,
)

pytestmark = pytest.mark.fault_injection


@pytest.fixture
def backend(monkeypatch, tmp_path):
    monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))
    monkeypatch.delenv("CORESMITH_LLM_LOG_ROOT", raising=False)
    b = fp.get_backend()
    b.reset()
    yield b
    b.reset()


# ---------------------------------------------------------------------------
# FaultSpec / FaultSchedule matching
# ---------------------------------------------------------------------------
class TestSchedule:
    def test_glob_match(self):
        spec = FaultSpec(FaultClass.EMPTY_RESPONSE, run_name_glob="generate_rtl*")
        assert spec.matches(run_name="generate_rtl:adder", glob_index=1, fired=0)
        assert not spec.matches(run_name="generate_tb", glob_index=1, fired=0)

    def test_call_index(self):
        spec = FaultSpec(FaultClass.EMPTY_RESPONSE, call_index=2)
        assert not spec.matches(run_name="x", glob_index=1, fired=0)
        assert spec.matches(run_name="x", glob_index=2, fired=0)

    def test_every_n(self):
        spec = FaultSpec(FaultClass.EMPTY_RESPONSE, every_n=3)
        assert not spec.matches(run_name="x", glob_index=1, fired=0)
        assert spec.matches(run_name="x", glob_index=3, fired=0)
        assert spec.matches(run_name="x", glob_index=6, fired=0)

    def test_max_faults(self):
        spec = FaultSpec(FaultClass.EMPTY_RESPONSE, max_faults=1)
        assert spec.matches(run_name="x", glob_index=1, fired=0)
        assert not spec.matches(run_name="x", glob_index=2, fired=1)

    def test_json_roundtrip(self):
        sched = FaultSchedule([
            FaultSpec(FaultClass.NO_FILE_WRITTEN, run_name_glob="generate_rtl*", call_index=1),
            FaultSpec(FaultClass.TIMEOUT_STRING, every_n=2),
        ])
        blob = sched.to_json()
        back = FaultSchedule.from_json(blob)
        assert [s.fault for s in back.specs] == [FaultClass.NO_FILE_WRITTEN, FaultClass.TIMEOUT_STRING]
        assert back.specs[0].call_index == 1
        assert back.specs[1].every_n == 2

    def test_from_json_accepts_specs_key(self):
        blob = '{"specs": [{"fault": "empty_response"}]}'
        sched = FaultSchedule.from_json(blob)
        assert sched.specs[0].fault is FaultClass.EMPTY_RESPONSE

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_FAULT_SCHEDULE", '[{"fault":"stall_string"}]')
        sched = FaultSchedule.from_env()
        assert sched.specs[0].fault is FaultClass.STALL_STRING

    def test_empty_json_is_empty(self):
        assert FaultSchedule.from_json("").specs == []


# ---------------------------------------------------------------------------
# Content-returning fault behaviors (via backend.generate directly)
# ---------------------------------------------------------------------------
class TestContentFaults:
    @pytest.mark.parametrize("fc,check", [
        (FaultClass.EMPTY_RESPONSE, lambda s: s == ""),
        (FaultClass.TIMEOUT_STRING, lambda s: s.startswith("[ClaudeLLM error") and "timed out" in s),
        (FaultClass.STALL_STRING, lambda s: s.startswith("[ClaudeLLM error") and "stalled" in s),
        (FaultClass.ERROR_TEXT_AS_CONTENT, lambda s: "[ClaudeLLM error" not in s and len(s) > 0),
        (FaultClass.TRUNCATED_JSON, lambda s: s.startswith("{") and not s.rstrip().endswith("}")),
        (FaultClass.FABRICATED_PASS, lambda s: '"passed": true' in s or '"passed":true' in s),
        (FaultClass.NO_FILE_WRITTEN, lambda s: len(s) > 0),
    ])
    def test_content_fault(self, backend, fc, check):
        backend.set_schedule(FaultSchedule.single(fc))
        out = backend.generate(None, "sys", "write rtl/x.v please", None)
        assert check(out), f"{fc.value} returned {out!r}"

    def test_provider_exception_raises(self, backend):
        backend.set_schedule(FaultSchedule.single(FaultClass.PROVIDER_EXCEPTION))
        with pytest.raises(RuntimeError) as ei:
            backend.generate(None, "s", "p", None)
        assert "sandbox" in str(ei.value)
        assert PROVIDER_EXCEPTION_MESSAGE in str(ei.value)


# ---------------------------------------------------------------------------
# Disk-side fault behaviors
# ---------------------------------------------------------------------------
class TestDiskFaults:
    _PROMPT = "Write the module to rtl/adder8.v as the block adder8."

    def test_success_writes_valid_rtl(self, backend, tmp_path):
        backend.set_schedule(FaultSchedule([]))  # no fault
        out = backend.generate(None, "sys", self._PROMPT, None)
        rtl = tmp_path / "rtl" / "adder8.v"
        assert rtl.exists()
        assert "module adder8" in rtl.read_text()
        assert rtl.stat().st_size >= 200
        assert "Done" in out

    def test_no_file_written_writes_nothing(self, backend, tmp_path):
        backend.set_schedule(FaultSchedule.single(FaultClass.NO_FILE_WRITTEN))
        backend.generate(None, "sys", self._PROMPT, None)
        assert not (tmp_path / "rtl" / "adder8.v").exists()

    def test_wrong_path_written_goes_to_scratch(self, backend, tmp_path):
        backend.set_schedule(FaultSchedule.single(FaultClass.WRONG_PATH_WRITTEN))
        backend.generate(None, "sys", self._PROMPT, None)
        assert not (tmp_path / "rtl" / "adder8.v").exists()
        scratch = tmp_path / "codex-call-scratch" / "adder8.v"
        assert scratch.exists()

    def test_json_disk_mismatch_writes_empty_target(self, backend, tmp_path):
        backend.set_schedule(FaultSchedule.single(FaultClass.JSON_DISK_MISMATCH))
        out = backend.generate(None, "sys", self._PROMPT, None)
        rtl = tmp_path / "rtl" / "adder8.v"
        assert rtl.exists() and rtl.read_text() == ""  # 0-char module
        assert '"status": "complete"' in out


# ---------------------------------------------------------------------------
# Real-guard integration: generate_rtl converts a faulted call into an error
# ---------------------------------------------------------------------------
class TestRealGuardIntegration:
    @pytest.fixture
    def rtl_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CORESMITH_LLM_PROVIDER", "fault")
        monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))
        monkeypatch.delenv("CORESMITH_LLM_LOG_ROOT", raising=False)
        import orchestrator.langgraph.pipeline_helpers as ph
        monkeypatch.setattr(ph, "PROJECT_ROOT", Path(tmp_path))
        b = fp.get_backend()
        b.reset()
        yield tmp_path
        b.reset()

    _BLOCK = {
        "name": "adder8",
        "rtl_target": "rtl/adder8.v",
        "python_source": "golden/adder8.py",
        "description": "8-bit adder",
    }

    def test_success_path_generate_rtl(self, rtl_env):
        from orchestrator.langgraph.pipeline_helpers import generate_rtl
        res = asyncio.run(generate_rtl(dict(self._BLOCK), attempt=1))
        assert not res.get("error")
        assert (Path(rtl_env) / "rtl" / "adder8.v").exists()

    def test_no_file_written_guard_fires(self, rtl_env):
        from orchestrator.langgraph.pipeline_helpers import generate_rtl
        fp.get_backend().set_schedule(FaultSchedule.single(FaultClass.NO_FILE_WRITTEN))
        res = asyncio.run(generate_rtl(dict(self._BLOCK), attempt=1))
        assert res.get("error")
        assert "did not write" in res["error"]

    def test_json_disk_mismatch_guard_fires(self, rtl_env):
        from orchestrator.langgraph.pipeline_helpers import generate_rtl
        fp.get_backend().set_schedule(FaultSchedule.single(FaultClass.JSON_DISK_MISMATCH))
        res = asyncio.run(generate_rtl(dict(self._BLOCK), attempt=1))
        # 0-char on-disk module -> postcondition (size < 200) fires.
        assert res.get("error")
