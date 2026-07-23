# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""FaultBackend -- a ClaudeLLM backend that injects faults (Package C).

Selected via ``CORESMITH_LLM_PROVIDER=fault``. ``coresmith_llm._generate_via_cli``
calls ``get_backend().generate(llm, system, prompt, resume_session_id)`` for every
LLM call. The backend consults the active :class:`FaultSchedule`, and either:
  - injects the matched fault (returns a failure content string, raises, or writes
    a wrong/absent/empty/stale artifact), OR
  - runs the CannedDesignScript success path (writes the expected artifacts).

Every call is appended to ``call_log`` for property assertions.
"""

from __future__ import annotations

import os
import threading

from orchestrator.testing.faults import (
    FaultClass,
    FaultSchedule,
    FaultSpec,
    PROVIDER_EXCEPTION_MESSAGE,
    STALL_CONTENT,
    TIMEOUT_CONTENT,
)
from orchestrator.testing.success_scripts import CannedDesignScript


def _project_root() -> str:
    return (
        os.environ.get("CORESMITH_PROJECT_ROOT", "").strip()
        or os.environ.get("CORESMITH_LLM_LOG_ROOT", "").strip()
        or os.getcwd()
    )


def _call_site() -> dict:
    """Read the (run_name, call_index) the ClaudeLLM.call set for this call."""
    try:
        from orchestrator.langchain.agents.coresmith_llm import _call_site_context

        return _call_site_context.get(None) or {}
    except Exception:
        return {}


class FaultBackend:
    """In-process LLM backend that injects scheduled faults."""

    def __init__(self, schedule: FaultSchedule | None = None) -> None:
        self._lock = threading.Lock()
        self.schedule = schedule if schedule is not None else FaultSchedule.from_env()
        self.success = CannedDesignScript()
        self.call_log: list[dict] = []
        # per-spec bookkeeping (index in schedule.specs -> counters)
        self._glob_counts: dict[int, int] = {}
        self._fired: dict[int, int] = {}

    # -- lifecycle -------------------------------------------------------
    def set_schedule(self, schedule: FaultSchedule) -> None:
        with self._lock:
            self.schedule = schedule
            self._glob_counts.clear()
            self._fired.clear()
            self.call_log.clear()

    def reset(self) -> None:
        self.set_schedule(FaultSchedule([]))

    # -- matching --------------------------------------------------------
    def _select_fault(self, run_name: str) -> FaultClass | None:
        """First matching spec wins. Updates per-spec glob/fired counters."""
        import fnmatch

        chosen: FaultClass | None = None
        for i, spec in enumerate(self.schedule.specs):
            if fnmatch.fnmatch(run_name or "", spec.run_name_glob):
                self._glob_counts[i] = self._glob_counts.get(i, 0) + 1
            gidx = self._glob_counts.get(i, 0)
            if chosen is None and spec.matches(
                run_name=run_name, glob_index=gidx, fired=self._fired.get(i, 0)
            ):
                self._fired[i] = self._fired.get(i, 0) + 1
                chosen = spec.fault
        return chosen

    # -- the ClaudeLLM backend contract ---------------------------------
    def generate(self, llm, system: str, prompt: str, resume_session_id):
        """Mirror ``_generate_via_cli``: return response text (or raise)."""
        site = _call_site()
        run_name = site.get("run_name", "")
        with self._lock:
            fault = self._select_fault(run_name)
            self.call_log.append({
                "run_name": run_name,
                "call_index": site.get("call_index"),
                "fault": fault.value if fault else None,
                "resume": bool(resume_session_id),
            })

        if fault is None:
            written = self.success.write_artifacts(
                prompt=prompt, system=system, project_root=_project_root(),
            )
            return self.success.response_text(written)

        return self._inject(fault, system=system, prompt=prompt)

    # -- fault behaviors -------------------------------------------------
    def _inject(self, fault: FaultClass, *, system: str, prompt: str) -> str:
        root = _project_root()
        if fault is FaultClass.TIMEOUT_STRING:
            return TIMEOUT_CONTENT
        if fault is FaultClass.STALL_STRING:
            return STALL_CONTENT
        if fault is FaultClass.EMPTY_RESPONSE:
            return ""
        if fault is FaultClass.ERROR_TEXT_AS_CONTENT:
            # Prose apology with NO "[ClaudeLLM error:" marker -- pins the
            # rtl_generator prose guard.
            return (
                "I'm sorry, I was unable to generate the RTL for this block "
                "because the specification is ambiguous. Please clarify."
            )
        if fault is FaultClass.NO_FILE_WRITTEN:
            # Plausible success text, but nothing written to disk.
            return "Done -- the module has been written and matches the reference."
        if fault is FaultClass.WRONG_PATH_WRITTEN:
            self._write_to_scratch(prompt, root)
            return (
                "Done. I wrote the module to my working directory "
                "(codex-call-*/). It should be picked up automatically."
            )
        if fault is FaultClass.STALE_ARTIFACT:
            # Intentionally write NOTHING: the harness pre-seeds an old-mtime
            # target file; advancing on it is the fail-open being pinned.
            return "Done -- reused the existing implementation, no changes needed."
        if fault is FaultClass.JSON_DISK_MISMATCH:
            self._write_empty_target(prompt, root)
            return (
                '{"verilog": "`include \\"generated.v\\"", "status": "complete", '
                '"module": "written"}'
            )
        if fault is FaultClass.TRUNCATED_JSON:
            return '{"verilog": "module foo(); assign x = 1\'b'
        if fault is FaultClass.FABRICATED_PASS:
            return (
                '{"passed": true, "verdict": "pass", "violations": [], '
                '"reason": "All checks passed."}'
            )
        if fault is FaultClass.PROVIDER_EXCEPTION:
            raise RuntimeError(PROVIDER_EXCEPTION_MESSAGE)
        if fault is FaultClass.CLI_HANG:
            # Only used in the watchdog unit test; sleep a SCALED short interval
            # so CORESMITH_TIMEOUT_MULTIPLIER can keep it sub-second.
            import time as _t

            from orchestrator._timeouts import scaled

            _t.sleep(min(scaled(2.0), 2.0))
            return STALL_CONTENT
        return ""  # unreachable

    # -- helpers ---------------------------------------------------------
    def _target_v_paths(self, prompt: str, root: str):
        import re
        from pathlib import Path

        out = []
        for m in re.finditer(r"[A-Za-z0-9_./-]+\.v\b", prompt or ""):
            rel = m.group(0)
            p = Path(rel) if rel.startswith("/") else Path(root) / rel
            if "rtl" in p.parts or "/rtl/" in str(p):
                out.append(p)
        return out

    def _write_to_scratch(self, prompt: str, root: str) -> None:
        from pathlib import Path

        from orchestrator.testing.success_scripts import _rtl_module

        for p in self._target_v_paths(prompt, root):
            scratch = Path(root) / f"codex-call-scratch" / p.name
            try:
                scratch.parent.mkdir(parents=True, exist_ok=True)
                scratch.write_text(_rtl_module(p.stem), encoding="utf-8")
            except OSError:
                pass

    def _write_empty_target(self, prompt: str, root: str) -> None:
        for p in self._target_v_paths(prompt, root):
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("", encoding="utf-8")  # 0-char module
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Module singleton (the seam coresmith_llm._get_testing_backend calls)
# ---------------------------------------------------------------------------
_BACKEND: FaultBackend | None = None
_BACKEND_LOCK = threading.Lock()


def get_backend() -> FaultBackend:
    global _BACKEND
    with _BACKEND_LOCK:
        if _BACKEND is None:
            _BACKEND = FaultBackend()
        return _BACKEND


def set_schedule(schedule: FaultSchedule) -> FaultBackend:
    b = get_backend()
    b.set_schedule(schedule)
    return b


def reset_backend() -> None:
    get_backend().reset()
