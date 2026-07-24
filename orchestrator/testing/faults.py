# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Fault taxonomy for the fault-injecting LLM provider (Package C).

A :class:`FaultSchedule` is a list of :class:`FaultSpec`, each of which matches a
subset of LLM calls (by ``run_name`` glob, absolute ``call_index``, or ``every_n``)
and names a :class:`FaultClass` to inject. The schedule is consulted by
``fault_provider.FaultBackend`` on every ``call()``.

Each FaultClass corresponds to a real failure mode observed in the OTEL/sqlite
forensics, and pins a specific fail-open site (see C3 in the plan):

  timeout_string       -> "[ClaudeLLM error: ... timed out ...]" content string
  stall_string         -> "[ClaudeLLM error: ... stalled ...]" content string
  empty_response       -> "" (empty content)
  error_text_as_content-> prose error with no marker (pins rtl_generator prose guard)
  no_file_written      -> plausible response but writes NO file (pins "did not write")
  wrong_path_written   -> writes to a codex-call-*/ scratch path, not the target
  stale_artifact       -> leaves an OLD-mtime file in place (pins no_stale_advance)
  json_disk_mismatch   -> returns JSON claiming success but on-disk module is 0 chars
  truncated_json       -> a JSON object cut off mid-string
  fabricated_pass      -> claims the gate/DV passed with no evidence
  provider_exception   -> raises the real codex '--sandbox' exit-2 error
  cli_hang             -> sleeps past the (scaled) stall threshold (watchdog unit test)
"""

from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass, field
from enum import Enum


class FaultClass(str, Enum):
    TIMEOUT_STRING = "timeout_string"
    STALL_STRING = "stall_string"
    EMPTY_RESPONSE = "empty_response"
    ERROR_TEXT_AS_CONTENT = "error_text_as_content"
    NO_FILE_WRITTEN = "no_file_written"
    WRONG_PATH_WRITTEN = "wrong_path_written"
    STALE_ARTIFACT = "stale_artifact"
    JSON_DISK_MISMATCH = "json_disk_mismatch"
    TRUNCATED_JSON = "truncated_json"
    FABRICATED_PASS = "fabricated_pass"
    PROVIDER_EXCEPTION = "provider_exception"
    CLI_HANG = "cli_hang"

    @classmethod
    def in_process(cls) -> list[FaultClass]:
        """The 11 pure in-process fault classes (all except CLI_HANG, which
        needs a real subprocess + scaled watchdog to exercise)."""
        return [c for c in cls if c is not cls.CLI_HANG]


# Content strings the ClaudeLLM CLI path returns on failure (never raised).
TIMEOUT_CONTENT = "[ClaudeLLM error: claude CLI timed out after 1200s (injected)]"
STALL_CONTENT = "[ClaudeLLM error: claude CLI stalled after 1200s (injected)]"
# The real error the codex --sandbox bug raises (A-Fix 6 motivation).
PROVIDER_EXCEPTION_MESSAGE = "error: unexpected argument '--sandbox' found"


@dataclass
class FaultSpec:
    """One fault rule.

    Matching (all provided constraints must hold):
      - ``run_name_glob``: fnmatch against the call's run_name (default '*').
      - ``call_index``: absolute 1-based index of the matching call to fault
        (counted over calls that pass the glob). ``None`` = any.
      - ``every_n``: fault every Nth matching call. ``None`` = every call.
    ``max_faults`` caps how many times this spec fires (``None`` = unlimited).
    """

    fault: FaultClass
    run_name_glob: str = "*"
    call_index: int | None = None
    every_n: int | None = None
    max_faults: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.fault, FaultClass):
            self.fault = FaultClass(str(self.fault))

    def matches(self, *, run_name: str, glob_index: int, fired: int) -> bool:
        """Whether this spec should fire for a call.

        ``glob_index`` is the 1-based count of calls (so far, inclusive) whose
        run_name matched this spec's glob. ``fired`` is how many times this spec
        has already fired.
        """
        if self.max_faults is not None and fired >= self.max_faults:
            return False
        if not fnmatch.fnmatch(run_name or "", self.run_name_glob):
            return False
        if self.call_index is not None and glob_index != self.call_index:
            return False
        if self.every_n is not None and (glob_index % self.every_n) != 0:
            return False
        return True

    def to_dict(self) -> dict:
        return {
            "fault": self.fault.value,
            "run_name_glob": self.run_name_glob,
            "call_index": self.call_index,
            "every_n": self.every_n,
            "max_faults": self.max_faults,
        }

    @classmethod
    def from_dict(cls, d: dict) -> FaultSpec:
        return cls(
            fault=FaultClass(d["fault"]),
            run_name_glob=d.get("run_name_glob", "*"),
            call_index=d.get("call_index"),
            every_n=d.get("every_n"),
            max_faults=d.get("max_faults"),
        )


@dataclass
class FaultSchedule:
    """An ordered list of FaultSpecs. The FIRST matching spec wins per call."""

    specs: list[FaultSpec] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps([s.to_dict() for s in self.specs])

    @classmethod
    def from_json(cls, blob: str) -> FaultSchedule:
        if not blob or not blob.strip():
            return cls([])
        data = json.loads(blob)
        # Accept either a bare list of specs or {"specs": [...]}.
        if isinstance(data, dict):
            data = data.get("specs", [])
        return cls([FaultSpec.from_dict(d) for d in data])

    @classmethod
    def from_env(cls, env_var: str = "CORESMITH_FAULT_SCHEDULE") -> FaultSchedule:
        import os

        return cls.from_json(os.environ.get(env_var, ""))

    @classmethod
    def single(cls, fault: FaultClass, **kw) -> FaultSchedule:
        """Convenience: a schedule with one spec faulting matching calls."""
        return cls([FaultSpec(fault=fault, **kw)])
