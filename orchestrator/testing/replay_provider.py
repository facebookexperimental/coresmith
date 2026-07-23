# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""ReplayBackend -- a ClaudeLLM backend that replays a recorded call corpus.

Selected via ``CORESMITH_LLM_PROVIDER=replay``.  ``coresmith_llm._generate_via_cli``
calls ``get_backend().generate(llm, system, prompt, resume_session_id)`` for every
LLM call; the backend serves the recorded response for the matching call and
re-applies the files that call wrote to disk (under the *current* project root),
so a graph node that reads its artifact back off disk behaves exactly as it did
during the recording -- without a live model.

Fixture layout (produced by ``scripts/make_replay_fixture.py``)::

    <fixture>/
      meta.json      # name, source, original_root, keep_prompts, engine_note
      calls.jsonl    # one record per recorded call (see below)
      files/<sha256> # content-addressed blobs (responses + writes)
      pre/           # optional: files that existed BEFORE the segment

Match cascade (per incoming call, keyed on the call-site ``run_name`` the
``ClaudeLLM.call`` stamped into the ContextVar):

  1. exact   -- ``(run_name, site_index)``: the next unconsumed recorded call for
                that run_name (site_index = how many we've served for it).
  2. digest  -- among unconsumed same-run_name calls, one whose normalized-prompt
                SHA-256 equals the incoming prompt's (survives root/timestamp edits).
  3. fuzzy   -- among unconsumed calls with a retained prompt, the highest
                ``token_jaccard`` >= ``CORESMITH_REPLAY_FUZZY_MIN`` (default 0.80).

On a miss: **strict** mode (default, and the tests' default) raises
:class:`ReplayMissError` with nearest-candidate diagnostics; **lenient**
(``CORESMITH_REPLAY_STRICT=0``) delegates to :class:`CannedDesignScript` so the
run limps forward (useful when replaying a *segment* of a longer run).
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Optional

from orchestrator.testing.prompt_norm import (
    normalize_prompt,
    prompt_digest,
    token_jaccard,
)
from orchestrator.testing.success_scripts import CannedDesignScript


class ReplayMissError(RuntimeError):
    """Raised in strict mode when no recorded call matches an incoming call."""


def _call_site() -> dict:
    """Read the (run_name, call_index) the ClaudeLLM.call set for this call."""
    try:
        from orchestrator.langchain.agents.coresmith_llm import _call_site_context

        return _call_site_context.get(None) or {}
    except Exception:
        return {}


def _fuzzy_min() -> float:
    try:
        return float(os.environ.get("CORESMITH_REPLAY_FUZZY_MIN", "0.80"))
    except (TypeError, ValueError):
        return 0.80


def _strict_default() -> bool:
    return os.environ.get("CORESMITH_REPLAY_STRICT", "1").strip().lower() not in (
        "0", "false", "no", "",
    )


def _current_root() -> str:
    return (
        os.environ.get("CORESMITH_PROJECT_ROOT", "").strip()
        or os.environ.get("CORESMITH_LLM_LOG_ROOT", "").strip()
        or os.getcwd()
    )


# ---------------------------------------------------------------------------
# Fixture model
# ---------------------------------------------------------------------------
class ReplayFixture:
    """A loaded replay corpus (meta + calls + content-addressed blobs)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        meta_path = self.root / "meta.json"
        self.meta: dict = (
            json.loads(meta_path.read_text()) if meta_path.exists() else {}
        )
        self.original_root: str = self.meta.get("original_root", "")
        self.calls: list[dict] = []
        calls_path = self.root / "calls.jsonl"
        if calls_path.exists():
            for line in calls_path.read_text().splitlines():
                line = line.strip()
                if line:
                    self.calls.append(json.loads(line))

    def blob_bytes(self, ref: str) -> bytes:
        """Read a content-addressed blob (``files/<sha>``)."""
        p = self.root / ref if not ref.startswith("/") else Path(ref)
        return p.read_bytes()

    def blob_text(self, ref: str) -> str:
        return self.blob_bytes(ref).decode("utf-8", errors="replace")

    def seed_pre(self, dest_root: str | Path) -> list[str]:
        """Copy the optional ``pre/`` tree into ``dest_root``. Returns paths."""
        pre = self.root / "pre"
        written: list[str] = []
        if not pre.is_dir():
            return written
        dest = Path(dest_root)
        for src in pre.rglob("*"):
            if src.is_file():
                rel = src.relative_to(pre)
                tgt = dest / rel
                tgt.parent.mkdir(parents=True, exist_ok=True)
                tgt.write_bytes(src.read_bytes())
                written.append(str(tgt))
        return written


class ReplayBackend:
    """In-process LLM backend that replays a recorded call corpus."""

    def __init__(
        self,
        fixture: Optional[ReplayFixture | str | Path] = None,
        *,
        strict: Optional[bool] = None,
    ) -> None:
        self._lock = threading.RLock()
        self.success = CannedDesignScript()
        self.call_log: list[dict] = []
        self._served: dict[str, int] = {}
        self._consumed: set[int] = set()
        self.strict = _strict_default() if strict is None else strict
        self.fixture: Optional[ReplayFixture] = None
        if fixture is None:
            env = os.environ.get("CORESMITH_REPLAY_FIXTURE", "").strip()
            if env:
                fixture = env
        if fixture is not None:
            self.set_fixture(fixture)

    # -- lifecycle -------------------------------------------------------
    def set_fixture(self, fixture: ReplayFixture | str | Path) -> None:
        with self._lock:
            if not isinstance(fixture, ReplayFixture):
                fixture = ReplayFixture(fixture)
            self.fixture = fixture
            self._by_run: dict[str, list[dict]] = {}
            for i, c in enumerate(fixture.calls):
                c.setdefault("_idx", i)
                self._by_run.setdefault(c.get("run_name", ""), []).append(c)
            for lst in self._by_run.values():
                lst.sort(key=lambda e: e.get("site_index", 0))
            self._served.clear()
            self._consumed.clear()
            self.call_log.clear()

    def reset(self) -> None:
        with self._lock:
            self._served.clear()
            self._consumed.clear()
            self.call_log.clear()

    # -- properties ------------------------------------------------------
    @property
    def unconsumed(self) -> list[dict]:
        """Recorded calls not yet served (order preserved)."""
        if not self.fixture:
            return []
        return [c for c in self.fixture.calls if c.get("_idx") not in self._consumed]

    # -- the ClaudeLLM backend contract ---------------------------------
    def generate(self, llm, system: str, prompt: str, resume_session_id) -> str:
        site = _call_site()
        run_name = site.get("run_name", "")
        with self._lock:
            entry, how = self._match(run_name, system, prompt)
            self.call_log.append({
                "run_name": run_name,
                "call_index": site.get("call_index"),
                "matched": how,
                "site_index": entry.get("site_index") if entry else None,
                "resume": bool(resume_session_id),
            })
            if entry is not None:
                self._consumed.add(entry.get("_idx"))
                self._served[run_name] = self._served.get(run_name, 0) + 1

        if entry is not None:
            self._apply_writes(entry)
            return self._response_text(entry)

        # miss
        if self.strict:
            raise ReplayMissError(self._miss_diagnostics(run_name, system, prompt))
        # lenient: keep the run moving via the canned success script.
        written = self.success.write_artifacts(
            prompt=prompt, system=system, project_root=_current_root(),
        )
        return self.success.response_text(written)

    # -- matching --------------------------------------------------------
    def _roots(self) -> list[str]:
        roots = [_current_root()]
        if self.fixture and self.fixture.original_root:
            roots.append(self.fixture.original_root)
        return roots

    def _match(self, run_name: str, system: str, prompt: str):
        if not self.fixture:
            return None, "no-fixture"
        entries = self._by_run.get(run_name, [])
        # 1. exact (run_name, site_index)
        k = self._served.get(run_name, 0)
        for e in entries:
            if e.get("site_index") == k and e.get("_idx") not in self._consumed:
                return e, "exact"
        # 2. digest among unconsumed same-run
        roots = self._roots()
        dg = prompt_digest(prompt, roots)
        for e in entries:
            if e.get("_idx") in self._consumed:
                continue
            if e.get("prompt_digest") and e["prompt_digest"] == dg:
                return e, "digest"
        # 3. fuzzy jaccard among unconsumed (same-run first, else any) with a
        #    retained prompt (only --keep-prompts fixtures carry prompt_norm).
        pool = [e for e in entries
                if e.get("_idx") not in self._consumed and e.get("prompt_norm")]
        if not pool:
            pool = [e for e in self.fixture.calls
                    if e.get("_idx") not in self._consumed and e.get("prompt_norm")]
        best, best_sc = None, 0.0
        for e in pool:
            sc = token_jaccard(prompt, e["prompt_norm"], roots)
            if sc > best_sc:
                best, best_sc = e, sc
        if best is not None and best_sc >= _fuzzy_min():
            return best, f"fuzzy:{best_sc:.2f}"
        return None, "miss"

    # -- serving ---------------------------------------------------------
    def _response_text(self, entry: dict) -> str:
        ref = entry.get("response_ref")
        if ref and self.fixture:
            try:
                return self.fixture.blob_text(ref)
            except OSError:
                pass
        return entry.get("response", "")

    def _apply_writes(self, entry: dict) -> None:
        """Materialize the files this call wrote, under the current root."""
        if not self.fixture:
            return
        root = Path(_current_root())
        for w in entry.get("writes", []) or []:
            rel = w.get("relpath", "")
            if not rel:
                continue
            # tolerate a stored <RUN>/ prefix
            rel = rel.replace("<RUN>/", "").replace("<RUN>", "").lstrip("/")
            tgt = root / rel
            try:
                data = self.fixture.blob_bytes(w["ref"]) if w.get("ref") else b""
                tgt.parent.mkdir(parents=True, exist_ok=True)
                tgt.write_bytes(data)
            except (OSError, KeyError):
                continue

    # -- diagnostics -----------------------------------------------------
    def _miss_diagnostics(self, run_name: str, system: str, prompt: str) -> str:
        roots = self._roots()
        dg = prompt_digest(prompt, roots)
        k = self._served.get(run_name, 0)
        lines = [
            "ReplayMiss: no recorded call matched this live call.",
            f"  live run_name = {run_name!r}, expected site_index = {k}",
            f"  live prompt_digest = {dg[:16]}...",
        ]
        same = [e for e in self._by_run.get(run_name, [])
                if e.get("_idx") not in self._consumed]
        if same:
            lines.append(f"  {len(same)} unconsumed call(s) for this run_name:")
            for e in same[:4]:
                jac = (
                    f", jaccard={token_jaccard(prompt, e['prompt_norm'], roots):.2f}"
                    if e.get("prompt_norm") else ""
                )
                lines.append(
                    f"    - site_index={e.get('site_index')} "
                    f"digest={str(e.get('prompt_digest'))[:16]}...{jac}"
                )
        else:
            names = sorted({e.get("run_name", "") for e in self.unconsumed})
            lines.append(
                "  no unconsumed calls for this run_name. "
                f"Unconsumed run_names: {names[:8]}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module singleton (the seam coresmith_llm._get_testing_backend calls)
# ---------------------------------------------------------------------------
_BACKEND: Optional[ReplayBackend] = None
_BACKEND_LOCK = threading.Lock()


def get_backend() -> ReplayBackend:
    global _BACKEND
    with _BACKEND_LOCK:
        if _BACKEND is None:
            _BACKEND = ReplayBackend()
        return _BACKEND


def set_fixture(fixture: ReplayFixture | str | Path, *, strict: Optional[bool] = None) -> ReplayBackend:
    b = get_backend()
    b.set_fixture(fixture)
    if strict is not None:
        b.strict = strict
    return b


def reset_backend() -> None:
    global _BACKEND
    with _BACKEND_LOCK:
        _BACKEND = None
