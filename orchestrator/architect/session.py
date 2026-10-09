# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Native sessions: ONE resumable agentic CLI session (claude / opencode /
codex) driven in sittings through the ``coresmith`` CLI.

Only cluster workers use this now (``CORESMITH_FANOUT=cluster``,
:class:`orchestrator.architect.cluster.ClusterSession`). The Architect -- the
coding agent that talks to the human and calls the CLI -- is outside the
engine: nothing here launches, resumes or decides for it. ``ArchitectSession``
is the historical name of the base class.

Mechanics:
* A runner (``runners.py``) runs ONE turn of a resumable session in the
  project root with file tools + a shell, streams the CLI's JSON events into
  ``transcript-<n>.jsonl`` and returns ``{ok, rc, session_id, text, cost_usd,
  cost_cumulative, turns, tokens, ...}``; ``stderr`` goes to
  ``stderr-<n>.log`` and its tail stays on the result (``stderr_tail``) so a
  failure is classified from the runner's diagnostics, never from the model's
  last message.
* Between invocations the runner (not the model) reads the session's
  ``stage_status``; the loop is bounded (``max_sittings``), stoppable
  (``STOP`` file) and ends ``waiting_for_answers`` when an invocation leaves an
  open must-answer question. There is no no-progress counter: an unchanged
  blocker list is not a reason to stop.
* Cost: ``claude --resume`` reports ``total_cost_usd`` cumulatively, so a
  invocation's cost is the delta against the previous cumulative figure
  (``session_cost_usd``); opencode reports each invocation's own cost
  (``cost_cumulative: False``), booked as is.
* Everything the model produced is on disk (transcript per invocation, prompts,
  ``status.json``); the project database holds what it registered.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from .runners import make_runner, resolve_model, resolve_runner

PROMPT_DIR = Path(__file__).resolve().parents[1] / "langchain" / "prompts"
DONE_STAGES = ("blocks", "integration", "acceptance", "backend")
# states a session ends in that the operator acts on (and resumes from)
WAITING_STATES = ("waiting_for_answers",)
STDERR_TAIL_CHARS = 2000


def answer_command(qid) -> str:
    return f'coresmith question answer Q{qid} --ruling "<your ruling>"'


def _read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except OSError:
        return ""


def _strip_format_fields(text: str) -> str:
    """Specialist prompts are str.format templates ({prd_context} ...); in an
    appendix they are reference material, so neutralise the placeholders."""
    import re
    return re.sub(r"\{([a-z_]+_context|shuttle_context|[a-z_]+)\}", lambda m: "<" + m.group(1) + ">", text)


class ArchitectSession:
    """Base class of a native session (historical name). Subclasses supply the
    system prompt, the prompts, the lease name and what "done" means; this
    class owns the runner, the files, the status record and the loop."""

    def __init__(self, project_root, *, model: str | None = None, max_turns: int = 300, max_sittings: int = 12,
                 claude_path: str | None = None, timeout_s: int = 4 * 3600, coresmith_bin: str | None = None,
                 runner: str | None = None, opencode_path: str | None = None):
        self.root = Path(project_root).resolve()
        self.dir = self.root / ".coresmith" / "architect"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_turns = max_turns
        self.max_sittings = max_sittings
        self.timeout_s = timeout_s
        self.claude_path = claude_path
        self.opencode_path = opencode_path
        self.coresmith_bin = coresmith_bin or str(Path(__file__).resolve().parents[2] / "bin" / "coresmith")
        # resolved lazily (subclasses move ``self.dir`` after this): see ``runner_name``
        self._runner_arg, self._model_arg = runner, model
        self._runner_name: str | None = None
        self._model: str | None = None
        self._runner = None

    # -- runner (claude | opencode | codex) --------------------------------------
    def _resolve_runner(self) -> None:
        """``runner=`` > ``CORESMITH_ARCHITECT_PROVIDER`` (the native-session
        runner selector; historical name) > the runner that created this
        session's recorded id (``status.json``: a session can only be resumed
        by its own CLI) > ``CORESMITH_LLM_PROVIDER``."""
        if self._runner_name:
            return
        st = self.status()
        recorded = st.get("runner") if st.get("session_id") else None
        explicit = self._runner_arg or os.environ.get("CORESMITH_ARCHITECT_PROVIDER")
        name = resolve_runner(explicit or recorded)
        model = self._model_arg or os.environ.get("CORESMITH_ARCHITECT_MODEL")
        if not model and not explicit and recorded and recorded == name:
            model = st.get("model")
        self._runner_name = name
        self._model = resolve_model(name, model)

    @property
    def runner_name(self) -> str:
        self._resolve_runner()
        return self._runner_name or "claude"

    @property
    def model(self) -> str:
        self._resolve_runner()
        return self._model or ""

    @model.setter
    def model(self, value: str) -> None:
        self._model_arg, self._model, self._runner_name, self._runner = value, None, None, None

    @property
    def runner(self):
        if self._runner is None:
            name = self.runner_name
            self._runner = make_runner(name, self.model, self.opencode_path if name == "opencode" else self.claude_path)
        return self._runner

    # -- identity ------------------------------------------------------------
    # Two sittings of the SAME session never run concurrently: each session
    # holds its own lease (a cluster worker's is ``cluster:<name>``).
    LEASE_HOLDER = "native session"
    BUSY_REASON = "session busy"

    def lease_name(self) -> str:
        return "native_session"

    def system_prompt(self) -> str:
        """The session's system prompt; subclasses supply it."""
        return ""

    # -- state ---------------------------------------------------------------
    @property
    def status_path(self) -> Path:
        return self.dir / "status.json"

    def status(self) -> dict:
        try:
            return json.loads(self.status_path.read_text())
        except (OSError, ValueError):
            return {"state": "idle", "session_id": "", "sittings": 0}

    def _write_status(self, **kw) -> dict:
        st = self.status()
        st.update(kw)
        st["ts"] = time.time()
        self.status_path.write_text(json.dumps(st, indent=2))
        return st

    def stop_requested(self) -> bool:
        return (self.dir / "STOP").exists()

    # -- tools ---------------------------------------------------------------
    def stage_status(self) -> dict:
        try:
            p = subprocess.run([self.coresmith_bin, "stage", "status", "--project-root", str(self.root), "--json"],
                               capture_output=True, text=True, timeout=120)
            return json.loads(p.stdout) if p.stdout.strip().startswith("{") else {"stage": "?", "blocked_by": [], "error": p.stderr[-500:]}
        except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
            return {"stage": "?", "blocked_by": [], "error": str(exc)}

    def open_questions(self) -> list[dict]:
        """Open must-answer questions (``[{id, text, item_id, asked_by}]``)."""
        try:
            from orchestrator.state_store.project_db import open_project
            return [{k: q.get(k) for k in ("id", "text", "item_id", "asked_by")}
                    for q in open_project(self.root).questions(open_only=True, must_answer=True)]
        except Exception:  # noqa: BLE001
            return []

    def run_status(self) -> str:
        try:
            p = subprocess.run([self.coresmith_bin, "status", "--project-root", str(self.root)],
                               capture_output=True, text=True, timeout=120)
            return (p.stdout or "")[-6000:]
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"(coresmith status unavailable: {exc})"

    def done(self, st: dict | None = None) -> bool:
        st = st or self.stage_status()
        return st.get("stage") in DONE_STAGES

    # -- prompts -------------------------------------------------------------
    def opening_prompt(self) -> str:
        return "\n".join(["Continue the work in this directory.", "", "## Run status (from `coresmith status`)", "```",
                          self.run_status(), "```"])

    def resume_prompt(self, st: dict, sitting: int) -> str:
        blockers = st.get("blocked_by") or []
        lines = [f"Invocation {sitting}: continue. The run is at stage `{st.get('stage')}`."]
        for b in blockers:
            lines.append(f"- {b.get('code')}: {b.get('text')}" + (f" [{b.get('count')}] {', '.join(b.get('ids') or [])[:400]}" if b.get("ids") else ""))
        lines += ["", "## Run status", "```", self.run_status(), "```"]
        return "\n".join(lines)

    # -- one invocation -----------------------------------------------------------
    def sitting_env(self, extra_env: dict | None = None) -> dict:
        """The CLI's environment: the project root, the caller's role/actor
        (``extra_env``) and the engine's ``bin/`` first on PATH so
        ``coresmith ...`` resolves in the model's shell on any runner."""
        env = dict(os.environ)
        env["CORESMITH_PROJECT_ROOT"] = str(self.root)
        bindir = str(Path(self.coresmith_bin).resolve().parent)
        path = env.get("PATH", "")
        if bindir not in path.split(os.pathsep):
            env["PATH"] = bindir + (os.pathsep + path if path else "")
        env.update(extra_env or {})
        return env

    def sit(self, prompt: str, *, resume: str = "", index: int = 1, name: str = "",
            timeout_s: int | None = None, max_turns: int | None = None,
            extra_env: dict | None = None) -> dict:
        """One turn of the session on the configured runner. Files are
        ``prompt-<index>.md`` / ``transcript-<index>.jsonl`` /
        ``stderr-<index>.log``, or ``<name>.md`` / ``<name>.jsonl`` /
        ``<name>.stderr.log`` when named. The stderr tail stays on the result
        (``stderr_tail``) for outcome classification."""
        system = self.system_prompt()
        (self.dir / "system.md").write_text(system)
        prompt_file = self.dir / (f"{name}.md" if name else f"prompt-{index}.md")
        transcript = self.dir / (f"{name}.jsonl" if name else f"transcript-{index}.jsonl")
        stderr_file = self.dir / (f"{name}.stderr.log" if name else f"stderr-{index}.log")
        prompt_file.write_text(prompt)
        t0 = time.time()
        runner = self.runner
        meta = {"runner": runner.name, "model": runner.model}
        try:
            res = runner.run(prompt, system_file=self.dir / "system.md", resume=resume, transcript=transcript,
                             cwd=self.root, env=self.sitting_env(extra_env), max_turns=int(max_turns or self.max_turns),
                             timeout_s=int(timeout_s or self.timeout_s))
        except (OSError, ValueError) as exc:
            return {"ok": False, "rc": -1, "error": str(exc), "session_id": resume, "elapsed_s": 0, **meta}
        stderr = res.pop("stderr", "") or ""
        stderr_file.write_text(stderr)
        res["stderr_tail"] = stderr[-STDERR_TAIL_CHARS:]
        res["text"] = (res.get("text") or "")[-4000:]
        return {**res, **meta, "elapsed_s": round(time.time() - t0, 1), "transcript": str(transcript)}

    # -- the loop ----------------------------------------------------------------
    def run(self) -> dict:
        """Invocations until :meth:`done`, the invocation bound (``max_sittings``), a STOP file or an
        open must-answer question. Holds the session's own lease
        (:meth:`lease_name`) so two invocations of the SAME session never
        overlap; sessions with different leases never block each other."""
        try:
            from orchestrator.state_store.leases import LeaseUnavailable, db_lease
            from orchestrator.state_store.project_db import open_project
            lease = db_lease(open_project(self.root), self.lease_name(), ttl_s=300, wait_s=0,
                             meta={"holder": self.LEASE_HOLDER})
        except Exception:  # noqa: BLE001 - no database: run unguarded
            lease, LeaseUnavailable = None, RuntimeError
        try:
            if lease is None:
                return self._run()
            with lease:
                return self._run()
        except LeaseUnavailable as exc:
            return self._write_status(state="busy", stop_reason=f"{self.BUSY_REASON}: {exc}")

    def _run(self) -> dict:
        st0 = self.status()
        sid = st0.get("session_id") or ""
        sittings = int(st0.get("sittings") or 0)
        resumed_from = st0.get("state") if st0.get("state") in WAITING_STATES else ""
        self._write_status(state="running", started_ts=st0.get("started_ts") or time.time(), stop_reason="",
                           waiting_questions=[], resumed_from=resumed_from, runner=self.runner_name, model=self.model)
        # cost: `claude --resume` reports the session's CUMULATIVE total_cost_usd
        total_cost = float(st0.get("cost_usd") or 0)
        if "session_cost_usd" in st0:
            session_cum = float(st0.get("session_cost_usd") or 0)
        else:   # a status from before the delta accounting: the last cumulative figure is the truth
            session_cum = float((st0.get("last") or {}).get("cost_usd") or 0) if sid else 0.0
            if sid and session_cum:
                total_cost = session_cum
        stage = self.stage_status()
        for _ in range(self.max_sittings):
            if self.stop_requested():
                return self._write_status(state="stopped", stop_reason="STOP file")
            if self.done(stage):
                return self._write_status(state="done", stage=stage.get("stage"), sittings=sittings,
                                          cost_usd=total_cost)
            sittings += 1
            prompt = self.opening_prompt() if not sid else self.resume_prompt(stage, sittings)
            res = self.sit(prompt, resume=sid, index=sittings)
            new_sid = res.get("session_id") or sid
            if new_sid != sid:
                session_cum = 0.0   # a new session starts its own cumulative figure
            sid = new_sid
            cum = res.get("cost_usd")
            delta = 0.0
            if cum is not None and not res.get("cost_cumulative", True):
                # a runner that reports the invocation's own cost (opencode)
                delta = max(0.0, float(cum))
                session_cum += delta
                cum = session_cum
            elif cum is not None:
                delta = max(0.0, float(cum) - session_cum)
                session_cum = max(session_cum, float(cum))
            total_cost += delta
            after = self.stage_status()
            from orchestrator.architect.outcomes import outcome
            result_outcome = outcome(res)
            if result_outcome["kind"] != "success":
                from orchestrator.state_store.project_db import open_project
                open_project(self.root).set_flag("agent_failure", {**result_outcome, "ts": time.time()})
                return self._write_status(
                    state=result_outcome["kind"], stop_reason=result_outcome["reason"],
                    session_id=sid, sittings=sittings, cost_usd=total_cost,
                    session_cost_usd=session_cum, last=dict(res, outcome=result_outcome),
                    stage=after.get("stage"))
            last = dict(res, text=res.get("text", "")[-1500:], cost_usd=round(delta, 6), session_cost_usd=cum)
            self._write_status(state="running", session_id=sid, sittings=sittings, cost_usd=total_cost,
                               session_cost_usd=session_cum, last=last, stage=after.get("stage"))
            if not res.get("ok") and not sid:
                return self._write_status(state="error", stop_reason=f"{self.runner_name} failed rc={res.get('rc')}: "
                                                                      f"{res.get('error', '')}")
            stage = after
            if self.done(stage):
                continue
            qs = self.open_questions()
            codes = {b.get("code") for b in stage.get("blocked_by") or []}
            if qs or (codes and codes <= {"OPEN_QUESTIONS"}):
                return self._write_status(
                    state="waiting_for_answers", stage=stage.get("stage"), blocked_by=stage.get("blocked_by"),
                    sittings=sittings, cost_usd=total_cost, waiting_questions=[
                        {**q, "answer": answer_command(q["id"])} for q in qs],
                    stop_reason="open must-answer question(s): answer them, then resume the session")
        if self.done(stage):
            return self._write_status(state="done", stage=stage.get("stage"), sittings=sittings, cost_usd=total_cost)
        return self._write_status(state="blocked", stage=stage.get("stage"), blocked_by=stage.get("blocked_by"),
                                  sittings=sittings, cost_usd=total_cost, stop_reason="max_sittings")
