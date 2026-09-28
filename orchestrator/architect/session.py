# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The architect sitting (step 3): ONE long-lived agentic session that can
only advance the run through the ``coresmith`` CLI.

Why one session: every architecture decision (PRD → SAD → executable SAD →
FRD → decomposition → fabric → contracts → ABI → uArch specs → model
refinement) is a trade-off against every other one; splitting them into
thirteen policies replaced a resident context with lossy documents (the
benchmark post-mortem). The sitting keeps the whole problem in one KV cache
and uses deterministic tools for everything that must be deterministic.

Mechanics:
* ``claude -p`` in the project root with file tools + Bash, stream-json,
  ``--max-turns`` per sitting; the session id from the ``system/init`` event
  is persisted and every later sitting ``--resume``s it (cache warm).
* Between sittings the runner (not the model) reads ``coresmith stage
  status``: done when the run has entered ``blocks``; otherwise the next
  sitting starts with the exact blockers. The loop is bounded
  (``max_sittings``) and stoppable (``.coresmith/architect/STOP``).
* Everything the model produced is on disk (``.coresmith/architect/``:
  transcript per sitting, prompts, status.json); the ontology holds what it
  registered. Recovery = restart with ``coresmith status``, never a summary.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parents[1] / "langchain" / "prompts"
APPENDIX = ["prd_spec.md", "sad_spec.md", "frd_spec.md", "block_diagram.md", "interface_definition.md",
            "ers_doc.md", "uarch_spec_generator.md", "systemc_model_generator.md", "frd_eval_generator.md",
            "skills/soc_fabric.md", "skills/systemc_tlm_lt.md"]
DONE_STAGES = ("blocks", "integration", "acceptance", "backend")


def _read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except OSError:
        return ""


def _strip_format_fields(text: str) -> str:
    """Specialist prompts are str.format templates ({prd_context} ...); in the
    appendix they are reference material, so neutralise the placeholders."""
    import re
    return re.sub(r"\{([a-z_]+_context|shuttle_context|[a-z_]+)\}", lambda m: "<" + m.group(1) + ">", text)


def build_system_prompt() -> str:
    head = _read(PROMPT_DIR / "architect.md")
    parts = [head, "", "# Appendix: the specialist guidance (reference; the stage contracts above win)"]
    for name in APPENDIX:
        t = _read(PROMPT_DIR / name)
        if t:
            parts += ["", f"## {name}", "", _strip_format_fields(t)]
    return "\n".join(parts)


class ArchitectSession:
    def __init__(self, project_root, *, model: str | None = None, max_turns: int = 300, max_sittings: int = 12,
                 claude_path: str | None = None, timeout_s: int = 4 * 3600, coresmith_bin: str | None = None):
        self.root = Path(project_root).resolve()
        self.dir = self.root / ".coresmith" / "architect"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.model = model or os.environ.get("CORESMITH_ARCHITECT_MODEL") or os.environ.get("CORESMITH_COORDINATOR_MODEL") \
            or os.environ.get("CORESMITH_MODEL") or "claude-opus-5-5"
        self.max_turns = max_turns
        self.max_sittings = max_sittings
        self.timeout_s = timeout_s
        self.claude_path = claude_path or os.environ.get("CLAUDE_CLI_PATH") or shutil.which("claude") or "claude"
        self.coresmith_bin = coresmith_bin or str(Path(__file__).resolve().parents[2] / "bin" / "coresmith")

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
        try:
            from orchestrator.state_store.project_db import open_project
            db = open_project(self.root)
            db.set_setting("architect_state", str(st.get("state", "")))
            if st.get("session_id"):
                db.set_setting("architect_session_id", str(st["session_id"]))
        except Exception:  # noqa: BLE001
            pass
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
        task = _read(self.root / "inputs" / "task.yaml")
        req = _read(self.root / "inputs" / "requirements.md")
        return "\n".join([
            "You are starting (or continuing) the architect sitting for this project. Work in this directory.",
            "", "## Run status (from `coresmith status`)", "```", self.run_status(), "```",
            "", "## Task", "```yaml", task[:4000], "```",
            "", "## Requirements (inputs/requirements.md, first 12k chars; read the file for the rest)", req[:12000],
            "", "## What to do now",
            "Read `coresmith status --json`. Do the work the current stage needs (write the documents, run the tools, "
            "answer or park questions), register every artifact with `coresmith register`, and call `coresmith stage next`. "
            "Repeat until the run enters the `blocks` stage. Never advance by any other means; never edit the engine.",
        ])

    def resume_prompt(self, st: dict, sitting: int) -> str:
        blockers = st.get("blocked_by") or []
        lines = [f"Sitting {sitting}: continue. The run is at stage `{st.get('stage')}`."]
        if blockers:
            lines.append("`coresmith stage next` is blocked by:")
            for b in blockers:
                lines.append(f"- {b.get('code')}: {b.get('text')}" + (f" [{b.get('count')}] {', '.join(b.get('ids') or [])[:400]}" if b.get("ids") else ""))
        else:
            lines.append("Nothing blocks the stage; run `coresmith stage next` and continue with the next stage.")
        lines += ["", "## Run status", "```", self.run_status(), "```",
                  "Resolve the blockers with the tools, register, then `coresmith stage next`."]
        return "\n".join(lines)

    # -- one sitting -----------------------------------------------------------
    def sit(self, prompt: str, *, resume: str = "", index: int = 1) -> dict:
        system = build_system_prompt()
        (self.dir / "system.md").write_text(system)
        (self.dir / f"prompt-{index}.md").write_text(prompt)
        cmd = [self.claude_path, "-p", "--output-format", "stream-json", "--verbose", "--model", self.model,
               "--max-turns", str(self.max_turns), "--permission-mode", "auto",
               "--disallowedTools", "Monitor,ScheduleWakeup,EnterPlanMode",
               "--append-system-prompt-file", str(self.dir / "system.md")]
        if resume:
            cmd += ["--resume", resume]
        env = dict(os.environ)
        env["CORESMITH_PROJECT_ROOT"] = str(self.root)
        env.setdefault("CORESMITH_ARCHITECT_SITTING", "1")
        t0 = time.time()
        transcript = self.dir / f"transcript-{index}.jsonl"
        try:
            with transcript.open("w") as out:
                p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=out, stderr=subprocess.PIPE, text=True,
                                     cwd=str(self.root), env=env)
                try:
                    _, err = p.communicate(prompt, timeout=self.timeout_s)
                    rc = p.returncode
                except subprocess.TimeoutExpired:
                    p.kill()
                    _, err = p.communicate()
                    rc = -9
        except OSError as exc:
            return {"ok": False, "rc": -1, "error": str(exc), "session_id": resume, "elapsed_s": 0}
        sid, text, cost, turns = resume, "", None, 0
        for raw in _read(transcript).splitlines():
            try:
                obj = json.loads(raw)
            except ValueError:
                continue
            if obj.get("type") == "system" and obj.get("session_id"):
                sid = str(obj["session_id"])
            elif obj.get("type") == "assistant":
                turns += 1
                for blk in (obj.get("message") or {}).get("content") or []:
                    if isinstance(blk, dict) and blk.get("type") == "text":
                        text = blk.get("text") or text
            elif obj.get("type") == "result":
                sid = str(obj.get("session_id") or sid)
                cost = obj.get("total_cost_usd", cost)
                text = obj.get("result") or text
        (self.dir / f"stderr-{index}.log").write_text(err or "")
        return {"ok": rc == 0, "rc": rc, "session_id": sid, "text": (text or "")[-4000:], "cost_usd": cost,
                "turns": turns, "elapsed_s": round(time.time() - t0, 1), "transcript": str(transcript)}

    # -- the loop ----------------------------------------------------------------
    def run(self) -> dict:
        st0 = self.status()
        sid = st0.get("session_id") or ""
        sittings = int(st0.get("sittings") or 0)
        self._write_status(state="running", started_ts=st0.get("started_ts") or time.time(), stop_reason="")
        total_cost = float(st0.get("cost_usd") or 0)
        for _ in range(self.max_sittings):
            if self.stop_requested():
                return self._write_status(state="stopped", stop_reason="STOP file")
            stage = self.stage_status()
            if self.done(stage):
                return self._write_status(state="done", stage=stage.get("stage"), sittings=sittings, cost_usd=total_cost)
            sittings += 1
            prompt = self.opening_prompt() if not sid else self.resume_prompt(stage, sittings)
            res = self.sit(prompt, resume=sid, index=sittings)
            sid = res.get("session_id") or sid
            total_cost += float(res.get("cost_usd") or 0)
            self._write_status(state="running", session_id=sid, sittings=sittings, cost_usd=total_cost,
                               last=dict(res, text=res.get("text", "")[-1500:]), stage=stage.get("stage"))
            if not res.get("ok") and not sid:
                return self._write_status(state="error", stop_reason=f"claude failed rc={res.get('rc')}: {res.get('error', '')}")
        stage = self.stage_status()
        if self.done(stage):
            return self._write_status(state="done", stage=stage.get("stage"), sittings=sittings, cost_usd=total_cost)
        return self._write_status(state="blocked", stage=stage.get("stage"), blocked_by=stage.get("blocked_by"),
                                  sittings=sittings, cost_usd=total_cost, stop_reason="max_sittings")
