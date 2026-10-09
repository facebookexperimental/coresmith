# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Native-session runners: the agentic CLI one cluster-worker turn runs on
(the Architect is outside the engine and never runs on these).

``ArchitectSession.sit`` is runner-neutral: it writes the prompt and the
system prompt, hands both to a runner and books what comes back. A runner runs
ONE turn of a resumable session in the project root with file tools + a shell,
streams the CLI's JSON events into the transcript file and returns
``{ok, rc, session_id, text, cost_usd, cost_cumulative, turns, tokens, ...}``.
``CORESMITH_ARCHITECT_PROVIDER`` / ``CORESMITH_ARCHITECT_MODEL`` are the
historical names of the native-session runner/model selectors.

* ``claude`` -- ``claude -p --output-format stream-json`` (``--resume <sid>``);
  ``total_cost_usd`` is CUMULATIVE for the session (the loop books deltas).
* ``opencode`` -- ``opencode --pure run --format json --thinking --auto`` with
  the model from ``CORESMITH_ARCHITECT_MODEL`` / ``CORESMITH_OPENCODE_MODEL`` /
  ``CORESMITH_MODEL`` (OpenRouter GLM, Kimi, Muse Spark ...), continued with
  ``--session <id>``. The system prompt is appended the way ``claude
  --append-system-prompt-file`` appends it: as an ``instructions`` file in the
  inline config (``OPENCODE_CONFIG_CONTENT``), which also defines the
  ``coresmith-sitting`` agent (``steps`` = the turn cap, every tool allowed,
  the interactive ``question`` / plan tools denied). Cost is the per-step
  ``cost`` of the turn (NOT cumulative); a provider that reports none leaves
  cost 0 with the tokens recorded and a ``cost_note``. The Muse Spark endpoint
  is registered exactly like the engine's own opencode calls
  (``coresmith_llm._inject_muse_spark_provider``, ``META_MODEL_API_KEY``).

* ``codex`` -- ``codex exec --json``, with the configured model and sandbox.

Runner selection (:func:`resolve_runner`): ``CORESMITH_ARCHITECT_PROVIDER``
(``claude`` | ``opencode`` | ``codex``), else the engine's resolved provider.
Unsupported bindings are refused before an invocation starts.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

RUNNERS = ("claude", "opencode", "codex")
_ALIASES = {"codex": "codex", "codex_cli": "codex", "openai": "codex",
            "claude": "claude", "claude_cli": "claude", "anthropic": "claude",
            "opencode": "opencode", "opencode_cli": "opencode", "openrouter": "opencode",
            "muse": "opencode", "glm": "opencode"}
OPENCODE_AGENT = "coresmith-sitting"
DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"


def resolve_runner(explicit: str | None = None) -> str:
    """``explicit`` > ``CORESMITH_ARCHITECT_PROVIDER`` > ``CORESMITH_LLM_PROVIDER``."""
    for raw in (explicit, os.environ.get("CORESMITH_ARCHITECT_PROVIDER")):
        v = str(raw or "").strip().lower()
        if v:
            if v not in _ALIASES:
                raise ValueError(f"unsupported architect runner {raw!r}: use one of {', '.join(RUNNERS)}")
            return _ALIASES[v]
    from orchestrator.langchain.agents.coresmith_llm import _detect_provider
    provider = _detect_provider()
    if provider not in _ALIASES:
        raise ValueError(f"unsupported architect provider {provider!r}; bind one of {RUNNERS}")
    return _ALIASES[provider]


def resolve_model(runner: str, explicit: str | None = None) -> str:
    """The invocation's model. claude: ``CORESMITH_ARCHITECT_MODEL`` >
    ``CORESMITH_COORDINATOR_MODEL`` > ``CORESMITH_MODEL`` > Opus. opencode:
    ``CORESMITH_ARCHITECT_MODEL`` > ``CORESMITH_OPENCODE_MODEL`` >
    ``CORESMITH_MODEL`` > the ``CORESMITH_OPENCODE_ENDPOINT`` default, with the
    engine's tier aliases (``opus-5`` ...) mapped like its own opencode calls."""
    if runner not in RUNNERS:
        raise ValueError(f"unsupported architect runner {runner!r}")
    if runner == "codex":
        from orchestrator.langchain.agents import coresmith_llm as llm
        m = (explicit or os.environ.get("CORESMITH_ARCHITECT_MODEL") or "").strip()
        return llm._CODEX_MODEL_MAP.get(m, m) if m else llm._resolve_model("", "codex_cli")
    if runner == "opencode":
        from orchestrator.langchain.agents import coresmith_llm as llm
        m = (explicit or os.environ.get("CORESMITH_ARCHITECT_MODEL") or "").strip()
        if not m:
            return llm._resolve_model("", "opencode_cli")
        table, _ = llm._opencode_endpoint_models(llm._opencode_endpoint())
        return table.get(m, m)
    return (explicit or os.environ.get("CORESMITH_ARCHITECT_MODEL") or os.environ.get("CORESMITH_COORDINATOR_MODEL")
            or os.environ.get("CORESMITH_MODEL") or DEFAULT_CLAUDE_MODEL)


def _read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except OSError:
        return ""


def _run_cli(cmd: list[str], prompt: str, *, transcript: Path, cwd: Path, env: dict, timeout_s: int,
             mode: str = "w", group: bool = False) -> tuple[int, str]:
    from orchestrator.processes import run
    with transcript.open(mode) as out:
        try:
            result = run(cmd, input=prompt, stdout=out, stderr=subprocess.PIPE, text=True,
                         cwd=str(cwd), env=env, timeout=timeout_s)
            return result.returncode, result.stderr or ""
        except subprocess.TimeoutExpired:
            return 124, f"invocation timed out after {timeout_s}s; transcript: {transcript}"


class SittingRunner:
    """One turn of a resumable agentic session. Subclasses implement
    :meth:`run`; :meth:`start` / :meth:`resume` are the two shapes of it."""
    name = ""
    cost_cumulative = False

    def __init__(self, binary: str, model: str):
        self.binary = binary
        self.model = model

    def run(self, prompt: str, *, system_file: Path, resume: str, transcript: Path, cwd: Path, env: dict,
            max_turns: int, timeout_s: int) -> dict:
        raise NotImplementedError

    def start(self, prompt: str, system: str, **kw) -> tuple[str, str, float | None, int]:
        """A new session: ``(session_id, text, cost_usd, turns)``. ``kw``:
        ``transcript``, ``cwd``, ``env``, ``max_turns``, ``timeout_s`` and an
        optional ``system_file`` (default: ``system.md`` next to the transcript)."""
        sf = kw.pop("system_file", None) or Path(kw["transcript"]).with_name("system.md")
        Path(sf).write_text(system)
        r = self.run(prompt, system_file=Path(sf), resume="", **kw)
        return r["session_id"], r["text"], r["cost_usd"], r["turns"]

    def resume(self, session_id: str, prompt: str, **kw) -> tuple[str, str, float | None, int]:
        """Continue ``session_id``: ``(session_id, text, cost_usd, turns)``."""
        sf = kw.pop("system_file", None) or Path(kw["transcript"]).with_name("system.md")
        r = self.run(prompt, system_file=Path(sf), resume=session_id, **kw)
        return r["session_id"], r["text"], r["cost_usd"], r["turns"]


class ClaudeRunner(SittingRunner):
    name = "claude"
    cost_cumulative = True   # `claude --resume` reports the session's running total

    def run(self, prompt, *, system_file, resume, transcript, cwd, env, max_turns, timeout_s) -> dict:
        cmd = [self.binary, "-p", "--output-format", "stream-json", "--verbose", "--model", self.model,
               "--max-turns", str(max_turns), "--permission-mode", "auto",
               "--disallowedTools", "Monitor,ScheduleWakeup,EnterPlanMode",
               "--append-system-prompt-file", str(system_file)]
        if resume:
            cmd += ["--resume", resume]
        rc, err = _run_cli(cmd, prompt, transcript=transcript, cwd=cwd, env=env, timeout_s=timeout_s)
        sid, text, cost, turns = resume, "", None, 0
        result_error = ""
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
                if obj.get("is_error"):
                    result_error = str(obj.get("result") or obj.get("errors") or obj.get("subtype"))
                cost = obj.get("total_cost_usd", cost)
                text = obj.get("result") or text
        return {"ok": rc == 0 and not result_error, "error": result_error, "rc": rc, "session_id": sid, "text": text or "", "cost_usd": cost,
                "cost_cumulative": True, "turns": turns, "stderr": err}


class CodexRunner(SittingRunner):
    name = "codex"

    def run(self, prompt, *, system_file, resume, transcript, cwd, env, max_turns, timeout_s) -> dict:
        from orchestrator.langchain.agents import coresmith_llm as llm
        cmd = llm.ClaudeLLM._build_codex_cmd(
            self.binary, self.model, str(cwd), env.get("CORESMITH_CODEX_SANDBOX", "workspace-write"),
            resume_session_id=resume or None,
            supported_flags=llm._codex_resume_supported_flags(self.binary) if resume else None,
            project_root=str(cwd))
        # Codex has no separate system-file flag; keep the shared worker instructions explicit.
        rc, err = _run_cli(cmd, _read(system_file) + "\n\n" + prompt, transcript=transcript,
                           cwd=cwd, env=env, timeout_s=timeout_s)
        raw = _read(transcript)
        text, usage = llm._parse_codex_json(raw)
        events = []
        for line in raw.splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:
                continue
        errors = [e.get("error") or e.get("message") for e in events
                  if e.get("type") in ("error", "turn.failed")]
        return {"ok": rc == 0 and not errors, "rc": rc, "stderr": err,
                "error": json.dumps(errors) if errors else "",
                "session_id": usage.get("session_id", resume), "text": text,
                "tokens": usage, "cost_usd": None, "cost_cumulative": False,
                "turns": sum(e.get("type") == "turn.completed" for e in events),
                "timed_out": rc == 124}


def opencode_config(system_file: Path, max_turns: int, model: str, base: str = "") -> str:
    """The inline ``OPENCODE_CONFIG_CONTENT`` for one invocation: the system
    prompt as an appended ``instructions`` file, the worker agent (its
    historical name ``coresmith-sitting`` is the config key) with the
    turn cap and its permissions, merged over the operator's own inline config
    and, for a ``meta-model-api/…`` model, the Muse Spark provider."""
    from orchestrator.langchain.agents import coresmith_llm as llm
    try:
        cfg = json.loads(base or "{}")
    except ValueError as exc:
        raise ValueError("OPENCODE_CONFIG_CONTENT must be valid JSON") from exc
    if not isinstance(cfg, dict):
        raise ValueError("OPENCODE_CONFIG_CONTENT must contain a JSON object")
    instr = [i for i in (cfg.get("instructions") or []) if i != str(system_file)]
    cfg["instructions"] = [*instr, str(system_file)]
    agents = cfg.setdefault("agent", {})
    agents[OPENCODE_AGENT] = {
        "mode": "primary",
        "description": "coresmith worker: drives the run through the coresmith CLI",
        "steps": int(max_turns),
        # every tool (file edit, bash, ...) without prompting; nobody answers an
        # interactive question in a headless invocation, and there is no plan mode
        "permission": {"*": "allow", "question": "deny", "plan_enter": "deny", "plan_exit": "deny"},
    }
    out = json.dumps(cfg)
    if model.startswith(llm.MUSE_SPARK_PROVIDER_ID + "/"):
        out = llm._inject_muse_spark_provider(out, model)
    return out


class OpenCodeRunner(SittingRunner):
    name = "opencode"
    cost_cumulative = False   # step_finish carries each step's own cost

    def command(self, *, resume: str, cwd: Path) -> list[str]:
        from orchestrator.langchain.agents import coresmith_llm as llm
        cmd = [self.binary, "--pure", "run", "--format", "json", "--thinking", "--auto",
               "--model", self.model, "--agent", OPENCODE_AGENT, "--dir", str(cwd)]
        variant = llm._resolve_opencode_variant(self.model, "architect")
        if variant:
            cmd += ["--variant", variant]
        if resume:
            cmd += ["--session", resume]
        return cmd

    def run(self, prompt, *, system_file, resume, transcript, cwd, env, max_turns, timeout_s) -> dict:
        from orchestrator.langchain.agents import coresmith_llm as llm
        env = dict(env)
        if self.model.startswith(llm.MUSE_SPARK_PROVIDER_ID + "/") and not env.get(llm.MUSE_SPARK_API_KEY_ENV, "").strip():
            transcript.write_text("")
            return {"ok": False, "rc": -1, "session_id": resume, "text": "", "cost_usd": None, "cost_cumulative": False,
                    "turns": 0, "stderr": "", "error": f"{llm.MUSE_SPARK_API_KEY_ENV} is not set: OpenCode cannot "
                    f"authenticate against the Meta Model API ({llm.MUSE_SPARK_BASE_URL})"}
        env["OPENCODE_CONFIG_CONTENT"] = opencode_config(system_file, max_turns, self.model,
                                                         env.get("OPENCODE_CONFIG_CONTENT", ""))
        try:
            retries = max(0, int(os.environ.get("CORESMITH_OPENCODE_MAX_RETRIES", "1") or 1))
        except ValueError:
            retries = 1
        sid, deadline, mode, attempts, errs = resume, time.time() + timeout_s, "w", 0, []
        text, usage, turns = "", {}, 0
        while True:
            attempts += 1
            cmd = self.command(resume=sid, cwd=cwd)
            start_len = len(_read(transcript)) if mode == "a" else 0
            rc, err = _run_cli(cmd, prompt, transcript=transcript, cwd=cwd, env=env, group=True, mode=mode,
                               timeout_s=max(1, int(deadline - time.time())))
            errs.append(err)
            chunk = _read(transcript)[start_len:]
            t, u = llm._parse_opencode_json(chunk)
            for raw in chunk.splitlines():
                try:
                    obj = json.loads(raw)
                except ValueError:
                    continue
                sid = str(obj.get("sessionID") or (obj.get("part") or {}).get("sessionID") or sid)
                if obj.get("type") == "step_start":
                    turns += 1
            text = t or text
            for k, v in u.items():   # numbers summed over attempts, terminal fields from the last one
                if k not in ("finish_reason", "provider_error") and isinstance(v, (int, float)):
                    usage[k] = usage.get(k, 0) + v
            for k in ("finish_reason", "provider_error"):
                usage.pop(k, None)
                if u.get(k):
                    usage[k] = u[k]
            transient = (llm._is_transient_opencode_failure(rc, chunk, err)
                         or (rc == 0 and bool(u.get("provider_error"))
                             and llm._is_transient_opencode_failure(1, u["provider_error"], "")))
            if transient and attempts <= retries and time.time() < deadline - 30:
                # resume the captured session (or start over) after a dropped stream
                prompt = ("The previous turn was interrupted by a provider error. Continue where you left off."
                          if sid else prompt)
                mode = "a"
                time.sleep(min(2 ** attempts, 8))
                continue
            break
        cost = float(usage.get("total_cost_usd") or 0.0)
        tokens = {k: usage[k] for k in ("input_tokens", "output_tokens", "reasoning_output_tokens",
                                        "cache_read_input_tokens", "cache_creation_input_tokens", "total_tokens")
                  if k in usage}
        res = {"ok": rc == 0 and not usage.get("provider_error"), "rc": rc, "session_id": sid, "text": text or "",
               "cost_usd": cost, "cost_cumulative": False, "turns": turns, "tokens": tokens,
               "stderr": "\n".join(errs), "attempts": attempts}
        if usage.get("finish_reason"):
            res["finish_reason"] = usage["finish_reason"]
        if usage.get("provider_error"):
            res["error"] = f"opencode provider error: {usage['provider_error']}"
        if not cost and tokens.get("total_tokens"):
            res["cost_note"] = "the provider reported no cost; tokens recorded, cost booked as 0"
        return res


def find_binary(runner: str, explicit: str | None = None) -> str:
    if runner not in RUNNERS:
        raise ValueError(f"unsupported architect runner {runner!r}")
    if runner == "codex":
        from orchestrator.langchain.agents.coresmith_llm import _find_codex_binary
        return explicit or _find_codex_binary()
    if runner == "opencode":
        if explicit:
            return explicit
        try:
            from orchestrator.langchain.agents.coresmith_llm import _find_opencode_binary
            return _find_opencode_binary()
        except FileNotFoundError:
            return "opencode"
    return explicit or os.environ.get("CLAUDE_CLI_PATH") or shutil.which("claude") or "claude"


def make_runner(runner: str, model: str, binary: str | None = None) -> SittingRunner:
    if runner not in RUNNERS:
        raise ValueError(f"unsupported architect runner {runner!r}")
    if (runner == "claude" and (model.startswith("gpt-") or "/" in model)) or (runner == "codex" and model.startswith("claude-")):
        raise ValueError(f"AGENT_BINDING_INVALID: {runner} cannot run {model}")
    cls = {"opencode": OpenCodeRunner, "claude": ClaudeRunner, "codex": CodexRunner}[runner]
    return cls(find_binary(runner, binary), model)
