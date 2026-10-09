# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""coresmithd -- FastAPI daemon driving one coresmith pipeline run.

Replaces the old ``run_pipeline.py`` headless auto-approver. The daemon stays
alive serving HTTP requests; an outer agent (Claude on cron, the ``coresmith``
CLI, another script) drives it by calling /run/start, /run/state, /run/resume,
etc. There is no auto-approve loop -- every interrupt is surfaced through
GET /run/state and waits for an explicit POST /run/resume from the outer
agent.

One daemon per project_root. The daemon picks a free 127.0.0.1 port and
writes ``<project_root>/.coresmith/daemon.json`` with ``{port, pid,
started_at}`` so the CLI can discover it. The file is removed on clean
shutdown.

Start with::

    CORESMITH_PROJECT_ROOT=<dir> venv/bin/python -m orchestrator.daemon.server

or via the ``coresmith daemon start`` CLI wrapper.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import functools
import json
import logging
import os
import signal
import socket
import sys
import time
from pathlib import Path
from typing import Any

# Resolve project root + telemetry before importing any graph code.
_PROJECT_ROOT = os.environ.get(
    "CORESMITH_PROJECT_ROOT",
    str(Path(__file__).resolve().parent.parent.parent),
)
os.environ["CORESMITH_PROJECT_ROOT"] = _PROJECT_ROOT

# The blocks override that was in the daemon's environment at startup. A
# /run/start with blocks_file sets CORESMITH_BLOCKS_FILE process-wide; a later
# run without blocks_file must fall back to this, not inherit the stale one.
_ENV_BLOCKS_FILE = os.environ.get("CORESMITH_BLOCKS_FILE")

# A-Fix 1: seed profile flag defaults BEFORE importing graph code -- the
# pipeline builder reads gate-enable helpers (e.g. block_goldens_enabled) at
# build time, so the profile must be applied first.
from orchestrator.profile import apply as _apply_profile

_apply_profile()

from orchestrator.telemetry import init_telemetry

init_telemetry(_PROJECT_ROOT)

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from orchestrator import module_build as MB
from orchestrator.graph_lifecycle import GraphLifecycle
from orchestrator.run_env import apply_persisted_env, format_override_notice
from orchestrator.state_store import builds as B

log = logging.getLogger("coresmithd")
log.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Driver-liveness watch (Section 7b): a run parked at an interrupt with no
# outer-agent /run/resume for too long is stalled -- WARN (repeating) + drop a
# STALLED_INTERRUPT marker so a human notices, without any paging infra.
# ---------------------------------------------------------------------------

_last_resume_ts: float = time.time()   # bumped on every /run/start + /run/resume
_STALL_THRESHOLD_S = float(os.environ.get("CORESMITH_INTERRUPT_STALL_S", "1800"))
_STALL_POLL_S = float(os.environ.get("CORESMITH_INTERRUPT_STALL_POLL_S", "300"))

# ---------------------------------------------------------------------------
# When was each interrupt RAISED? (first observed pending, which is the closest
# thing to a raise time available -- LangGraph interrupts carry no timestamp.)
# ---------------------------------------------------------------------------
# Both staleness answers used to be keyed to the wrong clock:
#
#   * ``_driver_liveness_watch`` measured "idle" as time since the last
#     /run/resume. On a HANDS-OFF run there is no resume after /run/start, so
#     the stall clock is already an hour old the moment the first interrupt is
#     raised -- the daemon logged STALLED_INTERRUPT for an interrupt that was
#     15 minutes old and dropped the marker file with idle_seconds=4488.
#   * ``_shape_state`` labelled an interrupt "stale_suspected" purely because
#     its block appears in the append-only ``completed_blocks``. In a two-pass
#     run EVERY block completes pass 1, so EVERY pass-2 interrupt is born
#     stale: a parked graded run reported ``stale_suspected: true`` and
#     ``live_interrupt_count: 0`` on a freshly raised, entirely live
#     ``contract_conformance_unrepairable`` -- and an automated consumer
#     reading ``live_interrupt_count == 0`` concludes there is nothing to act
#     on.
#
# A just-raised interrupt must read LIVE. So both are keyed to the interrupt
# itself: the stall clock runs from when THIS interrupt was raised, and the
# leftover suspicion needs the block's completion to be NEWER than the
# interrupt (i.e. the graph really did move past it) rather than merely present.
_interrupt_first_seen: dict[str, float] = {}


def _note_interrupts_seen(ids: set[str], now: float | None = None) -> None:
    """Record first-observation time for each pending interrupt id.

    Ids that are no longer pending are forgotten, so an id LangGraph reuses
    after a resume is timed from its new appearance rather than its old one.
    """
    t = time.time() if now is None else now
    for i in ids:
        _interrupt_first_seen.setdefault(i, t)
    for gone in set(_interrupt_first_seen) - ids:
        _interrupt_first_seen.pop(gone, None)


def _interrupt_raised_ts(intr_id: str, now: float | None = None) -> float:
    """When this interrupt was first seen pending. Unknown ids read as NOW --
    an interrupt we have never observed before is, by definition, new."""
    return _interrupt_first_seen.get(
        intr_id, time.time() if now is None else now)


def _completion_times(completed: list) -> dict:
    """block -> the time of its LATEST completion event (0.0 when unstamped).

    ``block_done_node`` stamps ``completed_at``; checkpoints written before that
    have none, and an unstamped completion is treated as NO EVIDENCE rather
    than as proof an interrupt is a leftover.
    """
    out: dict = {}
    for b in completed or []:
        if not isinstance(b, dict):
            continue
        name = b.get("name")
        if not name:
            continue
        try:
            ts = float(b.get("completed_at") or 0.0)
        except (TypeError, ValueError):
            ts = 0.0
        out[name] = max(ts, out.get(name, 0.0))
    return out

# ---------------------------------------------------------------------------
# D5: interrupt ids this daemon has already forwarded a resume for.
# ---------------------------------------------------------------------------
# `aget_state` reads the CHECKPOINT. Between a consumed `/run/resume` and the
# graph's next checkpoint write, that snapshot still carries the interrupt the
# resume just answered -- so `/run/state` reported `pending_interrupt_count: 1`
# while `status: running`. An outer agent polling state sees a pending interrupt
# on a run that is actively working and resumes it AGAIN. Double-resume is not
# harmless: the second decision lands on whatever the graph parks at next.
#
# The remedy has to be evidence-based, not a blanket "hide interrupts while
# running" -- PR #73 landed specifically because hiding live interrupts cost ~70
# minutes of a parked run. So: record the exact ids we forwarded a resume for,
# and discount ONLY those, and ONLY while the runner task is actually in flight.
# The moment the run stops (parked, done, error), the set is cleared and every
# count is raw again.
_consumed_interrupt_ids: set[str] = set()


def _pipeline_task_in_flight() -> bool:
    """True while the runner task is actually executing the graph."""
    return _pipeline.task is not None and not _pipeline.task.done()


def _consumed_now() -> set[str]:
    """Interrupt ids a live resume has already answered.

    Empty whenever the runner task is not in flight -- and the set is CLEARED
    then too, so a run that parks again on a same-id interrupt is reported at
    full strength. The suppression can only ever last as long as one in-flight
    resume.
    """
    global _consumed_interrupt_ids
    if not _pipeline_task_in_flight():
        if _consumed_interrupt_ids:
            _consumed_interrupt_ids = set()
        return set()
    return set(_consumed_interrupt_ids)


def _bind_interrupt_rows(pairs) -> None:
    """Bind LangGraph ``Interrupt.id`` to the coresmith row it belongs to
    (the payload carries ``interrupt_id`` when the park went through the
    interrupts table)."""
    try:
        db = _project_db()
    except Exception:  # noqa: BLE001
        return
    for lg_id, value in pairs:
        cid = value.get("interrupt_id") if isinstance(value, dict) else None
        if cid:
            with contextlib.suppress(Exception):
                db.bind_lg_id(cid, lg_id)


async def _count_pending_interrupts() -> int | None:
    """Count parked interrupts. ``None`` means COULD NOT DETERMINE, not zero.

    This previously returned 0 on any exception, so a failed graph/state probe
    was indistinguishable from "nothing is waiting". Observed consequence: the
    graph sat on ``[HUMAN] Intervention needed`` while ``/run/state`` reported
    ``pending_interrupt_count: 0`` and ``interrupts: []``; a subsequent
    ``POST /run/resume`` then returned ``{"resumed": true, "interrupts": 1}``.
    An outer agent polling state concludes there is nothing to drive and the
    run parks indefinitely -- in one session that cost ~70 minutes before the
    daemon's own ``STALLED_INTERRUPT`` log line gave it away.

    Absence of evidence is not evidence of absence: callers must treat ``None``
    as "unknown, go look" rather than as "clear".
    """
    try:
        await _pipeline.ensure_graph()
        snap = await _pipeline.graph.aget_state(
            {"configurable": {"thread_id": _pipeline.thread_id}}
        )
        consumed = _consumed_now()
        n = 0
        ids: set[str] = set()
        pairs = []
        if snap and snap.tasks:
            for t in snap.tasks:
                for i in t.interrupts:
                    ids.add(i.id)
                    pairs.append((i.id, i.value))
                    if i.id not in consumed:
                        n += 1
        _bind_interrupt_rows(pairs)
        # Stamp first-seen here too: this poll runs every _STALL_POLL_S whether
        # or not anyone calls /run/state, so an unattended run still learns when
        # its interrupt was raised.
        _note_interrupts_seen(ids)
        return n
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "interrupt probe FAILED (%s: %s) -- reporting UNKNOWN, not 0. "
            "Treat as 'go look', never as 'clear'.", type(exc).__name__, exc)
        return None


async def _driver_liveness_watch() -> None:
    """Poll for a stalled interrupt (pending > 0 with no resume for >30 min)."""
    marker = Path(_PROJECT_ROOT) / "STALLED_INTERRUPT"
    while True:
        try:
            await asyncio.sleep(_STALL_POLL_S)
            pending = await _count_pending_interrupts()
            if pending is None:
                log.warning(
                    "STALLED_INTERRUPT probe UNKNOWN (project_root=%s): the "
                    "interrupt count could not be determined. Not treating as "
                    "zero -- inspect the run.", _PROJECT_ROOT)
                continue
            if pending > 0:
                # Measure how long THIS interrupt has gone unanswered, not how
                # long since the last resume. On a hands-off run the only
                # resume is /run/start, so the old clock declared every
                # interrupt stalled the moment it was raised (observed:
                # idle_seconds=4488 on an interrupt 15 minutes old).
                now = time.time()
                oldest = min(
                    (_interrupt_first_seen[i] for i in _interrupt_first_seen),
                    default=now)
                # A resume after the interrupt appeared also restarts the clock:
                # the driver is demonstrably alive.
                unanswered_since = max(oldest, _last_resume_ts)
                idle = now - unanswered_since
                if idle >= _STALL_THRESHOLD_S:
                    log.warning(
                        "STALLED_INTERRUPT: %d pending interrupt(s) unanswered "
                        "for %.0f min (project_root=%s). The outer driver may "
                        "be dead -- resume or restart it.",
                        pending, idle / 60.0, _PROJECT_ROOT,
                    )
                    _stall = {
                        "pending_interrupt_count": pending,
                        "idle_seconds": round(idle),
                        "interrupt_raised_ts": oldest,
                        "last_resume_ts": _last_resume_ts,
                        "noted_at": now,
                    }
                    # C1: the fact lives in run_flags; the marker file is a view.
                    with contextlib.suppress(Exception):
                        _project_db().set_flag("stalled_interrupt", _stall)
                    try:
                        marker.write_text(json.dumps(_stall, indent=2))
                    except OSError:
                        pass
            else:
                # cleared -> remove any stale marker
                with contextlib.suppress(Exception):
                    _project_db().clear_flag("stalled_interrupt")
                with contextlib.suppress(OSError):
                    if marker.exists():
                        marker.unlink()
        except asyncio.CancelledError:
            break
        except Exception:  # noqa: BLE001 - liveness watch must never crash
            continue


def _backend_lifecycle():
    """The backend GraphLifecycle when a backend run can hold a park, else
    None. The MCP module (heavy) is only imported when it is already loaded
    or the interrupts table records a pending backend park (a daemon restart
    while the backend was parked)."""
    mod = sys.modules.get("orchestrator.mcp_server")
    if mod is None:
        try:
            pending = _project_db().interrupts(status="pending", graph="backend")
        except Exception:  # noqa: BLE001
            pending = []
        if not pending:
            return None
        try:
            mod = _backend_handle()
        except Exception:  # noqa: BLE001
            return None
    handle = getattr(mod, "_backend", None)
    if handle is None or not getattr(handle, "thread_id", ""):
        return None
    root = getattr(handle, "project_root", "") or ""
    if root and Path(str(root)).resolve() != Path(_PROJECT_ROOT).resolve():
        return None                     # another project's backend: never ours to answer
    return handle


async def _live_backend_parks() -> list[dict]:
    """Live parks of the backend graph (``graph: backend``):
    ``[{lg_id, interrupt_id, payload, graph}]`` (what ``backend resume``
    validates an answer against)."""
    handle = _backend_lifecycle()
    if handle is None:
        return []
    if handle.task is not None and not handle.task.done():
        return []                       # running: nothing is parked
    try:
        await handle.ensure_graph()
        snap = await handle.graph.aget_state({"configurable": {"thread_id": handle.thread_id}})
    except Exception:  # noqa: BLE001
        return []
    parks, pairs = [], []
    for t in (snap.tasks if snap and snap.tasks else []):
        for i in t.interrupts:
            val = i.value if isinstance(i.value, dict) else {"value": i.value}
            pairs.append((i.id, i.value))
            parks.append({"lg_id": i.id, "interrupt_id": str(val.get("interrupt_id") or ""),
                          "payload": {**val, "graph": "backend"}, "graph": "backend"})
    _bind_interrupt_rows(pairs)
    _note_interrupts_seen({i for i, _ in pairs})
    return parks


# ---------------------------------------------------------------------------
# Frontend -> backend handoff (opt-in): CORESMITH_AUTO_BACKEND=1
# ---------------------------------------------------------------------------
# The endpoint below is the SHAPE; this watch is what makes the handoff
# autonomous. Without it, `pipeline_done` is a state field nobody acts on: the
# chip_top gate-sim -- the only step that simulates the artifact that becomes
# silicon -- has only ever been reached by a human or a hand-written driver.
#
# Opt-in and one-shot: it fires at most once per daemon process, only when the
# frontend genuinely finished (pipeline_done AND no parked interrupt AND the
# pipeline task is not running), and it stops the backend after flat synthesis +
# the gate-sim verdict. It never runs P&R/DRC/LVS; that stays an explicit ask.

_AUTO_BACKEND_POLL_S = float(os.environ.get("CORESMITH_AUTO_BACKEND_POLL_S", "30"))
_auto_backend_fired = False


def _daemon_log(level: str, msg: str, *args, **kw) -> None:
    """Log somewhere a human will actually see it.

    The module's ``coresmithd`` logger has no handler: under uvicorn only the
    ``uvicorn.*`` loggers are configured, so everything sent to ``log`` goes
    nowhere -- the same trap ``_lifespan`` already documents for the profile-seed
    line. An autonomy step that starts real EDA by itself must not announce
    itself into a black hole, so route through ``uvicorn.error`` (which lands in
    daemon.log) and keep ``log`` as the fallback for a non-uvicorn host.
    """
    for logger in (logging.getLogger("uvicorn.error"), log):
        try:
            getattr(logger, level)(msg, *args, **kw)
            return
        except Exception:  # noqa: BLE001
            continue


def _auto_backend_enabled() -> bool:
    """True when the daemon should enter the backend itself on pipeline_done.

    Default OFF: entering the backend spends real EDA time, so it is an explicit
    opt-in rather than something a run acquires by upgrading the engine.
    """
    return (os.environ.get("CORESMITH_AUTO_BACKEND", "0") or "0").strip().lower() \
        not in ("", "0", "false", "no", "off")


async def _frontend_is_done() -> bool:
    """The frontend finished and is not waiting on anybody.

    Deliberately conservative -- all four must hold. A parked interrupt is NOT
    'done' even with pipeline_done set, because the run is waiting on a decision
    that could still change the RTL the backend would synthesize.
    """
    if _pipeline.task is not None and not _pipeline.task.done():
        return False
    try:
        await _pipeline.ensure_graph()
        snap = await _pipeline.graph.aget_state(
            {"configurable": {"thread_id": _pipeline.thread_id}}
        )
    except Exception:  # noqa: BLE001
        return False
    if not snap or not snap.values:
        return False
    if not snap.values.get("pipeline_done"):
        return False
    if snap.tasks and any(t.interrupts for t in snap.tasks):
        return False
    return True


async def _auto_backend_watch() -> None:
    """Poll for a finished frontend and hand off to the backend, once."""
    global _auto_backend_fired
    if not _auto_backend_enabled():
        return
    _daemon_log(
        "warning",
        "CORESMITH_AUTO_BACKEND=1: this daemon will enter flat synthesis + the "
        "chip_top gate-sim by itself when the frontend reaches pipeline_done "
        "(P&R/DRC/LVS still require an explicit `backend start --full`).")
    while not _auto_backend_fired:
        try:
            await asyncio.sleep(_AUTO_BACKEND_POLL_S)
            if not await _frontend_is_done():
                continue
            _auto_backend_fired = True     # one-shot, even if the launch fails
            _daemon_log(
                "warning",
                "AUTO-BACKEND: frontend reached pipeline_done with no parked "
                "interrupt -- entering flat synthesis + chip_top gate-sim.")
            from orchestrator import mcp_server as _mcp
            result = await _mcp.launch_backend(stop_after_gate_sim=True)
            try:
                from orchestrator.langgraph.event_stream import write_graph_event
                write_graph_event(_PROJECT_ROOT, "daemon", "auto_backend_start",
                                  {k: v for k, v in result.items()
                                   if k != "block_names"})
            except Exception:  # noqa: BLE001
                pass
            if result.get("error"):
                _daemon_log("error", "AUTO-BACKEND: launch refused: %s", result)
            else:
                _daemon_log("warning",
                            "AUTO-BACKEND: backend running (thread_id=%s)",
                            result.get("thread_id"))
        except asyncio.CancelledError:
            break
        except Exception:  # noqa: BLE001 - a handoff watch must never crash
            _daemon_log("warning", "AUTO-BACKEND: watch iteration failed",
                        exc_info=True)
            continue


# ---------------------------------------------------------------------------
# Lifecycle wiring
# ---------------------------------------------------------------------------

_pipeline = GraphLifecycle(
    name="pipeline",
    checkpoint_db=os.path.join(_PROJECT_ROOT, ".coresmith", "pipeline_checkpoint.db"),
    builder_fn_path="orchestrator.langgraph.pipeline_graph",
    builder_fn_name="build_pipeline_graph",
    project_root=_PROJECT_ROOT,
)

_architecture = GraphLifecycle(
    name="architecture",
    checkpoint_db=os.path.join(_PROJECT_ROOT, ".coresmith", "architecture_checkpoint.db"),
    builder_fn_path="orchestrator.langgraph.architecture_graph",
    builder_fn_name="build_architecture_graph",
    project_root=_PROJECT_ROOT,
)

# One recorded module build at a time: the SAME block subgraph the pipeline
# fans out, compiled on its own and driven on a persistent checkpoint whose
# thread is the build id (``coresmith build module``; see module_build.py).
_build = GraphLifecycle(
    name="build",
    checkpoint_db=MB.checkpoint_db_path(_PROJECT_ROOT),
    builder_fn_path="orchestrator.langgraph.pipeline_graph",
    builder_fn_name="build_block_graph",
    project_root=_PROJECT_ROOT,
)


# ---------------------------------------------------------------------------
# Persisted run env (<project_root>/.coresmith/env)
# ---------------------------------------------------------------------------
# The daemon reads its environment ONCE, at process start. `.coresmith/env` is
# the run's frozen config and operators edit it MID-RUN -- a chip lead appended
# several keys to a parked run -- but those edits were invisible to every later
# /run/restart-node and resume, so a knob the operator had already "set" was
# still unset in the graph until someone bounced the daemon. Re-apply the file
# before launching graph work, with the SAME persisted-wins semantics the CLI
# uses (one implementation: orchestrator.run_env).

# The CLI role scopes a SHELL (architect / worker / watchdog). The daemon is
# the engine: it must never run -- or hand its children -- a role, or a daemon
# started from a watchdog shell refuses its own synthesis agent
# (``ROLE_FORBIDDEN watchdog may not run tool``, MCU+FFT run 3).
_DAEMON_SCRUBBED_ENV = ("CORESMITH_ROLE", "CORESMITH_ACTOR")


def _clear_role_env(where: str) -> list[str]:
    """Drop ``CORESMITH_ROLE`` / ``CORESMITH_ACTOR`` from this process (and so
    from every child it spawns). ``CORESMITH_DAEMON_KEEP_ROLE=1`` keeps them
    (old behaviour)."""
    if os.environ.get("CORESMITH_DAEMON_KEEP_ROLE", "").strip().lower() in {"1", "true", "yes", "on"}:
        return []
    dropped = [k for k in _DAEMON_SCRUBBED_ENV if os.environ.pop(k, None) is not None]
    if dropped:
        _daemon_log("warning", "%s: cleared %s for the daemon and its children (a CLI role scopes a "
                    "shell, never the engine)", where, dropped)
    return dropped


def _apply_run_env(where: str) -> list[str]:
    """Re-read ``.coresmith/env`` into ``os.environ``; return the changed keys.

    The persisted file wins over whatever this process currently holds --
    including values the daemon itself set earlier in the run. Best-effort: an
    absent/unreadable file is a no-op and a failure never breaks the handler.
    """
    try:
        changes = apply_persisted_env(_PROJECT_ROOT)
    except Exception:  # noqa: BLE001 -- an env refresh must never fail a request
        _daemon_log("warning", "%s: persisted env refresh failed", where,
                    exc_info=True)
        _clear_role_env(where)
        return []
    _clear_role_env(where)
    if changes:
        # _daemon_log, not log: under uvicorn the `coresmithd` logger has no
        # handler, and a silent env swap is exactly what this fix is about.
        _daemon_log(
            "warning", "%s: reloaded .coresmith/env -- applied %s",
            where, [k for k, _, _ in changes],
        )
        notice = format_override_notice(changes)
        if notice:
            _daemon_log("warning", "%s: %s", where, notice)
    return [k for k, _, _ in changes]


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class StartRequest(BaseModel):
    max_attempts: int = 5
    target_clock_mhz: float | None = None   # None: inputs/task.yaml target_clock_mhz, else 50
    blocks_file: str = ""
    force: bool = False


class ResumeRequest(BaseModel):
    action: str = "approve"
    feedback: str = ""
    rtl_fix_description: str = ""
    block_actions: dict | None = None
    rationale: str = ""
    # C1-3: answer ONE parked branch (coresmith ``interrupt_id`` from
    # /run/interrupts or the interrupt payload). While the runner is in
    # flight the answer is queued in the interrupts table and applied at the
    # next superstep boundary (202) instead of being rejected (409).
    interrupt_id: str | None = None
    # Who answered (``cli`` by default; the Architect may pass any identity):
    # stored in ``decisions.actor`` and ``interrupts.resolved_by``.
    actor: str = "cli"
    # A revise's target blocks (the graph reads ``affected_blocks``).
    affected_blocks: list[str] | None = None


class RulingRequest(BaseModel):
    scope: str
    text: str
    rationale: str = ""
    source: str = "human"
    question_ref: str | None = None
    supersedes_id: int | None = None


class RevokeRulingRequest(BaseModel):
    reason: str = ""


class RestartBlockRequest(BaseModel):
    block_name: str
    from_node: str = "generate_rtl"
    uarch_feedback: str = ""
    max_attempts: int = 3


class RestartNodeRequest(BaseModel):
    node: str
    refresh_sidecars: bool = False


class BuildModuleRequest(BaseModel):
    model_config = {"extra": "forbid"}

    module: str
    seed_rtl: str = ""            # an existing source of the bound target: the starting implementation
    max_attempts: int = 3         # implementation attempts: misses/failures go back to the worker
    target_clock_mhz: float | None = None
    uarch_feedback: str = ""      # legacy clients receive SPEC_REVISION_REQUIRED


class BuildResumeRequest(BaseModel):
    build_id: str
    action: str = "approve"
    feedback: str = ""
    rtl_fix_description: str = ""
    rationale: str = ""
    interrupt_id: str | None = None
    actor: str = "cli"


class BuildAbortRequest(BaseModel):
    build_id: str
    reason: str = ""
    actor: str = "cli"


class ReviseBlocksRequest(BaseModel):
    blocks: list[str]
    feedback: str = ""


class ArchStartRequest(BaseModel):
    requirements: str = ""
    requirements_file: str = ""
    target_clock_mhz: float = 50.0
    pdk_config_path: str = ""
    max_rounds: int = 3


class ArchResumeRequest(BaseModel):
    action: str = "continue"
    feedback: str = ""
    rationale: str = ""


class BackendStartRequest(BaseModel):
    max_attempts: int = 3
    target_clock_mhz: float = 50.0
    # Default STOPS after flat synthesis + the chip_top gate-sim verdict.
    # P&R/DRC/LVS is hours of EDA that must be asked for explicitly.
    full: bool = False


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@contextlib.asynccontextmanager
async def _lifespan(_app: FastAPI):
    # This process runs the EDA tools: it identifies them by its own
    # resolution and records the resolved paths every other process (the
    # Architect's CLI) identifies a build's tools by (state_store/builds.py).
    os.environ["CORESMITH_TOOL_AUTHORITY"] = "1"
    # Defect 3 (rung1): the import-time _apply_profile() above logs its
    # profile-seed line BEFORE uvicorn configures logging, so it goes nowhere.
    # Re-emit it now (startup runs after uvicorn's logging is up) through the
    # uvicorn logger so it lands in daemon.log, plus a cheap pipeline_events
    # breadcrumb -- making profile seeding observable without new plumbing.
    try:
        from orchestrator import profile as _profile
        _profile.log_status(logging.getLogger("uvicorn.error"))
        try:
            from orchestrator.langgraph.event_stream import write_graph_event
            write_graph_event(
                _PROJECT_ROOT, "daemon", "profile_seeded",
                {"summary": _profile.status_line()},
            )
        except Exception:  # noqa: BLE001 -- observability must never fail startup
            pass
    except Exception:  # noqa: BLE001
        pass
    # Section 7b: start the driver-liveness watch (cheap, cancelled on shutdown).
    _watch_task = asyncio.create_task(_driver_liveness_watch())
    # Frontend -> backend handoff. Returns immediately when the opt-in is off.
    _auto_backend_task = asyncio.create_task(_auto_backend_watch())
    _lease_task = asyncio.create_task(_daemon_lease_renew())
    # No Architect is launched or resumed by the daemon: parks wait for the
    # Architect's own `coresmith resume`.
    from orchestrator.daemon.supervisor import watch
    _supervisor_task = asyncio.create_task(watch(sys.modules[__name__]))
    # Recorded builds the previous process left non-terminal are re-read from
    # their persistent checkpoints (parked / finished while nobody watched).
    try:
        recovered = await MB.recover_builds(_project_db(), _PROJECT_ROOT, _build)
        if recovered:
            _daemon_log("warning", "build recovery: %s", recovered)
    except Exception:  # noqa: BLE001 - recovery must never block startup
        _daemon_log("warning", "build recovery failed", exc_info=True)
    try:
        yield
    finally:
        for _t in (_watch_task, _auto_backend_task, _lease_task, _supervisor_task):
            _t.cancel()
            with contextlib.suppress(Exception):
                await _t


app = FastAPI(title="coresmithd", version="0.1", lifespan=_lifespan)


@app.get("/supervisor/status")
async def supervisor_status():
    return _project_db().get_flag("supervisor_status") or {"state": "disabled"}


@app.post("/supervisor/retry")
async def supervisor_retry():
    """Explicit infrastructure retry after the operator repairs the cause."""
    db = _project_db()
    db.clear_flag("agent_failure")
    root = Path(_PROJECT_ROOT) / ".coresmith"
    for path in root.glob("clusters/*/status.json"):
        if path.exists():
            status = json.loads(path.read_text())
            if status.get("state") in ("provider_blocked", "tool_failed"):
                status.update(state="retry_requested", previous_failure=status.get("stop_reason"))
                path.write_text(json.dumps(status, indent=2))
    pipeline = await run_state()
    if pipeline.get("values_empty"):
        db.clear_flag("supervisor_launch")
    return await supervisor_check()


@app.post("/supervisor/check")
async def supervisor_check():
    from orchestrator.daemon.supervisor import tick
    return await tick(sys.modules[__name__])


@app.get("/healthz")
async def healthz():
    return {
        "ok": True,
        "project_root": _PROJECT_ROOT,
        "status": _pipeline.status,
        "build_status": _build.status,
        "build_thread_id": _build.thread_id if _build.thread_id != "build" else None,
        "pid": os.getpid(),
    }


# ---------------------------------------------------------------------------
# Recorded module builds (coresmith build module|state|resume|pause)
# ---------------------------------------------------------------------------

def _refusal_response(refusal: MB.BuildRefusal) -> JSONResponse:
    return JSONResponse(status_code=refusal.http_status, content=refusal.to_json())


def _backend_lifecycle():
    """The backend GraphLifecycle when the MCP module (which owns it) has
    been imported into this process; None otherwise (no backend can be
    running here without it)."""
    mod = sys.modules.get("orchestrator.mcp_server")
    return getattr(mod, "_backend", None) if mod is not None else None


def _busy_lifecycles() -> list:
    """The graphs a module build must not interleave with."""
    out = [_pipeline]
    backend = _backend_lifecycle()
    if backend is not None:
        out.append(backend)
    return out


def _build_running() -> bool:
    return _build.task is not None and not _build.task.done()


def _refuse_if_build_running(what: str) -> None:
    """The converse guard: the pipeline and the backend never start, resume
    or re-enter while a module build is driving the workspace."""
    if _build_running():
        raise HTTPException(409, f"a module build is running (thread {_build.thread_id}); {what} shares the "
                                 "workspace -- wait for it to park or coresmith build pause")


def _serialized(what: str, graph: str | None = None):
    """Run a launching route inside the one launch section every transport
    shares (``module_build.launch_section``): the operation lock in this
    process and the ``graph_launch`` lease across processes, so its conflict
    checks and the start they guard are one critical section with every
    build start/resume and every other launch; a graph running in another
    process refuses it. The section is re-entrant for the same task, so a
    route that delegates to the MCP implementation (the backend routes) is
    one operation. When the route started ``graph`` (``pipeline`` /
    ``backend``), its lifecycle's task holds the ``graph:<graph>`` lease
    until it ends, so other processes see it."""
    def deco(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            db = _project_db()
            try:
                async with MB.launch_section(db, what=what):
                    if not graph:
                        return await fn(*args, **kwargs)
                    getter = (lambda: _pipeline) if graph == "pipeline" else _backend_lifecycle
                    async with MB.graph_ownership(db, graph, getter, meta={"what": what}):
                        return await fn(*args, **kwargs)
            except MB.BuildRefusal as refusal:
                return _refusal_response(refusal)
        return wrapper
    return deco


async def _watch_build(build_id: str) -> None:
    """When the build task ends, classify the non-success terminal states
    (``block_done_node`` itself commits ``completed``) and merge a completed
    build into the pipeline checkpoint."""
    outcome = await MB.watch_build(_project_db(), _PROJECT_ROOT, _build, build_id, pipeline=_pipeline)
    _daemon_log("info", "build %s -> %s", build_id, outcome)


async def _start_module_build(req: BuildModuleRequest, *, entry: str):
    """HTTP transport of :func:`orchestrator.module_build.start_build`."""
    try:
        out = await MB.start_build(_project_db(), _PROJECT_ROOT, _build, module=req.module, entry=entry,
                                   busy=_busy_lifecycles(), seed_rtl=req.seed_rtl, max_attempts=req.max_attempts,
                                   target_clock_mhz=req.target_clock_mhz, uarch_feedback=req.uarch_feedback,
                                   apply_env=lambda: _apply_run_env("build/module"), preflight=_preflight_or_400)
    except MB.BuildRefusal as refusal:
        return _refusal_response(refusal)
    asyncio.create_task(_watch_build(out["build_id"]))
    return out


@app.post("/build/module")
async def build_module(req: BuildModuleRequest):
    """``coresmith build module <name>``: one recorded build of one module
    through the existing block subgraph on a persistent checkpoint."""
    return await _start_module_build(req, entry="build_module")


@app.get("/build/state")
async def build_state(build_id: str = ""):
    db = _project_db()
    if not build_id:
        rows = B.builds_for(db, limit=1)
        if not rows:
            raise HTTPException(404, "no recorded builds")
        build_id = rows[0]["id"]
    try:
        out = await MB.state_with_parks(db, _PROJECT_ROOT, _build, build_id)
    except MB.BuildRefusal as refusal:
        return _refusal_response(refusal)
    with contextlib.suppress(Exception):
        _bind_interrupt_rows([(p["lg_id"], p["payload"]) for p in out["interrupts"]])
        _note_interrupts_seen({p["lg_id"] for p in out["interrupts"]})
    return out


@app.post("/build/resume")
async def build_resume(req: BuildResumeRequest):
    """Answer a build's park with ONE action (``approve`` for the uArch review,
    ``retry`` / ``fix_rtl`` / ``fix_tb`` / ``skip`` / ``abort`` ... per the park's
    ``supported_actions``). HTTP transport of
    :func:`orchestrator.module_build.resume_build`: the build's thread is
    selected from its recorded identity, its recorded inputs are re-validated
    against the live project, and only then is the checkpoint read."""
    try:
        out = await MB.resume_build(
            _project_db(), _PROJECT_ROOT, _build, req.build_id, action=req.action, busy=_busy_lifecycles(),
            feedback=req.feedback, rtl_fix_description=req.rtl_fix_description, rationale=req.rationale,
            interrupt_id=req.interrupt_id, actor=req.actor or "cli",
            apply_env=lambda: _apply_run_env("build/resume"),
            on_parks=lambda parks: _bind_interrupt_rows([(p["lg_id"], p["payload"]) for p in parks]))
    except MB.BuildRefusal as refusal:
        if refusal.code == "ACTION_UNSUPPORTED":
            raise HTTPException(400, f"{refusal.text}; allowed: {refusal.extra.get('allowed')}")
        if refusal.code in ("UNKNOWN_BUILD", "NO_SUCH_INTERRUPT"):
            raise HTTPException(404, refusal.text)
        if refusal.code in ("BUILD_TERMINAL", "BUILD_RUNNING", "PIPELINE_RUNNING", "BACKEND_RUNNING",
                            "NOTHING_TO_RESUME", "NOT_A_MODULE_BUILD"):
            raise HTTPException(409, refusal.text)
        return _refusal_response(refusal)
    if out.get("answered"):
        _record_decisions(ResumeRequest(action=req.action, feedback=req.feedback, rationale=req.rationale,
                                        actor=req.actor or "cli"), out["answered"])
    asyncio.create_task(_watch_build(req.build_id))
    return out


@app.post("/build/pause")
async def build_pause():
    paused = await _build.safe_pause()
    if paused:
        with contextlib.suppress(Exception):
            for row in B.active_builds(_project_db()):
                if row["thread_id"] == _build.thread_id:
                    B.mark_status(_project_db(), row["id"], "parked", error="paused by the operator", terminal=False)
    return {"paused": paused, "thread_id": _build.thread_id}


@app.post("/build/abort")
async def build_abort(req: BuildAbortRequest):
    """Mark a build that is not running ``aborted`` (history kept, nothing
    published) so its module can be built again; a running build is paused
    first."""
    try:
        return await MB.abort_build(_project_db(), _build, req.build_id, reason=req.reason, actor=req.actor or "cli")
    except MB.BuildRefusal as refusal:
        return _refusal_response(refusal)


@app.get("/run/state")
async def run_state():
    await _pipeline.ensure_graph()
    snap = await _pipeline.graph.aget_state(
        {"configurable": {"thread_id": _pipeline.thread_id}}
    )
    return _shape_state(snap)


@app.post("/run/start")
@_serialized("run start", graph="pipeline")
async def run_start(req: StartRequest):
    global _last_resume_ts
    # The stall clock and the consumed-interrupt set are reset only once every
    # refusal below has passed (a refused request starts nothing, so it must
    # not reset the state of the run that is actually there).
    if _pipeline.task is not None and not _pipeline.task.done():
        raise HTTPException(409, "pipeline already running; call /run/pause first")
    _refuse_if_build_running("run start")

    # Guard: a `run start` on an EXISTING run (paused / parked at an interrupt /
    # completed) would SILENTLY discard it via reset_for_new_run() -- wiping all
    # block/RTL progress back to completed_count=0. Refuse unless force=true so
    # the operator must explicitly clear. To continue an in-flight run, `resume`
    # it instead.
    if not req.force:
        try:
            await _pipeline.ensure_graph()
            snap = await _pipeline.graph.aget_state(
                {"configurable": {"thread_id": _pipeline.thread_id}}
            )
            vals = (snap.values if snap else {}) or {}
        except Exception:  # noqa: BLE001 - no prior state -> first run, proceed
            vals = {}
        completed = vals.get("completed_blocks") or []
        if completed or vals.get("pipeline_run_start") or vals.get("pipeline_done"):
            raise HTTPException(
                409,
                "a run already exists in this project root "
                f"(completed_blocks={len(completed)}, "
                f"pipeline_done={bool(vals.get('pipeline_done'))}). "
                "`run start` would DISCARD it and restart from scratch. To "
                "continue it, use `resume`. To intentionally start fresh, pass "
                "force=true (CLI: `run start --force`).",
            )

    # Mid-run edits to .coresmith/env (gate knobs, provider selectors) apply to
    # the work this request is about to launch, not only to the next daemon.
    env_updated = _apply_run_env("run/start")

    # The Architect's stage machine must have reached ``blocks`` (every module
    # ready: spec, target, built reference model with its declared checks,
    # worker binding) with every earlier stage still holding, and cluster
    # fan-out is refused. Checked before any tool preflight, baseline capture
    # or state reset. There is no empty-stage exemption and ``force`` never
    # bypasses it: ``force`` only means "replace the existing run". A DB that
    # cannot be read is a refusal, never a pass.
    try:
        refusal = MB.pipeline_start_refusal(_project_db(), _PROJECT_ROOT)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=409, content={
            "error": "STAGE_DB_UNREADABLE", "message": f"the project database could not be read: {exc}",
            "hint": "repair .coresmith/project.sqlite; the stage machine is not bypassed"})
    if refusal is not None:
        return _refusal_response(refusal)

    block_queue = _load_block_queue(req.blocks_file)
    if not block_queue:
        raise HTTPException(
            400,
            "No blocks found. Provide blocks_file or place blocks: in "
            "orchestrator/config.yaml.",
        )

    _preflight_or_400()
    refused = _requirements_gate_response(_PROJECT_ROOT)
    if refused is not None:
        return refused
    arch_warnings = _check_architecture_artifacts(_PROJECT_ROOT)

    # Every refusal has passed: this request starts a run.
    _last_resume_ts = time.time()  # Section 7b: run start resets the stall clock
    _consumed_interrupt_ids.clear()   # a fresh run answers nothing from the old

    # B3: persist the resolved block queue + initialize the scoreboard schema +
    # snapshot the oracle manifest so the harness (`coresmith verify ...`) can
    # resolve blocks and detect oracle tampering after the daemon parks. All
    # Queue/schema exports are best-effort; trust capture must succeed.
    try:
        from orchestrator.harness.blocks import persist_block_queue
        persist_block_queue(_PROJECT_ROOT, block_queue)
    except Exception:  # noqa: BLE001
        pass
    try:
        from orchestrator.state_store.store import Scoreboard
        Scoreboard(_PROJECT_ROOT).ensure_schema()
    except Exception:  # noqa: BLE001
        pass
    from orchestrator.state_store.trust import capture_run_baseline
    try:
        capture_run_baseline(_PROJECT_ROOT)
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc
    # C1: every run-scoped table (run_flags, decisions, interrupts) keys on this.
    with contextlib.suppress(Exception):
        _project_db().begin_run()
    # The discarded run's in-flight builds (rows the pipeline graph dispatched)
    # can never complete once its checkpoint is wiped: they are aborted, with
    # history, so their modules are buildable again.
    with contextlib.suppress(Exception):
        aborted = MB.abort_pipeline_builds(_project_db(), reason="the pipeline run was replaced (run start --force)")
        if aborted:
            _daemon_log("warning", "run start: aborted in-flight pipeline builds %s", aborted)

    await _pipeline.reset_for_new_run()

    _rotate_events(Path(_PROJECT_ROOT) / ".coresmith" / "pipeline_events.jsonl")

    from orchestrator.langgraph.pipeline_helpers import resolve_run_clock_mhz
    initial_state = {
        "project_root": _PROJECT_ROOT,
        "target_clock_mhz": resolve_run_clock_mhz(req.target_clock_mhz, _PROJECT_ROOT),
        "max_attempts": req.max_attempts,
        "block_queue": block_queue,
        "tier_list": [],
        "current_tier_index": 0,
        "completed_blocks": [],
        "integration_result": None,
        "pipeline_done": False,
        "pipeline_run_start": time.time(),
    }

    graph_config = {"configurable": {"thread_id": _pipeline.thread_id}}
    await _pipeline.safe_start(initial_state, graph_config)
    response = {
        "started": True,
        "block_count": len(block_queue),
        "status": _pipeline.status,
    }
    if env_updated:
        response["env_updated"] = env_updated
    if arch_warnings:
        response["warnings"] = arch_warnings
    return response


def _resume_action_error(
    action: str,
    block_actions: dict | None,
    interrupt_meta: list[tuple[str, list]],
) -> tuple[str, str, list] | None:
    """Validate a resume action against the parked interrupts' supported_actions.

    Returns ``(block_name, bad_action, allowed_actions)`` for the FIRST interrupt
    whose EFFECTIVE action is not in its declared ``supported_actions``, or
    ``None`` when every interrupt accepts its action. An interrupt that declares
    no ``supported_actions`` imposes no constraint (back-compat: not every
    interrupt enumerates them). ``interrupt_meta`` is ``[(block_name,
    supported_actions), ...]``; an interrupt's effective action is
    ``block_actions[block_name]`` when provided, else the default ``action``.

    This makes ``/run/resume`` reject an unsupported action with a 400 + the
    allowed list instead of silently forwarding it (e.g. ``approve`` sent to a
    DV-failure interrupt whose only stop action is ``abort``), which the graph
    node would otherwise map to its own default and appear to "proceed".
    """
    ba = block_actions or {}
    for block_name, supported in interrupt_meta:
        if not supported:
            continue
        effective = ba.get(block_name, action) if block_name else action
        if effective not in supported:
            return (block_name, effective, list(supported))
    return None


def _resume_tick_or_park(has_pending_interrupt: bool, has_next_nodes: bool) -> str:
    """Decide how ``POST /run/resume`` advances a not-running pipeline.

    Returns one of:
      - ``"resume"``: a parked interrupt is pending -> a real, *supported*
        action is required (validated against ``supported_actions``) and
        forwarded as ``Command(resume=...)``.
      - ``"tick"``: NO pending interrupt but the checkpoint still has next
        nodes (a stranded/paused run, e.g. after an ``aget_state``/
        ``aupdate_state`` recovery) -> re-invoke the graph with ``cmd=None`` to
        tick it forward, matching the ``/architecture/resume`` tick semantics
        (the pipeline endpoint previously only 409'd here, so a stranded run
        had no plain-tick path).
      - ``"none"``: nothing to do (no interrupt, no next nodes) -> HTTP 409.

    Guard: a PARKED run never ticks -- ``has_pending_interrupt`` wins so parked
    runs still require a real action (fail-closed against the supported_actions
    validation). Only a not-parked run with pending next nodes ticks.
    """
    if has_pending_interrupt:
        return "resume"
    if has_next_nodes:
        return "tick"
    return "none"


@app.post("/rulings")
async def rulings_add(req: RulingRequest):
    """Record an operator ruling (C2). Resolves any pending interrupt it
    answers; if the runner is idle and one was resolved, it is applied now."""
    from orchestrator.state_store.rulings import apply_ruling_to_interrupts
    try:
        db = _project_db()
        rid = db.add_ruling(req.scope, req.text, rationale=req.rationale, source=req.source,
                            question_ref=req.question_ref, supersedes_id=req.supersedes_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"rulings table unavailable: {exc}") from exc
    ruling = db.ruling(rid)
    resolved = apply_ruling_to_interrupts(db, ruling)
    with contextlib.suppress(Exception):
        db.export_rulings_view()
    applied = False
    if resolved and not _pipeline_task_in_flight():
        # Idle runner: a plain tick lets run_task's boundary applier consume
        # the queued answers for exactly those branches.
        with contextlib.suppress(Exception):
            await _pipeline.ensure_graph()
            await _pipeline.safe_resume(None, {"configurable": {"thread_id": _pipeline.thread_id}})
            applied = True
    return {"id": rid, "ruling": ruling, "resolved_interrupts": resolved,
            "applied_now": applied, "conflicts": ruling["conflicts"]}


@app.get("/rulings")
async def rulings_list(scope: str | None = None, all: bool = False):
    try:
        rows = _project_db().rulings(scope=scope or None, active_only=not all)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"rulings table unavailable: {exc}") from exc
    return {"rulings": rows, "count": len(rows)}


@app.post("/rulings/{ruling_id}/revoke")
async def rulings_revoke(ruling_id: int, req: RevokeRulingRequest):
    try:
        db = _project_db()
        ok = db.revoke_ruling(ruling_id, req.reason)
        db.export_rulings_view()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"rulings table unavailable: {exc}") from exc
    if not ok:
        raise HTTPException(404, f"no active ruling {ruling_id}")
    return {"revoked": True, "id": ruling_id}


@app.get("/run/interrupts")
async def run_interrupts(status: str | None = None):
    """The interrupts table for this run (pending by default when asked)."""
    try:
        rows = _project_db().interrupts(status=status or None)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"interrupts table unavailable: {exc}") from exc
    return {"interrupts": rows, "count": len(rows)}


def _resume_value(req: ResumeRequest) -> dict:
    """The resolution a resume hands the graph. ``reasoning`` mirrors the
    rationale (the revise-churn check reads it); ``actor`` says who answered."""
    val = {
        "action": req.action,
        "feedback": req.feedback,
        "rtl_fix_description": req.rtl_fix_description,
        "rationale": req.rationale,
        "reasoning": req.rationale,
        "block_actions": req.block_actions or {},
        "actor": req.actor or "cli",
    }
    if req.affected_blocks:
        val["affected_blocks"] = list(req.affected_blocks)
    return val


def _record_decisions(req: ResumeRequest, answered: list[dict]) -> None:
    """One ``decisions`` row per answered park, with the actor. Best-effort."""
    try:
        db = _project_db()
        for val in answered:
            block = val.get("block_name") or val.get("block") or ""
            action = (req.block_actions or {}).get(block, req.action) if block else req.action
            db.add_decision(action=action, interrupt_type=str(val.get("type") or ""), block=block,
                            reasoning=req.rationale or req.feedback or "",
                            interrupt_id=str(val.get("interrupt_id") or ""), actor=req.actor or "cli")
        with contextlib.suppress(OSError):
            db.export_decisions_view()
    except Exception as exc:  # noqa: BLE001
        log.warning("decision ledger unavailable (%s)", exc)


@app.post("/run/resume")
@_serialized("run resume", graph="pipeline")
async def run_resume(req: ResumeRequest):
    global _last_resume_ts, _consumed_interrupt_ids
    _last_resume_ts = time.time()  # Section 7b: the driver is alive
    with contextlib.suppress(Exception):
        _project_db().clear_flag("stalled_interrupt")
    with contextlib.suppress(OSError):
        _mk = Path(_PROJECT_ROOT) / "STALLED_INTERRUPT"
        if _mk.exists():
            _mk.unlink()
    _refuse_if_build_running("run resume")
    await _pipeline.ensure_graph()
    if _pipeline.task is not None and not _pipeline.task.done():
        if req.interrupt_id:
            # C1-3: a parked branch whose Send() siblings are still running.
            # LangGraph only resumes at superstep boundaries, so queue the
            # answer in the interrupts table; the branch picks it up in its
            # pre-park wait or the boundary applier in run_task resumes just
            # that branch when the step ends.
            resolution = _resume_value(req)
            try:
                row = _project_db().interrupt(req.interrupt_id)
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(500, f"interrupts table unavailable: {exc}") from exc
            if not row or row.get("status") != "pending":
                raise HTTPException(404, f"no pending interrupt {req.interrupt_id!r}")
            _payload = row.get("payload") or {}
            _meta = [(_payload.get("block", _payload.get("block_name", "")),
                      _payload.get("supported_actions", []))]
            _bad = _resume_action_error(req.action, req.block_actions, _meta)
            if _bad is not None:
                raise HTTPException(400, f"action '{_bad[1]}' not supported by the parked "
                                         f"interrupt; allowed: {_bad[2]}")
            try:
                ok = _project_db().resolve_interrupt(
                    req.interrupt_id, resolution, resolved_by=req.actor or "cli")
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(500, f"interrupts table unavailable: {exc}") from exc
            if not ok:
                raise HTTPException(404, f"no pending interrupt {req.interrupt_id!r}")
            _record_decisions(req, [_payload])
            return JSONResponse(status_code=202, content={
                "resumed": False, "queued": True, "interrupt_id": req.interrupt_id,
                "action": req.action,
                "note": "applied at the next superstep boundary (or sooner, if the "
                        "branch is waiting in CORESMITH_INTERRUPT_WAIT_S)",
            })
        raise HTTPException(409, "pipeline still running; pass interrupt_id to "
                                 "queue an answer for one parked branch")

    # A resume re-enters the graph in THIS process: pick up any .coresmith/env
    # edits the operator made while the run was parked.
    env_updated = _apply_run_env("run/resume")

    graph_config = {"configurable": {"thread_id": _pipeline.thread_id}}
    state_snapshot = await _pipeline.graph.aget_state(graph_config)

    # Collect all pending interrupt IDs so parallel-block runs all resume
    # with the same decision unless the caller provided per-block actions.
    interrupts: list[tuple[str, Any]] = []
    interrupt_meta: list[tuple[str, list]] = []
    if state_snapshot and state_snapshot.tasks:
        for task in state_snapshot.tasks:
            for intr in task.interrupts:
                _val = intr.value if isinstance(intr.value, dict) else {}
                # C1-3: a targeted resume answers ONE branch; the others stay
                # parked (they re-raise their interrupt on the next tick).
                if req.interrupt_id and _val.get("interrupt_id") != req.interrupt_id:
                    continue
                interrupts.append((intr.id, intr.value))
                interrupt_meta.append((
                    _val.get("block", _val.get("block_name", "")),
                    _val.get("supported_actions", []),
                ))
    _bind_interrupt_rows(interrupts)
    if req.interrupt_id and not interrupts:
        raise HTTPException(404, f"no parked interrupt {req.interrupt_id!r} in the checkpoint")

    _has_next = bool(state_snapshot and state_snapshot.next)
    _mode = _resume_tick_or_park(bool(interrupts), _has_next)
    if _mode == "none":
        raise HTTPException(409, "no pending interrupt")
    if _mode == "tick":
        # No parked interrupt but the graph still has next nodes: plain tick
        # (cmd=None) to advance a stranded/paused run without a fake action.
        _consumed_interrupt_ids.clear()
        await _pipeline.safe_resume(None, graph_config)
        _ticked = {
            "resumed": True,
            "ticked": True,
            "next_nodes": list(state_snapshot.next),
            "action": req.action,
            "status": _pipeline.status,
        }
        if env_updated:
            _ticked["env_updated"] = env_updated
        return _ticked

    # Reject an action the parked interrupt does not support (400 + allowed list)
    # rather than silently forwarding it into the graph.
    _bad = _resume_action_error(req.action, req.block_actions, interrupt_meta)
    if _bad is not None:
        _bn, _act, _allowed = _bad
        _where = f" (block '{_bn}')" if _bn else ""
        raise HTTPException(
            400,
            f"action '{_act}'{_where} not supported by the parked interrupt; "
            f"allowed: {_allowed}",
        )

    resume_value: Any = _resume_value(req)

    from langgraph.types import Command
    if len(interrupts) > 1 or req.interrupt_id:
        cmd = Command(resume={iid: resume_value for iid, _ in interrupts})
    else:
        cmd = Command(resume=resume_value)
    # C1-3: the rows behind these interrupts are answered by this resume
    # (resolved_by = the actor, then consumed), one decisions row each.
    with contextlib.suppress(Exception):
        _db = _project_db()
        for _, _v in interrupts:
            if isinstance(_v, dict) and _v.get("interrupt_id"):
                _db.resolve_interrupt(str(_v["interrupt_id"]), resume_value,
                                      resolved_by=req.actor or "cli")
        _db.consume_lg_interrupts([iid for iid, _ in interrupts])
    _record_decisions(req, [v for _, v in interrupts if isinstance(v, dict)])

    # D5: remember exactly which interrupts this resume answers, BEFORE the
    # graph starts running. Until it checkpoints again, aget_state still returns
    # them; without this record /run/state reports them as pending on a running
    # run and the outer agent resumes a second time.
    _consumed_interrupt_ids = {iid for iid, _ in interrupts}
    await _pipeline.safe_resume(cmd, graph_config)
    result = {"resumed": True, "interrupts": len(interrupts), "action": req.action}
    if env_updated:
        result["env_updated"] = env_updated
    return result


@app.post("/run/pause")
async def run_pause():
    return {"paused": await _pipeline.safe_pause()}


@app.post("/run/continue")
@_serialized("run continue", graph="pipeline")
async def run_continue():
    """Continue a pipeline that has next_nodes but no pending interrupt.

    Calls graph.ainvoke(None, config) to resume from current checkpoint
    without resetting progress. Safe when status=done but pipeline_done=False.
    """
    if _pipeline.task is not None and not _pipeline.task.done():
        raise HTTPException(409, "pipeline already running")
    _refuse_if_build_running("run continue")
    _apply_run_env("run/continue")
    await _pipeline.ensure_graph()
    snap = await _pipeline.graph.aget_state(
        {"configurable": {"thread_id": _pipeline.thread_id}}
    )
    if not snap or not snap.next:
        return {"continued": False, "reason": "no next nodes in checkpoint"}
    config = {"configurable": {"thread_id": _pipeline.thread_id}}
    await _pipeline.safe_start(None, config)
    return {"continued": True, "next_nodes": list(snap.next), "status": _pipeline.status}


@app.post("/run/restart-block")
async def run_restart_block(req: RestartBlockRequest):
    """Rebuild ONE block: a compatibility route onto the recorded module
    build (``POST /build/module``). The block runs through the same block
    subgraph on the build lifecycle's persistent checkpoint (one thread per
    build) under the same readiness and provenance rules. ``from_node``
    is accepted for compatibility; both entries implement the registered
    spec unchanged. Nonempty ``uarch_feedback`` is refused: the Architect
    must revise and register the spec before starting a new build.
    Requires the pipeline to be idle (pause first). Re-enter the pipeline
    afterwards with ``/run/restart-node``.
    """
    if req.from_node not in ("generate_uarch_spec", "generate_rtl"):
        raise HTTPException(400, f"invalid from_node {req.from_node!r}: generate_uarch_spec | generate_rtl")
    feedback = req.uarch_feedback if req.from_node == "generate_uarch_spec" else ""
    return await _start_module_build(
        BuildModuleRequest(module=req.block_name, max_attempts=req.max_attempts, uarch_feedback=feedback),
        entry="restart_block")


@app.post("/run/restart-node")
@_serialized("run restart-node", graph="pipeline")
async def run_restart_node(req: RestartNodeRequest):
    """Re-run the pipeline from the checkpoint where ``node`` is next, reusing
    every block's on-disk RTL/TB (engine follow-up #8/#10).

    Unlike ``run start --force`` (full pipeline restart + unconditional uarch
    spec regen), this forks from a specific node so a late-stage re-drive --
    re-run a skipped ``integration_check``, or ``validation_dv`` after a
    hand-patch -- does NOT regenerate already-passing blocks. Requires the
    pipeline to be idle (pause first if running).
    """
    if _pipeline.task is not None and not _pipeline.task.done():
        raise HTTPException(409, "pipeline already running -- pause first")
    _refuse_if_build_running("run restart-node")
    # Re-entering a node runs it with THIS process's env; the operator's
    # mid-run .coresmith/env edits must be live for that re-run (bug C).
    env_updated = _apply_run_env("run/restart-node")
    refreshed = []
    if req.refresh_sidecars:
        # #6: re-sync intact-RTL blocks' contract sidecars to live before
        # re-entering integration, so the staleness preflight does not force a
        # mass-regen of already-passing blocks.
        try:
            from orchestrator.state_store.project_db import open_project as _open_project
            _pdb = _open_project(_PROJECT_ROOT)
            for _name in _pdb.block_names():
                if _pdb.block_contract_version(_name):
                    _pdb.stamp_block_spec(_name)
                    refreshed.append(_name)
        except Exception:
            log.warning("restart-node: sidecar refresh failed", exc_info=True)
    result = await _pipeline.restart_from_node(req.node)
    if refreshed:
        result["sidecars_refreshed"] = refreshed
    if env_updated:
        result["env_updated"] = env_updated
    if result.get("error"):
        raise HTTPException(400, result["error"] + (
            " -- " + result["hint"] if result.get("hint") else ""))
    result["status"] = _pipeline.status
    return result


@app.post("/run/revise-blocks")
async def run_revise_blocks(req: ReviseBlocksRequest):
    """Operator-triggered targeted revise of named published blocks.

    Writes the integration-review revise plan ({block: reuse_spec=True}, the
    feedback -- else the block's human constraints -- as gate_feedback.txt,
    best/dv_best dropped for those blocks only) onto the latest checkpoint and
    re-enters the tier loop at init_tier. Other blocks are not redone.
    Requires the pipeline to be idle (pause first).
    """
    if _pipeline.task is not None and not _pipeline.task.done():
        raise HTTPException(409, "pipeline already running -- pause first")
    env_updated = _apply_run_env("run/revise-blocks")
    from orchestrator.langgraph.pipeline_graph import operator_revise_update
    result = await _pipeline.restart_with_update(
        lambda values: operator_revise_update(str(_PROJECT_ROOT), values, req.blocks, req.feedback),
        "integration_review")
    if result.get("error"):
        raise HTTPException(400, result["error"])
    if env_updated:
        result["env_updated"] = env_updated
    result["status"] = _pipeline.status
    return result


# ---------------------------------------------------------------------------
# Backend endpoints (flat synthesis -> chip_top gate-sim [-> P&R/DRC/LVS]).
#
# Until now the daemon's lifecycle stopped at the frontend: the chip_top
# gate-sim -- the only thing that ever simulates the artifact that becomes
# silicon -- lived in the backend graph and was reachable ONLY from an MCP
# client or a hand-written driver script. A run could reach pipeline_done and
# simply stop, with nobody to press the next button.
#
# Shape follows /architecture/*: a second graph gets its own
# /<graph>/start|state|pause backed by a GraphLifecycle. The implementation is
# shared with the MCP tool (mcp_server.launch_backend), exactly as
# /run/restart-block shares restart_block -- one implementation, two transports.
# ---------------------------------------------------------------------------

def _backend_handle():
    """The backend GraphLifecycle, imported lazily from the MCP server module.

    Lazy for the same reason ``/run/restart-block`` is: importing mcp_server
    pulls in the whole agent stack, and a daemon that only ever drives the
    frontend should not pay for it. It reads ``CORESMITH_PROJECT_ROOT``, which
    this daemon sets at import time, so both transports address the same
    checkpoint DB.
    """
    from orchestrator import mcp_server as _mcp
    return _mcp


@app.post("/backend/start")
@_serialized("backend start", graph="backend")
async def backend_start(req: BackendStartRequest):
    """Enter the backend: flat top synthesis + the chip_top gate-sim verdict.

    Stops there unless ``full=true``. Requires every block to have RTL +
    synthesis artifacts on disk (the shared launcher's own gate) -- so calling
    this before the frontend finished returns the missing-artifact list rather
    than starting a doomed run.
    """
    # Backend preflight: P&R needs a REACHABLE OpenROAD (the default nix
    # wrapper fails with `exec: nix: not found` on a host without nix); DRC/LVS
    # need klayout/magic/netgen (warnings). A --full start without OpenROAD is
    # refused up front (412 + remedy) instead of parking hours later.
    _refuse_if_build_running("backend start")
    pre = _backend_preflight()
    if req.full and not pre.get("ok", True):
        raise HTTPException(412, {"error": "backend_preflight_failed", "details": pre.get("errors", []),
                                  "warnings": pre.get("warnings", [])})
    try:
        _mcp = _backend_handle()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"backend launcher unavailable: {exc}") from exc
    result = await _mcp.launch_backend(
        max_attempts=req.max_attempts,
        target_clock_mhz=req.target_clock_mhz,
        stop_after_gate_sim=not req.full,
    )
    if result.get("error"):
        raise HTTPException(409, json.dumps(result))
    notes = list(pre.get("warnings") or []) + ([] if req.full else list(pre.get("errors") or []))
    if pre.get("openroad_how") == "native":
        notes.append(f"OpenROAD: nix wrapper without nix -> native {pre.get('openroad')}")
    if notes and isinstance(result, dict):
        result = {**result, "preflight_warnings": notes}
    return result


def _backend_preflight() -> dict:
    """``sky130.backend_tools_preflight``; probe failures fail closed."""
    try:
        from orchestrator.pdk.deployments.sky130 import backend_tools_preflight
        return backend_tools_preflight()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False,
                "errors": [f"backend preflight unavailable: {type(exc).__name__}: {exc}"],
                "warnings": []}


@app.get("/backend/state")
async def backend_state():
    """Backend graph snapshot, including the chip_top gate-sim verdict."""
    try:
        _mcp = _backend_handle()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"backend state unavailable: {exc}") from exc
    raw = await _mcp.get_backend_state()
    try:
        state = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (ValueError, TypeError):
        return {"raw": raw}
    # Surface the gate-sim verdict at the top level: it is the reason this
    # endpoint exists, and burying it in the raw checkpoint means an outer agent
    # has to know where to dig. Absence is reported as absence, never as a pass.
    try:
        snap = await _mcp._backend.graph.aget_state(
            {"configurable": {"thread_id": _mcp._backend.thread_id}}
        )
        vals = (snap.values if snap else {}) or {}
    except Exception:  # noqa: BLE001
        vals = {}
    state["chip_gate_sim"] = {
        "ok": vals.get("chip_gate_sim_ok"),
        "status": vals.get("chip_gate_sim_status", ""),
        "reason": vals.get("chip_gate_sim_reason", ""),
        "flat_netlist_path": vals.get("flat_netlist_path", ""),
        "stopped_after_gate_sim": bool(vals.get("stop_after_gate_sim")),
    }
    return state


class BackendResumeRequest(BaseModel):
    action: str = "retry"          # retry | skip | abort | accept
    constraint: str = ""
    feedback: str = ""             # alias of constraint (the CLI's --feedback)
    rationale: str = ""
    actor: str = "cli"
    interrupt_id: str = ""


async def _backend_park_meta() -> list[dict]:
    """The backend's live parks (payloads), for validating a backend resume."""
    try:
        return [p["payload"] for p in await _live_backend_parks()]
    except Exception:  # noqa: BLE001
        return []


@app.post("/backend/resume")
@_serialized("backend resume", graph="backend")
async def backend_resume(req: BackendResumeRequest):
    """Resume a parked backend interrupt (DRC/LVS/signoff ask_human).

    Delegates to the MCP resume_backend (one implementation, two transports
    -- the same split as /backend/start). Fire-and-forget is safe HERE
    because the daemon event loop hosts the created task; the gap this
    closes is that resume_backend was previously reachable only from a
    short-lived MCP process, which silently lost the resume. Two campaigns
    were run-blocked on this: backend start --full can only replay into the
    same park, so a DRC/LVS-parked backend was terminal from the outside."""
    _refuse_if_build_running("backend resume")
    try:
        _mcp = _backend_handle()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"backend resume unavailable: {exc}") from exc
    # Validate against the parked payload(s) and record who answered
    # (decisions, interrupts.resolved_by) -- the same contract /run/resume has.
    parked = await _backend_park_meta()
    if req.interrupt_id and parked:
        parked = [v for v in parked if v.get("interrupt_id") == req.interrupt_id]
        if not parked:
            raise HTTPException(404, f"no parked backend interrupt {req.interrupt_id!r}")
    meta = [(v.get("block_name", v.get("block", "")), v.get("supported_actions", [])) for v in parked]
    _bad = _resume_action_error(req.action, None, meta)
    if _bad is not None:
        raise HTTPException(400, f"action '{_bad[1]}' not supported by the parked backend "
                                 f"interrupt; allowed: {_bad[2]}")
    text = req.feedback or req.constraint
    raw = await _mcp.resume_backend(action=req.action, constraint=text)
    try:
        result = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (ValueError, TypeError):
        return {"raw": raw}
    if result.get("error"):
        raise HTTPException(409, json.dumps(result))
    with contextlib.suppress(Exception):
        db = _project_db()
        value = {"action": req.action, "constraint": text, "feedback": text,
                 "rationale": req.rationale, "actor": req.actor or "cli"}
        for v in parked:
            if v.get("interrupt_id"):
                db.resolve_interrupt(str(v["interrupt_id"]), value, resolved_by=req.actor or "cli")
                db.consume_interrupt(str(v["interrupt_id"]))
        _record_decisions(ResumeRequest(action=req.action, feedback=text, rationale=req.rationale,
                                        actor=req.actor or "cli"), parked)
    return result


@app.post("/backend/pause")
async def backend_pause():
    """Pause through the authoritative MCP implementation.

    Backend LLM calls run a blocking child process in an executor thread, so
    cancelling only the asyncio graph task leaves that process running. The MCP
    implementation reaps active CLI process groups before cancellation and is
    shared here to keep the two transports consistent.
    """
    try:
        _mcp = _backend_handle()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"backend pause unavailable: {exc}") from exc
    handle = _mcp._backend
    if handle.task is None or handle.task.done():
        return {"paused": False, "reason": "no running task"}

    raw = await _mcp.pause_backend()
    try:
        result = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (ValueError, TypeError):
        return {"paused": True, "raw": raw}
    if result.get("error"):
        raise HTTPException(409, json.dumps(result))
    return {"paused": True, **result}


# ---------------------------------------------------------------------------
# Architecture endpoints (PRD -> SAD -> FRD -> ERS -> block_diagram ->
# constraint check -> final review). Produces .coresmith/ers_spec.json
# and .coresmith/block_specs.json that the pipeline phase later consumes.
# ---------------------------------------------------------------------------

@app.post("/architecture/start")
async def architecture_start(req: ArchStartRequest):
    if _architecture.task is not None and not _architecture.task.done():
        raise HTTPException(409, "architecture already running; call /architecture/pause first")

    # The composition / model-integration gates read their knobs
    # (CORESMITH_REFERENCE_ENTRY, CORESMITH_MODEL_STIMULUS, ...) from the
    # environment at gate time, and the architecture graph is where they run.
    env_updated = _apply_run_env("architecture/start")

    requirements = req.requirements
    if not requirements and req.requirements_file:
        rf_path = Path(req.requirements_file)
        if not rf_path.is_absolute():
            rf_path = Path(_PROJECT_ROOT) / req.requirements_file
        if not rf_path.exists():
            raise HTTPException(400, f"requirements_file not found: {rf_path}")
        requirements = rf_path.read_text(encoding="utf-8")
    if not requirements:
        raise HTTPException(400, "Provide requirements or requirements_file")

    await _architecture.ensure_graph()

    from orchestrator.architecture.state import load_state, save_state
    from orchestrator.pdk import PDKConfig

    arch_state = load_state(_PROJECT_ROOT)
    arch_state.requirements = requirements
    arch_state.target_clock_mhz = req.target_clock_mhz

    pdk_summary = "No PDK configured"
    if req.pdk_config_path:
        pdk_path = Path(_PROJECT_ROOT) / req.pdk_config_path
        if pdk_path.exists():
            pdk = PDKConfig.from_yaml(str(pdk_path))
            arch_state.pdk_config = pdk.to_dict()
    if arch_state.pdk_config:
        pdk = PDKConfig.from_dict(arch_state.pdk_config)
        pdk_summary = pdk.to_summary()

    save_state(arch_state, _PROJECT_ROOT)

    await _architecture.reset_for_new_run()

    initial_state = {
        "project_root": _PROJECT_ROOT,
        "requirements": arch_state.requirements,
        "pdk_summary": pdk_summary,
        "target_clock_mhz": req.target_clock_mhz,
        "pdk_config": arch_state.pdk_config or {},
        "max_rounds": req.max_rounds,
        "round": 1,
        "phase": "prd",
        "prd_spec": None,
        "prd_questions": None,
        "sad_spec": None,
        "frd_spec": None,
        "ers_spec": None,
        "violations_history": [],
        "questions": [],
        "block_diagram": None,
        "memory_map": None,
        "clock_tree": None,
        "register_spec": None,
        "benchmark_data": arch_state.benchmark_results or None,
        "constraint_result": None,
        "human_feedback": arch_state.human_feedback or "",
        "human_response": None,
        "success": False,
        "error": "",
        "block_specs_path": "",
    }

    graph_config = {"configurable": {"thread_id": _architecture.thread_id}}
    await _architecture.safe_start(initial_state, graph_config)
    response = {
        "started": True,
        "status": _architecture.status,
        "requirements_length": len(requirements),
        "target_clock_mhz": req.target_clock_mhz,
        "pdk_summary": pdk_summary,
    }
    if env_updated:
        response["env_updated"] = env_updated
    return response


@app.get("/architecture/state")
async def architecture_state():
    await _architecture.ensure_graph()
    snap = await _architecture.graph.aget_state(
        {"configurable": {"thread_id": _architecture.thread_id}}
    )
    return _shape_arch_state(snap)


@app.post("/architecture/resume")
async def architecture_resume(req: ArchResumeRequest):
    valid = {"continue", "retry", "accept", "feedback", "abort"}
    if req.action not in valid:
        raise HTTPException(400, f"action must be one of {sorted(valid)}")
    if req.action == "feedback" and not req.feedback:
        raise HTTPException(400, "feedback is required when action=feedback")

    await _architecture.ensure_graph()
    if _architecture.task is not None and not _architecture.task.done():
        raise HTTPException(409, "architecture still running; nothing to resume")

    env_updated = _apply_run_env("architecture/resume")

    config = {"configurable": {"thread_id": _architecture.thread_id}}
    snap = await _architecture.graph.aget_state(config)

    resume_value: dict[str, Any] = {
        "action": req.action,
        "feedback": req.feedback,
        "rationale": req.rationale,
    }
    # PRD/ERS interrupts expect JSON-encoded answers via `feedback`. Parse and
    # promote so downstream routing finds them, matching mcp_server semantics.
    if req.feedback and req.action in ("continue", "feedback"):
        try:
            answers = json.loads(req.feedback)
            if isinstance(answers, dict):
                resume_value["answers"] = answers
                resume_value["action"] = "continue"
        except (json.JSONDecodeError, TypeError):
            pass

    from langgraph.types import Command
    has_pending = False
    if snap and snap.tasks:
        for t in snap.tasks:
            if t.interrupts:
                has_pending = True
                break

    if has_pending:
        cmd = Command(resume=resume_value)
    else:
        cmd = None  # plain tick to resume a paused run

    await _architecture.safe_resume(cmd, config)
    result = {"resumed": True, "action": resume_value["action"]}
    if env_updated:
        result["env_updated"] = env_updated
    return result


@app.post("/architecture/pause")
async def architecture_pause():
    return {"paused": await _architecture.safe_pause()}

def _shape_arch_state(snap) -> dict:
    base = {
        "status": _architecture.status,
        "thread_id": _architecture.thread_id,
        "project_root": _PROJECT_ROOT,
        "error_message": _architecture.error_message or None,
    }
    if not snap or not snap.values:
        base["values_empty"] = True
        return base
    values = snap.values
    interrupts: list[dict] = []
    if snap.tasks:
        for task in snap.tasks:
            for intr in task.interrupts:
                interrupts.append({"id": intr.id, "payload": intr.value})
    base.update({
        "phase": values.get("phase", ""),
        "round": values.get("round", 1),
        "max_rounds": values.get("max_rounds", 0),
        "has_prd": bool(values.get("prd_spec")),
        "has_sad": bool(values.get("sad_spec")),
        "has_frd": bool(values.get("frd_spec")),
        "has_ers": bool(values.get("ers_spec")),
        "has_block_diagram": bool(values.get("block_diagram")),
        "block_specs_path": values.get("block_specs_path", ""),
        "next_nodes": list(snap.next) if snap.next else [],
        "interrupts": interrupts,
        "pending_interrupt_count": len(interrupts),
        "interrupt_type": (
            interrupts[0]["payload"].get("type", "")
            if interrupts and isinstance(interrupts[0]["payload"], dict)
            else None
        ),
        "error": values.get("error", ""),
        "success": values.get("success", False),
    })
    return base


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_block_queue(blocks_file: str) -> list[dict]:
    """Resolve blocks the same way mcp_server.start_pipeline does."""
    from orchestrator.langgraph.pipeline_helpers import (
        get_sorted_block_queue,
        load_config,
    )

    if blocks_file:
        bf_path = Path(blocks_file)
        if not bf_path.is_absolute():
            bf_path = Path(_PROJECT_ROOT) / blocks_file
        if not bf_path.exists():
            raise HTTPException(400, f"blocks_file not found: {bf_path}")
        os.environ["CORESMITH_BLOCKS_FILE"] = str(bf_path)
    elif _ENV_BLOCKS_FILE:
        os.environ["CORESMITH_BLOCKS_FILE"] = _ENV_BLOCKS_FILE
    else:
        os.environ.pop("CORESMITH_BLOCKS_FILE", None)

    block_queue: list[dict] = []
    if not blocks_file:
        from orchestrator.state_store.project_db import open_project as _open_project
        block_queue = _open_project(_PROJECT_ROOT).block_specs()

    if not block_queue:
        config = load_config()
        block_queue = get_sorted_block_queue(config)

    return block_queue


def _preflight_or_400():
    from orchestrator.langgraph.pipeline_helpers import preflight_check
    check = preflight_check(["pipeline"])
    if not check["ok"]:
        raise HTTPException(412, {
            "error": "preflight_failed",
            "details": check["errors"],
            "warnings": check.get("warnings", []),
        })


_ENV_TRUE = {"1", "true", "yes", "on"}
_ENV_FALSE = {"0", "false", "no", "off"}


def requirements_registered(project_root) -> tuple[bool, list[str]]:
    """Whether the run's requirements are registered in the project DB.

    Returns ``(ok, missing)`` where ``missing`` holds stable codes:
    ``PRD_NOT_REGISTERED``, ``FRD_NOT_REGISTERED``, ``FRD_NO_MUST_HAVE`` (no
    must-have FRD item). A project DB that cannot be opened reports
    ``PROJECT_DB_UNREADABLE`` -- the gate fails closed.
    """
    try:
        from orchestrator.state_store.project_db import open_project
        db = open_project(project_root)
        missing = []
        if not db.artifact("prd"):
            missing.append("PRD_NOT_REGISTERED")
        if not db.artifact("frd"):
            missing.append("FRD_NOT_REGISTERED")
        if not db.items(artifact="frd", must_have=True):
            missing.append("FRD_NO_MUST_HAVE")
    except Exception as exc:  # noqa: BLE001
        log.warning("requirements gate: project DB unreadable: %s", exc)
        missing = ["PROJECT_DB_UNREADABLE"]
    return (not missing, missing)


def _requirements_gate_enabled() -> bool:
    """``/run/start`` requires registered requirements unless
    ``CORESMITH_REQUIRE_REQUIREMENTS`` is off (default ON) or the batch-eval
    escape hatch ``CORESMITH_SKIP_ARCH_WARN`` is truthy."""
    if (os.environ.get("CORESMITH_SKIP_ARCH_WARN", "") or "").strip().lower() in _ENV_TRUE:
        return False
    req = (os.environ.get("CORESMITH_REQUIRE_REQUIREMENTS", "1") or "1").strip().lower()
    return req not in _ENV_FALSE


def _stage_before_blocks(project_root) -> dict | None:
    """The stage machine's refusal for ``/run/start`` as a dict (None when the
    pipeline may start). Fail-closed: a project with no stage rows is at
    ``requirements``; a DB that cannot be read is a refusal."""
    from orchestrator.state_store.project_db import open_project
    try:
        refusal = MB.pipeline_start_refusal(open_project(project_root), project_root)
    except Exception as exc:  # noqa: BLE001
        return {"error": "STAGE_DB_UNREADABLE", "message": str(exc), "blocked_by": []}
    return refusal.to_json() if refusal is not None else None


def _requirements_gate_response(project_root) -> JSONResponse | None:
    """The HTTP 409 refusal for ``/run/start`` when the PRD / FRD / a
    must-have FRD item is not registered, else None. Architecture start is
    not gated."""
    if not _requirements_gate_enabled():
        return None
    ok, missing = requirements_registered(project_root)
    if ok:
        return None
    body = {
        "error": "requirements_not_registered",
        "missing": missing,
        "hint": ("register them: coresmith register prd|frd <path> (or the frd add verbs); "
                 "CORESMITH_SKIP_ARCH_WARN=1 bypasses for batch evaluation"),
    }
    return JSONResponse(status_code=409, content=body)


def _rotate_events(path: Path) -> Path | None:
    """Start a fresh pipeline events log for a new run.

    ``CORESMITH_ROTATE_EVENTS`` (default ``1``): a non-empty existing log is
    renamed to ``pipeline_events.<YYYYmmdd-HHMMSS>.jsonl`` next to it (keeping
    e.g. the Architect's ``stage_done`` events), then an empty file is
    created. ``0`` restores the old truncation. Returns the rotated path, or
    None when nothing was rotated.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rotate = (os.environ.get("CORESMITH_ROTATE_EVENTS", "1") or "1").strip().lower() not in _ENV_FALSE
    rotated = None
    if rotate and path.exists() and path.stat().st_size > 0:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        stem = path.name[: -len(".jsonl")] if path.name.endswith(".jsonl") else path.stem
        rotated = path.with_name(f"{stem}.{stamp}.jsonl")
        n = 1
        while rotated.exists():
            rotated = path.with_name(f"{stem}.{stamp}-{n}.jsonl")
            n += 1
        path.rename(rotated)
    path.write_text("")
    return rotated


def _registered_artifact_kinds(root: Path) -> set[str]:
    """Artifact kinds registered in the run's project DB (empty when the run
    has no DB -- never create one just to look)."""
    try:
        from orchestrator.state_store.stages import project_db_exists
        if not project_db_exists(root):
            return set()
        from orchestrator.state_store.project_db import open_project
        return {str(a.get("kind") or "") for a in open_project(root).artifacts()}
    except Exception:  # noqa: BLE001 - a warning helper must never fail run start
        return set()


def _check_architecture_artifacts(project_root: str) -> list[str]:
    """Return a list of warnings if the frontend pipeline is about to run
    without architecture-phase artifacts (PRD / ERS / block_diagram).

    The frontend pipeline can run against just `blocks.yaml` + a
    `generate_uarch_spec` per-block fallback, but the chip-level
    `integration_review` and `validation_dv` nodes need the architecture
    artifacts to do their job. When they're missing those nodes don't
    hard-fail — they soft-fail / abort silently — so the user thinks the
    run finished cleanly when it really skipped requirement validation.

    Set CORESMITH_SKIP_ARCH_WARN=1 to suppress this warning (the evaluation harness /
    rapid-iteration flows that intentionally skip the architecture phase).
    """
    if os.environ.get("CORESMITH_SKIP_ARCH_WARN", "").strip().lower() in {
        "1", "true", "yes", "on",
    }:
        return []

    root = Path(project_root)
    # Each artifact may live in more than one place: the architecture graph
    # writes arch/ers_spec.md, the CLI flow (`coresmith register ers`) renders
    # the ERS as JSON (arch/ers_spec.json, .coresmith/ers_spec.json -- the one
    # validation_dv reads). Any of them counts.
    # The PRD likewise: the architecture graph's .coresmith/prd_spec.json, or
    # the CLI flow's arch/prd_spec.{json,md} (`coresmith state write`). A
    # registered artifact row in the project DB (a file, or DB-sourced
    # ``db:<kind>``) counts as present too.
    expected = {
        "PRD spec": (root / ".coresmith" / "prd_spec.json", root / "arch" / "prd_spec.json",
                     root / "arch" / "prd_spec.md"),
        "ERS spec": (root / "arch" / "ers_spec.md", root / "arch" / "ers_spec.json",
                     root / ".coresmith" / "ers_spec.json"),
        "block diagram": (root / ".coresmith" / "block_diagram.json", root / "arch" / "block_diagram.json"),
    }
    registered = _registered_artifact_kinds(root)
    db_kind = {"PRD spec": "prd", "ERS spec": "ers", "block diagram": "block_diagram"}
    missing = [label for label, paths in expected.items()
               if db_kind[label] not in registered and not any(p.exists() for p in paths)]
    if not missing:
        return []

    return [
        (
            f"Architecture-phase artifacts missing: {', '.join(missing)}. "
            "The frontend pipeline will still run from blocks.yaml + uArch "
            "spec generation, but `integration_review` cannot verify "
            "cross-block data_width and `validation_dv` will soft-abort "
            "with 'No ERS found'. To exercise the full pipeline run "
            "`coresmith architecture start --requirements <spec>.md` first. "
            "Set CORESMITH_SKIP_ARCH_WARN=1 to silence this warning."
        )
    ]


def _shape_state(state_snapshot) -> dict:
    base = {
        "status": _pipeline.status,
        "thread_id": _pipeline.thread_id,
        "project_root": _PROJECT_ROOT,
        "error_message": _pipeline.error_message or None,
    }
    if not state_snapshot or not state_snapshot.values:
        base["values_empty"] = True
        return base

    values = state_snapshot.values
    _bind_pairs: list = []
    completed = values.get("completed_blocks", [])
    block_queue = values.get("block_queue", [])
    # Audit F9: completed_blocks is APPEND-ONLY across resumes / re-validation
    # passes -- the reference codec decoder accumulated 84 completion events for 21
    # blocks, so the raw len() reported completed_count=84 and
    # remaining_count=-63. Present attempt-scoped facts instead: one row per
    # unique block (the LATEST completion event wins -- it reflects the final
    # attempt), remaining floored at zero, and the raw append-only event count
    # preserved separately for forensics.
    latest_by_name: dict = {}
    for b in completed:
        name = b.get("name")
        if name:
            latest_by_name[name] = b
    completed_names = set(latest_by_name)

    # An interrupt whose block ALSO appears in completed_blocks used to be
    # dropped here as "stale". That silently hid LIVE interrupts: completed_blocks
    # is append-only across attempts, so a block that completed once and was
    # later re-entered (revise_interface, fix_rtl, a re-spec) is in the set
    # forever. The graph parks on `[HUMAN] Intervention needed`, /run/state
    # reports `pending_interrupt_count: 0` and `interrupts: []`, and the very
    # next POST /run/resume returns `{"resumed": true, "interrupts": 1}`.
    #
    # An outer agent polling state cannot tell "nothing to do" from "something
    # is waiting and I am hiding it", so the run parks until a human notices.
    # Surface every interrupt and LABEL the suspicion instead of acting on it --
    # a driver can then skip suspected-stale ones deliberately, which is a
    # different thing from never being told.
    #
    # D5: an interrupt a LIVE resume has already answered is not pending. The
    # checkpoint still carries it until the graph writes its next one, which is
    # how `/run/state` came to report `pending_interrupt_count: 1` alongside
    # `status: running` and drove outer agents to resume the same interrupt
    # twice. Discounted by ID, only while the runner task is in flight, and
    # still LISTED (with `consumed_by_resume`) so nothing is hidden.
    #
    # But "the block is in completed_blocks" alone is not evidence of a
    # leftover -- in a two-pass run EVERY block completes pass 1, so EVERY
    # pass-2 interrupt was born ``stale_suspected: true`` with
    # ``live_interrupt_count: 0``, and an automated driver reading that
    # concludes there is nothing to act on. The suspicion is now keyed to WHEN
    # THE INTERRUPT WAS RAISED: a leftover is an interrupt the graph has since
    # moved PAST, so the block's completion must be NEWER than the interrupt.
    # A just-raised interrupt reads LIVE, and an unstamped completion (a
    # checkpoint predating ``completed_at``) is no evidence at all -- absence of
    # evidence is not evidence of absence, which is this file's whole thesis.
    consumed = _consumed_now()
    now_ts = time.time()
    completion_ts = _completion_times(completed)
    _note_interrupts_seen(
        {intr.id
         for task in (state_snapshot.tasks or [])
         for intr in task.interrupts},
        now=now_ts,
    )
    interrupts: list[dict] = []
    suspected_stale = 0
    consumed_count = 0
    if state_snapshot.tasks:
        for task in state_snapshot.tasks:
            for intr in task.interrupts:
                payload = intr.value
                raised_ts = _interrupt_raised_ts(intr.id, now=now_ts)
                stale = False
                basis = "not_a_block_interrupt"
                if isinstance(payload, dict):
                    blk = payload.get("block", payload.get("block_name", ""))
                    if not blk:
                        basis = "not_a_block_interrupt"
                    elif blk not in completed_names:
                        basis = "block_not_completed"
                    else:
                        done_ts = completion_ts.get(blk, 0.0)
                        if not done_ts:
                            basis = "completion_unstamped"
                        elif raised_ts > done_ts:
                            basis = "raised_after_block_completed"
                        else:
                            basis = "raised_before_block_completed"
                            stale = True
                if stale:
                    suspected_stale += 1
                was_consumed = intr.id in consumed
                if was_consumed:
                    consumed_count += 1
                _bind_pairs.append((intr.id, payload))
                interrupts.append({
                    "id": intr.id,
                    "interrupt_id": (payload.get("interrupt_id")
                                     if isinstance(payload, dict) else None),
                    "payload": payload,
                    "stale_suspected": stale,
                    "stale_basis": basis,
                    "raised_ts": raised_ts,
                    "age_seconds": round(max(0.0, now_ts - raised_ts), 3),
                    "consumed_by_resume": was_consumed,
                })

    _bind_interrupt_rows(_bind_pairs)
    pending_interrupts = [i for i in interrupts if not i["consumed_by_resume"]]

    base.update({
        "passed_count": sum(b.get("success") is True for b in latest_by_name.values()),
        "failed_count": sum(b.get("success") is not True for b in latest_by_name.values()),
        "pending_count": max(0, len(block_queue) - len(latest_by_name)),
        "completed_count": len(latest_by_name),
        "completion_events": len(completed),
        "completed_blocks": [
            {"name": b.get("name"), "success": b.get("success"), "attempts": b.get("attempts", 1)}
            for b in latest_by_name.values()
        ],
        "total_blocks": len(block_queue),
        "remaining_count": max(0, len(block_queue) - len(latest_by_name)),
        "pipeline_done": values.get("pipeline_done", False),
        "frontend_outcome": (
            "awaiting_decision" if pending_interrupts else
            "failed" if values.get("pipeline_done") and (
                len(latest_by_name) < len(block_queue) or
                any(b.get("success") is not True for b in latest_by_name.values())) else
            "complete" if values.get("pipeline_done") and not state_snapshot.next else
            "running"),
        "next_nodes": list(state_snapshot.next) if state_snapshot.next else [],
        "interrupts": interrupts,
        # PENDING = still waiting on a decision. An interrupt whose resume is
        # already in flight is not waiting on anything.
        "pending_interrupt_count": len(interrupts) - consumed_count,
        # Split out so a driver can choose to skip suspected-stale ones
        # DELIBERATELY, rather than never being told they exist.
        "suspected_stale_interrupt_count": suspected_stale,
        "consumed_interrupt_count": consumed_count,
        "live_interrupt_count": max(
            0, len(interrupts) - consumed_count - suspected_stale),
        "interrupt_type": (
            pending_interrupts[0]["payload"].get("type", "")
            if pending_interrupts
            and isinstance(pending_interrupts[0]["payload"], dict)
            else None
        ),
    })

    for key in ("integration_result", "integration_dv_result", "validation_dv_result"):
        if values.get(key):
            base[key] = values[key]
    return base


# ---------------------------------------------------------------------------
# Daemon discovery file
# ---------------------------------------------------------------------------

def _daemon_file() -> Path:
    return Path(_PROJECT_ROOT) / ".coresmith" / "daemon.json"


_DAEMON_LEASE_TTL_S = 60.0
_daemon_lease_token: str | None = None


def _project_db():
    from orchestrator.state_store.project_db import open_project
    return open_project(_PROJECT_ROOT)


def _write_daemon_file(port: int):
    """Take the ``daemon`` lease for this project and publish daemon.json.

    C1: the lease (pid, port, token, expiry) is the ownership record; the file
    is a read-only view of it for ``bin/coresmith`` and older tools. A live
    foreign daemon refuses the start instead of being silently overwritten.
    """
    global _daemon_lease_token
    info = {"project_root": _PROJECT_ROOT, "port": port, "pid": os.getpid(),
            "started_at": time.time()}
    try:
        db = _project_db()
        token = db.acquire_lease("daemon", _DAEMON_LEASE_TTL_S, meta=info)
        if token is None:
            holder = db.lease("daemon") or {}
            raise SystemExit(
                f"error: another daemon (pid {holder.get('holder_pid')}@"
                f"{holder.get('holder_host')}, port {holder.get('meta', {}).get('port')}) "
                f"owns {_PROJECT_ROOT}; stop it or run "
                "`coresmith leases --steal daemon --reason ...`")
        _daemon_lease_token = token
    except SystemExit:
        raise
    except Exception as exc:
        raise RuntimeError("Cannot acquire daemon ownership; refusing startup") from exc
    df = _daemon_file()
    df.parent.mkdir(parents=True, exist_ok=True)
    df.write_text(json.dumps(info, indent=2))


async def _daemon_lease_renew() -> None:
    """Heartbeat the ``daemon`` lease at a third of its TTL."""
    while True:
        try:
            await asyncio.sleep(_DAEMON_LEASE_TTL_S / 3.0)
            if _daemon_lease_token:
                ok = await asyncio.to_thread(
                    _project_db().renew_lease, "daemon", _daemon_lease_token,
                    _DAEMON_LEASE_TTL_S)
                if not ok:
                    raise RuntimeError("daemon ownership was lost")
        except asyncio.CancelledError:
            break
        except Exception as exc:
            log.error("cannot retain daemon ownership: %s; stopping owned work", exc)
            from orchestrator.processes import cancel
            await asyncio.gather(_pipeline.safe_pause(), _architecture.safe_pause(),
                                 _backend_handle()._backend.safe_pause(), return_exceptions=True)
            await asyncio.to_thread(cancel)
            os.kill(os.getpid(), signal.SIGTERM)
            return


def _remove_daemon_file():
    """Remove daemon.json only if it still points at *this* process.

    Without the pid guard, a stale daemon that takes a while to finish
    uvicorn shutdown can race with a freshly-spawned replacement: the new
    daemon writes daemon.json, then the old daemon's finally / SIGTERM
    handler fires and deletes it. The CLI then reports `no daemon` while
    the replacement is happily serving HTTP -- which is exactly the bug
    seen on 2026-05-19 in the mcu3 run.
    """
    global _daemon_lease_token
    try:
        if _daemon_lease_token:
            with contextlib.suppress(Exception):
                _project_db().release_lease("daemon", _daemon_lease_token)
            _daemon_lease_token = None
        df = _daemon_file()
        if not df.exists():
            return
        try:
            info = json.loads(df.read_text())
        except Exception:
            df.unlink(missing_ok=True)
            return
        if info.get("pid") == os.getpid():
            df.unlink(missing_ok=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _pick_port(requested: int) -> int:
    if requested:
        return requested
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=0, help="bind port (0 = pick free)")
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    from orchestrator.state_store.trust import capture_run_baseline
    _apply_run_env("daemon start")
    capture_run_baseline(_PROJECT_ROOT)
    port = _pick_port(args.port)
    _write_daemon_file(port)

    def _sigterm(signum, frame):
        _remove_daemon_file()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)

    try:
        log.info("coresmithd starting at %s:%d for %s", args.host, port, _PROJECT_ROOT)
        uvicorn.run(app, host=args.host, port=port, log_level="info")
    finally:
        _remove_daemon_file()


if __name__ == "__main__":
    main()
