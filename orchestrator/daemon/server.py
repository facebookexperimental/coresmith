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

# A-Fix 1: seed profile flag defaults BEFORE importing graph code -- the
# pipeline builder reads gate-enable helpers (e.g. block_goldens_enabled) at
# build time, so the profile must be applied first.
from orchestrator.profile import apply as _apply_profile

_apply_profile()

from orchestrator.telemetry import init_telemetry

init_telemetry(_PROJECT_ROOT)

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from orchestrator.graph_lifecycle import GraphLifecycle

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
        if snap and snap.tasks:
            for t in snap.tasks:
                n += sum(1 for i in t.interrupts if i.id not in consumed)
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
                idle = time.time() - _last_resume_ts
                if idle >= _STALL_THRESHOLD_S:
                    log.warning(
                        "STALLED_INTERRUPT: %d pending interrupt(s) with no "
                        "/run/resume for %.0f min (project_root=%s). The outer "
                        "driver may be dead -- resume or restart it.",
                        pending, idle / 60.0, _PROJECT_ROOT,
                    )
                    try:
                        marker.write_text(json.dumps({
                            "pending_interrupt_count": pending,
                            "idle_seconds": round(idle),
                            "last_resume_ts": _last_resume_ts,
                            "noted_at": time.time(),
                        }, indent=2))
                    except OSError:
                        pass
            else:
                # cleared -> remove any stale marker
                with contextlib.suppress(OSError):
                    if marker.exists():
                        marker.unlink()
        except asyncio.CancelledError:
            break
        except Exception:  # noqa: BLE001 - liveness watch must never crash
            continue


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
    log.info(
        "CORESMITH_AUTO_BACKEND=1: this daemon will enter flat synthesis + the "
        "chip_top gate-sim by itself when the frontend reaches pipeline_done "
        "(P&R/DRC/LVS still require an explicit `backend start --full`).")
    while not _auto_backend_fired:
        try:
            await asyncio.sleep(_AUTO_BACKEND_POLL_S)
            if not await _frontend_is_done():
                continue
            _auto_backend_fired = True     # one-shot, even if the launch fails
            log.warning(
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
                log.error("AUTO-BACKEND: launch refused: %s", result)
            else:
                log.warning("AUTO-BACKEND: backend running (thread_id=%s)",
                            result.get("thread_id"))
        except asyncio.CancelledError:
            break
        except Exception:  # noqa: BLE001 - a handoff watch must never crash
            log.warning("AUTO-BACKEND: watch iteration failed", exc_info=True)
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


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class StartRequest(BaseModel):
    max_attempts: int = 5
    target_clock_mhz: float = 50.0
    blocks_file: str = ""
    force: bool = False


class ResumeRequest(BaseModel):
    action: str = "approve"
    feedback: str = ""
    rtl_fix_description: str = ""
    block_actions: dict | None = None
    rationale: str = ""


class RestartBlockRequest(BaseModel):
    block_name: str
    from_node: str = "generate_rtl"
    uarch_feedback: str = ""
    max_attempts: int = 3


class RestartNodeRequest(BaseModel):
    node: str
    refresh_sidecars: bool = False


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
    try:
        yield
    finally:
        for _t in (_watch_task, _auto_backend_task):
            _t.cancel()
            with contextlib.suppress(Exception):
                await _t


app = FastAPI(title="coresmithd", version="0.1", lifespan=_lifespan)


@app.get("/healthz")
async def healthz():
    return {
        "ok": True,
        "project_root": _PROJECT_ROOT,
        "status": _pipeline.status,
        "pid": os.getpid(),
    }


@app.get("/run/state")
async def run_state():
    await _pipeline.ensure_graph()
    snap = await _pipeline.graph.aget_state(
        {"configurable": {"thread_id": _pipeline.thread_id}}
    )
    return _shape_state(snap)


@app.post("/run/start")
async def run_start(req: StartRequest):
    global _last_resume_ts
    _last_resume_ts = time.time()  # Section 7b: run start resets the stall clock
    _consumed_interrupt_ids.clear()   # a fresh run answers nothing from the old
    if _pipeline.task is not None and not _pipeline.task.done():
        raise HTTPException(409, "pipeline already running; call /run/pause first")

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

    block_queue = _load_block_queue(req.blocks_file)
    if not block_queue:
        raise HTTPException(
            400,
            "No blocks found. Provide blocks_file or place blocks: in "
            "orchestrator/config.yaml.",
        )

    _preflight_or_400()
    arch_warnings = _check_architecture_artifacts(_PROJECT_ROOT)

    # B3: persist the resolved block queue + initialize the scoreboard schema +
    # snapshot the oracle manifest so the harness (`coresmith verify ...`) can
    # resolve blocks and detect oracle tampering after the daemon parks. All
    # best-effort -- must never block starting a run.
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
    try:
        from orchestrator.state_store.trust import write_oracle_manifest
        write_oracle_manifest(_PROJECT_ROOT)
    except Exception:  # noqa: BLE001
        pass

    await _pipeline.reset_for_new_run()

    events_path = Path(_PROJECT_ROOT) / ".coresmith" / "pipeline_events.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.write_text("")

    initial_state = {
        "project_root": _PROJECT_ROOT,
        "target_clock_mhz": req.target_clock_mhz,
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


@app.post("/run/resume")
async def run_resume(req: ResumeRequest):
    global _last_resume_ts, _consumed_interrupt_ids
    _last_resume_ts = time.time()  # Section 7b: the driver is alive
    with contextlib.suppress(OSError):
        _mk = Path(_PROJECT_ROOT) / "STALLED_INTERRUPT"
        if _mk.exists():
            _mk.unlink()
    await _pipeline.ensure_graph()
    if _pipeline.task is not None and not _pipeline.task.done():
        raise HTTPException(409, "pipeline still running; nothing to resume")

    graph_config = {"configurable": {"thread_id": _pipeline.thread_id}}
    state_snapshot = await _pipeline.graph.aget_state(graph_config)

    # Collect all pending interrupt IDs so parallel-block runs all resume
    # with the same decision unless the caller provided per-block actions.
    interrupts: list[tuple[str, Any]] = []
    interrupt_meta: list[tuple[str, list]] = []
    if state_snapshot and state_snapshot.tasks:
        for task in state_snapshot.tasks:
            for intr in task.interrupts:
                interrupts.append((intr.id, intr.value))
                _val = intr.value if isinstance(intr.value, dict) else {}
                interrupt_meta.append((
                    _val.get("block", _val.get("block_name", "")),
                    _val.get("supported_actions", []),
                ))

    _has_next = bool(state_snapshot and state_snapshot.next)
    _mode = _resume_tick_or_park(bool(interrupts), _has_next)
    if _mode == "none":
        raise HTTPException(409, "no pending interrupt")
    if _mode == "tick":
        # No parked interrupt but the graph still has next nodes: plain tick
        # (cmd=None) to advance a stranded/paused run without a fake action.
        _consumed_interrupt_ids.clear()
        await _pipeline.safe_resume(None, graph_config)
        return {
            "resumed": True,
            "ticked": True,
            "next_nodes": list(state_snapshot.next),
            "action": req.action,
            "status": _pipeline.status,
        }

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

    resume_value: Any = {
        "action": req.action,
        "feedback": req.feedback,
        "rtl_fix_description": req.rtl_fix_description,
        "rationale": req.rationale,
        "block_actions": req.block_actions or {},
    }

    from langgraph.types import Command
    if len(interrupts) > 1:
        cmd = Command(resume={iid: resume_value for iid, _ in interrupts})
    else:
        cmd = Command(resume=resume_value)

    # D5: remember exactly which interrupts this resume answers, BEFORE the
    # graph starts running. Until it checkpoints again, aget_state still returns
    # them; without this record /run/state reports them as pending on a running
    # run and the outer agent resumes a second time.
    _consumed_interrupt_ids = {iid for iid, _ in interrupts}

    await _pipeline.safe_resume(cmd, graph_config)
    return {"resumed": True, "interrupts": len(interrupts), "action": req.action}


@app.post("/run/pause")
async def run_pause():
    if _pipeline.task is None or _pipeline.task.done():
        return {"paused": False, "reason": "no running task"}
    # Reap any in-flight LLM CLI child (codex/claude) AND its whole process
    # group BEFORE cancelling the task. The blocking ``Popen`` runs in a thread
    # executor that ``task.cancel()`` cannot reach, so without this reap the
    # orphaned codex keeps running (2nd live occurrence: it kept burning tokens
    # after a run pause). Pausing mid-LLM-call therefore DISCARDS that call's
    # work -- the interrupted node simply re-runs from the last checkpoint on
    # resume (LangGraph node-boundary semantics), which is correct: a paused
    # in-flight generation was never committed to the graph state.
    try:
        from orchestrator.langchain.agents.coresmith_llm import (
            reap_active_cli_processes,
        )
        reaped = reap_active_cli_processes()
        if reaped:
            log.warning("run/pause reaped %d in-flight CLI process group(s)", reaped)
    except Exception:
        log.warning("run/pause: CLI reap failed", exc_info=True)
    _pipeline.task.cancel()
    try:
        await _pipeline.task
    except (asyncio.CancelledError, Exception):
        pass
    _pipeline.status = "paused"
    return {"paused": True}


@app.post("/run/continue")
async def run_continue():
    """Continue a pipeline that has next_nodes but no pending interrupt.

    Calls graph.ainvoke(None, config) to resume from current checkpoint
    without resetting progress. Safe when status=done but pipeline_done=False.
    """
    if _pipeline.task is not None and not _pipeline.task.done():
        raise HTTPException(409, "pipeline already running")
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
    """Regenerate ONE block from a specific node in its lifecycle.

    The daemon previously had no way to do this. ``/run/restart-node`` forks
    the graph but explicitly REUSES every block's on-disk RTL/TB, and
    ``run start --force`` regenerates the whole design. So after revising a
    uArch spec -- exactly what ``revise_interface`` and an architecture
    revision ask for -- the only HTTP-reachable options were "change nothing"
    or "rebuild everything", and the per-block path existed solely as an MCP
    tool. Observed cost: a spec fix that needed two blocks rebuilt had no way
    to be applied without discarding six good ones.

    ``from_node`` is ``generate_uarch_spec`` or ``generate_rtl``. The block
    runs in a standalone subgraph on its own thread, so the main pipeline
    checkpoint is untouched; re-enter it afterwards with ``/run/restart-node``.
    Requires the pipeline to be idle (pause first).
    """
    if _pipeline.task is not None and not _pipeline.task.done():
        raise HTTPException(409, "pipeline already running -- pause first")
    try:
        # Shared with the MCP tool of the same name: one implementation, two
        # transports. It reads CORESMITH_PROJECT_ROOT, which this daemon sets.
        from orchestrator.mcp_server import restart_block as _restart_block
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"restart_block unavailable: {exc}") from exc
    raw = await _restart_block(
        block_name=req.block_name,
        from_node=req.from_node,
        uarch_feedback=req.uarch_feedback,
        max_attempts=req.max_attempts,
    )
    try:
        return json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError):
        return {"result": raw}


@app.post("/run/restart-node")
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
    refreshed = []
    if req.refresh_sidecars:
        # #6: re-sync intact-RTL blocks' contract sidecars to live before
        # re-entering integration, so the staleness preflight does not force a
        # mass-regen of already-passing blocks.
        try:
            import json as _json

            from orchestrator.langgraph.pipeline_helpers import (
                refresh_current_sidecars,
            )
            bq = os.path.join(_PROJECT_ROOT, ".coresmith", "block_queue.json")
            names = []
            if os.path.exists(bq):
                data = _json.loads(open(bq).read())
                names = [b.get("name") for b in data if b.get("name")]
            refreshed = refresh_current_sidecars(_PROJECT_ROOT, names)
        except Exception:
            log.warning("restart-node: sidecar refresh failed", exc_info=True)
    result = await _pipeline.restart_from_node(req.node)
    if refreshed:
        result["sidecars_refreshed"] = refreshed
    if result.get("error"):
        raise HTTPException(400, result["error"] + (
            " -- " + result["hint"] if result.get("hint") else ""))
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
async def backend_start(req: BackendStartRequest):
    """Enter the backend: flat top synthesis + the chip_top gate-sim verdict.

    Stops there unless ``full=true``. Requires every block to have RTL +
    synthesis artifacts on disk (the shared launcher's own gate) -- so calling
    this before the frontend finished returns the missing-artifact list rather
    than starting a doomed run.
    """
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
    return result


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


@app.post("/backend/pause")
async def backend_pause():
    try:
        _mcp = _backend_handle()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"backend pause unavailable: {exc}") from exc
    handle = _mcp._backend
    if handle.task is None or handle.task.done():
        return {"paused": False, "reason": "no running task"}
    handle.task.cancel()
    try:
        await handle.task
    except (asyncio.CancelledError, Exception):  # noqa: BLE001
        pass
    handle.status = "paused"
    return {"paused": True}


# ---------------------------------------------------------------------------
# Architecture endpoints (PRD -> SAD -> FRD -> ERS -> block_diagram ->
# constraint check -> final review). Produces .coresmith/ers_spec.json
# and .coresmith/block_specs.json that the pipeline phase later consumes.
# ---------------------------------------------------------------------------

@app.post("/architecture/start")
async def architecture_start(req: ArchStartRequest):
    if _architecture.task is not None and not _architecture.task.done():
        raise HTTPException(409, "architecture already running; call /architecture/pause first")

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
    return {
        "started": True,
        "status": _architecture.status,
        "requirements_length": len(requirements),
        "target_clock_mhz": req.target_clock_mhz,
        "pdk_summary": pdk_summary,
    }


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
    return {"resumed": True, "action": resume_value["action"]}


@app.post("/architecture/pause")
async def architecture_pause():
    if _architecture.task is None or _architecture.task.done():
        return {"paused": False, "reason": "no running task"}
    _architecture.task.cancel()
    try:
        await _architecture.task
    except (asyncio.CancelledError, Exception):
        pass
    _architecture.status = "paused"
    return {"paused": True}

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

    block_queue: list[dict] = []
    specs_path = Path(_PROJECT_ROOT) / ".coresmith" / "block_specs.json"
    if not blocks_file and specs_path.exists():
        block_queue = json.loads(specs_path.read_text())

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
    expected = {
        "PRD spec": root / ".coresmith" / "prd_spec.json",
        "ERS spec": root / "arch" / "ers_spec.md",
        "block diagram": root / ".coresmith" / "block_diagram.json",
    }
    missing = [label for label, path in expected.items() if not path.exists()]
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
    consumed = _consumed_now()
    interrupts: list[dict] = []
    suspected_stale = 0
    consumed_count = 0
    if state_snapshot.tasks:
        for task in state_snapshot.tasks:
            for intr in task.interrupts:
                payload = intr.value
                stale = False
                if isinstance(payload, dict):
                    blk = payload.get("block", payload.get("block_name", ""))
                    stale = bool(blk) and blk in completed_names
                if stale:
                    suspected_stale += 1
                was_consumed = intr.id in consumed
                if was_consumed:
                    consumed_count += 1
                interrupts.append({
                    "id": intr.id,
                    "payload": payload,
                    "stale_suspected": stale,
                    "consumed_by_resume": was_consumed,
                })

    pending_interrupts = [i for i in interrupts if not i["consumed_by_resume"]]

    base.update({
        "completed_count": len(latest_by_name),
        "completion_events": len(completed),
        "completed_blocks": [
            {"name": b.get("name"), "success": b.get("success"), "attempts": b.get("attempts", 1)}
            for b in latest_by_name.values()
        ],
        "total_blocks": len(block_queue),
        "remaining_count": max(0, len(block_queue) - len(latest_by_name)),
        "pipeline_done": values.get("pipeline_done", False),
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


def _write_daemon_file(port: int):
    df = _daemon_file()
    df.parent.mkdir(parents=True, exist_ok=True)
    df.write_text(json.dumps({
        "project_root": _PROJECT_ROOT,
        "port": port,
        "pid": os.getpid(),
        "started_at": time.time(),
    }, indent=2))


def _remove_daemon_file():
    """Remove daemon.json only if it still points at *this* process.

    Without the pid guard, a stale daemon that takes a while to finish
    uvicorn shutdown can race with a freshly-spawned replacement: the new
    daemon writes daemon.json, then the old daemon's finally / SIGTERM
    handler fires and deletes it. The CLI then reports `no daemon` while
    the replacement is happily serving HTTP -- which is exactly the bug
    seen on 2026-05-19 in the mcu3 run.
    """
    try:
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
