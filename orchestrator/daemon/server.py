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
import json
import logging
import os
import signal
import socket
import sys
import time
from pathlib import Path
from typing import Any, Optional

# Resolve project root + telemetry before importing any graph code.
_PROJECT_ROOT = os.environ.get(
    "CORESMITH_PROJECT_ROOT",
    str(Path(__file__).resolve().parent.parent.parent),
)
os.environ["CORESMITH_PROJECT_ROOT"] = _PROJECT_ROOT

from orchestrator.telemetry import init_telemetry  # noqa: E402

init_telemetry(_PROJECT_ROOT)

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402
import uvicorn  # noqa: E402

from orchestrator.graph_lifecycle import GraphLifecycle  # noqa: E402

log = logging.getLogger("coresmithd")
log.setLevel(logging.INFO)


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


class ResumeRequest(BaseModel):
    action: str = "approve"
    feedback: str = ""
    rtl_fix_description: str = ""
    block_actions: Optional[dict] = None
    rationale: str = ""


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


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="coresmithd", version="0.1")


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
    if _pipeline.task is not None and not _pipeline.task.done():
        raise HTTPException(409, "pipeline already running; call /run/pause first")

    block_queue = _load_block_queue(req.blocks_file)
    if not block_queue:
        raise HTTPException(
            400,
            "No blocks found. Provide blocks_file or place blocks: in "
            "orchestrator/config.yaml.",
        )

    _preflight_or_400()
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
    return {"started": True, "block_count": len(block_queue), "status": _pipeline.status}


@app.post("/run/resume")
async def run_resume(req: ResumeRequest):
    await _pipeline.ensure_graph()
    if _pipeline.task is not None and not _pipeline.task.done():
        raise HTTPException(409, "pipeline still running; nothing to resume")

    state_snapshot = await _pipeline.graph.aget_state(
        {"configurable": {"thread_id": _pipeline.thread_id}}
    )
    if not state_snapshot or not state_snapshot.tasks:
        raise HTTPException(409, "no pending interrupt")

    # Collect all pending interrupt IDs so parallel-block runs all resume
    # with the same decision unless the caller provided per-block actions.
    interrupts: list[tuple[str, Any]] = []
    for task in state_snapshot.tasks:
        for intr in task.interrupts:
            interrupts.append((intr.id, intr.value))
    if not interrupts:
        raise HTTPException(409, "no pending interrupt")

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

    graph_config = {"configurable": {"thread_id": _pipeline.thread_id}}
    await _pipeline.safe_resume(cmd, graph_config)
    return {"resumed": True, "interrupts": len(interrupts), "action": req.action}


@app.post("/run/pause")
async def run_pause():
    if _pipeline.task is None or _pipeline.task.done():
        return {"paused": False, "reason": "no running task"}
    _pipeline.task.cancel()
    try:
        await _pipeline.task
    except (asyncio.CancelledError, Exception):
        pass
    _pipeline.status = "paused"
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
        load_config,
        get_sorted_block_queue,
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
    completed_names = {b.get("name") for b in completed if b.get("name")}

    interrupts: list[dict] = []
    if state_snapshot.tasks:
        for task in state_snapshot.tasks:
            for intr in task.interrupts:
                payload = intr.value
                if isinstance(payload, dict):
                    blk = payload.get("block", payload.get("block_name", ""))
                    if blk and blk in completed_names:
                        continue  # stale
                interrupts.append({"id": intr.id, "payload": payload})

    base.update({
        "completed_count": len(completed),
        "completed_blocks": [
            {"name": b.get("name"), "success": b.get("success"), "attempts": b.get("attempts", 1)}
            for b in completed
        ],
        "total_blocks": len(block_queue),
        "remaining_count": len(block_queue) - len(completed),
        "pipeline_done": values.get("pipeline_done", False),
        "next_nodes": list(state_snapshot.next) if state_snapshot.next else [],
        "interrupts": interrupts,
        "pending_interrupt_count": len(interrupts),
        "interrupt_type": (
            interrupts[0]["payload"].get("type", "")
            if interrupts and isinstance(interrupts[0]["payload"], dict)
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
