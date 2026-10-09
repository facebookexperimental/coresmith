"""Daemon-owned lifecycle supervision (opt-in: ``CORESMITH_SUPERVISOR=1``).

One responsibility: once the stage machine has reached ``blocks`` and no run
exists yet, start the frontend pipeline once (a durable, budgeted handoff).
It never launches, resumes or decides for an Architect -- the Architect is
the coding agent outside the engine that drives ``coresmith`` -- and never
answers design interrupts. A cluster worker's provider/tool failure is
reported as infrastructure state, never as a design verdict.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

_lock = asyncio.Lock()


def _read(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


def classify(pipeline, failure):
    if failure:
        return "blocked", failure.get("reason", "agent unavailable")
    if any(not i.get("consumed_by_resume") for i in pipeline.get("interrupts", [])):
        return "awaiting_decision", "pending review or question"
    if pipeline.get("frontend_outcome") == "failed":
        return "blocked", "frontend completed with failed blocks"
    if pipeline.get("frontend_outcome") == "complete":
        return "frontend_complete", "frontend complete; chip acceptance/backend must be checked separately"
    if pipeline.get("status") in ("error", "paused", "done"):
        return "blocked", pipeline.get("error_message") or "pipeline stopped before completion"
    return "running", ""


async def tick(server):
    """Idempotent, serialized check; all durable decisions live in ProjectDB."""
    async with _lock:
        db = server._project_db()
        if os.environ.get("CORESMITH_SUPERVISOR", "0") != "1":
            return {"state": "disabled"}
        from orchestrator.state_store.stages import status as stage_status
        root = Path(server._PROJECT_ROOT)
        now = time.time()
        stage = stage_status(db, root)
        pipeline = await server.run_state()
        failure = db.get_flag("agent_failure")
        # Cluster provider/tool failure is infrastructure state, not a design vote.
        if not failure:
            for path in (root / ".coresmith/clusters").glob("*/status.json"):
                worker = _read(path)
                if worker.get("state") in ("provider_blocked", "tool_failed"):
                    failure = {"kind": worker["state"], "reason": worker.get("stop_reason"),
                               "worker": path.parent.name}
                    break
        state, reason = classify(pipeline, failure)
        actions = []
        if state == "running" and not pipeline.get("pipeline_done"):
            if stage["stage"] == "blocks" and pipeline.get("values_empty"):
                prior = db.get_flag("supervisor_launch") or {}
                if not prior:
                    db.set_flag("supervisor_launch", {"ts": now, "state": "attempting"})
                    try:
                        result = await server.run_start(server.StartRequest())
                        if not isinstance(result, dict) or not result.get("started"):
                            raise RuntimeError(str(result))
                        db.set_flag("supervisor_launch", {"ts": now, "state": "started"})
                        actions.append("pipeline_started")
                        pipeline = await server.run_state()
                    except Exception as exc:
                        reason = f"pipeline launch failed: {exc}"
                        db.set_flag("supervisor_launch", {"ts": now, "state": "failed", "reason": reason})
                        state = "blocked"
                elif prior.get("state") != "started":
                    state, reason = "blocked", prior.get("reason", "pipeline launch interrupted; inspect checkpoint")
            elif stage["index"] < stage["stages"].index("blocks"):
                # Before ``blocks`` the Architect drives ``coresmith stage next``;
                # the supervisor launches nothing and waits.
                reason = f"stage {stage['stage']}: waiting for the Architect to reach blocks"
        previous = db.get_flag("supervisor_status") or {}
        progress = [stage["stage"], pipeline.get("passed_count", 0), pipeline.get("failed_count", 0)]
        last_progress = now if progress != previous.get("progress") else previous.get("last_progress_ts", now)
        record = {"state": state, "reason": reason, "ts": now, "stage": stage["stage"],
                  "progress": progress, "last_progress_ts": last_progress,
                  "passed": pipeline.get("passed_count", 0), "failed": pipeline.get("failed_count", 0),
                  "pending": pipeline.get("pending_count"), "pipeline_status": pipeline.get("status"),
                  "actions": actions, "failure": failure}
        db.set_flag("supervisor_status", record)
        directory = root / ".coresmith/supervisor"
        directory.mkdir(parents=True, exist_ok=True)
        tmp = directory / "status.tmp"
        tmp.write_text(json.dumps(record, indent=2))
        tmp.replace(directory / "status.json")
        if (state, reason) != (previous.get("state"), previous.get("reason")) or actions:
            with (directory / "events.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            if state == "blocked":
                server._daemon_log("error", "SUPERVISOR BLOCKED: %s", reason)
        return record


async def watch(server):
    while True:
        try:
            await tick(server)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            server._daemon_log("error", "SUPERVISOR CHECK FAILED: %s", exc)
            server._project_db().set_flag("supervisor_status", {
                "state": "blocked", "reason": f"supervisor check failed: {exc}", "ts": time.time()})
        await asyncio.sleep(30)
