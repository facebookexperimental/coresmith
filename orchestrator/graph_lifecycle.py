# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""GraphLifecycle -- run a LangGraph state machine as an asyncio background task.

Used by both the MCP server (orchestrator.mcp_server) and the FastAPI daemon
(orchestrator.daemon.server). The two share the same start / resume / pause /
state-recovery semantics; only the transport differs.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import time
import traceback
from typing import Any

logger = logging.getLogger(__name__)


class GraphLifecycle:
    """Manages the lifecycle of a running LangGraph graph.

    Holds the graph object, the AsyncSqlite checkpointer, the running asyncio
    task, and a coarse status string ("idle"/"running"/"interrupted"/"done"/
    "paused"/"error"). All start / resume / pause operations are serialised by
    an asyncio.Lock so concurrent callers cannot launch duplicate tasks.

    Parameters
    ----------
    name: short id for log messages and the LangGraph thread_id.
    checkpoint_db: filesystem path of the SQLite checkpoint database.
    builder_fn_path: dotted import path of the module exposing the graph
        builder ("orchestrator.langgraph.pipeline_graph").
    builder_fn_name: name of the builder function inside that module
        ("build_pipeline_graph").
    project_root: project root used for orphan-event recovery. Pass the same
        directory that holds .coresmith/pipeline_events.jsonl.
    """

    def __init__(
        self,
        name: str,
        checkpoint_db: str,
        builder_fn_path: str,
        builder_fn_name: str,
        project_root: str,
    ):
        self.name = name
        self.checkpoint_db = checkpoint_db
        self._builder_fn_path = builder_fn_path
        self._builder_fn_name = builder_fn_name
        self.project_root = project_root

        self.graph: Any = None
        self.checkpointer: Any = None
        self.task: asyncio.Task | None = None
        self.thread_id: str = name
        self.status: str = "idle"
        self.error_message: str = ""

        self._checkpointer_cm: Any = None
        self._lock = asyncio.Lock()
        self._last_config: dict | None = None
        self._watchdog: asyncio.Task | None = None
        self._watchdog_heals: int = 0

    # -- Recovery helpers ---------------------------------------------------

    def _close_orphaned_events(self) -> None:
        """Close orphaned graph_node_enter events from a prior crash."""
        try:
            from orchestrator.langgraph.event_stream import write_graph_event
            log_path = os.path.join(self.project_root, ".coresmith", "pipeline_events.jsonl")
            if not os.path.isfile(log_path):
                return
            with open(log_path, "r", encoding="utf-8") as fh:
                lines = fh.readlines()
            open_enters: dict[str, dict] = {}
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                etype = ev.get("event", "")
                node = ev.get("node", "")
                if etype == "graph_node_enter" and node:
                    open_enters[node] = ev
                elif etype == "graph_node_exit" and node:
                    open_enters.pop(node, None)
            for node, ev in open_enters.items():
                write_graph_event(self.project_root, node, "graph_node_exit", {
                    "block": ev.get("block", ""),
                    "server_restart": True,
                    "note": "Server restarted; closing orphaned enter event",
                })
        except Exception:
            logging.getLogger(__name__).warning(
                "%s: failed to close orphaned events", self.name, exc_info=True,
            )

    async def ensure_graph(self) -> None:
        """Lazily build the graph and checkpointer (thread-safe)."""
        if self.graph is not None:
            return

        async with self._lock:
            if self.graph is not None:
                return

            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            os.makedirs(os.path.dirname(self.checkpoint_db), exist_ok=True)
            self._checkpointer_cm = AsyncSqliteSaver.from_conn_string(self.checkpoint_db)
            self.checkpointer = await self._checkpointer_cm.__aenter__()

            try:
                await self.checkpointer.conn.execute("PRAGMA journal_mode=WAL")
                await self.checkpointer.conn.execute("PRAGMA synchronous=FULL")
                await self.checkpointer.conn.execute("PRAGMA busy_timeout=5000")
            except Exception:
                logging.getLogger(__name__).warning(
                    "%s: failed to set WAL pragmas", self.name, exc_info=True,
                )

            module = importlib.import_module(self._builder_fn_path)
            builder_fn = getattr(module, self._builder_fn_name)
            self.graph = builder_fn(checkpointer=self.checkpointer)

            # Startup recovery: detect a parked checkpoint from a prior process
            if self.thread_id and self.status == "idle":
                try:
                    config = {"configurable": {"thread_id": self.thread_id}}
                    state = await self.graph.aget_state(config)
                    if state and state.values:
                        if state.tasks:
                            for t in state.tasks:
                                if t.interrupts:
                                    self.status = "interrupted"
                                    break
                        if self.status == "idle":
                            self.status = "done"
                except Exception:
                    logging.getLogger(__name__).warning(
                        "%s: startup recovery check failed", self.name, exc_info=True,
                    )

            self._close_orphaned_events()

    async def cleanup(self) -> None:
        """Close the async SQLite checkpointer."""
        if self._checkpointer_cm is not None:
            try:
                await self._checkpointer_cm.__aexit__(None, None, None)
            except Exception:
                logging.getLogger(__name__).warning(
                    "%s: cleanup failed", self.name, exc_info=True,
                )
            self._checkpointer_cm = None
            self.graph = None
            self.checkpointer = None

    async def reset_for_new_run(self) -> None:
        """Wipe the checkpoint and rebuild the graph for a fresh run."""
        await self.cleanup()
        for suffix in ("", "-wal", "-shm", "-journal"):
            p = self.checkpoint_db + suffix
            if os.path.exists(p):
                os.unlink(p)
        self.graph = None
        self.status = "idle"
        self.error_message = ""
        await self.ensure_graph()

    async def run_task(self, initial_input: Any, config: dict) -> None:
        """Background task: drive the graph until interrupt / completion / error."""
        try:
            from orchestrator.langchain.agents.coresmith_llm import _breaker_context
            _breaker_context.set(self.name)
        except Exception:
            pass
        from langgraph.errors import GraphInterrupt
        try:
            from orchestrator.langchain.agents.coresmith_llm import CircuitBreakerOpen
        except Exception:
            CircuitBreakerOpen = type("CircuitBreakerOpen", (Exception,), {})
        try:
            self.status = "running"
            await self.graph.ainvoke(initial_input, config)
            state = await self.graph.aget_state(config)
            if state and state.tasks:
                for t in state.tasks:
                    if t.interrupts:
                        self.status = "interrupted"
                        return
            self.status = "done"
        except GraphInterrupt:
            self.status = "interrupted"
        except CircuitBreakerOpen:
            self.status = "error"
            self.error_message = traceback.format_exc()[:10000]
        except asyncio.CancelledError:
            self.status = "paused"
        except Exception:
            self.status = "error"
            self.error_message = traceback.format_exc()[:10000]

    # -- Wedge watchdog [dv-hardening-17] ------------------------------------
    #
    # Observed 3x live on armD (arch nodes: Escalate Constraints, interface
    # regen, Final Review): the runner task's ``await graph.ainvoke(...)``
    # parks FOREVER after the node's work has completed and checkpointed --
    # py-spy shows the event loop alive/idle, zero graph work on any thread,
    # no LLM subprocess, no exception anywhere. In-memory status stays
    # "running" (state polls lie) until a daemon restart re-reads the
    # checkpoint, which surfaces the pending interrupt cleanly. Root cause is
    # somewhere in the langgraph/aiosqlite await plumbing (dumps archived);
    # until it is found, this watchdog performs the SAME recovery in-process:
    # detect the wedge signature, cancel the zombie runner, resume from the
    # checkpoint (ainvoke(None) re-surfaces the interrupt).
    #
    # Wedge signature (ALL must hold, conservatively):
    #   - status == "running" and the runner task is alive
    #   - the events file has not been touched for CORESMITH_WEDGE_TIMEOUT_S
    #     (default 900s; heartbeats/graph events reset it)
    #   - the daemon process has NO live children (no codex/claude CLI, no
    #     make/verilator/yosys sims -- anything doing real work is a child)
    # Max 3 self-heals per handle, then status=error with a loud message.
    # CORESMITH_WEDGE_WATCHDOG=0 disables.

    @staticmethod
    def _wedge_watchdog_enabled() -> bool:
        return os.environ.get(
            "CORESMITH_WEDGE_WATCHDOG", "1"
        ).strip().lower() not in ("0", "false", "no", "off")

    @staticmethod
    def _daemon_has_children() -> bool:
        """True when this process has any live (non-zombie) direct child."""
        me = str(os.getpid())
        try:
            for pid in os.listdir("/proc"):
                if not pid.isdigit():
                    continue
                try:
                    with open(f"/proc/{pid}/stat") as fh:
                        parts = fh.read().split()
                    # stat: pid (comm) state ppid ...  -- comm may contain
                    # spaces but is parenthesised; find state after ')'
                    ridx = " ".join(parts).rindex(")")
                    tail = " ".join(parts)[ridx + 1:].split()
                    state, ppid = tail[0], tail[1]
                    if ppid == me and state != "Z":
                        return True
                except (OSError, ValueError, IndexError):
                    continue
        except OSError:
            return True  # cannot scan -> assume busy (never false-positive)
        return False

    def _wedge_suspected(self) -> bool:
        """Pure decision: does the current state match the wedge signature?"""
        if self.status != "running":
            return False
        if self.task is None or self.task.done():
            return False
        try:
            threshold = float(os.environ.get(
                "CORESMITH_WEDGE_TIMEOUT_S", "900") or 900)
        except ValueError:
            threshold = 900.0
        ev = os.path.join(self.project_root, ".coresmith",
                          "pipeline_events.jsonl")
        try:
            age = time.time() - os.path.getmtime(ev)
        except OSError:
            return False  # no events file yet -> too early to judge
        if age < threshold:
            return False
        if self._daemon_has_children():
            return False
        return True

    async def _watchdog_heal_once(self) -> str:
        """Cancel the zombie runner and resume from the checkpoint.

        Returns the action taken ("healed" | "gave_up" | "noop")."""
        if not self._wedge_suspected():
            return "noop"
        self._watchdog_heals += 1
        logger.error(
            "[WEDGE-WATCHDOG] %s: runner task wedged (status=running, no "
            "events, no children) -- self-heal %d/3: cancelling zombie "
            "runner and resuming from checkpoint",
            self.name, self._watchdog_heals,
        )
        task = self.task
        if self._watchdog_heals > 3:
            self.status = "error"
            self.error_message = (
                "wedge watchdog: runner task wedged 4x (langgraph await "
                "plumbing); self-heal limit reached -- restart the daemon "
                "and see /tmp/daemon_hang_*.txt py-spy dumps"
            )
            if task is not None:
                task.cancel()
            return "gave_up"
        if task is not None:
            task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=10)
            except (TimeoutError, asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        cfg = self._last_config or {
            "configurable": {"thread_id": self.thread_id}}
        async with self._lock:
            if self.task is not None and not self.task.done():
                return "noop"  # someone else resumed meanwhile
            self.task = asyncio.create_task(self.run_task(None, cfg))
        return "healed"

    async def _wedge_watchdog_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(60)
                t = self.task
                if t is None or t.done():
                    # runner finished normally; watchdog retires until the
                    # next safe_start/safe_resume re-arms it
                    return
                action = await self._watchdog_heal_once()
                if action == "gave_up":
                    return
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("[WEDGE-WATCHDOG] %s: watchdog crashed", self.name)

    def _arm_watchdog(self) -> None:
        if not self._wedge_watchdog_enabled():
            return
        if self._watchdog is not None and not self._watchdog.done():
            return
        self._watchdog = asyncio.create_task(self._wedge_watchdog_loop())

    async def safe_start(self, initial_input: Any, config: dict) -> None:
        """Spawn a fresh run_task (raises if one is already in flight)."""
        async with self._lock:
            if self.task is not None and not self.task.done():
                raise RuntimeError(f"{self.name} graph is already running")
            self._last_config = config
            self.task = asyncio.create_task(self.run_task(initial_input, config))
        self._arm_watchdog()

    async def safe_resume(self, resume_input: Any, config: dict) -> None:
        """Resume after an interrupt (raises if a task is already running)."""
        async with self._lock:
            if self.task is not None and not self.task.done():
                raise RuntimeError(f"{self.name} graph is already running")
            self._last_config = config
            self.task = asyncio.create_task(self.run_task(resume_input, config))
        self._arm_watchdog()

    async def restart_from_node(self, node_name: str) -> dict:
        """Re-run the graph from the checkpoint where ``node_name`` is next.

        The targeted counterpart of ``run start --force`` (engine follow-up
        #8/#10). ``--force`` restarts the WHOLE pipeline and pass-1 regenerates
        every uarch spec unconditionally, so a late-stage re-entry (e.g. re-run
        a skipped ``integration_check``, or re-drive ``validation_dv`` after a
        hand-patch) forces a full block-regen gauntlet that re-opens resolved
        feasibility/interface battles. This forks from the historical
        checkpoint whose ``next`` includes ``node_name`` and re-invokes from
        there, so every block's already-on-disk RTL/TB is reused verbatim --
        only ``node_name`` onward re-runs.

        Returns ``{restarted, node, checkpoint_id}`` or ``{error, ...}``.
        Requires the graph to be idle (not currently running).
        """
        if self.task is not None and not self.task.done():
            return {"error": "graph is already running -- pause first"}
        await self.ensure_graph()
        config = {"configurable": {"thread_id": self.thread_id}}
        found = None
        async for snap in self.graph.aget_state_history(config):
            if snap.next and node_name in snap.next:
                found = snap.config
                break
        if not found:
            return {
                "error": f"no checkpoint found with '{node_name}' as the next "
                         "node",
                "hint": "inspect the pipeline_events.jsonl timeline for the "
                        "exact node name (e.g. integration_check, "
                        "integration_dv, validation_dv, process_block).",
            }
        ckpt_id = found["configurable"].get("checkpoint_id")
        fork = {"configurable": {"thread_id": self.thread_id,
                                 "checkpoint_id": ckpt_id}}
        await self.safe_start(None, fork)
        return {"restarted": True, "node": node_name, "checkpoint_id": ckpt_id}
