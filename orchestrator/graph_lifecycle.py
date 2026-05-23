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
import traceback
from typing import Any, Optional


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
        self.task: Optional[asyncio.Task] = None
        self.thread_id: str = name
        self.status: str = "idle"
        self.error_message: str = ""

        self._checkpointer_cm: Any = None
        self._lock = asyncio.Lock()

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

    async def safe_start(self, initial_input: Any, config: dict) -> None:
        """Spawn a fresh run_task (raises if one is already in flight)."""
        async with self._lock:
            if self.task is not None and not self.task.done():
                raise RuntimeError(f"{self.name} graph is already running")
            self.task = asyncio.create_task(self.run_task(initial_input, config))

    async def safe_resume(self, resume_input: Any, config: dict) -> None:
        """Resume after an interrupt (raises if a task is already running)."""
        async with self._lock:
            if self.task is not None and not self.task.done():
                raise RuntimeError(f"{self.name} graph is already running")
            self.task = asyncio.create_task(self.run_task(resume_input, config))
