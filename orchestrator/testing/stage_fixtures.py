# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Stage fixtures: resume the pipeline from a captured checkpoint (Package C, C5).

A full arch->synth run is 4-6h; iterating on a *late* bug (integration DV, a
pass-2 RTL failure) means paying that cost every time. A **stage fixture** is a
frozen ``(checkpoint.db + artifacts + manifest)`` snapshot of a run parked at a
known point; :func:`materialize_stage` rehydrates it into a throwaway project
root so a test resumes the graph from there in seconds -- no live LLM, no EDA.

Format (produced by ``scripts/snapshot_stage.py``)::

    <fixture>/
      manifest.json   # stage, graph, thread_id, schema_fingerprint,
                      # langgraph_version, engine_commit, original_root,
                      # pending_interrupt, env, artifacts, has_checkpoint
      checkpoint.db   # (only when has_checkpoint) VACUUM'd single-file sqlite
      artifacts/      # tree rooted at the project root (relative paths)

Staleness policy: the checkpoint's schema is only valid for the exact graph
topology + state schema + langgraph version it was captured under. On a
mismatch, :func:`materialize_stage` calls ``pytest.skip`` (PRs stay green while
the fixture is regenerated) unless ``CORESMITH_STAGE_STRICT=1`` (nightly), which
turns the mismatch into a hard failure so a drifted fixture is noticed.

Async care (C7): the ``AsyncSqliteSaver`` is opened here and MUST be closed via
``StageContext.aclose()`` or pytest hangs at exit. The ``from_stage`` conftest
fixture closes every context it hands out.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# State channels merged with an append reducer -- must never be overwritten by
# the root-rewrite aupdate_state (it would DOUBLE the list).
_APPEND_KEYS = {"completed_blocks"}


def langgraph_version() -> str:
    try:
        import importlib.metadata as m

        return m.version("langgraph")
    except Exception:
        return "unknown"


def _graph_meta(graph_name: str):
    """Return (sorted node names, sorted state field names) for a graph kind."""
    from langgraph.checkpoint.memory import MemorySaver

    if graph_name == "block":
        from orchestrator.langgraph.pipeline_graph import (
            BlockState,
            build_block_subgraph,
        )
        g = build_block_subgraph().compile(checkpointer=MemorySaver())
        fields = set(BlockState.__annotations__)
    elif graph_name == "pipeline":
        from orchestrator.langgraph.pipeline_graph import (
            BlockState,
            OrchestratorState,
            build_pipeline_graph,
        )
        g = build_pipeline_graph(checkpointer=MemorySaver())
        fields = set(OrchestratorState.__annotations__) | set(BlockState.__annotations__)
    elif graph_name == "architecture":
        from orchestrator.langgraph.architecture_graph import (
            ArchGraphState,
            build_architecture_graph,
        )
        g = build_architecture_graph(checkpointer=MemorySaver())
        fields = set(ArchGraphState.__annotations__)
    else:
        raise ValueError(f"unknown graph kind {graph_name!r}")
    return sorted(g.nodes.keys()), sorted(fields)


def schema_fingerprint(graph_name: str = "block") -> str:
    """Stable 16-hex digest of (graph node names, state fields, langgraph ver).

    Two runs of the same engine on the same langgraph produce the same digest;
    a topology or schema change flips it, invalidating stale checkpoints.
    """
    nodes, fields = _graph_meta(graph_name)
    blob = json.dumps(
        {"graph": graph_name, "nodes": nodes, "fields": fields,
         "langgraph": langgraph_version()},
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _build_graph(graph_name: str, saver):
    if graph_name == "block":
        from orchestrator.langgraph.pipeline_graph import build_block_subgraph
        return build_block_subgraph().compile(checkpointer=saver)
    if graph_name == "pipeline":
        from orchestrator.langgraph.pipeline_graph import build_pipeline_graph
        return build_pipeline_graph(checkpointer=saver)
    if graph_name == "architecture":
        from orchestrator.langgraph.architecture_graph import build_architecture_graph
        return build_architecture_graph(checkpointer=saver)
    raise ValueError(f"unknown graph kind {graph_name!r}")


def _stage_strict() -> bool:
    return os.environ.get("CORESMITH_STAGE_STRICT", "0").strip().lower() in (
        "1", "true", "yes",
    )


def _rewrite(obj: Any, old: str, new: str) -> Any:
    if isinstance(obj, str):
        return obj.replace(old, new) if old and old in obj else obj
    if isinstance(obj, list):
        return [_rewrite(x, old, new) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_rewrite(x, old, new) for x in obj)
    if isinstance(obj, dict):
        return {k: _rewrite(v, old, new) for k, v in obj.items()}
    return obj


@dataclass
class StageContext:
    """A materialized stage: a project root + (optionally) a resumable graph."""

    project_root: str
    manifest: dict
    graph: Any = None
    config: Optional[dict] = None
    artifacts_copied: list = field(default_factory=list)
    _saver_cm: Any = None
    _saver: Any = None

    async def aget_state(self):
        return await self.graph.aget_state(self.config)

    async def aclose(self) -> None:
        """Close the AsyncSqliteSaver (avoids pytest hang-at-exit)."""
        if self._saver_cm is not None:
            try:
                await self._saver_cm.__aexit__(None, None, None)
            finally:
                self._saver_cm = None
                self._saver = None
                self.graph = None


async def materialize_stage(
    fixture_dir,
    project_root,
    monkeypatch,
    *,
    strict: Optional[bool] = None,
) -> StageContext:
    """Rehydrate a stage fixture into ``project_root`` and return a StageContext.

    Copies the fixture artifacts under ``project_root``, sets the manifest env +
    ``CORESMITH_PROJECT_ROOT`` and monkeypatches ``pipeline_helpers.PROJECT_ROOT``
    + ``pipeline_graph.PROJECT_ROOT`` (both freeze at import); on a checkpoint
    fixture it opens an ``AsyncSqliteSaver`` on a *copy* of the checkpoint and
    rewrites ``original_root -> project_root`` across the state via
    ``aupdate_state`` (append-reducer channels are left untouched). A schema
    fingerprint mismatch -> ``pytest.skip`` (or raise under ``CORESMITH_STAGE_STRICT``).
    """
    fixture_dir = Path(fixture_dir)
    project_root = str(project_root)
    manifest = json.loads((fixture_dir / "manifest.json").read_text())

    pr = Path(project_root)
    (pr / ".coresmith").mkdir(parents=True, exist_ok=True)

    # 1. artifacts
    copied: list[str] = []
    art_root = fixture_dir / "artifacts"
    if art_root.is_dir():
        for src in art_root.rglob("*"):
            if src.is_file():
                rel = src.relative_to(art_root)
                tgt = pr / rel
                tgt.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, tgt)
                copied.append(str(rel))

    # 2. env + frozen PROJECT_ROOTs
    for k, v in (manifest.get("env") or {}).items():
        monkeypatch.setenv(k, str(v))
    monkeypatch.setenv("CORESMITH_PROJECT_ROOT", project_root)
    import orchestrator.langgraph.pipeline_graph as pg
    import orchestrator.langgraph.pipeline_helpers as ph
    monkeypatch.setattr(ph, "PROJECT_ROOT", pr)
    monkeypatch.setattr(pg, "PROJECT_ROOT", pr)

    graph_name = manifest.get("graph", "block")
    strict = _stage_strict() if strict is None else strict
    ctx = StageContext(project_root=project_root, manifest=manifest,
                       artifacts_copied=copied)

    # No checkpoint (e.g. post-arch: the pipeline restarts fresh from
    # block_specs.json) -> the fingerprint is informational; artifacts are
    # schema-stable JSON, so we don't gate on the graph/langgraph version.
    if not manifest.get("has_checkpoint"):
        return ctx

    # 3. fingerprint / staleness -- enforced ONLY for checkpoint fixtures, whose
    # binary state is valid only for the exact topology + schema + langgraph it
    # was captured under.
    expected_fp = manifest.get("schema_fingerprint")
    if expected_fp:
        current_fp = schema_fingerprint(graph_name)
        if current_fp != expected_fp:
            msg = (
                f"stage fixture {fixture_dir.name!r} schema drift: manifest "
                f"fingerprint {expected_fp} != current {current_fp} "
                f"(graph={graph_name}, langgraph={langgraph_version()})"
            )
            if strict:
                raise RuntimeError(msg)
            import pytest
            pytest.skip(msg)

    # 4. checkpoint: open on a COPY (never mutate the committed fixture db)
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    ckpt_dst = pr / ".coresmith" / f"{graph_name}_stage_checkpoint.db"
    shutil.copy2(fixture_dir / "checkpoint.db", ckpt_dst)
    cm = AsyncSqliteSaver.from_conn_string(str(ckpt_dst))
    saver = await cm.__aenter__()
    graph = _build_graph(graph_name, saver)
    config = {"configurable": {"thread_id": manifest["thread_id"]}}
    ctx.graph = graph
    ctx.config = config
    ctx._saver_cm = cm
    ctx._saver = saver

    # 5. rewrite original_root -> project_root across the checkpointed state
    orig = manifest.get("original_root", "")
    if orig and orig != project_root:
        snap = await graph.aget_state(config)
        values = dict(snap.values or {})
        updates = {}
        for k, v in values.items():
            if k in _APPEND_KEYS:
                continue
            nv = _rewrite(v, orig, project_root)
            if nv != v:
                updates[k] = nv
        if updates:
            await graph.aupdate_state(config, updates)

    return ctx
