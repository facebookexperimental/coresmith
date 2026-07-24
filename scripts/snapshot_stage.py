#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Freeze a parked run into a stage fixture (Package C, C5).

Captures ``(checkpoint.db + artifacts + manifest)`` from a run parked at a known
point so tests can resume it in seconds via
``orchestrator.testing.stage_fixtures.materialize_stage``.

Safety:
  * REFUSES to snapshot a run that is not parked (still running / already done)
    unless ``--allow-unparked`` -- a mid-flight checkpoint is inconsistent.
  * The checkpoint is copied with ``sqlite3 .backup`` (which MERGES the WAL into
    a single consistent file) then ``VACUUM``'d -- never a raw file copy that
    could drop uncommitted WAL frames.

Per-stage artifact profiles (C5): only the artifacts a downstream resume needs
are captured, so fixtures stay small.

Usage:
    python3 scripts/snapshot_stage.py --project-root <run-dir> \\
        --stage post-uarch-pass1 --graph pipeline --thread-id <tid> \\
        --out orchestrator/tests/fixtures/stage/<name>
"""

from __future__ import annotations

import argparse
import asyncio
import fnmatch
import json
import shutil
import sqlite3
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Per-stage artifact globs (relative to project root). Later stages are supersets.
ARTIFACT_PROFILES: dict[str, list[str]] = {
    "post-arch": [
        ".coresmith/block_specs.json", ".coresmith/block_diagram.json",
        ".coresmith/prd_spec.json", ".coresmith/memory_map.json",
        "arch/prd_spec.md", "arch/sad_spec.md", "arch/frd_spec.md",
        "arch/ers_spec.md", "inputs/*",
    ],
    "post-uarch": [
        ".coresmith/block_specs.json", ".coresmith/block_diagram.json",
        ".coresmith/prd_spec.json", ".coresmith/_chip_model.py",
        "arch/*.md", "arch/uarch_specs/*.md", "arch/block_models/*",
        "blocks/**/*.json",
    ],
    "post-rtl": [
        ".coresmith/block_specs.json", "arch/*.md", "arch/uarch_specs/*.md",
        "arch/block_models/*", "blocks/**/*.json", "rtl/**/*.v", "tb/**/*.py",
    ],
}
# The block subgraph resumes from a per-block checkpoint; it needs the spec.
ARTIFACT_PROFILES["post-uarch-block"] = ["arch/uarch_specs/*.md", "arch/*.md"]

_DEFAULT_CKPT = {
    "pipeline": ".coresmith/pipeline_checkpoint.db",
    "architecture": ".coresmith/architecture_checkpoint.db",
    "backend": ".coresmith/backend_checkpoint.db",
    "block": ".coresmith/pipeline_checkpoint.db",
}


def _iter_profile_files(project_root: Path, stage: str, extra) -> list[Path]:
    patterns = list(ARTIFACT_PROFILES.get(stage, [])) + list(extra or [])
    out: list[Path] = []
    seen: set[Path] = set()
    for pat in patterns:
        for p in project_root.glob(pat):
            if p.is_file() and p not in seen:
                seen.add(p)
                out.append(p)
    return out


def backup_checkpoint(src_db: Path, dst_db: Path) -> None:
    """Consistent single-file copy: .backup (merges WAL) + VACUUM."""
    dst_db.parent.mkdir(parents=True, exist_ok=True)
    if dst_db.exists():
        dst_db.unlink()
    src = sqlite3.connect(str(src_db))
    try:
        dst = sqlite3.connect(str(dst_db))
        try:
            src.backup(dst)
            dst.execute("VACUUM")
            dst.commit()
        finally:
            dst.close()
    finally:
        src.close()


def _run_coro(coro):
    """Run ``coro`` to completion whether or not a loop is already running.

    ``snapshot`` is a sync CLI helper but is also called from async tests; when a
    loop is active we run the coroutine in a fresh loop on a worker thread so we
    never trip ``asyncio.run() cannot be called from a running event loop``.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(lambda: asyncio.run(coro)).result()


async def _read_park_state(ckpt_db: Path, thread_id: str, graph: str):
    """Return (parked: bool, pending_interrupt: dict|None) from the checkpoint."""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    from orchestrator.testing.stage_fixtures import _build_graph

    cm = AsyncSqliteSaver.from_conn_string(str(ckpt_db))
    saver = await cm.__aenter__()
    try:
        g = _build_graph(graph, saver)
        snap = await g.aget_state({"configurable": {"thread_id": thread_id}})
        interrupt = None
        for t in (snap.tasks or []):
            for it in (t.interrupts or []):
                interrupt = {"kind": "interrupt", "value": it.value}
                break
            if interrupt:
                break
        nxt = list(snap.next or ())
        parked = bool(interrupt) or bool(nxt)
        if interrupt is None and nxt:
            interrupt = {"kind": "node_boundary", "next": nxt}
        return parked, interrupt, bool(snap.values)
    finally:
        await cm.__aexit__(None, None, None)


def _engine_commit() -> str:
    try:
        import subprocess

        return subprocess.check_output(
            ["git", "-C", str(_REPO), "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def snapshot(
    *,
    project_root,
    out,
    stage: str,
    graph: str = "pipeline",
    thread_id: str = "",
    checkpoint_db=None,
    env: dict | None = None,
    extra_artifacts=(),
    require_parked: bool = True,
    original_root: str | None = None,
) -> dict:
    """Build a stage fixture at ``out``. Returns the manifest dict."""
    from orchestrator.testing.stage_fixtures import langgraph_version, schema_fingerprint

    project_root = Path(project_root)
    out = Path(out)
    ckpt_src = (
        Path(checkpoint_db) if checkpoint_db
        else project_root / _DEFAULT_CKPT.get(graph, _DEFAULT_CKPT["pipeline"])
    )
    has_checkpoint = ckpt_src.exists()

    pending_interrupt = None
    if has_checkpoint:
        parked, pending_interrupt, has_state = _run_coro(
            _read_park_state(ckpt_src, thread_id, graph)
        )
        if require_parked and not parked:
            raise SystemExit(
                f"refusing to snapshot: run at {project_root} is not parked "
                f"(no pending interrupt / next node). Use --allow-unparked to override."
            )

    out.mkdir(parents=True, exist_ok=True)
    art_dir = out / "artifacts"
    if art_dir.exists():
        shutil.rmtree(art_dir)
    art_dir.mkdir(parents=True)

    artifacts: list[str] = []
    for f in _iter_profile_files(project_root, stage, extra_artifacts):
        rel = f.relative_to(project_root)
        tgt = art_dir / rel
        tgt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, tgt)
        artifacts.append(str(rel))

    if has_checkpoint:
        backup_checkpoint(ckpt_src, out / "checkpoint.db")

    manifest = {
        "stage": stage,
        "graph": graph,
        "thread_id": thread_id,
        "schema_fingerprint": schema_fingerprint(graph),
        "langgraph_version": langgraph_version(),
        "engine_commit": _engine_commit(),
        "original_root": original_root if original_root is not None else str(project_root),
        "pending_interrupt": pending_interrupt,
        "env": env or {},
        "artifacts": sorted(artifacts),
        "has_checkpoint": has_checkpoint,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", required=True, choices=sorted(ARTIFACT_PROFILES))
    ap.add_argument("--graph", default="pipeline",
                    choices=["pipeline", "architecture", "backend", "block"])
    ap.add_argument("--thread-id", default="")
    ap.add_argument("--checkpoint-db", default=None)
    ap.add_argument("--allow-unparked", action="store_true")
    ap.add_argument("--original-root", default=None)
    ap.add_argument("--env", action="append", default=[],
                    help="KEY=VALUE env flag to record in the manifest (repeatable)")
    args = ap.parse_args(argv)

    env = {}
    for kv in args.env:
        if "=" in kv:
            k, v = kv.split("=", 1)
            env[k] = v

    manifest = snapshot(
        project_root=args.project_root, out=args.out, stage=args.stage,
        graph=args.graph, thread_id=args.thread_id, checkpoint_db=args.checkpoint_db,
        env=env, require_parked=not args.allow_unparked,
        original_root=args.original_root,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
