# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Block-queue resolution for the harness (mirrors the daemon's loader).

Resolution order (first hit wins):
  1. ``.coresmith/block_specs.json``      (architecture-phase output)
  2. ``.coresmith/block_queue.json``      (persisted at ``/run/start``)
  3. ``$CORESMITH_BLOCKS_FILE``           (explicit blocks.yaml override)
  4. ``load_config()`` -> ``get_sorted_block_queue`` (config.yaml fallback)

All imports of langgraph helpers are deferred so ``harness.cli`` (which imports
this module for the read-only subcommands) stays langgraph-free at import time.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def _retire_pin_mapped(project_root: str | Path, queue: list[dict]) -> list[dict]:
    """Drop a pad-adapter block a declared PRD pin map fully covers.

    The graph retires it at ``init_tier`` (and owns the partial-coverage park);
    this mirrors ONLY the unambiguous full-coverage case so every other reader
    of the queue -- the conformance stage's sibling list, ``coresmith verify``,
    the MCP status views -- reports the same block set the pipeline runs. Never
    parks and never raises: a bookkeeping helper must not gate anything.
    """
    try:
        from orchestrator.architecture import pin_map_retire as _pmr
        if not _pmr.retirement_enabled():
            return queue
        plan = _pmr.plan_retirement(project_root, queue)
        return _pmr.apply_retirement(queue, plan) if plan.retire else queue
    except Exception:  # noqa: BLE001
        return queue


def load_block_queue(project_root: str | Path) -> list[dict]:
    """Return the resolved block queue (list of block spec dicts)."""
    root = Path(project_root)
    return _retire_pin_mapped(root, _load_block_queue_raw(root))


def _load_block_queue_raw(project_root: str | Path) -> list[dict]:
    """The queue exactly as the architecture phase / config declared it."""
    root = Path(project_root)

    specs = root / ".coresmith" / "block_specs.json"
    if specs.exists():
        try:
            data = json.loads(specs.read_text())
            if data:
                return data
        except Exception:  # noqa: BLE001
            pass

    queue = root / ".coresmith" / "block_queue.json"
    if queue.exists():
        try:
            data = json.loads(queue.read_text())
            if data:
                return data
        except Exception:  # noqa: BLE001
            pass

    blocks_file = os.environ.get("CORESMITH_BLOCKS_FILE", "").strip()
    if blocks_file:
        bf = Path(blocks_file)
        if not bf.is_absolute():
            bf = root / blocks_file
        if bf.exists():
            try:
                from orchestrator.langgraph.pipeline_helpers import (
                    get_sorted_block_queue,
                    load_config,
                )
                os.environ["CORESMITH_BLOCKS_FILE"] = str(bf)
                return get_sorted_block_queue(load_config())
            except Exception:  # noqa: BLE001
                pass

    try:
        from orchestrator.langgraph.pipeline_helpers import (
            get_sorted_block_queue,
            load_config,
        )
        return get_sorted_block_queue(load_config())
    except Exception:  # noqa: BLE001
        return []


def load_block_spec(project_root: str | Path, name: str) -> dict | None:
    """The single block spec dict named ``name`` (or ``None`` if absent)."""
    for spec in load_block_queue(project_root):
        if isinstance(spec, dict) and spec.get("name") == name:
            return spec
    return None


def block_names(project_root: str | Path) -> list[str]:
    return [
        str(b.get("name"))
        for b in load_block_queue(project_root)
        if isinstance(b, dict) and b.get("name")
    ]


def persist_block_queue(project_root: str | Path, queue: list[dict]) -> bool:
    """Snapshot the resolved block queue to ``.coresmith/block_queue.json``.

    Called at ``/run/start`` so the harness can resolve blocks after the run
    began (esp. for runs driven from a blocks.yaml, whose queue would otherwise
    live only in checkpoint state). Best-effort -> returns success bool.
    """
    try:
        path = Path(project_root) / ".coresmith" / "block_queue.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(queue))
        return True
    except Exception:  # noqa: BLE001
        return False
