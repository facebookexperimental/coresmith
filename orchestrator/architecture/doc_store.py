# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Per-document file I/O layer for architecture artifacts.

Provides reader functions for each architecture document type. Writers
are co-located with the graph nodes in architecture_graph.py (the
``_persist_*`` helpers).

All readers return ``None`` when the file is missing or unparseable,
making them safe to call unconditionally.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def _read_json(path: Path) -> dict | list | None:
    """Read and parse a JSON file, returning None on any failure."""
    try:
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return None
        return json.loads(text)
    except (json.JSONDecodeError, OSError) as exc:
        log.debug("Failed to read %s: %s", path, exc)
        return None


def _read_text(path: Path) -> str | None:
    """Read a text file, returning None if missing or empty."""
    try:
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8")
        return text if text.strip() else None
    except OSError as exc:
        log.debug("Failed to read %s: %s", path, exc)
        return None


def read_prd(project_root: str) -> dict | None:
    return _read_json(Path(project_root) / ".coresmith" / "prd_spec.json")


def read_sad(project_root: str) -> str | None:
    return _read_text(Path(project_root) / "arch" / "sad_spec.md")


def read_frd(project_root: str) -> str | None:
    return _read_text(Path(project_root) / "arch" / "frd_spec.md")


def read_block_diagram(project_root: str) -> dict | None:
    return _read_json(Path(project_root) / ".coresmith" / "block_diagram.json")


def read_memory_map(project_root: str) -> dict | None:
    return _read_json(Path(project_root) / ".coresmith" / "memory_map.json")


def read_clock_tree(project_root: str) -> dict | None:
    return _read_json(Path(project_root) / ".coresmith" / "clock_tree.json")


def read_register_spec(project_root: str) -> dict | None:
    return _read_json(Path(project_root) / ".coresmith" / "register_spec.json")


def read_ers(project_root: str) -> dict | None:
    return _read_json(Path(project_root) / ".coresmith" / "ers_spec.json")


def read_block_specs(project_root: str) -> list | None:
    return _read_json(Path(project_root) / ".coresmith" / "block_specs.json")


def list_documents(project_root: str) -> dict[str, bool]:
    """Return a dict mapping document names to presence booleans."""
    root = Path(project_root)
    coresmith = root / ".coresmith"
    arch = root / "arch"

    md_only = {"sad_spec", "frd_spec"}
    json_docs = [
        "prd_spec", "block_diagram", "memory_map",
        "clock_tree", "register_spec", "ers_spec",
    ]

    result: dict[str, bool] = {}
    for doc in json_docs:
        result[doc] = (coresmith / f"{doc}.json").exists()
    for doc in md_only:
        result[doc] = (arch / f"{doc}.md").exists()

    return result
