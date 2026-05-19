# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Shared utilities for the socmate orchestrator.

Contains:
  - atomic_write: crash-safe file writes via tmp+rename
  - parse_llm_json: robust JSON extraction from LLM responses
  - smart_truncate: context-preserving text truncation
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Fix #15 -- Atomic file writes
# ---------------------------------------------------------------------------

def atomic_write(path: Path, content: str) -> None:
    """Write *content* to *path* atomically (POSIX rename semantics).

    Writes to a sibling ``.tmp`` file first, then calls ``os.replace``
    which is atomic on POSIX.  This prevents half-written state files
    when the process is killed mid-write.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(str(tmp), str(path))


# ---------------------------------------------------------------------------
# Fix #8 -- Robust LLM JSON parsing
# ---------------------------------------------------------------------------

def parse_llm_json(
    content: str,
    defaults: dict[str, Any],
    context: str = "",
) -> tuple[dict[str, Any], bool]:
    """Extract structured JSON from an LLM response.

    Tries multiple strategies in order:
      1. JSON code-block (```json ... ```)
      2. Raw JSON parse of the entire response
      3. Lenient cleanup (strip markdown fences, trailing commas,
         single→double quotes) then re-parse

    Returns:
        ``(result, parse_ok)`` where *parse_ok* is ``True`` if JSON
        was successfully parsed and ``False`` if defaults were used.
    """
    # Strategy 1: JSON code block
    json_match = re.search(r"```json\s*\n(.*?)```", content, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group(1))
            if isinstance(result, dict):
                merged = {**defaults, **result}
                return merged, True
        except (json.JSONDecodeError, TypeError):
            pass

    # Strategy 2: raw JSON
    try:
        result = json.loads(content)
        if isinstance(result, dict):
            merged = {**defaults, **result}
            return merged, True
    except (json.JSONDecodeError, TypeError):
        pass

    # Strategy 3: lenient cleanup
    cleaned = _lenient_json_cleanup(content)
    if cleaned:
        try:
            result = json.loads(cleaned)
            if isinstance(result, dict):
                merged = {**defaults, **result}
                return merged, True
        except (json.JSONDecodeError, TypeError):
            pass

    # All strategies failed
    fallback = dict(defaults)
    if content:
        fallback.setdefault("reasoning", content[:2000])
    fallback["_parse_failed"] = True
    return fallback, False


_CONFIRMATION_RE = re.compile(
    r"^([\w\s]+written to|I'?ve written|The .+ has been written|Done|File saved)",
    re.IGNORECASE,
)

_MIN_REAL_CONTENT_LEN = 200


def read_back_text(path: Path, llm_response: str) -> str:
    """Read a file the inner Claude wrote, falling back to llm_response.

    The inner Claude (ClaudeLLM) has Write-tool access and is told to
    write architecture docs to disk.  Its *response* is often just a
    confirmation like ``"SAD written to: arch/sad_spec.md"``.  This
    helper reads the real content from disk.
    """
    try:
        if path.exists():
            on_disk = path.read_text(encoding="utf-8").strip()
            if len(on_disk) > _MIN_REAL_CONTENT_LEN:
                return on_disk
    except OSError:
        pass
    return llm_response


def read_back_json(
    path: Path,
    llm_response: str,
    defaults: dict[str, Any],
    context: str = "",
) -> tuple[dict[str, Any], bool]:
    """Read a JSON file the inner Claude wrote, falling back to parse_llm_json.

    Same motivation as ``read_back_text`` but for structured JSON docs.
    Tries the on-disk file first, then falls back to parsing the LLM
    response text.
    """
    try:
        if path.exists():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                data = json.loads(text)
                if isinstance(data, dict):
                    merged = {**defaults, **data}
                    return merged, True
    except (json.JSONDecodeError, OSError):
        pass

    # Codex runs may be isolated in a per-call worktree under the project
    # root. In that mode a file-writing specialist can correctly write
    # ".socmate/<name>.json" inside "codex-call-*", return only a path
    # confirmation, and leave the canonical project-root file absent or stale.
    # Recover by looking for the newest matching artifact in those isolated
    # worktrees before falling back to parsing the response text.
    try:
        project_root = path.parent.parent
        rel = path.relative_to(project_root)
        candidates = sorted(
            project_root.glob(f"codex-call-*/{rel.as_posix()}"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for candidate in candidates:
            text = candidate.read_text(encoding="utf-8").strip()
            if not text:
                continue
            data = json.loads(text)
            if isinstance(data, dict):
                atomic_write(path, json.dumps(data, indent=2) + "\n")
                merged = {**defaults, **data}
                return merged, True
    except (ValueError, json.JSONDecodeError, OSError):
        pass
    return parse_llm_json(llm_response, defaults, context=context)


def _lenient_json_cleanup(content: str) -> str | None:
    """Try to recover JSON from slightly malformed LLM output."""
    # Strip leading/trailing markdown fences
    text = re.sub(r"^```\w*\s*\n?", "", content.strip())
    text = re.sub(r"\n?```\s*$", "", text)

    # Find the outermost { ... }
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    text = text[start : end + 1]

    # Remove trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)

    return text


# ---------------------------------------------------------------------------
# Fix #14 -- Smart truncation
# ---------------------------------------------------------------------------

def smart_truncate(
    text: str,
    max_chars: int,
    strategy: str = "head_tail",
) -> str:
    """Truncate *text* while preserving the most diagnostic information.

    Strategies:
      - ``"tail"``:  keep the last *max_chars* (best for error logs).
      - ``"head"``:  keep the first *max_chars*.
      - ``"head_tail"`` (default): keep first 40 % and last 60 %,
        with an omission notice in the middle.
    """
    if len(text) <= max_chars:
        return text

    omitted = len(text) - max_chars
    sep = f"\n\n...({omitted} chars omitted)...\n\n"
    sep_len = len(sep)

    if strategy == "tail":
        return f"...({omitted} chars omitted)...\n" + text[-max_chars:]

    if strategy == "head":
        return text[:max_chars] + f"\n...({omitted} chars omitted)..."

    # head_tail: keep first 40 % and last 60 %
    usable = max_chars - sep_len
    if usable < 40:
        # Degenerate case -- fall back to tail
        return text[-max_chars:]
    head = usable * 2 // 5
    tail = usable - head
    return text[:head] + sep + text[-tail:]
