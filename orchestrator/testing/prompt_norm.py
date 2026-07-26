# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Prompt normalization for record/replay (Package C, C1/C4.1).

The ``llm_calls.jsonl`` corpus records the FULL prompt+response of every LLM
call, but a prompt is full of run-specific volatile substrings -- the absolute
project root, timestamps, session ids -- that differ between the recording run
and a later replay run even though the *call* is logically the same. Replay
matching therefore normalizes prompts to a canonical form before digesting or
fuzzy-comparing them:

  - project roots        -> ``<RUN>``   (so ``/home/ubuntu/coresmith-runs/x`` and
                                          ``/tmp/pytest-.../t0`` collapse)
  - ISO / epoch times    -> ``<T>``
  - session / hex ids    -> ``<ID>``   (uuids, sha digests, codex ``tok_...``)

``prompt_digest`` is the primary fuzzy-independent key; ``token_jaccard`` is the
last-resort similarity used when neither the exact ``(run_name, site_index)`` key
nor the digest matches (see ``replay_provider``). All three are pure functions.
"""

from __future__ import annotations

import hashlib
import re

# ISO-8601-ish timestamps: 2026-05-13T07:11:34(.38)(Z|+00:00)
_ISO_TS = re.compile(
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"
)
# Epoch seconds (10 digits, 2001-2033), optional fractional part: 1778656294.38
_EPOCH_TS = re.compile(r"\b1[0-9]{9}(?:\.\d+)?\b")
# Bare date: 2026-05-13
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
# UUID
_UUID = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
# codex / openai style opaque tokens: tok_..., sess_..., call_...
_OPAQUE_TOKEN = re.compile(r"\b(?:tok|sess|session|call|resp|msg|thread)_[A-Za-z0-9]{6,}\b")
# Long bare hex strings (>=16 chars) -- sha digests, session ids.
_HEX = re.compile(r"\b[0-9a-fA-F]{16,}\b")
# Tokenizer for jaccard: identifier-ish runs plus the placeholder tokens.
_WORD = re.compile(r"<[A-Z]+>|[A-Za-z_][A-Za-z0-9_]*|\d+")

RUN_TOKEN = "<RUN>"
TIME_TOKEN = "<T>"
ID_TOKEN = "<ID>"


def normalize_prompt(text: str, roots=()) -> str:
    """Canonicalize a prompt for stable digest/similarity matching.

    ``roots`` is an iterable of absolute project-root strings to blank to
    ``<RUN>`` (longest first so nested roots don't partial-match). Callers pass
    both the fixture's recorded ``original_root`` and the live project root, so
    the recording and the replay prompt normalize to the same canonical text.
    """
    if not text:
        return ""
    out = text
    # 1. Explicit roots (longest first). Trailing slash tolerated.
    seen = {str(r).rstrip("/") for r in (roots or ()) if r}
    for r in sorted(seen, key=len, reverse=True):
        if r:
            out = out.replace(r, RUN_TOKEN)
    # 2. Generic run-dir heuristic (survives an unknown root): any path segment
    #    up to and including a ``coresmith-runs/<name>`` component.
    out = re.sub(r"/[^\s\"'`]*coresmith-runs/[^\s\"'`/]+", RUN_TOKEN, out)
    # 3. Volatile scalars.
    out = _ISO_TS.sub(TIME_TOKEN, out)
    out = _EPOCH_TS.sub(TIME_TOKEN, out)
    out = _DATE.sub(TIME_TOKEN, out)
    out = _UUID.sub(ID_TOKEN, out)
    out = _OPAQUE_TOKEN.sub(ID_TOKEN, out)
    out = _HEX.sub(ID_TOKEN, out)
    return out


def prompt_digest(text: str, roots=()) -> str:
    """SHA-256 (hex) of the normalized prompt -- the primary replay key."""
    norm = normalize_prompt(text, roots)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def _tokens(text: str) -> set[str]:
    return set(m.lower() if not m.startswith("<") else m for m in _WORD.findall(text))


def token_jaccard(a: str, b: str, roots=()) -> float:
    """Jaccard similarity of the two normalized prompts' token sets (0..1)."""
    ta = _tokens(normalize_prompt(a, roots))
    tb = _tokens(normalize_prompt(b, roots))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0
