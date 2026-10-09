"""Every file the exporter reads, with its identity and what was made of it.

A source gets a short id (``S12``); exported records reference it as
``{"s": "S12", "l": 345}`` (file line) or ``{"s": "S3", "k": "actions#17"}``
(database row). The UI shows the private snapshot path as text, never as a
download link.
"""
from __future__ import annotations

import collections
import datetime as _dt
import hashlib
import json
from pathlib import Path


def iso(ts) -> str | None:
    if ts is None:
        return None
    try:
        return _dt.datetime.fromtimestamp(float(ts), _dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def parse_iso(text) -> float | None:
    if not text or not isinstance(text, str):
        return None
    try:
        value = _dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
        if value.tzinfo is None:
            value = value.replace(tzinfo=_dt.timezone.utc)
        return value.timestamp()
    except ValueError:
        return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path):
    """Yield ``(line_number, record_or_None, raw_text)`` for every non-blank
    line; an unparseable line (a torn final line of a live log) yields None."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for n, line in enumerate(fh, 1):
            raw = line.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except ValueError:
                yield n, None, raw
                continue
            yield n, rec if isinstance(rec, dict) else {"__value__": rec}, raw


class Sources:
    def __init__(self, root: Path, manifest: dict | None = None):
        self.root = Path(root)
        self.items: dict[str, dict] = {}
        self._by_path: dict[str, str] = {}
        self._manifest = {}
        for f in (manifest or {}).get("files") or []:
            if isinstance(f, dict) and f.get("path"):
                self._manifest[str(f["path"])] = f

    def rel(self, path: Path) -> str:
        try:
            return str(Path(path).resolve().relative_to(self.root.resolve()))
        except ValueError:
            return str(path)

    def add(self, path: Path, role: str, arm: str | None = None) -> str:
        """Register ``path`` (idempotent) and return its source id."""
        rel = self.rel(path)
        if rel in self._by_path:
            return self._by_path[rel]
        sid = f"S{len(self.items) + 1}"
        p = Path(path)
        st = p.stat()
        man = self._manifest.get(rel) or {}
        sha = man.get("sha256")
        computed = sha256_file(p)
        rec = {"id": sid, "path": rel, "arm": arm, "role": role, "bytes": st.st_size, "sha256": computed,
               "manifest": {k: man.get(k) for k in ("mode", "source", "bytes", "sha256", "changed_during_copy",
                                                    "partial_final_line", "source_bytes_before", "source_bytes_after",
                                                    "elapsed_seconds") if k in man} or None,
               "manifest_sha_matches": (sha == computed) if sha else None,
               "records": 0, "kept": 0, "duplicates": 0, "unparseable": 0, "skipped": collections.Counter(),
               "notes": []}
        self.items[sid] = rec
        self._by_path[rel] = sid
        return sid

    def stat(self, sid: str, key: str, n: int = 1) -> None:
        self.items[sid][key] = self.items[sid].get(key, 0) + n

    def skip(self, sid: str, reason: str, n: int = 1) -> None:
        self.items[sid]["skipped"][reason] += n

    def note(self, sid: str, text: str) -> None:
        if text not in self.items[sid]["notes"]:
            self.items[sid]["notes"].append(text)

    def to_json(self) -> list[dict]:
        out = []
        for rec in self.items.values():
            d = dict(rec)
            d["skipped"] = dict(rec["skipped"])
            out.append(d)
        return out

    def unregistered_manifest_files(self) -> list[dict]:
        """Manifest entries the exporter did not read (shown on the coverage screen)."""
        seen = set(self._by_path)
        return [{"path": p, "mode": f.get("mode"), "bytes": f.get("bytes")}
                for p, f in sorted(self._manifest.items()) if p not in seen]


def ref(sid: str, line: int | None = None, key: str | None = None) -> dict:
    r = {"s": sid}
    if line is not None:
        r["l"] = line
    if key is not None:
        r["k"] = key
    return r
