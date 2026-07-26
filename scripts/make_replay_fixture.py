#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Distill a run's ``llm_calls.jsonl`` into a compact replay fixture.

Produces the layout ``orchestrator.testing.replay_provider.ReplayBackend`` reads::

    <out>/
      meta.json      # name, source, original_root, keep_prompts, engine_note, ...
      calls.jsonl    # one record per selected call (digest + response_ref [+ writes])
      files/<sha256> # content-addressed response bodies and (optional) writes
      pre/           # (not written here; created by snapshot tools if needed)

Design notes / footguns handled:
  * Pre-plan-2 corpora have NO ``run_name``/``call_index`` fields -- ``--infer-sites``
    recovers ``run_name`` by joining each llm_calls record to the ``pipeline_events``
    ``llm_end`` (``output_chars``==``response_len``) or ``llm_start``
    (``system_chars``==``system_prompt_len``) event nearest in time.
  * Raw prompts are DROPPED by default (only the normalized-prompt SHA-256 digest
    is kept -- enough for exact+digest replay matching). ``--keep-prompts`` retains
    the normalized prompt so fuzzy jaccard matching also works.
  * ``--auto-writes`` attributes on-disk files to a call when their mtime falls
    within that call's ``[llm_start, llm_end]`` window (+ slack). Telemetry/state
    files and oversized blobs are skipped.

Usage:
    python3 scripts/make_replay_fixture.py <run_dir_or_.coresmith> \\
        --out orchestrator/tests/fixtures/replay/<name> --name <name> \\
        --indices 60 --infer-sites [--auto-writes] [--keep-prompts]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Import prompt_norm without importing the whole orchestrator package.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from orchestrator.testing.prompt_norm import normalize_prompt, prompt_digest  # noqa: E402

# State/telemetry files that are outputs of the *harness*, never an LLM write.
_TELEMETRY_NAMES = {
    "llm_calls.jsonl", "pipeline_events.jsonl", "traces.db", "daemon.json",
    "daemon.log", "pipeline_results.json", "scoreboard.db",
}
_SKIP_DIR_PARTS = {".git", "__pycache__", "sim_build", "node_modules"}


# ---------------------------------------------------------------------------
# Loading + join
# ---------------------------------------------------------------------------
def _load_jsonl(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def resolve_paths(source: str) -> tuple[Path, Path, Path]:
    """Return (run_dir, llm_calls_path, events_path) from a run dir or .coresmith dir."""
    p = Path(source).resolve()
    if p.name == ".coresmith" or (p / "llm_calls.jsonl").exists():
        cor = p if p.name == ".coresmith" else p
        if (p / "llm_calls.jsonl").exists() and p.name != ".coresmith":
            cor = p
        else:
            cor = p if (p / "llm_calls.jsonl").exists() else (p / ".coresmith")
    else:
        cor = p / ".coresmith"
    llm = cor / "llm_calls.jsonl"
    ev = cor / "pipeline_events.jsonl"
    run_dir = cor.parent
    return run_dir, llm, ev


def infer_run_name(call: dict, starts: list[dict], ends: list[dict],
                   ts_window: float = 8.0) -> str | None:
    """Recover a call's run_name by joining to llm_end/llm_start events."""
    rl = call.get("response_len")
    ts = call.get("ts", 0.0)
    cands = [e for e in ends if e.get("output_chars") == rl and e.get("run_name")]
    if cands:
        best = min(cands, key=lambda e: abs(e.get("ts", 0.0) - ts))
        if abs(best.get("ts", 0.0) - ts) <= ts_window:
            return best.get("run_name")
    # Fallback: match a start by system prompt length (+ generous window: start
    # precedes the record by the call duration).
    syn = call.get("system_prompt_len")
    dur = call.get("duration_s", 0.0) or 0.0
    cands = [s for s in starts if s.get("system_chars") == syn and s.get("run_name")]
    if cands:
        best = min(cands, key=lambda s: abs(ts - dur - s.get("ts", 0.0)))
        if abs(ts - dur - best.get("ts", 0.0)) <= max(ts_window, dur + 5.0):
            return best.get("run_name")
    return None


def detect_original_root(calls: list[dict]) -> str:
    """Best-effort: recover the recording run's absolute project root."""
    import re
    from collections import Counter

    pr = re.compile(r"Project root:\s*(\S+)")
    home = re.compile(r"/home/[^/\s\"']+/[A-Za-z0-9_.-]+")
    counter: Counter = Counter()
    for c in calls[:40]:
        blob = (c.get("system_prompt", "") or "") + "\n" + (c.get("user_prompt", "") or "")
        m = pr.search(blob)
        if m:
            counter[m.group(1).rstrip("/")] += 5
        for hm in home.findall(blob):
            counter[hm.rstrip("/")] += 1
    return counter.most_common(1)[0][0] if counter else ""


# ---------------------------------------------------------------------------
# Auto-writes
# ---------------------------------------------------------------------------
def event_window(call: dict, starts: list[dict], ends: list[dict],
                 run_name: str | None) -> tuple[float, float] | None:
    """[start_ts, end_ts] window for the call, via matched run_name events."""
    end_ts = call.get("ts")
    start_ts = None
    if run_name:
        st = [s for s in starts if s.get("run_name") == run_name]
        en = [e for e in ends if e.get("run_name") == run_name]
        if en:
            best = min(en, key=lambda e: abs(e.get("ts", 0.0) - (end_ts or 0.0)))
            end_ts = best.get("ts", end_ts)
        if st and end_ts is not None:
            before = [s for s in st if s.get("ts", 0.0) <= end_ts]
            if before:
                start_ts = max(before, key=lambda s: s.get("ts", 0.0)).get("ts")
    if start_ts is None and end_ts is not None:
        start_ts = end_ts - (call.get("duration_s", 0.0) or 0.0)
    if start_ts is None or end_ts is None:
        return None
    return (start_ts, end_ts)


def find_writes(run_dir: Path, window: tuple[float, float], *, slack: float,
                size_cap: int) -> list[Path]:
    """Files whose mtime falls in [start-slack, end+slack]."""
    lo, hi = window[0] - slack, window[1] + slack
    hits: list[Path] = []
    for p in run_dir.rglob("*"):
        if not p.is_file():
            continue
        if any(part in _SKIP_DIR_PARTS for part in p.parts):
            continue
        if p.name in _TELEMETRY_NAMES:
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        if st.st_size > size_cap:
            continue
        if lo <= st.st_mtime <= hi:
            hits.append(p)
    return hits


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
def parse_indices(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def select_calls(calls: list[dict], *, indices: list[int] | None,
                 run_name_glob: str | None, max_calls: int | None,
                 run_names: list[str | None]) -> list[int]:
    import fnmatch

    chosen: list[int] = []
    for i, c in enumerate(calls):
        if indices is not None and i not in indices:
            continue
        if run_name_glob is not None:
            rn = run_names[i] or ""
            if not fnmatch.fnmatch(rn, run_name_glob):
                continue
        chosen.append(i)
    if max_calls is not None:
        chosen = chosen[:max_calls]
    return chosen


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_fixture(*, source: str, out: str, name: str,
                  indices: list[int] | None = None,
                  run_name_glob: str | None = None,
                  max_calls: int | None = None,
                  infer_sites: bool = False,
                  auto_writes: bool = False,
                  keep_prompts: bool = False,
                  write_slack: float = 5.0,
                  write_size_cap: int = 200_000,
                  original_root: str | None = None,
                  engine_note: str = "") -> dict:
    run_dir, llm_path, ev_path = resolve_paths(source)
    calls = _load_jsonl(llm_path)
    if not calls:
        raise SystemExit(f"no llm_calls records at {llm_path}")
    events = _load_jsonl(ev_path)
    starts = [e for e in events if e.get("event") == "llm_start"]
    ends = [e for e in events if e.get("event") == "llm_end"]

    orig = original_root if original_root is not None else detect_original_root(calls)

    # run_name per call.
    run_names: list[str | None] = []
    for c in calls:
        rn = c.get("run_name") or None
        if rn is None and infer_sites:
            rn = infer_run_name(c, starts, ends)
        run_names.append(rn)

    chosen = select_calls(
        calls, indices=indices, run_name_glob=run_name_glob,
        max_calls=max_calls, run_names=run_names,
    )
    if not chosen:
        raise SystemExit("no calls selected (check --indices/--run-name-glob)")

    out_dir = Path(out)
    (out_dir / "files").mkdir(parents=True, exist_ok=True)

    def _store(data: bytes) -> str:
        h = hashlib.sha256(data).hexdigest()
        (out_dir / "files" / h).write_bytes(data)
        return f"files/{h}"

    site_counter: dict[str, int] = {}
    records: list[dict] = []
    for i in chosen:
        c = calls[i]
        rn = run_names[i] or f"call_{i}"
        si = site_counter.get(rn, 0)
        site_counter[rn] = si + 1
        resp = c.get("response", "") or ""
        prompt = c.get("user_prompt", "") or ""
        system = c.get("system_prompt", "") or ""
        roots = [orig] if orig else []
        rec = {
            "run_name": rn,
            "site_index": si,
            "provider": c.get("provider", ""),
            "model": c.get("model", ""),
            "prompt_digest": prompt_digest(prompt, roots),
            "system_digest": prompt_digest(system, roots),
            "response_ref": _store(resp.encode("utf-8")),
            "response_len": len(resp),
            "timed_out": bool(c.get("timed_out")),
            "error": c.get("error", "") or "",
            "source_index": i,
            "writes": [],
        }
        if keep_prompts:
            rec["prompt_norm"] = normalize_prompt(prompt, roots)
            rec["system_norm"] = normalize_prompt(system, roots)

        if auto_writes:
            window = event_window(c, starts, ends, run_names[i])
            if window is not None:
                for wp in find_writes(run_dir, window, slack=write_slack,
                                      size_cap=write_size_cap):
                    try:
                        rel = wp.relative_to(run_dir).as_posix()
                    except ValueError:
                        continue
                    ref = _store(wp.read_bytes())
                    rec["writes"].append({
                        "relpath": rel, "ref": ref, "size": wp.stat().st_size,
                    })
        records.append(rec)

    with open(out_dir / "calls.jsonl", "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    meta = {
        "name": name,
        "source": str(llm_path),
        "original_root": orig,
        "keep_prompts": keep_prompts,
        "engine_note": engine_note,
        "n_calls": len(records),
        "run_names": sorted({r["run_name"] for r in records}),
        "total_writes": sum(len(r["writes"]) for r in records),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="run dir (parent of .coresmith) or a .coresmith dir")
    ap.add_argument("--out", required=True, help="fixture output directory")
    ap.add_argument("--name", default="", help="fixture name (defaults to --out basename)")
    ap.add_argument("--indices", default=None,
                    help="llm_calls line indices to keep, e.g. '60' or '11,12' or '5-9'")
    ap.add_argument("--run-name-glob", default=None,
                    help="keep calls whose (inferred) run_name matches this fnmatch glob")
    ap.add_argument("--max-calls", type=int, default=None)
    ap.add_argument("--infer-sites", action="store_true",
                    help="recover run_name from pipeline_events (pre-plan-2 logs)")
    ap.add_argument("--auto-writes", action="store_true",
                    help="attribute on-disk files to a call via mtime windows")
    ap.add_argument("--keep-prompts", action="store_true",
                    help="retain the normalized prompt (enables fuzzy matching)")
    ap.add_argument("--write-slack", type=float, default=5.0)
    ap.add_argument("--write-size-cap", type=int, default=200_000)
    ap.add_argument("--original-root", default=None,
                    help="override the auto-detected recording project root")
    ap.add_argument("--engine-note", default="",
                    help="what plumbing behavior this fixture pins")
    args = ap.parse_args(argv)

    name = args.name or Path(args.out).name
    meta = build_fixture(
        source=args.source, out=args.out, name=name,
        indices=parse_indices(args.indices) if args.indices else None,
        run_name_glob=args.run_name_glob, max_calls=args.max_calls,
        infer_sites=args.infer_sites, auto_writes=args.auto_writes,
        keep_prompts=args.keep_prompts, write_slack=args.write_slack,
        write_size_cap=args.write_size_cap, original_root=args.original_root,
        engine_note=args.engine_note,
    )
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
