# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""FRD evaluation on the SystemC SoC model (B2).

The uArch phase does not end with "the model compiles and clocks": every
FRD requirement is evaluated against the assembled model before any RTL is
lowered. This module is the deterministic half -- the requirement list the
harness must answer, the build/run of the agent-authored harness
(``model/frd_eval/*.cpp``, own ``sc_main``), the verdict protocol and the
summary the gate reads. The harness prints one line per requirement::

    FRD_EVAL {"id": "PERF-001", "status": "pass", "evidence": "..."}

with ``status`` in :data:`STATUSES`, and ``FRD_EVAL_DONE`` at the end.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

STATUSES = ("pass", "fail", "not_testable", "skipped")

# ``1. **ID**: PERF-001`` blocks as the FRD prompt formats them; the section
# heading above a block is its category.
_ID_RE = re.compile(r"\*\*ID\*\*:\s*`?([A-Z][A-Z0-9]*-[A-Z0-9-]*\d)`?", re.I)
_FIELD_RE = re.compile(r"\*\*(Requirement|Acceptance criteria|Priority|Model check)\*\*:\s*(.*)", re.I)
_LINE_RE = re.compile(r"^FRD_EVAL\s+(\{.*\})\s*$", re.M)


def extract_requirements(frd_text: str) -> list[dict]:
    """Every identified requirement of the FRD: ``{id, section, requirement,
    acceptance, priority, model_check}``. Tolerant of formatting drift: an id
    with no fields still yields a record (text = the block's first lines)."""
    reqs: list[dict] = []
    section = ""
    cur: dict | None = None
    for raw in frd_text.splitlines():
        line = raw.rstrip()
        h = re.match(r"^(#{2,3})\s+(.*)", line)
        if h:
            section = h.group(2).strip()
            cur = None
            continue
        m = _ID_RE.search(line)
        if m and "**ID**" in line:
            cur = {"id": m.group(1).upper(), "section": section, "requirement": "",
                   "acceptance": "", "priority": "", "model_check": ""}
            reqs.append(cur)
            continue
        if cur is None:
            continue
        f = _FIELD_RE.search(line)
        if f:
            key = f.group(1).lower()
            val = f.group(2).strip()
            if key == "requirement":
                cur["requirement"] = val
            elif key == "acceptance criteria":
                cur["acceptance"] = val
            elif key == "priority":
                cur["priority"] = val.strip("*` ").lower()
            else:
                cur["model_check"] = val
        elif line.strip() and cur["requirement"] and not cur["acceptance"] and line.startswith("     "):
            cur["requirement"] += " " + line.strip()
    seen: set[str] = set()
    uniq = []
    for r in reqs:
        if r["id"] not in seen:
            seen.add(r["id"])
            uniq.append(r)
    return uniq


def must_have(req: dict) -> bool:
    p = (req.get("priority") or "").lower()
    return "must" in p or p in ("hard", "p0", "required")


def write_requirements(model_dir, reqs: list[dict], *, frd_path: str = "arch/frd_spec.md") -> Path:
    d = Path(model_dir) / "frd_eval"
    d.mkdir(parents=True, exist_ok=True)
    out = d / "requirements.json"
    out.write_text(json.dumps({"frd": frd_path, "count": len(reqs), "requirements": reqs}, indent=2))
    return out


def harness_sources(model_dir) -> list[Path]:
    return sorted((Path(model_dir) / "frd_eval").glob("*.cpp"))


def build_harness(model_dir, *, timeout_s: int = 900) -> dict:
    md = Path(model_dir)
    if not harness_sources(md):
        return {"ok": False, "log": "no frd_eval/*.cpp harness sources"}
    env = dict(os.environ)
    from .toolchain import systemc_home
    if systemc_home():
        env["SYSTEMC_HOME"] = systemc_home()
    try:
        p = subprocess.run(["make", "-s", "frd_eval/frd_eval"], cwd=md, capture_output=True, text=True,
                           timeout=timeout_s, env=env)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "log": str(exc)}
    log = p.stdout + p.stderr
    (md / "frd_eval" / "build.log").write_text(log)
    return {"ok": p.returncode == 0 and (md / "frd_eval" / "frd_eval").exists(), "log": log[-8000:]}


def parse_results(text: str) -> list[dict]:
    out = []
    for m in _LINE_RE.finditer(text or ""):
        try:
            rec = json.loads(m.group(1))
        except ValueError:
            continue
        if not isinstance(rec, dict) or not rec.get("id"):
            continue
        st = str(rec.get("status") or "").lower()
        rec["id"] = str(rec["id"]).upper()
        rec["status"] = st if st in STATUSES else "fail"
        rec.setdefault("evidence", "")
        out.append(rec)
    return out


def run_harness(model_dir, *, timeout_s: int = 1800, args: list[str] | None = None) -> dict:
    md = Path(model_dir)
    exe = md / "frd_eval" / "frd_eval"
    if not exe.exists():
        return {"ok": False, "done": False, "results": [], "log": "frd_eval not built"}
    try:
        p = subprocess.run([str(exe), *(args or [])], cwd=md, capture_output=True, text=True, timeout=timeout_s)
        out = p.stdout + p.stderr
        rc = p.returncode
    except subprocess.TimeoutExpired as exc:
        out = ((exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")) \
            + f"\n[frd_eval] TIMEOUT after {timeout_s}s"
        rc = -1
    except OSError as exc:
        return {"ok": False, "done": False, "results": [], "log": str(exc)}
    (md / "frd_eval" / "run.log").write_text(out)
    results = parse_results(out)
    done = "FRD_EVAL_DONE" in out
    return {"ok": rc == 0 and done, "done": done, "rc": rc, "results": results, "log": out[-8000:]}


def summarize(reqs: list[dict], results: list[dict]) -> dict:
    """Per-status counts plus the ids that decide the gate: ``failed`` (any
    requirement the harness reports as failing) and ``unanswered_must`` (a
    must-have the harness gave no verdict for -- neither tested nor declared
    not testable with a reason)."""
    by_id = {r["id"]: r for r in results}
    counts = {s: 0 for s in STATUSES}
    for r in results:
        counts[r["status"]] += 1
    failed = sorted(r["id"] for r in results if r["status"] == "fail")
    unanswered_must = sorted(q["id"] for q in reqs if must_have(q) and q["id"] not in by_id)
    unreasoned = sorted(r["id"] for r in results if r["status"] == "not_testable" and not str(r.get("evidence") or "").strip())
    unknown = sorted(set(by_id) - {q["id"] for q in reqs})
    return {"counts": counts, "requirements": len(reqs), "answered": len(by_id), "failed": failed,
            "unanswered_must": unanswered_must, "not_testable_without_reason": unreasoned,
            "unknown_ids": unknown,
            "gate_ok": not failed and not unanswered_must and not unreasoned}


def write_report(project_root, model_dir, reqs: list[dict], run: dict, summary: dict) -> Path:
    """``.coresmith/frd_eval.json`` (machine) + ``model/frd_eval/REPORT.md`` (human)."""
    pr = Path(project_root)
    rec = {"summary": summary, "results": run.get("results", []), "done": run.get("done"), "rc": run.get("rc"),
           "requirements": [q["id"] for q in reqs]}
    (pr / ".coresmith").mkdir(exist_ok=True)
    out = pr / ".coresmith" / "frd_eval.json"
    out.write_text(json.dumps(rec, indent=2))
    by_id = {r["id"]: r for r in run.get("results", [])}
    L = ["# FRD evaluation on the SystemC SoC model", "",
         f"gate_ok: **{summary['gate_ok']}** -- {summary['counts']}", "",
         "| id | priority | status | evidence |", "|---|---|---|---|"]
    for q in reqs:
        r = by_id.get(q["id"], {})
        L.append(f"| {q['id']} | {q.get('priority','')} | {r.get('status','**unanswered**')} | "
                 f"{str(r.get('evidence',''))[:200].replace('|', '/')} |")
    for i in summary.get("unknown_ids") or []:
        L.append(f"| {i} | - | {by_id[i]['status']} | (not an FRD id) {str(by_id[i].get('evidence',''))[:120]} |")
    (Path(model_dir) / "frd_eval" / "REPORT.md").write_text("\n".join(L) + "\n")
    return out
