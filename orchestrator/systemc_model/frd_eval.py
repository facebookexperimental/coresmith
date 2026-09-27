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


def write_report(project_root, model_dir, reqs: list[dict], run: dict, summary: dict, *, name: str = "frd_eval") -> Path:
    """``.coresmith/<name>.json`` (machine) + ``<model_dir>/frd_eval/REPORT.md`` (human)."""
    pr = Path(project_root)
    rec = {"summary": summary, "results": run.get("results", []), "done": run.get("done"), "rc": run.get("rc"),
           "requirements": [q["id"] for q in reqs], "model_dir": str(model_dir)}
    (pr / ".coresmith").mkdir(exist_ok=True)
    out = pr / ".coresmith" / f"{name}.json"
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


def _log(msg: str) -> None:
    try:
        from orchestrator.langgraph.pipeline_helpers import GREEN, RED, YELLOW, log   # noqa: F401
    except Exception:  # noqa: BLE001
        print(msg)
        return
    colour = RED if ("gate_ok=False" in msg or "failing:" in msg or "author failed" in msg) \
        else (YELLOW if "attempt" in msg else GREEN)
    log(msg, colour)


async def evaluate(project_root, model_dir, blocks: list[str], *, arch: bool = False, agent=None,
                   timeout_s: int | None = None, repairs: int | None = None, db=None, record_sha: str = "") -> dict:
    """The FRD evaluated on a model (the SoC LT model, or the executable SAD
    when ``arch``): requirement list -> agent-authored harness
    (``<model_dir>/frd_eval/*.cpp``) -> build -> run -> verdicts. Repairs are
    bounded and only for harness defects (compile error, crash, hang, missing
    DONE); a ``fail`` verdict is the finding. Never raises. When ``db`` is
    given every verdict is recorded as a ``model_eval`` check (hash-bound)."""
    md = Path(model_dir)
    rec: dict = {"enabled": True, "gate_ok": None, "arch": arch}
    frd = Path(project_root) / "arch" / "frd_spec.md"
    if not frd.exists():
        rec["skipped"] = "no arch/frd_spec.md"
        return rec
    reqs = extract_requirements(frd.read_text(encoding="utf-8", errors="replace"))
    write_requirements(md, reqs)
    rec["requirements"] = len(reqs)
    if not reqs:
        rec["skipped"] = "the FRD has no identified requirements (**ID**: XXX-NNN blocks)"
        return rec
    timeout_s = timeout_s or int(os.environ.get("CORESMITH_FRD_EVAL_TIMEOUT_S", "1800") or 1800)
    repairs = max(0, int(os.environ.get("CORESMITH_FRD_EVAL_REPAIRS", "3") or 3) if repairs is None else repairs)
    if agent is None:
        from orchestrator.langchain.agents.frd_eval_generator import FRDEvalGenerator
        agent = FRDEvalGenerator()
    compiler_log, run_log, summary, run = "", "", None, None
    for attempt in range(1, repairs + 2):
        if not (attempt == 1 and harness_sources(md)):   # reuse an existing harness first
            try:
                await agent.generate(project_root=str(project_root), blocks=blocks, attempt=attempt,
                                     compiler_log=compiler_log, run_log=run_log, summary=summary, arch=arch)
            except Exception as exc:  # noqa: BLE001
                rec["error"] = f"harness author failed: {str(exc)[:300]}"
                _log(f"  [FRD-EVAL] {rec['error']}")
                break
        b = build_harness(md)
        rec["built"] = bool(b["ok"])
        if not b["ok"]:
            compiler_log, run_log = b["log"], ""
            _log(f"  [FRD-EVAL] harness build failed (attempt {attempt})")
            continue
        run = run_harness(md, timeout_s=timeout_s)
        summary = summarize(reqs, run["results"])
        rec.update({"done": run["done"], "rc": run.get("rc"), "summary": summary})
        if run["done"]:
            break
        compiler_log, run_log = "", run["log"]
        _log(f"  [FRD-EVAL] harness did not run to completion (attempt {attempt})")
    if run is not None:
        rec["report"] = str(write_report(project_root, md, reqs, run, summary, name="frd_eval_arch" if arch else "frd_eval"))
    rec["gate_ok"] = bool(run and run["done"] and summary and summary["gate_ok"])
    if summary:
        c = summary["counts"]
        _log(f"  [FRD-EVAL] {len(reqs)} requirement(s): pass={c['pass']} fail={c['fail']} "
             f"not_testable={c['not_testable']} skipped={c['skipped']} unanswered_must={len(summary['unanswered_must'])}"
             f" -> gate_ok={rec['gate_ok']}")
        if summary["failed"]:
            _log(f"  [FRD-EVAL] failing: {', '.join(summary['failed'][:12])}")
    if db is not None and run is not None and run.get("done"):
        for r in run["results"]:
            try:
                db.add_check(r["id"], "model_eval", r["status"], evidence=str(r.get("evidence") or ""),
                             sha=record_sha, actor="frd_eval" + ("_arch" if arch else ""))
            except Exception:  # noqa: BLE001
                pass
    return rec
