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

    FRD_EVAL {"id": "PERF-001", "status": "pass", "value": 207, "unit": "cycles", "evidence": "..."}

with ``status`` in :data:`STATUSES`, and ``FRD_EVAL_DONE`` at the end. A
bounded item (``bound_min``/``bound_max`` in ``requirements.json``) is
decided by its measured ``value`` (or a separate ``VALUE <id> <number>
[unit]`` line), never by the harness's own verdict: the value derives
pass/fail from the bounds, and a pass/fail verdict with no value is recorded
``tool_error`` ("no measured value from the model harness") --
``CORESMITH_MODEL_EVAL_REQUIRE_VALUE=0`` restores verdict-only judging.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

from orchestrator.processes import run as run_process

STATUSES = ("pass", "fail", "not_testable", "skipped")
# the engine's verdict for a bounded pass/fail that carries no value (never a harness status)
NO_VALUE_STATUS = "tool_error"
NO_VALUE_EVIDENCE = "no measured value from the model harness"

# ``1. **ID**: PERF-001`` blocks as the FRD prompt formats them; the section
# heading above a block is its category.
_ID_RE = re.compile(r"\*\*ID\*\*:\s*`?([A-Z][A-Z0-9]*-[A-Z0-9-]*\d)`?", re.I)
_FIELD_RE = re.compile(r"\*\*(Requirement|Acceptance criteria|Priority|Model check|Metric)\*\*:\s*(.*)", re.I)
_LINE_RE = re.compile(r"^FRD_EVAL\s+(\{.*\})\s*$", re.M)
_VALUE_RE = re.compile(r"^VALUE\s+([A-Za-z][A-Za-z0-9_-]*)\s+([-+0-9.eE]+)(?:\s+(\S+))?\s*$", re.M)


def require_value_enabled() -> bool:
    """``CORESMITH_MODEL_EVAL_REQUIRE_VALUE`` (default ``1``)."""
    return (os.environ.get("CORESMITH_MODEL_EVAL_REQUIRE_VALUE", "1") or "1").strip().lower() \
        not in {"0", "false", "no", "off"}


def _num(v):
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def bounded(req: dict | None) -> bool:
    return bool(req) and (req.get("bound_min") is not None or req.get("bound_max") is not None)


def status_from_bounds(req: dict, value: float) -> str:
    lo, hi = req.get("bound_min"), req.get("bound_max")
    return "pass" if (lo is None or value >= lo) and (hi is None or value <= hi) else "fail"


def _parse_metric(val: str) -> dict:
    """``fps; min 55; max -; unit fps`` -> ``{metric, bound_min, bound_max, unit}``
    (``-`` is open-ended; malformed/non-finite bounds are rejected)."""
    parts = [p.strip() for p in val.strip("`* ").split(";")]
    out = {"metric": parts[0] or None, "bound_min": None, "bound_max": None, "unit": None}
    for p in parts[1:]:
        key, _, v = p.partition(" ")
        key, v = key.lower(), v.strip()
        if key in ("min", "max"):
            if v in ("", "-"):
                out[f"bound_{key}"] = None
            else:
                parsed = _num(v.replace(",", ""))
                if parsed is None:
                    raise ValueError(f"malformed metric {key} bound: {v!r}")
                out[f"bound_{key}"] = parsed
        elif key == "unit":
            out["unit"] = None if v in ("", "-") else v
    return out


def extract_requirements(frd_text: str) -> list[dict]:
    """Every identified requirement of the FRD: ``{id, section, requirement,
    acceptance, priority, model_check}`` (plus ``metric``, ``bound_min``,
    ``bound_max``, ``unit`` when the block has a ``**Metric**`` line).
    Tolerant of formatting drift: an id with no fields still yields a record
    (text = the block's first lines)."""
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
            elif key == "metric":
                cur.update(_parse_metric(val))
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
        p = run_process(["make", "-s", "frd_eval/frd_eval"], cwd=md, capture_output=True, text=True,
                           timeout=timeout_s, env=env)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "log": str(exc)}
    log = p.stdout + p.stderr
    (md / "frd_eval" / "build.log").write_text(log)
    return {"ok": p.returncode == 0 and (md / "frd_eval" / "frd_eval").exists(), "log": log[-8000:]}


def parse_results(text: str) -> list[dict]:
    """``FRD_EVAL {...}`` records; a numeric ``value`` (and ``unit``) in the
    JSON, or a ``VALUE <id> <number> [unit]`` line, is carried as ``value``."""
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
        rec["value"] = _num(rec.get("value"))
        out.append(rec)
    by_id = {r["id"]: r for r in out}
    for m in _VALUE_RE.finditer(text or ""):
        rec = by_id.get(m.group(1).upper())
        v = _num(m.group(2))
        if rec is not None and v is not None:
            rec["value"] = v
            if m.group(3) and not rec.get("unit"):
                rec["unit"] = m.group(3)
    return out


def judge_values(reqs: list[dict], results: list[dict]) -> list[dict]:
    """Decide every bounded item by its value (in place, returns ``results``):
    a value derives pass/fail from the bounds (the harness's own verdict is
    kept as ``harness_status`` when it differs); a pass/fail without a value
    becomes ``tool_error`` with ``no_value``. ``not_testable``/``skipped``
    stand. A no-op with ``CORESMITH_MODEL_EVAL_REQUIRE_VALUE=0``."""
    if not require_value_enabled():
        return results
    by_id = {q["id"]: q for q in reqs}
    for r in results:
        q = by_id.get(r["id"])
        if not bounded(q) or r["status"] not in ("pass", "fail"):
            continue
        if r.get("value") is None:
            r["harness_status"] = r["status"]
            r["status"] = NO_VALUE_STATUS
            r["no_value"] = True
            r["evidence"] = f"{NO_VALUE_EVIDENCE} (harness said {r['harness_status']}): {r.get('evidence') or ''}".strip()
            continue
        derived = status_from_bounds(q, float(r["value"]))
        if derived != r["status"]:
            r["harness_status"] = r["status"]
            r["status"] = derived
    return results


def overlay_bounds(reqs: list[dict], db) -> list[dict]:
    """The DB item's metric/bounds/unit are authoritative (the rendered FRD
    may be stale or file-sourced): copy them onto the extracted requirements."""
    if db is None:
        return reqs
    for q in reqs:
        try:
            it = db.item(q["id"])
        except Exception:  # noqa: BLE001
            it = None
        if not it:
            continue
        for k in ("metric", "bound_min", "bound_max", "unit"):
            if it.get(k) is not None:
                q[k] = it[k]
    return reqs


def run_harness(model_dir, *, timeout_s: int = 1800, args: list[str] | None = None) -> dict:
    md = Path(model_dir)
    exe = md / "frd_eval" / "frd_eval"
    if not exe.exists():
        return {"ok": False, "done": False, "results": [], "log": "frd_eval not built"}
    try:
        p = run_process([str(exe), *(args or [])], cwd=md, capture_output=True, text=True, timeout=timeout_s)
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
    counts = {s: 0 for s in (*STATUSES, NO_VALUE_STATUS)}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    failed = sorted(r["id"] for r in results if r["status"] == "fail")
    unanswered_must = sorted(q["id"] for q in reqs if must_have(q) and q["id"] not in by_id)
    unreasoned = sorted(r["id"] for r in results if r["status"] == "not_testable" and not str(r.get("evidence") or "").strip())
    unknown = sorted(set(by_id) - {q["id"] for q in reqs})
    no_value = sorted(r["id"] for r in results if r.get("no_value"))
    return {"counts": counts, "requirements": len(reqs), "answered": len(by_id), "failed": failed,
            "unanswered_must": unanswered_must, "not_testable_without_reason": unreasoned,
            "unknown_ids": unknown, "no_value": no_value,
            "gate_ok": not failed and not unanswered_must and not unreasoned and not no_value}


def _bound_text(q: dict) -> str:
    """``[55..]`` / ``[..400]`` / ``[1..2]`` ('' when the item has no bounds)."""
    if not bounded(q):
        return ""
    lo = "" if q.get("bound_min") is None else f"{q['bound_min']:g}"
    hi = "" if q.get("bound_max") is None else f"{q['bound_max']:g}"
    return f"[{lo}..{hi}]"


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
         "| id | priority | status | value | bound | evidence |", "|---|---|---|---|---|---|"]
    for q in reqs:
        r = by_id.get(q["id"], {})
        val = "" if r.get("value") is None else f"{r['value']:g}{(' ' + str(r.get('unit'))) if r.get('unit') else ''}"
        bound = _bound_text(q)
        L.append(f"| {q['id']} | {q.get('priority','')} | {r.get('status','**unanswered**')} | {val} | {bound} | "
                 f"{str(r.get('evidence',''))[:200].replace('|', '/')} |")
    for i in summary.get("unknown_ids") or []:
        L.append(f"| {i} | - | {by_id[i]['status']} | (not an FRD id) {str(by_id[i].get('evidence',''))[:120]} |")
    (Path(model_dir) / "frd_eval" / "REPORT.md").write_text("\n".join(L) + "\n")
    return out


def _log(msg: str) -> None:
    try:
        from orchestrator.langgraph.pipeline_helpers import GREEN, RED, YELLOW, log  # noqa: F401
    except Exception:  # noqa: BLE001
        print(msg)
        return
    colour = RED if ("gate_ok=False" in msg or "failing:" in msg or "author failed" in msg) \
        else (YELLOW if "attempt" in msg else GREEN)
    log(msg, colour)


def model_check_inputs(db, project_root) -> tuple[list[str], dict[str, str]]:
    """What the SoC-model evaluation is required for and what each verdict
    is bound to: ``(scope_ids, record_shas)``. The scope is the must-have
    FRD requirements that DECLARE a model check -- the registered ontology
    item when the database knows the id, the parsed FRD block otherwise
    (``builds.model_check_scope``); each verdict's sha binds the assembled
    model + harness digest to the requirement as it read when judged
    (``builds.model_check_sha``)."""
    from orchestrator.state_store.builds import model_check_scope, model_check_sha, soc_model_digest
    digest = soc_model_digest(project_root, db=db)
    frd = Path(project_root) / "arch" / "frd_spec.md"
    parsed = extract_requirements(frd.read_text(encoding="utf-8", errors="replace")) if frd.exists() else []
    ids = model_check_scope(db, parsed)
    return ids, {i: model_check_sha(db, project_root, i, model_digest=digest) for i in ids}


async def evaluate(project_root, model_dir, blocks: list[str], *, arch: bool = False, agent=None,
                   timeout_s: int | None = None, repairs: int | None = None, db=None,
                   record_sha: str | dict[str, str] = "", author: bool = False,
                   scope_ids: list[str] | None = None) -> dict:
    """The FRD evaluated on a model (the SoC LT model, or the executable SAD
    when ``arch``): requirement list -> harness (``<model_dir>/frd_eval/*.cpp``)
    -> build -> run -> verdicts. Never raises. When ``db`` is given every
    verdict is recorded as a ``model_eval`` check (hash-bound).

    ``author=False`` (the default; ``coresmith model eval``) is a pure check:
    the harness must already exist. A missing harness, a build failure, an
    incomplete run (no ``FRD_EVAL_DONE``) or a failed run (``FRD_EVAL_DONE``
    printed but a non-zero exit) is reported with its diagnostics (``error``
    = ``HARNESS_MISSING`` / ``HARNESS_BUILD_FAILED`` / ``HARNESS_RUN_INCOMPLETE``
    / ``HARNESS_RUN_FAILED``) and nothing is written, repaired or generated.
    Only a run that exits 0 AND prints ``FRD_EVAL_DONE`` (``run["ok"]``) can
    pass the gate or record ``model_eval`` checks: verdict rows printed by a
    process that then crashed or returned non-zero are diagnostics, not
    authoritative evidence.

    ``author=True`` (``coresmith harness author``, and the frontend's uArch
    phase when it explicitly opts into model authoring) lets the harness agent
    write a missing harness and repair it a bounded number of times
    (``CORESMITH_FRD_EVAL_REPAIRS``); repairs are only for harness defects
    (compile error, crash, hang, missing DONE) -- a ``fail`` verdict is the
    finding.

    ``scope_ids`` (from :func:`model_check_inputs`) restricts the evaluation
    to the requirements that declare a model check: only those are written
    to ``requirements.json`` for the harness, judged, summarized and
    recorded; a verdict the harness prints for any other id is kept as
    ``out_of_scope_results`` and never decides the gate. A requirement with
    no declared model check (a Linux-boot must-have, a DRC constraint) keeps
    its RTL / chip-level acceptance and is never demanded of the model.
    ``record_sha`` may be one digest or ``{id: digest}`` per requirement."""
    md = Path(model_dir)
    rec: dict = {"enabled": True, "gate_ok": None, "arch": arch, "authoring": bool(author)}
    frd = Path(project_root) / "arch" / "frd_spec.md"
    if not frd.exists():
        rec["skipped"] = "no arch/frd_spec.md"
        return rec
    reqs = overlay_bounds(extract_requirements(frd.read_text(encoding="utf-8", errors="replace")), db)
    if scope_ids is not None:
        declared = {str(i) for i in scope_ids}
        excluded = [q["id"] for q in reqs if q["id"] not in declared]
        reqs = [q for q in reqs if q["id"] in declared]
        rec["scope"] = {"declared_model_checks": sorted(declared), "evaluated": [q["id"] for q in reqs],
                        "excluded": excluded,
                        "undeclared_in_frd": sorted(declared - {q["id"] for q in reqs})}
    write_requirements(md, reqs)
    rec["requirements"] = len(reqs)
    if not reqs:
        if scope_ids is not None:
            rec["skipped"] = ("no FRD requirement declares a model check: nothing is required of the SoC model "
                              "(the other requirements keep their RTL / chip-level acceptance)")
            rec["gate_ok"] = True
        else:
            rec["skipped"] = "the FRD has no identified requirements (**ID**: XXX-NNN blocks)"
        return rec
    timeout_s = timeout_s or int(os.environ.get("CORESMITH_FRD_EVAL_TIMEOUT_S", "1800") or 1800)
    author = bool(author or agent is not None)   # a supplied agent IS the explicit authoring request
    if author:
        repairs = max(0, int(os.environ.get("CORESMITH_FRD_EVAL_REPAIRS", "3") or 3) if repairs is None else repairs)
        if agent is None:
            from orchestrator.langchain.agents.frd_eval_generator import FRDEvalGenerator
            agent = FRDEvalGenerator()
    else:
        repairs = 0
    rec["harness_sources"] = [str(p) for p in harness_sources(md)]
    if not author and not rec["harness_sources"]:
        rec.update({"error": "HARNESS_MISSING", "built": False, "required": str(md / "frd_eval" / "frd_eval.cpp"),
                    "hint": "write the harness (own sc_main, one FRD_EVAL line per requirement id, FRD_EVAL_DONE at "
                            "the end; the ids are in frd_eval/requirements.json) or request one: "
                            "coresmith harness author" + (" --arch" if arch else "")})
        _log("  [FRD-EVAL] no harness supplied (HARNESS_MISSING); nothing authored")
        return rec
    compiler_log, run_log, summary, run = "", "", None, None
    for attempt in range(1, repairs + 2):
        if author and not (attempt == 1 and harness_sources(md)):   # reuse an existing harness first
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
            rec["build_log"] = (b["log"] or "")[-4000:]
            if not author:
                rec["error"] = "HARNESS_BUILD_FAILED"
                _log("  [FRD-EVAL] harness build failed (HARNESS_BUILD_FAILED); nothing repaired")
                break
            _log(f"  [FRD-EVAL] harness build failed (attempt {attempt})")
            continue
        run = run_harness(md, timeout_s=timeout_s)
        if scope_ids is not None:
            in_scope = {q["id"] for q in reqs}
            rec["out_of_scope_results"] = [r for r in run["results"] if r["id"] not in in_scope]
            run["results"] = [r for r in run["results"] if r["id"] in in_scope]
        judge_values(reqs, run["results"])
        summary = summarize(reqs, run["results"])
        rec.update({"done": run["done"], "rc": run.get("rc"), "run_ok": bool(run["ok"]), "summary": summary})
        if run["ok"]:                                   # exit 0 AND FRD_EVAL_DONE
            break
        # Incomplete (no DONE) or failed (DONE, then a non-zero exit): a harness
        # defect either way; the printed verdict rows are diagnostics only.
        failure = "HARNESS_RUN_INCOMPLETE" if not run["done"] else "HARNESS_RUN_FAILED"
        compiler_log, run_log = "", run["log"]
        what = ("did not run to completion" if not run["done"]
                else f"exited with code {run.get('rc')} after FRD_EVAL_DONE")
        if not author:
            rec["error"] = failure
            rec["run_log"] = (run["log"] or "")[-4000:]
            _log(f"  [FRD-EVAL] harness {what} ({failure}); nothing repaired")
            break
        _log(f"  [FRD-EVAL] harness {what} ({failure}, attempt {attempt})")
    if run is not None:
        rec["report"] = str(write_report(project_root, md, reqs, run, summary, name="frd_eval_arch" if arch else "frd_eval"))
    run_ok = bool(run and run["ok"])
    rec["gate_ok"] = bool(run_ok and summary and summary["gate_ok"])
    if run is not None and not run_ok and "error" not in rec:
        rec["error"] = "HARNESS_RUN_INCOMPLETE" if not run["done"] else "HARNESS_RUN_FAILED"
        rec["run_log"] = (run["log"] or "")[-4000:]
    if summary:
        c = summary["counts"]
        _log(f"  [FRD-EVAL] {len(reqs)} requirement(s): pass={c['pass']} fail={c['fail']} "
             f"not_testable={c['not_testable']} skipped={c['skipped']} unanswered_must={len(summary['unanswered_must'])}"
             f" -> gate_ok={rec['gate_ok']}" + ("" if run_ok else f" (run not ok: rc={run.get('rc')}, done={run['done']})"))
        if summary["failed"]:
            _log(f"  [FRD-EVAL] failing: {', '.join(summary['failed'][:12])}")
    if summary and summary.get("no_value"):
        _log(f"  [FRD-EVAL] bounded items without a measured value (tool_error): {', '.join(summary['no_value'][:12])}")
    if db is not None and run_ok:
        record_checks(db, reqs, run["results"], sha=record_sha, actor="frd_eval" + ("_arch" if arch else ""))
    return rec


def record_checks(db, reqs: list[dict], results: list[dict], *, sha: str | dict[str, str] = "",
                  actor: str = "frd_eval") -> list[int]:
    """One ``model_eval`` check per verdict. A bounded item with a value is
    recorded with the value (its status then follows the DB item's bounds);
    ``tool_error`` rows (no value) carry the reason. With
    ``CORESMITH_MODEL_EVAL_REQUIRE_VALUE=0`` the harness verdict is recorded
    as before (no value). ``sha`` is one digest for every check or
    ``{id: digest}`` (a verdict bound to its own requirement's inputs; an id
    without a digest is recorded unbound, which readiness treats as stale)."""
    by_id = {q["id"]: q for q in reqs}
    ids = []
    strict = require_value_enabled()
    for r in results:
        value = r.get("value") if strict and bounded(by_id.get(r["id"])) else None
        status = r["status"]
        ev = str(r.get("evidence") or "")
        if value is not None and r.get("unit"):
            ev = f"{value:g} {r['unit']}: {ev}" if ev else f"{value:g} {r['unit']}"
        if value is not None and status in ("pass", "fail"):
            # the status was derived from the (DB-overlaid) bounds; the DB derives it again
            attempts = ((status, float(value)), (None, float(value)), (status, None))
        else:
            attempts = ((status, None),)
        row_sha = (sha.get(r["id"], "") if isinstance(sha, dict) else sha) or ""
        for st, v in attempts:
            try:
                ids.append(db.add_check(r["id"], "model_eval", st, evidence=ev, sha=row_sha, actor=actor, value=v))
                break
            except Exception:  # noqa: BLE001 - a conflict / an item the DB does not know: next form
                continue
    return ids
