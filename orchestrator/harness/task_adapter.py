# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""WP-41: the task-adapter boundary for acceptance.

The engine does not know what a task's graded quantity is. A task ships an
ADAPTER -- a small task-owned Python module in ``inputs/task_adapter.py`` (or
``CORESMITH_TASK_ADAPTER=<path>``) -- that runs the task's own driver and
checker (ideally the published grader verbatim) against a candidate and
returns a receipt. The engine's job is the part that is the same for every
task: assemble the candidate and give it an identity, run the adapter in its
declared interpreter with a timeout, refuse an incomplete receipt, and park a
typed failure for the policy. Nothing about h264, QSPI, Caravel or ppabench
lives here; that is the adapter's business (review round 2, overfitting
table).

Adapter module contract
-----------------------
Header comments (parsed without importing the module, so an adapter may
depend on packages the engine's interpreter lacks)::

    # coresmith-python: /path/to/python      (default: the engine's interpreter)
    # coresmith-timeout-s: 5400              (default: CORESMITH_TASK_ADAPTER_TIMEOUT_S or 3600)

Module attributes:

``CASES``  list[str]  -- every case the receipt MUST report, declared up front.
``TOP``    str|None   -- the graded top module; when set, the candidate's top
                         must match or the oracle is incomplete.
``LABEL``  str        -- what this adapter is (e.g. "published grader
                         hidden_tb + throughput gate"; "author smoke test").
``grade(candidate: dict, workdir: str) -> dict``::

    candidate = {"top", "sources": [abs paths], "project_root", "inputs_dir",
                 "candidate_sha"}
    returns   = {"cases": {name: {"ok": bool, "kind": "functional_fail" |
                                  "budget_fail" | None, "cycles": int|None,
                                  "detail": str}},
                 "budgets": {name: {"ok": bool, "detail": str, ...}}  # optional
                 "detail": str, "artifacts": {...}}                     # optional

The adapter runs in a subprocess (``task_adapter_runner.py``, no engine
imports). Exit code, JSON shape, declared-vs-reported case set and boolean
``ok`` fields are checked here; anything short of a complete receipt is
``oracle_incomplete`` (kind: oracle_incomplete / adapter_defect /
infrastructure_error) and parks -- never a skip, never a pass.

Result shape: identical to ``acceptance_dv.run_acceptance_dv`` so the
Validation DV node's park payload (WP-19/WP-38) is unchanged, plus
``adapter``, ``label``, ``budgets``.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_RUNNER = Path(__file__).with_name("task_adapter_runner.py")
_HEADER_RE = re.compile(r"^#\s*coresmith-([a-z0-9-]+)\s*:\s*(.+?)\s*$")


def adapter_path(project_root: str) -> str:
    """The task adapter for this project, or "" when none is declared."""
    envp = (os.environ.get("CORESMITH_TASK_ADAPTER", "") or "").strip()
    if envp:
        # WP-52: a DECLARED adapter that is missing is a defect, not "no adapter".
        return envp if Path(envp).exists() else f"MISSING:{envp}"
    p = Path(project_root) / "inputs" / "task_adapter.py"
    return str(p) if p.exists() else ""


def read_header(path: str) -> dict:
    """``# coresmith-<key>: <value>`` lines from the top of the adapter."""
    out: dict = {}
    try:
        for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines()[:40]:
            m = _HEADER_RE.match(line.strip())
            if m:
                out[m.group(1)] = m.group(2)
    except OSError:
        pass
    return out


def _incomplete(reason: str, kind: str, **extra) -> dict:
    logger.warning("task adapter: INCOMPLETE (%s) -- %s", kind, reason)
    out = {"passed": False, "skipped": False, "oracle_incomplete": True,
           "kind": kind, "reason": reason, "cases": [],
           "violations": [{
               "type": "acceptance_dv_failure",
               "criterion": "acceptance_dv_oracle_incomplete",
               "kind": kind,
               "suggested_fix": "Not an RTL verdict: the task adapter did not "
                                "produce a complete receipt. Fix the adapter / "
                                "toolchain, then retry.",
           }]}
    out.update(extra)
    return out


def assemble_candidate(project_root: str, top_rtl: str, block_rtls: Any) -> dict:
    """Grade precisely the adopted manifest, including its configuration."""
    from orchestrator.harness.top_module import candidate_for_inputs
    rec = candidate_for_inputs(project_root, top_rtl, block_rtls)
    return {"top": rec["top_module"], "sources": rec["sources"],
            "project_root": str(Path(project_root).resolve()),
            "inputs_dir": str(Path(project_root).resolve() / "inputs"),
            "candidate_sha": rec["candidate_sha"], "dependencies": rec["dependencies"],
            "defines": rec["defines"], "parameters": rec["parameters"]}


def _rows_and_violations(receipt: dict, declared: list[str]) -> tuple[list, list, dict]:
    rows, violations = [], []
    cases = receipt.get("cases") or {}
    for name in declared:
        c = cases.get(name) or {}
        ok = c.get("ok")
        kind = c.get("kind") or (None if ok else "functional_fail")
        row = {"name": name, "ok": bool(ok), "cycles": c.get("cycles"),
               "criterion": "task_adapter", "kind": kind if not ok else None,
               "note": str(c.get("detail", ""))[:600]}
        rows.append(row)
        if not ok:
            violations.append({
                "type": "acceptance_dv_failure",
                "criterion": ("task_adapter_budget" if kind == "budget_fail"
                              else "task_adapter_functional"),
                "acceptance_case": name, "kind": kind,
                "gap_class": "mission",
                "suggested_fix": str(c.get("detail", ""))[:800] or
                                 "the task's own checker rejected this case",
            })
    budgets = receipt.get("budgets") or {}
    for bname, b in budgets.items():
        if not (isinstance(b, dict) and b.get("ok") is True):
            violations.append({
                "type": "acceptance_dv_failure",
                "criterion": "task_adapter_budget",
                "budget": bname, "kind": "budget_fail",
                "gap_class": "mission",
                "measured": {k: v for k, v in b.items() if k not in ("ok", "detail")},
                "suggested_fix": str(b.get("detail", ""))[:800] or
                                 f"over the task's {bname} budget",
            })
    return rows, violations, budgets


_SYSTEM_BWRAP = ("/usr/bin/bwrap", "/usr/local/bin/bwrap")


def _find_bwrap() -> str:
    """The bubblewrap binary for the adapter boundary (WP-77).

    ``CORESMITH_BWRAP`` wins; otherwise the SYSTEM bubblewrap is preferred over
    whatever ``PATH`` resolves first: EDA toolchain bundles (oss-cad-suite)
    ship their own ``bwrap`` that cannot create user namespaces on a normal
    host, and a run's PATH puts the toolchain first. Empty when none exists."""
    override = os.environ.get("CORESMITH_BWRAP", "").strip()
    if override:
        return override
    for cand in _SYSTEM_BWRAP:
        if os.access(cand, os.X_OK):
            return cand
    return shutil.which("bwrap") or ""


def _sandbox_argv(command: list[str], work: Path, project_root: str = "") -> list[str]:
    """Only an explicit owner opt-out permits an unsandboxed evaluator.

    WP-79: the mounts are ordered so the adapter can always read its own
    project and write its own work directory. ``--tmpfs /tmp`` gives the
    adapter private scratch, but it also masks anything living under /tmp, and
    a project there used to lose its candidate manifest, sources and inputs
    with a confusing "no receipt" failure. The project is therefore re-bound
    read-only after the tmpfs, and the work directory writable after that.
    """
    policy = os.environ.get("CORESMITH_ADAPTER_SANDBOX", "bwrap").strip().lower()
    if policy == "none":
        return command
    if policy != "bwrap":
        raise OSError(f"Unknown adapter sandbox policy {policy!r}")
    bwrap = _find_bwrap()
    if not bwrap:
        raise OSError("bubblewrap unavailable; adapter sandbox required")
    argv = [bwrap, "--ro-bind", "/", "/",
            "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp"]
    root = str(Path(project_root).resolve()) if project_root else ""
    if root and root != "/":
        argv += ["--ro-bind", root, root]
    argv += ["--bind", str(work), str(work),
             "--unshare-net", "--die-with-parent", *command]
    return argv


def _new_attempt(root: Path, sha: str) -> Path:
    base = root / ".coresmith/acceptance" / sha
    base.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        attempt = base / f"attempt-{index:06d}"
        try:
            attempt.mkdir()
            return attempt
        except FileExistsError:
            index += 1


def run_task_adapter(project_root: str, top_rtl: str, block_rtls: Any = None) -> dict | None:
    """Run the task adapter on the candidate. ``None`` when no adapter is declared."""
    apath = adapter_path(project_root)
    if not apath:
        return None
    if apath.startswith("MISSING:"):
        return _incomplete(f"declared task adapter not found: {apath[8:]}",
                           "adapter_defect", adapter=apath[8:])
    from orchestrator.harness.top_module import CandidateError, validated_candidate
    try:
        cand = assemble_candidate(project_root, top_rtl, block_rtls)
        if cand["candidate_sha"] != validated_candidate(project_root)["candidate_sha"]:
            raise CandidateError("Candidate changed before adapter invocation")
    except CandidateError as exc:
        return _incomplete(str(exc), exc.kind, adapter=apath)
    from orchestrator.state_store.trust import check_oracle_manifest
    integrity = check_oracle_manifest(project_root)
    if not integrity["ok"]:
        return _incomplete("Oracle trust baseline unavailable or changed", "infrastructure_error",
                           adapter=apath, oracle_integrity=integrity)
    hdr = read_header(apath)
    python = (os.environ.get("CORESMITH_TASK_ADAPTER_PYTHON", "") or
              hdr.get("python") or sys.executable)
    # A bare name (``# coresmith-python: python3``) resolves on PATH, so an
    # adapter can be portable instead of hardcoding one host's interpreter.
    if python and os.sep not in python:
        python = shutil.which(python) or python
    try:
        timeout = int(float(hdr.get("timeout-s") or os.environ.get(
            "CORESMITH_TASK_ADAPTER_TIMEOUT_S", "3600") or 3600))
    except ValueError:
        timeout = 3600
    if not Path(python).exists():
        return _incomplete(f"adapter interpreter not found: {python}", "adapter_defect",
                           adapter=apath)

    keep = _new_attempt(Path(project_root).resolve(), cand["candidate_sha"])
    work = keep / "work"
    work.mkdir()
    cand_json = keep / "candidate.json"
    receipt_json = keep / "receipt.json"
    provisional = work / ".runner-receipt.json"
    log_path = keep / "adapter.log"
    cand_json.write_text(json.dumps(cand, indent=1))
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TMPDIR"] = "/tmp"
    # Evaluators must not depend on a writable host home. Older application
    # libraries prefer HOME to APPDATA and do not honor XDG_CACHE_HOME.
    cache = work / ".cache"
    cache.mkdir()
    env.pop("HOME", None)
    env["APPDATA"] = str(cache)
    env["XDG_CACHE_HOME"] = str(cache)
    try:
        command = _sandbox_argv([python, str(_RUNNER), str(Path(apath).resolve()),
                                str(cand_json), str(provisional), str(work)], work,
                                project_root=project_root)
        with open(log_path, "x", encoding="utf-8") as lf:
            r = subprocess.run(command, stdout=lf, stderr=subprocess.STDOUT, text=True,
                               timeout=timeout, cwd=str(work), env=env)
        if provisional.is_file():
            # Only the engine can publish into the immutable attempt directory.
            with receipt_json.open("xb") as fh:
                fh.write(provisional.read_bytes())
    except subprocess.TimeoutExpired:
        return _incomplete(f"task adapter exceeded {timeout}s", "infrastructure_error",
                           adapter=apath, candidate_sha=cand["candidate_sha"],
                           captured_dir=str(keep), adapter_log=str(log_path))
    except OSError as exc:
        return _incomplete(f"task adapter could not start: {exc}", "infrastructure_error",
                           adapter=apath, candidate_sha=cand["candidate_sha"],
                           captured_dir=str(keep))
    try:
        if validated_candidate(project_root)["candidate_sha"] != cand["candidate_sha"]:
            raise CandidateError("Candidate changed during adapter evaluation")
    except CandidateError as exc:
        return _incomplete(str(exc), exc.kind, adapter=apath, candidate_sha=cand["candidate_sha"])
    integrity = check_oracle_manifest(project_root)
    if not integrity["ok"]:
        return _incomplete("Oracle changed during adapter evaluation", "infrastructure_error",
                           adapter=apath, oracle_integrity=integrity, captured_dir=str(keep))
    if not receipt_json.exists():
        tail = ""
        try:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-600:]
        except OSError:
            pass
        return _incomplete(f"task adapter exited rc={r.returncode} without a receipt: {tail}",
                           "adapter_defect" if r.returncode == 3 else "infrastructure_error",
                           adapter=apath, candidate_sha=cand["candidate_sha"],
                           captured_dir=str(keep), adapter_log=str(log_path))
    if r.returncode != 0:
        # WP-52: a receipt from a process that then failed is not a receipt.
        return _incomplete(f"task adapter exited rc={r.returncode} after writing a receipt",
                           "infrastructure_error", adapter=apath,
                           candidate_sha=cand["candidate_sha"], captured_dir=str(keep),
                           adapter_log=str(log_path))
    try:
        receipt = json.loads(receipt_json.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return _incomplete(f"task adapter receipt unreadable: {exc}", "oracle_incomplete",
                           adapter=apath, candidate_sha=cand["candidate_sha"],
                           captured_dir=str(keep), adapter_log=str(log_path))
    common = {"adapter": apath, "label": str(receipt.get("label", "")),
              "candidate_sha": cand["candidate_sha"], "sources": cand["sources"],
              "top": cand["top"], "captured_dir": str(keep), "adapter_log": str(log_path)}
    if receipt.get("error"):
        return _incomplete(f"task adapter raised: {str(receipt['error'])[:600]}",
                           "adapter_defect", **common)
    declared = receipt.get("declared_cases")
    if not isinstance(declared, list) or not declared or \
            len(set(map(str, declared))) != len(declared):
        return _incomplete("adapter declares no CASES (or duplicates)", "adapter_defect", **common)
    declared = [str(d) for d in declared]
    if receipt.get("declared_top") and str(receipt["declared_top"]) != cand["top"]:
        # WP-45: a wrong top is an INTEGRATION defect (the graded boundary was
        # not produced), not an oracle problem: park with fix actions.
        _why = (f"the task grades top {receipt['declared_top']!r} but the candidate's "
                f"top module is {cand['top']!r}; the graded boundary was not produced")
        summary = {"passed": False, "skipped": False, "kind": "boundary_mismatch",
                   "reason": _why, "cases": [], "violations": [{
                       "type": "acceptance_dv_failure",
                       "criterion": "task_adapter_boundary", "kind": "boundary_mismatch",
                       "gap_class": "integration", "suggested_fix": _why}],
                   "budgets": {}, "requested_cases": 0, "completed_cases": 0, **common}
        try:
            (keep / "summary.json").write_text(json.dumps(summary, default=str, indent=1))
            (Path(project_root) / ".coresmith" / "acceptance_dv.json").write_text(
                json.dumps(summary, default=str, indent=1))
        except OSError:
            pass
        return summary
    cases = receipt.get("cases")
    if not isinstance(cases, dict):
        return _incomplete("receipt has no cases dict", "oracle_incomplete", **common)
    missing = [n for n in declared if n not in cases]
    extra = [n for n in cases if n not in declared]
    bad = [n for n in declared if n in cases and (not isinstance(cases[n], dict)
                                                  or not isinstance(cases[n].get("ok"), bool))]
    if missing or extra or bad:
        return _incomplete(
            "receipt is incomplete: " + "; ".join(
                s for s in (f"missing {missing}" if missing else "",
                            f"undeclared {extra}" if extra else "",
                            f"non-boolean ok {bad}" if bad else "") if s),
            "oracle_incomplete", **common, requested_cases=len(declared),
            completed_cases=len(declared) - len(missing) - len(bad))
    # WP-52: budgets are validated like cases -- a dict per budget with a boolean ok.
    _budgets = receipt.get("budgets")
    if _budgets is not None and not isinstance(_budgets, dict):
        return _incomplete("receipt budgets is not a dict", "oracle_incomplete", **common)
    _bad_b = [k for k, v in (_budgets or {}).items()
              if not isinstance(v, dict) or not isinstance(v.get("ok"), bool)]
    if _bad_b:
        return _incomplete(f"budget(s) without a boolean ok: {_bad_b}", "oracle_incomplete",
                           **common)
    rows, violations, budgets = _rows_and_violations(receipt, declared)
    passed = all(r["ok"] for r in rows) and not violations
    kind = None
    if not passed:
        kind = ("functional_fail" if any(not r["ok"] and r.get("kind") != "budget_fail"
                                         for r in rows) else "budget_fail")
    reason = (f"{len(rows)} task-adapter case(s), {sum(1 for r in rows if not r['ok'])} "
              f"failing, {sum(1 for v in violations if v['criterion'] == 'task_adapter_budget')} "
              f"budget violation(s)" + (f" [{receipt.get('label')}]" if receipt.get("label") else ""))
    summary = {"passed": passed, "skipped": False, "kind": kind, "reason": reason,
               "cases": rows, "violations": violations, "budgets": budgets,
               "detail": str(receipt.get("detail", ""))[:2000],
               "requested_cases": len(declared), "completed_cases": len(declared),
               **common}
    try:
        (keep / "summary.json").write_text(json.dumps(summary, default=str, indent=1))
        out_json = Path(project_root) / ".coresmith" / "acceptance_dv.json"
        out_json.write_text(json.dumps(summary, default=str, indent=1))
    except OSError:
        pass
    logger.info("task adapter: %s (%s)", "PASSED" if passed else "FAILED", reason)
    return summary
