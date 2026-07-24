#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Headless auto-approver canary for the nightly CI (Package C, C6).

Drives the smallest real design (``examples/adder8``) end-to-end through the
coresmith daemon with NO human in the loop -- it polls ``coresmith state`` and
resumes every interrupt with a safe default action, until the pipeline reaches a
terminal state. It then asserts the frontend completed and (if the run got that
far) that ``integration_dv``/``validation_dv`` did not fail, and finally snapshots
a ``post-rtl`` stage fixture -- the nightly is the refresh source for the
in-repo stage fixtures + the replay corpus (``.coresmith/llm_calls.jsonl``).

This is CI-only: it needs a live LLM (CLAUDE_CODE_OAUTH_TOKEN / codex auth) and
EDA tools, so it runs in the nightly Docker image, never in PR CI. It shells out
to ``bin/coresmith`` (the tested CLI) rather than importing the daemon so it
exercises the same control surface an operator uses.

Default-answer policy (mirrors CLAUDE.md's outer-agent contract):
  uarch_spec_review / uarch_integration_review -> approve
  integration_check                            -> accept (else approve)
  integration_dv / validation_dv (failure)    -> stop (abort/skip) + FAIL verdict
                                                  (these interrupts ONLY fire on
                                                  failure -- never auto-approve one)
  per-block ask_human                          -> retry (bounded), then skip

A daemon HTTP 409 (task still running / no pending interrupt -- a transient race
in the poll loop) is retried with exponential backoff, then errors the run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_CLI = _REPO / "bin" / "coresmith"


def _cli(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_CLI), *args],
        text=True, capture_output=True, check=check,
    )


def _state(project_root: str) -> dict:
    cp = _cli("state", "--project-root", project_root, check=False)
    try:
        return json.loads(cp.stdout)
    except json.JSONDecodeError:
        return {"status": "unknown", "_raw": cp.stdout[-500:], "_err": cp.stderr[-500:]}


def _first_interrupt(state: dict) -> dict | None:
    for it in state.get("interrupts", []) or []:
        payload = it.get("payload", it) if isinstance(it, dict) else {}
        if payload:
            return payload
    return None


def _is_dv_failure(payload: dict) -> bool:
    """True when the interrupt is a FAILED integration_dv / validation_dv.

    Those nodes ONLY raise an interrupt on failure -- their ``type`` is
    ``integration_dv_failure`` / ``validation_dv_failure`` and their
    ``supported_actions`` are ``retry``/``fix_rtl``/``fix_tb``/``abort`` (never
    ``approve``). So reaching one at all means DV failed and the chip is NOT
    verified. Detect it robustly (type substring OR an explicit failed/passed
    flag) so the canary can never approve it."""
    itype = (payload.get("type") or "").lower()
    if "integration_dv" in itype or "validation_dv" in itype:
        return True
    if payload.get("failed") is True:
        return True
    if payload.get("passed") is False and "dv" in itype:
        return True
    return False


# A DV failure the canary retries ONCE per run (shared budget) when its contract
# audit classes it into one of these transient/locally-fixable categories with at
# least this confidence AND local_fix_possible. A transient flake or a regenerable
# testbench should not turn a fixable run into a terminal FAIL. Functional
# mismatch classes (LOCAL_RTL_BUG, TOP_WIRING_BUG, UARCH_*, ARCHITECTURE_ERROR)
# are NEVER retried -- they stop immediately.
_RETRY_MIN_CONFIDENCE = 0.85
# The retry-once categories in the contract-audit taxonomy (see
# orchestrator/langchain/prompts/contract_audit.md):
#   DV_PROCESS_ERROR -- missing/empty/header-only VCD or WaveKit audit, i.e. a
#     build/trace flake rather than a real design/test bug;
#   TESTBENCH_BUG    -- a regenerable testbench; the engine's own retry
#     regenerates the TB and it often passes (rung2 defect 5: the canary used to
#     stop on a high-confidence TESTBENCH_BUG the engine could have fixed).
_RETRY_ONCE_CATEGORIES = {"DV_PROCESS_ERROR", "TESTBENCH_BUG"}


def _load_contract_audit(payload: dict, project_root: str) -> dict:
    """Return the contract-audit dict for a DV-failure interrupt.

    The failure payload embeds the full ``contract_audit`` dict (and a
    ``contract_audit_path``); fall back to reading the JSON off disk if only the
    path is present, and finally to the newest ``.coresmith/contract_audit/*.json``
    in the run dir. Returns ``{}`` when nothing is available."""
    audit = payload.get("contract_audit")
    if isinstance(audit, dict) and audit:
        return audit
    for key in ("contract_audit_path", "audit_path"):
        p = payload.get(key)
        if p:
            try:
                return json.loads(Path(p).read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                pass
    try:
        audit_dir = Path(project_root) / ".coresmith" / "contract_audit"
        cands = sorted(audit_dir.glob("*.json"),
                       key=lambda q: q.stat().st_mtime, reverse=True)
        if cands:
            return json.loads(cands[0].read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        pass
    return {}


def _is_retryable_dv_failure(audit: dict) -> bool:
    """True when the audit classes the DV failure into a retry-once category
    (DV_PROCESS_ERROR / TESTBENCH_BUG) with confidence >= _RETRY_MIN_CONFIDENCE
    AND local_fix_possible -- the only cases the canary retries (shared budget of
    1 per run). Functional-mismatch classes always stop immediately."""
    if not isinstance(audit, dict):
        return False
    category = str(audit.get("category", "")).upper()
    if category not in _RETRY_ONCE_CATEGORIES:
        return False
    if not audit.get("local_fix_possible"):
        return False
    try:
        confidence = float(audit.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return confidence >= _RETRY_MIN_CONFIDENCE


# Back-compat alias (the predicate broadened beyond process errors in rung2).
_is_process_error = _is_retryable_dv_failure


def _pick_action(payload: dict, retries_seen: int, max_retries: int) -> dict:
    """Choose a resume action for an interrupt payload.

    A DV-failure interrupt is NEVER approved -- approving a failed
    integration_dv/validation_dv would ship an unverified chip. Pick the stop
    action its payload actually supports (abort, else skip)."""
    itype = (payload.get("type") or "").lower()
    actions = payload.get("supported_actions") or []

    def has(a):
        return a in actions or not actions

    if _is_dv_failure(payload):
        for a in ("abort", "skip"):
            if a in actions:
                return {"action": a}
        return {"action": actions[0] if actions else "abort"}
    if "integration_check" in itype:
        return {"action": "accept" if "accept" in actions else "approve"}
    if "spec_review" in itype or "integration_review" in itype:
        return {"action": "approve"}
    # Per-block failure: retry within budget, then skip.
    if retries_seen >= max_retries:
        for a in ("skip", "abort"):
            if a in actions:
                return {"action": a}
    for a in ("retry", "approve", "accept"):
        if has(a):
            return {"action": a}
    return {"action": actions[0] if actions else "abort"}


def _is_conflict(cp: subprocess.CompletedProcess) -> bool:
    """The CLI exits nonzero and echoes ``... -> 409 ...`` (or a conflict phrase)
    when the daemon returns HTTP 409 (task still running / no pending interrupt).
    Those are transient races in a polling loop and are safe to retry."""
    if cp.returncode == 0:
        return False
    text = f"{cp.stderr or ''}\n{cp.stdout or ''}".lower()
    return any(
        marker in text
        for marker in ("409", "still running", "nothing to resume",
                       "no pending interrupt")
    )


def _resume(project_root: str, action: dict, *, max_attempts: int = 4,
            base_backoff_s: float = 1.0, sleeper=time.sleep, runner=None) -> bool:
    """Send one resume; retry a 409 conflict with exponential backoff, then error.

    Returns True on success, False on an unrecoverable error (a persistent 409 or
    any non-conflict CLI error such as a 400 unsupported action / dead daemon).
    ``runner``/``sleeper`` are injectable so the decision is unit-testable without
    a live daemon."""
    def _default_runner() -> subprocess.CompletedProcess:
        return _cli("resume", "--project-root", project_root,
                    "--action", action["action"], check=False)

    run = runner or _default_runner
    for attempt in range(max_attempts):
        cp = run()
        if cp.returncode == 0:
            return True
        if _is_conflict(cp):
            if attempt < max_attempts - 1:
                sleeper(base_backoff_s * (2 ** attempt))
                continue
            print(f"[canary] resume conflict (HTTP 409) persisted after "
                  f"{max_attempts} attempts; giving up", flush=True)
            return False
        print(f"[canary] resume failed (rc={cp.returncode}): "
              f"{(cp.stderr or cp.stdout or '')[-300:]}", flush=True)
        return False
    return False


def run_canary(*, project_root: str, blocks_file: str, timeout_s: float,
               poll_s: float, max_retries: int) -> int:
    pr = Path(project_root)
    (pr / "inputs").mkdir(parents=True, exist_ok=True)

    _cli("daemon", "start", "--project-root", project_root, check=False)
    _cli("run", "start", "--project-root", project_root,
         "--blocks-file", blocks_file, check=False)

    deadline = time.monotonic() + timeout_s
    retries_seen = 0
    last_sig = None
    dv_failed = False
    resume_error = False
    dv_retry_budget = 1  # per-run: one autonomous retry for a PROCESS/infra DV flake
    while time.monotonic() < deadline:
        state = _state(project_root)
        status = state.get("status", "unknown")
        payload = _first_interrupt(state)
        if payload is not None:
            sig = (payload.get("type"),
                   payload.get("block_name") or payload.get("block"))
            retries_seen = retries_seen + 1 if sig == last_sig else 0
            last_sig = sig
            action = _pick_action(payload, retries_seen, max_retries)
            blk = payload.get("block_name") or payload.get("block")
            if _is_dv_failure(payload):
                # A failed integration_dv / validation_dv normally means the chip
                # is NOT verified -> stop cleanly + FAIL (never approve). BUT when
                # the contract audit classifies it into a retry-once category
                # (DV_PROCESS_ERROR trace/build flake OR a regenerable
                # TESTBENCH_BUG) with confidence >= 0.85 AND local_fix_possible,
                # retry ONCE per run -- a transient flake or a fixable TB should
                # not turn a fixable run terminal. A second failure of ANY kind
                # (shared budget exhausted) or a functional mismatch stops now.
                audit = _load_contract_audit(payload, project_root)
                supported = payload.get("supported_actions") or []
                if (dv_retry_budget > 0 and _is_retryable_dv_failure(audit)
                        and "retry" in supported):
                    dv_retry_budget -= 1
                    print(f"[canary] interrupt={payload.get('type')} block={blk} "
                          f"-> retry (retryable class "
                          f"{audit.get('category')} conf={audit.get('confidence')}, "
                          f"local_fix_possible={audit.get('local_fix_possible')}; "
                          f"retry budget now {dv_retry_budget})", flush=True)
                    if not _resume(project_root, {"action": "retry"}):
                        resume_error = True
                        break
                    time.sleep(poll_s)
                    continue
                print(f"[canary] interrupt={payload.get('type')} block={blk} -> "
                      f"{action['action']} (DV FAILED, class="
                      f"{audit.get('category', 'UNKNOWN')}); stopping with a "
                      f"failure verdict", flush=True)
                dv_failed = True
                _resume(project_root, action)
                break
            print(f"[canary] interrupt={payload.get('type')} "
                  f"block={blk} -> {action['action']}", flush=True)
            if not _resume(project_root, action):
                resume_error = True
                break
            time.sleep(poll_s)
            continue
        if status in ("done", "error", "completed"):
            break
        time.sleep(poll_s)

    final = _state(project_root)
    print("[canary] final state:", json.dumps({
        k: final.get(k) for k in ("status", "pipeline_done", "completed_count",
                                  "total_blocks")
    }), flush=True)

    # Snapshot a post-rtl fixture (best-effort; the artifact upload picks it up).
    try:
        snap_out = pr / ".coresmith" / "stage_snapshot_post_rtl"
        subprocess.run(
            [sys.executable, str(_REPO / "scripts" / "snapshot_stage.py"),
             "--project-root", project_root, "--stage", "post-rtl",
             "--graph", "pipeline", "--out", str(snap_out), "--allow-unparked"],
            check=False,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[canary] snapshot skipped: {exc}", flush=True)

    # Success = the frontend completed without erroring AND no DV failure /
    # resume error was seen. A failed integration_dv/validation_dv interrupt
    # (dv_failed) hard-fails the run even if pipeline_done is True -- the per-block
    # frontend can be done while the CHIP is unverified (see CLAUDE.md "What
    # 'done' means"), so pipeline_done alone is NOT sufficient.
    ok = (
        not dv_failed
        and not resume_error
        and bool(final.get("pipeline_done"))
        and final.get("status") != "error"
        and final.get("completed_count", 0) >= (final.get("total_blocks") or 1)
    )
    print(f"[canary] verdict: {'PASS' if ok else 'FAIL'}"
          + (" (DV FAILED)" if dv_failed else "")
          + (" (RESUME ERROR)" if resume_error else ""), flush=True)
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--blocks-file",
                    default=str(_REPO / "examples" / "adder8" / "blocks.yaml"))
    ap.add_argument("--timeout-s", type=float, default=1800.0)
    ap.add_argument("--poll-s", type=float, default=10.0)
    ap.add_argument("--max-retries", type=int, default=3)
    args = ap.parse_args(argv)
    return run_canary(
        project_root=args.project_root, blocks_file=args.blocks_file,
        timeout_s=args.timeout_s, poll_s=args.poll_s, max_retries=args.max_retries,
    )


if __name__ == "__main__":
    raise SystemExit(main())
