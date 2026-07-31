# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Honest DRC verdicts: "could not measure" is a state of its own.

THE DEFECT THIS CLOSES
----------------------
The macro DRC stage (``bin/gen_ram`` -> ``_run_magic_drc_on_gds``) rendered a
DRC run that produced NO REPORT AT ALL as::

    {"pass": false, "violation_count": -1}   ->   stage {"ok": false, "violations": -1}

Two real macros from a validation run (``w32_d64``, ``w9_d512``)
carry exactly that, and the ``precheck_magic_drc.rpt`` those verdicts name was
never written -- Magic produced nothing even though the binary was on PATH.
``-1`` was *intended* as a "could not measure" sentinel, but every consumer read
``ok: false`` / a negative count as "this macro HAS DRC violations", and the
macro was marked FAIL for a measurement that never happened.

That is the proxy-not-property dishonesty the engine's honest gates exist to
prevent: absence of evidence rendered as evidence of failure -- and, in the
KLayout sibling (a missing XML report counted as ``violation_count = 0``),
absence of evidence rendered as evidence of SUCCESS. Both are false verdicts;
neither is a DRC measurement.

THE VOCABULARY
--------------
Deliberately the SAME words :mod:`orchestrator.harness.gate_sim` uses, IMPORTED
from it rather than re-spelled, so the engine has exactly one status vocabulary:

``pass``     the DRC tool completed, wrote a non-empty report, and that report
             yielded a parseable count of ZERO. A real clean result.
``fail``     a real, measured violation count of N > 0.
``not_run``  NO MEASUREMENT EXISTS: no report, an empty report, a tool that
             never started / died / timed out, or a report carrying no parseable
             count. Neither clean nor dirty. It ALWAYS carries a ``reason``
             naming why nothing could be measured, and it is NEVER ``pass``.

FAIL-CLOSED RULES (in the order :func:`classify_drc` applies them)
-----------------------------------------------------------------
1. A POSITIVE count is a real measurement whatever the tool's exit status -- a
   report holding violation rects is evidence even from a tool that then died.
   So N > 0 is always ``fail``.
2. Everything else is only a CANDIDATE "clean", and clean must be EARNED by
   positive evidence: tool completed AND report exists AND report is non-empty
   AND the count parsed. Any missing link is ``not_run`` with a reason, never a
   pass and never a fabricated count.
3. ``violation_count`` is ``None`` when unmeasured. It is NEVER ``-1``: a
   negative "count" invites arithmetic (``count != 0`` -> "dirty",
   ``count >= 0`` -> "measured") that silently converts absence into a verdict.

This module is pure and import-cheap (stdlib + the gate_sim status constants).
"""

from __future__ import annotations

from pathlib import Path

from orchestrator.harness.gate_sim import (
    STATUS_FAIL,
    STATUS_NOT_RUN,
    STATUS_PASS,
)

__all__ = [
    "STATUS_FAIL",
    "STATUS_NOT_RUN",
    "STATUS_PASS",
    "classify_drc",
    "drc_stage",
    "drc_summary",
    "unmeasured_drc",
]

# The legacy sentinel the producers used to emit for "no verdict". Kept only so
# a value arriving from older code/JSON is normalized to the honest ``None``.
LEGACY_UNMEASURED_COUNT = -1

_NOT_A_COUNT = (
    " This is NOT a violation count and NOT a clean result -- no DRC "
    "measurement exists for this layout."
)


def _verdict(status: str, reason: str, count: int | None,
             report_path: str, tool: str) -> dict:
    v: dict = {
        "tool": tool,
        "status": status,
        "reason": reason,
        # `measured` is the question every consumer actually wants answered:
        # does a violation count exist at all?
        "measured": status in (STATUS_PASS, STATUS_FAIL),
        # `pass` is True ONLY for a real clean measurement. Kept for the
        # existing precheck consumers, which stay fail-closed on not_run.
        "pass": status == STATUS_PASS,
        # None -- never -1 -- when nothing was measured.
        "violation_count": count,
        "report_path": str(report_path or ""),
    }
    if status == STATUS_NOT_RUN:
        # Readers that only surface `error` (e.g. tapeout_diagnosis) still say
        # WHY there is no measurement instead of printing a bare "-1".
        v["error"] = reason
    return v


def unmeasured_drc(reason: str, *, report_path: str = "",
                   tool: str = "Magic") -> dict:
    """An explicit "no DRC measurement exists" verdict.

    For callers that know up front that no DRC ran (e.g. a generator whose
    config disables DRC). Never a pass, never a count, always a reason.
    """
    return _verdict(STATUS_NOT_RUN, reason.strip() or "no reason recorded",
                    None, report_path, tool)


def classify_drc(*, violation_count: int | None, report_path: str = "",
                 tool_ran: bool = True, tool_error: str = "",
                 tool: str = "Magic") -> dict:
    """Classify one DRC attempt into ``pass`` / ``fail`` / ``not_run``.

    Args:
        violation_count: the count the tool reported, or ``None``/negative when
            no count could be parsed. Negative is treated as UNMEASURED.
        report_path: the report the verdict claims to be derived from. Its
            existence and emptiness are checked here -- a verdict that names a
            report which was never written is the bug this module closes.
        tool_ran: did the tool actually complete (returncode 0 / no exception)?
        tool_error: tool diagnostics, folded into the ``reason`` when it failed.
        tool: tool name, for human-readable reasons ("Magic", "KLayout", ...).
    """
    count = violation_count
    if isinstance(count, bool):
        count = None
    if count is not None:
        try:
            count = int(count)
        except (TypeError, ValueError):
            count = None
    if count is not None and count < 0:
        count = None  # legacy -1 sentinel: absence, not a count

    rp = str(report_path or "")

    # Rule 1: a positive count is a measurement, however the tool exited.
    if count is not None and count > 0:
        where = f" in {rp}" if rp else ""
        return _verdict(STATUS_FAIL,
                        f"{tool} DRC measured {count} violation(s){where}",
                        count, rp, tool)

    # Rules 2/3: "clean" must be earned by positive evidence.
    if not tool_ran:
        detail = (tool_error or "").strip()
        why = (f"{tool} did not complete, so it produced no DRC measurement"
               + (f": {detail[:400]}" if detail else
                  " (no tool diagnostics were captured)"))
        return unmeasured_drc(why + _NOT_A_COUNT, report_path=rp, tool=tool)

    if not rp:
        return unmeasured_drc(
            f"no DRC report path was recorded for {tool}, so nothing could be "
            f"counted." + _NOT_A_COUNT, report_path=rp, tool=tool)

    p = Path(rp)
    if not p.is_file():
        return unmeasured_drc(
            f"{tool} left NO DRC report at {p} -- the report this verdict "
            f"would be derived from does not exist." + _NOT_A_COUNT,
            report_path=rp, tool=tool)

    try:
        text = p.read_text(errors="replace")
    except OSError as exc:
        return unmeasured_drc(
            f"the DRC report {p} could not be read ({exc})." + _NOT_A_COUNT,
            report_path=rp, tool=tool)

    if not text.strip():
        try:
            size = p.stat().st_size
        except OSError:
            size = 0
        return unmeasured_drc(
            f"the DRC report {p} is EMPTY ({size} bytes) -- {tool} wrote a "
            f"file but no result into it." + _NOT_A_COUNT,
            report_path=rp, tool=tool)

    if count is None:
        return unmeasured_drc(
            f"{tool} wrote {p} but it carries no parseable violation count, so "
            f"the layout was never actually judged." + _NOT_A_COUNT,
            report_path=rp, tool=tool)

    return _verdict(STATUS_PASS, f"{tool} DRC clean: 0 violations in {p}",
                    0, rp, tool)


def drc_summary(verdict: dict) -> str:
    """One line for a human / LLM / log reader.

    An unmeasured DRC says so IN WORDS, so no reader has to infer it from a
    bare ``-1`` (which reads as a violation) or a bare ``0`` (which reads as
    clean).
    """
    verdict = verdict or {}
    status = verdict.get("status") or STATUS_NOT_RUN
    tool = verdict.get("tool") or "DRC"
    if status == STATUS_PASS:
        return f"{tool} DRC: clean, 0 violations"
    if status == STATUS_FAIL:
        return f"{tool} DRC: {verdict.get('violation_count')} violation(s)"
    return (f"{tool} DRC COULD NOT BE MEASURED (status={STATUS_NOT_RUN}): "
            f"{verdict.get('reason') or 'no reason recorded'}")


def drc_stage(verdict: dict) -> dict:
    """Render a verdict as a ``gen_ram``-report ``stages[...]`` entry.

    ``ok`` is TRI-STATE and deliberately so:

    * ``True``  -- measured clean.
    * ``False`` -- measured N > 0 violations.
    * ``None``  -- NOT MEASURED. It is not ``False`` (that claims violations
      were found) and not ``True`` (that claims clean). ``if not ok`` still
      treats it as non-passing, so nothing downstream can read it as success,
      while ``ok is None`` tells a reader that checks the difference.

    ``violations`` is ``None`` when unmeasured -- never ``-1``.
    """
    verdict = verdict or {}
    status = verdict.get("status") or STATUS_NOT_RUN
    ok: bool | None
    if status == STATUS_PASS:
        ok = True
    elif status == STATUS_FAIL:
        ok = False
    else:
        ok = None
    return {
        "ok": ok,
        "status": status,
        "measured": bool(verdict.get("measured")),
        "violations": verdict.get("violation_count"),
        "reason": verdict.get("reason", ""),
        "report_path": verdict.get("report_path", ""),
        "detail": drc_summary(verdict),
    }
