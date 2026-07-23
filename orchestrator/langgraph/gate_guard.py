# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Fail-closed gate guard (A-Fix 2).

A load-bearing gate that RAISES must never be treated as a pass. Historically
several gate call sites wrapped the gate in ``try/except`` and returned
``passed=True`` on any error -- a fail-OPEN default that silently shipped a
block whose gate could not run (harness break, missing tool, a NameError in
the gate itself).

``gate_guard(name, fn, ...)`` runs a gate function fail-CLOSED:

- ``fn`` returns normally  -> ``GateResult(passed=True, value=<return>)`` (or
  ``passed=classify(value)`` when a ``classify`` callback is given). The raw
  return is preserved in ``.value`` so a caller that inspects it (e.g. a
  *violations list* whose emptiness means "pass") can.
- ``fn`` raises            -> ``GateResult(passed=False, skipped=False,
  error=<traceback tail>)``. An ERROR is **not** an honest skip.
- ``CORESMITH_GATE_FAIL_OPEN=1`` -> the single global rollback knob. Under it a
  raised gate is tolerated (``passed=True, skipped=True``) so a block still
  advances -- for a slow/odd host where fail-closed reddens harness errors.

Fail-closed is the default in BOTH profiles (strict and legacy): this is a bug
fix, not a policy choice, so no profile seeds ``CORESMITH_GATE_FAIL_OPEN``.

This module is intentionally dependency-light (only ``orchestrator.profile``)
so it imports without pulling in the LangGraph stack.
"""

from __future__ import annotations

import logging
import os
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

_LOG = logging.getLogger("coresmith.gate_guard")


def gate_fail_open_enabled() -> bool:
    """True when ``CORESMITH_GATE_FAIL_OPEN`` is set truthy.

    The single global rollback knob that reverts every fail-closed gate to the
    old fail-open behavior. Reads env directly (no profile seeds it).
    """
    try:
        from orchestrator.profile import flag_enabled
        return flag_enabled("CORESMITH_GATE_FAIL_OPEN", default=False)
    except Exception:  # noqa: BLE001 - a profile import hiccup must not break a gate
        raw = (os.environ.get("CORESMITH_GATE_FAIL_OPEN") or "").strip().lower()
        return raw in ("1", "true", "yes", "on")


@dataclass
class GateResult:
    """Outcome of running a gate through :func:`gate_guard`."""

    gate: str
    passed: bool
    skipped: bool = False
    reason: str = ""
    error: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    value: Any = None

    @property
    def errored(self) -> bool:
        """True iff the wrapped gate raised (a traceback tail was recorded)."""
        return bool(self.error)


def _tb_tail(exc: BaseException, limit: int = 2000) -> str:
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return tb[-limit:]


def gate_guard(
    name: str,
    fn: Callable[..., Any],
    *args: Any,
    classify: Optional[Callable[[Any], bool]] = None,
    **kwargs: Any,
) -> GateResult:
    """Run ``fn(*args, **kwargs)`` fail-closed and return a :class:`GateResult`.

    See the module docstring for the full contract.
    """
    try:
        value = fn(*args, **kwargs)
    except BaseException as exc:  # noqa: BLE001 - catching it is the whole point
        tail = _tb_tail(exc)
        reason = f"{type(exc).__name__}: {exc}"
        if gate_fail_open_enabled():
            _LOG.error(
                "gate %r raised, but CORESMITH_GATE_FAIL_OPEN is set -- "
                "tolerating (fail-open): %s", name, reason,
            )
            return GateResult(
                gate=name, passed=True, skipped=True, reason=reason, error=tail,
            )
        _LOG.error("gate %r raised (fail-closed, NOT a pass): %s", name, reason)
        return GateResult(
            gate=name, passed=False, skipped=False, reason=reason, error=tail,
        )
    passed = True if classify is None else bool(classify(value))
    return GateResult(gate=name, passed=passed, skipped=False, value=value)


def gate_error_violation(
    reason: str,
    error: str = "",
    *,
    gap_class: str = "block_math",
    criterion: str = "gate_error",
) -> dict:
    """Synthesize a single ``violation`` dict for a gate that ERRORED.

    A fail-closed gate whose invocation raised falls through to its node's
    normal violation/interrupt handling by injecting this one violation --
    rather than silently returning ``passed=True``. The shape matches what the
    model-integration / uarch-gate routers already consume.
    """
    return {
        "criterion": criterion,
        "gap_class": gap_class,
        "expected": "the deterministic gate to run to completion",
        "observed": f"gate raised: {reason}",
        "first_divergence_block": "",
        "affected_blocks": [],
        "suggested_fix": (
            "This is NOT a pass -- the gate itself errored (harness/environment). "
            "Fix the gate environment, then resume 'retry'."
            + (f"\nTraceback tail:\n{error}" if error else "")
        ),
    }
