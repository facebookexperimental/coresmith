# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Profile-based flag defaults (A-Fix 1).

A *profile* seeds environment defaults for the family of gate/feature flags the
engine already validated but ships default-off/unwired. The seeding is purely
``os.environ.setdefault`` -- an explicitly-set env var ALWAYS wins over a
profile default, and a profile default wins over a code default. Because the
profile only *seeds env*, every existing ``os.environ.get(...)`` reader (and
``monkeypatch.delenv`` in tests) keeps working unchanged.

Profiles:
  - ``strict`` (DEFAULT): turn the validated deterministic machinery ON
    (block goldens, PPA gate, fidelity gate, PDK characterization, latency
    audit, RTL-from-hardware-golden, seeded DV stimulus).
  - ``legacy``: seed nothing -- every flag falls back to its historical code
    default. The test suite pins this so its default-off assertions hold.

Select with ``CORESMITH_PROFILE=strict|legacy`` (default ``strict``).
"""

from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger(__name__)

# The flags the STRICT profile seeds to "1". Keys are the exact env var names
# the scattered ``*_enabled`` helpers already read, so no reader changes are
# needed beyond routing through the shared parser.
STRICT_DEFAULTS: dict[str, str] = {
    "CORESMITH_BLOCK_GOLDENS": "1",
    "CORESMITH_PPA_GATE": "1",
    "CORESMITH_FIDELITY_GATE": "1",
    "CORESMITH_PDK_CHAR": "1",
    "CORESMITH_LATENCY_AUDIT": "1",
    "CORESMITH_RTL_FROM_HW_GOLDEN": "1",
    # A-Fix 5a: seeded DV stimulus tier (defined here so the profile owns the
    # single source of truth; the model_integration gate reads the same flag).
    "CORESMITH_GATE_SEEDED_STIMULUS": "1",
}

LEGACY_DEFAULTS: dict[str, str] = {}

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}

_apply_lock = threading.Lock()
_applied = False
_seeded_keys: list[str] = []
# Profile keys that were ALREADY set in the environment when apply() ran (an
# explicit env var or a value seeded by an earlier process, e.g. the CLI before
# it spawned the daemon). Recorded so log_status() can report seeded-vs-preset
# even when this call seeded nothing.
_preexisting_keys: list[str] = []


def resolve_profile() -> str:
    """Return the active profile name: ``"strict"`` (default) or ``"legacy"``.

    Only an explicit ``CORESMITH_PROFILE=legacy`` selects legacy; anything else
    (unset, empty, ``strict``, or an unrecognized value) resolves to strict --
    fail-safe toward the validated machinery being ON.
    """
    val = (os.environ.get("CORESMITH_PROFILE", "") or "").strip().lower()
    if val == "legacy":
        return "legacy"
    if val and val != "strict":
        logger.warning("Unknown CORESMITH_PROFILE=%r; treating as 'strict'.", val)
    return "strict"


def _defaults_for(profile: str) -> dict[str, str]:
    return LEGACY_DEFAULTS if profile == "legacy" else STRICT_DEFAULTS


def status_line() -> str:
    """One-line human summary of the last :func:`apply` (profile + seed state).

    Reports the active profile, which flags this process seeded, and which were
    already set before it ran -- so the line is informative even when nothing
    was seeded (the common daemon case, since the CLI seeds them first and the
    daemon inherits the env).
    """
    profile = resolve_profile()
    seeded = ", ".join(_seeded_keys) if _seeded_keys else "(none)"
    already = ", ".join(_preexisting_keys) if _preexisting_keys else "(none)"
    return f"coresmith profile={profile} seeded=[{seeded}] already_set=[{already}]"


def log_status(target_logger: logging.Logger | None = None) -> None:
    """Emit the :func:`status_line` at INFO. Safe to call repeatedly.

    ``apply()`` runs at import time -- often BEFORE the daemon configures
    logging -- so that first line can go nowhere. Callers re-invoke this after
    logging is up (e.g. the daemon startup hook) to guarantee the profile-seed
    line is observable. Pass ``target_logger`` to route it through a logger that
    already has a handler (uvicorn's, so it lands in daemon.log).
    """
    (target_logger or logger).info("%s", status_line())


def apply(*, force: bool = False) -> list[str]:
    """Seed the active profile's env defaults (idempotent).

    ``os.environ.setdefault`` semantics: explicit env > profile default > code
    default, mechanically guaranteed. Returns the list of keys this call
    actually seeded (empty on legacy or when everything was already set). ALWAYS
    logs one INFO line via :func:`log_status` (profile name + seeded keys + keys
    already set), even when it seeds nothing.
    """
    global _applied
    with _apply_lock:
        if _applied and not force:
            return list(_seeded_keys)
        profile = resolve_profile()
        seeded: list[str] = []
        already: list[str] = []
        for key, value in _defaults_for(profile).items():
            if key not in os.environ:
                os.environ[key] = value
                seeded.append(key)
            else:
                already.append(key)
        _seeded_keys[:] = seeded
        _preexisting_keys[:] = already
        _applied = True
    log_status()
    return seeded


def ensure_applied() -> None:
    """Belt-and-suspenders: apply the profile once if it hasn't been yet.

    Called at the top of each gate-enable helper so a flag read is never seen
    before the profile has had a chance to seed its defaults, regardless of
    entrypoint.
    """
    if not _applied:
        apply()


def reset() -> None:
    """Undo the keys this process seeded and allow re-``apply()`` (test helper).

    Removes ONLY the keys the last ``apply()`` seeded (never an explicitly-set
    var) so a test can switch profiles cleanly.
    """
    global _applied
    with _apply_lock:
        for key in _seeded_keys:
            os.environ.pop(key, None)
        _seeded_keys.clear()
        _preexisting_keys.clear()
        _applied = False


def flag_enabled(name: str, default: bool = False) -> bool:
    """Canonical truthy parser for a ``CORESMITH_*`` flag.

    Unset -> ``default``. Set to a recognized truthy/falsy token -> that value.
    Unrecognized non-empty value -> ``default``. Empty string -> False (an
    explicit empty value historically meant "off" for the default-on gates).
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    v = raw.strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    return default


def flag_disabled(name: str, default: bool = False) -> bool:
    """Inverse of :func:`flag_enabled` (``True`` == the flag is OFF)."""
    return not flag_enabled(name, default=default)
