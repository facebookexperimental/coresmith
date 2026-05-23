"""Single source of truth for coresmith's subprocess timeouts.

Every long-running LLM / EDA tool call goes through one of these helpers so
that a single environment variable -- CORESMITH_TIMEOUT_MULTIPLIER -- can scale
every timeout in the pipeline at once.  This is the easiest knob to turn for
slow local models (e.g. Qwen 3.6 27B at ~25 tok/s on a single RTX PRO 6000,
which can need 22+ minutes for a single UARCH spec versus Claude's ~3 minutes).

Usage::

    from orchestrator._timeouts import scaled

    proc = subprocess.run(cmd, timeout=scaled(1200))
    # or with an existing per-site env override:
    proc = subprocess.run(cmd, timeout=scaled(1800, env="CORESMITH_TB_TIMEOUT"))

Setting CORESMITH_TIMEOUT_MULTIPLIER=3 triples every timeout for the run.
The per-site env vars still win for the *base* value, the multiplier is
applied on top.
"""
from __future__ import annotations

import os


def multiplier() -> float:
    """Read CORESMITH_TIMEOUT_MULTIPLIER fresh each call so live env changes
    take effect on the next subprocess invocation (the multiplier isn't
    cached at module-import time)."""
    try:
        return float(os.environ.get("CORESMITH_TIMEOUT_MULTIPLIER", "1.0"))
    except ValueError:
        return 1.0


def scaled(base_seconds: float, env: str | None = None) -> int:
    """Return base_seconds (or env override) multiplied by the timeout
    multiplier.  Always returns an int because most subprocess APIs prefer it."""
    if env:
        try:
            base_seconds = float(os.environ.get(env, str(base_seconds)))
        except ValueError:
            pass
    return int(base_seconds * multiplier())
