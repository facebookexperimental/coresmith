# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""PDK Characterization stage -- the single, shared, PDK-dependent characterization
that the architecture (PRD-time) runs ONCE and every downstream block consumes.

It folds the two halves together:
  * **arithmetic op-delay model** (``arith_characterize``): how much arithmetic
    fits in a clock period -> the input the µArch pipeline scheduler needs.
  * **memory PPA model** (``mem_characterize``): flop vs registered-flop vs SRAM
    macro area/Fmax per geometry -> the input the µArch memory-impl decision needs.

Both are PDK-dependent and shared by many blocks, so they are characterized once
at the PRD stage (not per block) and cached by PDK fingerprint; the stage
short-circuits to the cache on every run after the first.

Consumed at the µArch stage: ``predict_op_delay`` / ``ops_per_stage`` (scheduler)
and ``predict_mem`` (memory choice). This module is the one place that guarantees
both caches are warm before any block is specified.

Gated by ``CORESMITH_PDK_CHAR`` (default off until validated end-to-end); fails
OPEN (a characterization error never blocks the architecture run -- the downstream
gates remain the backstop).
"""
from __future__ import annotations

from typing import Any

from . import arith_characterize as _arith
from . import mem_characterize as _mem

# Re-export the consumer-facing predictors so the µArch agent imports one module.
from .arith_characterize import ops_per_stage, predict_op_delay
from .mem_characterize import predict_mem


def stage_enabled() -> bool:
    """Opt-in until validated end-to-end (default OFF; strict profile seeds it)."""
    from orchestrator.profile import ensure_applied, flag_enabled
    ensure_applied()
    return flag_enabled("CORESMITH_PDK_CHAR", default=False)


# A small default memory grid to warm the mem cache at the stage (the geometries
# a typical streaming datapath reaches). predict_mem still characterizes any
# unseen geometry lazily, so this is just a warm-start, not a hard list.
_DEFAULT_MEM_GRID = [
    ("1rw", 8, 256), ("1rw", 8, 1024), ("1rw1r", 8, 512), ("1rw1r", 8, 1024),
    ("1rw", 32, 256), ("1rw1r", 32, 512), ("1rw", 16, 64),
]


def ensure_pdk_characterized(pdk: dict | None = None, force: bool = False,
                             warm_memory: bool = True) -> dict[str, Any]:
    """Run (or load from cache) BOTH the arithmetic + memory PDK characterizations.

    Returns a summary dict. Fails open: any tool error is captured, not raised.
    """
    summary: dict[str, Any] = {"pdk_hash": None, "arith": None, "memory": None,
                               "errors": []}
    # --- arithmetic op-delay model (always; it's a fixed bounded sweep) ---
    try:
        doc = _arith.ensure_characterized(pdk=pdk, force=force)
        summary["pdk_hash"] = doc.get("pdk_hash")
        summary["arith"] = {
            "ops": sorted((doc.get("model") or {}).keys()),
            "cache": str(_arith._cache_path(pdk)),
            "delay_form": doc.get("delay_form"),
        }
    except Exception as exc:  # noqa: BLE001
        summary["errors"].append(f"arith: {exc}")

    # --- memory PPA model (load cache; warm a small grid if cold) ---
    try:
        table = _mem.load_table(pdk)
        if (force or not table) and warm_memory:
            table = _mem.characterize_memories(grid=_DEFAULT_MEM_GRID)
        summary["memory"] = {"rows": len(table or []),
                             "cache_dir": str(_mem.CACHE_DIR)}
        if summary["pdk_hash"] is None:
            summary["pdk_hash"] = _mem.pdk_hash(pdk)
    except Exception as exc:  # noqa: BLE001
        summary["errors"].append(f"memory: {exc}")

    return summary


def is_characterized(pdk: dict | None = None) -> bool:
    """True if both models are present + non-degenerate in cache."""
    arith_ok = _arith.is_characterized(pdk)
    try:
        mem_ok = bool(_mem.load_table(pdk))
    except Exception:  # noqa: BLE001
        mem_ok = False
    return arith_ok and mem_ok


__all__ = [
    "ensure_pdk_characterized",
    "is_characterized",
    "ops_per_stage",
    "predict_mem",
    "predict_op_delay",
    "stage_enabled",
]


if __name__ == "__main__":  # pragma: no cover
    from orchestrator.profile import apply as _apply_profile
    _apply_profile()
    import json
    import sys
    force = "--force" in sys.argv
    print(json.dumps(ensure_pdk_characterized(force=force), indent=2))
