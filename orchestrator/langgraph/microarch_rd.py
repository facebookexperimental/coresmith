# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Golden-reference resolution for the microarchitecture stage.

Resolves the run's reference-implementation ("golden") Python model from a run
directory, so downstream stages (architecture graph, harness CLI, trust store)
can locate the reference model without knowing the on-disk layout. Design-
specific quality goal-gates (e.g. a rate-distortion harness for a codec design)
are kept out of the generic engine and live in downstream design collateral.
"""
from __future__ import annotations

from pathlib import Path


def resolve_golden_path(run_dir: str) -> str | None:
    """Return the absolute path to the run's reference-implementation model.

    Prefers ``<run_dir>/inputs/golden.py``; otherwise falls back to the
    architecture composition resolver. Returns ``None`` if none is found.
    """
    cand = Path(run_dir) / "inputs" / "golden.py"
    if cand.exists():
        return str(cand.resolve())
    try:
        from orchestrator.architecture.composition import (
            resolve_reference_implementation,
        )
        p = resolve_reference_implementation(run_dir)
        if p and Path(p).exists():
            return str(Path(p).resolve())
    except Exception:  # noqa: BLE001
        pass
    return None
