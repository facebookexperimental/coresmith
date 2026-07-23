# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Host-flow stimulus + oracle plan for the deterministic QSPI-slave DV.

The deterministic BFM drives the DUT; the *expected* output comes from the
LLM-supplied golden **reference** (the pure ``run(stimulus)`` software model in
the run's ``inputs/*golden*.py`` / FRD reference entry) -- NOT from any
pin-driving harness. This module turns a run's acceptance stimulus + reference
into a :class:`StimulusPlan` (concrete register writes + expected OUT bytes)
following the QSPI-slave *host flow*:

    write CFG*, write IN window, CTRL.START, poll STATUS.DONE, read OUT window,
    compare to ``reference(stimulus)`` -- the oracle-not-golden principle.

The stimulus->pins mapping is the documented QSPI-slave convention (PROTOCOL.md
"Per-design mapping"): a stimulus ``dict`` contributes its ``key`` bytes (if
any) followed by its ``data``/``plaintext``/``payload`` bytes to the IN window,
and a block/length scalar to CFG0. This covers the aes exemplar and the
data-in/result-out designs (fft/raster). Anything that does not fit returns
``None`` so the caller falls back to the LLM BFM with a loud advisory.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .qspi_contract import QSPIContract


@dataclass
class StimulusPlan:
    """A concrete, deterministic host-flow drive plan + oracle."""

    cfg: list[tuple[int, int, int]] = field(default_factory=list)   # (addr, value, width_bytes)
    writes: list[tuple[int, bytes]] = field(default_factory=list)   # (addr, data)
    out_addr: int = 0x002000
    out_len: int = 0
    expected: bytes = b""
    case_name: str = ""

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "cfg": [[a, v, w] for (a, v, w) in self.cfg],
            "writes": [[a, d.hex()] for (a, d) in self.writes],
            "out_addr": self.out_addr,
            "out_len": self.out_len,
            "expected": self.expected.hex(),
            "case_name": self.case_name,
        }


def _flatten_bytes(value: Any) -> bytes:
    if value is None:
        return b""
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, int):
        return bytes([value & 0xFF])
    try:
        return bytes(int(v) & 0xFF for v in value)
    except (TypeError, ValueError):
        return b""


def _import_from_path(path: Path, mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _load_reference(project_root: str) -> Optional[Callable]:
    """Find the pure golden reference ``run(stimulus)`` for this run.

    Prefers ``inputs/*golden*.py`` with a module-level ``run`` (the FRD
    reference-entry convention). Returns the callable or None.
    """
    root = Path(project_root)
    candidates: list[Path] = []
    inp = root / "inputs"
    if inp.is_dir():
        candidates += sorted(inp.glob("*golden*.py"))
        candidates += sorted(inp.glob("*reference*.py"))
    for p in candidates:
        try:
            mod = _import_from_path(p, f"_cs_ref_{p.stem}")
        except Exception:  # noqa: BLE001
            continue
        for entry in ("run", "reference", "golden", "encrypt_stream"):
            fn = getattr(mod, entry, None)
            if callable(fn):
                return fn
    return None


def _load_acceptance_case(project_root: str) -> Optional[tuple[str, Any]]:
    """Load the FIRST acceptance case ``(name, stimulus)`` for this run.

    The deterministic DV uses the first acceptance case (the KAT) as its
    canonical vector -- deterministic and representative.
    """
    p = Path(project_root) / "inputs" / "acceptance_stimulus.py"
    if not p.exists():
        p2 = Path(project_root) / "inputs" / "model_stimulus.py"
        if not p2.exists():
            return None
        try:
            mod = _import_from_path(p2, "_cs_model_stim")
        except Exception:  # noqa: BLE001
            return None
        stim = getattr(mod, "stimulus", None)
        return ("model_stimulus", stim) if stim is not None else None
    try:
        mod = _import_from_path(p, "_cs_accept_stim")
    except Exception:  # noqa: BLE001
        return None
    cases = getattr(mod, "cases", None)
    if cases:
        name, stim = cases[0]
        return (str(name), stim)
    stim = getattr(mod, "stimulus", None)
    return ("stimulus", stim) if stim is not None else None


def build_plan_from_run(
    project_root: str, contract: QSPIContract
) -> Optional[StimulusPlan]:
    """Build a deterministic host-flow plan + oracle from the run artifacts.

    Returns None when a contract-faithful, self-contained plan cannot be
    derived (no acceptance stimulus, no pure reference, or a stimulus shape the
    host-flow convention does not cover) -- the caller then falls back to the
    LLM BFM with a loud advisory.
    """
    case = _load_acceptance_case(project_root)
    ref = _load_reference(project_root)
    if case is None or ref is None:
        return None
    case_name, stim = case

    # Map stimulus -> IN-window bytes + CFG0 scalar (host-flow convention).
    key_bytes = b""
    data_bytes = b""
    cfg0_val: Optional[int] = None
    if isinstance(stim, dict):
        key_bytes = _flatten_bytes(stim.get("key"))
        data_bytes = _flatten_bytes(
            stim.get("data", stim.get("plaintext", stim.get("payload")))
        )
        for k in ("n_blocks", "length", "count", "mode", "cfg0"):
            if k in stim:
                try:
                    cfg0_val = int(stim[k])
                    break
                except (TypeError, ValueError):
                    pass
    else:
        data_bytes = _flatten_bytes(stim)

    in_bytes = key_bytes + data_bytes
    if not in_bytes:
        return None

    # Reference oracle (pure). Must return bytes-like.
    try:
        expected = bytes(_flatten_bytes(ref(stim)))
    except Exception:  # noqa: BLE001
        return None
    if not expected:
        return None

    # CFG0 default: number of 16-byte blocks in the data window (aes/stream
    # convention) when the design did not name a scalar explicitly.
    if cfg0_val is None and data_bytes:
        cfg0_val = max(1, (len(data_bytes) + 15) // 16)

    plan = StimulusPlan(
        out_addr=contract.out_addr,
        out_len=len(expected),
        expected=expected,
        case_name=case_name,
    )
    if cfg0_val is not None:
        plan.cfg.append((contract.cfg0_addr, cfg0_val, 4))
    plan.writes.append((contract.in_addr, in_bytes))
    return plan


def load_plan_json(path: str) -> StimulusPlan:
    d = json.loads(Path(path).read_text())
    return StimulusPlan(
        cfg=[tuple(x) for x in d.get("cfg", [])],
        writes=[(a, bytes.fromhex(h)) for (a, h) in d.get("writes", [])],
        out_addr=d.get("out_addr", 0x002000),
        out_len=d.get("out_len", 0),
        expected=bytes.fromhex(d.get("expected", "")),
        case_name=d.get("case_name", ""),
    )
