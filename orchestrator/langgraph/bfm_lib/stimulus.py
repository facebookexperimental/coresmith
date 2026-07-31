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
and a block/length scalar to CFG0. This covers the block-cipher exemplar and the
data-in/result-out designs. Anything that does not fit returns ``None`` so the
caller falls back to the LLM BFM with a loud advisory.

TWO cases, not one. The canonical (first) acceptance case is the KAT: small,
deterministic, representative -- and therefore NOT a max-geometry test. A run
that ships only that case can pass every fixed-small-geometry check and still
carry an index/length counter that wraps below the declared maximum. So this
module also selects, GENERICALLY, the case that drives the largest geometry
(:func:`select_max_geometry_case`, ranked by IN-payload bytes then the CFG0
scalar) and bakes its golden the same generation-time way. No case-name
literals, no design-specific keys: the ranking is over magnitudes the host flow
actually drives on the pins.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    """Import a run-input module by path, with its OWN directory importable.

    The acceptance stimulus routinely imports its sibling golden by bare module
    name (``from <design>_golden import make_stimulus``). Loading it by path
    alone leaves that sibling unresolvable, so whether the plan builds at all
    depended on whether some EARLIER engine step happened to have put
    ``inputs/`` on ``sys.path`` first -- an import-order coincidence, and a
    silent fallback to the LLM BFM when it did not hold. The parent directory is
    inserted for the duration of the exec and removed after, so nothing leaks
    into the process's import path.
    """
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    parent = str(path.resolve().parent)
    injected = parent not in sys.path
    if injected:
        sys.path.insert(0, parent)
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    finally:
        if injected:
            try:
                sys.path.remove(parent)
            except ValueError:  # pragma: no cover - another thread removed it
                pass
    return mod


def _load_reference(project_root: str) -> Callable | None:
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


def load_acceptance_cases(project_root: str) -> list[tuple[str, Any]]:
    """EVERY acceptance case ``(name, stimulus)`` for this run, in declared order.

    ``[]`` when the run declares no acceptance stimulus at all. A single-stimulus
    module (``stimulus = ...`` rather than ``cases = [...]``) yields one entry, so
    callers never special-case the shape.
    """
    p = Path(project_root) / "inputs" / "acceptance_stimulus.py"
    if not p.exists():
        p2 = Path(project_root) / "inputs" / "model_stimulus.py"
        if not p2.exists():
            return []
        try:
            mod = _import_from_path(p2, "_cs_model_stim")
        except Exception:  # noqa: BLE001
            return []
        stim = getattr(mod, "stimulus", None)
        return [("model_stimulus", stim)] if stim is not None else []
    try:
        mod = _import_from_path(p, "_cs_accept_stim")
    except Exception:  # noqa: BLE001
        return []
    cases = getattr(mod, "cases", None)
    out: list[tuple[str, Any]] = []
    if cases:
        for entry in cases:
            try:
                name, stim = entry
            except (TypeError, ValueError):
                continue
            out.append((str(name), stim))
        return out
    stim = getattr(mod, "stimulus", None)
    return [("stimulus", stim)] if stim is not None else []


def _load_acceptance_case(project_root: str) -> tuple[str, Any] | None:
    """Load the FIRST acceptance case ``(name, stimulus)`` for this run.

    The deterministic DV uses the first acceptance case (the KAT) as its
    canonical vector -- deterministic and representative.
    """
    cases = load_acceptance_cases(project_root)
    return cases[0] if cases else None


def map_stimulus(stim: Any) -> tuple[bytes, int | None]:
    """``(IN-window bytes, CFG0 scalar)`` for one stimulus -- the host-flow
    convention, in ONE place so case selection and plan construction rank and
    drive the same magnitudes.

    A stimulus ``dict`` contributes ``key`` bytes then ``data``/``plaintext``/
    ``payload`` bytes; the CFG0 scalar comes from the first of ``n_blocks``,
    ``length``, ``count``, ``mode``, ``cfg0`` that is present, else defaults to
    the number of 16-byte blocks in the data window. A non-dict stimulus is
    flattened whole into the IN window.
    """
    key_bytes = b""
    data_bytes = b""
    cfg0_val: int | None = None
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

    if cfg0_val is None and data_bytes:
        cfg0_val = max(1, (len(data_bytes) + 15) // 16)
    return key_bytes + data_bytes, cfg0_val


def stimulus_scalars(stim: Any) -> dict[str, int]:
    """The integer-valued entries of a stimulus dict (``{}`` for other shapes).

    These are the design's OWN parameter names carrying the values this case
    drives -- the evidence a functional max-geometry marker is allowed to rest
    on. Names are opaque data: nothing here knows any design's vocabulary.
    """
    if not isinstance(stim, dict):
        return {}
    out: dict[str, int] = {}
    for k, v in stim.items():
        if isinstance(v, bool):
            continue
        if isinstance(v, int):
            out[str(k)] = v
        elif isinstance(v, float) and float(v).is_integer():
            out[str(k)] = int(v)
    return out


def select_max_geometry_case(
    cases: list[tuple[str, Any]]
) -> tuple[str, Any] | None:
    """The acceptance case that drives the LARGEST geometry, chosen generically.

    Ranked by ``(len(IN payload bytes), CFG0 scalar)`` -- the two magnitudes the
    QSPI host flow actually puts on the pins, so the winner is the case whose
    length/index counters run furthest. Ties keep the earlier-declared case, so
    selection is deterministic for a given acceptance module. ``None`` for an
    empty list.
    """
    best: tuple[str, Any] | None = None
    best_key: tuple[int, int] | None = None
    for name, stim in cases or []:
        try:
            in_bytes, cfg0 = map_stimulus(stim)
        except Exception:  # noqa: BLE001 - a malformed case is skipped, not fatal
            continue
        key = (len(in_bytes), int(cfg0 or 0))
        if best_key is None or key > best_key:
            best_key, best = key, (str(name), stim)
    return best


def _plan_from_case(
    case_name: str, stim: Any, ref: Callable, contract: QSPIContract
) -> StimulusPlan | None:
    """One acceptance case -> a concrete host-flow plan with a BAKED oracle.

    The golden reference is evaluated HERE, at generation time, exactly once per
    case -- the emitted testbench carries bytes, never an import of the model.
    """
    in_bytes, cfg0_val = map_stimulus(stim)
    if not in_bytes:
        return None
    try:
        expected = bytes(_flatten_bytes(ref(stim)))
    except Exception:  # noqa: BLE001
        return None
    if not expected:
        return None
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


@dataclass
class MaxGeoCase:
    """The largest acceptance case, with its baked plan and driven scalars.

    ``is_primary`` is True when the largest case IS the canonical first case --
    the generator then emits no second test (there is nothing extra to drive)
    but still advertises the ``# MAXGEO_CASE`` marker, because the primary test
    genuinely IS the max-geometry case and the artifact should say so.
    """

    plan: StimulusPlan
    scalars: dict[str, int] = field(default_factory=dict)
    in_bytes: int = 0
    out_bytes: int = 0
    cfg0: int = 0
    is_primary: bool = False


def build_max_geometry_case(
    project_root: str, contract: QSPIContract, primary_case_name: str = ""
) -> MaxGeoCase | None:
    """Select + bake the maximum-configuration acceptance case for this run.

    ``None`` when the run declares no acceptance cases, no pure reference, or the
    winning case's stimulus shape the host-flow convention does not cover -- the
    generator then emits exactly what it emitted before this existed.
    """
    cases = load_acceptance_cases(project_root)
    if not cases:
        return None
    ref = _load_reference(project_root)
    if ref is None:
        return None
    chosen = select_max_geometry_case(cases)
    if chosen is None:
        return None
    case_name, stim = chosen
    plan = _plan_from_case(case_name, stim, ref, contract)
    if plan is None:
        return None
    in_bytes = sum(len(d) for (_a, d) in plan.writes)
    cfg0 = plan.cfg[0][1] if plan.cfg else 0
    return MaxGeoCase(
        plan=plan,
        scalars=stimulus_scalars(stim),
        in_bytes=in_bytes,
        out_bytes=plan.out_len,
        cfg0=int(cfg0),
        is_primary=bool(primary_case_name) and case_name == primary_case_name,
    )


def build_plan_from_run(
    project_root: str, contract: QSPIContract
) -> StimulusPlan | None:
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
    return _plan_from_case(case_name, stim, ref, contract)


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
