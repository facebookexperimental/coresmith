# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Model-integration gate -- deterministic, LLM-free verification (v2).

This is the v2 replacement for the v1 ``composition`` gate. The methodology:

1. Per-block **Amaranth block models** (real clock / valid-ready / latency) are
   emitted by ``BlockGoldenGenerator`` into ``arch/block_models/<block>.py``.
2. An LLM **model-integration agent** wires those block models into a top-level
   Amaranth chip model and writes ``arch/block_models/_chip_model.py``, which
   exposes a module-level ``simulate(stimulus) -> observed`` that drives a
   stimulus through the chip model in Amaranth pysim and returns the
   chip-level output in the SAME shape the reference implementation returns.
3. THIS module is the deterministic gate: it imports ``_chip_model.simulate``,
   obtains a stimulus, runs the reference implementation to get the expected
   output, runs ``simulate`` to get the observed output, and compares them
   BIT-EXACT. On mismatch it returns one ``model_integration_failure`` violation.

Everything here is pure + deterministic and unit-testable with a real (fast)
Amaranth pysim but without an LLM or any EDA tooling.

Gated by ``CORESMITH_BLOCK_GOLDENS``: when off (the default),
:func:`run_model_integration_gate` is a no-op that returns ``[]``.
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
import os
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Reuse the v1 helpers that are still correct under v2.
from orchestrator.architecture.composition import (
    BLOCK_MODELS_DIRNAME,
    ReferenceInvocationError,
    _normalize_ref_output,
    _outputs_match,
    _run_reference,
    bit_exact_enabled,
    block_goldens_enabled,
    gate_allow_nondegenerate_enabled,
    gate_epsilon,
    output_has_float,
    outputs_close,
    parse_func_vectors,
    resolve_functional_acceptance,
    resolve_reference_entrypoint,
    resolve_reference_implementation,
    resolve_throughput_floor,
)
from orchestrator.architecture.fidelity import (
    compute_fidelity_derate,
    fidelity_gate_enabled,
    resolve_fidelity_metric,
    write_derate_ledger,
)

logger = logging.getLogger(__name__)

CHIP_MODEL_FILENAME = "_chip_model.py"


def _import_module_from_path(path: Path, mod_name: str):
    """Import a module from a file path under a private module name.

    Audit F2: the module's OWN directory goes on ``sys.path`` for the duration
    of the import. Project reference implementations and stimulus files live in
    ``inputs/`` and import sibling helpers by plain name (e.g. the reference_codec
    golden's ``import reference_codec_vectors``) -- without the parent dir on the path
    that import raises ``No module named ...`` and the gate used to swallow it
    as a no-op.
    """
    import sys

    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    parent = str(Path(path).resolve().parent)
    inserted = False
    if parent not in sys.path:
        sys.path.insert(0, parent)
        inserted = True
    try:
        spec.loader.exec_module(module)
    finally:
        if inserted:
            try:
                sys.path.remove(parent)
            except ValueError:
                pass
    return module


def _load_reference_module(path: str):
    """Import the reference implementation module from a file path."""
    return _import_module_from_path(Path(path), "_coresmith_reference_impl")


def _default_stimulus(entry_callable: Callable | None) -> Any:
    """Derive a small default stimulus from the reference entry signature.

    Policy (documented): a single short ascending list ``[1, 2, 3, 4]`` is
    produced and bound to the entry's FIRST positional parameter (the common
    "stream of samples / bytes / words" case). This is intentionally minimal:
    designs whose primary input is not a 1-D integer stream MUST supply an
    explicit stimulus via ``CORESMITH_MODEL_STIMULUS`` (a python file exposing
    a module-level ``stimulus``). The default exists only so a simple stream
    design (the common toy / objective-math case) can self-test without
    operator input. Returns ``None`` when no sensible default can be derived
    (the gate then logs + no-ops).
    """
    if entry_callable is None:
        return None
    try:
        sig = inspect.signature(entry_callable)
        positional = [
            p
            for p in sig.parameters.values()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
    except (TypeError, ValueError):
        positional = []
    # A single short ascending stream is the most broadly-valid default.
    default_stream = [1, 2, 3, 4]
    if not positional:
        # Zero-arg or *args-only: pass the bare list (simulate() decides).
        return default_stream
    if len(positional) == 1:
        return default_stream
    # Multiple parameters: bind the stream by name to the first param so the
    # reference and simulate() both receive a dict they can map.
    return {positional[0].name: default_stream}


def _stimulus_from_file(p: Path) -> tuple[Any, bool]:
    """Import ``p`` and return ``(p.stimulus, True)`` or ``(None, False)``."""
    if not p.is_file():
        return None, False
    try:
        mod = _import_module_from_path(p, "_coresmith_model_stimulus")
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "model integration gate: stimulus file %s failed to import: %s",
            p, exc,
        )
        return None, False
    stim = getattr(mod, "stimulus", None)
    if stim is None:
        logger.warning(
            "model integration gate: stimulus file %s has no `stimulus`", p
        )
        return None, False
    return stim, True


def _load_env_stimulus(project_root: str | None = None) -> tuple[Any, bool]:
    """Resolve the model-integration stimulus.

    Order: (1) ``CORESMITH_MODEL_STIMULUS`` env file, then (2) auto-discover
    ``<project_root>/inputs/model_stimulus.py`` (PR#12 finding #5). Returns
    ``(stimulus, found)``.

    Finding #5: when the env var was unset the gate fell straight through to a
    small DERIVED default (a bare ``[1,2,3,4]`` list) that a real golden's
    ``run()`` rejects -- so byte-exact blocks were marked failed for a pure
    setup reason. A run typically ships a project ``inputs/model_stimulus.py``
    exposing a valid ``stimulus``; auto-discovering it removes the footgun
    without requiring the operator to export the env var.
    """
    path = os.environ.get("CORESMITH_MODEL_STIMULUS", "").strip()
    if path:
        p = Path(path)
        if not p.is_file():
            logger.warning(
                "model integration gate: CORESMITH_MODEL_STIMULUS=%r not a file",
                path,
            )
        else:
            stim, ok = _stimulus_from_file(p)
            if ok:
                return stim, True
    # Auto-discover the project stimulus file.
    if project_root:
        cand = Path(project_root) / "inputs" / "model_stimulus.py"
        stim, ok = _stimulus_from_file(cand)
        if ok:
            logger.info(
                "model integration gate: using auto-discovered stimulus %s", cand
            )
            return stim, True
    return None, False


# ---------------------------------------------------------------------------
# A-Fix 5(a): two-tier gate stimulus (FIXED + anti-memorization SEEDED)
# ---------------------------------------------------------------------------

def _gate_seeded_stimulus_enabled() -> bool:
    """True -> also drive the gate with a FRESH SEEDED stimulus the chip model
    could not have memorized at generation time. Seeded ON in the STRICT
    profile (seeded by profile.STRICT_DEFAULTS); OFF in legacy."""
    try:
        from orchestrator.profile import ensure_applied, flag_enabled
        ensure_applied()
        return flag_enabled("CORESMITH_GATE_SEEDED_STIMULUS", default=False)
    except Exception:  # noqa: BLE001
        raw = (os.environ.get("CORESMITH_GATE_SEEDED_STIMULUS") or "").strip().lower()
        return raw in {"1", "true", "yes", "on"}


def _resolve_gate_seed() -> int:
    """The SEEDED-tier seed: a pinned value (``CORESMITH_DV_SEED_PIN``, for
    reproducible debugging) else a fresh cryptographic 63-bit seed. Delegates to
    the unified seed provider (single source of truth for the dev/gate split)."""
    from orchestrator.harness.seed_provider import gate_seed
    return gate_seed()


def _seeded_stimulus(entry_callable: Callable | None, seed: int) -> Any:
    """A FRESH seeded 1-D int stream for the SEEDED tier.

    Returns ``None`` when the reference is not a 1-D-stream entry (more than one
    positional parameter): the seeded tier then only runs if the reference
    module exports ``stimulus_for_seed(seed)``. This mirrors ``_default_stimulus``
    so a design the default tier could self-test is also seeded-testable.
    """
    if entry_callable is None:
        return None
    try:
        sig = inspect.signature(entry_callable)
        positional = [
            p
            for p in sig.parameters.values()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
    except (TypeError, ValueError):
        positional = []
    if len(positional) > 1:
        # Multi-parameter reference: cannot safely synthesize a stream. The
        # caller falls back to a ``stimulus_for_seed`` export if present.
        return None
    rng = random.Random(seed)
    n = rng.randint(6, 16)
    return [rng.randint(0, 255) for _ in range(n)]


def _stimulus_divergence(
    project_root: str, expected: Any, observed_norm: Any
) -> tuple[bool, str]:
    """Compare ONE extra-tier stimulus's expected vs observed using the same
    acceptance policy as the default tier (bit-exact / declared acceptance_fn /
    default reference-equivalence). Returns ``(diverged, criterion)``.

    The FIDELITY tier is NOT applied here (it needs a metric + derate ledger);
    extra stimulus tiers are gated OFF entirely when fidelity mode is active
    (see :func:`_run_extra_stimulus_tiers`). The deprecated non-degenerate
    escape hatch is intentionally ignored here -- catching a memorized/degenerate
    output under a fresh stimulus is the whole point.
    """
    if bit_exact_enabled():
        ok, kind = _gate_match(observed_norm, expected)
        return (not ok, "bit_exact" if kind == "exact" else "float_epsilon_divergence")
    accept = resolve_functional_acceptance(project_root)
    if accept is not None:
        try:
            ok = bool(accept(expected, observed_norm))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "model integration gate: acceptance predicate raised %s on an "
                "extra-tier stimulus -- treating as divergence", exc,
            )
            ok = False
        return (not ok, "functional_declared")
    ok, kind = _gate_match(observed_norm, expected)
    if ok:
        return (False, "")
    degenerate = _is_degenerate(observed_norm)
    crit = (
        "functional_nondegenerate" if degenerate
        else "functional_float_epsilon_divergence" if kind == "float_epsilon"
        else "functional_reference_divergence"
    )
    return (True, crit)


def detect_byte_shift(expected: Any, observed: Any, max_shift: int = 4) -> str:
    """Detect an ALIGNMENT divergence: observed == expected shifted by +/-k
    bytes (with inserted/duplicated head bytes or dropped head bytes).

    armD live: the gate reported 'unlocalized block_math' for
    observed == dup_head_byte + expected[:-1] -- the whole content chain was
    byte-exact and the real defect was an egress FIFO head-duplication (FWFT
    class). A shift detection names that class instantly instead of routing a
    content re-spec. Returns '' when no shift pattern matches.
    """
    try:
        if not (_is_byteseq(expected) and _is_byteseq(observed)):
            return ""
        e = bytes(expected) if isinstance(expected, (bytes, bytearray)) else bytes(
            v & 0xFF for v in expected)
        o = bytes(observed) if isinstance(observed, (bytes, bytearray)) else bytes(
            v & 0xFF for v in observed)
        if not e or not o or e == o:
            return ""
        n = min(len(e), len(o))
        if n < 8:
            return ""
        for k in range(1, max_shift + 1):
            # observed carries k EXTRA leading bytes (dup/inserted head):
            # o[k:] aligns with e
            if len(o) >= len(e) - max_shift and o[k:k + n - k] == e[:n - k]:
                return (
                    f"BYTE-SHIFT DETECTED: the observed stream equals the "
                    f"expected stream shifted by +{k} byte(s) (extra/duplicated "
                    f"head byte(s) {o[:k].hex()}). The CONTENT chain is "
                    f"byte-exact -- this is an egress FIFO / handshake "
                    f"ALIGNMENT bug (FWFT head-duplication / refill class), "
                    f"NOT content math. Inspect the egress-most block's FIFO "
                    f"pop/refill logic under backpressure; do NOT re-spec the "
                    f"datapath blocks."
                )
            # observed MISSING k leading bytes (head dropped): e[k:] aligns
            if o[:n - k] == e[k:k + n - k]:
                return (
                    f"BYTE-SHIFT DETECTED: the observed stream equals the "
                    f"expected stream with the first {k} byte(s) DROPPED "
                    f"(missing head {e[:k].hex()}). The content chain is "
                    f"byte-exact -- an ingress/egress handshake drops the "
                    f"first beat(s) (reset/priming class), not content math."
                )
    except Exception:  # noqa: BLE001
        return ""
    return ""


def _load_frd_func_vectors(project_root: str) -> list[dict]:
    """Machine-readable FRD FUNC vectors that carry an EXPLICIT structured
    stimulus (``stimulus_struct`` present). Prose-only vectors are skipped."""
    frd_path = Path(project_root) / "arch" / "frd_spec.md"
    if not frd_path.exists():
        return []
    try:
        frd_text = frd_path.read_text(encoding="utf-8")
    except OSError:
        return []
    return [
        v for v in parse_func_vectors(frd_text)
        if v.get("stimulus_struct") is not None
    ]


def _run_extra_stimulus_tiers(
    *,
    project_root: str,
    entry_callable: Callable | None,
    entry_name: str,
    ref_module: Any,
    simulate: Callable,
    chip_model_path: Path,
    models_dir: Path,
    sim_timeout: float,
) -> list[dict]:
    """Run the FIXED-FRD-vector and SEEDED anti-memorization stimulus tiers.

    Runs only after the default-tier functional check passed (caller guards).
    Each extra stimulus computes ``expected`` from the reference oracle and
    ``observed`` from the composed chip model, then compares via
    :func:`_stimulus_divergence`. A stimulus the reference cannot be invoked
    with, or that the sim cannot run (timeout / raise), is SKIPPED (logged) --
    never a false fail. A genuine divergence is a violation tagged with
    ``stimulus_tier`` (+ ``seed`` / ``func_id``).
    """
    violations: list[dict] = []

    def _one(stimulus: Any, *, tier: str, seed: int | None = None,
             func_id: str = "") -> None:
        try:
            expected = _run_reference(entry_callable, stimulus, reraise=True)
        except ReferenceInvocationError as exc:
            logger.info(
                "model integration gate: %s-tier stimulus skipped -- reference "
                "%r could not be invoked: %s", tier, entry_name, exc,
            )
            return
        if expected is None:
            logger.info(
                "model integration gate: %s-tier stimulus skipped -- reference "
                "returned None", tier,
            )
            return
        raw_observed, timed_out, sim_exc = _dispatch_simulate(
            simulate, chip_model_path, models_dir, stimulus, sim_timeout
        )
        if timed_out or sim_exc is not None:
            logger.info(
                "model integration gate: %s-tier stimulus skipped -- sim %s",
                tier, "timed out" if timed_out else f"raised {sim_exc}",
            )
            return
        observed_out, _cycles = _split_observed(raw_observed)
        observed_norm = _normalize_ref_output(observed_out)
        diverged, criterion = _stimulus_divergence(
            project_root, expected, observed_norm
        )
        if not diverged:
            return
        v = {
            "type": "model_integration_failure",
            "first_divergence_block": _first_divergence_block(project_root),
            "expected": expected,
            "observed": observed_norm,
            "gap_class": _classify_gap(project_root, expected, observed_norm),
            "criterion": criterion,
            "stimulus_tier": tier,
            "suggested_fix": (
                "The integrated chip model matched the reference on the FIXED "
                f"stimulus but DIVERGED under a {tier} stimulus"
                + (f" (seed={seed})" if seed is not None else "")
                + (f" ({func_id})" if func_id else "")
                + ". A pass on the fixed stimulus with a fresh-stimulus failure "
                "is the signature of a MEMORIZED / hardcoded output -- the "
                "composed block models do not implement the reference math. "
                "Inspect the diverging block's transcription in arch/block_models/."
            ),
        }
        if seed is not None:
            v["seed"] = seed
        if func_id:
            v["func_id"] = func_id
        violations.append(v)

    # Tier FIXED: machine-readable FRD FUNC vectors with explicit stimulus.
    for vec in _load_frd_func_vectors(project_root):
        _one(vec["stimulus_struct"], tier="fixed", func_id=vec.get("id", ""))

    # Tier SEEDED: a fresh seeded stimulus the chip model could not memorize.
    if _gate_seeded_stimulus_enabled():
        seed = _resolve_gate_seed()
        seeder = getattr(ref_module, "stimulus_for_seed", None)
        seeded_stim = None
        if callable(seeder):
            try:
                seeded_stim = seeder(seed)
            except Exception as exc:  # noqa: BLE001
                logger.info(
                    "model integration gate: reference stimulus_for_seed(%d) "
                    "raised %s -- falling back to derived seeded stream",
                    seed, exc,
                )
                seeded_stim = None
        if seeded_stim is None:
            seeded_stim = _seeded_stimulus(entry_callable, seed)
        if seeded_stim is None:
            logger.info(
                "model integration gate: SEEDED tier skipped -- reference is "
                "not a 1-D-stream entry and exports no stimulus_for_seed"
            )
        else:
            _one(seeded_stim, tier="seeded", seed=seed)

    return violations


def load_chip_model(project_root: str) -> Callable:
    """Import ``arch/block_models/_chip_model.py`` and return its ``simulate``.

    Shared loader (A-Fix 5d): both the model-integration gate and the chip-top
    RTL-vs-model equivalence gate need the composed Amaranth chip model's
    ``simulate(stimulus)``. Puts the block-models dir on ``sys.path`` for the
    import (the chip model imports its sibling block modules by plain name) and
    purges any stale same-named modules first. Raises ``RuntimeError`` (a
    missing dir / file / import failure / missing simulate) so callers surface a
    hard error rather than silently treating "no model" as a pass.
    """
    import sys

    root = Path(project_root)
    models_dir = root / "arch" / BLOCK_MODELS_DIRNAME
    chip_model_path = models_dir / CHIP_MODEL_FILENAME
    if not chip_model_path.exists():
        raise RuntimeError(f"no chip model at {chip_model_path}")

    inserted = False
    if str(models_dir) not in sys.path:
        sys.path.insert(0, str(models_dir))
        inserted = True
    _block_stems = {
        p.stem
        for p in models_dir.glob("*.py")
        if p.name != CHIP_MODEL_FILENAME and not p.name.startswith("__")
    }
    for _stem in _block_stems | {"_coresmith_chip_model"}:
        sys.modules.pop(_stem, None)
    try:
        chip_mod = _import_module_from_path(chip_model_path, "_coresmith_chip_model")
    finally:
        if inserted:
            try:
                sys.path.remove(str(models_dir))
            except ValueError:
                pass
    simulate = getattr(chip_mod, "simulate", None)
    if simulate is None or not callable(simulate):
        raise RuntimeError(
            f"{chip_model_path} exposes no callable simulate(stimulus)"
        )
    return simulate


def _broadcast_unlocalized_enabled() -> bool:
    """True (default) -> the gate BROADCASTS an unlocalizable output divergence.

    Default ON. Set ``CORESMITH_GATE_BROADCAST_UNLOCALIZED=0`` to restore the
    legacy behaviour where :func:`_first_divergence_block` named diagram block 0
    (a false-precise pointer that misdirected revise_uarch).
    """
    return os.environ.get(
        "CORESMITH_GATE_BROADCAST_UNLOCALIZED", "1"
    ).strip().lower() not in {"0", "false", "no", "off", ""}


def _load_block_diagram(project_root: str) -> dict:
    """Load ``.coresmith/block_diagram.json`` or return ``{}`` on any failure."""
    import json

    bd_path = Path(project_root) / ".coresmith" / "block_diagram.json"
    if not bd_path.exists():
        return {}
    try:
        bd = json.loads(bd_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return bd if isinstance(bd, dict) else {}


def _first_divergence_block(project_root: str, gap_class: str | None = None) -> str:
    """Return "" (unlocalized -> broadcast) for a composed-output divergence.

    A wrong-bytes / wrong-length divergence in the COMPOSED output cannot be
    localized to a single block without per-block reference *intermediates*,
    which the reference does not expose. The culprit may be ANYWHERE in the
    serialization chain — the SINK (rerun9: block_packer accept-side ack-and-drop)
    OR a MIDDLE block (rerun10: entropy_enc omitted the per-subblock prediction-mode
    field and emitted the frame/shape prefix per-subblock instead of per-MB).

    Naming any single block (diagram block 0, or a graph-terminal that turned out
    to be a status/control sideband sink) produces a FALSE-PRECISE
    ``affected_edge`` that makes ``_gate_localization_precise()`` SUPPRESS the
    broadcast and misdirect ``revise_uarch`` to the wrong block — proven to
    prevent convergence twice (block-0 in rerun9, lifecycle_status_ctrl in
    rerun10). So return "": the router then broadcasts the re-spec feedback + the
    serialization/axi_stream skill rules to ALL blocks, guaranteeing the real
    culprit is re-specced wherever it sits in the chain.

    ``gap_class`` is accepted (callers pass it) but no longer changes the result;
    a contract gap is just as unlocalizable as a value gap. Legacy single-block
    attribution (first declared block) is restored with
    ``CORESMITH_GATE_BROADCAST_UNLOCALIZED=0``.
    """
    if _broadcast_unlocalized_enabled():
        return ""
    bd = _load_block_diagram(project_root)
    blocks = [
        b.get("name") for b in (bd.get("blocks", []) or []) if b.get("name")
    ]
    return blocks[0] if blocks else ""


def _field_localization_enabled() -> bool:
    """True (default) -> the gate attributes a dict-output divergence to the
    PRODUCING SUB-CHAIN of only the diverging output field(s).

    This is the convergence fix for framework-HDL composition runs: a wrong-bytes
    divergence in ONE field of a multi-field output (e.g. a video codec's
    ``{"bitstream": ..., "recon": ..., "stats": ...}``) is localizable to the
    blocks that feed that field, so a gate-triggered re-spec touches only those
    blocks and KEEPS the (passing) blocks of the other fields' sub-chains.
    Without this the gate returns no ``affected_blocks`` and the daemon
    BROADCASTS the re-spec to every block -- re-rolling all blocks each pass,
    which is slow and has regressed already-good blocks.

    Set ``CORESMITH_GATE_FIELD_LOCALIZATION=0`` to restore the pure-broadcast
    behaviour.
    """
    return os.environ.get(
        "CORESMITH_GATE_FIELD_LOCALIZATION", "1"
    ).strip().lower() not in {"0", "false", "no", "off", ""}


def _bd_connections(bd: dict) -> list[dict]:
    """Normalised connection edges from a block diagram (``connections`` or
    legacy ``edges``); each edge is ``{"from","to","interface"}``-ish."""
    raw = bd.get("connections") or bd.get("edges") or []
    out: list[dict] = []
    for c in raw:
        if not isinstance(c, dict):
            continue
        src = c.get("from", c.get("from_block", ""))
        dst = c.get("to", c.get("to_block", ""))
        out.append({"from": src, "to": dst, "interface": c.get("interface", "")})
    return out


# A SIDEBAND (status / telemetry / monitor) edge -- NOT a functional DATAPATH
# edge. A monitor block fans IN from every datapath block via an
# ``event_axis``/``status_axis`` sideband; counting those edges as datapath would
# (a) hide true datapath sinks behind the monitor and (b) make a passing
# ``stats`` field falsely "prove good" every block that merely reports to the
# monitor. We exclude sideband edges when computing datapath sinks and producer
# sub-chains, so attribution follows the real data flow.
#
# The interface match is on the LEADING token of the interface name so a true
# sideband (``event_axis`` / ``status_axis``) is dropped but a DATAPATH wire that
# merely contains the word (``pixel_event_axis`` -- carries pixels) is KEPT. The
# monitor-block check below is the primary, reliable discriminator; this is a
# defensive secondary.
_SIDEBAND_INTERFACE_PREFIXES = ("event_", "status_", "monitor_", "telemetry_",
                                "debug_", "perf_", "counter_")
_SIDEBAND_INTERFACE_EXACT = ("event", "status", "monitor", "telemetry", "debug")
_SIDEBAND_BLOCK_TOKENS = ("status", "monitor", "lifecycle", "telemetry", "debug")


def _is_sideband_block(name: str, bd: dict) -> bool:
    """True if ``name`` is a status/telemetry MONITOR block (not on a datapath)."""
    ln = (name or "").lower()
    if any(t in ln for t in _SIDEBAND_BLOCK_TOKENS):
        return True
    for b in bd.get("blocks", []):
        if b.get("name") == name:
            desc = (b.get("description") or "").lower()
            # A block that explicitly does NOT participate in functional decisions
            # is a pure monitor (video_codec lifecycle_status_monitor says exactly this).
            if "without participating" in desc or "does not participate" in desc:
                return True
    return False


def _datapath_connections(conns: list[dict], bd: dict) -> list[dict]:
    """Functional-datapath edges only: drop edges into a monitor/status block or
    carrying a sideband (``event``/``status``) interface."""
    out: list[dict] = []
    for e in conns:
        iface = (e.get("interface") or "").lower()
        if iface in _SIDEBAND_INTERFACE_EXACT or iface.startswith(
            _SIDEBAND_INTERFACE_PREFIXES
        ):
            continue
        if _is_sideband_block(e.get("to", ""), bd):
            continue
        out.append(e)
    return out


def _producer_subchain(terminal: str, conns: list[dict]) -> set[str]:
    """All blocks that (transitively) FEED ``terminal`` -- backward reachability
    over the connection DAG, including ``terminal`` itself."""
    incoming: dict[str, list[str]] = {}
    for e in conns:
        if e["to"] and e["from"]:
            incoming.setdefault(e["to"], []).append(e["from"])
    seen: set[str] = set()
    stack = [terminal]
    while stack:
        b = stack.pop()
        if not b or b in seen:
            continue
        seen.add(b)
        stack.extend(incoming.get(b, []))
    return seen


# Field-key synonyms: an output dict field often does NOT literally appear in a
# block name/interface (e.g. ``recon`` is produced by an "inverse RECONSTRUCTION"
# core; ``stats`` by a "performance COUNTERS / STATUS monitor"). Expanding the
# field tokens with common synonyms lets a passing field map to its PRODUCER so
# that producer (and its prefix) is subtracted from the diverging field's culprit
# set. Purely additive heuristic -- it only ever HELPS attribution be sharper;
# an unmapped field still falls back to broadcast (honest, never false-precise).
_FIELD_SYNONYMS: dict[str, tuple[str, ...]] = {
    "recon": ("recon", "reconstruct", "reconstruction", "inverse", "rebuild"),
    "reconstruction": ("recon", "reconstruct", "reconstruction", "inverse"),
    "stats": ("stat", "stats", "status", "counter", "perf", "performance",
              "lifecycle", "monitor", "telemetry"),
    "status": ("status", "stat", "counter", "monitor", "lifecycle"),
    "bitstream": ("bitstream", "bytestream", "byte", "entropy", "entropy", "stream"),
    "stream": ("stream", "byte", "output", "adapter"),
}


def _terminal_for_field(field: str, bd: dict, conns: list[dict]) -> str | None:
    """Best-effort: the graph-TERMINAL (sink) block that emits output ``field``.

    A field maps to the block whose name / interface keys / description most
    resemble the field key (with common synonyms). Heuristic + generic: tokens of
    the field key (e.g. ``bitstream`` / ``recon`` / ``stats``) -- expanded via
    :data:`_FIELD_SYNONYMS` -- are matched against block names, the block's own
    interface names, and its description. Graph-sinks are preferred so the
    backward sub-chain reaches the whole producing tail. Returns ``None`` if no
    confident match (=> the field is treated as unmappable -> broadcast).
    """
    blocks = [b for b in bd.get("blocks", []) if b.get("name")]
    names = [b["name"] for b in blocks]
    by_name = {b["name"]: b for b in blocks}
    # Datapath-only edges for sink detection: a monitor/status sideband consumer
    # is not a datapath sink and must not mask the real functional-output block.
    dp = _datapath_connections(conns, bd)
    srcs = {e["from"] for e in dp if e["from"]}
    dsts = {e["to"] for e in dp if e["to"]}
    # Graph-terminal sinks: appear as a producer/consumer but feed nothing
    # downstream (no datapath edge leaves them), i.e. true outputs of the chip.
    sinks = [n for n in names if n in (srcs | dsts) and n not in srcs]
    fl = field.lower()
    ftoks = {t for t in fl.replace("-", "_").split("_") if t}
    for base, syns in _FIELD_SYNONYMS.items():
        if base in fl or any(t == base for t in ftoks):
            ftoks |= set(syns)

    def _score(label: str, weight: int = 1) -> int:
        if not label:
            return 0
        ll = label.lower()
        s = 0
        if fl and fl in ll:
            s += 4 * weight
        for t in ftoks:
            if len(t) >= 3 and t in ll:
                s += 2 * weight
        return s

    def _block_score(name: str) -> int:
        """Best evidence that block ``name`` PRODUCES ``field``: its own name
        (strongest), its interface keys (the ports it emits), then a weaker hit
        in its description. Sinks get a small bonus (true chip outputs)."""
        b = by_name.get(name, {})
        s = _score(name, weight=2)
        ifaces = b.get("interfaces") or {}
        if isinstance(ifaces, dict):
            for iface_name in ifaces:
                s = max(s, _score(iface_name, weight=1))
        s = max(s, _score(b.get("description", ""), weight=1) // 2)
        if name in sinks and s > 0:
            s += 1  # tie-break toward the graph-terminal output block
        return s

    best: tuple[int, str | None] = (0, None)
    for n in names:
        sc = _block_score(n)
        if sc > best[0]:
            best = (sc, n)
    # Then the interface name of an edge whose CONSUMER is a sink (that edge
    # carries the field to the boundary) -- catches a field named after the wire.
    for e in conns:
        if e["to"] in sinks:
            sc = _score(e.get("interface", ""))
            if sc > best[0]:
                best = (sc, e["to"])
    return best[1] if best[0] > 0 else None


def _localize_affected_blocks(
    project_root: str, expected: Any, observed: Any
) -> list[str]:
    """Attribute a divergence to the blocks feeding the diverging output FIELD(s).

    Only fires for a dict-vs-dict output with identical keys (the multi-output
    composition case). For each field, compare expected vs observed; the
    diverging fields' producing sub-chains are the candidate culprits. Blocks
    that feed ONLY a PASSING field are excluded so they are not re-rolled.

    Returns ``[]`` (=> unlocalizable => the caller broadcasts) when:
      * field localization is disabled, or
      * the output is not a same-keys dict (e.g. a sim-error STRING), or
      * EVERY field diverges (no passing sub-chain to protect -> broadcast), or
      * no diverging field maps to a block.
    """
    if not _field_localization_enabled():
        return []

    bd = _load_block_diagram(project_root)
    if not bd.get("blocks"):
        return []
    order = [b.get("name") for b in bd.get("blocks", []) if b.get("name")]

    # Only the dict-vs-dict (multi-field composed output) case is field-
    # localizable. A sim-error string / scalar / list output is unlocalizable
    # to a field here -> broadcast (return []).
    if not (isinstance(expected, dict) and isinstance(observed, dict)):
        return []
    if set(expected.keys()) != set(observed.keys()) or not expected:
        return []

    conns = _bd_connections(bd)
    # Producer sub-chains follow the FUNCTIONAL DATAPATH only. The status/monitor
    # sideband (every block -> lifecycle_status_monitor via event_axis) is NOT a
    # datapath: a passing ``stats`` field proves the monitor good, it does NOT
    # prove the datapath blocks that merely report to it good. Walking datapath
    # edges keeps the shared-prefix subtraction honest.
    dp_conns = _datapath_connections(conns, bd)

    diverging: list[str] = []
    passing: list[str] = []
    for k in expected:
        if _outputs_match(observed.get(k), expected.get(k)):
            passing.append(k)
        else:
            diverging.append(k)

    # All fields diverge (or none) -> nothing to protect; broadcast.
    if not diverging or not passing:
        return []

    div_terms: list[str] = []
    div_blocks: set[str] = set()
    for f in diverging:
        term = _terminal_for_field(f, bd, conns)
        if term is None:
            # A diverging field we cannot map -> don't risk a false-precise
            # localization; fall back to broadcast.
            return []
        div_terms.append(term)
        # Backward: every block feeding the field's value (its producers).
        div_blocks |= _producer_subchain(term, dp_conns)

    pass_blocks: set[str] = set()
    pass_terms: set[str] = set()
    for f in passing:
        term = _terminal_for_field(f, bd, conns)
        if term is not None:
            pass_terms.add(term)
            pass_blocks |= _producer_subchain(term, dp_conns)

    # Forward extension (bounded): a diverging field's value is also touched by
    # the datapath blocks that FORWARD/serialize it downstream to the chip
    # boundary -- they can corrupt it too (e.g. entropy_bitstream_engine ->
    # output_stream_adapter). Walk forward from each diverging terminal, but STOP
    # at any block that is a PASSING field's terminal or lies in a passing field's
    # producer region: the diverging value does not flow THROUGH a block that is
    # busy producing a (correct) different output, so crossing it would be a
    # false-positive (the recon-terminal -> entropy case). This keeps the
    # extension to the diverging field's OWN tail.
    outgoing: dict[str, list[str]] = {}
    for e in dp_conns:
        if e["from"] and e["to"]:
            outgoing.setdefault(e["from"], []).append(e["to"])
    for term in div_terms:
        stack = list(outgoing.get(term, []))
        while stack:
            nxt = stack.pop()
            if nxt in div_blocks:
                continue
            if nxt in pass_terms or nxt in pass_blocks:
                # Downstream block is dedicated to a passing output -> stop;
                # the diverging value does not pass cleanly through it.
                continue
            div_blocks.add(nxt)
            stack.extend(outgoing.get(nxt, []))

    # A block on BOTH a diverging and a passing field's datapath sub-chain (the
    # SHARED PREFIX, e.g. ingress->assemble->intra_core for both ``recon`` and
    # ``bitstream``) demonstrably produced CORRECT data for the passing field --
    # so it is proven-good and is NOT the culprit for the diverging field.
    # Re-spec only the blocks EXCLUSIVE to the diverging field(s) (the field's own
    # tail). This is the sharpest honest localization: it keeps every block that
    # any passing field exercised. (For codecv4: recon proves
    # input/pixel_block/intra good, so bitstream's exclusive tail =
    # entropy_bitstream_engine + output_stream_adapter.)
    exclusive = div_blocks - pass_blocks
    affected = exclusive if exclusive else (div_blocks - (pass_blocks - div_blocks))
    # Keep deterministic block-diagram order (``order`` computed at function top).
    return [n for n in order if n in affected]


def _split_observed(observed: Any) -> tuple[Any, int | None]:
    """Split a ``simulate()`` return into ``(output, cycles)``.

    The v2 ``simulate(stimulus)`` contract returns a ``(output, cycles)``
    2-tuple. Legacy / on-disk models that still return a bare output (a
    1-tuple-equivalent, i.e. anything that is not a 2-tuple) are tolerated:
    ``cycles`` is ``None`` and the throughput check is skipped (R6).

    A 2-tuple is recognised ONLY when its second element is an int/float cycle
    count; a 2-element list output (e.g. ``[a, b]``) is NOT mistaken for
    ``(output, cycles)``.
    """
    if (
        isinstance(observed, tuple)
        and len(observed) == 2
        and isinstance(observed[1], (int, float))
        and not isinstance(observed[1], bool)
    ):
        return observed[0], int(observed[1])
    return observed, None


def _flatten_scalars(obj: Any) -> list:
    """Recursively flatten ``obj`` to a list of Python scalars.

    Handles nested lists/tuples, numpy arrays (flattened to scalars), and
    bytes/bytearray (expanded to their integer byte values). Other objects are
    appended as-is. This is what makes degeneracy/equality checks numpy-safe:
    a bare ``v == first`` on a numpy array raises "truth value ambiguous", and
    a ``bytes`` bitstream must be treated as its byte sequence (not one opaque
    element that would look constant).
    """
    out: list = []
    stack = [obj]
    while stack:
        x = stack.pop()
        if x is None:
            out.append(None)
        elif isinstance(x, (bytes, bytearray)):
            out.extend(int(b) for b in x)
        elif isinstance(x, (list, tuple)):
            stack.extend(x)
        elif hasattr(x, "flat") and hasattr(x, "shape"):  # numpy ndarray
            try:
                out.extend(x.flatten().tolist())
            except Exception:  # noqa: BLE001
                out.append(x)
        else:
            out.append(x)
    return out


def _is_degenerate(observed: Any) -> bool:
    """True when ``observed`` is degenerate (all-zero / constant / None / empty).

    Used by the functional Tier-B non-degenerate fallback: a flat / collapsed
    output (the class of bug that shipped the codec flat-output) is rejected
    even without a declared acceptance predicate. Numpy/bytes-safe.
    """
    if observed is None:
        return True
    if isinstance(observed, (list, tuple, bytes, bytearray)) or (
        hasattr(observed, "flat") and hasattr(observed, "shape")
    ):
        flat = _flatten_scalars(observed)
        if not flat:
            return True
        first = flat[0]
        # All-constant (incl. all-zero) -> degenerate. Scalar `==` only (flat
        # holds Python scalars / ints), so no numpy-array truthiness ambiguity.
        if all(bool(v == first) for v in flat):
            return True
        return False
    if isinstance(observed, (int, float)):
        return observed == 0
    if isinstance(observed, str):
        return observed.strip() == ""
    if isinstance(observed, dict):
        if not observed:
            return True
        return all(_is_degenerate(v) for v in observed.values())
    return False


def _gate_match(observed: Any, expected: Any) -> tuple[bool, str]:
    """Compare composed output to the reference under the fixed-point/float
    policy. Returns ``(ok, criterion_suffix)``.

    DEFAULT = bit-exact (the design SHOULD use fixed-point so the integer RTL
    datapath reproduces the reference exactly). If the REFERENCE OUTPUT is
    float-valued, bit-exact reproduction is unrealistic, so we accept within an
    epsilon (CORESMITH_GATE_EPSILON) and LOUDLY surface that floats were detected
    -- the operator should confirm and prefer a fixed-point golden where
    possible (user policy: bias fixed-point; epsilon only for genuine floats).
    """
    if output_has_float(expected):
        eps = gate_epsilon()
        ok = outputs_close(observed, expected, eps)
        logger.warning(
            "model integration gate: FLOAT detected in the reference output -- "
            "bit-exact is unachievable for a float golden. Applying epsilon "
            "tolerance %g (CORESMITH_GATE_EPSILON). PREFER a fixed-point golden "
            "for a deterministic bit-exact gate; otherwise confirm epsilon is "
            "acceptable. match=%s", eps, ok,
        )
        return ok, "float_epsilon"
    return _outputs_match(observed, expected), "exact"


def _is_byteseq(v: Any) -> bool:
    """True for a byte/int sequence (a bitstream), not a dict/str/scalar."""
    if isinstance(v, (bytes, bytearray)):
        return True
    if isinstance(v, (list, tuple)) and v and all(isinstance(x, int) for x in v):
        return True
    return False


def first_divergence_offset(expected: Any, observed: Any) -> int:
    """Byte index of the first mismatch (-1 if one is a clean prefix of the other
    AND they are equal length; the shorter length otherwise)."""
    try:
        n = min(len(expected), len(observed))
    except TypeError:
        return -1
    for i in range(n):
        if expected[i] != observed[i]:
            return i
    return -1 if len(expected) == len(observed) else n


# ---------------------------------------------------------------------------
# Per-block golden-feasibility probe (Phase 2C) -- closes the best-effort swallow.
#
# _maybe_generate_block_golden used to wrap the whole generation in a
# try/except that SILENTLY swallowed every outcome. This probe makes the
# degenerate cases VISIBLE (and, under a flag, blocking): a block whose model
# fails to import/validate, whose golden reference cannot be resolved, or whose
# golden slice is never exercised on realistic input (an orphaned / mis-
# decomposed block).
#
# SCOPE (honest): a *sophisticated* stub -- one that drives the right ports but
# routes the primary DATA path to a terminal error (the reference codec's dct_token_engine
# emitting m_axis_error_completion instead of m_axis_fragment_coeff) -- is NOT
# caught here. Detecting that requires simulating the block MODEL and byte-
# comparing to the golden's boundary I/O, which needs a per-block Amaranth sim
# harness + a confident block->golden slice (Phase 2's generalized mapping). That
# is the documented follow-up. For the reference codec-class stub the uarch FEASIBILITY
# gate (Phase 1) is the live backstop: it stops the block at spec time, before a
# model is ever generated. This probe closes the *other* half -- the swallow.
# ---------------------------------------------------------------------------

def golden_feasibility_gate_enabled() -> bool:
    """CORESMITH_GOLDEN_FEASIBILITY_GATE (default OFF): when ON, a FAILED probe
    hard-fails the block. Default OFF = record + warn (visible, non-blocking) so
    the check rolls out without breaking live runs; the verdict is always written
    to ``.coresmith/blocks/<b>/golden_feasibility.json`` regardless."""
    return os.environ.get(
        "CORESMITH_GOLDEN_FEASIBILITY_GATE", "0"
    ).strip().lower() in ("1", "true", "yes", "on")


def _slice_reachability_probe(project_root: str, ref_path: str,
                              slice_path: Path) -> dict:
    """Instrument the block's golden-slice functions, run the golden reference on
    a stimulus, and report whether the slice is exercised + produces output.

    Robust by construction: concludes 'reachable'/'unreachable' ONLY when the
    slice functions resolve to module-level callables we can actually intercept
    (free-function goldens, e.g. the video codec). For method-based goldens (the reference codec's
    stateful decoder methods) module-level instrumentation does not apply, so we
    honest-SKIP rather than risk a false 'unreachable'. Never raises.
    """
    out = {"verdict": "skipped", "reason": "", "calls": {}}
    import json as _json
    if not slice_path.exists():
        out["reason"] = "no confident slice (needs Phase-2 generalized mapping)"
        return out
    try:
        slice_fns = _json.loads(
            slice_path.read_text(encoding="utf-8")).get("golden_functions", [])
    except Exception:  # noqa: BLE001
        out["reason"] = "slice sidecar unreadable"
        return out
    if not slice_fns:
        out["reason"] = "empty slice"
        return out
    if not ref_path:
        out["reason"] = "golden reference unresolved"
        return out
    try:
        ref_mod = _load_reference_module(ref_path)
        entry, _name = resolve_reference_entrypoint(project_root, ref_mod)
    except Exception as exc:  # noqa: BLE001
        out["reason"] = f"golden load failed: {exc}"
        return out
    if entry is None:
        out["reason"] = "golden entry unresolved"
        return out

    calls = {fn: {"n": 0, "produced": False} for fn in slice_fns}
    originals: dict = {}
    for fn in slice_fns:
        orig = getattr(ref_mod, fn, None)
        if not callable(orig):
            continue  # a method / non-module-level name -> not instrumentable
        originals[fn] = orig

        def _make(fnname, o):
            def _wrapped(*a, **k):
                r = o(*a, **k)
                c = calls[fnname]
                c["n"] += 1
                if r is not None and not (hasattr(r, "__len__") and len(r) == 0):
                    c["produced"] = True
                return r
            return _wrapped
        setattr(ref_mod, fn, _make(fn, orig))

    if not originals:
        # every slice fn is a method / not a module global -> cannot instrument
        # without risking a false 'unreachable'. Honest skip.
        out["reason"] = ("slice functions are methods / not module-level; "
                         "module-level reachability not applicable")
        return out

    # Stimulus: env-provided, else auto-discovered inputs/model_stimulus.py
    # (finding #5), else a small default derived from the entry sig.
    stim, found = _load_env_stimulus(project_root)
    if not found:
        try:
            stim = _default_stimulus(entry)
        except Exception:  # noqa: BLE001
            for fn, o in originals.items():
                setattr(ref_mod, fn, o)
            out["reason"] = "no stimulus available"
            return out
    try:
        entry(stim)
    except Exception as exc:  # noqa: BLE001
        # The golden itself raised on this stimulus -> inconclusive, not a fail.
        out["reason"] = f"golden run raised ({exc}); inconclusive"
        return out
    finally:
        for fn, o in originals.items():
            setattr(ref_mod, fn, o)

    out["calls"] = {k: v for k, v in calls.items() if k in originals}
    exercised = any(calls[fn]["n"] > 0 for fn in originals)
    produced = any(calls[fn]["produced"] for fn in originals)
    if not exercised:
        out["verdict"] = "unreachable"
        out["reason"] = (
            f"none of the block's {len(originals)} instrumentable golden-slice "
            f"fn(s) were called on the stimulus -- orphaned / mis-decomposed block")
    elif not produced:
        out["verdict"] = "unreachable"
        out["reason"] = ("block golden-slice fns ran but produced only empty/None "
                         "output on the stimulus")
    else:
        out["verdict"] = "reachable"
    return out


def check_golden_feasibility(project_root: str, block_name: str) -> dict:
    """Probe whether a generated block golden/model is non-degenerate.

    Returns ``{block, ran, passed, skipped, reason, checks}``; never raises. Used
    by ``_maybe_generate_block_golden`` (to close the swallow) and by the
    ``coresmith golden-check`` CLI. Writes the verdict to
    ``.coresmith/blocks/<b>/golden_feasibility.json``.
    """
    result = {"block": block_name, "ran": False, "passed": True,
              "skipped": False, "reason": "", "checks": {}}
    try:
        from orchestrator.architecture import composition as _composition
        if not _composition.block_goldens_enabled():
            result.update(skipped=True, reason="block goldens disabled")
            return _persist_golden_feasibility(project_root, block_name, result)
        root = Path(project_root)
        model_path = (root / "arch" / _composition.BLOCK_MODELS_DIRNAME
                      / f"{block_name}.py")
        if not model_path.exists():
            result.update(ran=True, passed=False, reason="no block model emitted")
            return _persist_golden_feasibility(project_root, block_name, result)
        # 1. model structural validity (import + @block factory present)
        from orchestrator.langchain.agents.block_golden_generator import (
            _validate_block_model_file,
        )
        problem = _validate_block_model_file(str(model_path), block_name)
        result["checks"]["model_valid"] = problem is None
        if problem is not None:
            result.update(ran=True, passed=False,
                          reason=f"model invalid: {problem}")
            return _persist_golden_feasibility(project_root, block_name, result)
        # 2. golden reference resolvable
        ref_path = _composition.resolve_generator_reference(project_root)
        result["checks"]["golden_resolvable"] = bool(ref_path)
        # 3. slice reachability (best-effort; free-function goldens only)
        slice_path = model_path.with_suffix(".slice.json")
        reach = _slice_reachability_probe(project_root, ref_path or "", slice_path)
        result["checks"]["slice_reachability"] = reach
        result["ran"] = True
        if reach.get("verdict") == "unreachable":
            result.update(passed=False, reason=reach.get("reason", "unreachable"))
        return _persist_golden_feasibility(project_root, block_name, result)
    except Exception as exc:  # noqa: BLE001 - the probe must never raise
        result.update(ran=False, skipped=True, reason=f"probe error: {exc}")
        return _persist_golden_feasibility(project_root, block_name, result)


def _persist_golden_feasibility(project_root: str, block_name: str,
                                result: dict) -> dict:
    try:
        import json as _json
        bdir = Path(project_root) / ".coresmith" / "blocks" / block_name
        bdir.mkdir(parents=True, exist_ok=True)
        (bdir / "golden_feasibility.json").write_text(
            _json.dumps(result, indent=2, default=str), encoding="utf-8")
    except OSError:
        pass
    return result


# A real header/framing prefix (parameter-set/group_header) is comfortably longer than
# this; a genuine width/framing mismatch diverges at or near offset 0.
_PREFIX_CLASSIFY_MIN = 8


def _prefix_classify_enabled() -> bool:
    return os.environ.get("CORESMITH_GATE_PREFIX_CLASSIFY", "1").strip() != "0"


def _classify_gap(project_root: str, expected: Any, observed: Any) -> str:
    """Best-effort gap classification: ``"block_math"`` vs ``"contract"``.

    Heuristic (section D, R7): a *shape* mismatch (different length / structure)
    between the composed model output and the reference output points at a
    handshake / width / framing contract that cannot compose -> ``"contract"``.
    A same-shape but value-divergent output points at a block whose transcribed
    math is wrong -> ``"block_math"`` (the common case, the default).

    PREFIX PROBE (CORESMITH_GATE_PREFIX_CLASSIFY, default on): for byte/seq
    outputs, length alone is misleading. A 238-byte golden vs a 50-byte composed
    output that is BYTE-EXACT through byte 31 (the parameter-set/group_header framing)
    and only then diverges/truncates is NOT a framing contract gap -- the chain
    composed correctly and ONE downstream block emitted wrong/short content. That
    is ``block_math`` (route to a single-block re-spec, not a whole-chip re-fan).
    Only a divergence at/near offset 0 (no shared framing) is a true ``contract``
    width/framing mismatch. Keystone fix: the length-only heuristic sent the Opus
    codec's entropy coding bug (byte-32 residual) down the contract path and re-fanned all
    8 blocks twice (>55% of that run's token budget).
    """
    if _prefix_classify_enabled() and _is_byteseq(expected) and _is_byteseq(observed):
        try:
            off = first_divergence_offset(expected, observed)
            if off >= _PREFIX_CLASSIFY_MIN:
                return "block_math"   # framing matched, content diverged -> 1 block
            if off == 0:
                # No shared prefix. LENGTH decides: a truncated/overlong stream
                # is a framing/width contract mismatch, but a SAME-LENGTH output
                # whose values diverge from element 0 is a wrong block's math
                # (the chain composed fine -- same shape, wrong content).
                if len(expected) != len(observed):
                    return "contract"
                return "block_math"
            # 0 < off < MIN -> ambiguous; fall through to the shape heuristic
        except Exception:  # noqa: BLE001
            pass

    def _shape(v: Any):
        # bytes/bytearray are length-bearing SEQUENCES, not opaque scalars: a
        # 57-byte golden vs a 37-byte truncated stream is a length (contract)
        # mismatch, NOT a same-shape value divergence. (The old heuristic saw
        # both as ("scalar",) -> wrongly block_math, defeating revise_uarch.)
        if isinstance(v, (list, tuple, bytes, bytearray)):
            return ("seq", len(v))
        if isinstance(v, dict):
            return ("map", tuple(sorted(v.keys())))
        # numpy arrays (and other objects exposing a length) compare by length.
        try:
            shape = getattr(v, "shape", None)
            if shape is not None:
                return ("seq", tuple(shape))
            if not isinstance(v, (str,)):
                return ("seq", len(v))
        except TypeError:
            pass
        return ("scalar",)

    try:
        if _shape(expected) != _shape(observed):
            return "contract"
    except Exception:  # noqa: BLE001
        return "block_math"
    return "block_math"


def _json_safe(obj: Any) -> Any:
    """Convert a value to a JSON-serialisable preview.

    Gate violation payloads carry ``expected``/``observed`` which for a real
    design are raw ``bytes`` (a bitstream) or numpy arrays. Those flow into the
    daemon's interrupt payload, and FastAPI's encoder raises (bytes -> utf-8
    decode of 0x81; ndarray -> not serialisable), so ``GET /run/state`` 500s and
    the outer agent cannot READ the parked failure to act on it. Stringify them.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (bytes, bytearray)):
        head = bytes(obj[:48]).hex()
        return f"<bytes len={len(obj)} hex={head}{'...' if len(obj) > 48 else ''}>"
    if hasattr(obj, "shape") and hasattr(obj, "flatten"):  # numpy ndarray
        try:
            flat = obj.flatten().tolist()
            return {"_ndarray_shape": list(obj.shape),
                    "preview": flat[:64],
                    "truncated": len(flat) > 64}
        except Exception:  # noqa: BLE001
            return repr(obj)[:300]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in list(obj)[:128]]
    return repr(obj)[:300]


def _sanitize_violations(viols: list[dict]) -> list[dict]:
    """Make every violation dict JSON-serialisable (see :func:`_json_safe`)."""
    return [_json_safe(v) if isinstance(v, dict) else _json_safe(v) for v in viols]


def _localize_from_traceback(models_dir, tb_text: str) -> tuple[str, str]:
    """(block_name, frame_desc) of the DEEPEST traceback frame inside a block
    model, else ("", ""). Turns an unlocalizable crash ("IndexError: list
    index out of range") into a targeted re-spec of the crashing block instead
    of a broadcast re-fan of every block."""
    if not tb_text:
        return "", ""
    import re as _re

    try:
        stems = {
            p.stem
            for p in Path(models_dir).glob("*.py")
            if not p.name.startswith("__")
        }
    except OSError:
        return "", ""
    best: tuple[str, str] = ("", "")
    for m in _re.finditer(r'File "([^"]+)", line (\d+)', tb_text):
        stem = Path(m.group(1)).stem
        if stem in stems:
            best = (stem, f"{Path(m.group(1)).name}:{m.group(2)}")
    if best[0] == "_chip_model":
        # a crash in the composition wiring itself, not a block's math
        return "_chip_model", best[1]
    return best


def _stall_autopsy_file(project_root: str) -> Path:
    return Path(project_root) / ".coresmith" / "stall_autopsy.json"


def _arm_stall_autopsy(project_root: str) -> None:
    """Point a composition edge monitor at this run's autopsy file.

    A simulation edge monitor may read CORESMITH_STALL_AUTOPSY_PATH and
    periodically dump per-edge handshake activity there -- surviving even a
    SIGKILLed sim subprocess. Stale files from a previous run are removed so a
    failure never reads an old autopsy. When no monitor writes the file, the
    summary is simply empty and the plumbing is a graceful no-op.
    """
    try:
        p = _stall_autopsy_file(project_root)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            p.unlink()
        os.environ["CORESMITH_STALL_AUTOPSY_PATH"] = str(p)
    except Exception:  # noqa: BLE001 - autopsy is best-effort diagnostics
        pass


def _stall_autopsy_summary(project_root: str, max_edges: int = 24) -> str:
    """Human/LLM-readable per-edge summary from the autopsy dump ('' if none)."""
    try:
        import json as _json

        p = _stall_autopsy_file(project_root)
        if not p.exists():
            return ""
        data = _json.loads(p.read_text(encoding="utf-8"))
        edges: dict = data.get("edges") or {}
        if not edges:
            return ""
        stalled = sorted(
            n for n, e in edges.items() if not e.get("transfers")
        )
        lines = [
            f"STALL AUTOPSY (per-edge handshake activity, sampled at cycle "
            f"{data.get('cycle')}):"
        ]
        if stalled:
            lines.append(
                "  ZERO-TRANSFER edges (likely stall point / first unwired "
                "handshake): " + ", ".join(stalled)
            )
        def _order(item):
            # zero-transfer edges first, then by staleness of last transfer
            name, e = item
            return (0 if not e.get("transfers") else 1,
                    e.get("last_transfer_cycle", -1))
        for name, e in sorted(edges.items(), key=_order)[:max_edges]:
            lines.append(
                f"  edge {name} [{e.get('kind')}]: "
                f"transfers={e.get('transfers')}, "
                f"last_transfer@{e.get('last_transfer_cycle')}, "
                f"valid_cycles={e.get('valid_cycles')}, "
                f"ready_cycles={e.get('ready_cycles')}, "
                f"now v={e.get('valid_now')}/r={e.get('ready_now')}"
            )
        if len(edges) > max_edges:
            lines.append(f"  ... ({len(edges) - max_edges} more edges)")
        fsm = data.get("fsm") or {}
        if fsm:
            lines.append(
                "  FSM states at last sample: "
                + ", ".join(f"{k}={v}" for k, v in list(fsm.items())[:16])
            )
        return "\n".join(lines)
    except Exception:  # noqa: BLE001
        return ""


def _attach_stall_autopsy(project_root: str, violations: list[dict]) -> list[dict]:
    """Attach the stall-autopsy summary to the first violation (best-effort)."""
    if not violations:
        return violations
    summary = _stall_autopsy_summary(project_root)
    if summary:
        try:
            violations[0]["stall_autopsy"] = summary
            violations[0]["suggested_fix"] = (
                str(violations[0].get("suggested_fix", "")) + "\n\n" + summary
            )
        except Exception:  # noqa: BLE001
            pass
    return violations


_SIGNATURE_ERROR_MARKERS = (
    "unexpected keyword argument",
    "positional argument",
    "missing 1 required",
    "missing required",
    "takes",  # "takes N positional arguments but M were given"
)


def _signature_feedback(models_dir, exc_text: str) -> str:
    """R2: exact block signatures appended to signature-shaped failures.

    When a gate failure's exception text looks like a call-signature mismatch
    (the Gemini ``qp_data`` class -- 4 blind retries until the exact signature
    was handed over), attach every block factory's REAL signature so the next
    regeneration can fix the wiring without archaeology. Returns "" when the
    failure is not signature-shaped or no signatures are extractable.
    """
    low = (exc_text or "").lower()
    if not any(m in low for m in _SIGNATURE_ERROR_MARKERS):
        return ""
    try:
        from orchestrator.architecture.composition_audit import (
            block_signature_appendix,
        )

        appendix = block_signature_appendix(models_dir)
    except Exception:  # noqa: BLE001 - feedback enrichment must never break the gate
        return ""
    return ("\n\n" + appendix) if appendix else ""


def _run_simulate_with_timeout(simulate, stimulus, timeout_s: float):
    """Run ``simulate(stimulus)`` in a daemon thread with a HARD timeout.

    A generated ``simulate()`` that waits unbounded for an output frame-end
    (``tlast``) the chip never asserts -- or a chip that free-runs -- would
    otherwise HANG the gate and the daemon. Returns ``(result, timed_out, exc)``.
    On timeout the worker is a daemon thread, so it cannot block process exit.
    """
    import threading

    box: dict = {}

    def _run():
        try:
            box["result"] = simulate(stimulus)
        except Exception as exc:  # noqa: BLE001
            # Stash the traceback ON the exception: a bare "IndexError: list
            # index out of range" is unlocalizable and forces the gate into a
            # broadcast re-spec of every block; the traceback names the
            # crashing block model file:line (targeted revise instead).
            import traceback as _tb

            try:
                exc._cs_tb = _tb.format_exc()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001 - some exc types are frozen
                pass
            box["exc"] = exc

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        return None, True, None
    return box.get("result"), False, box.get("exc")


# Runner executed by the (optionally PyPy) sim subprocess. Re-imports the chip
# model + its sibling block models from ``models_dir`` and runs simulate(),
# round-tripping stimulus/result as pickle protocol 2 (CPython<->PyPy compat).
_SIM_RUNNER_SRC = r'''
import sys, pickle, importlib.util, traceback
model_path, models_dir, in_path, out_path = sys.argv[1:5]
if models_dir and models_dir not in sys.path:
    sys.path.insert(0, models_dir)
out = {}
try:
    with open(in_path, "rb") as f:
        stim = pickle.load(f)
    spec = importlib.util.spec_from_file_location("_coresmith_chip_model", model_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sim = getattr(mod, "simulate", None)
    if sim is None or not callable(sim):
        raise RuntimeError("_chip_model.simulate missing / not callable")
    out = {"ok": True, "result": sim(stim)}
except BaseException as e:  # report everything; never hang
    out = {"ok": False, "exc": "%s: %s" % (type(e).__name__, e),
           "tb": traceback.format_exc()}
with open(out_path, "wb") as f:
    pickle.dump(out, f, protocol=2)
'''


def _run_simulate_subprocess(chip_model_path, models_dir, stimulus,
                             timeout_s: float, interpreter: str):
    """Run ``simulate(stimulus)`` in a subprocess under ``interpreter``.

    Set ``CORESMITH_SIM_PYTHON=pypy3`` to JIT the pure-Python model sim (the
    full-QCIF codec sim was ~430 s on CPython) WITHOUT running the whole
    CPython daemon (FastAPI/pydantic/numpy/otel -- slow under PyPy's cpyext)
    under PyPy. The subprocess timeout actually KILLS a runaway sim, unlike the
    un-interruptible daemon thread. Returns ``(result, timed_out, exc)``.
    """
    import pickle as _pickle
    import subprocess as _sp
    import tempfile as _tf

    with _tf.TemporaryDirectory(prefix="cs_sim_") as td:
        tdp = Path(td)
        in_path, out_path, runner = tdp / "stim.pkl", tdp / "res.pkl", tdp / "runner.py"
        try:
            in_path.write_bytes(_pickle.dumps(stimulus, protocol=2))
        except Exception as exc:  # noqa: BLE001 - non-picklable stimulus
            return None, False, RuntimeError(
                f"stimulus not picklable for subprocess sim: {exc}")
        runner.write_text(_SIM_RUNNER_SRC)
        try:
            proc = _sp.run(
                [interpreter, str(runner), str(chip_model_path),
                 str(models_dir), str(in_path), str(out_path)],
                timeout=timeout_s, capture_output=True,
            )
        except _sp.TimeoutExpired:
            return None, True, None
        except FileNotFoundError:
            return None, False, RuntimeError(
                f"CORESMITH_SIM_PYTHON interpreter not found: {interpreter!r}")
        if not out_path.exists():
            err = (proc.stderr or b"")[-800:].decode("utf-8", "replace")
            return None, False, RuntimeError(
                f"sim subprocess produced no result (rc={proc.returncode}): {err}")
        try:
            out = _pickle.loads(out_path.read_bytes())
        except Exception as exc:  # noqa: BLE001
            return None, False, RuntimeError(f"sim result unpickle failed: {exc}")
        if not out.get("ok"):
            err = RuntimeError(out.get("exc", "sim error"))
            # The runner captured the full traceback -- keep it (see the
            # thread-mode path: without it a crash is unlocalizable and the
            # gate broadcasts a re-spec to every block).
            try:
                err._cs_tb = out.get("tb", "")  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
            return None, False, err
        return out.get("result"), False, None


def _dispatch_simulate(simulate, chip_model_path, models_dir, stimulus,
                       timeout_s: float):
    """In-process thread by default; a subprocess under ``CORESMITH_SIM_PYTHON``
    (e.g. ``pypy3``) when set -- so the pure-Python model sim gets PyPy's JIT
    while the daemon stays on CPython."""
    interp = os.environ.get("CORESMITH_SIM_PYTHON", "").strip()
    if interp:
        return _run_simulate_subprocess(
            chip_model_path, models_dir, stimulus, timeout_s, interp)
    return _run_simulate_with_timeout(simulate, stimulus, timeout_s)


def _run_fidelity_tier(
    project_root: str, expected: Any, observed: Any, violations: list[dict]
) -> dict | None:
    """Fidelity tier of the composition gate (microarchitecture step 2).

    Scores the composed output against the declared metric + budget, records the
    derate ledger, and appends a violation ONLY when the result is below budget
    (or byte-mismatched with no usable score). A within-budget derate PASSES and
    is recorded; a derate above the ``escalate_floor`` passes but is flagged
    ``escalate`` in the ledger for chip-lead sign-off (consumed downstream by the
    integration agent / escalation, step 3).
    """
    ok_exact, _ = _gate_match(observed, expected)
    fid = compute_fidelity_derate(project_root, expected, observed)
    write_derate_ledger(project_root, fid, byte_exact=ok_exact)
    if ok_exact:
        return fid
    if fid is None or not fid.get("within_budget", False):
        violations.append(
            {
                "type": "model_integration_failure",
                "first_divergence_block": _first_divergence_block(project_root),
                "expected": expected,
                "observed": observed,
                # A below-budget FIDELITY score is by definition a CONTENT
                # failure: the composition wired and produced a scoreable
                # output; what's wrong is a block's math (armC live: the
                # 242B/11.85dB quant-transcription drop was classified
                # 'contract' by the shape heuristic -- 242!=133 length -- and
                # routed to broadcast/END instead of a targeted block re-spec,
                # twice flagged by the run drivers).
                "gap_class": "block_math",
                "criterion": "fidelity_below_budget",
                "fidelity": fid,
                "suggested_fix": (
                    "The integrated chip model's output is BELOW the declared "
                    "fidelity budget (CORESMITH_FIDELITY_GATE). Measured "
                    f"{fid.get('measured') if fid else 'n/a'} vs floor "
                    f"{fid.get('floor') if fid else 'n/a'} "
                    f"({fid.get('direction') if fid else ''}-is-better). Inspect "
                    "the diverging block's transcribed math and the "
                    "_chip_model.py wiring -- do NOT relax the budget to pass a "
                    "broken design."
                ),
            }
        )
    return fid


def _acceptance_stimulus_path(project_root: str):
    """Resolve the FRD acceptance-stimulus artifact ('' if none declared).

    Order: CORESMITH_ACCEPTANCE_STIMULUS env (a .py exposing module-level
    ``stimulus`` -- or ``cases``, a list of (name, stimulus) tuples) ->
    <root>/inputs/acceptance_stimulus.py -> <root>/arch/acceptance_stimulus.py.
    """
    envp = os.environ.get("CORESMITH_ACCEPTANCE_STIMULUS", "").strip()
    if envp and Path(envp).exists():
        return envp
    for cand in ("inputs/acceptance_stimulus.py", "arch/acceptance_stimulus.py"):
        c = Path(project_root) / cand
        if c.exists():
            return str(c)
    return ""


def full_model_dv_enabled() -> bool:
    """Full Model DV (uarch-stage tier 2): default ON, honest-skip without an
    acceptance artifact. CORESMITH_FULL_MODEL_DV=0 disables."""
    return os.environ.get(
        "CORESMITH_FULL_MODEL_DV", "1"
    ).strip().lower() not in ("0", "false", "no", "off")


def _run_full_model_dv(
    *,
    project_root: str,
    entry_callable: Callable | None,
    entry_name: str,
    simulate: Callable,
    chip_model_path: Path,
    models_dir: Path,
) -> list[dict]:
    """FULL MODEL DV [dv-hardening-14]: mission-scale model-vs-golden.

    The armC lesson: the fast composition gate certified a 30dB floor on a
    single-pixel_block stimulus (no cascade depth); real textured full-frame
    content landed at ~21-23dB and no downstream gate could see it
    (integration DV's oracle IS the model). This tier re-runs the SAME
    model-vs-golden comparison on the FRD's mission-scale acceptance stimulus
    (full frame / full audio segment / full program -- domain-generic: the
    artifact defines it) with its own ELASTIC budget (default 1200s, x1.5 on
    prior timeout, cap via CORESMITH_FULL_MODEL_DV_TIMEOUT_CAP_S). Pure
    Python+Amaranth -- no RTL sim -- so PyPy (CORESMITH_SIM_PYTHON) applies.

    HONEST SKIP (logged, no violation) when: tier disabled, no acceptance
    artifact, artifact unloadable, or reference not invokable with it.
    A timeout or divergence IS a violation (criterion full_model_dv_*).
    """
    if not full_model_dv_enabled():
        logger.info("full model dv: disabled -- skip")
        return []
    art = _acceptance_stimulus_path(project_root)
    if not art:
        logger.info(
            "full model dv: no acceptance stimulus artifact "
            "(CORESMITH_ACCEPTANCE_STIMULUS / inputs/acceptance_stimulus.py) "
            "-- SKIPPED-HONEST. The FRD acceptance-test mandate requires one "
            "for new designs."
        )
        return []
    try:
        mod = _import_module_from_path(Path(art), "_cs_acceptance_stimulus")
    except Exception as exc:  # noqa: BLE001
        logger.warning("full model dv: acceptance artifact %s failed to "
                       "import (%s) -- SKIPPED-HONEST", art, exc)
        return []
    cases = getattr(mod, "cases", None)
    if not cases:
        stim = getattr(mod, "stimulus", None)
        if stim is None:
            logger.warning("full model dv: %s exposes neither `cases` nor "
                           "`stimulus` -- SKIPPED-HONEST", art)
            return []
        cases = [("acceptance", stim)]

    _cap = float(os.environ.get(
        "CORESMITH_FULL_MODEL_DV_TIMEOUT_CAP_S", "3600") or 3600)
    _base = float(os.environ.get(
        "CORESMITH_FULL_MODEL_DV_TIMEOUT_S", "1200") or 1200)
    _state = Path(project_root) / ".coresmith" / "full_model_dv_timeout_state.json"
    _prior = 0
    try:
        import json as _json
        if _state.exists():
            _prior = int(_json.loads(_state.read_text()).get("timeouts", 0))
    except Exception:  # noqa: BLE001
        _prior = 0
    budget = min(_base * (1.5 ** _prior), _cap)

    violations: list[dict] = []
    for name, stim in list(cases)[:32]:
        try:
            expected = _run_reference(entry_callable, stim, reraise=True)
        except ReferenceInvocationError as exc:
            logger.info("full model dv: case %r skipped -- reference not "
                        "invokable: %s", name, exc)
            continue
        if expected is None:
            continue
        _arm_stall_autopsy(project_root)
        observed, timed_out, sim_exc = _dispatch_simulate(
            simulate, chip_model_path, models_dir, stim, budget,
        )
        if timed_out:
            try:
                import json as _json
                _state.parent.mkdir(parents=True, exist_ok=True)
                _state.write_text(_json.dumps({"timeouts": _prior + 1}))
            except Exception:  # noqa: BLE001
                pass
            violations.append(_attach_stall_autopsy(project_root, [{
                "type": "model_integration_failure",
                "criterion": "full_model_dv_timeout",
                "stimulus_tier": "acceptance",
                "acceptance_case": str(name),
                "first_divergence_block": _first_divergence_block(project_root),
                "expected": expected,
                "observed": f"full-model-dv simulate() did not return within "
                            f"{budget:.0f}s on acceptance case {name!r}",
                "gap_class": "contract",
                "suggested_fix": (
                    "The composed model cannot process the mission-scale "
                    "acceptance stimulus within the Full Model DV budget. "
                    "Next attempt auto-extends (x1.5, cap "
                    f"{_cap:.0f}s); use CORESMITH_SIM_PYTHON=pypy3 to JIT the "
                    "Amaranth sim if not already."
                ),
            }])[0])
            break
        if sim_exc is not None:
            _tb_text = str(getattr(sim_exc, "_cs_tb", "") or "")
            _tb_block, _tb_frame = _localize_from_traceback(models_dir, _tb_text)
            v = {
                "type": "model_integration_failure",
                "criterion": "full_model_dv_error",
                "stimulus_tier": "acceptance",
                "acceptance_case": str(name),
                "first_divergence_block": (
                    _tb_block if _tb_block and _tb_block != "_chip_model"
                    else _first_divergence_block(project_root)),
                "expected": expected,
                "observed": f"simulate() raised {type(sim_exc).__name__}: "
                            f"{sim_exc}"
                            + (f" (at {_tb_frame})" if _tb_frame else ""),
                "gap_class": "block_math",
                "suggested_fix": (
                    "The composed model crashed on the mission-scale "
                    "acceptance stimulus (it passed the fast gate stimulus -- "
                    "a scale/content-dependent path). "
                    + (f"Crash is inside {_tb_block!r} ({_tb_frame}). "
                       if _tb_block and _tb_block != "_chip_model" else "")
                ) + _tb_text[-800:],
            }
            if _tb_block and _tb_block != "_chip_model":
                v["affected_blocks"] = [_tb_block]
            violations.append(_attach_stall_autopsy(project_root, [v])[0])
            break
        observed_out, _cyc = _split_observed(observed)
        observed_norm = _normalize_ref_output(observed_out)
        if fidelity_gate_enabled() and resolve_fidelity_metric(project_root) is not None:
            before = len(violations)
            _run_fidelity_tier(project_root, expected, observed_norm, violations)
            for v in violations[before:]:
                v["criterion"] = "full_model_dv_below_budget"
                v["stimulus_tier"] = "acceptance"
                v["acceptance_case"] = str(name)
                v["gap_class"] = "block_math"
            if len(violations) > before:
                break
        else:
            ok, _kind = _gate_match(observed_norm, expected)
            if not ok:
                violations.append({
                    "type": "model_integration_failure",
                    "criterion": "full_model_dv_divergence",
                    "stimulus_tier": "acceptance",
                    "acceptance_case": str(name),
                    "first_divergence_block": _first_divergence_block(project_root),
                    "expected": expected,
                    "observed": observed_norm,
                    "gap_class": "block_math",
                    "suggested_fix": (
                        "The composed model diverges from the golden on the "
                        "mission-scale acceptance stimulus while passing the "
                        "fast gate stimulus -- a scale/content-dependent "
                        "divergence (cascade class). Localize via the first "
                        "divergence offset and re-spec the owning block."
                    ),
                })
                break
    if not violations:
        logger.info("full model dv: PASSED (%d acceptance case(s))",
                    min(len(list(cases)), 32))
    return violations


def _mark_gate_info(
    result_info: dict | None,
    *,
    skipped: bool,
    reason: str = "",
    checked_vectors: int = 0,
) -> None:
    """Audit F2: record whether the gate actually CHECKED anything.

    The gate returns a violations list where ``[]`` is indistinguishable
    between "checked and clean" and "never ran a comparison" -- the encoder
    CLI printed ``composed model == golden byte-exact`` after a reference
    import failure no-op. Callers pass ``result_info={}`` and get back
    ``{skipped, reason, checked_vectors}`` so a no-op can be reported as
    SKIP, never PASS.
    """
    if result_info is None:
        return
    result_info["skipped"] = skipped
    result_info["reason"] = reason
    result_info["checked_vectors"] = checked_vectors


def run_model_integration_gate(
    project_root: str, result_info: dict | None = None
) -> list[dict]:
    """Run the deterministic model-integration gate.

    Returns a list of violation dicts (empty == pass). A violation dict has:
    ``type`` (``"model_integration_failure"`` for a functional/bit-exact gap,
    ``"model_integration_throughput"`` for a cycle-budget gap),
    ``first_divergence_block``, ``expected``, ``observed``, ``gap_class``
    (``"block_math"`` vs ``"contract"``), ``suggested_fix``, and a
    ``result_summary`` ``{passed, functional, throughput_ok, observed_cycles}``.

    Criterion: FUNCTIONAL + THROUGHPUT by default (``CORESMITH_BIT_EXACT`` off);
    strict bytewise ``==`` reference when ``CORESMITH_BIT_EXACT`` is on.

    No-op (returns ``[]`` with a logged reason) when:
    - the feature flag is off, OR
    - there is no ``arch/block_models`` dir, OR
    - there is no reference implementation, OR
    - there is no ``arch/block_models/_chip_model.py`` (the integration agent
      hasn't run yet).

    Never raises (so it cannot crash the pipeline), but an UNEXPECTED internal
    error is reported as a VIOLATION -- not a silent no-op. A gate that crashed
    did not validate the decomposition, so it must park for the outer agent, not
    wave the run through. (The only no-ops are the deliberate "nothing to do"
    cases inside ``_run_gate_inner``: flag off, no models, no chip model.)
    """
    if not block_goldens_enabled():
        logger.info("model integration gate: CORESMITH_BLOCK_GOLDENS off -- no-op")
        _mark_gate_info(result_info, skipped=True,
                        reason="CORESMITH_BLOCK_GOLDENS off")
        return []

    try:
        return _sanitize_violations(
            _run_gate_inner(project_root, result_info=result_info)
        )
    except Exception as exc:  # noqa: BLE001 - gate must never crash the pipeline
        import traceback
        tb = traceback.format_exc()
        logger.warning(
            "model integration gate: unexpected internal error -- reporting as "
            "violation (NOT a pass): %s: %s\n%s",
            type(exc).__name__,
            exc,
            tb,
        )
        _mark_gate_info(result_info, skipped=False,
                        reason=f"gate internal error: {type(exc).__name__}")
        return [
            {
                "type": "model_integration_failure",
                "first_divergence_block": _first_divergence_block(project_root),
                "expected": "",
                "observed": f"gate internal error: {type(exc).__name__}: {exc}",
                "criterion": "gate_internal_error",
                "gap_class": "block_math",
                "suggested_fix": (
                    "The model-integration gate raised an unexpected error and "
                    "could NOT validate the decomposition. This is not a pass. "
                    f"Traceback:\n{tb}"
                ),
            }
        ]


def describe_gate_status(project_root: str) -> dict:
    """Classify whether the model-integration gate is APPLICABLE for this run.

    ``run_model_integration_gate`` returns ``[]`` (no violations) BOTH when it
    actually simulated the composed chip model and it matched the reference (a
    real PASS) AND when it never ran a comparison at all (a no-op SKIP: the
    two-golden flag is off, or there are no block models / no ``_chip_model.py``
    / no resolvable golden reference). An empty result is therefore a real PASS
    **only** when this reports ``applicable=True``. Otherwise the run is
    requirements-only / goldenless and the empty result must be recorded as
    SKIPPED-HONEST -- never reported as ``passed=True`` (rung2 defect 1: the
    mcu3 goldenless run had no ``inputs/golden.py`` -> pass-1
    ``_maybe_generate_block_golden`` skipped every block (it requires a
    resolvable golden / ``python_source``) -> ``arch/block_models/`` was never
    created -> the gate vacuously "passed" in 4ms).

    Cheap (path + glob checks + the golden-path resolver -- no imports/sim).

    Returns a dict::

        {"applicable": bool, "reason": str,
         "block_goldens_enabled": bool, "block_models_present": bool,
         "chip_model_present": bool, "reference_resolvable": bool}
    """
    root = Path(project_root)
    flag = block_goldens_enabled()
    models_dir = root / "arch" / BLOCK_MODELS_DIRNAME
    block_models_present = bool(
        models_dir.is_dir()
        and any(
            p
            for p in models_dir.glob("*.py")
            if p.name != CHIP_MODEL_FILENAME and not p.name.startswith("__")
        )
    )
    chip_model_present = (models_dir / CHIP_MODEL_FILENAME).exists()
    reference_resolvable = bool(resolve_reference_implementation(project_root))

    applicable = bool(
        flag
        and block_models_present
        and chip_model_present
        and reference_resolvable
    )

    if not flag:
        reason = (
            "CORESMITH_BLOCK_GOLDENS off; two-golden model-integration gate "
            "disabled (nothing to compose)"
        )
    elif applicable:
        reason = (
            "gate applicable: block models + _chip_model.py + golden reference "
            "all present"
        )
    else:
        missing = []
        if not reference_resolvable:
            missing.append(
                "no *_golden.py / CORESMITH_SOURCE_ROOT reference resolvable"
            )
        if not block_models_present:
            missing.append(
                "no block models in arch/block_models/ (pass-1 block-golden "
                "generation skips when no golden reference / python_source is "
                "available)"
            )
        if not chip_model_present:
            missing.append(
                "no arch/block_models/_chip_model.py (integration agent never "
                "stitched a composed model)"
            )
        # Lead with the canonical phrase so the outcome reads honestly.
        reason = "no golden reference; gate not applicable"
        if missing:
            reason += " (" + "; ".join(missing) + ")"

    return {
        "applicable": applicable,
        "reason": reason,
        "block_goldens_enabled": flag,
        "block_models_present": block_models_present,
        "chip_model_present": chip_model_present,
        "reference_resolvable": reference_resolvable,
    }


def _run_gate_inner(
    project_root: str, result_info: dict | None = None
) -> list[dict]:
    root = Path(project_root)

    models_dir = root / "arch" / BLOCK_MODELS_DIRNAME
    if not models_dir.is_dir() or not any(models_dir.glob("*.py")):
        logger.info(
            "model integration gate: no block models at %s -- no-op", models_dir
        )
        _mark_gate_info(result_info, skipped=True,
                        reason=f"no block models at {models_dir}")
        return []

    chip_model_path = models_dir / CHIP_MODEL_FILENAME
    if not chip_model_path.exists():
        logger.info(
            "model integration gate: no %s -- no-op (integration agent not run)",
            chip_model_path,
        )
        _mark_gate_info(result_info, skipped=True,
                        reason="no _chip_model.py (integration agent not run)")
        return []

    ref_path = resolve_reference_implementation(project_root)
    if not ref_path:
        logger.info(
            "model integration gate: no reference implementation -- no-op"
        )
        _mark_gate_info(result_info, skipped=True,
                        reason="no reference implementation resolvable")
        return []

    # Resolve the reference module + its callable entry point (the oracle).
    try:
        ref_module = _load_reference_module(ref_path)
    except Exception as exc:  # noqa: BLE001
        # Audit F2: a reference that RESOLVES but cannot be IMPORTED is a
        # broken oracle, not a goldenless run -- swallowing it as a no-op is
        # how the encoder CLI printed "composed model == golden byte-exact"
        # after "No module named 'reference_codec_vectors'". Report a violation.
        logger.warning(
            "model integration gate: reference import failed: %s -- violation "
            "(NOT a pass)", exc
        )
        _mark_gate_info(result_info, skipped=False,
                        reason=f"reference import failed: {exc}")
        return [
            {
                "type": "model_integration_failure",
                "first_divergence_block": "",
                "expected": "",
                "observed": f"reference {ref_path!r} failed to import: "
                            f"{type(exc).__name__}: {exc}",
                "criterion": "reference_import_failed",
                "gap_class": "contract",
                "suggested_fix": (
                    "The resolvable golden reference could not be imported, so "
                    "the gate has NO oracle and cannot validate the "
                    "decomposition. Sibling modules next to the reference are "
                    "importable (its directory is on sys.path during import) "
                    "-- fix the reference file or its imports. This is NOT a "
                    "pass."
                ),
            }
        ]

    entry_callable, entry_name = resolve_reference_entrypoint(
        project_root, ref_module
    )
    if entry_callable is None:
        logger.info(
            "model integration gate: no callable reference entry -- no-op"
        )
        _mark_gate_info(result_info, skipped=True,
                        reason="no callable reference entry point")
        return []
    logger.info("model integration gate: reference oracle entry = %s", entry_name)

    # Obtain the stimulus: explicit env file wins; else auto-discovered
    # inputs/model_stimulus.py (finding #5); else a small derived default.
    stimulus, found = _load_env_stimulus(project_root)
    if not found:
        stimulus = _default_stimulus(entry_callable)
        if stimulus is None:
            logger.info(
                "model integration gate: could not derive a default stimulus "
                "and CORESMITH_MODEL_STIMULUS unset -- no-op"
            )
            _mark_gate_info(result_info, skipped=True,
                            reason="no stimulus (CORESMITH_MODEL_STIMULUS "
                                   "unset, no derivable default)")
            return []
        logger.info(
            "model integration gate: using derived default stimulus %r", stimulus
        )

    # STATIC COMPOSITION AUDIT (pre-sim). Catches the mechanical wiring-defect
    # classes (bad kwargs, undriven nets, zero-width Signals, .symdict wiring)
    # in milliseconds WITH actionable feedback, instead of burning a full
    # simulate-and-diff round to discover them (the 2026-07 video_codec A/B burned 10+
    # attempts across two worker models on exactly these). Runs after all the
    # no-op early-returns above so gate no-op semantics are unchanged.
    from orchestrator.architecture.composition_audit import audit_violations

    audit_viols = audit_violations(chip_model_path, models_dir)
    if audit_viols:
        logger.info(
            "model integration gate: static composition audit found %d "
            "violation(s) -- skipping simulation (fix the wiring first)",
            len(audit_viols),
        )
        _mark_gate_info(result_info, skipped=False,
                        reason="static composition audit violations")
        return audit_viols

    # Import the integrated chip model + its simulate(). The chip model imports
    # its sibling block-model modules by plain name, so put the models dir on
    # sys.path for the duration of the import.
    import sys

    chip_mod = None
    inserted = False
    if str(models_dir) not in sys.path:
        sys.path.insert(0, str(models_dir))
        inserted = True
    # Purge any cached sibling block-model modules (and a prior chip model) so a
    # re-run / a different project picks up THIS dir's files rather than a stale
    # same-named module cached by an earlier import. The block models are
    # imported by plain name (e.g. `from add1 import add1`), so a same-named
    # module from another run would otherwise shadow them.
    _block_stems = {
        p.stem
        for p in models_dir.glob("*.py")
        if p.name != CHIP_MODEL_FILENAME and not p.name.startswith("__")
    }
    for _stem in _block_stems | {"_coresmith_chip_model"}:
        sys.modules.pop(_stem, None)
    try:
        chip_mod = _import_module_from_path(
            chip_model_path, "_coresmith_chip_model"
        )
    except Exception as exc:  # noqa: BLE001
        return [
            {
                "type": "model_integration_failure",
                "first_divergence_block": _first_divergence_block(project_root),
                "expected": "",
                "observed": f"_chip_model import raised {type(exc).__name__}: {exc}",
                "suggested_fix": (
                    "arch/block_models/_chip_model.py does not import; the "
                    "model-integration agent emitted a broken chip model. "
                    "Regenerate it (revise_uarch) or fix the wiring on disk."
                    + _signature_feedback(models_dir, str(exc))
                ),
            }
        ]
    finally:
        # The sibling block-model modules are now cached in sys.modules, so the
        # path entry is no longer needed for simulate().
        if inserted:
            try:
                sys.path.remove(str(models_dir))
            except ValueError:
                pass

    simulate = getattr(chip_mod, "simulate", None)
    if simulate is None or not callable(simulate):
        return [
            {
                "type": "model_integration_failure",
                "first_divergence_block": _first_divergence_block(project_root),
                "expected": "",
                "observed": "_chip_model.simulate is missing / not callable",
                "suggested_fix": (
                    "arch/block_models/_chip_model.py must expose a module-level "
                    "`def simulate(stimulus) -> observed`."
                ),
            }
        ]

    # Compute expected (reference) and observed (integrated Amaranth chip model).
    # A reference that CRASHES (wrong stimulus type, bad signature) provides no
    # oracle -- it must be a HARD failure, never a silent pass. _run_reference
    # with reraise=True raises ReferenceInvocationError on any call failure.
    try:
        expected = _run_reference(entry_callable, stimulus, reraise=True)
    except ReferenceInvocationError as exc:
        return [
            {
                "type": "model_integration_failure",
                "first_divergence_block": _first_divergence_block(project_root),
                "expected": "",
                "observed": f"reference entry {entry_name!r} could not be "
                            f"invoked: {exc}",
                "criterion": "reference_uninvokable",
                "gap_class": "contract",
                "suggested_fix": (
                    f"The reference entry point {entry_name!r} raised when "
                    f"called with the gate stimulus (type "
                    f"{type(stimulus).__name__}). The gate has NO oracle and "
                    "cannot validate the decomposition. Provide a stimulus that "
                    "matches the reference signature via CORESMITH_MODEL_STIMULUS "
                    "(a .py exposing module-level `stimulus`), and ensure the "
                    "integrated chip model's simulate(stimulus) accepts the same "
                    "object. This is NOT a pass."
                ),
            }
        ]
    if expected is None:
        logger.info(
            "model integration gate: reference returned None for stimulus "
            "-- no-op"
        )
        _mark_gate_info(result_info, skipped=True,
                        reason="reference returned None for the gate stimulus")
        return []

    # From here on the gate HAS an oracle and runs a real comparison: any
    # return below is a checked verdict, not a skip.
    _mark_gate_info(result_info, skipped=False, checked_vectors=1,
                    reason="default-tier stimulus checked")

    sim_timeout = float(os.environ.get("CORESMITH_GATE_SIM_TIMEOUT", "180") or 180)
    _arm_stall_autopsy(project_root)
    raw_observed, timed_out, sim_exc = _dispatch_simulate(
        simulate, chip_model_path, models_dir, stimulus, sim_timeout
    )
    if timed_out:
        return _attach_stall_autopsy(project_root, [
            {
                "type": "model_integration_failure",
                "first_divergence_block": _first_divergence_block(project_root),
                "expected": expected,
                "observed": f"simulate() did not return within {sim_timeout:.0f}s "
                            "(no output frame-end / runaway emission / deadlock)",
                "criterion": "simulate_timeout",
                "gap_class": "contract",
                "suggested_fix": (
                    "The integrated chip model's simulate() ran unbounded. Usually "
                    "the pipeline never asserts the output frame-end (m_axis_tlast) "
                    "so it free-runs: ensure every block PROPAGATES tlast and bounds "
                    "emission to the frame, and that simulate() has a hard cycle cap "
                    "(see prompt). Raise CORESMITH_GATE_SIM_TIMEOUT for a legitimately "
                    "long sim."
                ),
            }
        ])
    if sim_exc is not None:
        # The harness's EOF-witness assertion is a drain/EOF CONTRACT failure
        # (tlast not threaded to egress), not a math divergence -- route it so
        # localization/revise treats it as a contract gap.
        _eof_class = "EOF witness" in str(sim_exc)
        # Localize the crash from its traceback (captured by both dispatch
        # paths as exc._cs_tb): the deepest frame inside a block model names
        # the block -> TARGETED revise; a bare exception message forced a
        # broadcast re-spec of all blocks (observed live on armC: IndexError
        # with no location -> "unlocalized -> broadcast re-spec").
        _tb_text = str(getattr(sim_exc, "_cs_tb", "") or "")
        _tb_block, _tb_frame = _localize_from_traceback(models_dir, _tb_text)
        _tb_tail = ""
        if _tb_text:
            _tb_tail = "\n\nCrash traceback (tail):\n" + "\n".join(
                _tb_text.strip().splitlines()[-14:]
            )
        _fdb = _first_divergence_block(project_root)
        _ctx_gap = ""
        if _tb_block and _tb_block != "_chip_model":
            _fdb = _tb_block
            # Context-fidelity flag: if this block PASSED its own DV yet the
            # composed sim crashes inside it, its testbench never exercised
            # the crashing path while the composed context does (DV-closure
            # mechanism 2). Saying so redirects the fix: repair the block AND
            # note the stimulus hole -- observed live on armC (mode-3
            # neighbor-extension overrun invisible to the block TB).
            try:
                import json as _json

                _br = (Path(project_root) / ".coresmith" / "blocks"
                       / _tb_block / "best_result.json")
                if _br.exists() and _json.loads(
                    _br.read_text(encoding="utf-8")
                ).get("sim_passed"):
                    _ctx_gap = (
                        f"NOTE: {_tb_block!r} PASSED its own block DV -- its "
                        "testbench never exercises the crashing path, while "
                        "the composed chip context does (a block/integration "
                        "stimulus-parity gap). Fix the block's math AND "
                        "consider what stimulus class its DV was missing. "
                    )
            except Exception:  # noqa: BLE001 - enrichment only
                _ctx_gap = ""
        _viol = {
            "type": "model_integration_failure",
            "first_divergence_block": _fdb,
            "expected": expected,
            "observed": f"simulate() raised {type(sim_exc).__name__}: {sim_exc}"
                        + (f" (at {_tb_frame})" if _tb_frame else ""),
            "gap_class": "contract" if (_eof_class or _tb_block == "_chip_model")
                         else "block_math",
            "suggested_fix": (
                "The integrated Amaranth chip model raised during simulation. "
                + (
                    f"The crash is INSIDE block model {_tb_block!r} "
                    f"({_tb_frame}) -- fix that block's model; the wiring "
                    "delivered its inputs. " + _ctx_gap
                    if _tb_block and _tb_block != "_chip_model"
                    else (
                        f"The crash is in the composition wiring "
                        f"(_chip_model.py {_tb_frame}) -- fix the glue, not "
                        "the block models. "
                        if _tb_block == "_chip_model"
                        else "Inspect arch/block_models/_chip_model.py wiring "
                             "/ handshake and the per-block models it "
                             "instantiates. "
                    )
                )
                + _signature_feedback(models_dir, str(sim_exc))
                + _tb_tail
            ),
        }
        if _tb_block and _tb_block != "_chip_model":
            # consumed by the daemon's targeted-revise router
            _viol["affected_blocks"] = [_tb_block]
        return _attach_stall_autopsy(project_root, [_viol])

    # Throughput: simulate() returns (output, cycles); legacy 1-value returns
    # tolerated (cycles=None -> throughput check skipped, R6).
    observed_out, observed_cycles = _split_observed(raw_observed)
    observed_norm = _normalize_ref_output(observed_out)

    violations: list[dict] = []

    # --- Functional / bit-exact correctness branch -------------------------
    if bit_exact_enabled():
        functional_ok, _match_kind = _gate_match(observed_norm, expected)
        if not functional_ok:
            violations.append(
                {
                    "type": "model_integration_failure",
                    "first_divergence_block": _first_divergence_block(project_root),
                    "expected": expected,
                    "observed": observed_norm,
                    "gap_class": _classify_gap(project_root, expected, observed_norm),
                    "criterion": "bit_exact" if _match_kind == "exact"
                                 else "float_epsilon_divergence",
                    "suggested_fix": (
                        "CORESMITH_BIT_EXACT=1: the integrated chip model's "
                        "simulated output is not bytewise-equal to the reference "
                        "implementation. Inspect the per-block models in "
                        "arch/block_models/ for a block whose transcribed math "
                        "diverges, and the _chip_model.py wiring / handshake."
                    ),
                }
            )
    elif fidelity_gate_enabled() and resolve_fidelity_metric(project_root) is not None:
        # Fidelity tier (microarch step 2): quantitative score vs a declared
        # budget, with a recorded derate ledger. Distinguishes a within-budget
        # derate (PASS, recorded) from a below-budget break (FAIL) -- which the
        # binary accept/equivalence tiers below cannot.
        _run_fidelity_tier(project_root, expected, observed_norm, violations)
    else:
        # Functional default. Tier A: declared acceptance predicate.
        accept = resolve_functional_acceptance(project_root)
        if accept is not None:
            try:
                functional_ok = bool(accept(expected, observed_norm))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "model integration gate: acceptance predicate raised %s: "
                    "%s -- treating as fail", type(exc).__name__, exc,
                )
                functional_ok = False
            if not functional_ok:
                violations.append(
                    {
                        "type": "model_integration_failure",
                        "first_divergence_block": _first_divergence_block(
                            project_root
                        ),
                        "expected": expected,
                        "observed": observed_norm,
                        "gap_class": _classify_gap(
                            project_root, expected, observed_norm
                        ),
                        "criterion": "functional_declared",
                        "suggested_fix": (
                            "The declared functional-acceptance predicate "
                            "rejected the integrated chip model's output. "
                            "Inspect the named block's transcribed math and the "
                            "_chip_model.py wiring."
                        ),
                    }
                )
        elif gate_allow_nondegenerate_enabled():
            # Legacy escape hatch (opt-in): the OLD, too-weak Tier-B check that
            # accepted any NON-DEGENERATE output WITHOUT comparing it to the
            # reference. This let a composition streaming a handful of garbage
            # bytes pass (e.g. the codec that emitted ~0.5% of the reference
            # bytes yet "passed"). Kept only for designs with genuinely no usable
            # oracle comparison.
            if _is_degenerate(observed_norm):
                violations.append(
                    {
                        "type": "model_integration_failure",
                        "first_divergence_block": _first_divergence_block(
                            project_root
                        ),
                        "expected": expected,
                        "observed": observed_norm,
                        "gap_class": _classify_gap(
                            project_root, expected, observed_norm
                        ),
                        "criterion": "functional_nondegenerate",
                        "suggested_fix": (
                            "CORESMITH_GATE_ALLOW_NONDEGENERATE is set and the "
                            "integrated chip model's output is DEGENERATE "
                            "(all-zero / constant / empty). Fix the collapsed "
                            "datapath in arch/block_models/."
                        ),
                    }
                )
            else:
                logger.warning(
                    "model integration gate: CORESMITH_GATE_ALLOW_NONDEGENERATE "
                    "set -- accepting non-degenerate output %r WITHOUT checking it "
                    "against the reference (the weak legacy check). Unset it (and "
                    "declare validation_kpis acceptance_fn / "
                    "CORESMITH_FUNCTIONAL_ACCEPTANCE for lossy designs) for a "
                    "rigorous FRD-equivalence gate.",
                    observed_norm,
                )
        else:
            # Functional DEFAULT (no declared acceptance predicate): require
            # EQUIVALENCE TO THE REFERENCE (the FRD behaviour). We have a real
            # oracle here (expected is not None -- the None case no-op'd above),
            # so a non-degenerate but WRONG output must NOT pass. This is the fix
            # for the gate hole that let a composition reproducing ~0.5% of the
            # reference bytes through on mere non-degeneracy. Lossy/non-bit-exact
            # designs declare an acceptance_fn (Tier A) instead.
            _match_ok, _match_kind = _gate_match(observed_norm, expected)
            if not _match_ok:
                degenerate = _is_degenerate(observed_norm)
                violations.append(
                    {
                        "type": "model_integration_failure",
                        "first_divergence_block": _first_divergence_block(
                            project_root
                        ),
                        "expected": expected,
                        "observed": observed_norm,
                        "gap_class": _classify_gap(
                            project_root, expected, observed_norm
                        ),
                        "criterion": (
                            "functional_nondegenerate"
                            if degenerate
                            else (
                                "functional_float_epsilon_divergence"
                                if _match_kind == "float_epsilon"
                                else "functional_reference_divergence"
                            )
                        ),
                        "suggested_fix": (
                            (
                                "The integrated chip model's output is DEGENERATE "
                                "(all-zero / constant / empty) -- the composed "
                                "datapath collapsed. Fix the block models in "
                                "arch/block_models/."
                            )
                            if degenerate
                            else (detect_byte_shift(expected, observed_norm)
                                  + " ") if detect_byte_shift(expected, observed_norm)
                            else (
                                "The integrated chip model's output does NOT match "
                                "the reference (FRD behaviour) for the gate "
                                "stimulus -- the composed block models do not "
                                "implement the reference. Inspect the diverging "
                                "block's transcribed math and the _chip_model.py "
                                "wiring / framing (tlast, handshake). If the design "
                                "is INTENTIONALLY non-bit-exact (e.g. lossy), "
                                "declare an acceptance_fn in the ERS "
                                "validation_kpis or CORESMITH_FUNCTIONAL_ACCEPTANCE "
                                "(e.g. decode + PSNR>=thr). Only set "
                                "CORESMITH_GATE_ALLOW_NONDEGENERATE=1 if you accept "
                                "that the weak check can pass a wrong design."
                            )
                        ),
                    }
                )

    # --- Extra stimulus tiers (A-Fix 5a): FIXED FRD vectors + SEEDED ------
    # Anti-memorization: a chip model that hardcodes the default/fixed output
    # passes the tier above but DIVERGES under a fresh seeded stimulus. Only run
    # when the default tier PASSED functionally (a failing design already has a
    # violation) and NOT in fidelity mode (a lossy metric+ledger design would
    # false-fail a byte-equivalence seeded compare).
    _fidelity_mode = (
        fidelity_gate_enabled()
        and resolve_fidelity_metric(project_root) is not None
    )
    if not violations and not _fidelity_mode:
        violations.extend(
            _run_extra_stimulus_tiers(
                project_root=project_root,
                entry_callable=entry_callable,
                entry_name=entry_name,
                ref_module=ref_module,
                simulate=simulate,
                chip_model_path=chip_model_path,
                models_dir=models_dir,
                sim_timeout=sim_timeout,
            )
        )

    # FULL MODEL DV [dv-hardening-14] -- uarch-stage tier 2: mission-scale
    # model-vs-golden on the FRD acceptance stimulus, with its own elastic
    # budget. Runs only when everything above is clean (fast loop stays fast).
    if not violations:
        violations.extend(_run_full_model_dv(
            project_root=project_root,
            entry_callable=entry_callable,
            entry_name=entry_name,
            simulate=simulate,
            chip_model_path=chip_model_path,
            models_dir=models_dir,
        ))

    functional_passed = not violations

    # --- Throughput branch -------------------------------------------------
    throughput_ok: bool | None = None
    if observed_cycles is None:
        logger.info(
            "model integration gate: simulate() returned no cycle count "
            "(legacy 1-value return) -- skipping throughput check"
        )
    else:
        floor = resolve_throughput_floor(project_root)
        if floor is None:
            logger.info(
                "model integration gate: no throughput floor declared "
                "(validation_kpis / CORESMITH_THROUGHPUT_FLOOR_CYCLES) -- "
                "skipping throughput check"
            )
        else:
            throughput_ok = observed_cycles <= floor
            if not throughput_ok:
                violations.append(
                    {
                        "type": "model_integration_throughput",
                        "first_divergence_block": _first_divergence_block(
                            project_root
                        ),
                        "expected": f"<= {floor} cycles",
                        "observed": f"{observed_cycles} cycles",
                        "observed_cycles": observed_cycles,
                        "throughput_floor": floor,
                        "gap_class": "contract",
                        "suggested_fix": (
                            f"The integrated chip model took {observed_cycles} "
                            f"cycles, over the {floor}-cycle throughput floor "
                            "from the ERS validation_kpis. Pipeline / parallelise "
                            "the datapath in arch/block_models/ or revisit the "
                            "interface contract's handshake latency."
                        ),
                    }
                )

    # --- Per-field localization (convergence: targeted re-spec, not regen-all) -
    # When the composed output is a multi-field dict and only SOME fields diverge,
    # attribute the failure to the blocks feeding the diverging field(s) so a
    # gate-triggered re-spec touches only those blocks and KEEPS the passing
    # field's sub-chain. Populating ``affected_blocks`` flips the daemon router
    # (_gate_localization_precise) from BROADCAST (re-roll all blocks) to TARGETED.
    affected_blocks: list[str] = []
    try:
        affected_blocks = _localize_affected_blocks(
            project_root, expected, observed_norm
        )
    except Exception as exc:  # noqa: BLE001 - localization is best-effort
        logger.warning(
            "model integration gate: field localization raised %s: %s "
            "-- falling back to broadcast", type(exc).__name__, exc,
        )
        affected_blocks = []
    if affected_blocks:
        logger.info(
            "model integration gate: localized divergence to %d block(s): %s "
            "(targeted re-spec; passing-field blocks preserved)",
            len(affected_blocks), affected_blocks,
        )

    # --- Result summary ----------------------------------------------------
    # Recorded on every violation so the node/outer agent can read a uniform
    # summary regardless of which check failed.
    summary = {
        "passed": not violations,
        "functional": functional_passed,
        "throughput_ok": throughput_ok,
        "observed_cycles": observed_cycles,
    }
    for v in violations:
        v["result_summary"] = summary
        # Only attach to functional/value violations (a throughput violation is
        # not field-localizable). affected_blocks is consumed by the daemon's
        # _gate_affected_blocks() for targeted revise.
        if affected_blocks and v.get("type") == "model_integration_failure":
            v["affected_blocks"] = affected_blocks

    if not violations:
        logger.info(
            "model integration gate: PASSED (functional=%s throughput_ok=%s "
            "cycles=%s)", functional_passed, throughput_ok, observed_cycles,
        )
    # Attach the per-edge stall autopsy to divergence failures too: an
    # empty/partial output's autopsy names the first stalled handshake edge
    # (which block held ready low / never asserted valid).
    return _attach_stall_autopsy(project_root, violations)
