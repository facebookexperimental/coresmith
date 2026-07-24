"""Stimulus<->contract consistency guard (deterministic, domain-agnostic).

Runs BEFORE the frontend pipeline builds anything so a gate stimulus the
declared design contract cannot actually accept is caught in seconds, instead of
after a multi-hour arch + pass-1 run that dead-ends at the µarch integration
gate. This is the durable fix for the class of failure where the architecture
and the validation oracle (golden + stimulus) silently disagree about the input
domain -- e.g. the codec arch hardcoded a 640x360 frame while the gate fed a
16x16 in-contract frame the golden happily encodes.

Two GENERIC checks, no codec/MCU/domain knowledge:

1. ORACLE ACCEPTANCE -- the configured reference (the gate oracle) must be able
   to process the configured stimulus and return non-degenerate output. A
   reference that raises, or returns empty/constant output, means the
   stimulus+reference pair is misconfigured and every downstream gate verdict
   would be meaningless. (Catches a wrong CORESMITH_REFERENCE_ENTRY, a stimulus
   in the wrong shape/dtype, etc. -- works for codec, MCU, and pure-math
   references alike: it just calls the golden.)

2. STIMULUS-FIELD COVERAGE -- every CONFIG field the stimulus supplies (a dict
   stimulus separates the streamed payload from per-frame config such as qp / H
   / W / mode / entry_pc) must map to a declared EXTERNAL (boundary) input of
   the design. A config field with no boundary input means the design literally
   cannot receive that value at runtime -- it will hardcode or ignore it, and
   the composition can never honour it (the exact codec geometry failure). This
   is pure name reconciliation between the stimulus keys and the block diagram's
   boundary-input signal names, so it is domain-agnostic.

Severity: oracle-reject is ``error``; coverage gaps are ``warn`` (the boundary
extraction + name matching are best-effort). Behaviour is gated by
``CORESMITH_STIMULUS_CONTRACT_GUARD``: unset/``warn`` (default) logs + writes a
report and continues; ``strict`` makes the caller treat findings as blocking;
``off`` disables the guard entirely.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPORT_FILENAME = "stimulus_contract_guard.json"

# Generic aliases for the handful of cross-domain config concepts whose stimulus
# key rarely matches the RTL signal name verbatim. Deliberately small -- the
# token matcher handles the rest. Keys are normalised stimulus field names.
# NB: keep aliases to DISTINCTIVE single-concept tokens. Compound names like
# "frameheight" are deliberately excluded -- they substring-collide with common
# plumbing tokens ("frame") and cause false coverage matches.
_ALIASES: dict[str, list[str]] = {
    "h": ["height", "rows", "nrows", "numrows"],
    "height": ["h", "rows", "nrows", "numrows"],
    "w": ["width", "cols", "columns", "ncols", "numcols"],
    "width": ["w", "cols", "columns", "ncols", "numcols"],
    "qp": ["quant", "qindex", "qstep", "qscale"],
}


def guard_enabled() -> bool:
    """The guard runs unless explicitly disabled with ``...=off``."""
    return os.environ.get("CORESMITH_STIMULUS_CONTRACT_GUARD", "").strip().lower() \
        != "off"


def guard_strict() -> bool:
    """``...=strict`` makes findings blocking (default is warn-and-continue)."""
    return os.environ.get("CORESMITH_STIMULUS_CONTRACT_GUARD", "").strip().lower() \
        == "strict"


def _norm(name: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _tokenize(name: Any) -> set[str]:
    """Split a signal/interface name into normalised word tokens.

    Splits on non-alnum and camelCase boundaries: ``cfg_qp_i`` -> {cfg, qp, i};
    ``frameHeightIn`` -> {frame, height, in}. Used for word-boundary matching so
    a short field like ``h`` cannot spuriously match the ``h`` inside ``flush``.
    """
    s = str(name)
    # camelCase / digit boundaries -> spaces, then split on non-alnum.
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", s)
    parts = re.split(r"[^A-Za-z0-9]+", s)
    return {p.lower() for p in parts if p}


# Noise tokens that carry no semantic input identity (bus plumbing / direction).
_NOISE_TOKENS = {
    "i", "o", "in", "out", "s", "m", "axis", "tdata", "tvalid", "tready",
    "tlast", "tuser", "cfg", "config", "sig", "signal", "bus", "port", "if",
    "data", "valid", "ready", "clk", "rst", "reset", "n",
}


def _is_payload(value: Any) -> bool:
    """True if a stimulus field is the STREAMED PAYLOAD (exempt from coverage).

    The payload maps to the design's data-stream ingress, not a config port, so
    it is not subject to the config-field coverage check. Heuristic: anything
    array-like (numpy array, or a list/tuple longer than a small scalar-vector
    threshold, or nested) is payload; scalars and short vectors are config.
    """
    if hasattr(value, "shape") and hasattr(value, "dtype"):  # numpy-like
        return True
    if isinstance(value, (list, tuple)):
        if any(isinstance(v, (list, tuple)) or hasattr(v, "shape") for v in value):
            return True  # nested -> payload
        return len(value) > 8  # long flat vector -> payload stream
    if isinstance(value, (bytes, bytearray)):
        return len(value) > 8
    return False


def _stimulus_config_fields(stimulus: Any) -> dict[str, Any]:
    """Extract the CONFIG fields of a dict stimulus (drop the payload field(s))."""
    if not isinstance(stimulus, dict):
        return {}
    return {k: v for k, v in stimulus.items() if not _is_payload(v)}


# identifier immediately preceding a bit declaration, e.g. ``frame_h[10]`` or
# ``tdata[41:0]`` -- the robust way to pull real field/signal names out of a
# free-text ``signals``/``payload`` description without dragging in prose words
# (so ``payload_width = ...`` does NOT leak a spurious ``width`` token).
_FIELD_DECL_RE = re.compile(r"([A-Za-z]\w*)\s*\[")


def _iface_base(name: Any) -> str:
    """Strip a trailing ``_in``/``_out`` and any AXI master/slave prefix."""
    x = str(name).strip()
    low = x.lower()
    for pre in ("s_axis_", "m_axis_", "saxis_", "maxis_"):
        if low.startswith(pre):
            x = x[len(pre):]
            low = x.lower()
            break
    for suf in ("_in", "_out"):
        if low.endswith(suf):
            return x[: -len(suf)]
    return x


def _iface_is_input(iname: str, spec: Any) -> bool | None:
    """True=input, False=output, None=ambiguous. Handles _in/_out, AXI
    s_axis/m_axis naming, and pin ``*_i``/``*_o`` direction suffixes."""
    n = str(iname).lower()
    if n.endswith("_out") or n.startswith("m_axis") or n.startswith("maxis"):
        return False
    if n.endswith("_in") or n.startswith("s_axis") or n.startswith("saxis"):
        return True
    sig = (spec or {}).get("signals") if isinstance(spec, dict) else None
    if isinstance(sig, dict):
        has_i = any(str(s).lower().endswith("_i") for s in sig)
        has_o = any(str(s).lower().endswith("_o") for s in sig)
        if has_i and not has_o:
            return True
        if has_o and not has_i:
            return False
    return None


def _iface_tokens(iname: str, spec: Any) -> set[str]:
    """All field/signal tokens an interface exposes, across schema variants:
    dict ``signals`` ({name: width}), string ``signals``/``payload`` with
    ``name[width]`` field declarations, and list ``fields`` ([{name: ...}])."""
    toks = _tokenize(iname)
    if isinstance(spec, dict):
        sig = spec.get("signals")
        if isinstance(sig, dict):
            for s in sig:
                toks |= _tokenize(s)
        # Pull real field identifiers out of any free-text spec string.
        for key in ("signals", "payload", "semantic_contract", "description",
                    "fields"):
            val = spec.get(key)
            if isinstance(val, str):
                for m in _FIELD_DECL_RE.findall(val):
                    toks |= _tokenize(m)
            elif isinstance(val, list):
                for f in val:
                    if isinstance(f, dict) and f.get("name"):
                        toks |= _tokenize(f["name"])
                    elif isinstance(f, str):
                        toks |= _tokenize(f)
    return toks


def external_input_tokens(project_root: str) -> set[str] | None:
    """Tokens of the design's declared EXTERNAL (boundary) input signals.

    A boundary input is an INPUT interface that is NOT the consumer side of any
    internal block->block connection (i.e. it is driven from outside the chip).
    Schema-tolerant (dict or string ``signals``; ``_in``/``_out`` or AXI
    ``s_axis``/``m_axis`` naming; ``->``-style connection interfaces). Returns
    ``None`` when the block diagram is unavailable or exposes no recognisable
    input interface, so the caller skips the coverage check rather than emit
    false findings.
    """
    import json

    bd_path = Path(project_root) / ".coresmith" / "block_diagram.json"
    if not bd_path.exists():
        return None
    try:
        bd = json.loads(bd_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    # Internal consumer ports: the destination side of each connection. The
    # connection "interface" may be "m_axis_x -> s_axis_y" (consumer = right) or
    # a single name; normalise via _iface_base so it matches the block's port.
    conns = bd.get("connections") or []
    internal_consumed: set[tuple[str, str]] = set()
    for c in conns:
        if not isinstance(c, dict):
            continue
        iface = str(c.get("interface", ""))
        consumer = iface.split("->")[-1] if "->" in iface else iface
        internal_consumed.add((_norm(c.get("to")), _norm(_iface_base(consumer))))

    tokens: set[str] = set()
    found_any = False
    for b in bd.get("blocks", []) or []:
        if not isinstance(b, dict):
            continue
        bname = b.get("name")
        for iname, spec in (b.get("interfaces") or {}).items():
            if _iface_is_input(iname, spec) is not True:
                continue
            found_any = True
            if (_norm(bname), _norm(_iface_base(iname))) in internal_consumed:
                continue  # internally driven -> not a boundary input
            tokens |= _iface_tokens(iname, spec)
    if not found_any:
        # No interface schema we recognise -> signal "unknown" (skip check 2).
        return None
    return {t for t in tokens if t not in _NOISE_TOKENS}


def _field_covered(field: str, signal_tokens: set[str]) -> bool:
    """Is a config field name covered by some boundary-input token?"""
    nf = _norm(field)
    candidates = {nf} | {_norm(a) for a in _ALIASES.get(nf, [])}
    # Exact word-boundary match (robust for short names like ``qp``/``h``).
    if candidates & signal_tokens:
        return True
    # Substring match only for longer names (avoids ``h`` matching ``flush``).
    for c in candidates:
        if len(c) < 4:
            continue
        for t in signal_tokens:
            if len(t) < 4:
                continue
            if c in t or t in c:
                return True
    return False


def run_stimulus_contract_guard(project_root: str) -> list[dict]:
    """Run both generic checks; return a list of violation dicts (possibly empty).

    No-op (returns ``[]``) when the guard is disabled, no explicit stimulus is
    configured, or no reference is resolvable.
    """
    if not guard_enabled():
        return []

    # Lazy imports: reuse the gate's reference/stimulus machinery verbatim so the
    # guard's oracle check matches what the µarch gate will later do.
    from orchestrator.architecture import composition as _composition
    from orchestrator.architecture import model_integration as _mi

    violations: list[dict] = []

    stimulus, found = _mi._load_env_stimulus()
    if not found:
        # No explicit stimulus -> the gate uses a trivial derived stream that is
        # in-contract by construction; nothing to guard.
        return []

    # ---- Check 1: ORACLE ACCEPTANCE ---------------------------------------
    ref_path = _composition.resolve_reference_implementation(project_root)
    entry_callable = entry_name = None
    if ref_path:
        try:
            ref_module = _mi._load_reference_module(ref_path)
            entry_callable, entry_name = _composition.resolve_reference_entrypoint(
                project_root, ref_module
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("stimulus guard: reference import failed: %s", exc)
    if entry_callable is not None:
        try:
            expected = _composition._run_reference(
                entry_callable, stimulus, reraise=True
            )
            if _mi._is_degenerate(expected):
                violations.append({
                    "type": "oracle_degenerate_output",
                    "severity": "warn",
                    "message": (
                        f"the reference entry {entry_name!r} returned degenerate "
                        f"(empty/constant) output for the configured stimulus"
                    ),
                    "suggested_fix": (
                        "the gate oracle would pass vacuously -- confirm "
                        "CORESMITH_MODEL_STIMULUS is a representative, in-contract "
                        "input and CORESMITH_REFERENCE_ENTRY is the right entry."
                    ),
                })
        except _composition.ReferenceInvocationError as exc:
            violations.append({
                "type": "oracle_reference_rejects_stimulus",
                "severity": "error",
                "message": (
                    f"the reference entry {entry_name!r} raised on the configured "
                    f"stimulus: {exc}"
                ),
                "suggested_fix": (
                    "the gate oracle cannot process the stimulus -- fix "
                    "CORESMITH_MODEL_STIMULUS (shape/dtype/fields) or "
                    "CORESMITH_REFERENCE_ENTRY before building. Every gate verdict "
                    "would otherwise be meaningless."
                ),
            })

    # ---- Check 2: STIMULUS-FIELD COVERAGE ---------------------------------
    config_fields = _stimulus_config_fields(stimulus)
    if config_fields:
        signal_tokens = external_input_tokens(project_root)
        if signal_tokens is not None:
            for field in config_fields:
                if not _field_covered(field, signal_tokens):
                    violations.append({
                        "type": "stimulus_field_no_external_input",
                        "severity": "warn",
                        "field": field,
                        "message": (
                            f"stimulus supplies config field {field!r} but the "
                            f"design declares no external (boundary) input that "
                            f"carries it"
                        ),
                        "suggested_fix": (
                            f"add a top-level input port/field for {field!r} so the "
                            f"design derives it at runtime (and the chip model "
                            f"latches it), OR remove {field!r} from the stimulus. "
                            f"A config the design cannot receive will be hardcoded "
                            f"and the composition can never match the reference."
                        ),
                    })

    return violations


def format_violations(violations: list[dict]) -> str:
    """One-line-per-violation human summary."""
    lines = []
    for v in violations:
        sev = v.get("severity", "warn").upper()
        lines.append(f"  [{sev}] {v.get('type')}: {v.get('message')}")
    return "\n".join(lines)
