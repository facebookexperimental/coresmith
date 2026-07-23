# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Typed dimensional-parameter schema for the ERS (``ers_spec.json``).

This is the machine-readable declaration of a design's *parameter space* --
the axes whose extents drive index/address/counter widths and mode selects.
It is the deterministic feedstock that turns the maxgeo DV gate and the memory
manifest gate from prose-driven no-ops into fully-deterministic gates.

Schema (one object per parameter, in ``ers.parameters``)::

    {
      "name": "<free-form design parameter name>",   # data, NOT a fixed vocab
      "role": "dimension" | "mode" | "range",
      "min": <number>,                                # default 0
      "max": <number>,                                # extent (required for
                                                      #   dimension / range)
      "unit": "<string>",                             # default ""
      "boundary_values": [<number>, ...]              # default: [max] + every
                                                      #   2^n crossing <= max
    }

DOMAIN-GENERIC by construction: the parameter ``name`` is opaque data (width,
depth, burst_len, addr_range, frame_count, sample_rate, max_message_blocks,
key_modes -- whatever the design declares). No video/codec/domain vocabulary
lives here; the roles are value-semantics tags.

The boundary crisply: this describes the DESIGN-PARAMETER space (the axes RTL
index/address widths must survive at their maximum), NOT packaging/shuttle
facts (pad counts, die budget, PDK). Those are handled elsewhere.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

# The three parameter roles. ``dimension`` / ``range`` carry a numeric extent
# (max) that an RTL index/address/counter width must survive; ``mode`` is a
# control-select enumerated by its ``boundary_values``.
VALID_ROLES = ("dimension", "mode", "range")

# Roles that contribute a dimensional MAXIMUM to the maxgeo gate (a truncated
# width wraps at a 2^n boundary below the max). ``mode`` is deliberately not a
# geometry axis, so it is excluded from the maxgeo maxima -- it is still handed
# to the DV testbench prompt (via ``format_parameter_table``) so its modes get
# exercised.
_EXTENT_ROLES = ("dimension", "range")

# [rung3r2-fixes-3] ERS ``parameters`` presence-backstop provenance markers. When
# the generator OMITS the ``parameters`` key entirely on two consecutive attempts
# AND no dimensional axis is derivable from the machine-readable sources, the
# backstop writes an EMPTY block tagged with ``BACKSTOP_MARKER_KEY`` so the
# absence is loud + recorded on disk. A backstop-fabricated empty is NOT a
# genuine new-schema signal, so ``ers_has_parameters_block`` treats it as legacy
# (maxgeo has zero dims -> no-op; manifest strictness stays warn-only) -- exactly
# the "running in legacy mode" the open_item records. A design's own affirmed
# ``parameters: []`` (no marker) remains a present-signal, unchanged.
BACKSTOP_MARKER_KEY = "parameters_backstop"
BACKSTOP_EMPTY_MARKER = "empty_no_dims_derivable"


def _num(value: Any) -> Optional[float | int]:
    """Coerce a JSON scalar to a positive-or-zero number, else ``None``.

    Rejects bools (``True``/``False`` are ints in Python but never a dimension).
    Integer-valued floats collapse to ``int`` so ``640.0`` prints as ``640``.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if float(value).is_integer() else value
    if isinstance(value, str):
        s = value.strip().replace("_", "").replace(",", "")
        try:
            if "." in s:
                f = float(s)
                return int(f) if f.is_integer() else f
            return int(s)
        except ValueError:
            return None
    return None


def compute_boundary_values(max_val: Any) -> list:
    """Default ``boundary_values`` = ``[max]`` + every 2^n crossing <= max.

    A truncated index/address/counter width wraps exactly at a power-of-two
    boundary; enumerating the 2^n crossings up to (and including) the declared
    maximum gives the DV a per-crossing checklist without asking the LLM to
    compute it. Returns a sorted, de-duplicated list. ``[]`` for a non-positive
    or non-numeric max.
    """
    mx = _num(max_val)
    if mx is None or mx <= 0:
        return []
    vals: set = set()
    n = 1
    while n <= mx:
        vals.add(n)
        n *= 2
    vals.add(int(mx) if float(mx).is_integer() else mx)
    return sorted(vals)


def normalize_parameter(entry: Any) -> tuple[Optional[dict], list[str]]:
    """Validate + normalize ONE parameter object.

    Returns ``(normalized, errors)``. When ``errors`` is non-empty the entry is
    malformed and the caller drops it (recording the reason). Normalization:
    fills ``min`` (default 0), ``unit`` (default ""), coerces ``role`` (default
    ``dimension``; an unknown role is an error), and auto-computes
    ``boundary_values`` from ``max`` when omitted/empty (ensuring ``max`` is a
    member when the list is provided).
    """
    errors: list[str] = []
    if not isinstance(entry, dict):
        return None, ["parameter entry is not a JSON object"]

    raw_name = entry.get("name")
    if not isinstance(raw_name, str) or not raw_name.strip():
        return None, ["parameter missing a non-empty string 'name'"]
    name = raw_name.strip()

    raw_role = entry.get("role", "dimension")
    if not isinstance(raw_role, str) or raw_role.strip().lower() not in VALID_ROLES:
        errors.append(
            f"parameter {name!r} has invalid role {raw_role!r} "
            f"(must be one of {list(VALID_ROLES)})"
        )
    role = (raw_role.strip().lower() if isinstance(raw_role, str) else "dimension")
    if role not in VALID_ROLES:
        role = "dimension"

    max_val = _num(entry.get("max"))
    min_val = _num(entry.get("min"))
    if min_val is None:
        min_val = 0

    unit = entry.get("unit", "")
    if not isinstance(unit, str):
        unit = str(unit)

    # boundary_values: auto-compute from max when omitted/empty; else coerce +
    # ensure max is a member.
    bv_raw = entry.get("boundary_values")
    if bv_raw is None or (isinstance(bv_raw, list) and not bv_raw):
        boundary_values = compute_boundary_values(max_val) if max_val is not None else []
    elif isinstance(bv_raw, list):
        coerced = [nv for nv in (_num(v) for v in bv_raw) if nv is not None]
        boundary_values = sorted(set(coerced))
        if max_val is not None and max_val not in boundary_values:
            boundary_values = sorted(set(boundary_values) | {max_val})
    else:
        errors.append(f"parameter {name!r} 'boundary_values' is not a list")
        boundary_values = compute_boundary_values(max_val) if max_val is not None else []

    # An extent-bearing role must have a numeric max (or a derivable one).
    if role in _EXTENT_ROLES and max_val is None:
        if boundary_values:
            max_val = max(boundary_values)
        else:
            errors.append(
                f"parameter {name!r} role={role} has no numeric 'max' and no "
                "boundary_values to derive one from"
            )

    if errors:
        return None, errors

    return {
        "name": name,
        "role": role,
        "min": min_val,
        "max": max_val,
        "unit": unit,
        "boundary_values": boundary_values,
    }, []


def validate_parameters(params: Any) -> tuple[list[dict], list[str]]:
    """Validate + normalize the whole ``parameters`` list.

    Returns ``(normalized_list, errors)``. An empty list is valid (a design
    that declares no dimensional parameters -- the ERS prompt requires an
    explicit statement to accompany it, but structurally ``[]`` is accepted).
    Malformed entries are dropped and their reasons collected in ``errors``.
    """
    if params is None:
        return [], []
    if not isinstance(params, list):
        return [], [f"'parameters' must be a list, got {type(params).__name__}"]
    normed: list[dict] = []
    errors: list[str] = []
    for i, entry in enumerate(params):
        norm, errs = normalize_parameter(entry)
        if errs:
            errors.extend(f"parameters[{i}]: {e}" for e in errs)
            continue
        normed.append(norm)  # type: ignore[arg-type]
    return normed, errors


def validate_and_normalize_ers_parameters(
    ers_result: Any,
) -> tuple[Any, list[str]]:
    """Schema-check + normalize the ``parameters`` block in an ERS doc dict.

    The ERS doc-validation repair path: when the LLM-emitted block carries
    malformed entries they are DROPPED (deterministic repair, never a deadlock)
    and the reason is recorded in ``ers.open_items`` so it is visible to the
    operator + downstream review. Returns ``(ers_result, errors)`` with the
    block normalized in place. A doc that carries no ``parameters`` key is left
    untouched (legacy / pre-schema -- the presence of the key is the new-run
    signal, so we never synthesize one).
    """
    errors: list[str] = []
    if not isinstance(ers_result, dict):
        return ers_result, errors
    ers = ers_result.get("ers")
    if not isinstance(ers, dict) or "parameters" not in ers:
        return ers_result, errors
    normed, errs = validate_parameters(ers.get("parameters"))
    ers["parameters"] = normed
    if errs:
        errors.extend(errs)
        open_items = ers.get("open_items")
        if not isinstance(open_items, list):
            open_items = []
        open_items.append(
            "ERS parameters schema: dropped malformed dimensional-parameter "
            "entr" + ("y" if len(errs) == 1 else "ies") + " during validation: "
            + "; ".join(errs)
        )
        ers["open_items"] = open_items
    return ers_result, errors


# ---------------------------------------------------------------------------
# Deterministic derivation fallback ([rung3r2-fixes-3])
# ---------------------------------------------------------------------------
#
# When the generator omits the ``parameters`` key twice, the ERS backstop tries
# to DERIVE the dimensional axes from the same machine-readable sources the
# legacy maxgeo fallback (``pipeline_graph._declared_dimensions`` /
# ``_collect_declared_dims``) already harvests -- constraints/PRD fields carrying
# a ``{name, extent}`` pair, plus FRD FUNC-vector geometry. The harvest logic is
# mirrored here (kept dep-light so the architecture-phase backstop never has to
# import the pipeline graph); the name/extent key sets and the min-extent floor
# are intentionally IDENTICAL to ``pipeline_graph._DIM_*`` so a derived block
# matches exactly what the gate would have harvested. Keep the two in sync.
_DERIVE_NAME_KEYS = ("name", "parameter", "param", "dimension", "dim", "field", "id")
_DERIVE_EXTENT_KEYS = (
    "max", "maximum", "max_value", "maxval", "max_len", "max_length",
    "max_size", "max_depth", "max_count", "max_burst", "max_burst_len",
    "depth", "range", "capacity", "length", "size", "count",
)
_DERIVE_MIN_EXTENT = 8              # skip trivially-tiny dims (a 2-deep handshake)


def _harvest_candidate_dims(obj: Any, out: dict) -> None:
    """Recursively harvest ``{name: extent}`` candidate dimensions from an
    arbitrary JSON-ish structure. Name-agnostic; positive-integer extents only,
    at or above ``_DERIVE_MIN_EXTENT``. Mirrors
    ``pipeline_graph._collect_declared_dims``."""
    if isinstance(obj, dict):
        name = None
        for nk in _DERIVE_NAME_KEYS:
            val = obj.get(nk)
            if isinstance(val, str) and val.strip():
                name = val.strip()
                break
        if name is not None:
            best = None
            for ek in _DERIVE_EXTENT_KEYS:
                if ek in obj:
                    n = _num(obj.get(ek))
                    if (
                        isinstance(n, (int, float))
                        and not isinstance(n, bool)
                        and float(n).is_integer()
                        and n >= _DERIVE_MIN_EXTENT
                    ):
                        iv = int(n)
                        best = iv if best is None else max(best, iv)
            if best is not None:
                out[name] = max(best, out.get(name, 0))
        for v in obj.values():
            _harvest_candidate_dims(v, out)
    elif isinstance(obj, list):
        for item in obj:
            _harvest_candidate_dims(item, out)


def derive_parameters(
    sources: Any, frd_text: Optional[str] = None
) -> list[dict]:
    """Derive a normalized ``parameters`` list from machine-readable ``sources``.

    ``sources`` is an iterable of JSON-ish structures (PRD/SAD/FRD/block-diagram
    dicts + the ERS's own constraints); each harvestable ``{name: extent}`` pair
    becomes a ``role="dimension"`` parameter tagged ``derived: True`` (provenance
    for downstream review). ``frd_text`` -- when supplied -- is parsed for FRD
    FUNC-vector geometry (best-effort) and folded into the harvest. Deterministic
    (sorted by name). Returns ``[]`` when nothing dimensional is derivable. NEVER
    raises."""
    dims: dict = {}
    try:
        for src in sources or []:
            if src is not None:
                _harvest_candidate_dims(src, dims)
        if isinstance(frd_text, str) and frd_text.strip():
            try:
                from orchestrator.architecture.composition import parse_func_vectors
                _harvest_candidate_dims(parse_func_vectors(frd_text), dims)
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        return []
    out: list[dict] = []
    for name in sorted(dims):
        norm, errs = normalize_parameter(
            {"name": name, "role": "dimension", "max": dims[name]}
        )
        if errs or norm is None:
            continue
        norm["derived"] = True
        out.append(norm)
    return out


# ---------------------------------------------------------------------------
# Readers / consumers
# ---------------------------------------------------------------------------

def _load_ers(project_root: str | Path) -> Optional[dict]:
    """Load ``.coresmith/ers_spec.json``; ``None`` on any error."""
    try:
        p = Path(project_root) / ".coresmith" / "ers_spec.json"
        if not p.exists():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def ers_parameters_raw(ers_doc: Optional[dict]) -> Optional[list]:
    """Return the raw ``parameters`` list from an ERS doc dict (``ers`` sub-key
    or top-level), or ``None`` when the key is ABSENT. An empty list is
    returned as ``[]`` (present-but-empty), NOT ``None`` -- the distinction is
    the new-schema signal."""
    if not isinstance(ers_doc, dict):
        return None
    ers = ers_doc.get("ers", ers_doc)
    if not isinstance(ers, dict):
        return None
    block = ers.get("parameters")
    return block if isinstance(block, list) else None


def _is_backstop_empty(ers_doc: Optional[dict]) -> bool:
    """True when the ERS carries an EMPTY ``parameters`` block that the
    presence-backstop ([rung3r2-fixes-3]) fabricated because the generator
    omitted the key twice and no dims were derivable. Such a block is NOT a
    genuine new-schema signal -- it is recorded loudly on disk but the gates run
    in legacy mode. Reads via the ``ers`` sub-key nesting (never raw ``get``)."""
    if not isinstance(ers_doc, dict):
        return False
    ers = ers_doc.get("ers", ers_doc)
    if not isinstance(ers, dict):
        return False
    return (
        ers.get("parameters") == []
        and ers.get(BACKSTOP_MARKER_KEY) == BACKSTOP_EMPTY_MARKER
    )


def ers_has_parameters_block(project_root: str | Path) -> bool:
    """True when the run's ERS carries a ``parameters`` block (a list, possibly
    empty). This IS the new-schema / new-run signal: a run whose ERS was
    generated by the schema-aware generator always emits the key, so its
    presence -- regardless of emptiness -- flips manifest strictness on for
    that run without touching any global default. Legacy prose ERS docs lack
    the key -> False -> warn-only, unchanged.

    EXCEPTION ([rung3r2-fixes-3]): a backstop-fabricated empty block (the key was
    omitted twice + nothing derivable) is deliberately NOT counted -- it is a
    loud on-disk record, not a design-affirmed parameter space, so the gates stay
    in legacy mode. A design's OWN affirmed ``parameters: []`` (no marker) still
    counts, unchanged."""
    doc = _load_ers(project_root)
    if _is_backstop_empty(doc):
        return False
    return ers_parameters_raw(doc) is not None


def parameters_from_ers(project_root: str | Path) -> list[dict]:
    """Normalized (validated) parameter list for the run, or ``[]`` when the
    ERS has no parameters block / none survive validation. NEVER raises."""
    try:
        raw = ers_parameters_raw(_load_ers(project_root))
        if raw is None:
            return []
        normed, _errs = validate_parameters(raw)
        return normed
    except Exception:  # noqa: BLE001
        return []


def declared_maxima(params: list[dict]) -> dict:
    """``{name: max}`` for every extent-bearing (dimension/range) parameter --
    the maxgeo gate's dimensional maxima. ``mode`` params are excluded (they
    are control-selects, not geometry axes). Coerced to ``int`` (marker values
    are integers)."""
    out: dict = {}
    for p in params or []:
        if not isinstance(p, dict):
            continue
        if p.get("role") not in _EXTENT_ROLES:
            continue
        mx = _num(p.get("max"))
        if mx is None:
            bv = p.get("boundary_values")
            if isinstance(bv, list) and bv:
                mx = max((_num(v) for v in bv if _num(v) is not None), default=None)
        if mx is None or mx <= 0:
            continue
        name = p.get("name")
        if isinstance(name, str) and name.strip():
            out[name.strip()] = max(int(mx), out.get(name.strip(), 0))
    return out


def format_parameter_table(params: list[dict]) -> str:
    """Render the normalized parameter list as a compact, verbatim block for a
    DV testbench-generator prompt. Empty string when there are no parameters
    (so callers can gate on truthiness and keep byte-identical prompts when the
    design declares none)."""
    if not params:
        return ""
    lines = [
        "--- DIMENSIONAL PARAMETERS (verbatim from the ERS `parameters` block) ---",
        "These are the design's declared parameter axes. Use them directly; do "
        "NOT re-derive dimensional maxima from prose.",
    ]
    for p in params:
        if not isinstance(p, dict):
            continue
        name = p.get("name", "?")
        role = p.get("role", "dimension")
        mn = p.get("min", 0)
        mx = p.get("max")
        unit = p.get("unit", "")
        bv = p.get("boundary_values", [])
        seg = f"  {name}: role={role} min={mn}"
        if mx is not None:
            seg += f" max={mx}"
        if unit:
            seg += f" unit={unit}"
        seg += f" boundary_values={bv}"
        lines.append(seg)
    lines.append(
        "Drive EVERY dimension/range parameter at its declared max (emit the "
        "`# MAXGEO: <name>=<max> ...` marker covering each), and exercise each "
        "mode parameter across its boundary_values."
    )
    return "\n".join(lines)
