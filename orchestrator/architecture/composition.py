# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Composition gate -- deterministic, LLM-free.

This module wires every per-block *block-level golden model* together per the
block diagram, drives the FRD ``FUNC-NNN`` functional vectors through the
composed chip, and asserts the composed output equals the **reference
implementation**'s output. It is the "shift golden-fidelity LEFT" gate: it
catches a block whose golden math diverges from the reference implementation
(closing the ``QS-OPEN-001`` placeholder class) BEFORE end-of-pipeline DV.

Everything here is pure + deterministic and unit-testable without an LLM or any
EDA tooling.

Public surface:

- :func:`parse_func_vectors` -- tolerant markdown/regex parse of the FRD
  ``## Functional Vectors`` section into structured dicts.
- :func:`resolve_reference_implementation` -- locate the input software golden.
- :func:`load_block_goldens` -- import each ``arch/block_goldens/<block>.py``
  via importlib and instantiate its ``BlockGolden``.
- :func:`compose_and_run` -- topologically wire block goldens per the block
  diagram and run a stream of chip-level input transactions through them.
- :func:`run_composition_gate` -- the end-to-end gate; returns a list of
  violation dicts (empty == pass / no-op).

The feature is gated by ``CORESMITH_BLOCK_GOLDENS``: when off (the default),
:func:`run_composition_gate` is a no-op that returns ``[]``.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import json
import logging
import os
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

logger = logging.getLogger(__name__)

BLOCK_GOLDENS_DIRNAME = "block_goldens"  # v1, retired (under <project_root>/arch/)
BLOCK_MODELS_DIRNAME = "block_models"  # Amaranth block models (under arch/)


# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------

def block_goldens_enabled() -> bool:
    """True when ``CORESMITH_BLOCK_GOLDENS`` is set truthy (strict profile seeds it)."""
    from orchestrator.profile import ensure_applied, flag_enabled
    ensure_applied()
    return flag_enabled("CORESMITH_BLOCK_GOLDENS", default=False)


def bit_exact_enabled() -> bool:
    """True when ``CORESMITH_BIT_EXACT`` is set truthy.

    Default OFF: the model-integration gate accepts a *functional + throughput*
    match (decode + KPI, result-match, objective value). When ON the gate
    reverts to the strict bytewise ``==`` against the reference implementation.
    Mirrors :func:`block_goldens_enabled`.
    """
    return os.environ.get("CORESMITH_BIT_EXACT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def functional_blocks() -> set[str]:
    """Block names whose PER-BLOCK DV uses FUNCTIONAL-QUALITY acceptance instead
    of byte/bit-exact-vs-block-model.

    Read from ``CORESMITH_FUNCTIONAL_BLOCKS`` (comma/space separated). Default
    empty -> every block keeps the strict bit-exact block-DV comparison.

    Motivation (authorised 2026-06-23): a RATE-DISTORTION encoder block makes
    valid mode/quant/trellis choices that need NOT be byte-identical to one
    reference, so byte-exact-vs-block-model is the wrong block-DV bar for it. A
    listed block instead gets a FUNCTIONAL block testbench that decodes its
    output through the block-model's own inverse/reconstruction reference and
    asserts a real reconstruction-quality (PSNR) bound + structural validity +
    a sane rate bound -- a gate that still genuinely fails a garbage encoder
    (NOT a relaxed/always-pass assert). DETERMINISTIC blocks stay bit-exact by
    simply not being listed. Synth + PPA/FF-budget gates are UNAFFECTED for all
    blocks.
    """
    raw = os.environ.get("CORESMITH_FUNCTIONAL_BLOCKS", "")
    names: set[str] = set()
    for tok in raw.replace(",", " ").split():
        tok = tok.strip()
        if tok:
            names.add(tok)
    return names


def is_functional_block(block_name: str) -> bool:
    """True when ``block_name`` is in :func:`functional_blocks`."""
    return bool(block_name) and block_name in functional_blocks()


def gate_allow_nondegenerate_enabled() -> bool:
    """True when ``CORESMITH_GATE_ALLOW_NONDEGENERATE`` is set truthy.

    Default OFF. The model-integration gate's functional default (no declared
    acceptance predicate, ``CORESMITH_BIT_EXACT`` off) requires the composed
    chip model's output to MATCH THE REFERENCE (the FRD behaviour). Setting this
    flag reverts to the old, too-weak check that accepted any *non-degenerate*
    output without comparing it to the reference -- which let a composition
    streaming a handful of garbage bytes pass. Use only when there is genuinely
    no usable oracle comparison; intentionally non-bit-exact (lossy) designs
    should declare an acceptance_fn (see :func:`resolve_functional_acceptance`)
    instead. Mirrors :func:`bit_exact_enabled`.

    A-Fix 5(b): under the STRICT profile the escape hatch is DEPRECATED. If the
    flag is set while strict, log an ERROR and return ``False`` (the functional
    gate keeps its reference-equivalence requirement) -- a deliberate
    env>profile exception, because honoring a "pass any non-degenerate output"
    knob defeats the whole anti-gaming fix. The LEGACY profile still honors it.
    """
    flag_set = os.environ.get(
        "CORESMITH_GATE_ALLOW_NONDEGENERATE", ""
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not flag_set:
        return False
    try:
        from orchestrator.profile import resolve_profile
        profile = resolve_profile()
    except Exception:  # noqa: BLE001 - a profile hiccup must not weaken the gate
        profile = "strict"
    if profile != "legacy":
        logger.error(
            "CORESMITH_GATE_ALLOW_NONDEGENERATE is set but the STRICT profile "
            "DEPRECATES this escape hatch (it let a wrong composed model pass on "
            "mere non-degeneracy). IGNORING it -- the model-integration gate "
            "keeps its reference-equivalence requirement. Set "
            "CORESMITH_PROFILE=legacy to honor the flag."
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Functional acceptance + throughput floor resolution (criterion plumbing)
# ---------------------------------------------------------------------------

def _load_validation_kpis(project_root: str) -> list[dict]:
    """Best-effort load of the ERS ``validation_kpis`` list.

    Looks in ``.coresmith/ers_spec.json`` (the structured ERS) for a
    ``validation_kpis`` array of ``{acceptance_fn?, metric?, threshold?,
    cycles?/throughput?, ...}`` dicts. Returns ``[]`` when unavailable or
    malformed (the caller then falls back to env / non-degenerate tiers).
    """
    root = Path(project_root)
    for name in ("ers_spec.json", "prd_spec.json"):
        p = root / ".coresmith" / name
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        doc = data.get("ers", data.get("prd", data)) if isinstance(data, dict) else {}
        kpis = doc.get("validation_kpis") if isinstance(doc, dict) else None
        if isinstance(kpis, list) and kpis:
            return [k for k in kpis if isinstance(k, dict)]
    return []


def resolve_functional_acceptance(
    project_root: str,
) -> Optional[Callable[[Any, Any], bool]]:
    """Resolve a declared functional-acceptance predicate ``accept(expected, observed)``.

    Resolution order (first hit wins):

    1. ``CORESMITH_FUNCTIONAL_ACCEPTANCE`` env var -- a path to a ``.py``
       file exposing a module-level ``accept(expected, observed) -> bool``
       (mirrors ``CORESMITH_MODEL_STIMULUS``). For a codec this is typically
       decode + PSNR>=thr; for an MCU a result-match; for math an objective
       value.
    2. The ERS ``validation_kpis`` declaring an ``acceptance_fn`` entry of the
       form ``"path.py"`` or ``"path.py:accept"``; the named (or ``accept``)
       callable is loaded.

    Returns the callable, or ``None`` when no acceptance predicate is declared
    (the gate then uses the non-degenerate Tier-B fallback).
    """
    # 1. env override
    env_path = os.environ.get("CORESMITH_FUNCTIONAL_ACCEPTANCE", "").strip()
    if env_path:
        fn = _load_acceptance_callable(env_path)
        if fn is not None:
            return fn
        logger.warning(
            "model integration gate: CORESMITH_FUNCTIONAL_ACCEPTANCE=%r did "
            "not resolve to a callable accept(expected, observed)",
            env_path,
        )

    # 2. declared in ERS validation_kpis
    for kpi in _load_validation_kpis(project_root):
        decl = kpi.get("acceptance_fn")
        if isinstance(decl, str) and decl.strip():
            fn = _load_acceptance_callable(decl.strip(), project_root)
            if fn is not None:
                return fn
            logger.warning(
                "model integration gate: validation_kpis acceptance_fn %r did "
                "not resolve",
                decl,
            )
    return None


def _load_acceptance_callable(
    spec: str,
    project_root: str | None = None,
    default_func: str = "accept",
) -> Optional[Callable[[Any, Any], Any]]:
    """Load an ``<default_func>(expected, observed)`` callable from ``path.py[:name]``.

    ``spec`` is a filesystem path to a ``.py`` (optionally suffixed
    ``:funcname``; defaults to ``default_func``, normally ``accept``). Relative
    paths are resolved against ``project_root`` when given. Returns ``None`` on
    any failure. Reused for the fidelity-metric loader (``default_func``
    ``"fidelity"``) -- see ``orchestrator.architecture.fidelity``.
    """
    spec = spec.strip()
    if not spec:
        return None
    func_name = default_func
    path_part = spec
    # Only split a trailing ":name" (avoid splitting a Windows drive letter).
    if ":" in spec:
        head, _, tail = spec.rpartition(":")
        if head and tail and not tail.endswith(".py"):
            path_part, func_name = head, tail
    p = Path(path_part)
    if not p.is_absolute() and project_root:
        cand = Path(project_root) / p
        if cand.exists():
            p = cand
    if not p.is_file():
        return None
    try:
        mod = _import_module_from_path(p, "_coresmith_functional_acceptance")
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "model integration gate: acceptance module %s failed import: %s",
            p,
            exc,
        )
        return None
    fn = getattr(mod, func_name, None)
    if callable(fn):
        return fn
    return None


def resolve_throughput_floor(project_root: str) -> Optional[float]:
    """Resolve the maximum allowed cycle count (the throughput *floor*).

    The gate compares ``observed_cycles`` against this value; a run that takes
    MORE cycles than the floor is too slow and a throughput violation.

    Resolution order (first hit wins):

    1. ``CORESMITH_THROUGHPUT_FLOOR_CYCLES`` env var (an int/float cycle count).
    2. The ERS ``validation_kpis``: an explicit ``cycles`` / ``max_cycles``
       budget, else a ``throughput`` (samples/clk) combined with a declared
       ``stimulus_len`` -> cycles = stimulus_len / throughput.

    Returns the cycle floor as a float, or ``None`` when no throughput budget
    is declared (the gate then skips the throughput check).
    """
    env_floor = os.environ.get("CORESMITH_THROUGHPUT_FLOOR_CYCLES", "").strip()
    if env_floor:
        try:
            return float(env_floor)
        except ValueError:
            logger.warning(
                "model integration gate: CORESMITH_THROUGHPUT_FLOOR_CYCLES=%r "
                "not numeric",
                env_floor,
            )

    for kpi in _load_validation_kpis(project_root):
        for key in ("max_cycles", "cycles", "cycle_budget"):
            val = kpi.get(key)
            if isinstance(val, (int, float)) and val > 0:
                return float(val)
        thr = kpi.get("throughput")
        slen = kpi.get("stimulus_len")
        if (
            isinstance(thr, (int, float))
            and thr > 0
            and isinstance(slen, (int, float))
            and slen > 0
        ):
            return float(slen) / float(thr)
    return None


# ---------------------------------------------------------------------------
# FRD FUNC vector parsing
# ---------------------------------------------------------------------------

# A FUNC entry is a markdown bullet list under an `### FUNC-NNN ...` heading or
# a `- **ID**: FUNC-NNN` style bullet. We tolerate both layouts. We anchor on
# the FUNC-NNN id token and then collect the labelled fields that follow it
# until the next FUNC-NNN id.
_FUNC_ID_RE = re.compile(r"FUNC-(\d+)", re.IGNORECASE)
_FIELD_RE = re.compile(
    r"\*{0,2}(ID|Block(?:\s*/\s*I-?O\s*ports?)?|Stimulus|Expected\s*output|"
    r"Priority)\*{0,2}\s*[:\-]\s*(.*)",
    re.IGNORECASE,
)


def _func_section(frd_text: str) -> str:
    """Return only the ``## Functional Vectors`` section of the FRD, if present.

    Falls back to the whole document so a FRD that uses a slightly different
    heading still gets scanned.
    """
    if not frd_text:
        return ""
    # Match the heading, then up to the next same-or-higher level heading.
    # Stop only at a level-1/2 heading ("# " or "## ") so that "### FUNC-NNN"
    # subheadings inside the section do NOT terminate the capture.
    m = re.search(
        r"^#{1,2}\s*Functional\s+Vectors\s*$(.*?)(?=^#{1,2}\s|\Z)",
        frd_text,
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    if m:
        return m.group(1)
    return frd_text


def _coerce_scalar(value: str) -> Any:
    """Best-effort coercion of a field value to a Python literal.

    Tries ``ast.literal_eval`` (so ``[1, 2, 3]``, ``0x4F``, ``"foo"`` parse),
    then a bare-int parse, else returns the trimmed string. This is best-effort
    metadata; ``compose_and_run`` reads structured stimulus/expected when the
    caller supplies it, so an un-coercible value simply stays a string.
    """
    text = value.strip().strip("`").strip()
    if not text:
        return ""
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        pass
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"0x[0-9a-fA-F]+", text):
        return int(text, 16)
    return text


# Fenced ```json ... ``` block inside a FUNC entry (the machine-readable form).
_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```",
    re.IGNORECASE | re.DOTALL,
)
# Inline `Stimulus (structured): {...}` JSON form.
_INLINE_STIM_RE = re.compile(
    r"\*{0,2}Stimulus\s*\(structured\)\*{0,2}\s*[:\-]\s*(\{.*?\})\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _extract_structured(chunk: str) -> tuple[Optional[dict], Optional[Any]]:
    """Pull a machine-readable stimulus/expected pair out of a FUNC chunk.

    Recognises two shapes:

    1. A fenced json block (```json {"stimulus": {...}, "expected": {...}}```)
       where ``expected`` is optional.
    2. An inline ``Stimulus (structured): {...}`` line whose JSON is itself the
       stimulus dict.

    Returns ``(stimulus_struct, expected_struct)`` where each may be ``None``.
    """
    stimulus_struct: Optional[dict] = None
    expected_struct: Optional[Any] = None

    for m in _JSON_FENCE_RE.finditer(chunk):
        try:
            obj = json.loads(m.group(1))
        except (ValueError, TypeError):
            continue
        if not isinstance(obj, dict):
            continue
        if "stimulus" in obj:
            stim = obj.get("stimulus")
            if isinstance(stim, dict):
                stimulus_struct = stim
            if "expected" in obj:
                expected_struct = obj.get("expected")
            break
        # A bare JSON object with no "stimulus" key is treated as the stimulus
        # struct itself (lenient: lets a FUNC entry inline just the stimulus).
        if stimulus_struct is None:
            stimulus_struct = obj
            break

    if stimulus_struct is None:
        im = _INLINE_STIM_RE.search(chunk)
        if im:
            try:
                obj = json.loads(im.group(1))
                if isinstance(obj, dict):
                    stimulus_struct = obj
            except (ValueError, TypeError):
                pass

    return stimulus_struct, expected_struct


def parse_func_vectors(frd_text: str) -> list[dict]:
    """Parse the FRD ``## Functional Vectors`` section into structured dicts.

    Tolerant markdown/regex parse. Each returned dict has keys:
    ``id`` (e.g. ``"FUNC-001"``), ``block``, ``stimulus``, ``expected_output``,
    ``priority``, plus the machine-readable ``stimulus_struct`` (``dict|None``)
    and ``expected_struct`` (``Any|None``) extracted from a fenced ```json``
    block or an inline ``Stimulus (structured):`` line. Unknown / missing
    fields default to ``""`` (prose) or ``None`` (structured). The parser is
    line-oriented: it splits the section on FUNC-NNN id tokens and reads the
    labelled fields belonging to each.
    """
    section = _func_section(frd_text)
    if not section:
        return []

    # Split the section into chunks, one per FUNC-NNN occurrence. We find each
    # id position and slice up to the next id.
    ids = list(_FUNC_ID_RE.finditer(section))
    vectors: list[dict] = []
    for i, m in enumerate(ids):
        start = m.start()
        end = ids[i + 1].start() if i + 1 < len(ids) else len(section)
        chunk = section[start:end]
        func_id = f"FUNC-{int(m.group(1)):03d}"

        rec = {
            "id": func_id,
            "block": "",
            "stimulus": "",
            "expected_output": "",
            "priority": "",
            "stimulus_struct": None,
            "expected_struct": None,
        }
        stim_struct, exp_struct = _extract_structured(chunk)
        rec["stimulus_struct"] = stim_struct
        rec["expected_struct"] = exp_struct
        for line in chunk.splitlines():
            fm = _FIELD_RE.search(line)
            if not fm:
                continue
            label = fm.group(1).lower()
            raw = fm.group(2).strip()
            if label.startswith("block"):
                rec["block"] = raw
            elif label == "stimulus":
                rec["stimulus"] = _coerce_scalar(raw)
            elif label.startswith("expected"):
                rec["expected_output"] = _coerce_scalar(raw)
            elif label == "priority":
                rec["priority"] = raw
            elif label == "id":
                # The explicit ID field; keep the normalised one we derived.
                pass
        vectors.append(rec)

    # De-dup by id (a `- **ID**: FUNC-001` line plus a heading would otherwise
    # double-count); keep the first, richest occurrence.
    seen: dict[str, dict] = {}
    for v in vectors:
        if v["id"] not in seen:
            seen[v["id"]] = v
        else:
            # Merge: fill any blank fields from the later occurrence.
            for k, val in v.items():
                if not seen[v["id"]].get(k) and val:
                    seen[v["id"]][k] = val
    return list(seen.values())


# ---------------------------------------------------------------------------
# Reference implementation resolution
# ---------------------------------------------------------------------------

def resolve_generator_reference(project_root: str) -> str | None:
    """Reference source for the block-model / chip-model GENERATORS.

    The generators transcribe per-block MATH from the reference, so they need the
    FULL software golden (e.g. ``codec_golden.py``). The GATE, by contrast, needs
    a value it can compare to the chip's egress -- often a bytes-only WRAPPER
    pointed at by ``CORESMITH_SOURCE_ROOT``. Conflating the two starves the
    generators (a bytes-only wrapper has no per-block math -> empty stub blocks).

    Resolution: ``CORESMITH_GENERATOR_SOURCE`` (a file/dir) wins; otherwise fall
    back to :func:`resolve_reference_implementation` (so single-reference designs
    and existing setups are unchanged).
    """
    env_src = os.environ.get("CORESMITH_GENERATOR_SOURCE", "").strip()
    if env_src:
        p = Path(env_src)
        if p.is_file():
            return str(p.resolve())
        if p.is_dir():
            hits = sorted(p.glob("*_golden.py")) or sorted(p.glob("**/*_golden.py"))
            if hits:
                return str(hits[0].resolve())
    return resolve_reference_implementation(project_root)


def resolve_reference_implementation(project_root: str) -> str | None:
    """Locate the design's **reference implementation** (input software golden).

    Search order (first hit wins):

    0. ``CORESMITH_SOURCE_ROOT`` env var (a file, or a dir to scan) -- an
       EXPLICIT operator override always wins over auto-discovery, so a
       bitstream-only reference WRAPPER can be pointed at even when the PRD
       text names the raw ``*_golden.py`` (whose richer return type the gate
       cannot compare).
    1. An explicit path recorded in the PRD/requirements (a line naming a
       ``*_golden.py``) under ``<root>/arch`` or ``<root>/inputs``.
    2. ``examples/<design>/*_golden.py`` relative to the project root (a run
       dir that copied an example design).
    3. ``<root>/inputs/*_golden.py`` then ``<root>/inputs/*.py``.
    4. Any ``*_golden.py`` anywhere under the project root.

    Returns an absolute path string, or ``None`` when nothing is found (the
    gate then becomes a logged no-op).
    """
    root = Path(project_root)

    # 0. Explicit env override -- highest priority.
    env_src = os.environ.get("CORESMITH_SOURCE_ROOT", "").strip()
    if env_src:
        p = Path(env_src)
        if p.is_file():
            return str(p.resolve())
        if p.is_dir():
            hits = sorted(p.glob("*_golden.py")) or sorted(p.glob("**/*_golden.py"))
            if hits:
                return str(hits[0].resolve())

    # 1. Explicit reference cited in PRD / requirements text.
    for doc in (
        root / "arch" / "prd_spec.md",
        root / "inputs" / "requirements.md",
        root / "requirements.md",
    ):
        if doc.exists():
            try:
                text = doc.read_text(encoding="utf-8")
            except OSError:
                continue
            m = re.search(r"([^\s`'\"]+_golden\.py)", text)
            if m:
                cand = (root / m.group(1)).resolve()
                if cand.exists():
                    return str(cand)
                # Also try as an absolute / cwd-relative path.
                cand2 = Path(m.group(1))
                if cand2.exists():
                    return str(cand2.resolve())

    # 2. examples/<design>/*_golden.py under the project root.
    examples_dir = root / "examples"
    if examples_dir.is_dir():
        hits = sorted(examples_dir.glob("*/*_golden.py"))
        if hits:
            return str(hits[0].resolve())

    # 3. inputs/.
    inputs_dir = root / "inputs"
    if inputs_dir.is_dir():
        hits = sorted(inputs_dir.glob("*_golden.py")) or sorted(
            inputs_dir.glob("*.py")
        )
        if hits:
            return str(hits[0].resolve())

    # 5. Anywhere under the root.
    hits = sorted(root.glob("**/*_golden.py"))
    if hits:
        return str(hits[0].resolve())

    return None


# Regex for a declared reference entry point in PRD/FRD prose.
_REF_ENTRY_DECL_RE = re.compile(
    r"reference[_ ]entry[_ ]point[:=]\s*([A-Za-z_][\w.]*)",
    re.IGNORECASE,
)

# Public callable names we prefer when discovering an entry point.
_ENTRY_NAME_RE = re.compile(
    r"^(encode|decode|run|process|main|top|encode_image\w*|chip_top)\b",
    re.IGNORECASE,
)

# Preference order among conventional entry names. ``dir()`` returns names
# ALPHABETICALLY, so without an explicit intent ranking a codec golden that
# exposes both ``encode`` and ``decode`` resolved to ``decode`` ('d' < 'e') --
# the wrong oracle for an ENCODER design. Rank by the design's primary
# transform: encode before decode, generic drivers last. (Override always
# available via CORESMITH_REFERENCE_ENTRY or a declared reference_entry_point.)
_ENTRY_PRIORITY = (
    "encode_image", "encode", "decode", "chip_top", "top", "run", "process",
    "main",
)


def _entry_priority(name: str) -> tuple[int, str]:
    """Rank a conventional entry name by INTENT, not alphabetically.

    Lower sorts first. Only names that already matched ``_ENTRY_NAME_RE`` reach
    here, so the prefix check is safe. Unknown-but-matching names sort after the
    known set, tie-broken by name for determinism.
    """
    low = name.lower()
    for i, key in enumerate(_ENTRY_PRIORITY):
        if low == key or low.startswith(key):
            return (i, low)
    return (len(_ENTRY_PRIORITY), low)


def _public_callables(module) -> list[tuple[str, Callable]]:
    """Public (non-underscore) top-level callables defined in ``module``.

    Only includes functions/callables whose ``__module__`` is the module itself
    (so imported helpers like ``json.loads`` are not mistaken for entries).
    """
    out: list[tuple[str, Callable]] = []
    mod_name = getattr(module, "__name__", None)
    for name in dir(module):
        if name.startswith("_"):
            continue
        obj = getattr(module, name, None)
        if not callable(obj):
            continue
        # Prefer functions defined in this module; tolerate callables without a
        # __module__ (e.g. some builtins) by excluding them.
        obj_mod = getattr(obj, "__module__", None)
        if obj_mod is not None and mod_name is not None and obj_mod != mod_name:
            continue
        if inspect.isfunction(obj) or inspect.ismethod(obj):
            out.append((name, obj))
    return out


def resolve_reference_entrypoint(
    project_root: str,
    ref_module,
) -> Tuple[Optional[Callable], str]:
    """Resolve the callable that IS the design's executable oracle.

    Resolution order (first hit wins):

    1. ``CORESMITH_REFERENCE_ENTRY`` env var, formatted ``"func"`` or
       ``"module:func"``. ``"func"`` is ``getattr`` on ``ref_module``;
       ``"module:func"`` imports ``module`` (a submodule / dotted path) and
       reads ``func`` from it (falling back to ``getattr(ref_module, func)``).
    2. A declared entry in ``arch/prd_spec.md`` or ``arch/frd_spec.md`` matching
       ``reference_entry_point: <name>`` (the name may be dotted/attr-pathed and
       is resolved against ``ref_module``).
    3. Discovery on ``ref_module``: among its public top-level functions, prefer
       a name matching ``^(encode|decode|run|process|main|top|encode_image*|
       chip_top)``; else, if there is exactly one public top-level function,
       use it.

    Returns ``(callable_or_None, dotted_name_str)``. ``dotted_name_str`` is a
    best-effort human-readable name for logging even when the callable is None.
    """
    # 1. env override
    env_entry = os.environ.get("CORESMITH_REFERENCE_ENTRY", "").strip()
    if env_entry:
        fn = _resolve_dotted_entry(env_entry, ref_module)
        if fn is not None:
            return fn, env_entry
        logger.warning(
            "composition gate: CORESMITH_REFERENCE_ENTRY=%r did not resolve",
            env_entry,
        )

    # 2. declared in PRD / FRD prose
    root = Path(project_root)
    for doc in (root / "arch" / "prd_spec.md", root / "arch" / "frd_spec.md"):
        if not doc.exists():
            continue
        try:
            text = doc.read_text(encoding="utf-8")
        except OSError:
            continue
        m = _REF_ENTRY_DECL_RE.search(text)
        if m:
            decl = m.group(1)
            fn = _resolve_dotted_entry(decl, ref_module)
            if fn is not None:
                return fn, decl
            logger.warning(
                "composition gate: declared reference_entry_point %r "
                "did not resolve",
                decl,
            )

    # 3. discovery on the ref module
    if ref_module is not None:
        publics = _public_callables(ref_module)
        # Prefer a conventionally-named entry, RANKED BY INTENT -- not by the
        # alphabetical dir() order, which made an encoder golden exposing both
        # `encode` and `decode` resolve to `decode` ('d' < 'e').
        conventional = [
            (name, fn) for name, fn in publics if _ENTRY_NAME_RE.match(name)
        ]
        if conventional:
            conventional.sort(key=lambda item: _entry_priority(item[0]))
            if len(conventional) > 1:
                logger.warning(
                    "composition gate: multiple conventional reference entries "
                    "%s -- chose %r by intent priority (encode before decode); "
                    "set CORESMITH_REFERENCE_ENTRY or a declared "
                    "reference_entry_point to override",
                    [n for n, _ in conventional],
                    conventional[0][0],
                )
            name, fn = conventional[0]
            return fn, name
        # Else the single public top-level function, if unambiguous.
        if len(publics) == 1:
            name, fn = publics[0]
            return fn, name

    return None, env_entry or ""


def _resolve_dotted_entry(spec: str, ref_module) -> Optional[Callable]:
    """Resolve a ``"func"`` / ``"module:func"`` / ``"a.b.func"`` entry spec.

    ``module:func`` imports ``module`` (a dotted import path) and reads ``func``
    off it. A bare ``func`` or ``attr.path`` is resolved against ``ref_module``
    via successive ``getattr``. Returns the callable or ``None``.
    """
    spec = spec.strip()
    if not spec:
        return None

    if ":" in spec:
        mod_path, _, attr = spec.partition(":")
        target = None
        # Try importing as a submodule of the ref module first, then absolute.
        candidates = []
        ref_name = getattr(ref_module, "__name__", None)
        if ref_name:
            candidates.append(f"{ref_name}.{mod_path}")
        candidates.append(mod_path)
        for cand in candidates:
            try:
                target = importlib.import_module(cand)
                break
            except Exception:  # noqa: BLE001
                target = None
        if target is None:
            # Fall back: treat the whole thing as an attr path on ref_module.
            return _getattr_path(ref_module, attr)
        fn = _getattr_path(target, attr)
        if fn is None and ref_module is not None:
            fn = _getattr_path(ref_module, attr)
        return fn

    return _getattr_path(ref_module, spec)


def _getattr_path(obj, dotted: str) -> Optional[Callable]:
    """``getattr`` along a dotted path; return the callable or None."""
    if obj is None or not dotted:
        return None
    cur = obj
    for part in dotted.split("."):
        cur = getattr(cur, part, None)
        if cur is None:
            return None
    return cur if callable(cur) else None


# ---------------------------------------------------------------------------
# Block golden loading
# ---------------------------------------------------------------------------

def _import_module_from_path(path: Path, mod_name: str):
    """Import a module from a file path under a private module name."""
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_block_goldens(
    goldens_dir: str,
    block_names: list[str],
) -> dict[str, Any]:
    """Import + instantiate each block's golden model.

    Args:
        goldens_dir: directory containing ``<block>.py`` block goldens.
        block_names: the blocks to load (typically the block diagram's block
            names). A block with no ``<block>.py`` is silently skipped (the
            caller decides whether a missing golden is fatal).

    Returns:
        ``{block_name: BlockGolden instance}`` for every block that loaded.

    Raises:
        RuntimeError: if a present ``<block>.py`` fails to import, lacks PORTS,
            or lacks an instantiable ``BlockGolden``.
    """
    out: dict[str, Any] = {}
    d = Path(goldens_dir)
    for name in block_names:
        path = d / f"{name}.py"
        if not path.exists():
            continue
        try:
            module = _import_module_from_path(
                path, f"_coresmith_blkgolden_{name}"
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"block golden {path} failed to import: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        ports = getattr(module, "PORTS", None)
        if not isinstance(ports, dict) or "inputs" not in ports or "outputs" not in ports:
            raise RuntimeError(
                f"block golden {path} has no valid PORTS "
                "(need dict with 'inputs' and 'outputs')"
            )
        block_cls = getattr(module, "BlockGolden", None)
        if block_cls is None or not callable(block_cls):
            raise RuntimeError(f"block golden {path} has no BlockGolden class")
        try:
            instance = block_cls()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"block golden {path} BlockGolden() failed to instantiate: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        # Attach the declared ports so compose_and_run can map edges.
        instance._coresmith_ports = ports  # noqa: SLF001
        out[name] = instance
    return out


# ---------------------------------------------------------------------------
# Composition (topological wiring + execution)
# ---------------------------------------------------------------------------

def _block_ports(instance: Any) -> dict:
    ports = getattr(instance, "_coresmith_ports", None)
    if isinstance(ports, dict):
        return ports
    return {"inputs": [], "outputs": []}


def _topo_order(
    block_names: list[str],
    connections: list[dict],
) -> tuple[list[str], set[tuple[str, str]]]:
    """Kahn topological sort of blocks; returns (order, feedback_edges).

    Edges that would form a cycle (feedback paths) are detected and excluded
    from the ordering constraint, then returned separately so the caller can
    deliver them with a one-transaction delay. The remaining DAG defines the
    forward evaluation order within a transaction.
    """
    present = set(block_names)
    # Build adjacency from connections, ignoring chip-boundary endpoints
    # (a `from`/`to` that is not a known block name is a chip I/O endpoint).
    # A self-loop (src == dst) is always a feedback edge.
    edges: list[tuple[str, str]] = []
    self_loops: set[tuple[str, str]] = set()
    for c in connections:
        src = c.get("from", c.get("from_block", ""))
        dst = c.get("to", c.get("to_block", ""))
        if src in present and dst in present:
            if src == dst:
                self_loops.add((src, dst))
            else:
                edges.append((src, dst))

    # Detect feedback edges: an edge (u, v) is feedback if v can reach u via
    # other edges (i.e. it closes a cycle). We compute a tentative order via
    # DFS and mark back-edges.
    feedback: set[tuple[str, str]] = set()
    adj: dict[str, list[str]] = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)

    color: dict[str, int] = {n: 0 for n in block_names}  # 0=white,1=gray,2=black

    def dfs(u: str) -> None:
        color[u] = 1
        for v in adj.get(u, []):
            if color.get(v, 0) == 1:
                feedback.add((u, v))  # back-edge -> feedback
            elif color.get(v, 0) == 0:
                dfs(v)
        color[u] = 2

    for n in block_names:
        if color.get(n, 0) == 0:
            dfs(n)

    feedback |= self_loops

    # Kahn's algorithm on the DAG (forward edges only).
    forward = [(u, v) for (u, v) in edges if (u, v) not in feedback]
    indeg: dict[str, int] = {n: 0 for n in block_names}
    fadj: dict[str, list[str]] = defaultdict(list)
    for u, v in forward:
        indeg[v] += 1
        fadj[u].append(v)
    q = deque(sorted(n for n in block_names if indeg[n] == 0))
    order: list[str] = []
    while q:
        n = q.popleft()
        order.append(n)
        for v in sorted(fadj.get(n, [])):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    # Any block not ordered (shouldn't happen once feedback removed) appended
    # deterministically so it still runs.
    for n in block_names:
        if n not in order:
            order.append(n)
    return order, feedback


def compose_and_run(
    block_diagram: dict,
    block_goldens: dict[str, Any],
    chip_inputs: dict,
) -> dict:
    """Wire block goldens per the block diagram and run a transaction stream.

    Args:
        block_diagram: ``{"blocks": [...], "connections": [...]}``. Each
            connection is ``{"from","to","from_port","to_port", ...}`` (the
            ``interface`` field is used as a fallback port name).
        block_goldens: ``{block_name: BlockGolden instance}``.
        chip_inputs: ``{chip_input_port: [v0, v1, ...]}`` -- one list per
            chip-level input port, giving the value at each of N transactions.
            Scalars are accepted and treated as a single-transaction stream.

    Returns:
        ``{chip_output_port: [v0, v1, ...]}`` -- the chip-level outputs
        collected per transaction (only transactions that produced a value).

    Semantics:
        - Within a transaction, blocks are evaluated in topological order; a
          block's ``step()`` outputs are piped along forward edges to the
          consumers evaluated later in the same transaction.
        - Feedback edges (cycles) deliver a producer's *previous*-transaction
          output to the consumer on the *next* transaction (one-transaction
          delay), using the persistent block instance state.
        - A block that returns ``{}`` (latency / accumulation) simply provides
          no value on its outgoing edges that transaction.
        - chip-level outputs are edges whose ``to`` endpoint is not a block
          (a chip-boundary egress port) OR a block output port named in the
          block diagram's chip-output interface. We also expose any block
          output that has no forward consumer as a chip output under
          ``<block>.<port>`` when no explicit chip egress edge exists.
    """
    blocks_meta = block_diagram.get("blocks", []) or []
    connections = block_diagram.get("connections", []) or []
    block_names = [b.get("name", "") for b in blocks_meta if b.get("name")]
    # Only consider blocks we actually have goldens for.
    block_names = [n for n in block_names if n in block_goldens]
    present = set(block_names)

    order, feedback = _topo_order(block_names, connections)

    # Reset all blocks before a run so streams are deterministic.
    for inst in block_goldens.values():
        if hasattr(inst, "reset"):
            try:
                inst.reset()
            except Exception:  # noqa: BLE001 - reset is best-effort
                pass

    # Normalise chip inputs to per-port lists and find the stream length.
    norm_inputs: dict[str, list] = {}
    n_txn = 1
    for port, vals in chip_inputs.items():
        if isinstance(vals, list):
            norm_inputs[port] = vals
            n_txn = max(n_txn, len(vals))
        else:
            norm_inputs[port] = [vals]
    # Pad shorter input streams with None (no drive that transaction).
    for port in norm_inputs:
        if len(norm_inputs[port]) < n_txn:
            norm_inputs[port] = norm_inputs[port] + [None] * (
                n_txn - len(norm_inputs[port])
            )

    # Index connections for fast per-block lookup.
    # forward_in[block] = list of (src, from_port, to_port)
    forward_in: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    feedback_in: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    # chip_in_edges[block] = list of (chip_port, to_port)
    chip_in_edges: dict[str, list[tuple[str, str]]] = defaultdict(list)
    # chip_out_edges = list of (src_block, from_port, chip_out_port)
    chip_out_edges: list[tuple[str, str, str]] = []
    # consumed outputs: (block, port) that feed some block forward edge
    consumed: set[tuple[str, str]] = set()

    for c in connections:
        src = c.get("from", c.get("from_block", ""))
        dst = c.get("to", c.get("to_block", ""))
        fport = c.get("from_port") or c.get("interface") or ""
        tport = c.get("to_port") or c.get("interface") or ""
        if src in present and dst in present:
            if (src, dst) in feedback:
                feedback_in[dst].append((src, fport, tport))
            else:
                forward_in[dst].append((src, fport, tport))
            consumed.add((src, fport))
        elif src in present and dst not in present:
            # chip egress edge
            chip_out_edges.append((src, fport, dst or f"{src}.{fport}"))
            consumed.add((src, fport))
        elif src not in present and dst in present:
            # chip ingress edge
            chip_in_edges[dst].append((src, tport))

    # Per-block previous-transaction output cache for feedback delivery.
    prev_out: dict[str, dict] = {n: {} for n in block_names}

    chip_outputs: dict[str, list] = defaultdict(list)

    for t in range(n_txn):
        cur_out: dict[str, dict] = {}
        for name in order:
            inst = block_goldens[name]
            ports = _block_ports(inst)
            in_ports = ports.get("inputs", [])
            step_in: dict = {}

            # chip ingress
            for chip_port, to_port in chip_in_edges.get(name, []):
                key = to_port or chip_port
                # Prefer the chip stream keyed by the chip-side port; fall back
                # to the to_port name.
                val = None
                if chip_port in norm_inputs:
                    val = norm_inputs[chip_port][t]
                elif to_port in norm_inputs:
                    val = norm_inputs[to_port][t]
                elif key in norm_inputs:
                    val = norm_inputs[key][t]
                if val is not None:
                    step_in[key] = val

            # If the block declares input ports that match chip input stream
            # names directly (single-block designs, or unconnected ingress),
            # drive them too.
            for p in in_ports:
                if p in norm_inputs and p not in step_in:
                    if norm_inputs[p][t] is not None:
                        step_in[p] = norm_inputs[p][t]

            # forward edges (same transaction, upstream already ran)
            for src, fport, tport in forward_in.get(name, []):
                produced = cur_out.get(src, {})
                if fport in produced:
                    step_in[tport or fport] = produced[fport]

            # feedback edges (previous transaction's output, one-cycle delay)
            for src, fport, tport in feedback_in.get(name, []):
                produced = prev_out.get(src, {})
                if fport in produced:
                    step_in[tport or fport] = produced[fport]

            # Every declared input port must be present for block goldens that
            # strictly validate their input keys. Default any port not driven
            # this transaction -- feedback/sideband ports on the first
            # transaction (no producer output yet), or unwired ingress -- to 0
            # (the idle/no-activity value) so composition never raises mid-run.
            for p in in_ports:
                if p not in step_in:
                    step_in[p] = 0

            result = inst.step(step_in)
            cur_out[name] = result if isinstance(result, dict) else {}

        # collect chip-level outputs for this transaction
        for src, fport, chip_port in chip_out_edges:
            produced = cur_out.get(src, {})
            if fport in produced:
                chip_outputs[chip_port].append(produced[fport])
            elif not produced and fport == "":
                pass

        # If there are NO explicit chip egress edges, expose every unconsumed
        # block output port as a chip output (covers single-block + terminal
        # blocks the diagram didn't wire to a boundary egress).
        if not chip_out_edges:
            for name in order:
                produced = cur_out.get(name, {})
                for port, val in produced.items():
                    if (name, port) not in consumed:
                        chip_outputs[f"{name}.{port}"].append(val)

        prev_out = cur_out

    return dict(chip_outputs)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

def _load_reference_module(path: str):
    """Import the reference implementation module from a file path."""
    return _import_module_from_path(Path(path), "_coresmith_reference_impl")


def _vector_block(vec: dict) -> str:
    """Extract the bare block name from a FUNC vector's 'block' field."""
    raw = str(vec.get("block", "")).strip()
    # The block field is often "<block> / in -> out"; take the leading token.
    token = re.split(r"[\s/(]", raw, maxsplit=1)[0].strip()
    # Strip markdown/code-span backticks and trailing punctuation cruft
    # (FUNC "Block / I-O" fields look like "`frame_ctrl`;  drives ...").
    token = token.strip(" `;:,.\t\n")
    return token


def run_composition_gate(
    project_root: str, result_info: dict | None = None
) -> list[dict]:
    """DEPRECATED v1 entry point -- delegates to the v2 model-integration gate.

    v1 wired ad-hoc ``BlockGolden.step()`` Python goldens through an untimed
    ``compose_and_run`` harness and drove FRD FUNC vectors. v2 replaces that
    with Amaranth block models, an LLM model-integration agent that builds a
    top-level Amaranth chip model, and a deterministic pysim gate that
    compares the integrated model bit-exact to the reference implementation.

    This shim preserves the old callable so existing imports/tests don't break;
    it forwards to :func:`run_model_integration_gate`. Returns ``[]`` (no-op)
    when the feature flag is off.
    """
    if not block_goldens_enabled():
        logger.info("composition gate: CORESMITH_BLOCK_GOLDENS off -- no-op")
        if result_info is not None:
            result_info.update(skipped=True,
                               reason="CORESMITH_BLOCK_GOLDENS off",
                               checked_vectors=0)
        return []
    try:
        from orchestrator.architecture.model_integration import (
            run_model_integration_gate,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("composition gate: model_integration import failed: %s", exc)
        if result_info is not None:
            result_info.update(skipped=True,
                               reason=f"model_integration import failed: {exc}",
                               checked_vectors=0)
        return []
    return run_model_integration_gate(project_root, result_info=result_info)


def _run_composition_gate_v1(project_root: str) -> list[dict]:
    """RETIRED v1 implementation (kept for reference / not wired).

    Returns a list of violation dicts (empty == pass). No-op when the feature
    flag is off, no reference implementation is found, or there is no
    block_goldens dir / no block goldens.
    """
    if not block_goldens_enabled():
        logger.info("composition gate: CORESMITH_BLOCK_GOLDENS off -- no-op")
        return []

    root = Path(project_root)

    goldens_dir = root / "arch" / BLOCK_GOLDENS_DIRNAME
    if not goldens_dir.is_dir() or not any(goldens_dir.glob("*.py")):
        logger.info(
            "composition gate: no block goldens at %s -- no-op", goldens_dir
        )
        return []

    # A reference implementation is the PREFERRED oracle but is OPTIONAL:
    # objective-math designs (adder/CRC/MCU) have no executable reference, and
    # the gate then falls back to the FRD vector's hand-computed expected. So a
    # missing reference is NOT a no-op anymore -- we still drive vectors that
    # carry an explicit expected.
    ref_path = resolve_reference_implementation(project_root)
    if not ref_path:
        logger.info(
            "composition gate: no reference implementation -- will fall back "
            "to FRD expected (objective-math path)"
        )

    bd_path = root / ".coresmith" / "block_diagram.json"
    if not bd_path.exists():
        logger.info("composition gate: no block_diagram.json -- no-op")
        return []
    try:
        block_diagram = json.loads(bd_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [
            {
                "type": "composition_gate_error",
                "first_divergence_block": "",
                "expected": "",
                "observed": "",
                "suggested_fix": f"block_diagram.json unreadable: {exc}",
            }
        ]

    block_names = [
        b.get("name", "")
        for b in block_diagram.get("blocks", [])
        if b.get("name")
    ]
    try:
        block_goldens = load_block_goldens(str(goldens_dir), block_names)
    except RuntimeError as exc:
        return [
            {
                "type": "composition_gate_error",
                "first_divergence_block": "",
                "expected": "",
                "observed": "",
                "suggested_fix": str(exc),
            }
        ]
    if not block_goldens:
        logger.info("composition gate: no loadable block goldens -- no-op")
        return []

    # Load FRD FUNC vectors.
    frd_text = ""
    frd_path = root / "arch" / "frd_spec.md"
    if frd_path.exists():
        try:
            frd_text = frd_path.read_text(encoding="utf-8")
        except OSError:
            frd_text = ""
    vectors = parse_func_vectors(frd_text)

    # Try to obtain the reference module + a single callable entry point. The
    # entry point (when present) is the AUTHORITATIVE oracle: expected output is
    # computed by running it on each vector's stimulus.
    ref_module = None
    if ref_path:
        try:
            ref_module = _load_reference_module(ref_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("composition gate: reference import failed: %s", exc)

    entry_callable, entry_name = resolve_reference_entrypoint(
        project_root, ref_module
    )
    if entry_callable is not None:
        logger.info(
            "composition gate: reference oracle entry = %s", entry_name
        )
    else:
        logger.info(
            "composition gate: no callable reference entry -- using FRD "
            "expected as oracle (objective-math fallback)"
        )

    violations: list[dict] = []

    for vec in vectors:
        # 1. Determine the stimulus. Prefer the structured machine-readable
        #    stimulus; fall back to numeric coercion of the prose stimulus.
        stim_struct = vec.get("stimulus_struct")
        if isinstance(stim_struct, dict) and stim_struct:
            chip_inputs = _stimulus_to_chip_inputs(stim_struct, block_diagram)
            ref_stim: Any = stim_struct
        else:
            prose_stim = vec.get("stimulus", "")
            if prose_stim in ("", None):
                continue  # nothing to drive -- covered by per-block DV
            chip_inputs = _stimulus_to_chip_inputs(prose_stim, block_diagram)
            ref_stim = chip_inputs if chip_inputs is not None else prose_stim

        if chip_inputs is None:
            # Not machine-drivable -- SKIP (per-block DV still covers it).
            logger.info(
                "composition gate: vector %s stimulus not machine-drivable "
                "-- skipping", vec.get("id", ""),
            )
            continue

        # 2. Compute the observed composed output.
        try:
            composed = compose_and_run(block_diagram, block_goldens, chip_inputs)
        except Exception as exc:  # noqa: BLE001
            violations.append(
                {
                    "type": "composition_gate_failure",
                    "vector_id": vec.get("id", ""),
                    "first_divergence_block": _vector_block(vec),
                    "expected": "",
                    "observed": f"composition raised {type(exc).__name__}: {exc}",
                    "suggested_fix": (
                        "A block golden raised during composition; inspect "
                        f"block '{_vector_block(vec)}' golden math/representation."
                    ),
                }
            )
            continue

        composed_flat = _flatten_single(composed)

        # 3. Determine the EXPECTED via the oracle policy.
        if entry_callable is not None:
            # Reference implementation IS the oracle.
            expected = _run_reference(entry_callable, ref_stim)
            if expected is None:
                # Reference could not be run on this stimulus -- skip (logged).
                logger.info(
                    "composition gate: reference returned None for vector %s "
                    "-- skipping", vec.get("id", ""),
                )
                continue
            suggested_fix = (
                "composed block-goldens diverge from the reference "
                "implementation -- fix the named block's golden"
            )
        else:
            # No reference -> fall back to the structured expected, else the
            # prose-computed expected (objective-math designs).
            exp_struct = vec.get("expected_struct")
            if exp_struct is not None:
                expected = exp_struct
            else:
                expected = vec.get("expected_output", "")
            if expected in ("", None):
                # No oracle available at all -- skip.
                continue
            suggested_fix = (
                "Composed block-golden output diverges from the FUNC vector's "
                "expected output. Re-derive the named block's golden math "
                "(close any placeholder)."
            )

        # 4. Compare bit-exact; localize first divergence on mismatch.
        if not _outputs_match(composed_flat, expected):
            violations.append(
                {
                    "type": "composition_gate_failure",
                    "vector_id": vec.get("id", ""),
                    "first_divergence_block": _localize_divergence(
                        block_diagram, block_goldens, chip_inputs, vec
                    ),
                    "expected": expected,
                    "observed": composed_flat,
                    "suggested_fix": suggested_fix,
                }
            )

    return violations


def _stimulus_to_chip_inputs(stim: Any, block_diagram: dict) -> dict | None:
    """Coerce a FUNC vector stimulus into a ``compose_and_run`` chip_inputs map.

    Accepts:
      - dict: used as-is ({chip_port: value-or-list}).
      - list/scalar: bound to the first chip-ingress port name if one can be
        determined, else returns None (cannot drive deterministically).
    """
    def _numeric(v: Any) -> bool:
        if isinstance(v, bool):
            return False
        if isinstance(v, int):
            return True
        if isinstance(v, list):
            return all(_numeric(x) for x in v)
        return False

    if isinstance(stim, dict):
        # Only a fully-numeric {chip_port: value-or-list} mapping can be driven
        # deterministically; anything else is prose -> skip.
        if stim and all(_numeric(v) for v in stim.values()):
            return stim
        return None

    # A prose / non-numeric stimulus (the common case for FRD vectors written
    # in English) cannot be driven by the composition harness -- return None so
    # run_composition_gate SKIPS it (it stays covered by per-block DV) instead
    # of crashing. Only a bare int or list-of-ints binds to an ingress port.
    if not _numeric(stim):
        return None

    # Determine the chip's primary ingress port. A chip ingress edge is a
    # connection whose `from` is not a block. If none, use the first block's
    # first input port.
    block_names = {b.get("name") for b in block_diagram.get("blocks", [])}
    for c in block_diagram.get("connections", []):
        src = c.get("from", c.get("from_block", ""))
        if src and src not in block_names:
            port = c.get("from") or "in"
            return {port: stim}
    # Fall back to the first block's first input port.
    blocks = block_diagram.get("blocks", [])
    if blocks:
        ifaces = blocks[0].get("interfaces", {}) or {}
        in_ports = [
            p
            for p, info in ifaces.items()
            if (isinstance(info, dict) and info.get("direction") == "input")
        ] or list(ifaces.keys())
        if in_ports:
            return {in_ports[0]: stim}
    return None


def _flatten_single(composed: dict) -> Any:
    """If the composed output has exactly one port with one value, unwrap it.

    Keeps comparison robust to single-output designs where the FUNC vector's
    expected output is a bare value rather than ``{port: [value]}``.
    """
    if isinstance(composed, dict) and len(composed) == 1:
        (vals,) = composed.values()
        if isinstance(vals, list) and len(vals) == 1:
            return vals[0]
        if isinstance(vals, list):
            return vals
    return composed


def _deep_equal(a, b) -> bool:
    """Structural deep equality that handles dict / numpy ndarray / bytes / list
    leaves (the model-integration gate output is e.g. {bitstream: bytes, recon:
    [ndarray], stats: {...}}). Byte-EXACT: arrays via np.array_equal, bytes and
    lists elementwise. Bytes are normalised to list[int] so bytes==list[int] of
    the same values compares equal."""
    try:
        import numpy as _np
    except ImportError:
        _np = None
    if isinstance(a, (bytes, bytearray)):
        a = list(a)
    if isinstance(b, (bytes, bytearray)):
        b = list(b)
    if _np is not None and (isinstance(a, _np.ndarray) or isinstance(b, _np.ndarray)):
        try:
            aa = _np.asarray(a); bb = _np.asarray(b)
            return aa.shape == bb.shape and bool(_np.array_equal(aa, bb))
        except Exception:
            return False
    if isinstance(a, dict) or isinstance(b, dict):
        return (isinstance(a, dict) and isinstance(b, dict)
                and set(a.keys()) == set(b.keys())
                and all(_deep_equal(a[k], b[k]) for k in a))
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if not (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple))):
            return False
        return len(a) == len(b) and all(_deep_equal(x, y) for x, y in zip(a, b))
    try:
        return bool(a == b)
    except Exception:
        return False


def _outputs_match(observed: Any, expected: Any) -> bool:
    """Structural equality with light coercion for ints/lists/strings."""
    if _deep_equal(observed, expected):
        return True
    # Coerce single-element list vs scalar.
    if isinstance(observed, list) and len(observed) == 1 and _deep_equal(observed[0], expected):
        return True
    if isinstance(expected, list) and len(expected) == 1 and _deep_equal(expected[0], observed):
        return True
    # Coerce stringified ints.
    try:
        if int(observed) == int(expected):
            return True
    except (TypeError, ValueError):
        pass
    return False


def gate_epsilon() -> float:
    """Tolerance for the FLOAT-output gate path (CORESMITH_GATE_EPSILON, default
    1e-6). Used only when the reference output is float-valued -- bit-exact float
    reproduction across an integer RTL datapath is unrealistic, so the user
    policy is: bias to fixed-point (deterministic, bit-exact) by default, and
    when the golden genuinely outputs floats, accept within epsilon."""
    try:
        return float(os.environ.get("CORESMITH_GATE_EPSILON", "1e-6") or 1e-6)
    except (TypeError, ValueError):
        return 1e-6


def output_has_float(obj: Any) -> bool:
    """True if the reference output contains any floating-point value (Python
    float or numpy floating), recursively. bools/ints are NOT floats. Used to
    decide whether the gate must allow an epsilon tolerance vs require bit-exact.
    """
    if isinstance(obj, bool):
        return False
    if isinstance(obj, float):
        return True
    dtype = getattr(obj, "dtype", None)
    if dtype is not None:
        try:
            import numpy as _np
            return bool(_np.issubdtype(dtype, _np.floating)) or (
                bool(_np.issubdtype(dtype, _np.complexfloating))
            )
        except Exception:  # noqa: BLE001
            return False
    if isinstance(obj, (list, tuple)):
        return any(output_has_float(x) for x in obj)
    if isinstance(obj, dict):
        return any(output_has_float(v) for v in obj.values())
    return False


def _flatten_numbers(obj: Any) -> list:
    """Flatten a nested output into a flat list of leaves (numbers kept as
    numbers; non-numeric leaves kept for strict !=)."""
    out: list = []

    def _rec(x: Any) -> None:
        if isinstance(x, bool):
            out.append(int(x))
        elif isinstance(x, (int, float)):
            out.append(x)
        elif hasattr(x, "tolist") and not isinstance(x, (str, bytes)):
            _rec(x.tolist())  # numpy array/scalar
        elif isinstance(x, (list, tuple)):
            for y in x:
                _rec(y)
        elif isinstance(x, dict):
            for y in x.values():
                _rec(y)
        else:
            out.append(x)

    _rec(obj)
    return out


def outputs_close(observed: Any, expected: Any, eps: float) -> bool:
    """Structure-aware numeric closeness: same flattened length, numeric leaves
    within ``abs(a-b) <= eps + eps*abs(b)`` (combined abs+rel), non-numeric
    leaves bytewise-equal. The epsilon comparison for float-output designs."""
    a = _flatten_numbers(observed)
    b = _flatten_numbers(expected)
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            if abs(x - y) > eps + eps * abs(y):
                return False
        elif x != y:
            return False
    return True


def _normalize_ref_output(value: Any) -> Any:
    """Normalise a reference return into a comparable structure.

    Tuples become lists (JSON-comparable); everything else is returned as-is.
    Nested tuples are handled recursively.
    """
    if isinstance(value, tuple):
        return [_normalize_ref_output(v) for v in value]
    if isinstance(value, list):
        return [_normalize_ref_output(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize_ref_output(v) for k, v in value.items()}
    return value


class ReferenceInvocationError(RuntimeError):
    """The reference entry callable could not be invoked on the stimulus.

    Raised by :func:`_run_reference` when ``reraise=True`` and the call fails
    (bad signature, wrong stimulus type, exception inside the reference). The
    model-integration gate treats this as a HARD failure: a reference that
    cannot be invoked provides no oracle, so the run must NOT pass vacuously.
    """


def _run_reference(
    entry_callable: Optional[Callable],
    stimulus_struct: Any,
    *,
    reraise: bool = False,
) -> Any:
    """Run the reference entry callable on a structured stimulus.

    ``stimulus_struct`` is normally a ``{chip_port: value-or-list}`` dict. It is
    mapped to the callable's signature by parameter NAME where keys match
    declared parameters; otherwise the values are passed positionally in port
    order. A non-dict stimulus is passed as a single positional argument.

    On failure (bad signature, wrong stimulus type, exception inside the
    reference): when ``reraise`` is False (legacy/v1 path) returns ``None``;
    when True (the model-integration gate) raises
    :class:`ReferenceInvocationError` so the caller surfaces a HARD violation
    instead of silently treating "no oracle" as a pass.
    The successful return is normalised via :func:`_normalize_ref_output`.
    """
    if entry_callable is None:
        return None
    try:
        if isinstance(stimulus_struct, dict):
            try:
                sig = inspect.signature(entry_callable)
                param_names = [
                    p.name
                    for p in sig.parameters.values()
                    if p.kind
                    in (
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    )
                ]
                accepts_var_kw = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in sig.parameters.values()
                )
            except (TypeError, ValueError):
                param_names = []
                accepts_var_kw = False

            keys = list(stimulus_struct.keys())
            # By-name mapping when the param names line up with stimulus keys.
            if param_names and (
                accepts_var_kw or set(keys).issubset(set(param_names))
            ) and all(k in param_names for k in keys if not accepts_var_kw):
                if set(keys) & set(param_names) or accepts_var_kw:
                    result = entry_callable(**stimulus_struct)
                    return _normalize_ref_output(result)

            # Single-parameter callable that wants the whole dict (e.g.
            # ``def run(stim): ...``) -- pass the dict as one positional arg.
            if len(param_names) == 1 and param_names[0] not in keys:
                result = entry_callable(stimulus_struct)
                return _normalize_ref_output(result)

            # Otherwise pass values positionally in port (insertion) order.
            result = entry_callable(*[stimulus_struct[k] for k in keys])
            return _normalize_ref_output(result)

        # Non-dict stimulus: single positional argument.
        result = entry_callable(stimulus_struct)
        return _normalize_ref_output(result)
    except Exception as exc:  # noqa: BLE001 - reference signature/behaviour
        logger.warning(
            "composition gate: reference entry call failed: %s: %s",
            type(exc).__name__,
            exc,
        )
        if reraise:
            raise ReferenceInvocationError(
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return None


def _localize_divergence(
    block_diagram: dict,
    block_goldens: dict[str, Any],
    chip_inputs: dict,
    vec: dict,
) -> str:
    """Best-effort first-divergence block naming.

    If the FUNC vector names a block, trust that (it is the authored owner of
    the vector). Otherwise return the empty string (caller reports the
    composed-output mismatch without a specific block). A finer-grained
    per-block-vs-reference-intermediate comparison would require the reference
    implementation to expose intermediates, which is not part of the contract.
    """
    named = _vector_block(vec)
    if named and named in block_goldens:
        return named
    return named or ""
