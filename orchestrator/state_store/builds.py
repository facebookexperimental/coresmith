# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Recorded module builds: identity, provenance, evidence and lineage.

A *build* is one recorded attempt to take one module through the existing
block LangGraph (uArch review -> RTL -> lint -> assertions -> testbench + DV
+ coverage -> synthesis + timing -> done). The ``builds`` table in the
project database is the Architect's experiment ledger:

* ``inputs_json`` -- the *architecture inputs* the build started from, as
  content hashes: the registered uArch spec, the module's contract edges and
  their version, the bound build target's *configuration* (top, sources,
  includes, defines, parameters, assets -- never the RTL bytes, which the
  build itself writes), the SystemC reference model with the headers it
  compiles against and the FRD harness, the requirement items the module
  owns, the resolved worker binding and the tool/PDK/clock context. The
  engine revision and an explicit seed are recorded as starting context, not
  as staleness axes: a repair of a seeded implementation in the normal loop
  is a valid new result, and the record says the seed was modified.
* ``result_json`` -- the terminal evidence: the measurement rows recorded
  under the build id, the produced implementation identity (the complete
  final target revision -- sources, include-dir headers, discovered
  ``$readmem`` assets -- plus the testbench and the synthesis reports), the
  graph thread and checkpoint namespace the completion ran under.

Rows are never overwritten: a new attempt is a new row, and the published
``best`` of a module is a *selection* that points at the build it came from.
A build's evidence is *stale* when any architecture input it recorded no
longer matches the live project, or when any produced file changed after the
build; the row and its results stay as history.

Nothing here opens a LangGraph or runs a tool; this module is imported by
``project_db`` at schema time and must stay free of graph imports at module
level.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

BUILDS_SCHEMA = """
CREATE TABLE IF NOT EXISTS builds (
    id TEXT PRIMARY KEY,
    module TEXT NOT NULL,
    run_id TEXT NOT NULL DEFAULT '',
    entry TEXT NOT NULL,                 -- build_module | run_start | restart_block
    graph TEXT NOT NULL,                 -- build | pipeline
    thread_id TEXT NOT NULL DEFAULT '',
    checkpoint_ns TEXT,
    status TEXT NOT NULL,                -- dispatched | running | parked | completed | failed | failed_persistence | error | aborted
    requested_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL,
    inputs_json TEXT NOT NULL,
    worker_json TEXT,
    seed_json TEXT,
    result_json TEXT,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_builds_module ON builds(module, requested_at);
CREATE TABLE IF NOT EXISTS build_candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    build_id TEXT NOT NULL,
    module TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    ts REAL NOT NULL,
    outcome TEXT NOT NULL,               -- feasible | target_miss | unmeasured
    evaluation_json TEXT NOT NULL,       -- targets / functional: values, gaps, tool receipts
    files_json TEXT                      -- the candidate implementation {path: sha256}
);
CREATE INDEX IF NOT EXISTS idx_candidates_build ON build_candidates(build_id, attempt);
"""

BUILD_ACTIVE = ("dispatched", "running", "parked")
BUILD_TERMINAL = ("completed", "failed", "failed_persistence", "error", "aborted")
BUILD_STATUSES = BUILD_ACTIVE + BUILD_TERMINAL
BUILD_ENTRIES = ("build_module", "run_start", "restart_block")

# The architecture-input axes a build's evidence is bound to. A change on any
# of them makes the evidence stale; the row stays.
_IDENTITY_AXES = (
    ("spec", "the registered uArch spec"),
    ("contracts", "the module's interface contract edges"),
    ("target", "the bound build target's configuration"),
    ("model", "the SystemC reference model or its headers"),
    ("harness", "the FRD evaluation harness"),
    ("items", "the requirement items the module owns"),
    ("targets", "the module's FRD target allocation (bounds, units, priorities, measurement bindings and conditions)"),
    ("acceptance", "the acceptance testbench (the build's fixed oracle) or a module it imports"),
    ("worker", "the worker binding"),
    ("tooling", "the tool, PDK, clock or workload context"),
)

_ITEM_IDENTITY_FIELDS = ("text", "kind", "bound_min", "bound_max", "unit", "acceptance", "model_check", "metric")


# ---------------------------------------------------------------- hashing
def file_sha256(path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _canon_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _j(value: Any) -> str:
    return json.dumps(value, default=str, sort_keys=True)


def _uj(text: str | None, default: Any):
    if not text:
        return default
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return default


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def new_build_id(module: str) -> str:
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    return f"b-{module}-{stamp}-{uuid.uuid4().hex[:6]}"


def is_primitive_spec(spec: dict | None) -> bool:
    spec = spec or {}
    return bool(spec.get("primitive")) or str(spec.get("kind") or "").lower() == "primitive"


def _block_spec(db, name: str) -> dict | None:
    try:
        for b in db.block_specs():
            if b.get("name") == name:
                return b
    except Exception:  # noqa: BLE001
        return None
    return None


# ---------------------------------------------------------------- identities
_INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.M)
_MAX_MODEL_DEPS = 400


def _model_dependencies(cpp: Path, model_dir: Path) -> tuple[dict[str, str | None], list[str], str | None]:
    """The local headers ``cpp`` compiles against, found by following quoted
    ``#include`` directives from the file (relative to the including file,
    then to the model directory -- the ``-I.`` of the generated Makefile).
    Returns ``({path: sha256}, unresolved_includes, error)``; a file that
    cannot be read is an error, never an empty valid identity."""
    deps: dict[str, str | None] = {}
    unresolved: list[str] = []
    todo = [cpp]
    seen: set[str] = set()
    try:
        while todo:
            cur = todo.pop(0)
            key = str(cur.resolve())
            if key in seen:
                continue
            seen.add(key)
            if len(seen) > _MAX_MODEL_DEPS:
                return deps, unresolved, f"more than {_MAX_MODEL_DEPS} model dependencies"
            text = cur.read_text(encoding="utf-8", errors="replace")
            if cur != cpp:
                deps[str(cur)] = hashlib.sha256(text.encode("utf-8", "surrogateescape")).hexdigest()
            for inc in _INCLUDE_RE.findall(text):
                cand = None
                for base in (cur.parent, model_dir):
                    p = base / inc
                    if p.is_file():
                        cand = p
                        break
                if cand is None:
                    if inc not in unresolved:
                        unresolved.append(inc)
                    continue
                todo.append(cand)
    except OSError as exc:
        return deps, unresolved, f"{type(exc).__name__}: {exc}"
    return dict(sorted(deps.items())), sorted(unresolved), None


def model_identity(root, name: str, db=None) -> dict:
    """The SystemC reference model of ``name`` as ``coresmith model build``
    records it: ``sha16`` is the first 16 hex digits of the SHA-256 of the
    implementation (the ``models.sha`` column uses exactly this
    construction); ``deps_sha256`` covers the implementation together with
    every local header it includes (``models.deps_sha``). The implementation
    is the registered ``models.path`` when a row names one, else the
    convention ``model/<name>_model.cpp``."""
    from orchestrator.systemc_model.conventions import model_name
    root = Path(root)
    md = root / "model"
    cp = md / f"{model_name(name)}.cpp"
    source = "convention"
    row = None
    if db is not None and hasattr(db, "model_for"):
        try:
            row = db.model_for(name)
        except Exception:  # noqa: BLE001
            row = None
    if row and str(row.get("path") or "").strip():
        reg = Path(str(row["path"]))
        cp = reg if reg.is_absolute() else (root / reg)
        source = "registered"
    hp = md / f"{model_name(name)}.h"
    cpp_sha = file_sha256(cp)
    deps, unresolved, error = _model_dependencies(cp, md) if cpp_sha else ({}, [], None)
    header_sha = file_sha256(hp)
    if header_sha is not None:
        deps.setdefault(str(hp), header_sha)
    return {"path": str(cp), "path_source": source, "exists": cpp_sha is not None,
            "sha16": cpp_sha[:16] if cpp_sha else "", "sha256": cpp_sha,
            "header_path": str(hp), "header_sha256": header_sha,
            "deps": deps, "unresolved_includes": unresolved, "error": error,
            "deps_sha256": _canon_sha({"cpp": cpp_sha, "deps": deps}) if cpp_sha else ""}


def model_sha16(root, name: str, db=None) -> str:
    """The value ``models.sha`` holds for ``name`` ('' when the file is absent)."""
    return model_identity(root, name, db)["sha16"]


def model_deps_sha(root, name: str, db=None) -> str:
    """The value ``models.deps_sha`` holds for ``name``: implementation + headers."""
    return model_identity(root, name, db)["deps_sha256"]


def soc_model_digest(root, names: list[str] | None = None, db=None) -> str:
    """One digest over the assembled SoC model's inputs, built from the same
    dependency manifests the per-model identity uses: every block model's
    implementation (the registered path or the convention) with every local
    header it includes, the generated top and common headers, and the FRD
    harness sources with their includes. A declared model check's verdict is
    bound to it (with the requirement it judged, see :func:`model_check_sha`).
    ``names`` defaults to the registered blocks (``db``) or, without a
    database, to the conventional ``model/*_model.cpp`` files."""
    root = Path(root)
    md = root / "model"
    files: dict[str, Any] = {}
    if names is None:
        if db is not None and hasattr(db, "block_specs"):
            try:
                names = [b["name"] for b in db.block_specs()]
            except Exception:  # noqa: BLE001
                names = None
        if names is None:
            names = sorted(p.name[:-len("_model.cpp")] for p in md.glob("*_model.cpp")) if md.is_dir() else []
    for n in sorted(set(names)):
        ident = model_identity(root, n, db)
        files[f"model:{n}"] = {"path": ident["path"], "sha256": ident["sha256"], "error": ident["error"]}
        files.update(ident["deps"])
    for fname in ("soc_model_top.h", "cs_model_common.h"):
        if (md / fname).is_file():
            files[str(md / fname)] = file_sha256(md / fname)
    files.update(harness_identity(root)["files"])
    return _canon_sha(files)


def item_digest(it: dict | None) -> str:
    """The identity of a requirement item as a verdict's input: text, kind,
    bounds, unit, acceptance, declared model check, metric."""
    it = it or {}
    return _canon_sha({k: it.get(k) for k in _ITEM_IDENTITY_FIELDS})


def model_check_sha(db, root, item_id: str, *, model_digest: str | None = None) -> str:
    """What a ``model_eval`` verdict for ``item_id`` is bound to: the assembled
    model + harness digest AND the requirement as it was judged (its text and
    bounds). A bound change re-evaluates; a model change re-evaluates."""
    it = None
    if db is not None and hasattr(db, "item"):
        try:
            it = db.item(item_id)
        except Exception:  # noqa: BLE001
            it = None
    return _canon_sha({"model": model_digest or soc_model_digest(root, db=db), "item": item_digest(it)})


def model_check_scope(db, parsed: list[dict] | None = None) -> list[str]:
    """The ids of the live must-have requirements that *declare* a model
    check: the exact scope the SoC-model evaluation is required for. The
    registered ontology item decides for an id it knows; an id the database
    does not hold (an FRD evaluated before registration) is read from the
    parsed FRD (``parsed``: ``frd_eval.extract_requirements``). Other
    requirements keep their RTL / chip-level acceptance."""
    from orchestrator.state_store.ontology import item_must_have
    known: dict[str, dict] = {}
    if db is not None:
        try:
            known = {it["id"]: it for it in db.items(artifact="frd")}
        except Exception:  # noqa: BLE001
            known = {}
    out = set()
    for iid, it in known.items():
        if not item_must_have(it) or it.get("status") in ("retired", "waived"):
            continue
        if str(it.get("model_check") or "").strip():
            out.add(iid)
    for q in parsed or []:
        qid = str(q.get("id") or "")
        if not qid or qid in known:
            continue
        prio = str(q.get("priority") or "").lower()
        if ("must" in prio or prio in ("hard", "p0", "required")) and str(q.get("model_check") or "").strip():
            out.add(qid)
    return sorted(out)


def harness_identity(root) -> dict:
    """The FRD harness (``model/frd_eval/*.cpp|*.h``) with every local header
    its sources include (resolved next to the source, then in the model
    directory: the harness compiles with ``-I. -Ifrd_eval``)."""
    md = Path(root) / "model"
    fe = md / "frd_eval"
    srcs = sorted(fe.glob("*.cpp")) + sorted(fe.glob("*.h")) if fe.is_dir() else []
    files: dict[str, str | None] = {}
    errors: list[str] = []
    for p in srcs:
        files[str(p)] = file_sha256(p)
        deps, _unresolved, err = _model_dependencies(p, md)
        for dep, sha in deps.items():
            files.setdefault(dep, sha)
        if err:
            errors.append(f"{p.name}: {err}")
    return {"sources": [str(p) for p in srcs], "files": files, "errors": errors,
            "sha256": _canon_sha({"files": files, "errors": errors}) if srcs else None}


def target_identity(root, name: str) -> dict | None:
    """The bound build target of ``name`` (None when unbound): its
    configuration digest (top, sources, cwd, includes, defines, parameters,
    assets) -- the architecture input -- plus, as context, the per-file
    hashes of the sources that exist and the full revision
    (``targets.revision``: configuration + sources + dependencies + assets)
    when every file exists. The bytes are the implementation, which the
    build writes; they are judged by :func:`produced_outputs` after it."""
    from orchestrator.harness.targets import load, revision
    try:
        target = load(root, name, require_files=False)
    except (ValueError, OSError) as exc:
        return {"name": name, "error": str(exc)}
    if target is None:
        return None
    files = list(target.get("sources") or []) + list(target.get("assets") or [])
    hashes = {p: file_sha256(p) for p in files}
    present = all(v is not None for v in hashes.values())
    out = {"name": name, "top": target.get("top"), "cwd": target.get("cwd"), "sources": list(target.get("sources") or []),
           "include_dirs": list(target.get("include_dirs") or []), "assets": list(target.get("assets") or []),
           "config_sha256": _canon_sha(target), "files_sha256": hashes, "sources_present": present,
           "revision": None, "revision_error": None}
    if present:
        try:
            out["revision"] = revision(target)
        except (OSError, ValueError, RuntimeError) as exc:  # noqa: BLE001 - dependency discovery failed
            out["revision_error"] = str(exc)[:300]
    else:
        out["revision_error"] = "source files not present at dispatch (fresh build)"
    return out


def module_contracts(db, name: str) -> dict:
    edges = []
    try:
        for c in (db.contracts() or {}).get("contracts") or []:
            if name in (c.get("producer_block"), c.get("consumer_block")):
                edges.append(c)
    except Exception:  # noqa: BLE001 - no contracts registered
        edges = []
    edges.sort(key=lambda c: str(c.get("edge_id") or ""))
    return {"edge_ids": [c.get("edge_id") for c in edges], "sha256": _canon_sha(edges),
            "version": int(db.block_contract_version(name) or 0) if hasattr(db, "block_contract_version") else 0}


def owned_items(db, name: str) -> list[str]:
    out: list[str] = []
    for lk in db.links(to_id=f"block:{name}", rel="owned_by"):
        if lk["from_id"] not in out:
            out.append(lk["from_id"])
    return out


def owned_item_digests(db, name: str) -> dict[str, str]:
    """``{item_id: digest}`` of every live must-have item the module owns
    (text, kind, bounds, unit, acceptance, declared model check)."""
    from orchestrator.state_store.ontology import item_must_have
    out: dict[str, str] = {}
    for iid in owned_items(db, name):
        it = db.item(iid)
        if not it or not item_must_have(it) or it.get("status") in ("retired", "waived"):
            continue
        out[iid] = item_digest(it)
    return out


_PROVIDER_SHORT = {"claude_cli": "claude", "codex_cli": "codex", "opencode_cli": "opencode", "kimi_cli": "kimi",
                   "agy_cli": "agy"}
_TESTING_PROVIDERS = ("fault", "replay")


def worker_binding(root) -> dict:
    """The worker (LLM) binding the engine's helpers would run with, resolved
    by the adapter itself: the persisted ``.coresmith/env`` is applied over
    the live process environment (the same persisted-wins rule as
    ``run_env.apply_persisted_env``) and handed to
    ``coresmith_llm._detect_provider`` / ``_resolve_model`` as a mapping, so
    ``model`` is the exact model id the provider call would use (aliases
    resolved through the adapter's catalogue) without mutating the process
    environment. Only the provider and the deciding selector are recorded,
    nothing else of either environment. ``explicit`` is True only when the
    provider and the selector that decides its model are declared in the
    persisted file (or the provider is a test-only backend, which needs no
    model)."""
    from orchestrator.langchain.agents import coresmith_llm as L
    from orchestrator.run_env import load_persisted_env
    try:
        persisted = load_persisted_env(root)
    except Exception:  # noqa: BLE001
        persisted = {}
    effective = {**os.environ, **persisted}
    raw = (effective.get("CORESMITH_LLM_PROVIDER") or "").strip().lower()
    source = "persisted" if (persisted.get("CORESMITH_LLM_PROVIDER") or "").strip() else ("environment" if raw else "default")
    error = None
    try:
        provider_key = L._detect_provider(effective)
    except ValueError as exc:
        provider_key, error = "", str(exc)[:200]
    testing = provider_key in _TESTING_PROVIDERS
    provider = _PROVIDER_SHORT.get(provider_key, provider_key)
    model_var, declared = L.selected_model_var(provider_key or "claude_cli", effective) if provider_key else ("", "")
    model = ""
    if provider_key and not testing:
        try:
            model = L._resolve_model("", provider_key, effective)
        except ValueError as exc:
            error = error or str(exc)[:200]
    model_source = ("persisted" if (persisted.get(model_var) or "").strip() else "environment") if model_var else ""
    candidates = L.MODEL_SELECTOR_VARS.get(provider_key, ()) if provider_key else ("CORESMITH_LLM_PROVIDER",)
    missing = []
    if source != "persisted":
        missing.append("CORESMITH_LLM_PROVIDER")
    if not testing and model_source != "persisted":
        missing.append(" | ".join(candidates) if candidates else "a model selector")
    if error:
        missing.append(error)
    return {"provider": provider, "provider_key": provider_key, "provider_declared": raw, "provider_source": source,
            "model": model, "model_declared": declared, "model_var": model_var, "model_source": model_source,
            "testing_provider": testing, "explicit": not missing, "missing": missing, "error": error,
            "env_file": str(Path(root) / ".coresmith" / "env")}


_TOOL_VERSION_CACHE: dict[tuple, dict] = {}
# The flag each tool prints its version with (OpenSTA does not know --version:
# it would open its shell and print a splash line).
_VERSION_FLAG = {"sta": "-version"}
TOOL_PATHS = Path(".coresmith") / "tool_paths.json"


def _tool_env(root) -> dict:
    """The environment a tool is resolved and queried in: the persisted
    ``.coresmith/env`` over the process environment (the daemon applies the
    same file)."""
    env = dict(os.environ)
    if root is not None:
        try:
            from orchestrator.run_env import load_persisted_env
            env.update(load_persisted_env(root))
        except Exception:  # noqa: BLE001
            pass
    return env


def tool_authority() -> bool:
    """Whether this process runs the tools (the daemon sets
    ``CORESMITH_TOOL_AUTHORITY=1``): it identifies them by its own resolution
    and records the resolved paths for every other process."""
    return (os.environ.get("CORESMITH_TOOL_AUTHORITY", "") or "").strip().lower() in {"1", "true", "yes", "on"}


def recorded_tool_paths(root) -> dict[str, str]:
    """``{tool: resolved path}`` the tool-running process last recorded."""
    try:
        data = json.loads((Path(root) / TOOL_PATHS).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return {k: str(v) for k, v in (data.get("paths") or {}).items() if v} if isinstance(data, dict) else {}


def _resolve_tool(binary: str, env: dict) -> tuple[str | None, str | None]:
    """``(path, error)`` of the binary that actually runs: ``binary`` on the
    PATH of ``env``; the engine's ``bin/sta`` shim is followed to the OpenSTA
    it execs (``CORESMITH_REAL_STA``, default ``sta-real``)."""
    path = shutil.which(binary, path=env.get("PATH"))
    if not path:
        return None, f"{binary} not on PATH"
    if binary == "sta":
        try:
            with open(path, "rb") as fh:
                head = fh.read(4096)
        except OSError:
            head = b""
        if b"CoreSmith OpenSTA compatibility shim" in head:
            real = (env.get("CORESMITH_REAL_STA") or "sta-real").strip()
            resolved = shutil.which(real, path=env.get("PATH")) if real else None
            if not resolved:
                return None, f"the engine's sta shim execs CORESMITH_REAL_STA={real or '<empty>'}, which is not available"
            path = resolved
    return str(Path(path).resolve()), None


def _tool_version(binary: str, root=None, *, own: bool = False) -> dict:
    """The identity of the tool that runs as ``binary``: ``{path, sha256,
    version, error, resolved_by}``. The tool-running process (``own`` or
    :func:`tool_authority`) resolves it on its own PATH; any other process
    identifies the binary the tool-running process RECORDED (so a caller's
    PATH never changes the identity), falling back to its own resolution
    when nothing is recorded. ``sha256`` is the binary's content hash;
    ``version`` is the first line of a SUCCESSFUL version query only (run in
    the same environment the binary was resolved in) -- an error text is
    never a version. An absent tool has ``path``/``sha256``/``version`` None
    and the reason in ``error``. Never raises."""
    env = _tool_env(root)
    path, error, how = None, None, "path"
    if not own and not tool_authority() and root is not None:
        rec = recorded_tool_paths(root).get(binary)
        if rec:
            if not Path(rec).is_file():
                return {"path": rec, "sha256": None, "version": None, "resolved_by": "recorded",
                        "error": f"the recorded {binary} binary {rec} no longer exists"}
            path, how = rec, "recorded"
    if path is None:
        path, error = _resolve_tool(binary, env)
    if not path:
        return {"path": None, "sha256": None, "version": None, "error": error, "resolved_by": how}
    try:
        st_ = os.stat(path)
        key = (path, st_.st_mtime_ns, st_.st_size)
    except OSError as exc:
        return {"path": path, "sha256": None, "version": None, "error": str(exc)[:200], "resolved_by": how}
    if key not in _TOOL_VERSION_CACHE:
        version, err = None, None
        try:
            p = subprocess.run([path, _VERSION_FLAG.get(binary, "--version")], capture_output=True, text=True,
                               timeout=20, stdin=subprocess.DEVNULL, env=env)
            line = ((p.stdout or p.stderr or "").strip().splitlines() or [""])[0][:120]
            if p.returncode == 0 and line:
                version = line
            else:
                err = f"version query exited {p.returncode}: {line}"[:200]
        except (OSError, subprocess.TimeoutExpired) as exc:
            err = f"version query failed: {type(exc).__name__}"
        _TOOL_VERSION_CACHE[key] = {"path": path, "sha256": file_sha256(path), "version": version, "error": err}
    return {**_TOOL_VERSION_CACHE[key], "resolved_by": how}


def _tool_version_text(entry: Any) -> str | None:
    """A recorded tool's version when it is one (records written before the
    binary hash stored an error text or ``unavailable`` as the version)."""
    v = entry.get("version") if isinstance(entry, dict) else entry
    v = str(v or "").strip()
    low = v.lower()
    if not v or not re.search(r"\d", v) or any(w in low for w in ("unavailable", "unknown", "error", "not found",
                                                                   "coresmith_real_sta")):
        return None
    return v


def _tool_known(entry: Any) -> bool:
    return bool((entry.get("sha256") if isinstance(entry, dict) else None) or _tool_version_text(entry))


def _same_tool(a: Any, b: Any) -> bool | None:
    """Whether two recorded identities of one tool name the same tool: True
    only for a POSITIVE match (the binary hashes when both sides have one,
    else the versions), False for a difference, None when either side could
    not identify the tool -- unknown is never a match."""
    sa = a.get("sha256") if isinstance(a, dict) else None
    sb = b.get("sha256") if isinstance(b, dict) else None
    if sa and sb:
        return sa == sb
    va, vb = _tool_version_text(a), _tool_version_text(b)
    if va and vb:
        return va == vb
    return None


def _engine_liberty() -> dict:
    """The liberty file and PDK root the engine's synthesis actually uses
    (``pipeline_helpers.LIBERTY_FILE`` / ``PDK_ROOT``): one resolver, the
    engine's. Reported as unavailable -- never guessed -- when the engine
    helpers cannot be imported."""
    try:
        from orchestrator.langgraph import pipeline_helpers as ph
    except Exception as exc:  # noqa: BLE001
        return {"liberty": None, "liberty_sha256": None, "pdk": None, "pdk_root": None,
                "error": f"engine PDK resolution unavailable: {type(exc).__name__}"}
    lib = Path(getattr(ph, "LIBERTY_FILE", "") or "")
    present = bool(str(lib)) and lib.is_file()
    return {"liberty": str(lib) if present else None, "liberty_sha256": file_sha256(lib) if present else None,
            "pdk": lib.parents[3].name if present and len(lib.parents) > 3 else ("generic (no liberty)"),
            "pdk_root": str(getattr(ph, "PDK_ROOT", "") or "") or None, "error": None}


_TOOLS = ("verilator", "yosys", "sta")


def tooling_snapshot(root, *, target_clock_mhz: float | None, persist: bool = False) -> dict:
    """The tool / PDK / clock context. ``persist``: this process is about to
    run the tools (a build dispatch in the daemon): identify them by its own
    resolution and record the resolved paths (``.coresmith/tool_paths.json``)
    every other process identifies them by."""
    generic = (os.environ.get("CORESMITH_SYNTH_GENERIC", "") or "").strip().lower() in {"1", "true", "yes", "on"}
    tools = {tool: _tool_version(tool, root, own=persist) for tool in _TOOLS}
    if persist and root is not None:
        try:
            p = Path(root) / TOOL_PATHS
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({"paths": {k: v.get("path") for k, v in tools.items() if v.get("path")},
                                       "recorded_at": time.time(), "pid": os.getpid()}, indent=2))
            os.replace(tmp, p)
        except OSError:
            pass
    return {**tools, **_engine_liberty(), "synth_generic": generic,
            "clock_mhz": target_clock_mhz, "workload": None}


def engine_revision() -> str:
    try:
        from orchestrator.utils import engine_git_sha
        return engine_git_sha() or ""
    except Exception:  # noqa: BLE001
        return ""


def module_inputs(db, root, name: str, *, seed_path: str | None = None,
                  target_clock_mhz: float | None = None, persist_tools: bool = False) -> dict:
    """The architecture-input identity of module ``name`` right now. An
    engine primitive's uArch spec is generated by its build (not an input)
    and is recorded as such."""
    from orchestrator.state_store import module_targets as MT
    root = Path(root)
    spec = root / "arch" / "uarch_specs" / f"{name}.md"
    primitive = is_primitive_spec(_block_spec(db, name))
    model = model_identity(root, name, db)
    row = db.model_for(name) if hasattr(db, "model_for") else None
    alloc = MT.allocation(db, root, name)
    inputs = {
        "module": name,
        "primitive": primitive,
        "spec": (None if primitive else {"path": str(spec), "sha256": file_sha256(spec)}),
        "contracts": module_contracts(db, name),
        "target": target_identity(root, name),
        "model": {**model, "row": ({k: row.get(k) for k in ("sha", "deps_sha", "spec_contract_version", "build_ok",
                                                            "smoke_ok", "ts")} if row else None),
                  "soc_digest": soc_model_digest(root, db=db)},
        "harness": harness_identity(root),
        "items": owned_item_digests(db, name),
        # the FRD targets and the fixed acceptance oracle the build is judged by
        "targets": {k: alloc[k] for k in ("targets", "functional", "deferred", "unverified", "frd_revision", "digest")},
        "acceptance": alloc["acceptance"],
        "worker": worker_binding(root),
        "tooling": tooling_snapshot(root, target_clock_mhz=target_clock_mhz, persist=persist_tools),
        "engine_sha": engine_revision(),
        "seed": None,
        "captured_at": time.time(),
    }
    if seed_path:
        sp = Path(seed_path)
        inputs["seed"] = {"path": str(sp), "sha256": file_sha256(sp)}
    inputs["digest"] = identity_digest(inputs)
    return inputs


def _axis_value(inputs: dict, axis: str) -> Any:
    v = inputs.get(axis)
    if axis == "worker":
        return {k: (v or {}).get(k) for k in ("provider", "model")}
    if axis == "tooling":
        t = v or {}
        return {"pdk": t.get("pdk"), "liberty_sha256": t.get("liberty_sha256"), "clock_mhz": t.get("clock_mhz"),
                "workload": t.get("workload"), "synth_generic": t.get("synth_generic"),
                **{tool: ((t.get(tool) or {}).get("sha256") if isinstance(t.get(tool), dict) else None)
                   or _tool_version_text(t.get(tool)) for tool in _TOOLS}}
    if axis == "model":
        return {k: (v or {}).get(k) for k in ("sha16", "deps_sha256", "soc_digest")}
    if axis == "target":
        return {"config_sha256": v.get("config_sha256")} if v else None
    if axis == "contracts":
        return {k: (v or {}).get(k) for k in ("sha256", "version")}
    if axis == "harness":
        return (v or {}).get("sha256")
    if axis == "targets":
        return (v or {}).get("digest")
    if axis == "acceptance":
        from orchestrator.state_store.module_targets import acceptance_digest
        return acceptance_digest(v)
    if axis == "spec":
        return (v or {}).get("sha256") if v else None
    return v


def identity_digest(inputs: dict) -> str:
    return _canon_sha({axis: _axis_value(inputs, axis) for axis, _ in _IDENTITY_AXES})


def stale_reasons(stored: dict, live: dict) -> list[str]:
    """Why evidence bound to ``stored`` does not apply to ``live`` (empty =
    compatible). The target is compared by configuration: its bytes are the
    implementation the build writes, judged by :func:`implementation_changed`
    after completion. The seed is starting context, not an axis."""
    out = []
    for axis, label in _IDENTITY_AXES:
        if axis == "tooling":
            a, b = _axis_value(stored, axis), _axis_value(live, axis)
            changed = [k for k in a if k not in _TOOLS and a.get(k) != b.get(k)]
            unconfirmed = []
            for tool in _TOOLS:
                ta, tb = (stored.get("tooling") or {}).get(tool), (live.get("tooling") or {}).get(tool)
                same = _same_tool(ta, tb)
                if same is False:
                    changed.append(tool)
                elif same is None and _tool_known(ta) != _tool_known(tb):
                    why = (tb or {}).get("error") if isinstance(tb, dict) and _tool_known(ta) else "unidentified when recorded"
                    unconfirmed.append(f"{tool} ({why or 'unidentified'})")
            if changed:
                out.append(f"{axis}: {label} changed since the build ({', '.join(changed)})")
            if unconfirmed:
                out.append(f"{axis}: the tool the evidence was measured with cannot be identified now -- unknown, not "
                           f"a match ({'; '.join(unconfirmed)})")
            continue
        a, b = _axis_value(stored, axis), _axis_value(live, axis)
        if a != b:
            out.append(f"{axis}: {label} changed since the build")
    return out


# ---------------------------------------------------------------- rows
def _row(r: sqlite3.Row | None) -> dict | None:
    if r is None:
        return None
    d = dict(r)
    for k in ("inputs", "worker", "seed", "result"):
        d[k] = _uj(d.pop(f"{k}_json", None), None)
    return d


def record_dispatch(db, *, build_id: str, module: str, entry: str, graph: str, thread_id: str,
                    inputs: dict, worker: dict | None = None, seed: dict | None = None,
                    checkpoint_ns: str | None = None) -> dict:
    """Allocate the build row before anything runs. Raises on failure: a
    build that cannot be recorded is not started."""
    if entry not in BUILD_ENTRIES:
        raise ValueError(f"unknown build entry {entry!r}")
    with db._tx() as con:
        con.execute(
            "INSERT INTO builds(id, module, run_id, entry, graph, thread_id, checkpoint_ns, status, requested_at, "
            "inputs_json, worker_json, seed_json) VALUES (?,?,?,?,?,?,?,'dispatched',?,?,?,?)",
            (build_id, module, db.run_id() if hasattr(db, "run_id") else "", entry, graph, thread_id, checkpoint_ns,
             time.time(), _j(inputs), _j(worker or inputs.get("worker") or {}), _j(seed) if seed else None))
    return get_build(db, build_id)


def find_active_build(db, module: str, *, entry: str, thread_id: str, checkpoint_ns: str | None = None,
                      run_id: str | None = None) -> dict | None:
    """The active build row a graph task already allocated for exactly this
    place (module, entry, thread, checkpoint namespace, run): the replay of
    an init node after a crash reuses its identity instead of allocating a
    second row."""
    for b in active_builds(db, module):
        if b["entry"] != entry or b["thread_id"] != thread_id:
            continue
        if checkpoint_ns is not None and (b.get("checkpoint_ns") or "") != (checkpoint_ns or ""):
            continue
        if run_id is not None and (b.get("run_id") or "") != (run_id or ""):
            continue
        return b
    return None


def mark_started(db, build_id: str, *, checkpoint_ns: str | None = None, thread_id: str | None = None) -> None:
    with db._tx() as con:
        con.execute("UPDATE builds SET status='running', started_at=COALESCE(started_at, ?), "
                    "checkpoint_ns=COALESCE(?, checkpoint_ns), thread_id=COALESCE(?, thread_id) "
                    "WHERE id=? AND status<>'completed'",
                    (time.time(), checkpoint_ns, thread_id, build_id))


def mark_status(db, build_id: str, status: str, *, error: str | None = None, result: dict | None = None,
                terminal: bool | None = None) -> None:
    """Move a build to ``status``. A build already ``completed`` is never
    moved back (its completion is the committed terminal record)."""
    if status not in BUILD_STATUSES:
        raise ValueError(f"bad build status {status!r}")
    terminal = status in BUILD_TERMINAL if terminal is None else terminal
    with db._tx() as con:
        row = con.execute("SELECT status FROM builds WHERE id=?", (build_id,)).fetchone()
        if row is None:
            raise ValueError(f"unknown build {build_id}")
        if row["status"] == "completed":
            return
        con.execute("UPDATE builds SET status=?, error=?, result_json=COALESCE(?, result_json), "
                    "finished_at=CASE WHEN ? THEN ? ELSE finished_at END WHERE id=?",
                    (status, error, _j(result) if result is not None else None, int(terminal), time.time(), build_id))


def get_build(db, build_id: str) -> dict | None:
    with db._conn() as con:
        return _row(con.execute("SELECT * FROM builds WHERE id=?", (build_id,)).fetchone())


def builds_for(db, module: str | None = None, *, limit: int = 200) -> list[dict]:
    with db._conn() as con:
        if module:
            rows = con.execute("SELECT * FROM builds WHERE module=? ORDER BY requested_at DESC, id LIMIT ?",
                               (module, limit)).fetchall()
        else:
            rows = con.execute("SELECT * FROM builds ORDER BY requested_at DESC, id LIMIT ?", (limit,)).fetchall()
    return [_row(r) for r in rows]


def active_builds(db, module: str | None = None) -> list[dict]:
    return [b for b in builds_for(db, module, limit=10000) if b["status"] in BUILD_ACTIVE]


def latest_build(db, module: str, *, statuses: tuple[str, ...] | None = None) -> dict | None:
    for b in builds_for(db, module, limit=10000):
        if statuses is None or b["status"] in statuses:
            return b
    return None


# ---------------------------------------------------------------- candidates
def _candidate_row(r) -> dict:
    d = dict(r)
    d["evaluation"] = _uj(d.pop("evaluation_json", None), {})
    d["files"] = _uj(d.pop("files_json", None), {})
    return d


def record_candidate(db, *, build_id: str, module: str, attempt: int, outcome: str,
                     evaluation: dict, files: dict | None = None) -> int:
    """One evaluated candidate of a build (an attempt that reached the target
    evaluation), kept whatever its outcome: the experiment ledger the
    Architect compares. Raises on failure (an unrecorded evaluation cannot
    be published)."""
    with db._tx() as con:
        cur = con.execute(
            "INSERT INTO build_candidates(build_id, module, attempt, ts, outcome, evaluation_json, files_json) "
            "VALUES (?,?,?,?,?,?,?)",
            (build_id, module, int(attempt), time.time(), outcome, _j(evaluation),
             _j(files) if files is not None else None))
        return int(cur.lastrowid)


def candidates_for(db, build_id: str) -> list[dict]:
    """Every evaluated candidate of ``build_id``, oldest first."""
    with db._conn() as con:
        rows = con.execute(
            "SELECT id, build_id, module, attempt, ts, outcome, evaluation_json, files_json "
            "FROM build_candidates WHERE build_id=? ORDER BY id", (build_id,)).fetchall()
    return [_candidate_row(r) for r in rows]


def latest_candidate(db, build_id: str, attempt: int | None = None) -> dict | None:
    rows = [c for c in candidates_for(db, build_id) if attempt is None or int(c["attempt"]) == int(attempt)]
    return rows[-1] if rows else None


def binds_targets(build: dict | None) -> bool:
    """Whether a recorded build is judged by FRD targets."""
    inputs = (build or {}).get("inputs") or {}
    return bool((inputs.get("targets") or {}).get("digest"))


def candidate_problems(db, build: dict, attempt: int, files: dict[str, str | None]) -> tuple[list[str], dict | None]:
    """Why the candidate of ``attempt`` cannot be published (empty = it can):
    a build that binds FRD targets publishes only an attempt whose recorded
    evaluation is ``feasible`` -- every required target measured from tool
    receipts and met -- and whose evaluated files are the files on disk."""
    cand = None
    for c in reversed(candidates_for(db, build["id"])):
        if int(c["attempt"]) == int(attempt):
            cand = c
            break
    if not binds_targets(build):
        return [], cand
    if cand is None:
        return [f"FRD_TARGETS_NOT_EVALUATED: attempt {attempt} has no recorded target evaluation"], None
    ev = cand.get("evaluation") or {}
    if cand["outcome"] != "feasible":
        return [f"FRD_TARGETS_NOT_MET: attempt {attempt} is {cand['outcome']} (missed {ev.get('missed') or []}, "
                f"unmeasured {ev.get('unmeasured') or []})"], cand
    changed = [p for p, sha in (cand.get("files") or {}).items() if files.get(p) != sha]
    if changed:
        return ["FRD_TARGETS_STALE: the implementation changed after its target evaluation: "
                + ", ".join(Path(p).name for p in changed[:4])], cand
    # every receipt the evaluation used still describes its (snapshotted)
    # evidence files, the liberty and the acceptance files: content and
    # provenance, not just a recorded verdict
    from orchestrator.state_store.module_targets import candidate_receipt_problems
    inputs = build.get("inputs") or {}
    alloc = {**(inputs.get("targets") or {}), "module": build.get("module"), "acceptance": inputs.get("acceptance")}
    receipt_problems = candidate_receipt_problems(alloc, ev)
    if receipt_problems:
        return ["FRD_TARGETS_STALE: the evidence of the evaluation no longer holds: "
                + "; ".join(receipt_problems)[:600]], cand
    return [], cand


def stamp_candidate_checks(db, cand: dict, *, build_id: str, revision: str | None) -> list[dict]:
    """``block_dv`` checks for the items a published candidate measured: each
    target with its measured value (pass/fail derived from the item's bounds)
    and each functional requirement whose acceptance tests passed, bound to
    the published revision, with the tool receipt as evidence. Returns the
    stamped ``[{item, status, value}]``."""
    out = []
    ev = cand.get("evaluation") or {}
    actor = f"build:{build_id}"
    for t in ev.get("targets") or []:
        if t.get("value") is None or t.get("status") not in ("pass", "fail"):
            continue
        rec = [{k: r.get(k) for k in ("kind", "test", "measure", "value", "unit")} for r in t.get("receipts") or []]
        db.add_check(t["id"], "block_dv", None, value=float(t["value"]), sha=(revision or "")[:16], actor=actor,
                     evidence=_j({"build_id": build_id, "candidate": cand.get("id"), "attempt": cand.get("attempt"),
                                  "measured": rec})[:2000])
        out.append({"item": t["id"], "status": t["status"], "value": t["value"]})
    for f in ev.get("functional") or []:
        if f.get("status") != "pass":
            continue
        db.add_check(f["id"], "block_dv", "pass", sha=(revision or "")[:16], actor=actor,
                     evidence=_j({"build_id": build_id, "candidate": cand.get("id"), "tests": f.get("tests")})[:2000])
        out.append({"item": f["id"], "status": "pass", "value": None})
    return out


# ---------------------------------------------------------------- evidence
def _scoreboard(root):
    from orchestrator.state_store.store import Scoreboard
    return Scoreboard(root)


def evidence_for_build(root, build_id: str) -> dict:
    return _scoreboard(root).rows_for_build(build_id)


def verify_build_evidence(root, build_id: str, *, attempt: int, timing_required: bool) -> tuple[list[str], dict]:
    """The committed rows a passing attempt must have left behind, read back
    from the database (never trusted from in-memory flags), judged by their
    LATEST outcome:

    * the latest RTL-scope DV row of the build is a gate-sourced, non-skipped
      pass of the current attempt (an earlier pass never outranks a later
      failure or a later attempt);
    * the latest coverage row belongs to the current attempt and is a
      MEASURED closure: a finite percentage against a declared finite floor,
      ``passed`` true. A row that says coverage was not measured -- the
      line-coverage gate disabled, no ``coverage.dat``, no
      ``verilator_coverage``, nothing annotated, "not evaluated" -- is an
      unavailable measurement, never evidence; there is no diagnostic reason
      that makes required closure inapplicable;
    * the latest synthesis PPA row of the current attempt carries a finite
      cell count, a positive verdict (``ppa_ok``), and -- when timing is
      required -- a finite measured WNS.

    Returns ``(problems, rows)``; an empty list means the evidence is complete."""
    rows = evidence_for_build(root, build_id)
    problems: list[str] = []
    attempt = int(attempt)
    dv = [r for r in rows["dv"] if r.get("scope") == "rtl"]
    if not dv:
        problems.append(f"no dv_results row for build {build_id}")
    else:
        last = dv[-1]
        if int(last.get("attempt") or 0) != attempt:
            problems.append(f"the latest dv_results row is attempt {last.get('attempt')}, not the current attempt {attempt}")
        elif last.get("source") != "gate":
            problems.append("the latest dv_results row is not the gate's own verdict")
        elif last.get("passed") != 1 or last.get("skipped"):
            problems.append(f"the latest dv_results row of attempt {attempt} is not a non-skipped pass")
    cov = rows["coverage"]
    if not cov:
        problems.append(f"no coverage_results row for build {build_id}")
    else:
        last = cov[-1]
        unc = _uj(last.get("uncovered"), {}) or {}
        if last.get("attempt") is None or int(last.get("attempt") or 0) != attempt:
            problems.append(f"the latest coverage_results row is not attempt {attempt}'s "
                            f"(attempt {last.get('attempt')})")
        elif unc.get("applicable") is False or last.get("pct") is None:
            problems.append("line coverage was not measured for this attempt ("
                            f"{str(unc.get('reason') or 'no percentage recorded').strip()}): a build needs measured "
                            "closure -- CORESMITH_LINE_COV_GATE must be on, the simulation must write coverage.dat "
                            "and verilator_coverage must be on PATH; an unavailable measurement is never evidence")
        else:
            if not _finite(last.get("pct")):
                problems.append("coverage_results row carries no finite coverage percentage")
            if not _finite(unc.get("floor")):
                problems.append("coverage_results row carries no declared floor (threshold)")
            if unc.get("passed") is not True:
                problems.append("coverage closure is not a recorded pass (coverage row passed is not true)")
    ppa = [r for r in rows["ppa"] if r.get("probe") == "synth"]
    if not ppa:
        problems.append(f"no synthesis ppa_history row for build {build_id}")
    else:
        last = ppa[-1]
        if int(last.get("attempt") or 0) != attempt:
            problems.append(f"the latest synthesis ppa_history row is attempt {last.get('attempt')}, not {attempt}")
        else:
            if not _finite(last.get("cells")):
                problems.append("synthesis ppa_history row has no finite cell count")
            if last.get("ppa_ok") is None:
                problems.append("synthesis ppa_history row carries no PPA verdict (ppa_ok is NULL: the PPA gate did "
                                "not run -- CORESMITH_PPA_GATE must be on for a build to complete)")
            elif last.get("ppa_ok") != 1:
                problems.append("synthesis ppa_history row carries no positive PPA verdict (ppa_ok)")
            if timing_required and not _finite(last.get("wns_ns")):
                problems.append("timing is required but the synthesis ppa_history row has no finite measured WNS")
            if last.get("power_basis") not in (None, "measured", "estimated", "unavailable"):
                problems.append("ppa_history row carries an unknown power basis")
    return problems, rows


_IMPORT_RE = re.compile(r"^\s*(?:from\s+([A-Za-z_][\w.]*)\s+import|import\s+([A-Za-z_][\w.]*))", re.M)
_MAX_TB_DEPS = 200


def tb_dependencies(tb_path) -> dict[str, str | None]:
    """The local Python modules a cocotb testbench imports (``import x`` /
    ``from x import``), resolved next to the testbench as ``x.py`` or
    ``x/__init__.py`` (and ``a.b`` as ``a/b.py``), followed transitively:
    the generated VIPs and helpers the test actually runs with. Standard
    library and site-packages imports are not local files and are not
    recorded. ``{path: sha256}``."""
    tb = Path(tb_path)
    out: dict[str, str | None] = {}
    if not tb.is_file():
        return out
    base = tb.parent
    todo = [tb]
    seen: set[str] = set()
    while todo and len(seen) < _MAX_TB_DEPS:
        cur = todo.pop(0)
        key = str(cur.resolve())
        if key in seen:
            continue
        seen.add(key)
        try:
            text = cur.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in _IMPORT_RE.finditer(text):
            mod = (m.group(1) or m.group(2) or "").strip()
            if not mod:
                continue
            rel = Path(*mod.split("."))
            for cand in (base / rel.with_suffix(".py"), base / rel / "__init__.py", cur.parent / rel.with_suffix(".py"),
                         cur.parent / rel / "__init__.py"):
                if cand.is_file():
                    if str(cand) not in out and cand.resolve() != tb.resolve():
                        out[str(cand)] = file_sha256(cand)
                        todo.append(cand)
                    break
    return dict(sorted(out.items()))


def produced_outputs(root, module: str, target: dict | None, *, rtl_path: str = "", tb_path: str = "") -> dict:
    """The identity of what the build produced or verified: the COMPLETE
    final target revision (``targets.revision``: configuration, every
    source, every file of every include dir, the discovered ``$readmem``
    assets), the testbench with the local modules it imports
    (:func:`tb_dependencies`), the netlist and the STA report. ``missing``
    names required files that do not exist: a build whose implementation is
    not on disk has produced nothing publishable."""
    root = Path(root)
    files: dict[str, str | None] = {}
    missing: list[str] = []
    revision = None
    revision_error = None
    if target and target.get("sources"):
        try:
            from orchestrator.harness.targets import load, revision_payload
            t = load(root, module, require_files=True)
            if t is None:
                missing.append(f"target {module} is no longer bound")
            else:
                payload = revision_payload(t)
                files.update(payload["files"])
                revision = payload["sha256"]
        except (ValueError, OSError, RuntimeError) as exc:
            revision_error = str(exc)[:300]
            for p in list(target.get("sources") or []) + list(target.get("assets") or []):
                sha = file_sha256(p)
                files[str(p)] = sha
                if sha is None:
                    missing.append(str(p))
            if not missing:
                missing.append(f"target revision unavailable: {revision_error}")
    elif rtl_path:
        sha = file_sha256(rtl_path)
        files[str(rtl_path)] = sha
        if sha is None:
            missing.append(str(rtl_path))
    else:
        missing.append("no bound target and no RTL path")
    if tb_path:
        sha = file_sha256(tb_path)
        files[str(tb_path)] = sha
        if sha is None:
            missing.append(str(tb_path))
        else:
            files.update(tb_dependencies(tb_path))
    syn = root / "syn" / "output" / module
    for name in (f"{module}_netlist.v", f"{module}_report.txt", "sta_report.txt"):
        p = syn / name
        if p.is_file():
            files[str(p)] = file_sha256(p)
    return {"files_sha256": files, "revision": revision, "revision_error": revision_error,
            "missing": missing, "captured_at": time.time()}


def snapshot_artifacts(root, build_id: str, files: dict[str, str | None]) -> dict:
    """Immutable copies of what the build produced -- the sources, the
    testbench and its local modules, the netlist and the reports -- under
    ``.coresmith/builds/<build_id>/``, keyed by the original path (the next
    build of the module overwrites the originals). A file inside the project
    keeps its relative path; a file outside it is placed under
    ``external/<path digest>/<name>`` so two ``defs.vh`` from different
    vendor trees never collide. Every copy is read back and verified against
    the recorded hash; a copy that already exists is kept when it matches
    and is an error when it does not (a completed snapshot is never
    rewritten). Returns ``{dir, files: {original: copy}, errors}``; a file
    that cannot be copied or verified is an error, never a silent gap."""
    root = Path(root)
    dest = root / ".coresmith" / "builds" / build_id
    copied: dict[str, str] = {}
    errors: list[str] = []
    for orig, sha in sorted(files.items()):
        if sha is None:
            errors.append(f"{orig}: no recorded hash (the file did not exist)")
            continue
        src = Path(orig)
        try:
            rel = src.resolve().relative_to(root.resolve())
        except ValueError:
            rel = Path("external") / hashlib.sha256(str(src.resolve()).encode()).hexdigest()[:16] / src.name
        target = dest / rel
        try:
            if target.exists():
                have = file_sha256(target)
                if have != sha:
                    errors.append(f"{src}: an earlier snapshot at {target} holds other bytes; snapshots are immutable")
                    continue
                copied[str(src)] = str(target)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            got = file_sha256(target)
            if got != sha:
                errors.append(f"{src}: the copy's bytes ({(got or 'unreadable')[:12]}) do not match the recorded hash "
                              f"({sha[:12]}); the source changed while it was being archived")
                continue
            copied[str(src)] = str(target)
        except OSError as exc:
            errors.append(f"{src}: {exc}")
    return {"dir": str(dest), "files": copied, "errors": errors}


def implementation_changed(root, build: dict) -> list[str]:
    """Files the build recorded as its produced output whose bytes differ
    now (a hand edit after the build), plus a target-revision marker when a
    dependency the revision covers changed without any recorded file
    changing (a header added to an include dir). A recorded file with no
    hash is never treated as valid: the published pass no longer describes
    the implementation on disk."""
    result = build.get("result") or {}
    produced = result.get("produced") or {}
    recorded = produced.get("files_sha256") or {}
    changed = [p for p, sha in recorded.items() if sha is None or file_sha256(p) != sha]
    rev = produced.get("revision")
    if rev and not changed:
        try:
            from orchestrator.harness.targets import load, revision
            t = load(root, build["module"], require_files=True)
            if t is not None and revision(t) != rev:
                changed.append(f"<target revision of {build['module']}: dependencies changed>")
        except (ValueError, OSError, RuntimeError) as exc:
            changed.append(f"<target revision of {build['module']} unavailable: {str(exc)[:120]}>")
    return changed


# ---------------------------------------------------------------- selection
def current_build_status(db, root, module: str, *, target_clock_mhz: float | None = None) -> dict:
    """How the module's published ``best`` stands against the live inputs:
    ``{best, build, ok, reasons}``. ``ok`` is True only for a ``best`` that
    names a ``completed`` build of this module whose recorded inputs still
    match and whose produced implementation is unchanged."""
    best = db.result(module, "best") or None
    out: dict = {"module": module, "best": best, "build": None, "ok": False, "reasons": []}
    if not best or not best.get("done"):
        out["reasons"].append("no published pass")
        return out
    bid = str(best.get("build_id") or "")
    if not bid:
        out["reasons"].append("the published pass names no recorded build (a manual or diagnostic publication)")
        return out
    build = get_build(db, bid)
    out["build"] = build
    if build is None:
        out["reasons"].append(f"build {bid} is not recorded")
        return out
    if build["module"] != module:
        out["reasons"].append(f"build {bid} belongs to module {build['module']}, not {module}")
        return out
    if build["status"] != "completed":
        out["reasons"].append(f"build {bid} is {build['status']}, not completed")
        return out
    stored = build.get("inputs") or {}
    live = module_inputs(db, root, module, seed_path=(stored.get("seed") or {}).get("path"),
                         target_clock_mhz=target_clock_mhz if target_clock_mhz is not None
                         else (stored.get("tooling") or {}).get("clock_mhz"))
    out["stale"] = stale_reasons(stored, live)
    out["reasons"].extend(out["stale"])
    changed = implementation_changed(root, build)
    if changed:
        out["implementation_changed"] = changed
        out["reasons"].append("implementation edited after the build: " + ", ".join(Path(p).name for p in changed[:4]))
    out["ok"] = not out["reasons"]
    return out


def _abs(root, p) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (Path(root) / q)


def composition_identity(db, root, *, top_rtl_path: str | None = None, files: list | None = None,
                         context: dict | None = None) -> dict:
    """What a chip-level verdict (integration DV, validation DV, backend
    signoff) is measured on: every block's current recorded build, the bytes
    of the integrated top and of every other input the measuring node
    declares (``files``: the chip testbench, the block RTL it compiled, the
    ERS it validated against, the SDC), plus the ``context`` it ran under
    (the liberty digest, the clock). ``top_rtl_path`` defaults to the latest
    elaborated snapshot's. The manifest is written to
    ``.coresmith/compositions/<sha256>.json`` so a gate can re-check every
    component later (:func:`composition_status`); a verdict recorded under a
    digest whose components no longer match describes another composition."""
    root = Path(root)
    modules: dict[str, str | None] = {}
    for spec in (db.block_specs() if hasattr(db, "block_specs") else []):
        cur = current_build_status(db, root, spec["name"])
        modules[spec["name"]] = (cur.get("build") or {}).get("id") if cur["ok"] else None
    top = top_rtl_path
    if not top:
        top, _problem = selected_top(db, root)
    paths = ([str(_abs(root, top))] if top else []) + [str(_abs(root, p)) for p in (files or []) if p]
    file_hashes = {p: file_sha256(p) for p in dict.fromkeys(paths)}
    manifest = {"modules": modules, "top_rtl_path": str(_abs(root, top)) if top else None,
                "files": file_hashes, "context": dict(context or {}),
                "missing": sorted(p for p, h in file_hashes.items() if h is None)}
    manifest["sha256"] = _canon_sha({k: manifest[k] for k in ("modules", "files", "context")})
    manifest["top_sha256"] = file_hashes.get(manifest["top_rtl_path"]) if manifest["top_rtl_path"] else None
    try:
        d = root / ".coresmith" / "compositions"
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{manifest['sha256']}.json"
        if not p.exists():
            p.write_text(_j({**manifest, "recorded_at": time.time()}))
    except OSError:
        pass
    return manifest


def selected_top(db, root) -> tuple[str | None, str | None]:
    """``(path, problem)``: the integrated top the project currently
    selects. The canonical adoption is the hierarchy-validated candidate
    receipt (``harness.top_module.write_candidate_receipt``, the sole
    adoption operation of the single-block, Caravel and Integration Lead
    paths): when a receipt exists its ``top_rtl_path`` is the selected top,
    and a receipt that no longer validates (sources changed, a dependency
    moved) is reported as a problem rather than silently ignored. Without a
    receipt the deterministic shell's latest elaborated snapshot selects
    the top. ``(None, None)`` when nothing selects one."""
    try:
        from orchestrator.harness.top_module import (
            CandidateError,
            read_candidate_receipt,
            validated_candidate,
        )
        rec = read_candidate_receipt(root)
    except Exception:  # noqa: BLE001 - the harness is not importable here: the shell snapshot decides
        rec = None
    if rec:
        try:
            cur = validated_candidate(root)
            return (str(_abs(root, cur["top_rtl_path"])) if cur.get("top_rtl_path") else None), None
        except CandidateError as exc:
            top = rec.get("top_rtl_path")
            return (str(_abs(root, top)) if top else None), f"the adopted candidate no longer validates: {exc}"
    if not hasattr(db, "latest_integration_snapshot"):
        return None, None
    snap = db.latest_integration_snapshot() or {}
    top = snap.get("rtl_path")
    return (str(_abs(root, top)) if top else None), None


def composition_status(db, root, sha: str | None) -> dict:
    """Whether the composition a verdict named (its recorded manifest) is
    the one the project has NOW: every block's current build is the one
    recorded, the top the verdict measured is the top the project selects
    today, every declared input existed when measured and has the recorded
    bytes now, and the recorded tooling context (the liberty digest) is the
    engine's current one. A declared input that was missing when measured
    is never positive evidence. ``{ok, reasons, manifest}``."""
    root = Path(root)
    out: dict = {"ok": False, "reasons": [], "manifest": None}
    if not sha:
        out["reasons"].append("the verdict names no composition")
        return out
    p = root / ".coresmith" / "compositions" / f"{sha}.json"
    manifest = _uj(p.read_text(encoding="utf-8") if p.is_file() else None, None)
    if not isinstance(manifest, dict):
        out["reasons"].append(f"no recorded manifest for composition {sha[:12]}")
        return out
    out["manifest"] = manifest
    for name, bid in (manifest.get("modules") or {}).items():
        cur = current_build_status(db, root, name)
        now = (cur.get("build") or {}).get("id") if cur["ok"] else None
        if now != bid:
            out["reasons"].append(f"{name}: current build is {now or 'none'}, the verdict was measured on {bid or 'none'}")
    measured_top = manifest.get("top_rtl_path")
    current_top, top_problem = selected_top(db, root)
    if top_problem:
        out["reasons"].append(top_problem)
    if measured_top and current_top and str(Path(measured_top).resolve()) != str(Path(current_top).resolve()):
        out["reasons"].append(f"the selected top is now {Path(current_top).name}; the verdict measured "
                              f"{Path(measured_top).name}")
    elif measured_top and not current_top:
        out["reasons"].append("the project selects no integrated top now; the verdict measured one")
    elif current_top and not measured_top:
        out["reasons"].append("the verdict measured no integrated top; the project selects one now")
    for path, recorded in (manifest.get("files") or {}).items():
        if recorded is None:
            out["reasons"].append(f"{Path(path).name}: the declared input did not exist when the verdict was measured")
        elif file_sha256(path) != recorded:
            out["reasons"].append(f"{Path(path).name}: bytes changed since the verdict")
    ctx = manifest.get("context") or {}
    if "liberty_sha256" in ctx and ctx["liberty_sha256"] != _engine_liberty().get("liberty_sha256"):
        out["reasons"].append("the liberty the verdict was measured with is not the engine's current one")
    out["ok"] = not out["reasons"]
    return out


# ---------------------------------------------------------------- lineage
def _checkpoint_threads(checkpoint_db: Path) -> dict[str, int]:
    """``{thread_id: checkpoint count}`` of a LangGraph SQLite checkpoint file."""
    if not checkpoint_db.is_file():
        return {}
    try:
        con = sqlite3.connect(f"file:{checkpoint_db}?mode=ro", uri=True, timeout=3.0)
        try:
            rows = con.execute("SELECT thread_id, COUNT(*) FROM checkpoints GROUP BY thread_id").fetchall()
        finally:
            con.close()
        return {str(t): int(n) for t, n in rows}
    except sqlite3.Error:
        return {}


def _checkpoint_namespaces(checkpoint_db: Path, thread_id: str) -> set[str]:
    """The checkpoint namespaces recorded for ``thread_id`` (the subgraph
    tasks that ran under it)."""
    if not checkpoint_db.is_file():
        return set()
    try:
        con = sqlite3.connect(f"file:{checkpoint_db}?mode=ro", uri=True, timeout=3.0)
        try:
            rows = con.execute("SELECT DISTINCT checkpoint_ns FROM checkpoints WHERE thread_id=?",
                               (thread_id,)).fetchall()
        finally:
            con.close()
        return {str(r[0] or "") for r in rows}
    except sqlite3.Error:
        return set()


def event_logs(root) -> list[Path]:
    """Every graph event log of the project, oldest first: the logs ``run
    start`` rotated aside (``pipeline_events.<stamp>.jsonl``) and the active
    ``pipeline_events.jsonl`` last."""
    cdir = Path(root) / ".coresmith"
    active = cdir / "pipeline_events.jsonl"
    rotated = sorted(p for p in cdir.glob("pipeline_events*.jsonl") if p != active and p.is_file())
    return rotated + ([active] if active.is_file() else [])


def _events_for_build(root, build_id: str) -> dict:
    """``{count, nodes, files}`` of the graph events that carry ``build_id``,
    read from every event log (a build that ran before a ``run start``
    rotation has its events in a rotated log)."""
    out: dict = {"count": 0, "nodes": [], "files": []}
    for path in event_logs(root):
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if build_id in line:
                        try:
                            ev = json.loads(line)
                        except ValueError:
                            continue
                        if ev.get("build_id") == build_id:
                            out["count"] += 1
                            node = str(ev.get("node") or "")
                            if node and node not in out["nodes"]:
                                out["nodes"].append(node)
                            if path.name not in out["files"]:
                                out["files"].append(path.name)
        except OSError:
            continue
    return out


def namespace_recorded(ns: str, recorded: set[str]) -> bool:
    """Whether checkpoint namespace ``ns`` -- or an ancestor of it on a ``|``
    segment boundary -- has checkpoints. A subgraph completion runs in a
    nested task (``process_block:<id>|block_done:<id>``) whose checkpoints are
    written under its checkpointed parent (``process_block:<id>``)."""
    if not ns:
        return False
    parts = ns.split("|")
    return any("|".join(parts[:i]) in recorded for i in range(len(parts), 0, -1))


def _graph_completion(root, build: dict, ck: dict) -> dict:
    """Whether a completed build's record is the graph's own: its completion
    names the thread it ran under, that thread has checkpoints in the graph's
    database (and, for a pipeline fan-out, the subgraph namespace the
    completion names), its graph events carry the build id through init and
    completion, and its required evidence rows verify."""
    result = build.get("result") or {}
    graph = build.get("graph") or ""
    thread = build.get("thread_id") or ""
    reasons = []
    if build.get("status") != "completed":
        reasons.append(f"build is {build.get('status')}")
    if not result.get("thread_id"):
        reasons.append("the completion names no graph thread")
    elif result.get("thread_id") != thread:
        reasons.append("the completion ran on another thread than the recorded one")
    if ck.get(graph, {}).get(thread, 0) <= 0:
        reasons.append(f"no checkpoints for thread {thread!r} in the {graph} graph")
    elif graph == "pipeline":
        ns = str(result.get("checkpoint_ns") or "")
        if not namespace_recorded(ns, _checkpoint_namespaces(Path(root) / ".coresmith" / "pipeline_checkpoint.db",
                                                             thread)):
            reasons.append("the completion's checkpoint namespace is not recorded in the pipeline checkpoint")
    ev = _events_for_build(root, build["id"])
    if "Init Block" not in ev["nodes"] or "Block Done" not in ev["nodes"]:
        reasons.append("graph events do not carry the build id through init and completion")
    if build.get("status") == "completed":
        problems, _ = verify_build_evidence(root, build["id"], attempt=int(result.get("attempt") or 0),
                                            timing_required=bool(result.get("timing_required")))
        reasons.extend(f"evidence: {p}" for p in problems)
    return {"ok": not reasons, "reasons": reasons, "events": ev}


def lineage(db, root, module: str | None = None) -> dict:
    """Machine-readable graph lineage per module: every recorded build, whether
    its LangGraph thread has checkpoints on disk, how many graph events carry
    its id, which rows it left, and whether the module's published pass is a
    graph-built, current one. ``intended_workflow`` is True for a module only
    when its current pass satisfies all of that (:func:`_graph_completion`);
    it says nothing about chip-level functional acceptance."""
    root = Path(root)
    ck = {"build": _checkpoint_threads(root / ".coresmith" / "build_checkpoint.db"),
          "pipeline": _checkpoint_threads(root / ".coresmith" / "pipeline_checkpoint.db")}
    sb = _scoreboard(root)
    modules = [module] if module else sorted({b["module"] for b in builds_for(db, limit=100000)}
                                             | {b["name"] for b in db.block_specs()})
    out: dict = {"project_root": str(root), "modules": {}, "generated_at": time.time()}
    for name in modules:
        rows = []
        for b in builds_for(db, name, limit=100000):
            r = sb.rows_for_build(b["id"])
            ev = _events_for_build(root, b["id"])
            rows.append({"id": b["id"], "entry": b["entry"], "graph": b["graph"], "status": b["status"],
                         "thread_id": b["thread_id"], "checkpoint_ns": b.get("checkpoint_ns"),
                         "checkpoints": ck.get(b["graph"], {}).get(b["thread_id"], 0),
                         "events": ev["count"], "event_nodes": ev["nodes"], "event_files": ev["files"],
                         "rows": {k: len(v) for k, v in r.items()},
                         "candidates": len(candidates_for(db, b["id"])),
                         "requested_at": b["requested_at"], "finished_at": b.get("finished_at"),
                         "inputs_digest": (b.get("inputs") or {}).get("digest"), "error": b.get("error")})
        cur = current_build_status(db, root, name)
        cur_build = cur.get("build") or {}
        completion = _graph_completion(root, cur_build, ck) if cur_build else {"ok": False, "reasons": ["no current build"]}
        out["modules"][name] = {
            "builds": rows,
            "published": {"ok": cur["ok"], "build_id": (cur.get("best") or {}).get("build_id"),
                          "reasons": cur["reasons"],
                          "graph_checkpoints_present": bool(cur_build) and ck.get(cur_build.get("graph", ""), {}).get(cur_build.get("thread_id", ""), 0) > 0,
                          "graph_completion": completion},
            "intended_workflow": bool(cur["ok"] and completion["ok"]),
        }
    out["intended_workflow"] = bool(out["modules"]) and all(m["intended_workflow"] for m in out["modules"].values())
    return out


def _candidate_targets(db, build: dict) -> dict | None:
    """The FRD target values of a build's published (else latest) candidate."""
    cands = candidates_for(db, build["id"])
    if not cands:
        return None
    pub = (build.get("result") or {}).get("candidate_id")
    cand = next((c for c in cands if c["id"] == pub), cands[-1])
    ev = cand.get("evaluation") or {}
    return {"candidate_id": cand["id"], "attempt": cand["attempt"], "outcome": cand["outcome"],
            "values": {t["id"]: {"value": t.get("value"), "unit": t.get("unit"), "status": t.get("status")}
                       for t in ev.get("targets") or []}}


def compare(db, root, build_a: str, build_b: str) -> dict:
    """Two recorded builds side by side with their provenance: status,
    inputs digest, the stale axes between them, and the latest synthesis PPA
    and DV rows of each. Scope and context travel with every number."""
    a, b = get_build(db, build_a), get_build(db, build_b)
    if a is None or b is None:
        raise ValueError(f"unknown build: {build_a if a is None else build_b}")
    sb = _scoreboard(root)

    def _metrics(build: dict) -> dict:
        rows = sb.rows_for_build(build["id"])
        ppa = [r for r in rows["ppa"] if r.get("probe") == "synth"]
        dv = [r for r in rows["dv"] if r.get("scope") == "rtl"]
        cov = rows["coverage"]
        last = ppa[-1] if ppa else {}
        return {"status": build["status"], "module": build["module"], "entry": build["entry"],
                "inputs_digest": (build.get("inputs") or {}).get("digest"),
                "ppa": {k: last.get(k) for k in ("stage", "tool", "pdk", "clock_mhz", "workload", "cells", "ff",
                                                 "area_um2", "wns_ns", "tns_ns", "power_mw", "power_basis", "ppa_ok")},
                "dv": ({k: dv[-1].get(k) for k in ("passed", "skipped", "tests_passed", "tests_total", "attempt")}
                       if dv else None),
                "coverage": ({k: cov[-1].get(k) for k in ("pct", "points_hit", "points_total", "attempt")} if cov else None),
                "targets": _candidate_targets(db, build)}
    return {"a": _metrics(a), "b": _metrics(b), "same_module": a["module"] == b["module"],
            "input_differences": stale_reasons(a.get("inputs") or {}, b.get("inputs") or {})}
