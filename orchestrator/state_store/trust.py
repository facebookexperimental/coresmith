# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Oracle-integrity manifest.

The gate's authority rests on the golden reference + the fixed stimulus + the
ERS/FRD specs being the SAME artifacts the run started with. An agent that
edits the golden (or a stimulus file) to make its RTL "match" is tampering with
the oracle. ``write_oracle_manifest`` snapshots SHA-256 of those files at
``/run/start``; ``check_oracle_manifest`` recomputes at gate-accept time and
reports any drift as an ``ORACLE_TAMPER`` violation.

The baseline is owner-controlled outside the project. Missing or malformed
baselines fail closed; only explicit daemon/run starts capture a new baseline.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

_MANIFEST_NAME = "oracle_manifest.json"


def _manifest_path(project_root: str | Path) -> Path:
    root = Path(project_root).resolve()
    trust_dir = Path(os.environ.get("CORESMITH_TRUST_DIR") or
                     Path.home() / ".coresmith/trust").expanduser().resolve()
    if trust_dir.is_relative_to(root):
        raise ValueError("CORESMITH_TRUST_DIR must be outside the project")
    project_id = hashlib.sha256(str(root).encode()).hexdigest()
    return trust_dir / project_id / _MANIFEST_NAME


def _sha256(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:  # noqa: BLE001
        return None


def _oracle_files(project_root: str | Path) -> list[Path]:
    """The files whose integrity underwrites the gate verdict.

    * the resolved golden reference implementation
    * everything under ``inputs/`` (requirements + stimulus + golden copy)
    * the architecture specs (``arch/{ers,frd,prd}_spec.md``)
    """
    root = Path(project_root)
    files: list[Path] = []
    seen: set[str] = set()

    def _add(p: Path) -> None:
        try:
            rp = p.resolve()
        except Exception:  # noqa: BLE001
            rp = p
        key = str(rp)
        # C2: the run's own state dir (rulings view, DB, views) is engine
        # output, never an oracle. Rulings deliberately live there so adding
        # one mid-run is not tampering.
        if ".coresmith" in rp.parts:
            return
        if p.is_file() and key not in seen:
            seen.add(key)
            files.append(p)

    try:
        from orchestrator.langgraph.microarch_rd import resolve_golden_path
        gp = resolve_golden_path(str(root))
        if gp:
            _add(Path(gp))
    except Exception:  # noqa: BLE001
        pass

    from orchestrator.harness.task_adapter import adapter_path
    adapter = adapter_path(str(root))
    if adapter and not adapter.startswith("MISSING:"):
        _add(Path(adapter))

    inputs = root / "inputs"
    if inputs.is_dir():
        for p in sorted(inputs.rglob("*")):
            # dv-hardening-18b (armD false-positive): Python bytecode caches
            # regenerate whenever the golden is legitimately IMPORTED (the
            # Full Model DV / acceptance tiers import it from inputs/), which
            # fail-closed the tamper guard on __pycache__ churn and escalated
            # a healthy block. Caches are derived artifacts, not oracle
            # content -- integrity is carried by the .py sources.
            if "__pycache__" in p.parts or p.suffix == ".pyc":
                continue
            _add(p)

    for name in ("ers_spec.md", "frd_spec.md", "prd_spec.md"):
        _add(root / "arch" / name)

    return files


def _digest_map(project_root: str | Path) -> dict[str, str]:
    """relpath (posix) -> sha256 for every oracle file present now."""
    root = Path(project_root)
    out: dict[str, str] = {}
    for p in _oracle_files(root):
        digest = _sha256(p)
        if digest is None:
            raise OSError(f"Cannot hash oracle file: {p}")
        try:
            rel = p.resolve().relative_to(root.resolve())
            key = rel.as_posix()
        except Exception:  # noqa: BLE001
            key = str(p)
        out[key] = digest
    return out


def write_oracle_manifest(project_root: str | Path) -> dict | None:
    """Snapshot oracle hashes into the external owner trust directory.

    Best-effort: returns the manifest dict on success, ``None`` on failure.
    """
    try:
        files = _digest_map(project_root)
        manifest = {"created_ts": time.time(), "files": files}
        path = _manifest_path(project_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as fh:
            temporary = Path(fh.name)
            json.dump(manifest, fh, indent=2)
        try:
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return manifest
    except Exception:  # noqa: BLE001
        return None


def capture_run_baseline(project_root: str | Path) -> dict:
    manifest = write_oracle_manifest(project_root)
    if manifest is None:
        raise RuntimeError("Cannot capture the external oracle trust baseline")
    return manifest


# The architecture SPEC files an AUTHORIZED *pre-RTL* feasibility revise may
# legitimately edit (freeze register bits, add an opcode, repartition an
# interface). Editing these during honest-feasibility triage is design work,
# not oracle tampering. The golden reference and the stimulus under inputs/ are
# deliberately EXCLUDED -- editing THOSE to make RTL "match" is the exact cheat
# the manifest exists to catch, so they can never be re-baselined.
_SPEC_REBASELINE_ALLOWED = frozenset({
    "arch/ers_spec.md", "arch/frd_spec.md", "arch/prd_spec.md",
})


def rebaseline_oracle_specs(project_root: str | Path) -> list[str]:
    """Re-snapshot ONLY the architecture spec files in an existing manifest,
    leaving the golden + stimulus (``inputs/``) hashes at their run-start
    values.

    Call this at an AUTHORIZED pre-RTL design edit -- a uarch feasibility
    ``revise_interface`` where the chip-lead fixed a blocker by editing the
    frozen ERS/interface. That edit is legitimate triage, so the spec's new
    content becomes the baseline going forward; but the golden/stimulus stay
    frozen, so a later attempt to edit the golden to force a match is STILL
    caught as ``ORACLE_TAMPER``. No-op (returns ``[]``) when there is no
    manifest or nothing spec-side drifted. Returns the re-baselined relpaths."""
    path = _manifest_path(project_root)
    if not path.exists():
        return []
    try:
        manifest = json.loads(path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        return []
    recorded = manifest.get("files") or {}
    if not recorded:
        return []
    now = _digest_map(project_root)
    rebaselined: list[str] = []
    for rel in list(recorded.keys()):
        if rel not in _SPEC_REBASELINE_ALLOWED:
            continue  # golden/stimulus/inputs are never re-baselineable
        cur = now.get(rel)
        if cur is not None and cur != recorded[rel]:
            recorded[rel] = cur
            rebaselined.append(rel)
    if rebaselined:
        manifest["files"] = recorded
        manifest["last_spec_rebaseline_ts"] = time.time()
        try:
            path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        except Exception:  # noqa: BLE001
            return []
    return sorted(rebaselined)


def check_oracle_manifest(project_root: str | Path) -> dict[str, Any]:
    """Recompute oracle hashes and compare to the recorded manifest.

    Returns a dict::

        {"ok": bool, "checked": bool, "changed": [...], "missing": [...],
         "added": [...], "violation": {...} | None}

    Missing/unreadable baselines are infrastructure failures. A complete empty
    baseline is valid and still detects newly added immutable oracle files.
    """
    result: dict[str, Any] = {
        "ok": True, "checked": False, "changed": [], "missing": [],
        "added": [], "violation": None,
    }
    try:
        path = _manifest_path(project_root)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        recorded = manifest["files"]
        if not isinstance(recorded, dict) or any(
                not isinstance(k, str) or not isinstance(v, str) or len(v) != 64
                for k, v in recorded.items()):
            raise ValueError("Malformed oracle baseline")
        now = _digest_map(project_root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result.update(ok=False, violation={"category": "ORACLE_BASELINE_ERROR",
                      "kind": "infrastructure_error", "reason": str(exc)})
        return result
    result["checked"] = True
    changed: list[str] = []
    missing: list[str] = []
    for rel, want in recorded.items():
        have = now.get(rel)
        if have is None:
            missing.append(rel)
        elif have != want:
            changed.append(rel)
    # An oracle file that APPEARED under inputs/ after run start is tamper too:
    # dropping a doctored ``inputs/golden.py`` into a run whose golden resolved
    # elsewhere SHADOWS the real oracle (resolve_golden_path prefers it) while
    # every recorded hash still matches. Scoped to inputs/ deliberately -- the
    # arch specs AND a golden resolved outside the run dir are both written /
    # re-resolved legitimately after /run/start (the PRD naming the reference
    # is itself an architecture-stage output).
    added = sorted(rel for rel in now
                   if rel not in recorded and rel.startswith("inputs/"))

    if changed or missing or added:
        # PR#12 finding #9: partition drift into AMENDABLE architecture specs
        # (arch/{ers,frd,prd}_spec.md -- a legitimate pre-RTL feasibility
        # revise edits these) vs IMMUTABLE oracle (the golden + inputs/
        # stimulus/requirements -- editing THOSE to make RTL match is the
        # cheat this guard exists to catch). An authorized ERS amendment (e.g.
        # the cs_sram reconciliation) used to cascade ORACLE_TAMPER across
        # every re-processed block because a spec edit was treated identically
        # to golden tampering, with the impossible remedy "restore the golden."
        # A spec-ONLY drift is now auto-re-baselined (matching the existing
        # rebaseline_oracle_specs intent) and recorded as an advisory, not a
        # fail. Set CORESMITH_STRICT_ORACLE_MANIFEST=1 to keep any drift a
        # hard fail.
        drift = set(changed) | set(missing) | set(added)
        immutable_drift = sorted(d for d in drift
                                 if d not in _SPEC_REBASELINE_ALLOWED)
        spec_drift = sorted(d for d in drift if d in _SPEC_REBASELINE_ALLOWED)
        _strict = os.environ.get(
            "CORESMITH_STRICT_ORACLE_MANIFEST", "").strip().lower() in {
                "1", "true", "yes", "on"}
        result["changed"] = sorted(changed)
        result["missing"] = sorted(missing)
        result["added"] = added

        if immutable_drift or _strict or missing:
            # A real oracle changed/vanished (or strict mode) -> TAMPER.
            result["ok"] = False
            detail = (
                "oracle artifacts changed since run start "
                f"(modified={sorted(changed)}, missing={sorted(missing)}, "
                f"added={added}). The golden reference / stimulus that "
                "underwrites the gate MUST NOT be edited -- or SHADOWED by a "
                "new file -- to make RTL match: restore them and re-run."
            )
            result["violation"] = {
                "criterion": "oracle_integrity",
                "category": "ORACLE_TAMPER",
                "gap_class": "block_math",
                "severity": "error",
                "detail": detail,
                "changed": sorted(changed),
                "missing": sorted(missing),
                "added": added,
                "immutable_drift": immutable_drift,
                "suggested_fix": (
                    "NOT a pass -- an IMMUTABLE oracle (golden/stimulus/"
                    "requirements) was changed, is missing, or was shadowed "
                    "by a file added after run start. Restore the "
                    "original files, then resume. (Architecture SPEC edits "
                    "are re-baselineable; immutable oracle edits are not.)"
                ),
            }
        else:
            # Spec-only, authorized-amendment class -> re-baseline + advise.
            rebaselined = rebaseline_oracle_specs(project_root)
            result["ok"] = True
            result["spec_rebaselined"] = rebaselined or spec_drift
            result["advisory"] = (
                "authorized architecture-spec amendment re-baselined "
                f"({spec_drift}); golden/stimulus integrity intact. Recorded "
                "as a carried-forward provenance note, not a tamper fail."
            )
    return result
