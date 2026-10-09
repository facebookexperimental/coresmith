# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""``coresmith state check``: a deterministic audit of the project database
rows against the files on disk.

Every problem carries a stable code; ``error`` problems make the verb exit 1,
warnings alone exit 0:

    BLOCK_RTL_MISSING         a block row's rtl_target file does not exist
    BLOCK_TB_MISSING          a block row's testbench file does not exist
    PRIMITIVE_UNMATERIALIZED  a primitive block row with no rtl_target
    ARTIFACT_MISSING          a file-sourced artifact's file is gone
    ARTIFACT_STALE            a file-sourced artifact's file changed since it was registered
    VIEW_MISSING              a DB-sourced artifact's rendered view (or the ERS view) is missing
    UARCH_UNREGISTERED        arch/uarch_specs/<b>.md on disk with no uarch:<b> artifact
    FRD_UNVERIFIED            (warning) a must-have FRD item with no verifier
    PINS_SHELL_MISMATCH       (warning) the latest shell's boundary differs from the pins

Stage-aware: while the stage machine is before ``blocks`` (``stages.current``;
a run with no stage rows is not judged), BLOCK_RTL_MISSING, BLOCK_TB_MISSING
and PRIMITIVE_UNMATERIALIZED are expected and reported as ``info``.

Wired from ``orchestrator.harness.cli`` (``state check``). Imports nothing
from ``orchestrator.langgraph``.
"""

from __future__ import annotations

from pathlib import Path

EXIT_PASS, EXIT_FAIL = 0, 1

# DB-sourced artifact kind -> the rendered views that must exist
DB_VIEWS = {
    "frd": ("arch/frd_spec.md",),
    "prd": ("arch/prd_spec.md", ".coresmith/prd_spec.json"),
    "contracts": (".coresmith/interface_contracts.json",),
    "ers": (".coresmith/ers_spec.json",),
    "pins": (".coresmith/pins.json",),
}
ERS_VIEW = ".coresmith/ers_spec.json"
# expected before the blocks stage (the RTL/TB/primitive do not exist yet)
PRE_BLOCKS_CODES = ("BLOCK_RTL_MISSING", "BLOCK_TB_MISSING", "PRIMITIVE_UNMATERIALIZED")


def _before_blocks(db) -> str:
    """The current stage when the stage machine is in use and before
    ``blocks`` ('' otherwise)."""
    try:
        from orchestrator.state_store import stages
        if not db.stage_rows():
            return ""
        cur = stages.current(db)
        return cur if stages.STAGES.index(cur) < stages.STAGES.index("blocks") else ""
    except Exception:  # noqa: BLE001
        return ""


def _p(code: str, where: str, text: str, severity: str = "error") -> dict:
    return {"code": code, "where": where, "text": text, "severity": severity}


def _abs(root: Path, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else root / q


def audit(db, root) -> list[dict]:
    """Every rows-vs-disk problem of the project (see the module docstring)."""
    from orchestrator.state_store.ontology import file_sha, item_must_have
    root = Path(root)
    out: list[dict] = []
    # blocks
    try:
        specs = db.block_specs()
    except Exception:  # noqa: BLE001
        specs = []
    for b in specs:
        name = str(b.get("name") or "")
        rtl, tb = str(b.get("rtl_target") or ""), str(b.get("testbench") or "")
        primitive = bool(b.get("primitive") or str(b.get("kind") or "").lower() == "primitive")
        if rtl and not _abs(root, rtl).exists():
            out.append(_p("BLOCK_RTL_MISSING", name, f"rtl_target {rtl} does not exist"))
        if tb and not _abs(root, tb).exists():
            out.append(_p("BLOCK_TB_MISSING", name, f"testbench {tb} does not exist"))
        if primitive and not rtl:
            out.append(_p("PRIMITIVE_UNMATERIALIZED", name,
                          "primitive block with no rtl_target (its generated RTL was never recorded)"))
    # artifacts
    arts = db.artifacts()
    kinds = {a["kind"] for a in arts}
    for a in arts:
        kind, path = a["kind"], str(a.get("path") or "")
        if path.startswith("db:"):
            for v in DB_VIEWS.get(kind, ()):
                if not (root / v).exists():
                    out.append(_p("VIEW_MISSING", kind, f"{v} (the rendered view of db:{kind}) does not exist; "
                                                        "coresmith state write"))
            continue
        f = _abs(root, path)
        if not f.is_file():
            out.append(_p("ARTIFACT_MISSING", kind, f"registered file {path} does not exist"))
            continue
        try:
            sha = file_sha(f)
        except OSError as exc:
            out.append(_p("ARTIFACT_MISSING", kind, f"registered file {path} unreadable: {exc}"))
            continue
        if a.get("sha") and sha != a["sha"]:
            out.append(_p("ARTIFACT_STALE", kind, f"{path} changed since it was registered (sha {a['sha']} -> "
                                                  f"{sha}); coresmith register {kind.split(':')[0]} {path}"
                                                  + (f" --block {kind.split(':', 1)[1]}" if ":" in kind else "")))
    if "ers" in kinds and not (root / ERS_VIEW).exists() and not str((db.artifact("ers") or {}).get("path") or "").startswith("db:"):
        out.append(_p("VIEW_MISSING", "ers", f"{ERS_VIEW} (read by validation_dv) does not exist; "
                                             "coresmith register ers <path> writes it"))
    # uArch specs on disk without an artifact
    ud = root / "arch" / "uarch_specs"
    if ud.is_dir():
        for md in sorted(ud.glob("*.md")):
            if f"uarch:{md.stem}" not in kinds:
                out.append(_p("UARCH_UNREGISTERED", md.stem, f"{md.relative_to(root)} is not registered; "
                                                             f"coresmith register uarch --block {md.stem} {md.relative_to(root)}"))
    # must-have FRD items with no verifier
    for it in db.items(artifact="frd"):
        if item_must_have(it) and not db.verifiers(item_id=it["id"]):
            out.append(_p("FRD_UNVERIFIED", it["id"], "must-have with no verifier (coresmith frd verifier ...)",
                          "warning"))
    # pins vs the latest shell boundary
    try:
        pins = {p["name"] for p in db.pins()}
        snap = db.latest_integration_snapshot()
    except Exception:  # noqa: BLE001
        pins, snap = set(), None
    if pins and snap and snap.get("boundary") is not None:
        extra = sorted(set(snap["boundary"]) - pins - {"clk", "rst_n"})
        missing = sorted(pins - set(snap["boundary"]))
        if extra or missing:
            out.append(_p("PINS_SHELL_MISMATCH", "shell", f"the latest shell boundary differs from the pins "
                                                          f"(extra {extra}, missing {missing}); coresmith shell assemble",
                          "warning"))
    stage = _before_blocks(db)
    if stage:
        for q in out:
            if q["code"] in PRE_BLOCKS_CODES and q["severity"] == "error":
                q["severity"] = "info"
                q["text"] += f" (expected before the blocks stage; stage is {stage})"
    return out


def cmd_state_check(args) -> int:
    from orchestrator.harness import cli
    db = cli._state_db(args)
    probs = audit(db, db.root)
    errors = [q for q in probs if q["severity"] == "error"]
    warns = [q for q in probs if q["severity"] == "warning"]
    infos = [q for q in probs if q["severity"] == "info"]
    if errors or warns:
        head = f"state check: {len(errors)} problem(s), {len(warns)} warning(s)"
    else:
        head = "state check: OK (rows and disk agree)"
    if infos:
        head += f", {len(infos)} expected at this stage (info)"
    cli._emit(args, {"ok": not errors, "problems": probs, "errors": len(errors), "warnings": len(warns),
                     "info": len(infos)},
              "\n".join([head] + cli._problems_lines(errors + warns + infos)))
    return EXIT_FAIL if errors else EXIT_PASS
