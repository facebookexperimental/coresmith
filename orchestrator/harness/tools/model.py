# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""``coresmith model ... --arch`` and ``coresmith fabric derive``: the
executable SAD (architecture model) as tools for the Architect. Exit
contract (``cli._model_exit``): a missing or invalid ``arch_model.json`` /
unbuilt model is an input error (``error`` in ``INPUT_ERRORS``, ``required``
names the path; exit 2); a compile or run failure is a check failure (exit
1); only a missing SystemC toolchain is ``tool_error`` (exit 3)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from orchestrator.state_store.ontology import file_sha


def _spec_shape_problems(d) -> list[str]:
    """Why a decoded ``arch_model.json`` is not an architecture spec at all
    (the shape ``ArchSpec.from_json`` reads): an object with ``components``
    and ``links`` as lists of objects and ``fabric`` as an object."""
    if not isinstance(d, dict):
        return [f"the document must be a JSON object, not {type(d).__name__}"]
    problems = []
    for key in ("components", "links"):
        val = d.get(key)
        if val is None:
            continue
        if not isinstance(val, list):
            problems.append(f"{key} must be a list, not {type(val).__name__}")
            continue
        bad = [i for i, c in enumerate(val) if not isinstance(c, dict)]
        if bad:
            problems.append(f"{key}[{bad[0]}] must be an object, not {type(val[bad[0]]).__name__}")
    fab = d.get("fabric")
    if fab is not None and not isinstance(fab, dict):
        problems.append(f"fabric must be an object, not {type(fab).__name__}")
    return problems


def _load_spec(md: Path) -> tuple[object | None, dict | None]:
    """``(spec, None)`` when ``model/arch/arch_model.json`` loads and validates;
    ``(None, error)`` with ``ARCH_SPEC_MISSING`` / ``ARCH_SPEC_INVALID`` (the
    required path and the ``problems``) otherwise. Only the parse of the
    document is guarded: a compiler or tool failure happens later and is
    reported as such."""
    from orchestrator.systemc_model import arch_model as am
    p = md / "arch_model.json"
    if not p.exists():
        return None, {"ok": False, "error": "ARCH_SPEC_MISSING", "required": str(p),
                      "hint": "coresmith model init --arch writes a template; edit components/links from the SAD"}
    invalid = {"ok": False, "error": "ARCH_SPEC_INVALID", "required": str(p)}
    try:
        doc = json.loads(p.read_text())
    except ValueError as exc:
        return None, {**invalid, "problems": [f"not JSON: {str(exc)[:300]}"]}
    problems = _spec_shape_problems(doc)
    if problems:
        return None, {**invalid, "problems": problems}
    try:
        spec = am.ArchSpec.from_json(doc)
    except (ValueError, TypeError, AttributeError, KeyError) as exc:   # a field of the wrong type
        return None, {**invalid, "problems": [f"{type(exc).__name__}: {str(exc)[:300]}"]}
    errs = spec.validate()
    if errs:
        return None, {**invalid, "problems": list(errs)}
    return spec, None

SPEC_TEMPLATE = {
    "name": "soc", "clock_mhz": 100, "addr_width": 32,
    "components": [
        {"name": "cpu", "kind": "initiator", "instances": 1, "energy_pj_per_txn": 0},
        {"name": "ram", "kind": "target", "base": "0x80000000", "size": "0x8000000", "latency_cycles": 8,
         "bytes_per_cycle": 8, "protocol": "axi4"},
    ],
    "fabric": {"latency_cycles": 2, "bytes_per_cycle": 8},
    "links": [],
}


def arch_init(project_root) -> dict:
    from orchestrator.systemc_model.arch_model import arch_dir
    md = arch_dir(project_root)
    md.mkdir(parents=True, exist_ok=True)
    p = md / "arch_model.json"
    if p.exists():
        return {"ok": True, "path": str(p), "created": False}
    p.write_text(json.dumps(SPEC_TEMPLATE, indent=2))
    return {"ok": True, "path": str(p), "created": True,
            "hint": "edit components/links from the SAD, then: coresmith model build --arch"}


def arch_build(project_root) -> dict:
    """``coresmith model build --arch``: generate + compile the architecture
    model from ``arch_model.json``. Missing/invalid spec: input error (exit 2);
    missing toolchain: ``tool_error`` (exit 3); compiler diagnostics: a build
    failure with the log (exit 1)."""
    from orchestrator.systemc_model import arch_model as am
    from orchestrator.systemc_model.toolchain import detect
    md = am.arch_dir(project_root)
    spec, bad = _load_spec(md)
    if bad:
        return bad
    tc = detect()
    if not tc["ok"]:
        return {"ok": False, "error": "systemc toolchain: " + tc["reason"], "tool_error": True}
    am.write_build(project_root, spec, systemc_home=tc.get("systemc_home") or "")
    b = am.build(md)
    out = {"ok": b["ok"], "log": b["log"][-3000:], "dir": str(md), "components": len(spec.components),
           "tool_error": False}
    if not b["ok"]:
        out["error"] = "ARCH_MODEL_BUILD_FAILED"
    return out


def arch_run(project_root, *, ns: int = 10000) -> dict:
    """``coresmith model run --arch``: the smoke scenario of the built model.
    Not built: input error (exit 2); a failed run: exit 1 with the log."""
    from orchestrator.systemc_model import arch_model as am
    md = am.arch_dir(project_root)
    exe = md / "arch_model"
    if not exe.exists():
        return {"ok": False, "error": "ARCH_MODEL_NOT_BUILT", "required": str(exe), "stats": None,
                "hint": "coresmith model build --arch"}
    r = am.run(md, ns=ns)
    out = {"ok": r["ok"], "log": r["log"][-2000:], "stats": r.get("stats")}
    if r["ok"]:
        out["stats_path"] = str(md / "stats.json")
    else:
        out["error"] = "ARCH_MODEL_RUN_FAILED"
    return out


def arch_register(db, project_root) -> dict:
    """Register the arch model artifact (sha over the spec + generated top).
    A missing spec is an input error (exit 2)."""
    from orchestrator.systemc_model import arch_model as am
    md = am.arch_dir(project_root)
    p = md / "arch_model.json"
    if not p.exists():
        return {"ok": False, "error": "ARCH_SPEC_MISSING", "required": str(p), "hint": "coresmith model init --arch"}
    sha = file_sha(p)
    art = db.register_artifact("arch_model", str(p.relative_to(Path(project_root))), sha=sha,
                               meta={"stats": bool((md / "stats.json").exists())}, registered_by="cli")
    return {"ok": True, "artifact": art}


def arch_eval(db, project_root, *, timeout_s: int | None = None, agent=None, repairs: int | None = None) -> dict:
    """``coresmith model eval --arch``: the FRD evaluated on the executable SAD
    with the harness that exists under ``model/arch/frd_eval/``; verdicts
    become ``model_eval`` checks. A pure check: it never authors or repairs the
    harness (``coresmith harness author --arch`` does that explicitly). A
    caller that passes an explicit ``agent`` is asking for authoring (the CLI
    never does); ``repairs`` only applies then."""
    from orchestrator.harness.tools.integrate import frd_input_error
    from orchestrator.systemc_model import arch_model as am
    from orchestrator.systemc_model import frd_eval as fe
    md = am.arch_dir(project_root)
    if not (md / "arch_model_top.h").exists():
        return {"ok": False, "gate_ok": None, "error": "ARCH_MODEL_NOT_BUILT", "required": str(md / "arch_model_top.h"),
                "hint": "coresmith model build --arch"}
    bad = frd_input_error(project_root, db)
    if bad:
        return {"ok": False, "gate_ok": None, **bad}
    spec_p = md / "arch_model.json"
    sha = file_sha(spec_p) if spec_p.exists() else ""
    blocks = []
    try:
        blocks = [c.name for c in am.load_spec(project_root).components]
    except Exception:  # noqa: BLE001
        pass
    rec = asyncio.run(fe.evaluate(project_root, md, blocks, arch=True, timeout_s=timeout_s, db=db,
                                  record_sha=sha, agent=agent, repairs=repairs, author=agent is not None))
    if rec.get("gate_ok") is not None:
        arch_register(db, project_root)
    return {"ok": bool(rec.get("gate_ok")), **rec}


def fabric_derive(db, project_root, *, name: str | None = None, headroom: float = 2.0, write: bool = True) -> dict:
    from orchestrator.systemc_model import arch_model as am
    md = am.arch_dir(project_root)
    stats = am.read_stats(md)
    if stats is None:
        return {"ok": False, "error": "no model/arch/stats.json (coresmith model run --arch, or the FRD harness)"}
    try:
        spec = am.load_spec(project_root)
    except (OSError, ValueError) as exc:
        return {"ok": False, "error": f"arch_model.json: {exc}"}
    fs = am.derive_fabric(spec, stats, name=name, headroom=headroom)
    problems = []
    try:
        from orchestrator.fabric import FabricSpec
        problems = FabricSpec.from_json(fs).validate()
    except Exception as exc:  # noqa: BLE001
        problems = [f"FabricSpec check unavailable: {exc}"]
    # The project database row is the single source of truth: set_fabric_spec
    # validates, versions and re-exports the read-only .coresmith/fabric_spec.json view.
    if write and not problems:
        res = db.set_fabric_spec(str(fs["name"]), fs)
        if not res.get("ok"):
            return {"ok": False, "fabric": fs, "problems": res.get("problems") or ["fabric spec refused"], "path": ""}
        view = Path(project_root) / ".coresmith" / "fabric_spec.json"
        return {"ok": True, "fabric": fs, "problems": [], "path": str(view),
                "version": res.get("version"), "changed": res.get("changed")}
    return {"ok": not problems, "fabric": fs, "problems": problems, "path": ""}
