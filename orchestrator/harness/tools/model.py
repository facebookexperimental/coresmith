# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""``coresmith model ...`` and ``coresmith fabric derive`` (architect sitting,
step 2): the executable SAD as tools."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from orchestrator.state_store.ontology import file_sha

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
    from orchestrator.systemc_model import arch_model as am
    from orchestrator.systemc_model.toolchain import detect
    md = am.arch_dir(project_root)
    p = md / "arch_model.json"
    if not p.exists():
        return {"ok": False, "error": "no model/arch/arch_model.json (coresmith model init --arch)"}
    try:
        spec = am.ArchSpec.from_json(json.loads(p.read_text()))
    except (ValueError, TypeError) as exc:
        return {"ok": False, "error": f"arch_model.json: {exc}"}
    errs = spec.validate()
    if errs:
        return {"ok": False, "error": "spec invalid", "problems": errs}
    tc = detect()
    if not tc["ok"]:
        return {"ok": False, "error": "systemc toolchain: " + tc["reason"], "tool_error": True}
    am.write_build(project_root, spec, systemc_home=tc.get("systemc_home") or "")
    b = am.build(md)
    return {"ok": b["ok"], "log": b["log"][-3000:], "dir": str(md), "components": len(spec.components),
            "tool_error": not b["ok"]}


def arch_run(project_root, *, ns: int = 10000) -> dict:
    from orchestrator.systemc_model import arch_model as am
    md = am.arch_dir(project_root)
    r = am.run(md, ns=ns)
    out = {"ok": r["ok"], "log": r["log"][-2000:], "stats": r.get("stats")}
    if r["ok"]:
        out["stats_path"] = str(md / "stats.json")
    return out


def arch_register(db, project_root) -> dict:
    """Register the arch model artifact (sha over the spec + generated top)."""
    from orchestrator.systemc_model import arch_model as am
    md = am.arch_dir(project_root)
    p = md / "arch_model.json"
    if not p.exists():
        return {"ok": False, "error": "no model/arch/arch_model.json"}
    sha = file_sha(p)
    art = db.register_artifact("arch_model", str(p.relative_to(Path(project_root))), sha=sha,
                               meta={"stats": bool((md / "stats.json").exists())}, registered_by="cli")
    return {"ok": True, "artifact": art}


def arch_eval(db, project_root, *, agent=None, timeout_s: int | None = None, repairs: int | None = None) -> dict:
    """FRD evaluation on the executable SAD; verdicts become ``model_eval`` checks."""
    from orchestrator.systemc_model import arch_model as am
    from orchestrator.systemc_model import frd_eval as fe
    md = am.arch_dir(project_root)
    if not (md / "arch_model_top.h").exists():
        return {"ok": False, "error": "arch model not built (coresmith model build --arch)"}
    spec_p = md / "arch_model.json"
    sha = file_sha(spec_p) if spec_p.exists() else ""
    blocks = []
    try:
        blocks = [c.name for c in am.load_spec(project_root).components]
    except Exception:  # noqa: BLE001
        pass
    rec = asyncio.run(fe.evaluate(project_root, md, blocks, arch=True, agent=agent, timeout_s=timeout_s,
                                  repairs=repairs, db=db, record_sha=sha))
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
    out_p = Path(project_root) / ".coresmith" / "fabric_spec.json"
    if write and not problems:
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(fs, indent=2))
        try:
            db.set_setting("fabric_spec_sha", file_sha(out_p))
        except Exception:  # noqa: BLE001
            pass
    return {"ok": not problems, "fabric": fs, "problems": problems, "path": str(out_p) if write and not problems else ""}
