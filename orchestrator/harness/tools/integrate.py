# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""``coresmith vip generate`` / ``coresmith shell assemble`` / ``coresmith
model refine|eval`` (architect sitting, step 6): the interfaces and
model_eval stage tools, wrapping the graph's own functions so the sitting and
the graph produce identical artifacts."""
from __future__ import annotations

import asyncio
from pathlib import Path


def vip_generate(db, pr) -> dict:
    from orchestrator.langgraph.vip_lib.codegen import write_all_vips
    contracts = list((db.contracts() or {}).get("contracts") or [])
    if not contracts:
        return {"ok": False, "error": "no contracts registered (coresmith register contracts <json>)"}
    idx = write_all_vips(pr, contracts, contract_version=db.contracts_version())
    errs = {k: v.get("error") for k, v in (idx.get("edges") or {}).items() if v.get("error")}
    slices = 0
    try:
        from orchestrator.langgraph.pipeline_graph import _write_contract_slice
        for b in db.block_specs():
            _write_contract_slice(str(pr), b["name"])
            slices += 1
    except Exception as exc:  # noqa: BLE001
        errs["contract_slices"] = str(exc)[:200]
    return {"ok": not errs, "vips": len(idx.get("edges") or {}), "errors": errs, "contract_slices": slices,
            "index": str(Path(pr) / ".coresmith" / "vip_index.json")}


def shell_assemble(db, pr, *, tier=None, all_real: bool = False) -> dict:
    from orchestrator.langgraph.pipeline_graph import _shell_assemble
    res = _shell_assemble(str(pr), db.block_specs(), tier=tier, all_real=all_real)
    if res is None:
        return {"ok": False, "error": "shell integration does not apply (chassis declared or disabled)"}
    asm, elab, snap = res
    ok = bool(elab.get("ok")) and not asm.wiring_errors and bool(asm.boundary_ports)
    return {"ok": ok, "top": asm.module_name, "rtl_path": asm.rtl_path, "real": [b for b in asm.instantiated if b not in asm.stubs],
            "stubs": list(asm.stubs), "wires": asm.wires, "boundary_ports": len(asm.boundary_ports),
            "wiring_errors": asm.wiring_errors[:20], "elaborated": elab.get("ok"), "elab_reason": elab.get("reason", ""),
            "elab_errors": (elab.get("errors") or [])[:20],
            "hint": "" if asm.boundary_ports else "no boundary ports: the top's port set must equal the declared top (task.yaml top / soc_top_ports.svh)"}


def model_refine(db, pr, *, blocks: list[str] | None = None) -> dict:
    """Per-block SystemC models (agent-authored where missing), assembly,
    build, smoke and the FRD evaluation -- the uArch phase's model work as a
    tool. ``blocks`` limits which models are (re)authored; the assembly is
    always whole-SoC."""
    from orchestrator.langgraph.pipeline_graph import _uarch_phase_models
    from orchestrator.systemc_model.conventions import model_name
    specs = db.block_specs()
    if blocks:
        md = Path(pr) / "model"
        for b in blocks:
            for ext in (".cpp",):
                p = md / f"{model_name(b)}{ext}"
                if p.exists():
                    p.rename(p.with_suffix(p.suffix + ".prev"))   # force re-authoring of the named blocks
    rec = asyncio.run(_uarch_phase_models(str(pr), specs))
    fe = rec.get("frd_eval") or {}
    return {"ok": bool(rec.get("build_ok")) and bool(rec.get("smoke_ok")) and (fe.get("gate_ok") is not False),
            "build_ok": rec.get("build_ok"), "smoke_ok": rec.get("smoke_ok"), "missing_models": rec.get("missing_models"),
            "frd_eval": {k: fe.get(k) for k in ("gate_ok", "summary", "report", "skipped", "error")},
            "blocks": {b: v.get("source") for b, v in (rec.get("blocks") or {}).items()},
            "build_log": (rec.get("build_log") or "")[-1500:], "smoke_log": (rec.get("smoke_log") or "")[-800:]}


def model_eval(db, pr) -> dict:
    from orchestrator.systemc_model import frd_eval as fe
    md = Path(pr) / "model"
    if not (md / "soc_model_top.h").exists():
        return {"ok": False, "error": "SoC model not assembled (coresmith model refine --all)"}
    names = [b["name"] for b in db.block_specs()]
    rec = asyncio.run(fe.evaluate(pr, md, names, arch=False, db=db))
    return {"ok": bool(rec.get("gate_ok")), **rec}
