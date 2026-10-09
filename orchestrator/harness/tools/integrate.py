# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""``coresmith vip generate`` / ``coresmith shell assemble`` / ``coresmith
model build|eval|author`` / ``coresmith harness author``: the interface and
executable-model tactics as tools, wrapping the graph's own functions so the
Architect and the graph produce identical artifacts.

Checks (``model build``, ``model eval``) consume the files that exist and
never construct an authoring agent, edit a source or repair a harness.
Authoring is its own verb (``model author``, ``harness author``) and only
touches what it was asked for.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

# ``error`` codes that mean "the input is missing or invalid" (CLI exit 2,
# the required path is named in ``required``).
INPUT_ERRORS = ("HARNESS_MISSING", "SOC_MODEL_NOT_ASSEMBLED", "ARCH_MODEL_NOT_BUILT", "ARCH_SPEC_MISSING",
                "ARCH_SPEC_INVALID", "FRD_MISSING", "FRD_NO_REQUIREMENTS", "UNKNOWN_BLOCK", "PRIMITIVE_BLOCK",
                "NO_BLOCKS")


def frd_input_error(pr, db=None) -> dict | None:
    """The FRD an evaluation needs: ``FRD_MISSING`` (no ``arch/frd_spec.md``)
    or ``FRD_NO_REQUIREMENTS`` (no ``**ID**: XXX-NNN`` blocks to evaluate),
    with the required path; None when the FRD is usable."""
    from orchestrator.systemc_model import frd_eval as fe
    frd = Path(pr) / "arch" / "frd_spec.md"
    if not frd.exists():
        return {"error": "FRD_MISSING", "required": str(frd),
                "hint": "register the FRD (coresmith register frd <path> / coresmith state write)"}
    reqs = fe.extract_requirements(frd.read_text(encoding="utf-8", errors="replace"))
    if not reqs:
        return {"error": "FRD_NO_REQUIREMENTS", "required": str(frd),
                "hint": "the FRD has no identified requirements (**ID**: XXX-NNN blocks); nothing can be evaluated"}
    return None


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
    try:
        pins = db.pins()
    except Exception:  # noqa: BLE001
        pins = []
    hint = ""
    if not pins:
        hint = ("no chip pins declared: coresmith pin add <name> --dir in|out|inout [--width N] --from <block>.<port> "
                "(and clk/rst_n: --kind clock|reset)")
    elif any(str(e).startswith("SHELL_UNDECLARED_PORT") for e in asm.wiring_errors):
        hint = "every block port is a contract edge or a pin: coresmith pin add ... --from <block>.<port>"
    elif not asm.boundary_ports:
        hint = "no boundary ports: the top's port set is the declared pins (coresmith pin list)"
    return {"ok": ok, "top": asm.module_name, "rtl_path": asm.rtl_path, "real": [b for b in asm.instantiated if b not in asm.stubs],
            "stubs": list(asm.stubs), "wires": asm.wires, "boundary_ports": len(asm.boundary_ports),
            "boundary": [b.get("name") for b in asm.boundary_ports], "pins": len(pins),
            "wiring_errors": asm.wiring_errors[:20], "elaborated": elab.get("ok"), "elab_reason": elab.get("reason", ""),
            "elab_errors": (elab.get("errors") or [])[:20], "hint": hint}


def model_build(db, pr) -> dict:
    """``coresmith model build`` (no ``--arch``): assemble the SoC model from the
    registered blocks and contracts, generate the fabric primitive's model and
    the missing block headers/contract slices, compile and smoke it with the
    implementations that exist, and record a ``models`` row per block for what
    was actually built. Never authors, repairs or renames an implementation:
    missing ``model/<block>_model.cpp`` files are listed in ``missing_models``."""
    from orchestrator.langgraph.pipeline_graph import _uarch_phase_models
    specs = db.block_specs()
    if not specs:
        return {"ok": False, "error": "NO_BLOCKS", "required": "a registered block diagram "
                "(coresmith register block_diagram <json>)"}
    rec = asyncio.run(_uarch_phase_models(str(pr), specs, author=False, evaluate=False))
    if rec.get("error") == "MODEL_PATH_UNSUPPORTED":
        bad = rec.get("unsupported_model_paths") or {}
        return {"ok": False, "error": "MODEL_PATH_UNSUPPORTED", "unsupported_model_paths": bad,
                "required": [f"model/{b}_model.cpp" for b in sorted(bad)],
                "hint": "the SoC model build compiles model/<block>_model.cpp; a models row that names another "
                        "existing file is not a supported binding. Migration: move the implementation to the "
                        "required path and remove (or rename away) the file the row names; with that file gone the "
                        "next model build re-binds the row to the conventional file (its old verdicts cleared) and "
                        "records that file's build/smoke result"}
    missing = sorted(rec.get("missing_models") or [])
    out = {"ok": bool(rec.get("build_ok")) and bool(rec.get("smoke_ok")),
           "build_ok": rec.get("build_ok"), "smoke_ok": rec.get("smoke_ok"), "missing_models": missing,
           "tool_error": bool(rec.get("parked_reason")), "parked_reason": rec.get("parked_reason", ""),
           "blocks": {b: v.get("source") for b, v in (rec.get("blocks") or {}).items()},
           "build_log": (rec.get("build_log") or "")[-1500:], "smoke_log": (rec.get("smoke_log") or "")[-800:],
           "model_dir": str(Path(pr) / "model")}
    if rec.get("rebound_model_paths"):
        # rows whose migrated-away file is gone, re-bound to the conventional file
        out["rebound_model_paths"] = dict(rec["rebound_model_paths"])
    if missing:
        out["required"] = [f"model/{m}_model.cpp" for m in missing]
        out["hint"] = ("write each implementation against its generated model/<block>_model.h, or request one: "
                       "coresmith model author --block <block>")
    return out


def model_eval(db, pr) -> dict:
    """``coresmith model eval`` (no ``--arch``): the FRD evaluated on the
    assembled SoC model with the harness that exists under ``model/frd_eval/``.
    A pure check: it never authors or repairs the harness."""
    from orchestrator.systemc_model import frd_eval as fe
    md = Path(pr) / "model"
    if not (md / "soc_model_top.h").exists():
        return {"ok": False, "gate_ok": None, "error": "SOC_MODEL_NOT_ASSEMBLED", "required": str(md / "soc_model_top.h"),
                "hint": "assemble and build the SoC model first: coresmith model build"}
    bad = frd_input_error(pr, db)
    if bad:
        return {"ok": False, "gate_ok": None, **bad}
    names = [b["name"] for b in db.block_specs()]
    # The evaluation is scoped to the FRD items that DECLARE a model check
    # (the others keep their RTL / chip-level acceptance), and every verdict
    # is bound to the exact model + harness bytes AND the requirement it
    # judged (readiness compares that digest: a changed model, harness, or
    # requirement text/bounds re-evaluates).
    from orchestrator.state_store.builds import soc_model_digest
    scope, shas = fe.model_check_inputs(db, pr)
    rec = asyncio.run(fe.evaluate(pr, md, names, arch=False, db=db, author=False, record_sha=shas, scope_ids=scope))
    return {"ok": bool(rec.get("gate_ok")), "model_digest": soc_model_digest(pr, db=db), "model_check_scope": scope, **rec}


def _author_outcome(out: dict) -> dict:
    """Normalise one generator result: ``provider_error`` (the call failed and
    the response is an error banner) outranks a file that happens to exist."""
    err = str(out.get("response_error") or "")
    return {"written": bool(out.get("written")) and not err, "files_written": out.get("files_written") or [],
            "notes": out.get("notes") or "", "provider_error": err}


def model_author(db, pr, blocks: list[str], *, diagnostics: str = "") -> dict:
    """``coresmith model author --block <b> [--block <b2>]``: one explicit
    SystemC model-author call per NAMED block. Nothing else is authored: a
    block that is not named stays missing. ``diagnostics`` (the text of a
    previous ``model build`` log) turns the call into a repair request."""
    from orchestrator.langgraph.pipeline_graph import author_block_models
    specs = db.block_specs()
    known = {b["name"]: b for b in specs}
    unknown = sorted(b for b in blocks if b not in known)
    if unknown:
        return {"ok": False, "error": "UNKNOWN_BLOCK", "unknown": unknown,
                "required": "a block registered in the block diagram (coresmith blocks)"}
    prims = sorted(b for b in blocks if str(known[b].get("kind") or "").lower() == "primitive")
    if prims:
        return {"ok": False, "error": "PRIMITIVE_BLOCK", "primitive": prims,
                "hint": "a primitive's model is generated by coresmith model build, never authored"}
    rec = asyncio.run(author_block_models(str(pr), specs, list(blocks), compiler_log=diagnostics))
    results = {b: _author_outcome(v) for b, v in (rec.get("blocks") or {}).items()}
    if rec.get("parked_reason"):
        return {"ok": False, "tool_error": True, "error": rec["parked_reason"], "blocks": results}
    provider_errors = {b: r["provider_error"] for b, r in results.items() if r["provider_error"]}
    return {"ok": all(r["written"] for r in results.values()) and not provider_errors,
            "blocks": results, "provider_errors": provider_errors,
            "written": sorted(b for b, r in results.items() if r["written"]),
            "not_written": sorted(b for b, r in results.items() if not r["written"]),
            "model_dir": str(Path(pr) / "model")}


def harness_author(db, pr, *, arch: bool = False, diagnostics: str = "") -> dict:
    """``coresmith harness author [--arch]``: one explicit FRD-harness author
    call for the SoC model (or the executable architecture model with
    ``--arch``). The model must already be built; ``diagnostics`` (a previous
    ``model eval`` build log) makes it a repair request."""
    from orchestrator.systemc_model import frd_eval as fe
    pr = Path(pr)
    if arch:
        from orchestrator.systemc_model import arch_model as am
        md = am.arch_dir(pr)
        if not (md / "arch_model_top.h").exists():
            return {"ok": False, "error": "ARCH_MODEL_NOT_BUILT", "required": str(md / "arch_model_top.h"),
                    "hint": "coresmith model build --arch"}
        try:
            blocks = [c.name for c in am.load_spec(pr).components]
        except Exception:  # noqa: BLE001
            blocks = []
    else:
        md = pr / "model"
        if not (md / "soc_model_top.h").exists():
            return {"ok": False, "error": "SOC_MODEL_NOT_ASSEMBLED", "required": str(md / "soc_model_top.h"),
                    "hint": "coresmith model build"}
        blocks = [b["name"] for b in db.block_specs()]
    bad = frd_input_error(pr, db)
    if bad:
        return {"ok": False, **bad}
    frd = pr / "arch" / "frd_spec.md"
    reqs = fe.overlay_bounds(fe.extract_requirements(frd.read_text(encoding="utf-8", errors="replace")), db)
    fe.write_requirements(md, reqs)
    from orchestrator.langchain.agents.frd_eval_generator import FRDEvalGenerator
    agent = FRDEvalGenerator()
    out = asyncio.run(agent.generate(project_root=str(pr), blocks=blocks, attempt=2 if diagnostics else 1,
                                     compiler_log=diagnostics, arch=arch))
    res = _author_outcome(out)
    res.update({"ok": res["written"] and not res["provider_error"], "sources": out.get("sources") or [],
                "harness_dir": str(md / "frd_eval"), "requirements": len(reqs)})
    return res
