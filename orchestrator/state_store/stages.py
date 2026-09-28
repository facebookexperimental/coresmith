# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The deterministic state machine of a run (architect sitting, step 1).

``coresmith stage next`` is the only way a run advances. Each stage has an
entry function that inspects the ontology and returns the exact reasons it
is blocked -- item ids, missing artifacts, unanswered questions -- never a
judgement. The LLM reads the list and goes and does the work.

Stages: requirements -> arch_model -> decomposition -> interfaces -> uarch ->
model_eval -> blocks -> integration -> acceptance -> backend. The last three
are entered by the pipeline graph (tier loop, acceptance, backend) and only
recorded here.
"""
from __future__ import annotations

import json
from pathlib import Path

from orchestrator.state_store.ontology import item_must_have

STAGES = ("requirements", "arch_model", "decomposition", "interfaces", "uarch", "model_eval",
          "blocks", "integration", "acceptance", "backend")


def _b(code: str, what: str, ids: list | None = None) -> dict:
    return {"code": code, "text": what, "ids": sorted(ids or [])[:50], "count": len(ids or [])}


def _entry_requirements(db, pr: Path) -> list[dict]:
    out = []
    for k in ("prd", "frd"):
        if not db.artifact(k):
            out.append(_b("MISSING_ARTIFACT", f"register the {k.upper()}: coresmith register {k} <path>"))
    if out:
        return out
    frd_items = [i for i in db.items(artifact="frd")]
    kpis = db.items(kind="KPI", artifact="prd")
    covered = {lk["to_id"] for lk in db.links(rel="derives_from")} | {lk["to_id"] for lk in db.links(rel="covers")}
    missing = [k["id"] for k in kpis if k["id"] not in covered]
    if missing:
        out.append(_b("KPI_UNCOVERED", "every PRD validation KPI needs an FRD item that derives from it "
                      "(cite the KPI id in the FRD requirement text)", missing))
    no_acc = [i["id"] for i in frd_items if not i.get("acceptance")]
    if no_acc:
        out.append(_b("FRD_NO_ACCEPTANCE", "FRD items without acceptance criteria", no_acc))
    no_prio = [i["id"] for i in frd_items if not i.get("priority")]
    if no_prio:
        out.append(_b("FRD_NO_PRIORITY", "FRD items without a priority", no_prio))
    no_mc = [i["id"] for i in frd_items if item_must_have(i) and not i.get("model_check")]
    if no_mc:
        out.append(_b("FRD_NO_MODEL_CHECK", "must-have FRD items without a '**Model check**' line "
                      "(how the executable architecture model observes it, or why it cannot)", no_mc))
    qs = db.questions(open_only=True, must_answer=True)
    if qs:
        out.append(_b("OPEN_QUESTIONS", "answer with a ruling: coresmith question answer <id> --ruling <text>",
                      [f"Q{q['id']}" for q in qs]))
    return out


def _entry_arch_model(db, pr: Path) -> list[dict]:
    out = []
    if not db.artifact("arch_model"):
        return [_b("MISSING_ARTIFACT", "register the executable architecture model: coresmith model register --arch <dir>")]
    latest = {(c["item_id"], c["kind"]): c for c in db.checks(kind="model_eval", latest=True)}
    musts = [i for i in db.items(artifact="frd") if item_must_have(i)]
    unanswered = [i["id"] for i in musts if (i["id"], "model_eval") not in latest]
    failed = [i["id"] for i in musts if latest.get((i["id"], "model_eval"), {}).get("status") == "fail"]
    noreason = [i["id"] for i in musts if latest.get((i["id"], "model_eval"), {}).get("status") == "not_testable"
                and not (latest[(i["id"], "model_eval")].get("evidence") or "").strip()]
    if unanswered:
        out.append(_b("MODEL_EVAL_MISSING", "must-have FRD items with no model_eval check (run coresmith model eval)", unanswered))
    if failed:
        out.append(_b("MODEL_EVAL_FAILED", "FRD items the architecture model fails", failed))
    if noreason:
        out.append(_b("MODEL_EVAL_NO_REASON", "not_testable without a reason", noreason))
    return out


def _entry_decomposition(db, pr: Path) -> list[dict]:
    out = []
    if not db.artifact("block_diagram"):
        return [_b("MISSING_ARTIFACT", "register the block diagram: coresmith register block_diagram <json>")]
    blocks = {b["name"] for b in db.block_specs()} if hasattr(db, "block_specs") else set()
    owned = {lk["from_id"] for lk in db.links(rel="owned_by")}
    musts = [i["id"] for i in db.items(artifact="frd") if item_must_have(i) and i["kind"] not in ("PHYS", "MPW")]
    unowned = [i for i in musts if i not in owned]
    if unowned:
        out.append(_b("REQ_UNOWNED", "must-have FRD items no block owns (add 'owns': [ids] to blocks or link: "
                      "coresmith link <item> block:<name> owned_by)", unowned))
    fabric = [b for b in (db.block_specs() if hasattr(db, "block_specs") else []) if b.get("fabric") or b.get("primitive")]
    if not fabric and len(blocks) >= 4:
        out.append(_b("NO_FABRIC", "no fabric primitive in the diagram (derive it from the arch model: coresmith fabric derive)", []))
    if not db.artifact("ers"):
        out.append(_b("MISSING_ARTIFACT", "register the ERS (per-block requirements): coresmith register ers <json>"))
    return out


def _entry_interfaces(db, pr: Path) -> list[dict]:
    out = []
    if not db.artifact("contracts"):
        out.append(_b("MISSING_ARTIFACT", "register the interface contracts: coresmith register contracts <json>"))
    if not db.artifact("abi"):
        out.append(_b("MISSING_ARTIFACT", "register the HW/SW ABI (memory map, register maps, GPU ISA): coresmith register abi <md>"))
    if out:
        return out
    if not (pr / ".coresmith" / "vip_index.json").exists():
        out.append(_b("VIP_MISSING", "generate the interface VIPs: coresmith vip generate"))
    snap = db.latest_integration_snapshot() if hasattr(db, "latest_integration_snapshot") else None
    if not snap or not snap.get("elaborated"):
        out.append(_b("SHELL_NOT_ELABORATED", "the stub shell top must elaborate: coresmith shell assemble"))
    elif not snap.get("boundary_ports"):
        out.append(_b("SHELL_NO_BOUNDARY", "the shell has no boundary ports: its port set must equal the declared top"))
    return out


def _entry_uarch(db, pr: Path) -> list[dict]:
    out = []
    blocks = [b["name"] for b in db.block_specs()] if hasattr(db, "block_specs") else []
    missing = [b for b in blocks if not db.artifact(f"uarch:{b}") and not _is_primitive(db, b)]
    if missing:
        out.append(_b("UARCH_MISSING", "register a uArch spec per block: coresmith register uarch --block <b> <md>", missing))
    perf = {i["id"] for i in db.items(kind="PERF", artifact="frd") if item_must_have(i)}
    cited = {lk["to_id"] for lk in db.links(rel="cites")}
    uncited = sorted(perf - cited)
    if perf and uncited and not missing:
        out.append(_b("PERF_UNCITED", "must-have PERF items no uArch spec cites in its §6.1 budget", uncited))
    return out


def _entry_model_eval(db, pr: Path) -> list[dict]:
    out = []
    blocks = [b["name"] for b in db.block_specs()] if hasattr(db, "block_specs") else []
    rows = {m["block"]: m for m in db.models()} if hasattr(db, "models") else {}
    nobuild = [b for b in blocks if not rows.get(b, {}).get("build_ok")]
    if nobuild:
        out.append(_b("MODEL_NOT_BUILT", "block models missing or not building: coresmith model refine <block>", nobuild))
    fe = pr / ".coresmith" / "frd_eval.json"
    if not fe.exists():
        out.append(_b("FRD_EVAL_MISSING", "run the FRD evaluation on the refined model: coresmith model eval", []))
    else:
        try:
            s = json.loads(fe.read_text()).get("summary") or {}
            if not s.get("gate_ok"):
                out.append(_b("FRD_EVAL_FAILED", "FRD evaluation not passing", (s.get("failed") or []) + (s.get("unanswered_must") or [])))
        except ValueError:
            out.append(_b("FRD_EVAL_UNREADABLE", ".coresmith/frd_eval.json unreadable", []))
    return out


def _entry_blocks(db, pr: Path) -> list[dict]:
    return []   # entered by the pipeline graph once model_eval is done


def _is_primitive(db, block: str) -> bool:
    for b in db.block_specs():
        if b["name"] == block:
            return bool(b.get("primitive") or b.get("kind") == "primitive")
    return False


_ENTRY = {"requirements": _entry_requirements, "arch_model": _entry_arch_model, "decomposition": _entry_decomposition,
          "interfaces": _entry_interfaces, "uarch": _entry_uarch, "model_eval": _entry_model_eval, "blocks": _entry_blocks}


def entry(db, project_root, stage: str) -> list[dict]:
    """Why ``stage`` cannot be marked done (empty = its exit criteria hold)."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; stages: {STAGES}")
    fn = _ENTRY.get(stage)
    return fn(db, Path(project_root)) if fn else []


def current(db) -> str:
    rows = {r["name"]: r for r in db.stage_rows()}
    for s in STAGES:
        if rows.get(s, {}).get("status") != "done":
            return s
    return STAGES[-1]


def status(db, project_root) -> dict:
    cur = current(db)
    blockers = entry(db, project_root, cur)
    return {"stage": cur, "index": STAGES.index(cur), "stages": STAGES, "blocked_by": blockers,
            "can_advance": not blockers, "done": [r["name"] for r in db.stage_rows() if r["status"] == "done"]}


def advance(db, project_root, *, actor: str = "cli") -> dict:
    """Mark the current stage done when its exit criteria hold and activate
    the next one. Refuses (``advanced=False``) with the blockers otherwise."""
    cur = current(db)
    blockers = entry(db, project_root, cur)
    if blockers:
        db.stage_set(cur, STAGES.index(cur), "active", blocked_by=blockers)
        return {"advanced": False, "stage": cur, "blocked_by": blockers}
    db.stage_set(cur, STAGES.index(cur), "done")
    nxt = STAGES[min(STAGES.index(cur) + 1, len(STAGES) - 1)]
    if nxt != cur:
        db.stage_set(nxt, STAGES.index(nxt), "active")
    try:
        from orchestrator.langgraph.event_stream import write_graph_event
        write_graph_event(str(project_root), "Stage", "stage_done", {"stage": cur, "next": nxt, "actor": actor})
    except Exception:  # noqa: BLE001
        pass
    return {"advanced": True, "done": cur, "stage": nxt, "blocked_by": entry(db, project_root, nxt)}
