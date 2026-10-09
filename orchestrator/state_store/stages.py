# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The deterministic state machine of a run, driven by the Architect (the
coding agent calling the CLI).

``coresmith stage next`` is the only way a stage is marked done. Each stage
has an entry function that inspects the ontology and the stored engine
results and returns the exact reasons it is blocked -- item ids, missing
artifacts, unanswered questions, missing or stale evidence -- never a
judgement. The Architect reads the list and goes and does the work.

Stages: requirements -> decomposition -> interfaces -> uarch -> blocks ->
integration -> acceptance -> backend.

* ``requirements`` .. ``interfaces`` are the shared architecture: documents,
  decomposition, contracts and pins. A project with no stage rows starts at
  ``requirements``; there is no legacy exemption.
* ``uarch`` is architecture readiness: every module has its registered uArch
  spec, its bound build target, its built and smoked SystemC reference model
  (current bytes, current contract version) with the model checks its
  requirements declare, and the worker binding the engine's helpers will run
  with. ``coresmith build module <name>`` needs the shared stages done and
  that module ready; it does not wait for unrelated modules.
* ``blocks`` completes only on recorded builds: every block's published
  pass (authored modules and engine primitives alike) names a ``completed``
  build whose recorded inputs still match the project and whose produced
  implementation is unchanged (``state_store/builds.py``). A pass published
  by hand or by a diagnostic verb carries no build and does not count.
* ``integration``, ``acceptance`` and ``backend`` complete only on the
  graph's own positive gates recorded against the current composition (every
  block's current build + the integrated top's bytes, named by a digest on
  each row): the elaborated real-RTL top plus the integration DV row, the
  validation DV row (which includes the acceptance DV), and a backend signoff
  PPA row -- each the LATEST verdict of its scope for the current run.

A must-have item whose only authoritative verdict is a failing ``model_eval``
check is still reported as an advisory for the RTL stages
(``MODEL_ONLY_FAILED``); but where a requirement *declares* a model check,
readiness requires that check to pass on the current model
(``MODEL_CHECK_FAILED``), and a later RTL pass never masks it.
"""
from __future__ import annotations

import os
from pathlib import Path

from orchestrator.state_store.ontology import item_must_have

STAGES = ("requirements", "decomposition", "interfaces", "uarch",
          "blocks", "integration", "acceptance", "backend")
# Stage names earlier revisions recorded; kept so a stored row is recognised.
LEGACY_STAGES = ("arch_model", "model_eval")
# The check kind a reference model produces. Advisory for the RTL stages;
# required at readiness for every requirement that declares a model check.
MODEL_CHECK_KIND = "model_eval"
# Stages whose completion a recorded build or graph gate establishes.
RESULT_STAGES = ("blocks", "integration", "acceptance", "backend")


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
    qs = db.questions(open_only=True, must_answer=True)
    if qs:
        out.append(_b("OPEN_QUESTIONS", "answer with a ruling: coresmith question answer <id> --ruling <text>",
                      [f"Q{q['id']}" for q in qs]))
    return out


def _entry_decomposition(db, pr: Path) -> list[dict]:
    out = []
    if not db.artifact("block_diagram"):
        return [_b("MISSING_ARTIFACT", "register the block diagram: coresmith register block_diagram <json>")]
    owned = {lk["from_id"] for lk in db.links(rel="owned_by")}
    musts = [i["id"] for i in db.items(artifact="frd") if item_must_have(i) and i["kind"] not in ("PHYS", "MPW")]
    unowned = [i for i in musts if i not in owned]
    if unowned:
        out.append(_b("REQ_UNOWNED", "must-have FRD items no block owns (add 'owns': [ids] to blocks or link: "
                      "coresmith link <item> block:<name> owned_by)", unowned))
    # A declared fabric primitive is validated by the fabric resolver and the
    # shell assembly; an explicit custom interconnect is a legitimate design.
    # There is no block-count heuristic that demands a primitive.
    if not db.artifact("ers"):
        out.append(_b("MISSING_ARTIFACT", "register the ERS (per-block requirements): coresmith register ers <json>"))
    return out


def _entry_interfaces(db, pr: Path, *, check_shell: bool = True) -> list[dict]:
    out = []
    # a registered document, or edges authored line by line (`contract add`
    # flips the artifact to db:contracts)
    if not db.artifact("contracts") and not (hasattr(db, "contract_rows") and db.contract_rows()):
        out.append(_b("MISSING_ARTIFACT", "define the interface contracts: coresmith contract add ... "
                      "(or coresmith register contracts <json>)"))
    if not db.artifact("abi"):
        out.append(_b("MISSING_ARTIFACT", "register the HW/SW ABI (memory map, register maps, GPU ISA): coresmith register abi <md>"))
    if out:
        return out
    if not (pr / ".coresmith" / "vip_index.json").exists():
        out.append(_b("VIP_MISSING", "generate the interface VIPs: coresmith vip generate"))
    if not check_shell:
        from orchestrator.state_store.pins import pin_problems
        pins = db.pins() if hasattr(db, "pins") else []
        if not pins and not infer_boundary_enabled():
            out.append(_b("PINS_MISSING", "declare the chip pins: coresmith pin add"))
        for pin in pins:
            out.extend(_b(p["code"], p["text"], [p["where"]])
                       for p in pin_problems(db, pin, replacing=pin["name"]))
        return out
    snap = db.latest_integration_snapshot() if hasattr(db, "latest_integration_snapshot") else None
    if infer_boundary_enabled():
        # CORESMITH_SHELL_INFER_BOUNDARY=1: the pre-pins heuristic
        if not snap or not snap.get("elaborated"):
            out.append(_b("SHELL_NOT_ELABORATED", "the stub shell top must elaborate: coresmith shell assemble"))
        elif not snap.get("boundary_ports"):
            out.append(_b("SHELL_NO_BOUNDARY", "the shell has no boundary ports: its port set must equal the declared top"))
        return out
    pins = db.pins() if hasattr(db, "pins") else []
    if not pins:
        out.append(_b("PINS_MISSING", "declare the chip pins: coresmith pin add <name> --dir in|out|inout [--width N] "
                      "--from <block>.<port>; clk/rst_n: coresmith pin add clk --dir in --kind clock "
                      "(every block port is a contract edge or a pin)"))
    if not snap or not snap.get("elaborated"):
        out.append(_b("SHELL_NOT_ELABORATED", "the stub shell top must elaborate: coresmith shell assemble"))
        errs = [str(e) for e in (snap or {}).get("wiring_errors") or []]
        undeclared = [e.split()[1].rstrip(":") for e in errs if e.startswith("SHELL_UNDECLARED_PORT ")]
        if undeclared:
            out.append(_b("SHELL_UNDECLARED_PORT", "block ports on no contract edge and no pin: "
                          "coresmith pin add <name> --dir .. --from <block>.<port> (or a contract edge)", undeclared))
        mism = [e for e in errs if e.startswith(("SHELL_BOUNDARY_MISMATCH", "PIN_DOUBLE_DRIVEN", "SHELL_PIN_"))]
        if mism:
            out.append(_b("SHELL_BOUNDARY_MISMATCH", "the shell's boundary differs from the declared pins: "
                          + "; ".join(mism)[:600]))
    elif pins:
        boundary = snap.get("boundary")
        names = {p["name"] for p in pins}
        if boundary is None:
            if int(snap.get("boundary_ports") or 0) != len(names):
                out.append(_b("SHELL_BOUNDARY_MISMATCH", "the shell was assembled before the pins changed: "
                              "coresmith shell assemble"))
        else:
            extra, missing = sorted(set(boundary) - names), sorted(names - set(boundary))
            # the synthesized clk/rst_n are accepted whether or not they are pins
            extra = [n for n in extra if n not in ("clk", "rst_n")]
            if extra or missing:
                out.append(_b("SHELL_BOUNDARY_MISMATCH", "the shell's boundary differs from the declared pins "
                              "(re-run coresmith shell assemble after pin changes)",
                              [f"extra:{n}" for n in extra] + [f"missing:{n}" for n in missing]))
    return out


def _entry_uarch(db, pr: Path) -> list[dict]:
    """Architecture readiness of every module (the exit of ``uarch``): the
    per-module readiness of :func:`module_ready` for each non-primitive
    block, merged by code."""
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
    merged: dict[str, dict] = {}
    for name in blocks:
        if _is_primitive(db, name):
            continue
        for blk in module_ready(db, pr, name, include_spec=False):
            cur = merged.get(blk["code"])
            if cur is None:
                merged[blk["code"]] = dict(blk)
            else:
                ids = sorted(set(cur["ids"]) | set(blk["ids"]))
                cur["ids"], cur["count"] = ids[:50], len(ids)
    out.extend(merged.values())
    return out


def _declared_model_items(db, name: str) -> list[dict]:
    """Live must-have items the module owns that *declare* a model check."""
    from orchestrator.state_store.builds import owned_items
    out = []
    for iid in owned_items(db, name):
        it = db.item(iid)
        if not it or not item_must_have(it) or it.get("status") in ("retired", "waived"):
            continue
        if str(it.get("model_check") or "").strip():
            out.append(it)
    return out


def module_ready(db, pr: Path, name: str, *, include_spec: bool = True) -> list[dict]:
    """Why module ``name`` cannot be built (empty = ready). Deterministic,
    from declared inputs only; nothing here runs a tool or a worker.

    * ``UARCH_MISSING`` -- no registered uArch spec.
    * ``TARGET_UNBOUND`` / ``TARGET_INVALID`` -- no (valid) bound build target
      (``coresmith target bind <name> --file <json>``).
    * ``MODEL_MISSING`` / ``MODEL_INVALID`` / ``MODEL_NOT_BUILT`` /
      ``MODEL_STALE`` -- the SystemC reference model is absent, its
      dependencies cannot be read, it is not built and smoked, or its
      recorded build is for other bytes (the implementation OR a header it
      includes), an older contract version, or predates dependency tracking
      (``coresmith model build``). The implementation is the registered
      ``models.path`` when a row names one, else ``model/<name>_model.cpp``.
    * ``MODEL_CHECK_MISSING`` / ``MODEL_CHECK_FAILED`` / ``MODEL_CHECK_STALE``
      -- a requirement the module owns declares a model check and the latest
      ``model_eval`` verdict is absent, failing, or bound to another model,
      harness or requirement text/bounds (``coresmith model eval``). Only
      declared model checks are required; other requirements keep their
      RTL-level scope.
    * ``WORKER_UNBOUND`` -- the worker binding is not declared in the
      persisted ``.coresmith/env`` (provider and its model).
    * ``FRD_TARGET_INVALID`` / ``FRD_TARGET_UNBOUND`` / ``FRD_TARGET_OWNER_MISMATCH`` /
      ``ACCEPTANCE_TB_MISSING`` / ``ACCEPTANCE_TEST_MISSING`` /
      ``ACCEPTANCE_TB_AMBIGUOUS`` -- the FRD target allocation the module's
      build would bind is unusable (``state_store.module_targets``).

    An engine primitive declares none of these: its RTL, testbench and spec
    are generated by its build.
    """
    from orchestrator.state_store import builds as B
    pr = Path(pr)
    out: list[dict] = []
    if _is_primitive(db, name):
        return out
    if include_spec and not db.artifact(f"uarch:{name}"):
        out.append(_b("UARCH_MISSING", "register a uArch spec per block: coresmith register uarch --block <b> <md>", [name]))
    tgt = B.target_identity(pr, name)
    if tgt is None:
        out.append(_b("TARGET_UNBOUND", "bind the module's build target: coresmith target bind <name> --file <json>", [name]))
    elif tgt.get("error"):
        out.append(_b("TARGET_INVALID", f"the bound target is invalid: {tgt['error']}", [name]))
    model = B.model_identity(pr, name, db)
    row = db.model_for(name) if hasattr(db, "model_for") else None
    if not model["exists"]:
        out.append(_b("MODEL_MISSING", f"write the SystemC reference model {Path(model['path']).name} "
                      "(coresmith model author --block <b>), then coresmith model build", [name]))
    elif model.get("error"):
        out.append(_b("MODEL_INVALID", f"the reference model's dependencies cannot be read: {model['error']}", [name]))
    elif row is None or not (row.get("build_ok") and row.get("smoke_ok")):
        out.append(_b("MODEL_NOT_BUILT", "build and smoke the reference model: coresmith model build", [name]))
    elif (str(row.get("sha") or "") != model["sha16"]
          or not str(row.get("deps_sha") or "")
          or str(row.get("deps_sha") or "") != model["deps_sha256"]
          or _version(row.get("spec_contract_version")) != _version(db.block_contract_version(name))):
        out.append(_b("MODEL_STALE", "the recorded model build is for other bytes (implementation or an included "
                      "header), an older contract version, or predates dependency tracking: coresmith model build",
                      [name]))
    declared = _declared_model_items(db, name)
    if declared:
        digest = B.soc_model_digest(pr, db=db)
        missing_c, failed_c, stale_c = [], [], []
        for it in declared:
            chk = db.latest_check(it["id"], MODEL_CHECK_KIND) if hasattr(db, "latest_check") else None
            if chk is None:
                missing_c.append(it["id"])
            elif chk.get("status") != "pass":
                failed_c.append(it["id"])
            elif str(chk.get("sha") or "") != B.model_check_sha(db, pr, it["id"], model_digest=digest):
                stale_c.append(it["id"])
        if missing_c:
            out.append(_b("MODEL_CHECK_MISSING", "requirements that declare a model check have no model_eval verdict: "
                          "coresmith model eval", missing_c))
        if failed_c:
            out.append(_b("MODEL_CHECK_FAILED", "declared model checks whose latest model_eval verdict is not a pass "
                          "(an RTL pass never masks it): fix the model or the requirement", failed_c))
        if stale_c:
            out.append(_b("MODEL_CHECK_STALE", "declared model checks whose verdict was bound to other model or "
                          "harness bytes, or to the requirement as it read before an edit: coresmith model eval",
                          stale_c))
    worker = B.worker_binding(pr)
    if not worker["explicit"]:
        out.append(_b("WORKER_UNBOUND", "declare the worker binding in .coresmith/env: " + ", ".join(worker["missing"]),
                      [name]))
    # The FRD targets the module's build will be judged by must be bindable:
    # finite bounds, a measurement binding per required target, one owner,
    # an existing acceptance testbench defining every bound test.
    from orchestrator.state_store import module_targets as MT
    for p in MT.allocation(db, pr, name)["problems"]:
        out.append(_b(p["code"], f"{p['where']}: {p['text']}", [p["where"]]))
    return out


def _version(v) -> str:
    """A contract version as the model row and the ontology may each spell
    it (``0`` / ``"0"`` / ``""`` / None are the same version)."""
    try:
        return str(int(str(v if v is not None else 0).strip() or 0))
    except (TypeError, ValueError):
        return str(v).strip()


def shared_ready(db, pr: Path) -> list[dict]:
    """Why no module can be built yet: the shared architecture stages
    (requirements, decomposition, interfaces) are not done, or a done stage
    no longer holds. ``STAGE_MACHINE_UNUSED`` when the project has no stage
    rows: it starts at ``requirements``; nothing is exempt."""
    rows = {r["name"]: r for r in db.stage_rows()}
    if not rows:
        return [_b("STAGE_MACHINE_UNUSED", "the project has no stage rows: it starts at requirements "
                   "(coresmith stage status / coresmith stage next)", [])]
    out = []
    for s in STAGES[:STAGES.index("uarch")]:
        if rows.get(s, {}).get("status") != "done":
            out.append(_b("STAGE_BEFORE_UARCH", f"stage {s} is not done: coresmith stage next", [s]))
            break
    out.extend(regressed_stages(db, pr, upto="uarch", check_shell=False))
    return out


def regressed_stages(db, pr, *, upto: str | None = None, check_shell: bool = True) -> list[dict]:
    """Done stages whose STRUCTURAL exit criteria no longer hold on the
    current inputs (``STAGE_REGRESSED``): a registered document gone, a pin
    set changed, a module no longer ready. Stored rows are history and are
    not rewritten; effective completion is what the inputs support now. A
    failed must-have verdict (``MUST_HAVE_FAILED``) is a downstream RESULT,
    not a regression of the architecture: it blocks advancing past the
    current stage but never prohibits the repair build that would fix it."""
    rows = {r["name"]: r for r in db.stage_rows()}
    out = []
    for s in STAGES:
        if upto and STAGES.index(s) >= STAGES.index(upto):
            break
        if rows.get(s, {}).get("status") != "done":
            continue
        # A broken implementation cannot prevent the module build needed to
        # repair it. Stage advancement still requires shell elaboration.
        blockers = (_entry_interfaces(db, Path(pr), check_shell=False)
                    if s == "interfaces" and not check_shell else entry(db, pr, s, include_failed=False))
        if blockers:
            out.append({"code": "STAGE_REGRESSED", "stage": s,
                        "text": f"stage {s} was marked done but its exit criteria no longer hold",
                        "ids": [b["code"] for b in blockers][:50], "count": len(blockers), "blocked_by": blockers})
    return out


def _primitive_materialized(pr: Path, spec: dict) -> bool:
    tgt = str(spec.get("rtl_target") or "").strip()
    if tgt and (pr / tgt).is_file():
        return True
    try:
        from orchestrator.langgraph.integration_helpers import primitive_rtl_target
        p = primitive_rtl_target(pr, spec)
    except Exception:  # noqa: BLE001
        p = None
    return bool(p and p.is_file())


def _entry_blocks(db, pr: Path) -> list[dict]:
    """The exit criteria of ``blocks`` (what ``stage next`` into integration
    needs): every block -- authored module or engine primitive -- has a
    published ``best`` that names a ``completed`` recorded build whose inputs
    still match the project and whose produced implementation is unchanged
    (``BLOCK_NOT_BUILT`` / ``PRIMITIVE_NOT_BUILT`` otherwise: a pass published
    by hand, by ``block-done`` outside a build, or by a build that went stale
    is not a deliverable, and a primitive's materialized file alone is not a
    graph receipt); and every must-have item owned by a published block is
    verified at RTL level or above -- the latest check of its top-ranked kind
    passes and, when the item is bounded, carries the measured value. There
    is no switch that opens this exit."""
    from orchestrator.state_store import builds as B
    from orchestrator.state_store.ontology import check_rank
    specs = db.block_specs() if hasattr(db, "block_specs") else []
    out: list[dict] = []
    unpublished, unmaterialized, not_built, prim_not_built, published = [], [], [], [], set()
    for spec in specs:
        name = spec["name"]
        best = db.result(name, "best") or {}
        primitive = B.is_primitive_spec(spec)
        if not best or not best.get("done"):
            if primitive:
                (prim_not_built if _primitive_materialized(pr, spec) else unmaterialized).append(
                    f"{name} (materialized, no graph build)" if _primitive_materialized(pr, spec) else name)
            else:
                unpublished.append(name)
            continue
        cur = B.current_build_status(db, pr, name)
        if cur["ok"]:
            published.add(name)
        else:
            (prim_not_built if primitive else not_built).append(f"{name} ({'; '.join(cur['reasons'])[:160]})")
    if unpublished:
        out.append(_b("BLOCKS_UNPUBLISHED", "blocks with no published pass: build them through the module graph "
                      "(coresmith build module <block>)", unpublished))
    if not_built:
        out.append(_b("BLOCK_NOT_BUILT", "published passes that are not a current recorded build (no build id, a "
                      "build that is not completed, inputs changed since, or the implementation was edited after "
                      "it): coresmith build module <block>", not_built))
    if unmaterialized:
        out.append(_b("PRIMITIVE_UNMATERIALIZED", "engine primitives whose rtl_target is not materialized "
                      "(coresmith build module <primitive>, or the pipeline's tier loop)", unmaterialized))
    if prim_not_built:
        out.append(_b("PRIMITIVE_NOT_BUILT", "engine primitives without a current graph build receipt (a "
                      "materialized file alone is not verification): coresmith build module <primitive>",
                      prim_not_built))
    owner: dict[str, str] = {}
    for lk in db.links(rel="owned_by"):
        to = str(lk.get("to_id") or "")
        if to.startswith("block:") and to[6:] in published:
            owner[lk["from_id"]] = to[6:]
    unverified, unmeasured = [], []
    rtl_rank = check_rank("block_dv")
    for item_id in sorted(owner):
        it = db.item(item_id) if hasattr(db, "item") else None
        if not it or not item_must_have(it) or it.get("status") in ("waived", "retired"):
            continue
        verifiers = db.verifiers(item_id=item_id) if hasattr(db, "verifiers") else []
        later_scopes = {"integration", "validation", "acceptance", "backend", "signoff"}
        if verifiers and all(v.get("kind") == "chip" or
                             str((v.get("args") or {}).get("scope") or "").lower() in later_scopes
                             for v in verifiers):
            continue  # explicitly verified at a later chip/acceptance scope
        latest = db.checks(item_id, latest=True)
        if not latest:
            unverified.append(item_id)
            continue
        top = max(check_rank(c["kind"]) for c in latest)
        chk = max((c for c in latest if check_rank(c["kind"]) == top), key=lambda c: (c["ts"], c["id"]))
        ok = top >= rtl_rank and (chk["status"] == "pass" or (chk["status"] == "not_testable"
                                                             and (chk.get("evidence") or "").strip()))
        if not ok:
            unverified.append(item_id)
        elif chk["status"] == "pass" and (it.get("bound_min") is not None or it.get("bound_max") is not None) \
                and chk.get("value") is None:
            unmeasured.append(item_id)
    if unverified:
        out.append(_b("OWNED_ITEM_UNVERIFIED", "must-have items owned by a published block with no passing "
                      "RTL-level (block_dv or higher) check: re-verify the block (coresmith block-done)", unverified))
    if unmeasured:
        out.append(_b("BOUNDED_ITEM_UNMEASURED", "bounded must-have items whose passing check carries no measured "
                      "value: the block testbench must report the number for the item", unmeasured))
    return out


def _is_primitive(db, block: str) -> bool:
    for b in db.block_specs():
        if b["name"] == block:
            return bool(b.get("primitive") or b.get("kind") == "primitive")
    return False


# ---------------------------------------------------------------------------
# Positive gates past ``blocks``: the graph's own recorded verdicts, bound to
# the current composition (every module's current build) by time and run.
# ---------------------------------------------------------------------------

def _composition(db, pr) -> tuple[list[dict], float | None, str]:
    """``(blockers, newest_build_finished_at, run_id)``: the blocks' current
    builds. A block without a current build blocks every later stage; the
    newest completion time bounds the evidence that can speak for this
    composition. Which composition a verdict measured is re-checked per row
    from the manifest it recorded (``builds.composition_status``)."""
    from orchestrator.state_store import builds as B
    newest = None
    blockers = []
    for spec in db.block_specs() if hasattr(db, "block_specs") else []:
        name = spec["name"]
        cur = B.current_build_status(db, pr, name)
        if not cur["ok"]:
            blockers.append(name)
            continue
        fin = float((cur.get("build") or {}).get("finished_at") or 0)
        newest = fin if newest is None else max(newest, fin)
    out = []
    if blockers:
        out.append(_b("COMPOSITION_NOT_BUILT", "blocks without a current recorded build; later-stage evidence "
                      "cannot speak for this composition", blockers))
    rid = db.run_id() if hasattr(db, "run_id") else ""
    return out, newest, rid


def _names_current_composition(db, pr, row: dict) -> bool:
    from orchestrator.state_store import builds as B
    return B.composition_status(db, pr, row.get("composition_sha"))["ok"]


def _gate_rows(db, pr, scope: str, *, after: float | None, run_id: str,
               rows: list[dict] | None = None) -> list[dict]:
    """The gate-sourced verdict of ``scope`` that speaks for the current
    composition: the LATEST such row of this run must be a non-skipped pass
    (an earlier pass never outranks a later failure), recorded after
    ``after``, and the composition manifest it recorded -- every block's
    build, the top, the testbench and every other input the node declared,
    the tooling context -- must still be the project's. A row of another
    run, a row whose composition changed or that names none, and any row
    older than a block's current build never qualify. Returns ``[row]`` or
    ``[]``."""
    from orchestrator.state_store.store import Scoreboard
    cand = [r for r in (rows if rows is not None else Scoreboard(pr).dv_rows())
            if r.get("scope") == scope and r.get("source") == "gate" and (r.get("run_id") or "") == (run_id or "")]
    if not cand:
        return []
    last = max(cand, key=lambda r: (float(r.get("ts") or 0), int(r.get("id") or 0)))
    if last.get("passed") != 1 or last.get("skipped"):
        return []
    if after is not None and float(last.get("ts") or 0) < after:
        return []
    if not _names_current_composition(db, pr, last):
        return []
    return [last]


def _bound_hdl_top(db, name: str) -> str:
    """The HDL top module a registered module builds as: the ``top`` of its
    existing target binding (``coresmith target bind``), else its own name."""
    try:
        doc = db.target_binding(name) if hasattr(db, "target_binding") else None
    except Exception:  # noqa: BLE001
        doc = None
    top = str((doc or {}).get("top") or "").strip() if isinstance(doc, dict) else ""
    return top or name


def _elaboration(db, pr: Path) -> tuple[dict | None, str | None]:
    """``(evidence, problem)``: the project's positive elaboration evidence
    for the integrated top. The canonical adoption is the hierarchy-
    validated candidate receipt (``harness.top_module.write_candidate_receipt``,
    the sole adoption operation of the single-block, Caravel and Integration
    Lead paths): when a receipt exists it is the evidence -- its elaborated
    cells are the real blocks, a registered block it does not instantiate is
    not integrated -- and a receipt that no longer validates (the top or a
    source edited, a dependency moved, a replacement adopted) is a problem,
    never silently ignored. Without a receipt the deterministic shell's
    latest elaborated snapshot is the evidence. The evidence is normalized to
    ``{source, elaborated, stub_blocks, ts, rtl_path}``."""
    rec = None
    try:
        from orchestrator.harness.top_module import (
            CandidateError,
            read_candidate_receipt,
            validated_candidate,
        )
        rec = read_candidate_receipt(pr)
    except Exception:  # noqa: BLE001 - the harness is not importable here: the shell snapshot decides
        rec = None
    if rec:
        try:
            cur = validated_candidate(pr)
        except CandidateError as exc:
            return None, f"the adopted candidate no longer validates: {exc}"
        # the elaborated hierarchy: the receipt lists the root's descendants;
        # the adopted root itself is part of it (a single-block top IS the
        # block). A registered module is present when the HDL top its
        # existing target binding names (``coresmith target bind``, else the
        # module's own name) is one of these cells.
        cells = {str(c) for c in cur.get("elaborated_cells") or []} | {str(cur.get("top_module") or "")}
        blocks = [b["name"] for b in db.block_specs()] if hasattr(db, "block_specs") else []
        absent = [b for b in blocks if _bound_hdl_top(db, b) not in cells]
        try:
            import calendar
            import time as _time
            ts = float(calendar.timegm(_time.strptime(str(cur.get("written_at") or ""), "%Y-%m-%dT%H:%M:%SZ")))
        except (ValueError, TypeError, OverflowError):
            ts = 0.0
        return {"source": "candidate", "elaborated": True, "stub_blocks": absent,
                "ts": ts, "rtl_path": cur.get("top_rtl_path"), "top_module": cur.get("top_module")}, None
    snap = db.latest_integration_snapshot() if hasattr(db, "latest_integration_snapshot") else None
    if not snap:
        return None, None
    return {"source": "shell", "elaborated": bool(snap.get("elaborated")),
            "stub_blocks": [str(s) for s in snap.get("stub_blocks") or []], "ts": float(snap.get("ts") or 0),
            "rtl_path": snap.get("rtl_path")}, None


def _entry_integration(db, pr: Path) -> list[dict]:
    """``integration`` is done when the current composition's real-RTL top
    elaborates with no stubs -- the validated candidate receipt (the
    canonical adoption) or, without one, an ``integration_snapshots`` row --
    newer than every current build, and the graph's integration DV recorded,
    as its latest chip-scope verdict for this run, a pass that names this
    exact composition. A receipt that no longer validates, a snapshot or a
    row older than a block's build, or a row recorded under another
    composition (the top edited since), describes a previous composition."""
    out, newest, rid = _composition(db, pr)
    if out:
        return out
    ev, problem = _elaboration(db, pr)
    if problem:
        out.append(_b("INTEGRATION_CANDIDATE_STALE", problem + " (re-adopt the current top)", []))
    elif not ev or not ev.get("elaborated"):
        out.append(_b("INTEGRATION_NOT_ELABORATED", "the real-RTL top has not elaborated for this composition "
                      "(the graph's shell_update / integration_check)", []))
    elif ev.get("stub_blocks"):
        out.append(_b("INTEGRATION_STUBS", "the adopted top's elaborated hierarchy does not contain every registered "
                      "module's bound HDL top" if ev["source"] == "candidate"
                      else "the latest elaborated top still instantiates stubs", list(ev["stub_blocks"])))
    elif newest is not None and (int(ev["ts"]) if ev["source"] == "candidate" else ev["ts"]) < (
            int(newest) if ev["source"] == "candidate" else newest):
        # the receipt's written_at has one-second resolution: compare seconds
        out.append(_b("INTEGRATION_SNAPSHOT_STALE", "the latest elaborated top predates a block's current build", []))
    if not _gate_rows(db, pr, "chip", after=newest, run_id=rid):
        out.append(_b("INTEGRATION_DV_MISSING", "no passing integration DV (the latest chip-scope gate row of this "
                      "run, whose recorded composition -- builds, top, testbench, inputs -- is still the "
                      "project's) for the current composition", []))
    return out


def _entry_acceptance(db, pr: Path) -> list[dict]:
    """``acceptance`` is done when the graph's validation DV -- which runs the
    acceptance DV, the chip-top synthesizability check and the die rollup
    before recording -- left, as its latest validation-scope verdict for this
    run, a pass naming the current composition, after a passing integration
    DV of the same composition."""
    out, newest, rid = _composition(db, pr)
    if out:
        return out
    chip = _gate_rows(db, pr, "chip", after=newest, run_id=rid)
    if not chip:
        out.append(_b("INTEGRATION_DV_MISSING", "no passing integration DV recorded for this composition", []))
        return out
    since = max(float(r.get("ts") or 0) for r in chip)
    val = _gate_rows(db, pr, "validation", after=newest, run_id=rid)
    if not val:
        out.append(_b("VALIDATION_DV_MISSING", "no passing validation DV (acceptance DV included; the latest "
                      "validation-scope gate row of this run, naming this composition) recorded for the current "
                      "composition", []))
    elif float(val[0].get("ts") or 0) < since:
        out.append(_b("VALIDATION_DV_STALE", "the passing validation DV predates the latest integration DV", []))
    return out


def _entry_backend(db, pr: Path) -> list[dict]:
    """``backend`` is done when the backend graph's LATEST signoff row for
    the chip top of the current composition (``ppa_history`` with
    ``probe=backend``: DRC + LVS + extracted timing + chip gate-sim [+
    precheck]) is ``ppa_ok``, names this composition and this run, and was
    recorded after the validation DV. Power may be recorded as unavailable
    or estimated; it is context, not the signoff."""
    from orchestrator.state_store.store import Scoreboard
    out, newest, rid = _composition(db, pr)
    if out:
        return out
    val = _gate_rows(db, pr, "validation", after=newest, run_id=rid)
    if not val:
        out.append(_b("VALIDATION_DV_MISSING", "no passing validation DV recorded for this composition", []))
        return out
    since = max(float(r.get("ts") or 0) for r in val)
    ok = False
    for block in {r.get("block") for r in val}:
        cand = [r for r in Scoreboard(pr).ppa_rows(str(block))
                if r.get("probe") == "backend" and (r.get("run_id") or "") == (rid or "")]
        if not cand:
            continue
        last = max(cand, key=lambda r: (float(r.get("ts") or 0), int(r.get("id") or 0)))
        if (last.get("ppa_ok") == 1 and float(last.get("ts") or 0) >= since
                and _names_current_composition(db, pr, last)):
            ok = True
    if not ok:
        out.append(_b("BACKEND_SIGNOFF_MISSING", "no backend signoff (the latest probe=backend ppa_history row of "
                      "this run for the chip top, ppa_ok, naming this composition) recorded after the validation "
                      "DV (coresmith backend start)", []))
    return out


_ENTRY = {"requirements": _entry_requirements, "decomposition": _entry_decomposition,
          "interfaces": _entry_interfaces, "uarch": _entry_uarch, "blocks": _entry_blocks,
          "integration": _entry_integration, "acceptance": _entry_acceptance, "backend": _entry_backend}


def stage_fail_blocks_enabled() -> bool:
    """A failed must-have item blocks every stage (CORESMITH_STAGE_FAIL_BLOCKS,
    default on; ``0`` disables)."""
    return (os.environ.get("CORESMITH_STAGE_FAIL_BLOCKS", "1") or "1").strip().lower() \
        not in {"0", "false", "no", "off"}


def _authoritative_kind(db, item_id: str) -> str:
    """The kind of the check that decides ``item_id``'s status: the latest check
    of the highest-ranked kind present ('' when the item has no checks)."""
    from orchestrator.state_store.ontology import check_rank
    latest = db.checks(item_id, latest=True) if hasattr(db, "checks") else []
    if not latest:
        return ""
    top = max(check_rank(c["kind"]) for c in latest)
    chk = max((c for c in latest if check_rank(c["kind"]) == top), key=lambda c: (c["ts"], c["id"]))
    return str(chk.get("kind") or "")


def _split_failed(db) -> tuple[list[str], list[str]]:
    """Failed must-have item ids as ``(blocking, model_only)``. A failure whose
    authoritative check is a ``model_eval`` verdict is advisory: the model is
    optional evidence and never gates the RTL work that produces the
    higher-ranked check (the item keeps its derived ``failed`` status and its
    checks; a later RTL-level pass supersedes it under the normal rank rule)."""
    blocking, advisory = [], []
    for i in db.items(status="failed"):
        if not item_must_have(i):
            continue
        if _authoritative_kind(db, i["id"]) == MODEL_CHECK_KIND:
            advisory.append(i["id"])
        else:
            blocking.append(i["id"])
    return blocking, advisory


def _must_have_failed(db) -> list[dict]:
    failed, _ = _split_failed(db)
    if not failed:
        return []
    return [_b("MUST_HAVE_FAILED", "must-have items whose authoritative check failed (coresmith frd show <id>: "
               "fix the design and re-verify, or change the requirement)", failed)]


def advisories(db) -> list[dict]:
    """Informational notes ``stage status`` prints next to the blockers. They
    never affect ``can_advance``."""
    _, model_only = _split_failed(db)
    if not model_only:
        return []
    return [_b("MODEL_ONLY_FAILED", "must-have items whose only authoritative verdict is a failing model_eval "
               "check: model evidence is advisory; an RTL-level check (block_dv or higher) decides "
               "(coresmith frd show <id>)", model_only)]


def entry(db, project_root, stage: str, *, include_failed: bool = True) -> list[dict]:
    """Why ``stage`` cannot be marked done (empty = its exit criteria hold).
    Every stage is also blocked by a failed must-have item
    (``MUST_HAVE_FAILED``; CORESMITH_STAGE_FAIL_BLOCKS=0 disables) unless
    ``include_failed`` is False (the structural criteria alone, as
    :func:`regressed_stages` re-evaluates them)."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; stages: {STAGES}")
    fn = _ENTRY.get(stage)
    out = fn(db, Path(project_root)) if fn else []
    if include_failed and stage_fail_blocks_enabled():
        out = out + _must_have_failed(db)
    return out


def current(db) -> str:
    rows = {r["name"]: r for r in db.stage_rows()}
    for s in STAGES:
        if rows.get(s, {}).get("status") != "done":
            return s
    return STAGES[-1]


def status(db, project_root) -> dict:
    cur = current(db)
    blockers = entry(db, project_root, cur)
    rows = db.stage_rows()
    regressed = regressed_stages(db, project_root)
    return {"stage": cur, "index": STAGES.index(cur), "stages": STAGES, "blocked_by": blockers,
            "can_advance": not blockers and not regressed, "advisories": advisories(db),
            "regressed": regressed,
            "done": [r["name"] for r in rows if r["status"] == "done"],
            "legacy_rows": [r["name"] for r in rows if r["name"] not in STAGES],
            "unused": not rows}


def advance(db, project_root, *, actor: str = "cli") -> dict:
    """Mark the current stage done when its exit criteria hold and activate
    the next one. Refuses (``advanced=False``) with the blockers otherwise."""
    cur = current(db)
    blockers = entry(db, project_root, cur)
    regressed = regressed_stages(db, project_root)
    if regressed:
        # an earlier stage no longer holds: the machine does not move past it
        blockers = blockers + regressed
    if blockers:
        db.stage_set(cur, STAGES.index(cur), "active", blocked_by=blockers)
        return {"advanced": False, "stage": cur, "blocked_by": blockers, "advisories": advisories(db)}
    db.stage_set(cur, STAGES.index(cur), "done")
    nxt = STAGES[min(STAGES.index(cur) + 1, len(STAGES) - 1)]
    if nxt != cur:
        db.stage_set(nxt, STAGES.index(nxt), "active")
    payload = {"stage": cur, "next": nxt, "actor": actor}
    if cur == "interfaces" and contract_autolock_enabled():
        # the interface contracts and the chip pins freeze with the stage: a
        # locked edge/pin changes only through `--unlock --reason`
        # (best-effort; never blocks the advance)
        try:
            payload["locked_edges"] = int(db.lock_contracts())
        except Exception:  # noqa: BLE001
            payload["locked_edges"] = 0
        try:
            payload["locked_pins"] = int(db.lock_pins()) if hasattr(db, "lock_pins") else 0
        except Exception:  # noqa: BLE001
            payload["locked_pins"] = 0
    try:
        from orchestrator.langgraph.event_stream import write_graph_event
        write_graph_event(str(project_root), "Stage", "stage_done", payload)
    except Exception:  # noqa: BLE001
        pass
    res = {"advanced": True, "done": cur, "stage": nxt, "blocked_by": entry(db, project_root, nxt),
           "advisories": advisories(db)}
    for k in ("locked_edges", "locked_pins"):
        if k in payload:
            res[k] = payload[k]
    return res


def infer_boundary_enabled() -> bool:
    """``CORESMITH_SHELL_INFER_BOUNDARY=1``: the chip boundary is inferred from
    the ports no edge covers (no ``PINS_MISSING``; the old ``SHELL_NO_BOUNDARY``
    check). Default off: the boundary is the declared pins."""
    return (os.environ.get("CORESMITH_SHELL_INFER_BOUNDARY", "0") or "0").strip().lower() \
        in {"1", "true", "yes", "on"}


def model_eval_require_value() -> bool:
    """``CORESMITH_MODEL_EVAL_REQUIRE_VALUE`` (default ``1``): a bounded item's
    model_eval verdict must carry the measured value; ``0`` restores the
    verdict-only behaviour."""
    return (os.environ.get("CORESMITH_MODEL_EVAL_REQUIRE_VALUE", "1") or "1").strip().lower() \
        not in {"0", "false", "no", "off"}


def contract_autolock_enabled() -> bool:
    """Lock every interface-contract edge when ``interfaces`` completes
    (CORESMITH_CONTRACT_AUTOLOCK, default on; ``0`` disables)."""
    return (os.environ.get("CORESMITH_CONTRACT_AUTOLOCK", "1") or "1").strip().lower() \
        not in {"0", "false", "no", "off"}


def project_db_exists(project_root) -> bool:
    """Whether the run has a project DB at all. Graph call sites check this
    before opening one: opening imports and re-exports legacy JSON as read-only
    views, a side effect a run that never used the stage machine must not get
    from mere stage bookkeeping."""
    from orchestrator.state_store.project_db import DB_NAME
    return (Path(project_root) / ".coresmith" / DB_NAME).exists()


def record_entered(db, project_root, stage: str, *, actor: str = "graph") -> dict:
    """Record that the pipeline graph entered ``stage`` (blocks, integration,
    acceptance, backend).

    A no-op when the stages table is empty: the run was never driven through the
    Architect's stage machine, and fabricating ``done`` rows for stages nobody ran would lie.
    Otherwise each earlier stage must satisfy its normal exit gate before it
    can be marked done. The first blocked stage remains active and the later
    entry is refused; graph execution is not evidence that skipped work passed.
    Re-entering an active/done stage changes nothing. Emits a
    ``Stage``/``stage_entered`` graph event when something changed.
    """
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; stages: {STAGES}")
    rows = {r["name"]: r for r in db.stage_rows()}
    if not rows:
        return {"recorded": False, "reason": "stage machine unused"}
    if rows.get(stage, {}).get("status") in ("active", "done"):
        return {"recorded": True, "stage": stage, "newly_done": []}
    idx = STAGES.index(stage)
    newly_done = []
    for i, s in enumerate(STAGES[:idx]):
        if rows.get(s, {}).get("status") != "done":
            blockers = entry(db, project_root, s)
            if blockers:
                db.stage_set(s, i, "active", blocked_by=blockers)
                return {"recorded": False, "stage": stage,
                        "blocked_stage": s, "blocked_by": blockers,
                        "newly_done": newly_done}
            db.stage_set(s, i, "done")
            newly_done.append(s)
    db.stage_set(stage, idx, "active")
    try:
        from orchestrator.langgraph.event_stream import write_graph_event
        write_graph_event(str(project_root), "Stage", "stage_entered",
                          {"stage": stage, "newly_done": newly_done, "actor": actor})
    except Exception:  # noqa: BLE001
        pass
    return {"recorded": True, "stage": stage, "newly_done": newly_done}
