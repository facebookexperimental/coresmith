# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The FRD targets a module build must meet, as the build binds them.

There is no separate target store. A module's targets are the typed FRD
items the Architect already writes (``coresmith frd add|edit``: metric,
unit, ``--min`` / ``--max``, priority, ``derives_from``), allocated to the
module by ownership (``--owner``), and measured through their verifier
binding (``coresmith frd verifier``):

* a ``cocotb`` verifier bound to the module names a test of one of the
  module's **acceptance testbench** files; the test records the quantity
  with :func:`orchestrator.harness.measure.record` during the engine's own
  simulation (throughput, latency, any sim-measured number);
* an ``eda`` verifier names a quantity the engine's tools measure on the
  build's synthesized netlist under the condition its ``args`` declare:
  ``area_um2`` (yosys ``stat -liberty``; ``area_scope`` ``total`` counts
  bound memory macros from their Liberty, ``std_cell`` is the explicit
  standard-cell subtotal) or ``power_mw`` (OpenSTA ``report_power``,
  vectorless with the declared primary-input ``activity`` / ``duty``).

A live owned item with a metric and a finite bound is a **required** target
when it is a must-have and **advisory** otherwise; an item whose verifiers
are all chip-scope is **deferred** to integration/validation. Owned items
with module-scope cocotb verifiers and no bound are **functional**
requirements: their tests must pass in the acceptance run.

:func:`allocation` computes the binding with every problem that makes it
unusable (the build refuses it readably); :func:`evaluate` judges one
candidate ONLY from tool receipts it validates itself -- the method's
shape, the hashes of every file the tool read or wrote, and the value
re-derived from the tool's own output file. A scalar, a test name or a
JSONL row without a valid receipt is never a measurement. Nothing here runs
a tool or a worker, and nothing imports LangGraph.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

LATER_SCOPES = ("integration", "validation", "acceptance", "backend", "signoff")
MEASURING_KINDS = ("cocotb", "eda")

# The quantities an ``eda`` verifier can bind: the base unit the tools report
# in, the item units they convert to (factor = base units per item unit) and
# the measurement conditions a verifier may declare (anything else is refused:
# a condition the engine would not apply is never silently ignored).
EDA_MEASURES: dict[str, dict] = {
    "area_um2": {"label": "area", "base_unit": "um2",
                 "units": {"um2": 1.0, "um^2": 1.0, "µm²": 1.0, "um²": 1.0, "mm2": 1e6, "mm^2": 1e6},
                 "conditions": {"area_scope": ("total", "std_cell")},
                 "method": "yosys stat -liberty on the build's synthesized netlist (+ bound macro Liberty area)"},
    "power_mw": {"label": "power", "base_unit": "mW",
                 "units": {"mw": 1.0, "w": 1000.0, "uw": 1e-3, "µw": 1e-3},
                 "conditions": {"activity": (0.0, 2.0), "duty": (0.0, 1.0)},
                 "method": "OpenSTA report_power on the mapped netlist at the build SDC clock (vectorless: "
                           "set_power_activity -input with the declared activity/duty, propagated)"},
}
POWER_DEFAULTS = {"activity": 0.1, "duty": 0.5}
AREA_DEFAULTS = {"area_scope": "total"}
# Labels a cocotb verifier may carry: the test itself defines its workload.
COCOTB_LABELS = ("workload", "note")
COMMON_ARGS = ("scope",)

SIM_RECEIPT = "cocotb_sim"
AREA_RECEIPT = "yosys_stat_area"
POWER_RECEIPT = "opensta_report_power"

OUTCOMES = ("feasible", "target_miss", "unmeasured")


def _canon_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _sha(path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except (OSError, TypeError, ValueError):
        return None


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def _problem(code: str, where: str, text: str) -> dict:
    return {"code": code, "where": where, "text": text}


def later_scope(v: dict) -> bool:
    """A verifier that explicitly belongs after module DV (chip tests, or an
    ``args.scope`` of integration / validation / acceptance / backend /
    signoff)."""
    scope = str((v.get("args") or {}).get("scope") or "").strip().lower() if isinstance(v.get("args"), dict) else ""
    return v.get("kind") == "chip" or scope in LATER_SCOPES


def unit_factor(measure: str, unit: str | None) -> float | None:
    """Base units of ``measure`` per one ``unit`` (None: not convertible)."""
    spec = EDA_MEASURES.get(measure)
    if not spec:
        return None
    u = str(unit or "").strip()
    return spec["units"].get(u) or spec["units"].get(u.lower())


def entry_name(entry: str) -> str:
    return str(entry or "").rsplit(".", 1)[-1]


def defines_test(tb_text: str, entry: str) -> bool:
    """Whether a cocotb file defines the test ``entry`` (a decorated
    ``def``/``async def`` of that name)."""
    name = re.escape(entry_name(entry))
    return bool(re.search(r"@cocotb\.test\b", tb_text)) and bool(
        re.search(rf"^\s*(?:async\s+)?def\s+{name}\s*\(", tb_text, re.M))


def _abs(root: Path, p: str) -> Path:
    q = Path(p)
    return (q if q.is_absolute() else root / q).resolve()


def _rel(root: Path, p: Path) -> str:
    try:
        return str(p.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(p)


def acceptance_module(module: str) -> str:
    """The cocotb module name the PRIMARY acceptance file runs under (the
    engine copies it to ``sim_build/<module>/test_<module>.py``); every other
    acceptance file runs under its own file stem."""
    return f"test_{module}"


def supplemental_module(module: str) -> str:
    return f"test_{module}_supplemental"


def supplemental_path(acceptance_path: str | Path, module: str) -> Path:
    """Where the worker writes supplemental tests: next to the primary
    acceptance file."""
    return Path(acceptance_path).parent / f"{supplemental_module(module)}.py"


def _owners(db, item_id: str) -> set[str]:
    return {str(lk["to_id"])[6:] for lk in db.links(from_id=item_id, rel="owned_by")
            if str(lk.get("to_id") or "").startswith("block:")}


def _module_scope(v: dict, module: str, owners: set[str]) -> bool:
    """A verifier the module build runs: a cocotb test or an eda measurement
    bound to this module (or unbound to any block when the module is the
    item's only owner), not a later-scope one."""
    if v.get("kind") not in MEASURING_KINDS or later_scope(v):
        return False
    block = str(v.get("block") or "")
    return block == module or (not block and owners == {module})


def condition(binding: dict) -> dict:
    """The explicit measurement condition of an eda binding (defaults filled)."""
    args = binding.get("args") if isinstance(binding.get("args"), dict) else {}
    if binding.get("entry") == "power_mw":
        return {k: float(args.get(k, d)) for k, d in POWER_DEFAULTS.items()}
    if binding.get("entry") == "area_um2":
        return {"area_scope": str(args.get("area_scope", AREA_DEFAULTS["area_scope"]))}
    return {}


def measure_key(binding: dict) -> str:
    """The key of one distinct measurement: an eda quantity under its
    condition (two power targets at different activities are two runs)."""
    c = condition(binding)
    if binding.get("entry") == "power_mw":
        return f"power_mw@activity={c['activity']:g},duty={c['duty']:g}"
    if binding.get("entry") == "area_um2":
        return f"area_um2@{c['area_scope']}"
    return str(binding.get("entry") or "")


def _arg_problems(v: dict) -> list[str]:
    """Conditions a verifier declares that the engine would not apply."""
    args = v.get("args") if isinstance(v.get("args"), dict) else {}
    if v.get("args") not in (None, {}) and not isinstance(v.get("args"), dict):
        return [f"verifier #{v.get('id')} args must be a JSON object"]
    out = []
    if v.get("kind") == "cocotb":
        bad = sorted(set(args) - set(COCOTB_LABELS) - set(COMMON_ARGS))
        if bad:
            out.append(f"verifier #{v.get('id')}: {bad} are not measurement conditions a cocotb binding applies "
                       f"(the test defines its own stimulus; labels allowed: {list(COCOTB_LABELS)})")
        return out
    spec = EDA_MEASURES.get(str(v.get("entry") or "")) or {}
    allowed = spec.get("conditions") or {}
    bad = sorted(set(args) - set(allowed) - set(COMMON_ARGS))
    if bad:
        out.append(f"verifier #{v.get('id')}: unsupported measurement condition(s) {bad} for {v.get('entry')}; "
                   f"supported: {sorted(allowed)} (the build clock and liberty are the build's own)")
    for k, rng in allowed.items():
        if k not in args:
            continue
        if isinstance(rng, tuple) and rng and isinstance(rng[0], str):
            if args[k] not in rng:
                out.append(f"verifier #{v.get('id')}: {k} {args[k]!r} is not one of {list(rng)}")
        elif not (_finite(args[k]) and rng[0] <= float(args[k]) <= rng[1]):
            out.append(f"verifier #{v.get('id')}: {k} {args[k]!r} out of range [{rng[0]}, {rng[1]}]")
    return out


def _binding(root: Path, v: dict) -> dict:
    args = v.get("args") if isinstance(v.get("args"), dict) else {}
    out = {"verifier": v.get("id"), "kind": v.get("kind"), "entry": str(v.get("entry") or ""), "args": args,
           "block": str(v.get("block") or "")}
    if v.get("kind") == "cocotb":
        out["path"] = _rel(root, _abs(root, str(v.get("path") or "")))
    else:
        out["path"] = ""
        out["method"] = (EDA_MEASURES.get(out["entry"]) or {}).get("method", "")
        out["condition"] = condition(out)
        out["measure"] = measure_key(out)
    return out


def allocation(db, root, module: str) -> dict:
    """The target allocation of ``module`` as a build would bind it now.

    ``{module, targets, functional, deferred, unverified, acceptance,
    frd_revision, problems, advisories, digest}``. ``problems`` are
    refusal-grade (codes ``FRD_TARGET_INVALID`` / ``FRD_TARGET_UNBOUND`` /
    ``FRD_TARGET_OWNER_MISMATCH`` / ``ACCEPTANCE_TB_MISSING`` /
    ``ACCEPTANCE_TEST_MISSING`` / ``ACCEPTANCE_TB_AMBIGUOUS``); a build with
    any of them is refused before anything runs. ``digest`` is the identity
    of what the build is judged by (None when the module binds nothing)."""
    from orchestrator.state_store.builds import (
        _block_spec,
        file_sha256,
        item_digest,
        owned_items,
        tb_dependencies,
    )
    from orchestrator.state_store.ontology import item_must_have
    root = Path(root)
    targets: list[dict] = []
    functional: list[dict] = []
    deferred: list[str] = []
    unverified: list[str] = []
    problems: list[dict] = []
    advisories: list[dict] = []
    cocotb_paths: dict[str, list[str]] = {}
    for iid in owned_items(db, module):
        it = db.item(iid)
        if not it or it.get("status") in ("retired", "waived"):
            continue
        owners = _owners(db, iid)
        vs = db.verifiers(item_id=iid)
        mine = [v for v in vs if _module_scope(v, module, owners)]
        later = [v for v in vs if later_scope(v)]
        lo, hi = it.get("bound_min"), it.get("bound_max")
        bounded = lo is not None or hi is not None
        must = item_must_have(it)
        if bounded and len(owners) > 1:
            problems.append(_problem("FRD_TARGET_OWNER_MISMATCH", iid,
                                     f"a bounded target has one owner; {iid} is owned by {', '.join(sorted(owners))}"))
            continue
        derives = sorted(lk["to_id"] for lk in db.links(from_id=iid, rel="derives_from"))
        bad = []
        incomplete = [v for v in mine if v["kind"] == "cocotb"
                      and not (str(v.get("path") or "").strip() and str(v.get("entry") or "").strip())]
        if incomplete:
            bad.append("cocotb verifier " + ", ".join(f"#{v['id']}" for v in incomplete) + " needs --path and --entry")
        for v in mine:
            bad += _arg_problems(v)
        eda = [v for v in mine if v["kind"] == "eda"]
        if bounded:
            if not str(it.get("metric") or "").strip():
                bad.append("no --metric")
            if lo is not None and not _finite(lo):
                bad.append(f"--min {lo!r} is not finite")
            if hi is not None and not _finite(hi):
                bad.append(f"--max {hi!r} is not finite")
            if _finite(lo) and _finite(hi) and float(lo) > float(hi):
                bad.append(f"--min {lo} > --max {hi}")
            for v in eda:
                entry = str(v.get("entry") or "")
                if entry not in EDA_MEASURES:
                    bad.append(f"eda verifier #{v['id']} measures {entry!r}; one of {sorted(EDA_MEASURES)}")
                elif unit_factor(entry, it.get("unit")) is None:
                    bad.append(f"{entry} is measured in {EDA_MEASURES[entry]['base_unit']}; unit "
                               f"{it.get('unit') or '(none)'} is not one of {sorted(EDA_MEASURES[entry]['units'])}")
        elif eda:
            bad.append("an eda measurement needs a bound to judge (--metric with --min/--max)")
        if bad:
            problems.append(_problem("FRD_TARGET_INVALID", iid, "; ".join(bad)))
            continue
        for v in mine:
            if v["kind"] == "cocotb":
                cocotb_paths.setdefault(str(_abs(root, str(v["path"]))), []).append(iid)
        if bounded:
            if not mine:
                if later:
                    deferred.append(iid)
                elif must:
                    others = sorted({v["kind"] for v in vs})
                    problems.append(_problem(
                        "FRD_TARGET_UNBOUND", iid,
                        f"required target {iid} ({it.get('metric')}) has no measurement binding for {module}"
                        + (f" (its verifiers are {', '.join(others)}, which a module build does not run)" if others else "")
                        + ": coresmith frd verifier " + iid + " --kind cocotb --path <acceptance tb> --entry <test> "
                        "(the test calls harness.measure.record) or --kind eda --entry area_um2|power_mw"))
                else:
                    advisories.append(_problem("FRD_TARGET_UNBOUND", iid, "advisory target with no measurement "
                                               "binding: not measured"))
                continue
            targets.append({
                "id": iid, "artifact": it.get("artifact"), "kind": it.get("kind"), "text": str(it.get("text") or "")[:200],
                "metric": it.get("metric"), "unit": it.get("unit") or "", "bound_min": lo, "bound_max": hi,
                "priority": it.get("priority") or "", "required": must, "derives_from": derives,
                "item_digest": item_digest(it), "bindings": [_binding(root, v) for v in mine]})
        elif mine:
            functional.append({"id": iid, "required": must, "item_digest": item_digest(it),
                               "derives_from": derives, "bindings": [_binding(root, v) for v in mine]})
        elif later:
            deferred.append(iid)
        elif must:
            unverified.append(iid)
    # a verifier bound to this module for an item another block owns
    for v in db.verifiers(block=module):
        if v.get("kind") not in MEASURING_KINDS or later_scope(v):
            continue
        it = db.item(v["item_id"])
        if not it or it.get("status") in ("retired", "waived"):
            continue
        owners = _owners(db, v["item_id"])
        if module not in owners:
            problems.append(_problem("FRD_TARGET_OWNER_MISMATCH", v["item_id"],
                                     f"verifier #{v['id']} measures {v['item_id']} on {module}'s build, but the item "
                                     f"is owned by {', '.join(sorted(owners)) or 'no block'}: rebind it "
                                     f"(frd verifier --block) or move ownership (frd edit --owner {module})"))
    acceptance = _acceptance(root, module, cocotb_paths, targets + functional, _block_spec(db, module) or {},
                             problems, file_sha256, tb_dependencies)
    frd = db.artifact("frd") if hasattr(db, "artifact") else None
    out = {"module": module, "targets": targets, "functional": functional, "deferred": sorted(deferred),
           "unverified": sorted(unverified), "acceptance": acceptance,
           "frd_revision": ({"sha": frd.get("sha"), "version": frd.get("version"), "path": frd.get("path")}
                            if frd else None),
           "problems": problems, "advisories": advisories}
    out["digest"] = allocation_digest(out)
    return out


def _acceptance(root: Path, module: str, cocotb_paths: dict[str, list[str]], bound: list[dict], spec: dict,
                problems: list[dict], file_sha256, tb_dependencies) -> dict | None:
    """The acceptance testbench FILES the module-scope cocotb verifiers name:
    all of them are fixed inputs, all are simulated (the primary one -- the
    block's declared ``testbench`` when it is among them -- as
    ``test_<module>``, every other one under its own file stem), and every
    cocotb binding learns the module name its test runs under."""
    if not cocotb_paths:
        return None
    paths = sorted(cocotb_paths)
    declared = str(_abs(root, str(spec.get("testbench")))) if spec.get("testbench") else ""
    primary = declared if declared in cocotb_paths else paths[0]
    ordered = [primary] + [p for p in paths if p != primary]
    run_module = {primary: acceptance_module(module)}
    for p in ordered[1:]:
        run_module[p] = Path(p).stem
    names = list(run_module.values()) + [supplemental_module(module)]
    clashes = sorted({n for n in names if names.count(n) > 1})
    if clashes:
        problems.append(_problem("ACCEPTANCE_TB_AMBIGUOUS", module,
                                 f"acceptance files would run under the same cocotb module name {clashes}: rename "
                                 f"one (files: {', '.join(_rel(root, Path(p)) for p in ordered)})"))
    files = []
    for p in ordered:
        path = Path(p)
        entries = sorted({b["entry"] for t in bound for b in t["bindings"]
                          if b["kind"] == "cocotb" and str(_abs(root, b["path"])) == p})
        for t in bound:
            for b in t["bindings"]:
                if b["kind"] == "cocotb" and str(_abs(root, b["path"])) == p:
                    b["module"] = run_module[p]
        if not path.is_file():
            problems.append(_problem("ACCEPTANCE_TB_MISSING", _rel(root, path),
                                     f"the acceptance testbench bound by {', '.join(cocotb_paths[p])} does not "
                                     "exist; it is a build input the Architect supplies"))
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        missing = [e for e in entries if not defines_test(text, e)]
        if missing:
            problems.append(_problem("ACCEPTANCE_TEST_MISSING", _rel(root, path),
                                     "bound tests not defined as @cocotb.test in this acceptance file: "
                                     + ", ".join(missing)))
        files.append({"path": _rel(root, path), "abs_path": str(path), "sha256": file_sha256(path),
                      "deps": {_rel(root, Path(d)): sha for d, sha in tb_dependencies(path).items()},
                      "entries": entries, "module": run_module[p]})
    if not files:
        return None
    first = files[0]
    return {"files": files, "path": first["path"], "abs_path": first["abs_path"], "sha256": first["sha256"],
            "module": first["module"], "modules": [f["module"] for f in files],
            "supplemental": _rel(root, supplemental_path(Path(first["abs_path"]), module))}


def acceptance_files(acc: dict | None) -> list[dict]:
    """The acceptance files of a recorded allocation (one entry per file)."""
    if not acc:
        return []
    return list(acc.get("files") or [{k: acc.get(k) for k in ("path", "abs_path", "sha256", "deps", "module")}])


def _binding_identity(b: dict) -> dict:
    return {k: b.get(k) for k in ("kind", "path", "entry", "args", "block")}


def allocation_digest(alloc: dict | None) -> str | None:
    """What the build's evidence is judged by: every target and functional
    requirement (item content, bounds, unit, priority, derivation, the
    measurement bindings with their conditions). None when nothing is bound."""
    alloc = alloc or {}
    targets = alloc.get("targets") or []
    functional = alloc.get("functional") or []
    if not targets and not functional:
        return None
    return _canon_sha({
        "targets": [{**{k: t.get(k) for k in ("id", "artifact", "item_digest", "priority", "required", "metric",
                                             "unit", "bound_min", "bound_max", "derives_from")},
                     "bindings": [_binding_identity(b) for b in t.get("bindings") or []]} for t in targets],
        "functional": [{**{k: f.get(k) for k in ("id", "item_digest", "required", "derives_from")},
                        "bindings": [_binding_identity(b) for b in f.get("bindings") or []]} for f in functional],
    })


def acceptance_digest(acc: dict | None) -> str | None:
    if not acc:
        return None
    return _canon_sha([{k: f.get(k) for k in ("path", "sha256", "deps")} for f in acceptance_files(acc)])


def measurability(alloc: dict, *, liberty_present: bool, synth_generic: bool, sta_problem: str | None) -> list[dict]:
    """Why a REQUIRED target's bound measurement method cannot run in this
    environment (``FRD_TARGET_UNMEASURABLE``): area needs the liberty-mapped
    synthesis; power needs it and OpenSTA. An advisory target never gates:
    it is measured where possible and reported unmeasured otherwise."""
    out = []
    for t in alloc.get("targets") or []:
        if not t.get("required"):
            continue
        for b in t["bindings"]:
            if b["kind"] != "eda":
                continue
            why = []
            if synth_generic:
                why.append("CORESMITH_SYNTH_GENERIC maps to generic gates (no liberty area/power)")
            elif not liberty_present:
                why.append("no liberty file: synthesis is generic")
            if b["entry"] == "power_mw" and sta_problem:
                why.append(sta_problem)
            if why:
                out.append(_problem("FRD_TARGET_UNMEASURABLE", t["id"], f"{b['entry']} cannot be measured here: "
                                    + "; ".join(why)))
    return out


# ---------------------------------------------------------------- tool output parsing
_AREA_RE = re.compile(r"Chip area for (?:module|top module)\s+'?\\?([^':]*)'?:\s*([0-9.eE+-]+)")
_POWER_TOTAL_RE = re.compile(
    r"^\s*Total\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)", re.M)


def parse_chip_area(report_text: str) -> float | None:
    """The last ``Chip area for module`` of a yosys ``stat -liberty`` report."""
    found = _AREA_RE.findall(report_text or "")
    if not found:
        return None
    try:
        v = float(found[-1][1])
    except ValueError:
        return None
    return v if math.isfinite(v) and v >= 0 else None


def parse_power_report(text: str) -> dict[str, float] | None:
    """OpenSTA ``report_power``'s ``Total`` row (internal, switching, leakage,
    total -- Watts) as mW; None when absent or not finite."""
    m = None
    for m in _POWER_TOTAL_RE.finditer(text or ""):
        pass
    if m is None:
        return None
    try:
        vals = [float(m.group(i)) for i in range(1, 5)]
    except ValueError:
        return None
    if not all(math.isfinite(v) and v >= 0 for v in vals):
        return None
    return {"internal_mw": vals[0] * 1e3, "switching_mw": vals[1] * 1e3, "leakage_mw": vals[2] * 1e3,
            "power_mw": vals[3] * 1e3}


def tool_errors(text: str) -> list[str]:
    """``Error`` lines in a tool's output (OpenSTA can print one and still
    exit 0 after a failed link or SDC read)."""
    return [ln.strip()[:200] for ln in (text or "").splitlines() if re.match(r"^\s*(?:\[?ERROR|Error)\b", ln)]


def _liberty_cell_body(lib_path: str, cell: str) -> str | None:
    """The text of ``cell``'s group in a Liberty file (None if absent)."""
    try:
        text = Path(lib_path).read_text(errors="replace")
    except (OSError, TypeError):
        return None
    m = re.search(r'cell\s*\(\s*"?' + re.escape(cell) + r'"?\s*\)\s*\{', text)
    if not m:
        return None
    depth, i = 1, m.end()
    while i < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
    return text[m.end():i - 1]


def liberty_cell_area(lib_path: str, cell: str) -> float | None:
    """The cell-level ``area`` attribute of ``cell`` in a Liberty file."""
    body = _liberty_cell_body(lib_path, cell)
    if body is None:
        return None
    depth = 0
    for i, c in enumerate(body):
        depth += {"{": 1, "}": -1}.get(c, 0)
        if depth == 0 and body.startswith("area", i) and (i == 0 or not (body[i - 1].isalnum() or body[i - 1] == "_")):
            am = re.match(r"area\s*:\s*([0-9.eE+-]+)\s*;", body[i:])
            if am:
                try:
                    v = float(am.group(1))
                except ValueError:
                    return None
                return v if math.isfinite(v) and v >= 0 else None
    return None


def liberty_cell_has_power(lib_path: str, cell: str) -> bool:
    """Whether ``cell`` is power-characterized in its Liberty (internal or
    leakage power data): a timing/area-only model is not a power model."""
    body = _liberty_cell_body(lib_path, cell)
    return bool(body) and bool(re.search(r"\b(internal_power|leakage_power|cell_leakage_power)\b", body))


# ---------------------------------------------------------------- receipts
def _hash_problems(rec: dict, pairs, *, optional: tuple = ()) -> list[str]:
    """Every named file exists and has its recorded hash. Only a key in
    ``optional`` (the simulation's measurements.jsonl, which a run without a
    measurement never writes) may be absent -- and then on both sides."""
    out = []
    for path_key, sha_key in pairs:
        p, want = rec.get(path_key), rec.get(sha_key)
        if not isinstance(p, str) or not p:
            out.append(f"no {path_key}")
            continue
        got = _sha(p)
        if want is None:
            if path_key not in optional:
                out.append(f"the required {path_key} {Path(p).name} had no recorded hash (it did not exist)")
            elif got is not None:
                out.append(f"{Path(p).name} exists but the receipt recorded none")
        elif got != want:
            out.append(f"{Path(p).name} does not match its recorded hash")
    return out


def sim_receipt_problems(rec: Any, alloc: dict | None = None) -> list[str]:
    """Why ``rec`` is not the receipt of an engine simulation that ran the
    recorded acceptance files (empty = valid): its kind, the hashes of
    results.xml / measurements.jsonl and of every simulated input, and --
    with ``alloc`` -- that the acceptance files simulated are the recorded
    ones."""
    if not isinstance(rec, dict) or rec.get("kind") != SIM_RECEIPT:
        return ["no simulation receipt (the evidence of an engine simulation run)"]
    if rec.get("error"):
        return [str(rec["error"])]
    out = []
    if "measurements_sha256" not in rec or not _finite(rec.get("captured_at")):
        out.append("incomplete simulation receipt")
    out += _hash_problems(rec, (("results_xml", "results_sha256"), ("measurements", "measurements_sha256")),
                          optional=("measurements",))
    inputs = rec.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        out.append("the receipt lists no simulated inputs")
    else:
        out += [f"the simulated input {Path(p).name} had no hash (it did not exist)" for p, sha in inputs.items()
                if not sha]
        out += [f"{Path(p).name} changed after the simulation" for p, sha in inputs.items() if sha and _sha(p) != sha]
        for f in acceptance_files((alloc or {}).get("acceptance")):
            if inputs.get(str(Path(f["abs_path"]).resolve())) != f.get("sha256"):
                out.append(f"the simulation did not run the recorded acceptance file {f['path']}")
    return out


def eda_receipt_problems(m: Any, binding: dict) -> list[str]:
    """Why ``m`` (``{value, receipt}``) is not a tool measurement of
    ``binding`` (empty = valid): the receipt's kind for the measure, its
    condition equals the binding's, every file it names has its recorded
    hash, and ``value`` equals what the tool's own output file says."""
    if not isinstance(m, dict) or not _finite(m.get("value")):
        return ["no measurement"]
    rec = m.get("receipt")
    entry = binding.get("entry")
    want = {"area_um2": AREA_RECEIPT, "power_mw": POWER_RECEIPT}.get(entry)
    if not isinstance(rec, dict) or rec.get("kind") != want:
        return [f"no {entry} tool receipt"]
    out = _hash_problems(rec, (("report_path", "report_sha256"), ("netlist_path", "netlist_sha256"),
                               ("liberty", "liberty_sha256")))
    # ONE netlist: the timing verdict, the area and the power are of the same
    # bytes (a fan-out-repaired netlist never borrows the unrepaired one's
    # cheaper area or power)
    if rec.get("netlist_selection_error"):
        out.append(f"netlist selection failed: {rec['netlist_selection_error']}")
    if rec.get("timing_netlist_sha256") and rec.get("timing_netlist_sha256") != rec.get("netlist_sha256"):
        out.append("the timing verdict was measured on another netlist than this measurement "
                   f"(timing {str(rec['timing_netlist_sha256'])[:12]}, measured {str(rec.get('netlist_sha256'))[:12]})")
    for path_key, sha_key in (("timing_report_path", "timing_report_sha256"),
                              ("original_netlist_path", "original_netlist_sha256"),
                              ("original_report_path", "original_report_sha256")):
        if rec.get(path_key) and _sha(rec[path_key]) != rec.get(sha_key):
            out.append(f"{Path(rec[path_key]).name} does not match its recorded hash")
    for lib in rec.get("macro_libs") or []:
        if _sha(lib.get("path")) != lib.get("sha256"):
            out.append(f"macro liberty {Path(str(lib.get('path'))).name} does not match its recorded hash")
    try:
        text = Path(rec.get("report_path") or "").read_text(errors="replace")
    except OSError:
        text = ""
    cond = condition(binding)
    if entry == "area_um2":
        std = parse_chip_area(text)
        if std is None:
            out.append("the yosys report has no chip area")
        else:
            macro = sum(float(x.get("area_um2") or 0) * int(x.get("count") or 0) for x in rec.get("macros") or [])
            derived = std if cond["area_scope"] == "std_cell" else std + macro
            if rec.get("area_scope") != cond["area_scope"]:
                out.append(f"the receipt measured area scope {rec.get('area_scope')}, the target binds "
                           f"{cond['area_scope']}")
            if cond["area_scope"] == "total" and rec.get("macros_unresolved"):
                out.append("macro area unknown for " + ", ".join(map(str, rec["macros_unresolved"])))
            if not math.isclose(derived, float(m["value"]), rel_tol=1e-9, abs_tol=1e-9):
                out.append("the value does not match the yosys report")
    elif entry == "power_mw":
        out += _hash_problems(rec, (("sdc_path", "sdc_sha256"),))
        for mac in rec.get("macros") or []:
            if not liberty_cell_has_power(mac.get("lib"), mac.get("name")):
                out.append(f"macro {mac.get('name')} has no power characterization in its Liberty")
        if any("Creating black box" in ln for ln in text.splitlines()):
            out.append("OpenSTA created a black box: the figure leaves a cell out")
        for k in ("activity", "duty"):
            if not _finite(rec.get(k)) or not math.isclose(float(rec[k]), cond[k], rel_tol=0, abs_tol=1e-12):
                out.append(f"the receipt's {k} {rec.get(k)} is not the target's {cond[k]}")
        if not _finite(rec.get("clock_period_ns")) or float(rec["clock_period_ns"]) <= 0:
            out.append("no clock: the SDC the power was estimated at defines no period")
        errs = tool_errors(text)
        if errs:
            out.append("OpenSTA reported errors: " + "; ".join(errs[:3]))
        parsed = parse_power_report(text)
        if parsed is None:
            out.append("the OpenSTA report has no Total power")
        elif not math.isclose(parsed["power_mw"], float(m["value"]), rel_tol=1e-9, abs_tol=1e-12):
            out.append("the value does not match the OpenSTA report")
    return out


# ---------------------------------------------------------------- evaluation
def _tests(rec: dict) -> dict:
    from orchestrator.harness.tools.block import _cocotb_results
    return _cocotb_results(rec["results_xml"])


def _rows(rec: dict) -> list[dict]:
    from orchestrator.harness import measure as M
    return M.read(rec["measurements"]) if rec.get("measurements_sha256") else []


def _status_in(results: dict, entry: str, module: str | None) -> str | None:
    """The verdict of test ``entry`` run under cocotb module ``module`` (a
    same-named test of another module never answers for it)."""
    short = entry_name(entry)
    if not module:
        return results.get(entry) or results.get(short)
    for key, status in results.items():
        cls, _, name = key.rpartition(".")
        if name == short and cls and cls.split(".")[-1] == module:
            return status
    return None


def _pick(rows: list[dict], item_id: str, entry: str, module: str) -> tuple[dict | None, str]:
    """The last measurement of ``item_id`` recorded by the bound test
    ``entry`` in its own acceptance module; ``(row, reason-if-none)``."""
    mine = [r for r in rows if str(r.get("item") or "").strip() == item_id]
    if not mine:
        return None, f"no measurement of {item_id} was recorded (the test must call harness.measure.record)"
    short = entry_name(entry)
    by_test = [r for r in mine if entry_name(str(r.get("test") or "")) == short]
    if not by_test:
        return None, f"{item_id} was measured, but not by the bound test {short} (record(..., test={short!r}))"
    attributed = [r for r in by_test if str(r.get("module") or "") == module]
    if not attributed:
        return None, (f"the measurement of {item_id} by {short} was not recorded by the acceptance module {module} "
                      "(only the acceptance testbench's own bound tests measure targets)")
    return attributed[-1], ""


def _judge(t: dict, value: float) -> str:
    lo, hi = t.get("bound_min"), t.get("bound_max")
    return "pass" if (lo is None or value >= float(lo)) and (hi is None or value <= float(hi)) else "fail"


def _gap(t: dict, value: float) -> dict:
    lo, hi = t.get("bound_min"), t.get("bound_max")
    if hi is not None and value > float(hi):
        d = value - float(hi)
        return {"side": "max", "bound": float(hi), "over": d, "relative": (d / abs(float(hi))) if float(hi) else None}
    if lo is not None and value < float(lo):
        d = float(lo) - value
        return {"side": "min", "bound": float(lo), "under": d, "relative": (d / abs(float(lo))) if float(lo) else None}
    margin = min([x for x in ((float(hi) - value) if hi is not None else None,
                              (value - float(lo)) if lo is not None else None) if x is not None], default=None)
    return {"side": None, "margin": margin}


def evaluate(alloc: dict, *, sim: dict | None, eda: dict | None) -> dict:
    """One candidate judged from tool receipts the function validates itself.

    ``sim``: ``{"receipt": <cocotb_sim receipt>}`` of this candidate's
    acceptance simulation (None when there is none). The test verdicts and
    measurement rows are READ FROM the receipt's own results.xml and
    measurements.jsonl after their hashes check out -- caller-supplied
    ``tests`` / ``rows`` are ignored. ``eda``: ``{measure_key: {"value",
    "receipt"} | {"error"}}``; a value counts only when its receipt is the
    method's, for the binding's condition, and agrees with the tool report.

    Returns ``{outcome, targets: [...], functional: [...], missed,
    unmeasured}``: ``feasible`` only when every required target is measured
    and inside its bounds and every required functional test passed;
    ``unmeasured`` when any required measurement is missing or unproven
    (never a pass); else ``target_miss``. Advisory targets are judged and
    reported, never gating."""
    rec = (sim or {}).get("receipt") if isinstance(sim, dict) else None
    sim_probs = sim_receipt_problems(rec, alloc)
    tests, rows = ({}, []) if sim_probs else (_tests(rec), _rows(rec))
    eda = eda or {}
    out_targets = []
    for t in alloc.get("targets") or []:
        values, reasons, receipts, unmeasured, failed = [], [], [], False, False
        for b in t["bindings"]:
            if b["kind"] == "cocotb":
                if sim_probs:
                    unmeasured = True
                    reasons.append("no valid acceptance-simulation receipt: " + "; ".join(sim_probs)[:300])
                    continue
                module = b.get("module") or (alloc.get("acceptance") or {}).get("module")
                st = _status_in(tests, b["entry"], module)
                if st is None:
                    unmeasured = True
                    reasons.append(f"test {module}.{entry_name(b['entry'])} did not run (not in results.xml)")
                    continue
                if st != "pass":
                    failed = True
                    reasons.append(f"test {b['entry']} {st}")
                    continue
                row, why = _pick(rows, t["id"], b["entry"], module)
                if row is None:
                    unmeasured = True
                    reasons.append(why)
                    continue
                try:
                    v = float(row.get("value"))
                except (TypeError, ValueError):
                    v = float("nan")
                unit = str(row.get("unit") or "").strip()
                if not math.isfinite(v):
                    unmeasured = True
                    reasons.append(f"{b['entry']} recorded a non-finite value {row.get('value')!r}")
                    continue
                if t["unit"] and unit != t["unit"]:
                    unmeasured = True
                    reasons.append(f"{b['entry']} recorded unit {unit or '(none)'}; the target is in {t['unit']}")
                    continue
                values.append(v)
                receipts.append({"kind": "sim", "test": b["entry"], "module": module, "value": v, "unit": unit,
                                 "measurement": {k: row.get(k) for k in ("item", "value", "unit", "test", "module", "ts")},
                                 "receipt": rec})
            else:
                key = b.get("measure") or measure_key(b)
                m = eda.get(key) or {}
                probs = [str(m["error"])] if m.get("error") else eda_receipt_problems(m, b)
                if probs:
                    unmeasured = True
                    reasons.append(f"{key}: " + "; ".join(probs)[:300])
                    continue
                factor = unit_factor(b["entry"], t["unit"]) or 1.0
                v = float(m["value"]) / factor
                values.append(v)
                receipts.append({"kind": "eda", "measure": key, "value": v, "unit": t["unit"],
                                 "base_value": float(m["value"]), "base_unit": EDA_MEASURES[b["entry"]]["base_unit"],
                                 "receipt": m.get("receipt")})
        status = "unmeasured" if unmeasured else ("fail" if failed else None)
        value = None
        if values:
            judged = [(v, _judge(t, v)) for v in values]
            failing = [v for v, s in judged if s == "fail"]
            if failing:
                value = failing[0]
            else:
                value = max(values) if t["bound_max"] is not None and t["bound_min"] is None else min(values)
            if status is None:
                status = "fail" if failing else "pass"
        status = status or "unmeasured"
        out_targets.append({"id": t["id"], "required": t["required"], "metric": t["metric"], "unit": t["unit"],
                            "bound_min": t["bound_min"], "bound_max": t["bound_max"], "status": status,
                            "value": value, "values": values, "gap": _gap(t, value) if value is not None else None,
                            "reasons": reasons, "receipts": receipts})
    out_functional = []
    for f in alloc.get("functional") or []:
        if sim_probs:
            out_functional.append({"id": f["id"], "required": f["required"], "status": "unmeasured",
                                   "tests": {b["entry"]: None for b in f["bindings"]},
                                   "reasons": ["no valid acceptance-simulation receipt: " + "; ".join(sim_probs)[:300]]})
            continue
        statuses = {b["entry"]: _status_in(tests, b["entry"], b.get("module") or (alloc.get("acceptance") or {}).get("module"))
                    for b in f["bindings"]}
        st = ("unmeasured" if any(s is None for s in statuses.values()) else
              "pass" if all(s == "pass" for s in statuses.values()) else "fail")
        out_functional.append({"id": f["id"], "required": f["required"], "status": st, "tests": statuses})
    netlists = {r["receipt"].get("netlist_sha256") for t in out_targets for r in t["receipts"]
                if r["kind"] == "eda" and isinstance(r.get("receipt"), dict)}
    if len(netlists) > 1:
        for t in out_targets:
            if any(r["kind"] == "eda" for r in t["receipts"]):
                t["status"] = "unmeasured"
                t["reasons"].append("the candidate's area/power measurements name different netlists")
    req = [x for x in out_targets + out_functional if x["required"]]
    unmeasured = [x["id"] for x in req if x["status"] == "unmeasured"]
    missed = [x["id"] for x in req if x["status"] == "fail"]
    outcome = "unmeasured" if unmeasured else ("target_miss" if missed else "feasible")
    needs_sim = bool(alloc.get("functional")) or any(b["kind"] == "cocotb" for t in alloc.get("targets") or []
                                                     for b in t["bindings"])
    return {"outcome": outcome, "targets": out_targets, "functional": out_functional,
            "unmeasured": unmeasured, "missed": missed, "sim_problems": sim_probs if needs_sim else [],
            "sim_receipt": rec if needs_sim and not sim_probs else None}


def candidate_receipt_problems(alloc: dict, evaluation: dict) -> list[str]:
    """Re-validate every receipt a recorded evaluation used (publication):
    the simulation receipt and each eda receipt still describe their files."""
    out = []
    if evaluation.get("sim_receipt") is not None or any(
            r.get("kind") == "sim" for t in evaluation.get("targets") or [] for r in t.get("receipts") or []):
        out += [f"simulation: {p}" for p in sim_receipt_problems(evaluation.get("sim_receipt"), alloc)]
    elif any(f.get("status") == "pass" for f in evaluation.get("functional") or []):
        out.append("functional tests passed without a simulation receipt")
    for t in evaluation.get("targets") or []:
        binding_by_key = {b.get("measure") or measure_key(b): b
                          for at in alloc.get("targets") or [] if at["id"] == t["id"] for b in at["bindings"]}
        for r in t.get("receipts") or []:
            if r.get("kind") == "sim":
                if r.get("receipt") != evaluation.get("sim_receipt"):
                    out.append(f"{t['id']}: its measurement names another simulation receipt")
                continue
            b = binding_by_key.get(r.get("measure"))
            if b is None:
                out.append(f"{t['id']}: {r.get('measure')} is not a binding of the recorded target")
                continue
            out += [f"{t['id']}: {p}" for p in eda_receipt_problems({"value": r.get("base_value"),
                                                                      "receipt": r.get("receipt")}, b)]
    return out


def gap_report(module: str, evaluation: dict) -> str:
    """The actionable text a target miss hands the implementation worker."""
    lines = [f"TARGET MISS: {module} passed its acceptance tests and synthesis, but required FRD targets are not met.",
             "Change the RTL implementation (microarchitecture, pipelining, sharing, encoding) to meet them; keep the",
             "interface contract, the acceptance testbench and every passing test unchanged. Measured this attempt:"]
    for t in evaluation.get("targets") or []:
        bound = " and ".join(x for x in ((f">= {t['bound_min']:g}" if t["bound_min"] is not None else ""),
                                         (f"<= {t['bound_max']:g}" if t["bound_max"] is not None else "")) if x)
        val = "unmeasured" if t["value"] is None else f"{t['value']:g} {t['unit']}".strip()
        tag = "REQUIRED" if t["required"] else "advisory"
        line = f"  - {t['id']} [{tag}] {t['metric']} = {val}; target {bound} {t['unit']}".rstrip() + f" -> {t['status'].upper()}"
        g = t.get("gap") or {}
        if t["status"] == "fail" and g.get("side"):
            amount = g.get("over", g.get("under"))
            rel = f" ({g['relative'] * 100:.1f}%)" if g.get("relative") is not None else ""
            line += f": {'over' if g['side'] == 'max' else 'under'} by {amount:g}{rel}"
        if t["reasons"]:
            line += " | " + "; ".join(t["reasons"])[:300]
        lines.append(line)
    for f in evaluation.get("functional") or []:
        if f["status"] != "pass":
            lines.append(f"  - {f['id']} [{'REQUIRED' if f['required'] else 'advisory'}] tests "
                         + ", ".join(f"{k}={v or 'not run'}" for k, v in f["tests"].items()))
    return "\n".join(lines)


def brief(alloc: dict) -> str:
    """The build brief the implementation and verification workers read:
    the targets with their measurement method and conditions, the fixed
    acceptance testbench files and where supplemental tests go."""
    if not alloc or not (alloc.get("targets") or alloc.get("functional") or alloc.get("acceptance")):
        return ""
    lines = [f"## BUILD TARGETS for {alloc.get('module')} (FRD-linked; the build completes only when every REQUIRED "
             "target is measured and met)"]
    for t in alloc.get("targets") or []:
        bound = " and ".join(x for x in ((f">= {t['bound_min']:g}" if t["bound_min"] is not None else ""),
                                         (f"<= {t['bound_max']:g}" if t["bound_max"] is not None else "")) if x)
        how = []
        for b in t["bindings"]:
            if b["kind"] == "cocotb":
                how.append(f"measured by acceptance test {b['entry']} ({b['path']})"
                           + (f", workload {b['args'].get('workload')}" if (b.get("args") or {}).get("workload") else ""))
            else:
                how.append(f"{b['entry']} by {b.get('method') or 'the engine EDA flow'}"
                           + (f", condition {json.dumps(b.get('condition'))}" if b.get("condition") else ""))
        lines.append(f"- {t['id']} [{'REQUIRED' if t['required'] else 'advisory'}] {t['metric']} {bound} "
                     f"{t['unit']}".rstrip() + f" -- {'; '.join(how)}. {t['text']}")
    for f in alloc.get("functional") or []:
        lines.append(f"- {f['id']} [{'REQUIRED' if f['required'] else 'advisory'}] functional: tests "
                     + ", ".join(b["entry"] for b in f["bindings"]) + " must pass")
    acc = alloc.get("acceptance")
    if acc:
        lines += ["", "ACCEPTANCE TESTBENCH (the Architect's fixed oracle, READ-ONLY): "
                  + ", ".join(f["path"] for f in acceptance_files(acc)),
                  "Never edit, weaken, skip or replace it; the build records its hash and refuses a changed oracle.",
                  f"Supplemental tests (only to close coverage of uncovered behaviour) go in {acc['supplemental']}; "
                  "they may not record measurements."]
    return "\n".join(lines)
