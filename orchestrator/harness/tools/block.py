# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""``coresmith block-status <b>`` / ``coresmith block-done <b>`` (the
Architect's step 4): the block gate as a tool.

A cluster worker (one long-lived session owning several blocks) writes RTL,
assertions and testbenches with its file tools and can only *finish* a block
through ``block_done``: contract conformance -> lint + simulation (with the
edge VIPs) -> full synthesis -> timing. Only a pass publishes ``best`` (the
same record ``block_done_node`` writes), so the tier loop, the shell
assembly and the final integration see exactly what the graph would have
produced. Tool failures are typed ``tool_error`` and never count as a design
failure.
"""
from __future__ import annotations

import hashlib
import math
import os
import time
from pathlib import Path


def _spec(pr, name: str) -> dict:
    from orchestrator.harness import blocks as B
    return B.load_block_spec(pr, name) or {"name": name}


def _paths(pr, spec: dict) -> tuple[str, str]:
    from orchestrator.harness import verify as V
    return V._resolve_rtl_path(Path(pr), spec), V._resolve_tb_path(Path(pr), spec, None)


def _env_on(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def _cocotb_results(path) -> dict[str, str]:
    """A cocotb JUnit ``results.xml`` -> ``{testcase name: pass|fail|skipped}``.
    A name passes iff no instance has a failure/error/skipped child, fails if
    any instance has a failure/error, and is skipped otherwise. Each case is
    also keyed ``<classname>.<name>``. Missing/unparsable file -> ``{}``."""
    import xml.etree.ElementTree as ET
    try:
        root = ET.parse(str(path)).getroot()
    except (OSError, ET.ParseError):
        return {}
    by_name: dict[str, list] = {}
    for row in root.iter("testcase"):
        n = row.get("name")
        if not n:
            continue
        keys = [n] + ([f"{row.get('classname')}.{n}"] if row.get("classname") else [])
        for k in keys:
            by_name.setdefault(k, []).append(row)
    out = {}
    for n, rows in by_name.items():
        if any(r.find(t) is not None for r in rows for t in ("failure", "error")):
            out[n] = "fail"
        elif all(r.find("skipped") is None for r in rows):
            out[n] = "pass"
        else:
            out[n] = "skipped"
    return out


def _same_path(pr: Path, a: str, b: str) -> bool:
    if not a or not b:
        return False
    pa, pb = Path(a), Path(b)
    pa = pa if pa.is_absolute() else pr / pa
    pb = pb if pb.is_absolute() else pr / pb
    return pa.resolve() == pb.resolve()


def measurements_file(pr, name: str) -> Path:
    """The canonical measurements file of a block's simulation (see
    :mod:`orchestrator.harness.measure`)."""
    return Path(pr) / "sim_build" / name / "measurements.jsonl"


def _pick_measurement(rows: list[dict], item_id: str, entry: str) -> dict | None:
    """The last measurement of ``item_id`` recorded by ``entry`` (short or
    dotted name). Evidence from a different test never satisfies a verifier."""
    mine = [r for r in rows if str(r.get("item") or "").strip() == item_id]
    if not mine:
        return None
    short = entry.rsplit(".", 1)[-1]
    by_test = [r for r in mine if str(r.get("test") or "") in (entry, short)
               or str(r.get("test") or "").rsplit(".", 1)[-1] == short]
    return by_test[-1] if by_test else None


def _bounded(it: dict | None) -> bool:
    return bool(it) and (it.get("bound_min") is not None or it.get("bound_max") is not None)


def _owned_items(db, name: str) -> list[str]:
    owned: list[str] = []
    for lk in db.links(to_id=f"block:{name}", rel="owned_by"):
        if lk["from_id"] not in owned:
            owned.append(lk["from_id"])
    return owned


def _later_scope_verifier(v: dict) -> bool:
    """Whether a verifier explicitly belongs after block DV."""
    scope = str((v.get("args") or {}).get("scope") or "").strip().lower()
    return v.get("kind") == "chip" or scope in {"integration", "validation", "acceptance", "backend", "signoff"}


def stamp_checks(db, pr, *, kind: str, items: list[str], select, results_xml, measurements,
                 rtl_sha: str = "", actor: str = "", skip_missing: bool = False) -> dict:
    """Per-item checks of ``kind`` from one cocotb ``results.xml`` (the shared
    core of ``block-done``'s ``block_dv`` stamping and the chip-level
    ``integration_dv`` / ``validation_dv`` stamping).

    For each live item in ``items``: its verifiers for which ``select(v)`` is
    true decide; an item with no verifier at all is ``unverified``, one whose
    verifiers are all elsewhere is ``deferred``. A bounded item (metric +
    min/max) is never stamped ``pass`` from the cocotb verdict alone: when its
    test passed, the measurement the testbench recorded (``measurements``,
    see :mod:`orchestrator.harness.measure`) decides through the item's
    bounds; no measurement -> ``tool_error`` and the item is listed in
    ``unmeasured_items``. ``skip_missing``: a selected verifier whose test is
    not in this ``results.xml`` is skipped instead of stamped ``tool_error``
    (a path-less chip verifier names a test of the integration OR the
    validation testbench)."""
    from orchestrator.harness import measure as M
    pr = Path(pr)
    xml = Path(results_xml)
    results = _cocotb_results(xml)
    mfile = Path(measurements)
    mrows = M.read(mfile)
    checks, unverified, deferred, unmeasured = [], [], [], []
    for iid in items:
        it = db.item(iid)
        if it is not None and it.get("status") in ("retired", "waived"):
            continue
        vs = db.verifiers(item_id=iid)
        mine = [v for v in vs if v.get("entry") and select(v)]
        if not vs:
            unverified.append(iid)
            continue
        if not mine:
            deferred.append(iid)
            continue
        bounded = _bounded(it)
        outcomes = []
        for v in mine:
            entry = v["entry"]
            st = results.get(entry) or results.get(entry.rsplit(".", 1)[-1] if "." in entry else "")
            value = None
            if st is None:
                if skip_missing:
                    continue
                status = "tool_error"
                ev = (f"cocotb test {entry} not found in results.xml" if xml.exists()
                      else f"cocotb test {entry}: no results.xml at {xml}")
            elif st == "pass" and bounded:
                m = _pick_measurement(mrows, iid, entry)
                if m is None:
                    status = "tool_error"
                    ev = ("no measurement recorded for a bounded item; TB must call harness.measure.record "
                          f"({entry} passed; looked in {mfile})")
                    if iid not in unmeasured:
                        unmeasured.append(iid)
                else:
                    try:
                        value = float(m["value"])
                    except (TypeError, ValueError):
                        value = None
                    expected_unit = str(it.get("unit") or "").strip()
                    actual_unit = str(m.get("unit") or "").strip()
                    if value is None or not math.isfinite(value):
                        status = "tool_error"
                        ev = f"{entry}: measurement is not a finite number ({m.get('value')!r})"
                        value = None
                        if iid not in unmeasured:
                            unmeasured.append(iid)
                    elif expected_unit and actual_unit != expected_unit:
                        status = "tool_error"
                        ev = (f"{entry}: measurement unit {actual_unit or '(missing)'} does not match "
                              f"required unit {expected_unit}")
                        value = None
                        if iid not in unmeasured:
                            unmeasured.append(iid)
                    else:
                        from orchestrator.state_store.ontology import derive_status_from_bounds
                        status = derive_status_from_bounds(it, value)
                        ev = (f"{entry}: pass; measured {value:g} {actual_unit or expected_unit} "
                              f"(test {m.get('test') or '-'}, {mfile})").replace("  ", " ")
            else:
                status = st
                ev = f"{entry}: {st} ({xml})"
            outcomes.append((v, status, value, ev))
            row = {"item": iid, "entry": entry, "status": status, "verifier": v["id"]}
            if value is not None:
                row["value"] = value
            checks.append(row)
        if outcomes:
            statuses = [o[1] for o in outcomes]
            aggregate_status = ("fail" if "fail" in statuses else
                                "tool_error" if "tool_error" in statuses else
                                "skipped" if "skipped" in statuses else "pass")
            measured_values = [o[2] for o in outcomes if o[2] is not None]
            aggregate_value = None
            if aggregate_status == "pass" and measured_values:
                # One scalar is stored for compatibility, with all individual
                # values retained in evidence. Pick the conservative edge for
                # a one-sided bound so the aggregate cannot overstate margin.
                aggregate_value = (max(measured_values) if it.get("bound_max") is not None
                                   and it.get("bound_min") is None else min(measured_values))
            elif aggregate_status == "fail" and measured_values:
                from orchestrator.state_store.ontology import derive_status_from_bounds
                failing = [v for v in measured_values
                           if derive_status_from_bounds(it, v) == "fail"]
                aggregate_value = failing[0] if failing else None
            evidence = "; ".join(o[3] for o in outcomes)
            db.add_check(iid, kind, aggregate_status, evidence=evidence, sha=rtl_sha,
                         actor=actor, value=aggregate_value)
    failed = sorted({c["item"] for c in checks if c["status"] != "pass"})
    verified = sorted({c["item"] for c in checks} - set(failed))
    return {"item_checks": checks, "verified_items": verified, "failed_items": failed,
            "unverified_items": unverified, "deferred_items": deferred, "unmeasured_items": unmeasured,
            "results_xml": str(xml), "measurements": str(mfile)}


def stamp_item_checks(db, pr, name: str, tb: str, *, rtl_sha: str = "", actor: str = "",
                      measurements=None) -> dict:
    """Per-item ``block_dv`` checks from the block's cocotb ``results.xml``:
    each item owned by the block is checked through its own cocotb verifiers
    (bound to this block, or to its tb path); items with no verifier get no
    check. Bounded items are decided by the recorded measurement (``measurements``
    or ``sim_build/<block>/measurements.jsonl``); see :func:`stamp_checks`.
    Returns ``{item_checks, verified_items, failed_items, unverified_items,
    deferred_items, unmeasured_items, results_xml, measurements}``."""
    pr = Path(pr)

    def select(v: dict) -> bool:
        return v["kind"] == "cocotb" and (v.get("block") == name or _same_path(pr, v.get("path") or "", tb))
    return stamp_checks(db, pr, kind="block_dv", items=_owned_items(db, name), select=select,
                        results_xml=pr / "sim_build" / name / "results.xml",
                        measurements=Path(measurements) if measurements else measurements_file(pr, name),
                        rtl_sha=rtl_sha, actor=actor)


# ---------------------------------------------------------------------------
# Chip-level numbers into the FRD: integration_dv / validation_dv (from the
# chip testbench's results.xml + measurements.jsonl), synth (flat cell count),
# sta (chip WNS).
# ---------------------------------------------------------------------------
CHIP_DV_KINDS = {"integration": "integration_dv", "validation": "validation_dv"}


def chip_item_checks_enabled() -> bool:
    """``CORESMITH_CHIP_ITEM_CHECKS`` (default ``1``): chip-level simulations
    and the backend stamp per-item checks; ``0`` = they record none (old)."""
    return os.environ.get("CORESMITH_CHIP_ITEM_CHECKS", "1").strip().lower() not in ("0", "false", "no", "off")


def stamp_chip_checks(db, pr, *, kind: str, tb_path: str, sim_dir, actor: str = "engine") -> dict:
    """``integration_dv`` / ``validation_dv`` checks from a chip-level cocotb
    run (``sim_dir`` = ``sim_build/<scope>``): every live item with a verifier
    of kind ``chip`` (or any verifier whose path is this chip testbench) is
    stamped the same way ``block-done`` stamps ``block_dv``. A path-less
    ``chip`` verifier is stamped only by the run whose ``results.xml`` holds
    its test."""
    pr = Path(pr)
    sim_dir = Path(sim_dir)

    def by_path(v: dict) -> bool:      # bound to THIS chip testbench: a missing test is a tool_error
        return bool(v.get("path")) and _same_path(pr, v["path"], tb_path)

    def by_kind(v: dict) -> bool:      # path-less chip verifier: stamped by the run that holds its test
        return not v.get("path") and v["kind"] == "chip"
    vs = db.verifiers()
    common = {"kind": kind, "results_xml": sim_dir / "results.xml",
              "measurements": sim_dir / "measurements.jsonl", "actor": actor}
    out = stamp_checks(db, pr, items=sorted({v["item_id"] for v in vs if by_path(v)}), select=by_path, **common)
    rest = stamp_checks(db, pr, items=sorted({v["item_id"] for v in vs if by_kind(v)}), select=by_kind,
                        skip_missing=True, **common)
    for k in ("item_checks", "unmeasured_items"):
        out[k] = out[k] + [x for x in rest[k] if x not in out[k]]
    failed = sorted({c["item"] for c in out["item_checks"] if c["status"] != "pass"})
    out.update(failed_items=failed, verified_items=sorted({c["item"] for c in out["item_checks"]} - set(failed)),
               kind=kind)
    return out


def chip_top_items(db, metrics) -> list[dict]:
    """Live items whose ``metric`` is one of ``metrics`` and which no block
    other than the chip top owns (a block's ERS budget is that block's, not
    the chip's)."""
    want = {str(m).lower() for m in metrics}
    from orchestrator.harness.top_module import declared_top, read_candidate_receipt
    receipt = read_candidate_receipt(db.root) or {}
    top = str(receipt.get("top_module") or declared_top(db.root) or "").strip()
    out = []
    for it in db.items():
        if str(it.get("metric") or "").strip().lower() not in want:
            continue
        owners = {lk["to_id"].split(":", 1)[1] for lk in db.links(from_id=it["id"], rel="owned_by")
                  if str(lk["to_id"]).startswith("block:")}
        if not top or owners != {top}:
            continue
        out.append(it)
    return out


CELL_METRICS = ("cells", "cell_count", "std_cells")
WNS_METRICS = ("wns_ns", "ns", "wns")


def record_metric_checks(db, *, kind: str, metrics, value, evidence: str, actor: str = "engine") -> list[dict]:
    """One ``kind`` check carrying ``value`` on every bounded chip-level item
    whose metric is in ``metrics`` (pass/fail derived from the item's bounds;
    unbounded items are skipped). Returns ``[{item, status, value}]``;
    best-effort per item."""
    if value is None or not chip_item_checks_enabled():
        return []
    try:
        value = float(value)
    except (TypeError, ValueError):
        return []
    rows = []
    for it in chip_top_items(db, metrics):
        if not _bounded(it):
            continue
        try:
            cid = db.add_check(it["id"], kind, None, evidence=evidence, actor=actor, value=value)
        except Exception:  # noqa: BLE001
            continue
        st = next((c["status"] for c in reversed(db.checks(it["id"], kind=kind)) if c["id"] == cid), "?")
        rows.append({"item": it["id"], "status": st, "value": value})
    return rows


def block_status(db, pr, name: str) -> dict:
    pr = Path(pr)
    spec = _spec(pr, name)
    rtl, tb = _paths(pr, spec)
    best = db.result(name, "best")
    dv = db.result(name, "dv_best")
    owned = [lk["from_id"] for lk in db.links(to_id=f"block:{name}", rel="owned_by")]
    cites = [lk["to_id"] for lk in db.links(from_id=f"block:{name}", rel="cites")]
    edges = []
    try:
        for c in (db.contracts() or {}).get("contracts") or []:
            if name in (c.get("producer_block"), c.get("consumer_block")):
                edges.append(c.get("edge_id"))
    except Exception:  # noqa: BLE001
        pass
    vip_dir = pr / ".coresmith" / "vip"
    vips = sorted(str(p.relative_to(pr)) for p in vip_dir.glob("*.py") if any(e and e in p.name for e in edges)) if vip_dir.exists() else []
    return {
        "block": name, "tier": spec.get("tier"), "cluster": spec.get("cluster") or spec.get("subsystem") or "",
        "primitive": bool(spec.get("primitive") or spec.get("kind") == "primitive"),
        "rtl_path": rtl, "rtl_exists": Path(rtl).exists(), "tb_path": tb, "tb_exists": Path(tb).exists(),
        "uarch_spec": f"arch/uarch_specs/{name}.md", "contract_slice": f".coresmith/blocks/{name}/contract_slice.json",
        "edges": edges, "vips": vips, "owned_items": owned, "cites": cites,
        "done": bool(best and best.get("done")), "best": best, "dv_best": dv,
        "attempts": len(db.attempts(name)) if hasattr(db, "attempts") else None,
        "coverage": db.result(name, "coverage") if hasattr(db, "result") else None,
    }


def block_done(db, pr, name: str, *, target_clock_mhz: float = 50.0, seed: int | None = None,
               attempt: int | None = None, actor: str = "cluster") -> dict:
    """Run the whole gate on the on-disk RTL/TB and publish ``best`` on a pass."""
    pr = Path(pr)
    spec = _spec(pr, name)
    rtl, tb = _paths(pr, spec)
    out: dict = {"block": name, "ok": False, "stages": {}, "tool_error": False, "rtl_path": rtl}
    if not Path(rtl).exists():
        out["stages"]["rtl"] = {"ok": False, "reason": f"no RTL at {rtl}"}
        return out
    rtl_sha = hashlib.sha256(Path(rtl).read_bytes()).hexdigest()[:16]
    out["rtl_sha"] = rtl_sha
    attempt = attempt or (len(db.attempts(name)) + 1 if hasattr(db, "attempts") else 1)
    t0 = time.time()

    # 1. contract conformance (report-only stage; here it is the gate)
    try:
        from orchestrator.langgraph.contract_conformance import run_conformance_stage
        conf = run_conformance_stage(str(pr), name, rtl, tb_path=tb if Path(tb).exists() else "")
        out["stages"]["conformance"] = {"ok": bool(conf.get("ok", True)), "ran": bool(conf.get("ran")),
                                        "missing": (conf.get("after_missing") or conf.get("before_missing") or [])[:20],
                                        "deviations": (conf.get("deviations") or [])[:10]}
        if conf.get("ran") and not conf.get("ok", True):
            out["reason"] = "RTL does not conform to the interface contract"
            return out
    except Exception as exc:  # noqa: BLE001
        out["stages"]["conformance"] = {"ok": None, "tool_error": str(exc)[:300]}
        out["tool_error"] = True

    # 2. lint + simulation (VIPs, assertions) with coverage
    from orchestrator.harness import verify as V
    # the TB records measurements of bounded items here (harness.measure);
    # a stale file from an earlier simulation must not decide this one
    mfile = measurements_file(pr, name)
    try:
        mfile.unlink()
    except OSError:
        pass
    from orchestrator.harness.sim_evidence import capture, input_hashes
    evidence_inputs = input_hashes([rtl, tb])
    (pr / "sim_build" / name / "results.xml").unlink(missing_ok=True)
    prev_m = os.environ.get("CORESMITH_MEASUREMENTS")
    os.environ["CORESMITH_MEASUREMENTS"] = str(mfile)
    try:
        try:   # the gate's own DV run is a dv_results row too (the scorecard reads the latest one)
            from orchestrator.state_store.store import Scoreboard
            _sb = Scoreboard(pr)
        except Exception:  # noqa: BLE001
            _sb = None
        dv = V.verify_rtl(pr, spec, attempt=attempt, seed=seed, coverage=True, record_source=actor,
                          scoreboard=_sb)
    finally:
        if prev_m is None:
            os.environ.pop("CORESMITH_MEASUREMENTS", None)
        else:
            os.environ["CORESMITH_MEASUREMENTS"] = prev_m
    out["stages"]["dv"] = {"ok": dv.passed, "verdict": dv.verdict, "infra": dv.infra_error, "skipped": dv.skipped,
                           "log": dv.log_path, "details": {k: v for k, v in (dv.details or {}).items() if k in ("stage", "seed", "coverage", "tests", "failed")}}
    if dv.infra_error:
        out["tool_error"] = True
        out["reason"] = f"DV tool error: {dv.verdict}"
        return out
    if not dv.passed:
        out["reason"] = f"DV failed: {dv.verdict}"
        return out
    try:
        evidence = capture(pr / "sim_build" / name / "results.xml", evidence_inputs)
        if not evidence["passed"]:
            raise ValueError("Simulation XML contains failed or skipped tests")
    except (OSError, ValueError) as exc:
        out.update(tool_error=True, reason=f"DV evidence invalid: {exc}")
        return out
    dv_rec = {"sim_passed": True, "seed": seed, "attempt": attempt, "rtl_sha": rtl_sha, "ts": time.time(),
              "source": actor, "verdict": dv.verdict, "simulation_evidence": evidence}
    try:
        db.set_result(name, "dv_best", dv_rec)
    except Exception:  # noqa: BLE001
        pass

    # 3. full synthesis + 4. timing
    if spec.get("golden_exempt") is None:
        pass
    try:
        from orchestrator.langgraph.pipeline_graph import (
            _evaluate_ppa_gate,
            _timing_ok_from_ppa_meta,
        )
        from orchestrator.langgraph.pipeline_helpers import synthesize_block
    except Exception as exc:  # noqa: BLE001
        out["stages"]["synth"] = {"ok": None, "tool_error": str(exc)[:300]}
        out["tool_error"] = True
        out["reason"] = "synthesis tooling unavailable"
        return out
    res = synthesize_block(spec, rtl, target_clock_mhz, attempt)
    synth_ok = bool(res.get("success"))
    out["stages"]["synth"] = {"ok": synth_ok, "gate_count": res.get("gate_count"), "ff": res.get("ff_count"),
                              "area_um2": res.get("chip_area_um2"), "log": res.get("log_path", "")}
    if not synth_ok:
        out["tool_error"] = bool(res.get("infra_error") or res.get("tool_error"))
        out["reason"] = "synthesis failed" + (" (tool error)" if out["tool_error"] else "")
        return out
    try:
        ppa_ok, violations, meta = _evaluate_ppa_gate(str(pr), name, rtl, res, require_gate_flag=False)
    except Exception as exc:  # noqa: BLE001
        out["stages"]["timing"] = {"ok": None, "tool_error": str(exc)[:300]}
        out["tool_error"] = True
        out["reason"] = "timing tooling failed"
        return out
    timing_ok = _timing_ok_from_ppa_meta(meta)
    wns, tns = meta.get("wns_ns"), meta.get("tns_ns")
    # A block is published on a MEASURED timing pass only: no WNS (the STA
    # crashed on an SRAM black box, or was never run) is "not measured", a
    # negative WNS or TNS is a fail -- never a pass (blocks used to be published
    # at WNS -68 ns and TNS -54 us because the verdict was None; the workers
    # refused to accept it, the tool did not).
    measured = wns is not None
    try:
        neg = (wns is not None and float(wns) < 0) or (tns is not None and float(tns) < 0)
    except (TypeError, ValueError):
        neg = False
    if neg:
        timing_ok = False
    elif not measured:
        timing_ok = None
    out["stages"]["timing"] = {"ok": timing_ok, "measured": measured, "wns_ns": wns, "tns_ns": tns,
                               "sta_report": meta.get("sta_report_path", ""), "ppa_ok": ppa_ok,
                               "violations": [str(v)[:160] for v in (violations or [])[:8]]}
    if timing_ok is False:
        out["reason"] = f"timing violated (WNS {wns} ns, TNS {tns} ns)"
        return out
    if not measured and os.environ.get("CORESMITH_BLOCK_DONE_ALLOW_UNMEASURED_TIMING", "").strip().lower() not in ("1", "true", "yes"):
        out["tool_error"] = True
        out["reason"] = ("timing not measured (no WNS: the STA did not run or crashed, e.g. on an SRAM black box) -- "
                         "characterise the macro or provide its liberty; CORESMITH_BLOCK_DONE_ALLOW_UNMEASURED_TIMING=1 waives")
        return out
    if ppa_ok is False:
        out["reason"] = "PPA budget violated: " + "; ".join(str(v)[:100] for v in (violations or [])[:3])
        return out

    # Requirement evidence is part of publication, not bookkeeping after it.
    # A passing simulation cannot blanket-pass named verifiers, and a stamping
    # failure must not leave a published ``best`` record behind.
    try:
        item_result = stamp_item_checks(
            db, pr, name, tb, rtl_sha=rtl_sha, actor=actor)
        out.update(item_result)
    except Exception as exc:  # noqa: BLE001
        out["item_checks_error"] = str(exc)[:300]
        out["reason"] = "requirement evidence stamping failed"
        out["tool_error"] = True
        return out
    from orchestrator.state_store.ontology import item_must_have
    required = {
        iid for iid in _owned_items(db, name)
        if (db.item(iid) and item_must_have(db.item(iid))
            and db.item(iid).get("status") not in ("waived", "retired"))
    }
    later_scope = {
        iid for iid in required
        if (db.verifiers(item_id=iid)
            and all(_later_scope_verifier(v) for v in db.verifiers(item_id=iid)))
    }
    evidence_failures = required & set(
        item_result.get("failed_items", [])
        + item_result.get("unverified_items", [])
        + item_result.get("unmeasured_items", []))
    evidence_failures |= ((required - later_scope)
                          & set(item_result.get("deferred_items", [])))
    if evidence_failures:
        out["reason"] = ("required item evidence failed or is missing: "
                         + ", ".join(sorted(evidence_failures)))
        return out

    # publish
    best = {**dv_rec, "synth_success": True, "timing_ok": timing_ok, "timing_required": bool(meta.get("timing_required")),
            "gate_count": res.get("gate_count"), "ff_count": res.get("ff_count"), "wns_ns": meta.get("wns_ns"),
            "attempt": attempt, "done": True, "published_by": actor}
    db.set_result(name, "best", best)
    try:
        from orchestrator.langgraph.event_stream import write_graph_event
        write_graph_event(str(pr), "Block Done", "block_published", {"block": name, "rtl_sha": rtl_sha, "actor": actor,
                                                                    "wns_ns": meta.get("wns_ns"), "gate_count": res.get("gate_count")})
    except Exception:  # noqa: BLE001
        pass
    out.update({"ok": True, "best": best, "elapsed_s": round(time.time() - t0, 1)})
    return out
