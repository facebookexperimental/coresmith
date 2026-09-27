# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""``coresmith block-status <b>`` / ``coresmith block-done <b>`` (architect
sitting, step 4): the block gate as a tool.

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
import json
import os
import time
from pathlib import Path


def _spec(pr, name: str) -> dict:
    from orchestrator.harness import blocks as B
    return B.load_block_spec(pr, name) or {"name": name}


def _paths(pr, spec: dict) -> tuple[str, str]:
    from orchestrator.harness import verify as V
    return V._resolve_rtl_path(Path(pr), spec), V._resolve_tb_path(Path(pr), spec, None)


def block_status(db, pr, name: str) -> dict:
    pr = Path(pr)
    spec = _spec(pr, name)
    rtl, tb = _paths(pr, spec)
    best = db.result(name, "best")
    dv = db.result(name, "dv_best")
    owned = [l["from_id"] for l in db.links(to_id=f"block:{name}", rel="owned_by")]
    cites = [l["to_id"] for l in db.links(from_id=f"block:{name}", rel="cites")]
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
    dv = V.verify_rtl(pr, spec, attempt=attempt, seed=seed, coverage=True, record_source=actor)
    out["stages"]["dv"] = {"ok": dv.passed, "verdict": dv.verdict, "infra": dv.infra_error, "skipped": dv.skipped,
                           "log": dv.log_path, "details": {k: v for k, v in (dv.details or {}).items() if k in ("stage", "seed", "coverage", "tests", "failed")}}
    if dv.infra_error:
        out["tool_error"] = True
        out["reason"] = f"DV tool error: {dv.verdict}"
        return out
    if not dv.passed:
        out["reason"] = f"DV failed: {dv.verdict}"
        return out
    dv_rec = {"sim_passed": True, "seed": seed, "attempt": attempt, "rtl_sha": rtl_sha, "ts": time.time(),
              "source": actor, "verdict": dv.verdict}
    try:
        db.set_result(name, "dv_best", dv_rec)
    except Exception:  # noqa: BLE001
        pass

    # 3. full synthesis + 4. timing
    if spec.get("golden_exempt") is None:
        pass
    try:
        from orchestrator.langgraph.pipeline_graph import _evaluate_ppa_gate, _timing_ok_from_ppa_meta
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
    # negative WNS or TNS is a fail -- never a pass (coresmith3: rv_l1i0/1 were
    # published at WNS -68 ns and gpu_mem at TNS -54 us because the verdict was
    # None; the workers refused to accept it, the tool did not).
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
    if not measured and not os.environ.get("CORESMITH_BLOCK_DONE_ALLOW_UNMEASURED_TIMING", "").strip().lower() in ("1", "true", "yes"):
        out["tool_error"] = True
        out["reason"] = ("timing not measured (no WNS: the STA did not run or crashed, e.g. on an SRAM black box) -- "
                         "characterise the macro or provide its liberty; CORESMITH_BLOCK_DONE_ALLOW_UNMEASURED_TIMING=1 waives")
        return out
    if ppa_ok is False:
        out["reason"] = "PPA budget violated: " + "; ".join(str(v)[:100] for v in (violations or [])[:3])
        return out

    # publish
    best = {**dv_rec, "synth_success": True, "timing_ok": timing_ok, "timing_required": bool(meta.get("timing_required")),
            "gate_count": res.get("gate_count"), "ff_count": res.get("ff_count"), "wns_ns": meta.get("wns_ns"),
            "attempt": attempt, "done": True, "published_by": actor}
    db.set_result(name, "best", best)
    for iid in [l["from_id"] for l in db.links(to_id=f"block:{name}", rel="owned_by")]:
        try:
            db.add_check(iid, "block_dv", "pass", evidence=f"{name} passed conformance+DV+synth+timing (WNS {meta.get('wns_ns')})",
                         sha=rtl_sha, actor=actor)
        except Exception:  # noqa: BLE001
            pass
    try:
        from orchestrator.langgraph.event_stream import write_graph_event
        write_graph_event(str(pr), "Block Done", "block_published", {"block": name, "rtl_sha": rtl_sha, "actor": actor,
                                                                    "wns_ns": meta.get("wns_ns"), "gate_count": res.get("gate_count")})
    except Exception:  # noqa: BLE001
        pass
    out.update({"ok": True, "best": best, "elapsed_s": round(time.time() - t0, 1)})
    return out
