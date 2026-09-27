"""Architect sitting step 4: the block gate as a tool and cluster fan-out."""
import asyncio
import json
import stat
from pathlib import Path

import pytest

from orchestrator.harness.tools import block as bt
from orchestrator.langgraph import pipeline_graph as pg
from orchestrator.state_store.project_db import open_project


class _R:
    def __init__(self, passed, verdict="", infra=False, details=None):
        self.passed, self.verdict, self.infra_error, self.skipped, self.details, self.log_path = passed, verdict, infra, False, details or {}, ""


def _project(tmp_path, blocks=("alu", "ctl")):
    db = open_project(tmp_path)
    db.import_block_diagram({"blocks": [{"name": b, "tier": 1, "subsystem": "cpu"} for b in blocks] + [{"name": "dma", "tier": 1, "subsystem": "mem"}],
                             "connections": []})
    (tmp_path / "rtl").mkdir(exist_ok=True)
    for b in blocks + ("dma",):
        (tmp_path / "rtl" / f"{b}.v").write_text(f"module {b}(input clk); endmodule\n")
    db.link_items("PERF-001", "block:alu", "owned_by")
    return db


def test_block_specs_carry_subsystem_and_cluster(tmp_path):
    db = open_project(tmp_path)
    db.import_block_diagram({"blocks": [{"name": "alu", "tier": 1, "subsystem": "cpu", "cluster": "core0", "instances": 2, "owns": ["PERF-001"]},
                                        {"name": "dma", "tier": 1}], "connections": []})
    specs = {b["name"]: b for b in db.block_specs()}
    assert specs["alu"]["subsystem"] == "cpu" and specs["alu"]["cluster"] == "core0" and specs["alu"]["owns"] == ["PERF-001"]
    assert "subsystem" not in specs["dma"] and pg._cluster_of(specs["dma"]) == "all" and pg._cluster_of(specs["alu"]) == "core0"


def test_block_done_publishes_best_only_on_a_full_pass(tmp_path, monkeypatch):
    db = _project(tmp_path)
    import orchestrator.harness.verify as V
    import orchestrator.langgraph.contract_conformance as cc
    import orchestrator.langgraph.pipeline_helpers as ph
    monkeypatch.setattr(cc, "run_conformance_stage", lambda pr, n, rtl, **k: {"ran": True, "ok": True})
    monkeypatch.setattr(V, "verify_rtl", lambda pr, spec, **k: _R(True, "sim OK"))
    monkeypatch.setattr(ph, "synthesize_block", lambda spec, rtl, clk, att: {"success": True, "gate_count": 1234, "ff_count": 100})
    monkeypatch.setattr(pg, "_evaluate_ppa_gate", lambda pr, n, rtl, res, require_gate_flag=True: (True, [], {"wns_ns": 1.5, "timing_required": True}))
    res = bt.block_done(db, tmp_path, "alu", target_clock_mhz=64.0)
    assert res["ok"] and res["best"]["done"] and res["best"]["wns_ns"] == 1.5
    assert db.result("alu", "best")["gate_count"] == 1234 and db.result("alu", "dv_best")["sim_passed"]
    assert db.checks("PERF-001", kind="block_dv")[0]["status"] == "pass"
    st = bt.block_status(db, tmp_path, "alu")
    assert st["done"] and st["owned_items"] == ["PERF-001"] and st["cluster"] == "cpu"
    # timing miss: no best
    monkeypatch.setattr(pg, "_evaluate_ppa_gate", lambda pr, n, rtl, res, require_gate_flag=True: (False, ["slack"], {"wns_ns": -2.0, "timing_verdict_failed": True}))
    res = bt.block_done(db, tmp_path, "ctl")
    assert not res["ok"] and "timing" in res["reason"] and db.result("ctl", "best") is None
    assert db.result("ctl", "dv_best")["sim_passed"]
    # DV tool error is typed, never a design failure
    monkeypatch.setattr(V, "verify_rtl", lambda pr, spec, **k: _R(False, "verilator missing", infra=True))
    res = bt.block_done(db, tmp_path, "ctl")
    assert not res["ok"] and res["tool_error"] and "tool error" in res["reason"]
    # conformance failure stops before DV
    monkeypatch.setattr(cc, "run_conformance_stage", lambda pr, n, rtl, **k: {"ran": True, "ok": False, "after_missing": ["s_q_valid"]})
    res = bt.block_done(db, tmp_path, "ctl")
    assert not res["ok"] and res["stages"]["conformance"]["missing"] == ["s_q_valid"] and "dv" not in res["stages"]
    # missing RTL
    assert "no RTL" in bt.block_done(db, tmp_path, "ghost")["stages"]["rtl"]["reason"]


def _state(tmp_path):
    return {"project_root": str(tmp_path), "target_clock_mhz": 64.0, "max_attempts": 3, "tier_list": [0, 1], "current_tier_index": 1,
            "block_queue": [{"name": "fab", "tier": 0, "kind": "primitive", "primitive": "cs_fabric"},
                            {"name": "alu", "tier": 1, "subsystem": "cpu"}, {"name": "ctl", "tier": 1, "cluster": "cpu"},
                            {"name": "dma", "tier": 1, "subsystem": "mem"}, {"name": "misc", "tier": 1}]}


def test_fan_out_groups_blocks_by_cluster_and_keeps_primitives_on_the_block_path(tmp_path, monkeypatch):
    monkeypatch.setenv("CORESMITH_FANOUT", "cluster")
    monkeypatch.setenv("CORESMITH_UARCH_PHASE", "0")
    monkeypatch.setattr(pg, "_stale_specs", lambda pr, names: [])
    sends = pg.fan_out_tier(_state(tmp_path))
    by_node = {}
    for s in sends:
        by_node.setdefault(s.node, []).append(s.arg)
    assert sorted(by_node) == ["process_cluster"]
    clusters = {a["cluster"]: [b["name"] for b in a["cluster_blocks"]] for a in by_node["process_cluster"]}
    assert clusters == {"all": ["misc"], "cpu": ["alu", "ctl"], "mem": ["dma"]}
    st = dict(_state(tmp_path), current_tier_index=0)
    sends = pg.fan_out_tier(st)
    assert [s.node for s in sends] == ["process_block"] and sends[0].arg["current_block"]["name"] == "fab"
    monkeypatch.setenv("CORESMITH_FANOUT", "block")
    assert all(s.node == "process_block" for s in pg.fan_out_tier(_state(tmp_path)))


def test_process_cluster_node_reports_published_blocks(tmp_path, monkeypatch):
    db = _project(tmp_path)
    db.set_result("alu", "best", {"done": True, "attempt": 2, "gate_count": 10})

    class _FakeSession:
        def __init__(self, pr, cluster, blocks, **k):
            _FakeSession.seen = (cluster, list(blocks), k.get("target_clock_mhz"))

        def run(self):
            return {"state": "blocked", "sittings": 3, "cost_usd": 1.0}
    import orchestrator.architect.cluster as cl
    monkeypatch.setattr(cl, "ClusterSession", _FakeSession)
    out = asyncio.run(pg.process_cluster_node({"project_root": str(tmp_path), "cluster": "cpu", "target_clock_mhz": 64.0,
                                               "cluster_blocks": [{"name": "alu"}, {"name": "ctl"}]}))
    done = {c["name"]: c["success"] for c in out["completed_blocks"]}
    assert done == {"alu": True, "ctl": False} and _FakeSession.seen == ("cpu", ["alu", "ctl"], 64.0)
    assert out["completed_blocks"][0]["attempts"] == 2 and out["completed_blocks"][1]["attempts"] == 3


def test_cluster_session_prompts_and_done(tmp_path, monkeypatch):
    from orchestrator.architect.cluster import ClusterSession, build_cluster_prompt
    _project(tmp_path)
    ROOT = Path(__file__).resolve().parents[2]
    s = ClusterSession(tmp_path, "cpu", ["alu", "ctl"], coresmith_bin=str(ROOT / "bin" / "coresmith"))
    st = s.stage_status()
    assert st["pending"] == ["alu", "ctl"] and st["blocked_by"][0]["code"] == "BLOCK_NOT_DONE"
    assert not s.done(st)
    p = s.opening_prompt()
    assert "cluster `cpu`" in p and "**alu**" in p and "block-done" in p
    open_project(tmp_path).set_result("alu", "best", {"done": True})
    open_project(tmp_path).set_result("ctl", "best", {"done": True})
    assert s.done()
    sp = build_cluster_prompt()
    assert "coresmith block-done" in sp and "## rtl_generator.md" in sp and "## testbench_generator.md" in sp
