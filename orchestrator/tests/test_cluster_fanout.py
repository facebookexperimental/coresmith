# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The Architect's step 4: the block gate as a tool and cluster fan-out."""
import asyncio
from pathlib import Path

from orchestrator.harness.tools import block as bt
from orchestrator.langgraph import pipeline_graph as pg
from orchestrator.state_store.project_db import open_project


class _R:
    def __init__(self, passed, verdict="", infra=False, details=None):
        self.passed, self.verdict, self.infra_error, self.skipped, self.details, self.log_path = passed, verdict, infra, False, details or {}, ""


def _simulated_pass(pr, spec, **k):
    """A stand-in for ``verify_rtl`` that leaves what a real passing cocotb run
    leaves: the JUnit ``results.xml`` with executed, passing tests. The block
    gate publishes only from that evidence (hashed with the RTL and the TB)."""
    xml = Path(pr) / "sim_build" / spec["name"] / "results.xml"
    xml.parent.mkdir(parents=True, exist_ok=True)
    xml.write_text('<testsuites><testsuite name="all" tests="1">'
                   f'<testcase name="smoke" classname="test_{spec["name"]}" time="0.1"/></testsuite></testsuites>\n')
    return _R(True, "sim OK")


def _project(tmp_path, blocks=("alu", "ctl")):
    db = open_project(tmp_path)
    db.import_block_diagram({"blocks": [{"name": b, "tier": 1, "subsystem": "cpu"} for b in blocks] + [{"name": "dma", "tier": 1, "subsystem": "mem"}],
                             "connections": []})
    (tmp_path / "rtl").mkdir(exist_ok=True)
    (tmp_path / "tb" / "cocotb").mkdir(parents=True, exist_ok=True)
    for b in blocks + ("dma",):
        (tmp_path / "rtl" / f"{b}.v").write_text(f"module {b}(input clk); endmodule\n")
        # the block gate publishes only with a supplied testbench on disk: its
        # sim evidence hashes the RTL and the TB that were actually simulated
        (tmp_path / "tb" / "cocotb" / f"test_{b}.py").write_text(
            f"import cocotb\n\n\n@cocotb.test()\nasync def smoke(dut):\n    assert dut._name == '{b}'\n")
    # the item alu owns, verified by the TB's `smoke` test: a published pass
    # stamps its block_dv check from results.xml (per-item, never blanket)
    db.upsert_item("frd", {"id": "PERF-001", "kind": "PERF", "text": "alu completes the smoke sequence",
                           "priority": "must_have"})
    db.link_items("PERF-001", "block:alu", "owned_by")
    db.add_verifier("PERF-001", "cocotb", path="tb/cocotb/test_alu.py", entry="smoke", block="alu")
    return db


def test_block_specs_carry_subsystem_and_cluster(tmp_path):
    db = open_project(tmp_path)
    db.import_block_diagram({"blocks": [{"name": "alu", "tier": 1, "subsystem": "cpu", "cluster": "core0", "instances": 2, "owns": ["PERF-001"]},
                                        {"name": "dma", "tier": 1}], "connections": []})
    specs = {b["name"]: b for b in db.block_specs()}
    assert specs["alu"]["subsystem"] == "cpu" and specs["alu"]["cluster"] == "core0" and specs["alu"]["owns"] == ["PERF-001"]
    assert "subsystem" not in specs["dma"] and pg._cluster_of(specs["dma"]) == "tier1" and pg._cluster_of(specs["alu"]) == "core0"


def test_block_done_publishes_best_only_on_a_full_pass(tmp_path, monkeypatch):
    # legacy blanket block_dv stamping (per-item stamping: test_cli_frd.py)
    monkeypatch.setenv("CORESMITH_BLOCK_DV_BLANKET", "1")
    db = _project(tmp_path)
    import orchestrator.harness.verify as V
    import orchestrator.langgraph.contract_conformance as cc
    import orchestrator.langgraph.pipeline_helpers as ph
    monkeypatch.setattr(cc, "run_conformance_stage", lambda pr, n, rtl, **k: {"ran": True, "ok": True})
    monkeypatch.setattr(V, "verify_rtl", _simulated_pass)
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
    # unmeasured timing (STA crashed on an SRAM black box) is a tool error, never a pass
    monkeypatch.setattr(pg, "_evaluate_ppa_gate", lambda pr, n, rtl, res, require_gate_flag=True: (None, [], {"wns_ns": None}))
    res = bt.block_done(db, tmp_path, "ctl")
    assert not res["ok"] and res["tool_error"] and "not measured" in res["reason"] and db.result("ctl", "best") is None
    monkeypatch.setenv("CORESMITH_BLOCK_DONE_ALLOW_UNMEASURED_TIMING", "1")
    assert bt.block_done(db, tmp_path, "ctl")["ok"] and db.result("ctl", "best")["timing_ok"] is None
    monkeypatch.delenv("CORESMITH_BLOCK_DONE_ALLOW_UNMEASURED_TIMING")
    db.clear_result("ctl", "best")
    # negative TNS with a non-negative WNS is still a fail
    monkeypatch.setattr(pg, "_evaluate_ppa_gate", lambda pr, n, rtl, res, require_gate_flag=True: (True, [], {"wns_ns": 0.0, "tns_ns": -54366.0}))
    res = bt.block_done(db, tmp_path, "ctl")
    assert not res["ok"] and "timing violated" in res["reason"]
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
    assert clusters == {"tier1": ["misc"], "cpu": ["alu", "ctl"], "mem": ["dma"]}
    st = dict(_state(tmp_path), current_tier_index=0)
    sends = pg.fan_out_tier(st)
    assert [s.node for s in sends] == ["process_block"] and sends[0].arg["current_block"]["name"] == "fab"
    monkeypatch.setenv("CORESMITH_FANOUT", "block")
    assert all(s.node == "process_block" for s in pg.fan_out_tier(_state(tmp_path)))


def _bare_tier(clusters=(None, None, None, None)):
    blocks = [{"name": "fab", "tier": 0, "kind": "primitive", "primitive": "cs_fabric"}]
    for n, c in zip(("sram", "uart", "gpio", "fft64"), clusters):
        blocks.append({"name": n, "tier": 1, **({"cluster": c} if c else {})})
    return blocks


def _fan(tmp_path, monkeypatch, queue, idx=1):
    monkeypatch.setenv("CORESMITH_FANOUT", "cluster")
    monkeypatch.setenv("CORESMITH_UARCH_PHASE", "0")
    monkeypatch.setattr(pg, "_stale_specs", lambda pr, names: [])
    st = {"project_root": str(tmp_path), "target_clock_mhz": 64.0, "max_attempts": 3, "tier_list": [0, 1],
          "current_tier_index": idx, "block_queue": queue}
    return pg.fan_out_tier(st)


def test_unclustered_blocks_all_dispatch_under_the_tier_default_cluster(tmp_path, monkeypatch):
    logs = []
    monkeypatch.setattr(pg, "log", lambda msg, *a, **k: logs.append(msg))
    sends = _fan(tmp_path, monkeypatch, _bare_tier())
    assert [s.node for s in sends] == ["process_cluster"]
    assert sends[0].arg["cluster"] == "tier1"
    assert sorted(b["name"] for b in sends[0].arg["cluster_blocks"]) == ["fft64", "gpio", "sram", "uart"]
    assert any("assigned the tier default cluster" in m and "sram" in m for m in logs)


def test_mixed_tier_keeps_declared_clusters(tmp_path, monkeypatch):
    sends = _fan(tmp_path, monkeypatch, _bare_tier(("mem", None, None, "accel")))
    got = {s.arg["cluster"]: sorted(b["name"] for b in s.arg["cluster_blocks"]) for s in sends}
    assert got == {"accel": ["fft64"], "mem": ["sram"], "tier1": ["gpio", "uart"]}


def test_default_clusters_never_collide_across_tiers():
    # a design-wide "all" shared .coresmith/clusters/all between tiers
    assert pg._cluster_of({"name": "a", "tier": 1}) != pg._cluster_of({"name": "b", "tier": 2})


def test_an_empty_dispatch_routes_to_the_completion_gate_not_end(tmp_path, monkeypatch):
    # nothing in the current tier (e.g. a stale tier_list): never an empty Send
    # list, which ends the graph silently with blocks pending
    assert _fan(tmp_path, monkeypatch, [{"name": "x", "tier": 5}], idx=1) == ["pipeline_complete"]


def test_init_tier_parks_pipeline_incomplete_for_an_undispatchable_block(tmp_path, monkeypatch):
    monkeypatch.setenv("CORESMITH_FANOUT", "cluster")
    monkeypatch.setattr(pg, "_cluster_of", lambda b: "" if b["name"] == "uart" else "c")
    parked = []

    def _fake_park(payload, **k):
        parked.append(payload)
        return {"action": "retry"}
    monkeypatch.setattr(pg, "_park", _fake_park)
    monkeypatch.setattr(pg, "_record_stage", lambda *a, **k: None)
    monkeypatch.setattr(pg, "_stamp_engine_sha", lambda pr: None)
    st = {"project_root": str(tmp_path), "target_clock_mhz": 64.0, "max_attempts": 3, "tier_list": [0, 1],
          "current_tier_index": 1, "block_queue": _bare_tier(), "completed_blocks": []}
    asyncio.run(pg.init_tier_node(st))
    assert parked and parked[0]["type"] == "pipeline_incomplete" and parked[0]["missing_blocks"] == ["uart"]
    assert "could not assign a cluster" in parked[0]["reason"]
    parked.clear()
    monkeypatch.setenv("CORESMITH_FANOUT", "block")
    asyncio.run(pg.init_tier_node(st))
    assert not parked


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


def test_integrate_tools_wrap_the_graph_functions(tmp_path, monkeypatch):
    from orchestrator.harness.tools import integrate as it
    db = open_project(tmp_path)
    assert it.vip_generate(db, tmp_path)["ok"] is False           # no contracts
    db.import_block_diagram({"blocks": [{"name": "req", "tier": 1}, {"name": "rsp", "tier": 1}], "connections": []})
    db.import_contracts({"contracts": [{"edge_id": "req__m_q__to__rsp__s_q", "producer_block": "req", "producer_port": "m_q",
                                        "consumer_block": "rsp", "consumer_port": "s_q", "handshake_protocol": "req_resp",
                                        "data_width_bits": 8, "timing": {"req_to_rsp_cycles": {"exact": 1}}}]})
    r = it.vip_generate(db, tmp_path)
    assert r["ok"] and r["vips"] == 1 and (tmp_path / ".coresmith" / "vip_index.json").exists()
    assert (tmp_path / ".coresmith" / "blocks" / "req" / "contract_slice.json").exists()
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs" / "task.yaml").write_text("top: tiny\n")
    monkeypatch.setenv("CORESMITH_DETERMINISTIC_TOP", "1")
    s = it.shell_assemble(db, tmp_path)
    assert s["stubs"] == ["req", "rsp"] and s["real"] == [] and "boundary_ports" in s
    ev = it.model_eval(db, tmp_path)
    assert ev["ok"] is False and ev["error"] == "SOC_MODEL_NOT_ASSEMBLED" and ev["required"].endswith("soc_model_top.h")


def test_cluster_fanout_is_an_explicit_opt_in(monkeypatch):
    """The block path is the default; no legacy Architect flag flips it."""
    monkeypatch.delenv("CORESMITH_FANOUT", raising=False)
    assert pg.cluster_fanout_enabled() is False
    monkeypatch.setenv("CORESMITH_ARCHITECT_SITTING", "1")
    assert pg.cluster_fanout_enabled() is False
    monkeypatch.setenv("CORESMITH_FANOUT", "cluster")
    assert pg.cluster_fanout_enabled() is True
    monkeypatch.setenv("CORESMITH_FANOUT", "block")
    assert pg.cluster_fanout_enabled() is False
    assert not hasattr(pg, "architect_sitting_enabled")
