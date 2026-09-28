"""Operator-triggered targeted revise of published blocks (`coresmith run revise-blocks`).

The targeted-revise plan ({block: reuse_spec}) used to come only from an
integration-review or DV-failure decision, so an operator with a defect in two
published blocks (e.g. rv_core0/rv_core1's simulation-only CTF monitor) had
"change nothing" (restart-node) or "rebuild everything" (run start --force).
The new verb writes the same plan onto the latest checkpoint as the
integration_review node's output: route_after_integration_review re-enters
init_tier and only the named blocks are redone. No LLM is involved here.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from orchestrator.graph_lifecycle import GraphLifecycle
from orchestrator.langgraph import pipeline_graph as pg

QUEUE = [{"name": "rv_core0", "tier": 3, "cluster": "cpu"},
         {"name": "rv_core1", "tier": 3, "cluster": "cpu"},
         {"name": "rv_fpu0", "tier": 3, "cluster": "cpu"},
         {"name": "uart", "tier": 1, "cluster": "io"}]
RULE = "ACCEPTANCE defect: the CTF monitor opens +ctf with \"w\" in every hart"


def _project(root):
    (root / "arch/uarch_specs").mkdir(parents=True)
    db = pg._db(str(root))
    for b in QUEUE:
        (root / "arch/uarch_specs" / f"{b['name']}.md").write_text(f"# {b['name']}\n")
        db.set_result(b["name"], "best", {"done": True})
        db.set_result(b["name"], "dv_best", {"sim_passed": True})
    for name in ("rv_core0", "rv_core1"):
        db.add_constraint(name, RULE, source="human")
        db.add_constraint(name, "learned: keep ce low in reset", source="lint")
    return db


def _values(root):
    return {"project_root": str(root), "target_clock_mhz": 64.0, "max_attempts": 3,
            "block_queue": QUEUE, "tier_list": [0, 1, 2, 3], "current_tier_index": 4,
            "completed_blocks": [{"name": b["name"], "success": True} for b in QUEUE],
            "integration_review_action": "approve", "pipeline_done": False}


def test_plan_reuses_spec_writes_ledger_feedback_and_drops_only_named_results(tmp_path):
    db = _project(tmp_path)
    update = pg.operator_revise_update(str(tmp_path), _values(tmp_path), ["rv_core0", "rv_core1"])
    assert update["revise_blocks"] == {"rv_core0": True, "rv_core1": True}
    assert update["integration_review_action"] == "revise" and update["current_tier_index"] == 0
    for name in ("rv_core0", "rv_core1"):
        fb = (tmp_path / ".coresmith/blocks" / name / "gate_feedback.txt").read_text()
        assert "OPERATOR REVISION (tier 3; MANDATORY)" in fb and RULE in fb
        assert "learned" not in fb  # only the operator's (human) rules
        assert db.result(name, "best") is None and db.result(name, "dv_best") is None
    for name in ("rv_fpu0", "uart"):
        assert db.result(name, "best")["done"] is True
        assert not (tmp_path / ".coresmith/blocks" / name / "gate_feedback.txt").exists()


def test_explicit_feedback_and_refusals(tmp_path):
    db = _project(tmp_path)
    pg.operator_revise_update(str(tmp_path), _values(tmp_path), ["rv_core0"], "use one $fopen")
    assert "use one $fopen" in (tmp_path / ".coresmith/blocks/rv_core0/gate_feedback.txt").read_text()
    with pytest.raises(ValueError, match="not_a_block"):
        pg.operator_revise_update(str(tmp_path), _values(tmp_path), ["not_a_block"])
    with pytest.raises(ValueError, match="uart"):
        pg.operator_revise_update(str(tmp_path), _values(tmp_path), ["uart"])  # no ledger, no text
    assert db.result("uart", "best")["done"] is True


@pytest.mark.asyncio
async def test_revise_enters_init_tier_from_latest_checkpoint(tmp_path, monkeypatch):
    _project(tmp_path)
    life = GraphLifecycle("pipeline", str(tmp_path / ".coresmith/pipeline_checkpoint.db"),
                          "orchestrator.langgraph.pipeline_graph", "build_pipeline_graph", str(tmp_path))
    try:
        await life.ensure_graph()
        config = {"configurable": {"thread_id": "pipeline"}}
        # A finished run: integration_check failed closed and the graph ended.
        await life.graph.aupdate_state(config, _values(tmp_path), as_node="integration_check")
        life.safe_start = AsyncMock()
        result = await life.restart_with_update(
            lambda v: pg.operator_revise_update(str(tmp_path), v, ["rv_core0", "rv_core1"]),
            "integration_review")
        snap = await life.graph.aget_state(config)
        assert snap.next == ("init_tier",)
        assert snap.values["revise_blocks"] == {"rv_core0": True, "rv_core1": True}
        assert len(snap.values["completed_blocks"]) == len(QUEUE)
        life.safe_start.assert_awaited_once()
        assert life.safe_start.await_args.args[1]["configurable"]["checkpoint_id"] == result["checkpoint_id"]

        # The tier loop redoes only the planned blocks: one cpu cluster worker
        # for rv_core0/rv_core1; rv_fpu0 (same tier and cluster) is not redone.
        monkeypatch.setenv("CORESMITH_FANOUT", "cluster")
        state = {**snap.values, "current_tier_index": 3}
        sends = pg.fan_out_tier(state)
        assert [(s.node, [b["name"] for b in s.arg["cluster_blocks"]]) for s in sends] == [
            ("process_cluster", ["rv_core0", "rv_core1"])]
    finally:
        await life.cleanup()


def test_cluster_worker_sees_pending_feedback(tmp_path):
    from orchestrator.architect.cluster import ClusterSession
    _project(tmp_path)
    pg.operator_revise_update(str(tmp_path), _values(tmp_path), ["rv_core0"])
    sess = ClusterSession.__new__(ClusterSession)
    sess.root, sess.blocks = tmp_path, ["rv_core0", "rv_fpu0"]
    text = "\n".join(sess.feedback_lines({"rv_core0": {"done": False}, "rv_fpu0": {"done": True}}))
    assert "MANDATORY revision feedback for rv_core0" in text and RULE in text
    assert "rv_fpu0" not in text
    assert not sess.feedback_lines({"rv_core0": {"done": True}})
