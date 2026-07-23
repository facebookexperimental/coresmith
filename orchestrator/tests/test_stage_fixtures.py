# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Stage-fixture tests (Package C, C4/C5).

Covers the schema fingerprint, the committed post-arch fixture, the checkpoint
snapshot->resume round-trip (built synthetically so it needs no live LLM run),
and the staleness policy (skip vs strict-fail on fingerprint drift).
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

from orchestrator.testing import stage_fixtures as sf

pytestmark = pytest.mark.stage_fixture

_REPO = Path(__file__).resolve().parents[2]
_STAGE_FIXTURES = Path(__file__).parent / "fixtures" / "stage"


def _load_snapshot_module():
    path = _REPO / "scripts" / "snapshot_stage.py"
    spec = importlib.util.spec_from_file_location("snapshot_stage", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# schema_fingerprint
# ---------------------------------------------------------------------------
class TestSchemaFingerprint:
    def test_stable_and_hex(self):
        a = sf.schema_fingerprint("block")
        b = sf.schema_fingerprint("block")
        assert a == b and len(a) == 16
        int(a, 16)  # valid hex

    def test_differs_across_graphs(self):
        assert sf.schema_fingerprint("block") != sf.schema_fingerprint("pipeline")

    def test_unknown_graph_raises(self):
        with pytest.raises(ValueError):
            sf.schema_fingerprint("nope")


# ---------------------------------------------------------------------------
# Committed post-arch fixture (no checkpoint -- pipeline starts from block_specs)
# ---------------------------------------------------------------------------
class TestPostArchFixture:
    @pytest.mark.asyncio
    async def test_materialize_copies_artifacts_and_sets_env(self, from_stage, tmp_path):
        root = tmp_path / "pa_root"
        ctx = await from_stage("adder8_post_arch", project_root=root)
        # Artifacts landed under the new root.
        assert (root / ".coresmith" / "block_specs.json").exists()
        assert (root / "arch" / "frd_spec.md").exists()
        # No checkpoint -> no resumable graph.
        assert ctx.graph is None
        assert ctx.manifest["stage"] == "post-arch"
        # Env from the manifest is applied.
        import os
        assert os.environ["CORESMITH_PROJECT_ROOT"] == str(root)
        assert os.environ["CORESMITH_PROFILE"] == "legacy"

    @pytest.mark.asyncio
    async def test_block_queue_loads_from_materialized_specs(self, from_stage, tmp_path):
        root = tmp_path / "pa_root2"
        await from_stage("adder8_post_arch", project_root=root)
        from orchestrator.harness.blocks import load_block_queue

        queue = load_block_queue(str(root))
        assert [b["name"] for b in queue] == ["adder8"]
        assert queue[0]["rtl_target"] == "rtl/adder8/adder8.v"


# ---------------------------------------------------------------------------
# Synthetic checkpoint snapshot -> resume (post-uarch, no live LLM)
# ---------------------------------------------------------------------------
_BLOCK = {
    "name": "adder8", "rtl_target": "rtl/adder8.v", "tier": 1,
    "python_source": "golden/adder8.py", "testbench": "tb/test_adder8.py",
    "description": "8-bit adder",
}


def _block_state(project_root: str) -> dict:
    return {
        "project_root": project_root, "target_clock_mhz": 50.0, "max_attempts": 3,
        "pipeline_run_start": 0.0, "current_block": dict(_BLOCK), "attempt": 1,
        "phase": "init", "uarch_approved": False, "lint_clean": False,
        "sim_passed": False, "synth_success": False, "synth_gate_count": 0,
        "rtl_path": "", "tb_path": "", "debug_action": "", "step_log_paths": {},
        "preserve_testbench": False, "force_regen_tb": False, "human_response": None,
        "completed_blocks": [],
    }


async def _drive_to_post_uarch(record_root: Path, thread_id: str) -> None:
    """Drive a fresh block subgraph to a post-uarch node-boundary checkpoint."""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    from orchestrator.langgraph.pipeline_graph import build_block_subgraph

    (record_root / ".coresmith").mkdir(parents=True, exist_ok=True)
    db = record_root / ".coresmith" / "pipeline_checkpoint.db"
    cm = AsyncSqliteSaver.from_conn_string(str(db))
    saver = await cm.__aenter__()
    try:
        # interrupt_after freezes the run right after the uarch spec is written,
        # i.e. a deterministic "post-uarch" checkpoint.
        graph = build_block_subgraph(two_pass=False).compile(
            checkpointer=saver, interrupt_after=["generate_uarch_spec"])
        cfg = {"configurable": {"thread_id": thread_id}}
        await asyncio.wait_for(
            graph.ainvoke(_block_state(str(record_root)), cfg), timeout=30)
        snap = await graph.aget_state(cfg)
        assert snap.next == ("review_uarch_spec",), snap.next  # parked post-uarch
    finally:
        await cm.__aexit__(None, None, None)


@pytest.fixture
def synth_env(monkeypatch):
    """Fault provider (no faults) + EDA stubs + sub-second watchdog."""
    monkeypatch.setenv("CORESMITH_PROFILE", "legacy")
    monkeypatch.setenv("CORESMITH_LLM_PROVIDER", "fault")
    monkeypatch.setenv("CORESMITH_TIMEOUT_MULTIPLIER", "0.01")
    monkeypatch.setenv("CORESMITH_RTL_MODEL_EQUIV", "0")
    monkeypatch.delenv("CORESMITH_LLM_LOG_ROOT", raising=False)
    from orchestrator.testing import fault_provider as fp
    from orchestrator.testing.eda_stubs import stub_eda
    from orchestrator.testing.faults import FaultSchedule
    stub_eda(monkeypatch)
    b = fp.get_backend()
    b.reset()
    b.set_schedule(FaultSchedule([]))  # no faults -> canned success artifacts
    yield b
    b.reset()


async def _make_post_uarch_fixture(tmp_path, monkeypatch, thread_id):
    """Drive to a post-uarch checkpoint + snapshot it -> (fixture_dir, manifest)."""
    record_root = tmp_path / f"record_{thread_id}"
    monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(record_root))
    import orchestrator.langgraph.pipeline_graph as pg
    import orchestrator.langgraph.pipeline_helpers as ph
    monkeypatch.setattr(ph, "PROJECT_ROOT", record_root)
    monkeypatch.setattr(pg, "PROJECT_ROOT", record_root)
    await _drive_to_post_uarch(record_root, thread_id)
    snap_mod = _load_snapshot_module()
    fixture_dir = tmp_path / f"fx_{thread_id}"
    manifest = snap_mod.snapshot(
        project_root=record_root, out=fixture_dir, stage="post-uarch-block",
        graph="block", thread_id=thread_id,
        checkpoint_db=record_root / ".coresmith" / "pipeline_checkpoint.db",
        require_parked=True, original_root=str(record_root),
    )
    return fixture_dir, manifest


class TestSyntheticCheckpointResume:
    @pytest.mark.asyncio
    async def test_snapshot_then_resume_skips_uarch_regen(
            self, synth_env, monkeypatch, tmp_path):
        fault_backend = synth_env
        thread_id = "synthblock"

        fixture_dir, manifest = await _make_post_uarch_fixture(
            tmp_path, monkeypatch, thread_id)
        assert any("Uarch Spec" in e["run_name"] for e in fault_backend.call_log)
        assert manifest["has_checkpoint"] is True
        assert manifest["pending_interrupt"]["next"] == ["review_uarch_spec"]
        assert (fixture_dir / "checkpoint.db").exists()

        # Materialize into a NEW root + resume.
        new_root = tmp_path / "resume"
        new_root.mkdir()
        ctx = await sf.materialize_stage(fixture_dir, str(new_root), monkeypatch)
        try:
            # root rewritten across state
            snap = await ctx.aget_state()
            assert snap.values.get("project_root") == str(new_root)
            assert snap.next == ("review_uarch_spec",)  # still parked post-uarch

            n0 = len(fault_backend.call_log)
            await asyncio.wait_for(ctx.graph.ainvoke(None, ctx.config), timeout=30)
            snap2 = await ctx.aget_state()
            # P1: terminated within budget.
            assert snap2.next == ()
            # The core value: resume did NOT regenerate the uarch spec.
            resumed = [e["run_name"] for e in fault_backend.call_log[n0:]]
            assert resumed, "resume should have made downstream LLM calls"
            assert not any("Uarch Spec" in rn for rn in resumed), resumed
        finally:
            await ctx.aclose()


# ---------------------------------------------------------------------------
# Staleness policy: checkpoint fingerprint drift -> skip (default) / raise (strict)
# ---------------------------------------------------------------------------
class TestStalenessPolicy:
    async def _stale_ckpt_fixture(self, tmp_path, monkeypatch, thread_id):
        """A real checkpoint fixture whose manifest fingerprint is corrupted."""
        fixture_dir, manifest = await _make_post_uarch_fixture(
            tmp_path, monkeypatch, thread_id)
        manifest["schema_fingerprint"] = "deadbeefdeadbeef"  # force drift
        (fixture_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return fixture_dir

    @pytest.mark.asyncio
    async def test_mismatch_skips_by_default(self, synth_env, monkeypatch, tmp_path):
        fx = await self._stale_ckpt_fixture(tmp_path, monkeypatch, "stale_skip")
        monkeypatch.delenv("CORESMITH_STAGE_STRICT", raising=False)
        root = tmp_path / "r"
        root.mkdir()
        with pytest.raises(pytest.skip.Exception):
            await sf.materialize_stage(fx, str(root), monkeypatch)

    @pytest.mark.asyncio
    async def test_mismatch_raises_under_strict(self, synth_env, monkeypatch, tmp_path):
        fx = await self._stale_ckpt_fixture(tmp_path, monkeypatch, "stale_raise")
        monkeypatch.setenv("CORESMITH_STAGE_STRICT", "1")
        root = tmp_path / "r2"
        root.mkdir()
        with pytest.raises(RuntimeError) as ei:
            await sf.materialize_stage(fx, str(root), monkeypatch)
        assert "schema drift" in str(ei.value)
