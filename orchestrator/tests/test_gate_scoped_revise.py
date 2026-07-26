# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Gate-scoped revise + OPERATOR_MODEL_PIN (dv-hardening-7).

armC live findings, 2026-07-05: every µarch-gate revise round re-drew specs,
reviews, and models for all NON-implicated blocks (~5 wasted LLM rounds per
iteration), and the model regen even clobbered an operator hand-patch (the
mode-3 corner clamp) because there was a spec pin but no model pin.

Disk-first signals (no graph state): _last_gate_signature.txt == "a gate
failure iteration is in progress" (cleared on real pass);
blocks/<b>/gate_feedback.txt == "the gate implicated this block".
"""

from __future__ import annotations

from pathlib import Path

import pytest

import orchestrator.langgraph.pipeline_helpers as ph
from orchestrator.langgraph.pipeline_helpers import gate_scoped_reuse_reason


def _arm(root: Path, *, gate_failing: bool, implicated: bool,
         block: str = "blk") -> None:
    cs = root / ".coresmith"
    (cs / "blocks" / block).mkdir(parents=True, exist_ok=True)
    if gate_failing:
        (cs / "_last_gate_signature.txt").write_text("sig")
    if implicated:
        (cs / "blocks" / block / "gate_feedback.txt").write_text("fb")


class TestGateScopedReuseReason:
    def test_scopes_unimplicated_block_during_failing_gate(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CORESMITH_GATE_SCOPED_REVISE", raising=False)
        _arm(tmp_path, gate_failing=True, implicated=False)
        assert gate_scoped_reuse_reason(str(tmp_path), "blk") != ""

    def test_implicated_block_not_scoped(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CORESMITH_GATE_SCOPED_REVISE", raising=False)
        _arm(tmp_path, gate_failing=True, implicated=True)
        assert gate_scoped_reuse_reason(str(tmp_path), "blk") == ""

    def test_no_gate_iteration_no_scope(self, tmp_path, monkeypatch):
        # clean first pass: signature absent -> normal generation everywhere
        monkeypatch.delenv("CORESMITH_GATE_SCOPED_REVISE", raising=False)
        _arm(tmp_path, gate_failing=False, implicated=False)
        assert gate_scoped_reuse_reason(str(tmp_path), "blk") == ""

    def test_broadcast_writes_feedback_for_all_so_nothing_scoped(
        self, tmp_path, monkeypatch
    ):
        # init_tier writes gate_feedback.txt for every tier block on a
        # broadcast -- so every block is "implicated" and none are scoped.
        monkeypatch.delenv("CORESMITH_GATE_SCOPED_REVISE", raising=False)
        for b in ("a", "b"):
            _arm(tmp_path, gate_failing=True, implicated=True, block=b)
        assert gate_scoped_reuse_reason(str(tmp_path), "a") == ""
        assert gate_scoped_reuse_reason(str(tmp_path), "b") == ""

    def test_kill_switch(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_GATE_SCOPED_REVISE", "0")
        _arm(tmp_path, gate_failing=True, implicated=False)
        assert gate_scoped_reuse_reason(str(tmp_path), "blk") == ""


class TestModelRegenSkips:
    """_maybe_generate_block_golden: pin + scope must PREVENT the generator
    call; an implicated block must still regenerate."""

    @pytest.fixture
    def project(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ph, "PROJECT_ROOT", tmp_path)
        monkeypatch.setenv("CORESMITH_BLOCK_GOLDENS", "1")
        monkeypatch.delenv("CORESMITH_IGNORE_SPEC_PINS", raising=False)
        monkeypatch.delenv("CORESMITH_GATE_SCOPED_REVISE", raising=False)
        models = tmp_path / "arch" / "block_models"
        models.mkdir(parents=True)
        (models / "blk.py").write_text("# existing model (hand-patched)\n")
        return tmp_path

    def _install_generator_probe(self, monkeypatch):
        calls = []

        class _FakeGen:
            def __init__(self, *a, **k):
                pass

            async def generate(self, **kw):
                calls.append(kw.get("block_name"))
                return {}

        import orchestrator.langchain.agents.block_golden_generator as g

        monkeypatch.setattr(g, "BlockGoldenGenerator", _FakeGen)
        return calls

    async def test_model_pin_blocks_regen(self, project, monkeypatch):
        calls = self._install_generator_probe(monkeypatch)
        pin = project / ".coresmith" / "blocks" / "blk" / "OPERATOR_MODEL_PIN"
        pin.parent.mkdir(parents=True)
        pin.write_text("hand-patched mode-3 clamp")
        await ph._maybe_generate_block_golden({"name": "blk"})
        assert calls == []
        assert "hand-patched" in (
            project / "arch" / "block_models" / "blk.py"
        ).read_text()

    async def test_scope_blocks_regen_for_unimplicated(self, project, monkeypatch):
        calls = self._install_generator_probe(monkeypatch)
        _arm(project, gate_failing=True, implicated=False)
        await ph._maybe_generate_block_golden({"name": "blk"})
        assert calls == []

    async def test_implicated_block_still_regens(self, project, monkeypatch):
        calls = self._install_generator_probe(monkeypatch)
        _arm(project, gate_failing=True, implicated=True)
        # reference resolution etc. will run; give it a golden to find
        (project / "inputs").mkdir()
        (project / "inputs" / "toy_golden.py").write_text("def run(s):\n    return s\n")
        monkeypatch.setenv("CORESMITH_SOURCE_ROOT",
                           str(project / "inputs" / "toy_golden.py"))
        await ph._maybe_generate_block_golden({"name": "blk"})
        assert calls == ["blk"]

    async def test_spec_pin_path_still_regens_model(self, tmp_path, monkeypatch):
        """armC defect 3: OPERATOR_SPEC_PIN's early return skipped the model
        regen entirely, so guidance pinned INTO the spec never reached the
        model. The pinned-spec path must still call the model generator."""
        import orchestrator.langgraph.pipeline_graph as pg

        monkeypatch.setenv("CORESMITH_BLOCK_GOLDENS", "1")
        monkeypatch.delenv("CORESMITH_IGNORE_SPEC_PINS", raising=False)
        # project scaffolding: pinned spec + pin file
        (tmp_path / "arch" / "uarch_specs").mkdir(parents=True)
        (tmp_path / "arch" / "uarch_specs" / "blk.md").write_text("# pinned spec")
        pin = tmp_path / ".coresmith" / "blocks" / "blk" / "OPERATOR_SPEC_PIN"
        pin.parent.mkdir(parents=True)
        pin.write_text("mandatory clamp")

        regen_calls = []

        async def _probe(block, callbacks=None):
            regen_calls.append(block["name"])

        monkeypatch.setattr(ph, "_maybe_generate_block_golden", _probe)
        monkeypatch.setattr(pg, "write_graph_event", lambda *a, **k: None)
        monkeypatch.setattr(pg, "_callbacks", lambda s: [])

        state = {"current_block": {"name": "blk"}, "project_root": str(tmp_path)}
        result = await pg.generate_uarch_spec_node(state)
        assert result == {"uarch_approved": False, "phase": "uarch"}
        assert regen_calls == ["blk"], (
            "pinned-spec path must regen the model from the pinned spec"
        )

    async def test_pin_ignored_with_escape_env(self, project, monkeypatch):
        calls = self._install_generator_probe(monkeypatch)
        monkeypatch.setenv("CORESMITH_IGNORE_SPEC_PINS", "1")
        pin = project / ".coresmith" / "blocks" / "blk" / "OPERATOR_MODEL_PIN"
        pin.parent.mkdir(parents=True)
        pin.write_text("x")
        (project / "inputs").mkdir()
        (project / "inputs" / "toy_golden.py").write_text("def run(s):\n    return s\n")
        monkeypatch.setenv("CORESMITH_SOURCE_ROOT",
                           str(project / "inputs" / "toy_golden.py"))
        await ph._maybe_generate_block_golden({"name": "blk"})
        assert calls == ["blk"]


class TestGoldenModelWrapperProvisioning:
    """armC pass-2 [dv-hardening-9]: architecture-driven runs have no
    per-block python_source; the wrapper must still be created from the
    block MODEL so the cocotb TB's `from <b>_model import ...` resolves."""

    def test_wrapper_from_block_model_without_python_source(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(ph, "PROJECT_ROOT", tmp_path)
        monkeypatch.setenv("CORESMITH_BLOCK_GOLDENS", "1")
        models = tmp_path / "arch" / "block_models"
        models.mkdir(parents=True)
        (models / "blk.py").write_text("VALUE = 42\n")
        ph.create_golden_model_wrapper("blk", "")
        wrapper = tmp_path / "tb" / "cocotb" / "blk_model.py"
        assert wrapper.exists(), "wrapper must be provisioned from the block model"
        assert 'import_module("arch.block_models.blk")' in wrapper.read_text()

    def test_no_sources_no_wrapper(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ph, "PROJECT_ROOT", tmp_path)
        monkeypatch.setenv("CORESMITH_BLOCK_GOLDENS", "1")
        ph.create_golden_model_wrapper("blk", "")
        assert not (tmp_path / "tb" / "cocotb" / "blk_model.py").exists()

    def test_python_source_path_still_works(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ph, "PROJECT_ROOT", tmp_path)
        monkeypatch.delenv("CORESMITH_BLOCK_GOLDENS", raising=False)
        (tmp_path / "golden").mkdir()
        (tmp_path / "golden" / "blk_ref.py").write_text("VALUE = 1\n")
        ph.create_golden_model_wrapper("blk", "golden/blk_ref.py")
        wrapper = tmp_path / "tb" / "cocotb" / "blk_model.py"
        assert wrapper.exists()
        assert 'import_module("golden.blk_ref")' in wrapper.read_text()
