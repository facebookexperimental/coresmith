# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the RTL-from-hardware-golden rollout (microarchitecture restructure
step 1).

When ``CORESMITH_RTL_FROM_HW_GOLDEN`` is on, ``rtl_reference_source`` points the
RTL generator at the per-block MyHDL hardware golden
(``arch/block_models/<block>.py``) instead of the float reference golden, so the
RTL becomes a lowering of the proven model rather than an independent
re-transcription. Default-off preserves legacy behavior.
"""
from __future__ import annotations

from pathlib import Path

import orchestrator.langgraph.pipeline_helpers as ph


def _write_block_model(root: Path, name: str) -> None:
    d = root / "arch" / "block_models"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.py").write_text(
        f"def {name}(stimulus):\n    return (b'', 0)\n", encoding="utf-8"
    )


def test_default_off_uses_float_golden(tmp_path, monkeypatch):
    monkeypatch.setattr(ph, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("CORESMITH_RTL_FROM_HW_GOLDEN", raising=False)
    _write_block_model(tmp_path, "blk")  # model present, but feature off
    block = {"name": "blk", "python_source": "inputs/golden.py"}

    src, is_hw = ph.rtl_reference_source(block)

    assert src == "inputs/golden.py"
    assert is_hw is False


def test_enabled_uses_hw_golden_when_model_present(tmp_path, monkeypatch):
    monkeypatch.setattr(ph, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("CORESMITH_RTL_FROM_HW_GOLDEN", "1")
    _write_block_model(tmp_path, "blk")
    block = {"name": "blk", "python_source": "inputs/golden.py"}

    src, is_hw = ph.rtl_reference_source(block)

    assert src == str(Path("arch") / "block_models" / "blk.py")
    assert is_hw is True


def test_enabled_falls_back_when_model_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(ph, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("CORESMITH_RTL_FROM_HW_GOLDEN", "1")
    # no block model on disk -> must fall back to the float golden, never crash
    block = {"name": "blk", "python_source": "inputs/golden.py"}

    src, is_hw = ph.rtl_reference_source(block)

    assert src == "inputs/golden.py"
    assert is_hw is False


def test_falsey_env_values_keep_legacy(tmp_path, monkeypatch):
    monkeypatch.setattr(ph, "PROJECT_ROOT", tmp_path)
    _write_block_model(tmp_path, "blk")
    block = {"name": "blk", "python_source": "inputs/golden.py"}
    for val in ("0", "", "false", "off", "no"):
        monkeypatch.setenv("CORESMITH_RTL_FROM_HW_GOLDEN", val)
        src, is_hw = ph.rtl_reference_source(block)
        assert (src, is_hw) == ("inputs/golden.py", False), val


def test_block_hw_golden_rel_resolves_path(tmp_path, monkeypatch):
    monkeypatch.setattr(ph, "PROJECT_ROOT", tmp_path)
    _write_block_model(tmp_path, "luma_tqrecon_lane_array")
    rel = ph.block_hw_golden_rel({"name": "luma_tqrecon_lane_array"})
    assert rel == str(Path("arch") / "block_models" / "luma_tqrecon_lane_array.py")
    # absent block -> ''
    assert ph.block_hw_golden_rel({"name": "does_not_exist"}) == ""
    assert ph.block_hw_golden_rel({}) == ""


def test_rtl_generator_accepts_hw_golden_flag():
    """The agent signature must accept the new kwarg (back-compat default)."""
    import inspect

    from orchestrator.langchain.agents.rtl_generator import RTLGeneratorAgent

    sig = inspect.signature(RTLGeneratorAgent.generate)
    assert "reference_is_hw_golden" in sig.parameters
    assert sig.parameters["reference_is_hw_golden"].default is False
