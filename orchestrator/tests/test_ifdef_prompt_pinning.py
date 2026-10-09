# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Workers are told the functional equivalence enforced by the ifdef gate."""
from __future__ import annotations

from pathlib import Path

_PROMPTS = Path(__file__).resolve().parent.parent / "langchain" / "prompts"


def _norm(t: str) -> str:
    return " ".join(t.lower().split())


def test_rtl_generator_md_pins_one_implementation_rule():
    txt = _norm((_PROMPTS / "rtl_generator.md").read_text())
    assert "simulation and synthesis must use the same functional logic" in txt
    assert "simulation-only assertions and tracing may be guarded" in txt


def test_pipeline_contract_skill_pins_rule():
    txt = _norm((_PROMPTS / "skills" / "pipeline_contract.md").read_text())
    assert "one implementation per module" in txt
    assert "split-brain" in txt
    assert "ifndef synthesis" in txt
