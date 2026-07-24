# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Prompt-pinning: the "exactly ONE implementation per module" rule (the rung3
split-brain ban) must be present in the RTL-generator system prompt, its inline
fallback, and the pipeline_contract skill -- so the model is told the discipline,
not only rejected by the deterministic gate after the fact.
"""
from __future__ import annotations

from pathlib import Path

_PROMPTS = Path(__file__).resolve().parent.parent / "langchain" / "prompts"


def _norm(t: str) -> str:
    return " ".join(t.lower().split())


def test_rtl_generator_md_pins_one_implementation_rule():
    txt = _norm((_PROMPTS / "rtl_generator.md").read_text())
    assert "exactly one implementation per module" in txt
    assert "ifdef" in txt or "ifndef" in txt
    # names the allowed carve-out (debug/trace/assertion) and the wrapper split
    assert "$dumpvars" in txt or "$display" in txt or "assert" in txt
    assert "cs_" in txt  # the cs_* wrapper library is the only legit split


def test_rtl_generator_inline_fallback_pins_rule():
    # The SYSTEM_PROMPT inline fallback (used if the .md is missing) must carry it too.
    from orchestrator.langchain.agents import rtl_generator as rg
    src = Path(rg.__file__).read_text()
    # the inline literal block, not the file it loads
    assert "one implementation per module" in _norm(src)
    assert "ifndef synthesis" in _norm(src)


def test_pipeline_contract_skill_pins_rule():
    txt = _norm((_PROMPTS / "skills" / "pipeline_contract.md").read_text())
    assert "one implementation per module" in txt
    assert "split-brain" in txt
    assert "ifndef synthesis" in txt
