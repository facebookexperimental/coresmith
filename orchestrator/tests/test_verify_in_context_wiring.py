# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The `verify_in_context` skill must be wired into the code-gen agents (rung1).

Item 11 shipped prompts/skills/verify_in_context.md but no agent's load_skills
pulled it in, so the CLI verify surface was never advertised to the models that
generate RTL / testbenches / models. It must be present in the rtl_generator,
testbench_generator, and microarch build-model system prompts, UNCONDITIONALLY
(present whether or not CORESMITH_PROMPT_SLIM is set)."""
from __future__ import annotations

import importlib

import pytest

from orchestrator.langchain.prompts.skills import load_skill

# Standalone anchors from verify_in_context.md: the CLI invocation surface + the
# fresh-seed gate discipline. A structural rewrite that dropped either would be
# caught. Defect 1 (rung1-fixes-2): the skill now instructs invoking the CLI via
# $CORESMITH_CLI (PATH is unreliable in codex agent shells), so the anchor is the
# env-var handle + the `verify` verb rather than a bare `coresmith verify`.
_ANCHOR_CLI = "CORESMITH_CLI"
_ANCHOR_VERB = "verify rtl"
_ANCHOR_SEED = "FRESH seed"


def _has_skill(text: str) -> bool:
    return _ANCHOR_CLI in text and _ANCHOR_VERB in text and _ANCHOR_SEED in text


def test_skill_file_is_standalone():
    body = load_skill("verify_in_context")
    assert body, "verify_in_context.md must exist"
    assert _has_skill(body)
    # It should stand alone: verify commands + state queries + fresh-seed note.
    assert "contracts" in body
    assert "dv-status" in body
    # Documents the fallback-to-plain-coresmith when the var is unset.
    assert "CORESMITH_CLI:-coresmith" in body


@pytest.mark.parametrize("slim", ["0", "1"])
class TestSkillWiredBothBranches:
    def test_rtl_generator_prompt(self, monkeypatch, slim):
        monkeypatch.setenv("CORESMITH_PROMPT_SLIM", slim)
        import orchestrator.langchain.agents.rtl_generator as rg
        importlib.reload(rg)
        assert _has_skill(rg.SYSTEM_PROMPT)

    def test_testbench_generator_prompt(self, monkeypatch, slim):
        monkeypatch.setenv("CORESMITH_PROMPT_SLIM", slim)
        import orchestrator.langchain.agents.testbench_generator as tg
        importlib.reload(tg)
        assert _has_skill(tg.SYSTEM_PROMPT)

    def test_microarch_build_models_skills(self, monkeypatch, slim):
        monkeypatch.setenv("CORESMITH_PROMPT_SLIM", slim)
        # _load_shared_skills() reads from disk fresh each call and feeds the
        # BUILD_MODELS_SYSTEM prompt (run_build_models_node).
        import orchestrator.langgraph.microarch_exp as mx
        assert _has_skill(mx._load_shared_skills())
