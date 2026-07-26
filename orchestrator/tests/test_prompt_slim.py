# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Item-11 (B4) prompt slimming -- both env-gate branches."""

from __future__ import annotations

from orchestrator.langchain.agents import rtl_generator as rg
from orchestrator.langchain.agents.contract_lookup import (
    format_block_contracts_prompt,
    format_block_contracts_prompt_slim,
)
from orchestrator.langgraph import microarch_exp as mx


class TestSlimFlag:
    def test_default_on(self, monkeypatch):
        monkeypatch.delenv("CORESMITH_PROMPT_SLIM", raising=False)
        assert rg._prompt_slim_enabled() is True
        assert mx._prompt_slim_enabled() is True

    def test_off(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_PROMPT_SLIM", "0")
        assert rg._prompt_slim_enabled() is False
        assert mx._prompt_slim_enabled() is False


_VIEW = {
    "defaults": {"default_packing_convention": "lsb_first"},
    "edges": [
        {
            "edge_id": "e1", "role": "producer",
            "producer_block": "a", "consumer_block": "b",
            "bootstrap_policy": {"required": True, "policy_type": "reset_seed",
                                 "rationale": "drive seed on cycle 1"},
        },
    ],
}


class TestContractSlim:
    def test_slim_has_pointer_and_bootstrap_no_json_dump(self):
        out = format_block_contracts_prompt_slim("a", _VIEW)
        # Defect 1 (rung1-fixes-2): the CLI pointer now invokes via
        # $CORESMITH_CLI (PATH is unreliable in codex agent shells).
        assert "contracts a" in out
        assert "CORESMITH_CLI" in out
        assert "reset_seed" in out
        assert "```json" not in out  # no full edge dump

    def test_full_has_json_dump(self):
        out = format_block_contracts_prompt("a", _VIEW)
        assert "```json" in out
        assert "reset_seed" in out

    def test_empty_edges_returns_empty(self):
        assert format_block_contracts_prompt_slim("a", {"edges": []}) == ""


class TestMicroarchBuildPromptSlim:
    def _golden(self):
        return "\n".join(f"line{i} = {i}" for i in range(200))

    def test_slim_uses_path_and_head_not_full(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_PROMPT_SLIM", "1")
        golden = self._golden()
        out = mx.build_build_models_prompt(
            str(tmp_path), ["blk"], {"blk": "spec"}, "inputs/golden.py", golden, "",
        )
        assert "inputs/golden.py" in out
        assert "coresmith verify model" in out
        # head only -> line39 present, line150 absent.
        assert "line39" in out
        assert "line150" not in out

    def test_full_inlines_whole_golden(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_PROMPT_SLIM", "0")
        golden = self._golden()
        out = mx.build_build_models_prompt(
            str(tmp_path), ["blk"], {"blk": "spec"}, "inputs/golden.py", golden, "",
        )
        assert "line150" in out  # full source inlined
