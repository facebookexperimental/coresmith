# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Engine follow-ups #6/#8/#10: refresh_current_sidecars (staleness pileup) +
the restart_from_node graph-fork helper. Pure/near-pure; no LLM/EDA."""
from __future__ import annotations

import hashlib
import json

import pytest


def _rtl_sha1(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


class TestRefreshCurrentSidecars:
    def _block(self, root, name, contract_sha, rtl_text, rtl_rel, passed=True):
        bd = root / ".coresmith" / "blocks" / name
        bd.mkdir(parents=True)
        (bd / "uarch_spec_contract_sha1").write_text("OLD_" + contract_sha)
        rtl = root / rtl_rel
        rtl.parent.mkdir(parents=True, exist_ok=True)
        rtl.write_text(rtl_text)
        (bd / "best_result.json").write_text(json.dumps({
            "sim_passed": passed,
            "rtl_sha1": _rtl_sha1(rtl_text),
            "rtl_target": rtl_rel,
            "contract_sha1": "OLD_" + contract_sha,
        }))

    def test_refreshes_intact_passing_block(self, tmp_path, monkeypatch):
        from orchestrator.langgraph import pipeline_helpers as ph
        monkeypatch.setattr(ph, "block_contract_sha1",
                            lambda pr, n: "LIVE_" + n)
        self._block(tmp_path, "blkA", "x", "module a; endmodule",
                    "rtl/a.v")
        refreshed = ph.refresh_current_sidecars(str(tmp_path), ["blkA"])
        assert refreshed == ["blkA"]
        sc = (tmp_path / ".coresmith" / "blocks" / "blkA"
              / "uarch_spec_contract_sha1").read_text()
        assert sc == "LIVE_blkA"
        best = json.loads((tmp_path / ".coresmith" / "blocks" / "blkA"
                           / "best_result.json").read_text())
        assert best["contract_sha1"] == "LIVE_blkA"

    def test_skips_overwritten_rtl(self, tmp_path, monkeypatch):
        from orchestrator.langgraph import pipeline_helpers as ph
        monkeypatch.setattr(ph, "block_contract_sha1",
                            lambda pr, n: "LIVE_" + n)
        self._block(tmp_path, "blkB", "x", "module b; endmodule", "rtl/b.v")
        # overwrite the RTL so its hash no longer matches best_result.rtl_sha1
        (tmp_path / "rtl" / "b.v").write_text("module b_WRONG; endmodule")
        refreshed = ph.refresh_current_sidecars(str(tmp_path), ["blkB"])
        assert refreshed == []  # RTL not intact -> not vouched for

    def test_skips_unpassed_block(self, tmp_path, monkeypatch):
        from orchestrator.langgraph import pipeline_helpers as ph
        monkeypatch.setattr(ph, "block_contract_sha1",
                            lambda pr, n: "LIVE_" + n)
        self._block(tmp_path, "blkC", "x", "module c; endmodule", "rtl/c.v",
                    passed=False)
        assert ph.refresh_current_sidecars(str(tmp_path), ["blkC"]) == []

    def test_skips_block_touching_changed_edge(self, tmp_path, monkeypatch):
        from orchestrator.langgraph import pipeline_helpers as ph
        monkeypatch.setattr(ph, "block_contract_sha1",
                            lambda pr, n: "LIVE_" + n)
        self._block(tmp_path, "wrapper", "x", "module w; endmodule", "rtl/w.v")
        import orchestrator.langchain.agents.contract_lookup as cl
        monkeypatch.setattr(
            cl, "load_block_contracts",
            lambda pr, n: [{"edge_id": "qspi_read_out__to__wrapper"}])
        refreshed = ph.refresh_current_sidecars(
            str(tmp_path), ["wrapper"], changed_edge_substrings=["qspi_read_out"])
        assert refreshed == []  # participates in the changed edge -> skip


class TestRestartFromNode:
    @pytest.mark.asyncio
    async def test_forks_from_matching_checkpoint(self):
        import types
        from orchestrator.graph_lifecycle import GraphLifecycle

        # a fake graph exposing aget_state_history + a spy safe_start
        class _Snap:
            def __init__(self, nxt, cid):
                self.next = nxt
                self.config = {"configurable": {"thread_id": "pipeline",
                                                 "checkpoint_id": cid}}

        class _Graph:
            async def aget_state_history(self, config):
                for s in (_Snap(("validation_dv",), "c3"),
                          _Snap(("integration_check",), "c2"),
                          _Snap(("process_block",), "c1")):
                    yield s

        lc = GraphLifecycle.__new__(GraphLifecycle)
        lc.name = "pipeline"
        lc.thread_id = "pipeline"
        lc.task = None
        lc.graph = _Graph()
        started = {}

        async def _ensure():
            return None

        async def _safe_start(inp, cfg):
            started["cfg"] = cfg

        lc.ensure_graph = _ensure
        lc.safe_start = _safe_start

        res = await lc.restart_from_node("integration_check")
        assert res["restarted"] is True
        assert res["checkpoint_id"] == "c2"
        assert started["cfg"]["configurable"]["checkpoint_id"] == "c2"

    @pytest.mark.asyncio
    async def test_unknown_node_errors(self):
        from orchestrator.graph_lifecycle import GraphLifecycle

        class _Graph:
            async def aget_state_history(self, config):
                if False:
                    yield None  # empty history

        lc = GraphLifecycle.__new__(GraphLifecycle)
        lc.name = "pipeline"
        lc.thread_id = "pipeline"
        lc.task = None
        lc.graph = _Graph()

        async def _ensure():
            return None
        lc.ensure_graph = _ensure
        res = await lc.restart_from_node("nonexistent_node")
        assert "error" in res
