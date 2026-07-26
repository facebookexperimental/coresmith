# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""C5: sim-pass provenance spans RTL + TB + the block's interface-contract
slice, and the block-model sidecar records which contract a model was
generated against. The fragment_metadata_memory livelock: a recorded pass with
a MATCHING rtl_sha1 but from an obsolete 48-bit TB/contract era was honored
forever by the skip-regen fast path."""

import json

from orchestrator.langgraph.pipeline_helpers import block_contract_sha1


def _write_contracts(root, contracts):
    (root / ".coresmith").mkdir(exist_ok=True)
    (root / ".coresmith" / "interface_contracts.json").write_text(
        json.dumps({"contracts": contracts}))


def _edge(producer, consumer, width):
    return {"producer_block": producer, "consumer_block": consumer,
            "signal": f"{producer}_to_{consumer}", "width": width}


class TestBlockContractSha1:
    def test_stable_and_nonempty(self, tmp_path):
        _write_contracts(tmp_path, [_edge("a", "b", 48)])
        h1 = block_contract_sha1(tmp_path, "a")
        h2 = block_contract_sha1(tmp_path, "a")
        assert h1 and h1 == h2

    def test_changes_when_own_edge_changes(self, tmp_path):
        _write_contracts(tmp_path, [_edge("a", "b", 48)])
        before = block_contract_sha1(tmp_path, "a")
        _write_contracts(tmp_path, [_edge("a", "b", 56)])  # widen 48 -> 56
        after = block_contract_sha1(tmp_path, "a")
        assert before and after and before != after

    def test_unchanged_when_other_blocks_edge_changes(self, tmp_path):
        # Hashing the block's OWN slice keeps a chip-lead edit to one block's
        # contract from invalidating every other block's recorded pass.
        _write_contracts(tmp_path, [_edge("a", "b", 48), _edge("c", "d", 8)])
        before = block_contract_sha1(tmp_path, "a")
        _write_contracts(tmp_path, [_edge("a", "b", 48), _edge("c", "d", 16)])
        after = block_contract_sha1(tmp_path, "a")
        assert before and before == after

    def test_missing_file_is_empty(self, tmp_path):
        assert block_contract_sha1(tmp_path, "a") == ""


class TestPassProvenance:
    def test_omits_unavailable_axes(self, tmp_path):
        # No contract file + no TB -> only rtl_sha1 recorded; absent keys mean
        # "axis not checked" so older best_result.json files stay honored.
        from orchestrator.langgraph.pipeline_graph import _pass_provenance

        rtl = tmp_path / "block.v"
        rtl.write_text("module m; endmodule\n")
        prov = _pass_provenance(tmp_path, "a", str(rtl), "")
        assert set(prov) == {"rtl_sha1"}

    def test_full_provenance_when_available(self, tmp_path):
        from orchestrator.langgraph.pipeline_graph import _pass_provenance

        _write_contracts(tmp_path, [_edge("a", "b", 48)])
        rtl = tmp_path / "block.v"
        rtl.write_text("module m; endmodule\n")
        tb = tmp_path / "test_block.py"
        tb.write_text("async def test(): pass\n")
        prov = _pass_provenance(tmp_path, "a", str(rtl), str(tb))
        assert set(prov) == {"rtl_sha1", "tb_sha1", "contract_sha1"}
        # TB edit flips only the TB axis.
        tb.write_text("async def test2(): pass\n")
        prov2 = _pass_provenance(tmp_path, "a", str(rtl), str(tb))
        assert prov2["rtl_sha1"] == prov["rtl_sha1"]
        assert prov2["contract_sha1"] == prov["contract_sha1"]
        assert prov2["tb_sha1"] != prov["tb_sha1"]
