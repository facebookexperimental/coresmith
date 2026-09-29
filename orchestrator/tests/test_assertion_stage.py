# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""A3: spec invariants must exist as assertions; phantom claims are rejected."""
from __future__ import annotations

import asyncio

from orchestrator.langgraph import assertion_stage as A
from orchestrator.langgraph import pipeline_graph as pg
from orchestrator.state_store.project_db import open_project

_SPEC = """# core -- spec
## 4. Algorithm Mapping
### 4a. Cross-Block Semantic Invariants

| ID | Invariant | Ports | Hook |
|---|---|---|---|
| INV-L1D-SWMR-001 (INV-002) | at most one E/M copy | tags | SVA |
| INV-L1D-SNPLAT-003 (IFACE-013) | snp_rsp_valid(t+1) == snp_valid(t) | s_snoop | SVA |

- **Invariant ID**: INV-BULLET-004 -- a bullet-style invariant
- INV-L1D-SWMR-001 mentioned again is not a new entry

## 5. Reset and Initialization
INV-NOT-IN-4A-009 must not be picked up.
"""

_EDGE = {"edge_id": "req__m_q__to__core__s_q", "producer_block": "req", "producer_port": "m_q",
         "consumer_block": "core", "consumer_port": "s_q", "handshake_protocol": "req_resp",
         "data_width_bits": 8, "fields": [{"name": "addr", "width": 8, "msb": 7, "lsb": 0}],
         "sideband_signals": [{"name": "req_valid"}, {"name": "rsp_valid"}],
         "timing": {"req_to_rsp_cycles": {"exact": 1}}}

_STREAM = {"edge_id": "core__m_axis_o__to__sink__s_axis_o", "producer_block": "core",
           "producer_port": "m_axis_o", "consumer_block": "sink", "consumer_port": "s_axis_o",
           "handshake_protocol": "axi_stream", "data_width_bits": 8,
           "fields": [{"name": "tdata", "width": 8, "msb": 7, "lsb": 0}],
           "timing": {"valid_to_ready_max_stall": 2}}


def _project(tmp_path, with_spec=True):
    db = open_project(tmp_path)
    db.import_contracts({"contracts": [_EDGE, _STREAM]})
    if with_spec:
        d = tmp_path / "arch" / "uarch_specs"
        d.mkdir(parents=True)
        (d / "core.md").write_text(_SPEC)
    return db


class TestChecklist:
    def test_spec_invariants_from_table_and_bullets_only_in_4a(self):
        rows = A.spec_invariants(_SPEC)
        assert [r["id"] for r in rows] == ["INV-L1D-SWMR-001", "INV-L1D-SNPLAT-003", "INV-BULLET-004"]
        assert rows[0]["aliases"] == ["INV-002"]
        assert "snp_rsp_valid" in rows[1]["text"]

    def test_contract_invariants_per_timing_rule(self, tmp_path):
        _project(tmp_path)
        ids = {i["id"]: i for i in A.contract_invariants(str(tmp_path), "core")}
        assert "TIM-req__m_q__to__core__s_q-latency" in ids
        assert ids["TIM-req__m_q__to__core__s_q-latency"]["role"] == "consumer"
        assert "TIM-core__m_axis_o__to__sink__s_axis_o-hold" in ids
        assert "TIM-core__m_axis_o__to__sink__s_axis_o-stall" in ids
        assert "TIM-req__m_q__to__core__s_q-reset_idle" in ids


class TestFindAndPhantom:
    def test_tags_cover_assertions_within_the_window(self):
        rtl = """module core(input clk, input rst_n, input a, output reg q);
`ifndef SYNTHESIS
  // INV: INV-L1D-SWMR-001, INV-002
  always @(posedge clk) if (rst_n && !(a || !q)) $error("SWMR");
  // INV: INV-FAR-999
  //
  //
  //
  //
  assert property (@(posedge clk) disable iff (!rst_n) a |-> ##1 q);
`endif
`ifdef NEVER
  // INV: INV-HIDDEN-001
  assert property (@(posedge clk) a);
`endif
endmodule
"""
        f = A.find_assertions(rtl)
        assert len(f["assertions"]) == 2
        assert {"INV-L1D-SWMR-001", "INV-002"} <= f["covered"]
        assert "INV-FAR-999" not in f["covered"]      # tag too far from the assertion
        assert "INV-HIDDEN-001" not in f["tags"]      # ifdef'd out as sim sees it

    def test_phantom_claims(self):
        rtl = """  assign wlast = cnt == len;  // checked by assertion
  // asserted in the TB
  wire x;
  wire y;
  wire z;
  // INV: INV-OK-1
  always @(posedge clk) if (!(x)) $error("x");  // enforced by assertion
"""
        ph = A.phantom_assertion_claims(rtl)
        assert [p["line"] for p in ph] == [1, 2]      # line 7 has the assertion on it


class TestEvaluate:
    def _rtl(self, tmp_path, text):
        f = tmp_path / "core.v"
        f.write_text(text)
        return f

    def test_missing_and_vip_sva_coverage(self, tmp_path, monkeypatch):
        from orchestrator.langgraph import vip_lib as V
        monkeypatch.setenv("CORESMITH_ASSERTION_STAGE", "1")
        monkeypatch.setenv("CORESMITH_VIP_SVA_BIND", "1")
        _project(tmp_path)
        V.write_all_vips(tmp_path, [_EDGE, _STREAM])
        rtl = self._rtl(tmp_path, """module core(input clk, input rst_n);
`ifndef SYNTHESIS
  // INV: INV-L1D-SWMR-001
  always @(posedge clk) if (rst_n && 0) $error("x");
  // INV: INV-L1D-SNPLAT-003
  always @(posedge clk) if (rst_n && 0) $error("y");
`endif
endmodule
""")
        r = A.evaluate(str(tmp_path), "core", rtl)
        assert r["ok"] is False and r["mode"] == "gate"
        assert [m["id"] for m in r["missing"]] == ["INV-BULLET-004"]
        by = {c["id"]: c["by"] for c in r["covered"]}
        assert by["INV-L1D-SWMR-001"] == "rtl"
        assert by["TIM-req__m_q__to__core__s_q-latency"] == "vip_sva"
        assert by["TIM-core__m_axis_o__to__sink__s_axis_o-hold"] == "vip_sva"
        assert (tmp_path / ".coresmith/blocks/core/assertions_checklist.json").exists()
        assert "INV-BULLET-004" in A.feedback_text(r)

    def test_modes(self, tmp_path, monkeypatch):
        _project(tmp_path)
        rtl = self._rtl(tmp_path, "module core(input clk);\nendmodule\n")
        monkeypatch.setenv("CORESMITH_ASSERTION_STAGE", "advisory")
        r = A.evaluate(str(tmp_path), "core", rtl)
        assert r["ok"] is True and r["missing"] and r["mode"] == "advisory"
        monkeypatch.setenv("CORESMITH_ASSERTION_STAGE", "1")
        assert A.evaluate(str(tmp_path), "core", rtl)["ok"] is False
        monkeypatch.setenv("CORESMITH_ASSERTION_STAGE", "0")
        assert A.mode() == "off"


class TestNode:
    def test_graph_and_routing(self):
        g = pg.build_block_subgraph()
        assert "assertion_check" in g.nodes
        assert pg.route_after_rtl({"lint_clean": True}) == "assertion_check"
        assert pg.route_after_assertions({"assertion_ok": False}) == "diagnose"
        assert pg.route_after_assertions({}) == "generate_testbench"

    def test_node_gates_and_writes_feedback(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_ASSERTION_STAGE", "1")
        _project(tmp_path)
        rtl = tmp_path / "core.v"
        rtl.write_text("module core(input clk); // verified by assertion elsewhere\nendmodule\n")
        st = {"project_root": str(tmp_path), "current_block": {"name": "core"}, "rtl_path": str(rtl),
              "attempt": 1, "phase": "lint"}
        out = asyncio.run(pg.assertion_check_node(st))
        assert out == {"assertion_ok": False, "phase": "assertions"}
        prev = (tmp_path / ".coresmith/blocks/core/previous_error.txt").read_text()
        assert "INV-L1D-SWMR-001" in prev and "Phantom" in prev
        (tmp_path / ".coresmith/blocks/core/constraints.json").write_text("[]")
        d = asyncio.run(pg.diagnose_node({**st, "phase": "assertions"}))
        assert d["debug_action"] == "retry_rtl"
        assert pg._db(str(tmp_path)).diagnosis("core")["category"] == "ASSERTION_COVERAGE"
        monkeypatch.setenv("CORESMITH_ASSERTION_STAGE", "0")
        assert asyncio.run(pg.assertion_check_node(st)) == {"assertion_ok": True}

    def test_runtime_assertion_failure_is_an_rtl_bug(self, tmp_path):
        bdir = tmp_path / ".coresmith/blocks/core"
        bdir.mkdir(parents=True)
        (bdir / "constraints.json").write_text("[]")
        (bdir / "previous_error.txt").write_text("%Error: core.v:12: Assertion failed in TOP.core: INV-1\n")
        st = {"project_root": str(tmp_path), "current_block": {"name": "core"}, "attempt": 1, "phase": "sim"}
        d = asyncio.run(pg.diagnose_node(st))
        assert d["debug_action"] == "retry_rtl"
        assert pg._db(str(tmp_path)).diagnosis("core")["category"] == "ASSERTION_FAILED"
