# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Engine defects exposed by the MCU+FFT SoC evaluation run.

1. a primitive block's generated testbench is the testbench of record (the LLM
   TB that replaced it segfaulted Verilator); CORESMITH_PRIMITIVE_LLM_TB=1
   restores the LLM author;
2. every block-subgraph route target exists (``retry_sim`` routed to a
   non-existent ``simulate`` node and was silently dropped);
3. the materializer writes rtl_target / testbench / module_name to the block row;
4. uArch spec writers register ``uarch:<block>`` in the ontology;
5. ``verify synth --full`` without a WNS is a tool error, not a pass;
7. preflight catches an old Verilator, unreachable OpenSTA, missing yosys-slang
   and the engine bin/ off PATH.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import orchestrator.langgraph.pipeline_helpers as ph
from orchestrator.langgraph import pipeline_graph as pg
from orchestrator.state_store.project_db import open_project

_FAB = {"name": "soc", "masters": [{"name": "hart0"}],
        "slaves": [{"name": "ram", "protocol": "axi4", "base": 0x80000000, "size": 0x1000000}]}
# As the architect registers it: no rtl_target / testbench / module_name.
_PRIM = {"name": "soc_fabric", "kind": "primitive", "primitive": "cs_fabric", "tier": 0, "fabric": _FAB}


def _fake_generate(spec, out_dir, tb_dir=None, **kw):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{spec.module_name}.v").write_text(
        f"module {spec.module_name}(input wire clk, input wire rst_n); endmodule\n")
    tbd = Path(tb_dir or out_dir)
    tbd.mkdir(parents=True, exist_ok=True)
    (tbd / f"test_{spec.module_name}.py").write_text("# generated cocotbext-axi tb\n")
    return SimpleNamespace(module=spec.module_name, rtl_path=str(out / f"{spec.module_name}.v"),
                           tb_path=str(tbd / f"test_{spec.module_name}.py"), cached=False,
                           ports=[{"name": "clk", "dir": "input", "width": 1},
                                  {"name": "rst_n", "dir": "input", "width": 1},
                                  {"name": "m_ram_awaddr", "dir": "output", "width": 32}])


def _materialize(tmp_path, monkeypatch):
    import orchestrator.fabric as fabric
    monkeypatch.setattr(fabric, "generate_fabric", _fake_generate)
    db = open_project(tmp_path)
    db.import_block_diagram({"blocks": [_PRIM, {"name": "mcu", "tier": 1}], "connections": []})
    out = asyncio.run(pg.materialize_primitive_node(
        {"project_root": str(tmp_path), "current_block": dict(_PRIM), "attempt": 1}))
    return db, out


# --------------------------------------------------------------------------- 1
class TestPrimitiveTestbenchOfRecord:
    def _gates_off(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_CONTRACT_CONFORMANCE_GATE", "0")
        monkeypatch.setenv("CORESMITH_CONTRACT_PORT_GATE", "0")

    def test_generated_tb_is_simulated_never_authored(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CORESMITH_PRIMITIVE_LLM_TB", raising=False)
        self._gates_off(monkeypatch)
        _db, out = _materialize(tmp_path, monkeypatch)
        assert out["current_block"]["testbench"] == "tb/cocotb/test_cs_fabric_soc.py"
        state = {**out, "project_root": str(tmp_path), "attempt": 1}
        authored, fixed, sims = [], [], []

        async def llm_tb(*a, **k):
            authored.append(a)
            return {"test_count": 1}

        async def llm_fix(*a, **k):
            fixed.append(a)
            return "fixed"

        def sim(block, rtl, tb, attempt, **k):
            sims.append(tb)
            return {"passed": False, "log": "ImportError: cocotb test framework\nSegmentation fault"}

        with patch.object(pg, "generate_testbench", llm_tb), \
                patch.object(pg, "fix_testbench_errors", llm_fix), \
                patch.object(pg, "run_simulation", sim):
            res = asyncio.run(pg.generate_testbench_node(state))
        assert not authored and not fixed
        assert sims and Path(sims[0]) == tmp_path / "tb/cocotb/test_cs_fabric_soc.py"
        assert res["tb_path"] == str(tmp_path / "tb/cocotb/test_cs_fabric_soc.py")
        assert (tmp_path / "tb/cocotb/test_cs_fabric_soc.py").read_text().startswith("# generated")
        assert not (tmp_path / "tb/cocotb/test_soc_fabric.py").exists()

    def test_env_gate_restores_the_llm_author(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_PRIMITIVE_LLM_TB", "1")
        self._gates_off(monkeypatch)
        _db, out = _materialize(tmp_path, monkeypatch)
        assert "testbench" not in out["current_block"]  # old behaviour: not written back
        state = {**out, "project_root": str(tmp_path), "attempt": 1, "tb_path": ""}
        authored = []

        async def llm_tb(block, **k):
            authored.append(block["testbench"])
            p = tmp_path / block["testbench"]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("# llm tb\n")
            return {"test_count": 1}

        with patch.object(pg, "generate_testbench", llm_tb), \
                patch.object(pg, "run_simulation", lambda *a, **k: {"passed": False, "log": "x"}):
            asyncio.run(pg.generate_testbench_node(state))
        assert authored == ["tb/cocotb/test_soc_fabric.py"]


# --------------------------------------------------------------------------- 2
class TestRouteTargetsExist:
    _ACTIONS = ("retry_rtl", "retry_tb", "retry_synth", "retry_sim", "retry_rtl_timing",
                "ask_human", "escalate", "something_unknown")

    def test_every_route_decision_target_is_a_subgraph_node(self):
        nodes = set(pg.build_block_subgraph().compile().get_graph().nodes)
        blocks = [{"name": "alu"}, {"name": "soc_fabric", "kind": "primitive"}]
        for blk in blocks:
            assert pg._rtl_node({"current_block": blk}) in nodes
            for act in self._ACTIONS:
                target = pg.route_decision({"current_block": blk, "debug_action": act})
                assert target in nodes, (blk["name"], act, target)
        for label in pg.route_decision.__edge_labels__:
            assert label in nodes, label

    def test_retry_sim_reruns_the_sim_with_the_tb_kept(self):
        expected = {"retry_rtl": "generate_rtl", "retry_tb": "generate_testbench",
                    "retry_synth": "synthesize", "retry_sim": "generate_testbench",
                    "retry_rtl_timing": "timing_fix", "ask_human": "ask_human",
                    "escalate": "block_done"}
        for act, node in expected.items():
            assert pg.route_decision({"current_block": {"name": "alu"}, "debug_action": act}) == node
        assert pg.route_decision({"current_block": {"name": "alu"},
                                  "debug_action": "retry_sim"}) == "generate_testbench"
        # a primitive's sim retry does not need re-materialization either
        assert pg.route_decision({"current_block": {"name": "f", "kind": "primitive"},
                                  "debug_action": "retry_sim"}) == "generate_testbench"

    def test_decide_sets_sim_retry_only_for_retry_sim(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pg, "_db", lambda pr: SimpleNamespace(diagnosis=lambda b: {"category": "X"}))
        base = {"project_root": str(tmp_path), "current_block": {"name": "alu"}, "attempt": 1,
                "max_attempts": 3}
        assert asyncio.run(pg.decide_node({**base, "debug_action": "retry_sim"}))["sim_retry"] is True
        assert asyncio.run(pg.decide_node({**base, "debug_action": "retry_rtl"}))["sim_retry"] is False

    def test_sim_retry_keeps_the_existing_tb(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_CONTRACT_CONFORMANCE_GATE", "0")
        monkeypatch.setenv("CORESMITH_CONTRACT_PORT_GATE", "0")
        monkeypatch.setenv("CORESMITH_INTERFACE_VIP", "0")
        rtl = tmp_path / "alu.v"
        rtl.write_text("module alu(input wire clk); endmodule\n")
        tb = tmp_path / "tb/cocotb/test_alu.py"
        tb.parent.mkdir(parents=True)
        tb.write_text("# tb\n")
        authored = []

        async def llm_tb(*a, **k):
            authored.append(a)
            return {}

        state = {"project_root": str(tmp_path), "attempt": 1, "rtl_path": str(rtl),
                 "current_block": {"name": "alu", "testbench": "tb/cocotb/test_alu.py"},
                 "sim_retry": True, "force_regen_tb": False, "preserve_testbench": False}
        with patch.object(pg, "generate_testbench", llm_tb), \
                patch.object(pg, "run_simulation", lambda *a, **k: {"passed": False, "log": "x"}):
            out = asyncio.run(pg.generate_testbench_node(state))
        assert not authored and out["sim_retry"] is False


# --------------------------------------------------------------------------- 3
class TestMaterializerWritesTheBlockRow:
    def test_row_and_state_name_the_generated_files(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CORESMITH_PRIMITIVE_LLM_TB", raising=False)
        db, out = _materialize(tmp_path, monkeypatch)
        blk = out["current_block"]
        assert blk["rtl_target"] == "rtl/interconnect/cs_fabric_soc.v"
        assert blk["testbench"] == "tb/cocotb/test_cs_fabric_soc.py"
        assert blk["module_name"] == "cs_fabric_soc"
        rows = {b["name"]: b for b in db.block_specs()}
        assert rows["soc_fabric"]["rtl_target"] == "rtl/interconnect/cs_fabric_soc.v"
        assert rows["soc_fabric"]["testbench"] == "tb/cocotb/test_cs_fabric_soc.py"
        assert rows["soc_fabric"]["kind"] == "primitive" and rows["soc_fabric"]["fabric"]["name"] == "soc"
        assert "mcu" in rows  # the rest of the queue is untouched
        with db._conn() as conn:
            extra = json.loads(conn.execute(
                "SELECT extra_json FROM blocks WHERE name='soc_fabric'").fetchone()["extra_json"])
        assert extra["module_name"] == "cs_fabric_soc"
        # the harness path resolution now finds the generated RTL, not rtl/<name>.v
        from orchestrator.harness.verify import _resolve_rtl_path, _resolve_tb_path
        assert Path(_resolve_rtl_path(tmp_path, rows["soc_fabric"])).exists()
        assert Path(_resolve_tb_path(tmp_path, rows["soc_fabric"], None)).exists()
        # the block_specs.json view the backend's missing_rtl gate reads
        view = json.loads((tmp_path / ".coresmith/block_specs.json").read_text())
        fab = next(b for b in view if b["name"] == "soc_fabric")
        assert (tmp_path / fab["rtl_target"]).exists()


# --------------------------------------------------------------------------- 4
_SPEC = """# alu
## 2. Interface
| port | dir |
## 3. Microarchitecture
x
## 4. Behaviour
### 4a Cross-Block Semantic Invariants
- INV-ALU-001: sum is a + b mod 2^32. Meets PERF-004.
## 5. Reset
r
### 6a Output Timing Contract
t
## 9. Verilog Interface Stub
module alu(); endmodule
"""


class TestUarchRegistration:
    def test_helper_registers_uarch_items(self, tmp_path):
        spec = tmp_path / "arch/uarch_specs/alu.md"
        spec.parent.mkdir(parents=True)
        spec.write_text(_SPEC)
        assert pg._register_uarch_best_effort(str(tmp_path), "alu", spec) is True
        db = pg._db(str(tmp_path))
        art = db.artifact("uarch:alu")
        assert art and art["path"] == "arch/uarch_specs/alu.md" and art["registered_by"] == "graph"
        assert any(i["id"] == "INV-ALU-001" for i in db.items(artifact="uarch:alu"))

    def test_helper_never_raises(self, tmp_path):
        assert pg._register_uarch_best_effort(str(tmp_path), "nope", tmp_path / "missing.md") is False
        bad = tmp_path / "bad.md"
        bad.write_text("# no sections\n")
        assert pg._register_uarch_best_effort(str(tmp_path), "bad", bad) is False

    def test_uarch_phase_writer_registers(self, tmp_path, monkeypatch):
        async def fake_single(todo, **k):
            for b in todo:
                p = tmp_path / "arch/uarch_specs" / f"{b['name']}.md"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(_SPEC)
            return {"written": [b["name"] for b in todo], "missing": []}
        monkeypatch.setattr(ph, "generate_uarch_specs_single_context", fake_single)
        asyncio.run(pg._uarch_phase_specs(str(tmp_path), [{"name": "alu"}]))
        assert pg._db(str(tmp_path)).artifact("uarch:alu")

    def test_adoption_registers(self, tmp_path):
        reviewed = tmp_path / "arch/uarch_specs_review/alu.md"
        reviewed.parent.mkdir(parents=True)
        reviewed.write_text(_SPEC)
        res = pg._adopt_reviewed_specs(str(tmp_path), ["alu"], {"alu": str(reviewed)})
        assert res.ok, res.error
        assert pg._db(str(tmp_path)).artifact("uarch:alu")["path"] == "arch/uarch_specs/alu.md"

    def test_materialized_primitive_spec_registers(self, tmp_path, monkeypatch):
        db, _out = _materialize(tmp_path, monkeypatch)
        assert pg._db(str(tmp_path)).artifact("uarch:soc_fabric")


# --------------------------------------------------------------------------- 5
class TestVerifySynthFullNeedsTiming:
    def _run(self, tmp_path, monkeypatch, meta):
        from orchestrator.harness import verify as V
        rtl = tmp_path / "rtl/alu.v"
        rtl.parent.mkdir(parents=True)
        rtl.write_text("module alu(); endmodule\n")
        monkeypatch.setattr(ph, "synthesize_block", lambda *a, **k: {
            "success": True, "ff_count": 3, "gate_count": 10, "chip_area_um2": 1.0})
        monkeypatch.setattr(pg, "_evaluate_ppa_gate", lambda *a, **k: (None, [], meta))
        return V.verify_synth(tmp_path, {"name": "alu"}, full=True)

    def test_no_wns_is_a_tool_error(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CORESMITH_VERIFY_SYNTH_ALLOW_UNMEASURED", raising=False)
        r = self._run(tmp_path, monkeypatch, {"wns_ns": None})
        assert not r.passed and r.infra_error and r.exit_code == 3
        assert "timing not measured" in r.verdict and r.details["tool_error"] is True

    def test_waiver_passes_unmeasured(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_VERIFY_SYNTH_ALLOW_UNMEASURED", "1")
        r = self._run(tmp_path, monkeypatch, {"wns_ns": None})
        assert r.passed and not r.infra_error and "NOT measured" in r.verdict

    def test_measured_wns_passes_and_negative_fails(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CORESMITH_VERIFY_SYNTH_ALLOW_UNMEASURED", raising=False)
        r = self._run(tmp_path, monkeypatch, {"wns_ns": 1.5, "tns_ns": 0.0})
        assert r.passed and r.details["wns_ns"] == 1.5

    def test_negative_wns_fails(self, tmp_path, monkeypatch):
        r = self._run(tmp_path, monkeypatch, {"wns_ns": -0.4, "tns_ns": -2.0})
        assert not r.passed and not r.infra_error and "timing violated" in r.verdict


# --------------------------------------------------------------------------- 7
class TestToolPreflight:
    @pytest.fixture
    def env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CORESMITH_ALLOW_NO_OPENRAM", "1")
        monkeypatch.delenv("CORESMITH_SYNTH_GENERIC", raising=False)
        monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))
        monkeypatch.setattr(ph, "LIBERTY_FILE", tmp_path)          # "exists": PDK-mapped flow
        monkeypatch.setattr(ph, "PDK_ROOT", tmp_path)
        (tmp_path / "sky130A").mkdir()
        real_sta = tmp_path / "real_sta"
        real_sta.write_text("#!/bin/sh\n")
        real_sta.chmod(0o755)
        monkeypatch.setenv("CORESMITH_REAL_STA", str(real_sta))
        monkeypatch.setenv("PATH", str(ph._ENGINE_BIN))
        tools = {"verilator": "/opt/v/bin/verilator", "yosys": "/usr/bin/yosys",
                 "sta": str(ph._ENGINE_BIN / "sta")}
        monkeypatch.setattr(
            ph.shutil, "which",
            lambda n, *a, **k: tools.get(n) or (
                n if Path(n).is_file() and os.access(n, os.X_OK) else None))
        version = {"out": "Verilator 5.036 2025-04-27 rev v5.036"}

        def fake_run(cmd, *a, **k):
            return subprocess.CompletedProcess(cmd, 0, version["out"], "")
        monkeypatch.setattr(ph, "run_process", fake_run)
        return SimpleNamespace(tools=tools, version=version, root=tmp_path)

    def test_clean_host_is_ok(self, env):
        r = ph.preflight_check(["pipeline"])
        assert r["ok"], r
        assert not [w for w in r["warnings"] if "PATH" in w or "sta" in w.lower()]

    def test_old_verilator_is_an_error(self, env):
        env.version["out"] = "Verilator 5.020 2024-01-01 rev (Debian 5.020-1)"
        r = ph.preflight_check(["pipeline"])
        assert not r["ok"]
        assert any("5.020" in e and "5.036" in e and "PATH" in e for e in r["errors"])

    def test_unreadable_verilator_version_is_an_error(self, env):
        env.version["out"] = "garbage"
        assert any("--version" in e for e in ph.preflight_check(["pipeline"])["errors"])

    def test_sta_shim_without_real_sta_is_an_error(self, env, monkeypatch):
        monkeypatch.setenv("CORESMITH_REAL_STA", str(env.root / "nope"))
        r = ph.preflight_check(["pipeline"])
        assert any("CORESMITH_REAL_STA" in e and "nope" in e for e in r["errors"])

    def test_sta_shim_resolves_explicit_command_from_path(self, env, monkeypatch):
        monkeypatch.setenv("CORESMITH_REAL_STA", "sta-real")
        env.tools["sta-real"] = "/run/tools/sta-real"
        assert ph._sta_problem() is None

    def test_no_sta_on_path_is_an_error(self, env):
        env.tools.pop("sta")
        assert any("no `sta` on PATH" in e for e in ph.preflight_check(["pipeline"])["errors"])

    def test_no_sta_is_only_a_warning_for_generic_synth(self, env, monkeypatch):
        env.tools.pop("sta")
        monkeypatch.setenv("CORESMITH_SYNTH_GENERIC", "1")
        r = ph.preflight_check(["pipeline"])
        assert not any("sta" in e for e in r["errors"])
        assert any("no `sta` on PATH" in w for w in r["warnings"])

    def test_primitive_fabric_needs_yosys_slang(self, env, monkeypatch):
        (env.root / ".coresmith").mkdir()
        (env.root / ".coresmith/block_diagram.json").write_text(json.dumps(
            {"blocks": [{"name": "soc_fabric", "kind": "primitive"}]}))
        import orchestrator.fabric.generate as fg
        monkeypatch.setattr(fg, "slang_available", lambda yb=None: False)
        r = ph.preflight_check(["pipeline"])
        assert any("yosys-slang" in e and "CORESMITH_FABRIC_YOSYS" in e for e in r["errors"])
        monkeypatch.setattr(fg, "slang_available", lambda yb=None: True)
        assert ph.preflight_check(["pipeline"])["ok"]

    def test_no_primitive_no_slang_probe(self, env, monkeypatch):
        import orchestrator.fabric.generate as fg
        monkeypatch.setattr(fg, "slang_available", lambda yb=None: pytest.fail("probed"))
        assert ph.preflight_check(["pipeline"])["ok"]

    def test_engine_bin_off_path_is_a_warning(self, env, monkeypatch):
        monkeypatch.setenv("PATH", "/usr/bin")
        r = ph.preflight_check(["pipeline"])
        assert r["ok"]
        assert any("bin/" in w and "PATH" in w for w in r["warnings"])
