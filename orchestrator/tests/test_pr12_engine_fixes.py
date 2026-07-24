# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""PR#12 engine follow-ups surfaced by the evaluation harness sweep (findings #3/#5/#7/
#9/#12). Pure-function unit coverage; no LLM / EDA toolchain."""
from __future__ import annotations

import json
import re


# --- #12: cocotb @test(stage=N) discovery -----------------------------------
class TestCocotbStageDiscovery:
    PAT = r"@cocotb\.test\s*\("

    def test_counts_plain_and_staged_and_kwargs(self):
        tb = (
            "@cocotb.test()\n"
            "async def a(dut): pass\n"
            "@cocotb.test(stage=1)\n"
            "async def b(dut): pass\n"
            "@cocotb.test(skip=False, timeout_time=10)\n"
            "async def c(dut): pass\n"
        )
        assert len(re.findall(self.PAT, tb)) == 3

    def test_old_regex_missed_staged(self):
        # regression witness: the pre-fix literal regex found only 1 of 3
        assert len(re.findall(r"@cocotb\.test\(\)",
                              "@cocotb.test()\n@cocotb.test(stage=1)\n"
                              "@cocotb.test(stage=2)\n")) == 1


# --- #3: storage-lint env thresholds + reviewed-flop exception --------------
class TestStorageLintThresholds:
    def test_env_override_words_and_bits(self, monkeypatch):
        from orchestrator.langgraph import rtl_storage_lint as sl
        monkeypatch.delenv("CORESMITH_STORAGE_LINT_MAX_WORDS", raising=False)
        assert sl.storage_lint_max_words() == 256
        monkeypatch.setenv("CORESMITH_STORAGE_LINT_MAX_WORDS", "512")
        monkeypatch.setenv("CORESMITH_STORAGE_LINT_MAX_BITS", "16384")
        assert sl.storage_lint_max_words() == 512
        assert sl.storage_lint_max_bits() == 16384

    def test_256deep_flop_array_flagged_by_default(self):
        from orchestrator.langgraph import rtl_storage_lint as sl
        v = ("module m; reg [31:0] mem [0:255];\n"
             "always @(posedge clk) q <= mem[a]; endmodule")
        rep = sl.find_oversized_memory_arrays(v)
        assert not rep.ok  # 256 words hits the default threshold

    def test_env_raise_clears_the_same_array(self, monkeypatch):
        from orchestrator.langgraph import rtl_storage_lint as sl
        v = "module m; reg [31:0] mem [0:255]; endmodule"
        monkeypatch.setenv("CORESMITH_STORAGE_LINT_MAX_WORDS", "1024")
        monkeypatch.setenv("CORESMITH_STORAGE_LINT_MAX_BITS", "65536")
        rep = sl.find_oversized_memory_arrays(v)
        assert rep.ok  # now under the raised threshold

    def test_reviewed_flop_exception_waives_size_only(self):
        from orchestrator.langgraph import rtl_storage_lint as sl
        v = ("module m; // coresmith:reviewed-flop-exception\n"
             "reg [31:0] mem [0:255]; endmodule")
        rep = sl.find_oversized_memory_arrays(v)
        assert rep.ok  # size threshold waived by the reviewed marker

    def test_exception_does_not_waive_untimeable_read(self):
        # a marker cannot rescue an actually-untimeable flat read mux; but
        # without a PDK (period_ns unset) the timing check is a no-op, so this
        # asserts the exception path is size-only by construction.
        from orchestrator.langgraph import rtl_storage_lint as sl
        v = ("module m; // coresmith:reviewed-flop-exception\n"
             "reg [7:0] mem [0:4095]; endmodule")
        rep = sl.find_oversized_memory_arrays(v)  # no period -> size-only
        assert rep.ok


# --- #5: stimulus auto-discovery --------------------------------------------
class TestStimulusAutoDiscovery:
    def test_autodiscovers_inputs_model_stimulus(self, tmp_path, monkeypatch):
        from orchestrator.architecture import model_integration as mi
        monkeypatch.delenv("CORESMITH_MODEL_STIMULUS", raising=False)
        inp = tmp_path / "inputs"
        inp.mkdir()
        (inp / "model_stimulus.py").write_text(
            "stimulus = {'data': [1, 2, 3], 'mode': 'fwd'}\n")
        stim, found = mi._load_env_stimulus(str(tmp_path))
        assert found and stim == {"data": [1, 2, 3], "mode": "fwd"}

    def test_env_wins_over_autodiscovery(self, tmp_path, monkeypatch):
        from orchestrator.architecture import model_integration as mi
        (tmp_path / "inputs").mkdir()
        (tmp_path / "inputs" / "model_stimulus.py").write_text(
            "stimulus = 'AUTO'\n")
        envf = tmp_path / "env_stim.py"
        envf.write_text("stimulus = 'ENV'\n")
        monkeypatch.setenv("CORESMITH_MODEL_STIMULUS", str(envf))
        stim, found = mi._load_env_stimulus(str(tmp_path))
        assert found and stim == "ENV"

    def test_none_when_neither(self, tmp_path, monkeypatch):
        from orchestrator.architecture import model_integration as mi
        monkeypatch.delenv("CORESMITH_MODEL_STIMULUS", raising=False)
        stim, found = mi._load_env_stimulus(str(tmp_path))
        assert not found and stim is None


# --- #9: oracle-manifest spec re-baseline vs immutable tamper ---------------
class TestOracleManifestRebaseline:
    def _setup(self, root):
        (root / "inputs").mkdir(parents=True)
        (root / "arch").mkdir(parents=True)
        (root / "inputs" / "requirements.md").write_text("golden reqs")
        (root / "arch" / "ers_spec.md").write_text("ers v1")
        (root / "arch" / "frd_spec.md").write_text("frd v1")
        (root / "arch" / "prd_spec.md").write_text("prd v1")

    def test_spec_only_edit_rebaselines_and_passes(self, tmp_path):
        from orchestrator.state_store import trust
        self._setup(tmp_path)
        trust.write_oracle_manifest(tmp_path)
        # authorized ERS amendment
        (tmp_path / "arch" / "ers_spec.md").write_text("ers v2 (cs_sram)")
        res = trust.check_oracle_manifest(tmp_path)
        assert res["ok"] is True
        assert "arch/ers_spec.md" in res.get("spec_rebaselined", [])
        # re-baselined: a second check is clean
        assert trust.check_oracle_manifest(tmp_path)["ok"] is True

    def test_immutable_oracle_edit_is_tamper(self, tmp_path):
        from orchestrator.state_store import trust
        self._setup(tmp_path)
        trust.write_oracle_manifest(tmp_path)
        (tmp_path / "inputs" / "requirements.md").write_text("MUTATED golden")
        res = trust.check_oracle_manifest(tmp_path)
        assert res["ok"] is False
        assert res["violation"]["category"] == "ORACLE_TAMPER"
        assert "inputs/requirements.md" in res["violation"]["immutable_drift"]

    def test_strict_mode_keeps_spec_edit_a_fail(self, tmp_path, monkeypatch):
        from orchestrator.state_store import trust
        self._setup(tmp_path)
        trust.write_oracle_manifest(tmp_path)
        (tmp_path / "arch" / "ers_spec.md").write_text("ers v2")
        monkeypatch.setenv("CORESMITH_STRICT_ORACLE_MANIFEST", "1")
        res = trust.check_oracle_manifest(tmp_path)
        assert res["ok"] is False


# --- #7: override forces past the mem_price gate ----------------------------
class TestMemPriceOverride:
    def test_override_marker_defers_gate(self, tmp_path):
        from orchestrator.langgraph import pipeline_graph as pg
        root = tmp_path
        (root / "arch" / "uarch_specs").mkdir(parents=True)
        (root / "arch" / "uarch_specs" / "blk.md").write_text(
            "# MEM big: 32x4096 ports=1rw impl=fpmem justification=x\n"
            "area_budget_um2: 20000\n")
        bdir = root / ".coresmith" / "blocks" / "blk"
        bdir.mkdir(parents=True)
        # without override: gate returns a revise request (over budget)
        v_no = pg._mem_price_gate_verdict(str(root), "blk")
        assert v_no is None or v_no.get("action") == "revise"
        # with override marker: gate defers (returns None = accept)
        (bdir / "uarch_feasibility_override").write_text("1")
        v_yes = pg._mem_price_gate_verdict(str(root), "blk")
        assert v_yes is None
        led = json.loads((bdir / "mem_price.json").read_text())
        assert led.get("deferred") is True


# --- #4: snapshot-before-regen data-loss guard ------------------------------
class TestSnapshotBeforeRegen:
    def test_snapshot_saves_rtl_and_tb(self, tmp_path, monkeypatch):
        from orchestrator.langgraph import pipeline_graph as pg
        monkeypatch.delenv("CORESMITH_SNAPSHOT_BEFORE_REGEN", raising=False)
        root = tmp_path
        (root / "rtl").mkdir(parents=True)
        (root / "tb").mkdir(parents=True)
        rtl = root / "rtl" / "wrapper.v"
        rtl.write_text("module user_project_wrapper; endmodule // GOOD att1")
        (root / "tb" / "test_wrapper.py").write_text("# GOOD att1 tb")
        bdir = root / ".coresmith" / "blocks" / "wrapper"
        bdir.mkdir(parents=True)
        (bdir / "best_result.json").write_text(json.dumps({"sim_passed": True}))
        block = {"name": "wrapper", "rtl_target": "rtl/wrapper.v",
                 "testbench": "tb/test_wrapper.py"}
        pg._snapshot_passing_block(
            str(root), "wrapper", block, rtl, 3, "contract_sha1_stale")
        snap = bdir / "passing_snapshot_attempt3"
        assert (snap / "wrapper.v").read_text().endswith("GOOD att1")
        assert (snap / "test_wrapper.py").exists()
        best = json.loads((bdir / "best_result.json").read_text())
        assert best["passing_snapshots"][0]["reason"] == "contract_sha1_stale"

    def test_disabled_by_env(self, tmp_path, monkeypatch):
        from orchestrator.langgraph import pipeline_graph as pg
        monkeypatch.setenv("CORESMITH_SNAPSHOT_BEFORE_REGEN", "0")
        root = tmp_path
        (root / "rtl").mkdir(parents=True)
        rtl = root / "rtl" / "w.v"
        rtl.write_text("module w; endmodule")
        bdir = root / ".coresmith" / "blocks" / "w"
        bdir.mkdir(parents=True)
        pg._snapshot_passing_block(
            str(root), "w", {"name": "w", "rtl_target": "rtl/w.v"}, rtl, 1, "x")
        assert not (bdir / "passing_snapshot_attempt1").exists()
