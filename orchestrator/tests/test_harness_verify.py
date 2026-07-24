# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the verify harness (wrapped deterministic checks).

These NEVER invoke real verilator/yosys -- the wrapped standalone functions
are monkeypatched so the tests exercise the harness glue + fail-closed
equivalence semantics only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.harness import verify as V
from orchestrator.state_store.store import Scoreboard


@pytest.fixture
def pr(tmp_path) -> Path:
    (tmp_path / ".coresmith").mkdir()
    return tmp_path


def _make_rtl(pr: Path, name: str) -> dict:
    (pr / "rtl").mkdir(exist_ok=True)
    (pr / "rtl" / f"{name}.v").write_text(f"module {name}(); endmodule\n")
    (pr / "tb" / "cocotb").mkdir(parents=True, exist_ok=True)
    (pr / "tb" / "cocotb" / f"test_{name}.py").write_text("# tb\n")
    # A non-empty uarch spec so verify_synth reaches parse_ff_budget (it skips
    # budget parsing entirely when the spec file is empty/absent).
    (pr / "arch" / "uarch_specs").mkdir(parents=True, exist_ok=True)
    (pr / "arch" / "uarch_specs" / f"{name}.md").write_text(
        f"# {name}\nflip_flop_budget: 100\n"
    )
    return {"name": name, "rtl_target": f"rtl/{name}.v",
            "testbench": f"tb/cocotb/test_{name}.py"}


class TestVerifyResult:
    def test_exit_codes(self):
        assert V.VerifyResult(True).exit_code == 0
        assert V.VerifyResult(False).exit_code == 1
        assert V.VerifyResult(False, skipped=True).exit_code == 4
        assert V.VerifyResult(False, infra_error=True).exit_code == 3

    def test_json_and_human(self):
        r = V.VerifyResult(True, verdict="ok", log_path="/x")
        j = r.to_json()
        assert j["passed"] is True and j["exit_code"] == 0
        assert "PASS" in r.to_human()


class TestVerifyModel:
    def _patch(self, monkeypatch, *, elab_err=None, missing=None, feasible=True):
        import orchestrator.langgraph.microarch_exp as mx
        monkeypatch.setattr(mx, "elaborate_block_model", lambda p, b: elab_err)
        monkeypatch.setattr(mx, "_read_block_diagram", lambda p: {})
        monkeypatch.setattr(mx, "_expected_ports_for_block", lambda bd, b: ["s_axis"])
        monkeypatch.setattr(mx, "_factory_params", lambda p, b: ["s_axis_tdata"])
        monkeypatch.setattr(mx, "check_interface_constraint",
                            lambda params, exp: missing or [])
        monkeypatch.setattr(mx, "_read_target_clock_mhz", lambda p: 50.0)
        monkeypatch.setattr(mx, "_read_uarch_specs", lambda p, bs: {})
        monkeypatch.setattr(mx, "_size_one_model",
                            lambda mp, b, mhz, spec: {"feasible": feasible,
                                                      "detail": "over budget"})

    def test_elaboration_error_fails(self, pr, monkeypatch):
        self._patch(monkeypatch, elab_err="SyntaxError line 3")
        r = V.verify_model(pr, "adder")
        assert r.exit_code == 1
        assert r.details["stage"] == "elaborate"

    def test_missing_interface_fails(self, pr, monkeypatch):
        self._patch(monkeypatch, missing=["m_axis"])
        r = V.verify_model(pr, "adder")
        assert r.exit_code == 1
        assert r.details["missing"] == ["m_axis"]

    def test_infeasible_fails(self, pr, monkeypatch):
        self._patch(monkeypatch, feasible=False)
        r = V.verify_model(pr, "adder")
        assert r.exit_code == 1
        assert r.details["stage"] == "size"

    def test_pass(self, pr, monkeypatch):
        self._patch(monkeypatch)
        assert V.verify_model(pr, "adder").exit_code == 0

    def test_skip_size_passes_without_sizing(self, pr, monkeypatch):
        self._patch(monkeypatch, feasible=False)  # would fail if sized
        assert V.verify_model(pr, "adder", skip_size=True).exit_code == 0


class TestRunBlockEquivGate:
    def _enable(self, monkeypatch, *, goldens=True, equiv=True, fail_open=False):
        import orchestrator.architecture.composition as comp
        import orchestrator.langgraph.gate_guard as gg
        import orchestrator.langgraph.rtl_model_equiv as rme
        monkeypatch.setattr(comp, "block_goldens_enabled", lambda: goldens)
        monkeypatch.setattr(rme, "rtl_model_equiv_enabled", lambda: equiv)
        monkeypatch.setattr(gg, "gate_fail_open_enabled", lambda: fail_open)

    def _set_equiv(self, monkeypatch, results):
        import orchestrator.langgraph.rtl_model_equiv as rme
        it = iter(results)
        monkeypatch.setattr(rme, "check_rtl_model_equivalence",
                            lambda *a, **k: next(it))

    def test_not_applicable_when_goldens_off(self, pr, monkeypatch):
        self._enable(monkeypatch, goldens=False)
        out = V.run_block_equiv_gate("adder", str(pr / "rtl.v"), pr, seed=1)
        assert out["ran"] is False

    def test_byte_exact_pass(self, pr, monkeypatch):
        self._enable(monkeypatch)
        self._set_equiv(monkeypatch, [
            {"passed": True, "skipped": False, "checked_vectors": 64},
        ])
        out = V.run_block_equiv_gate("adder", str(pr / "r.v"), pr, seed=1)
        assert out["ran"] and out["passed"] and out["checked_vectors"] == 64

    def test_divergence_fails(self, pr, monkeypatch):
        self._enable(monkeypatch)
        self._set_equiv(monkeypatch, [
            {"passed": False, "skipped": False, "reason": "byte 0 differs"},
        ])
        out = V.run_block_equiv_gate("adder", str(pr / "r.v"), pr, seed=1)
        assert out["ran"] and not out["passed"] and not out["skipped"]
        assert "byte 0" in out["reason"]
        assert out["prev_error_text"]

    def test_honest_skip_non_blocking(self, pr, monkeypatch):
        self._enable(monkeypatch)
        self._set_equiv(monkeypatch, [
            {"passed": False, "skipped": True, "reason": "non-AXIS interface"},
        ])
        out = V.run_block_equiv_gate("adder", str(pr / "r.v"), pr, seed=1)
        assert out["skipped"] and not out["failed_closed"]

    def test_harness_error_retries_then_fails_closed(self, pr, monkeypatch):
        self._enable(monkeypatch, fail_open=False)
        # both attempts return harness_error -> fail closed
        self._set_equiv(monkeypatch, [
            {"passed": False, "skipped": True, "harness_error": True,
             "reason": "build timeout"},
            {"passed": False, "skipped": True, "harness_error": True,
             "reason": "build timeout"},
        ])
        out = V.run_block_equiv_gate("adder", str(pr / "r.v"), pr, seed=1)
        assert out["failed_closed"] is True
        assert out["prev_error_text"]

    def test_harness_error_fail_open_stays_skip(self, pr, monkeypatch):
        self._enable(monkeypatch, fail_open=True)
        self._set_equiv(monkeypatch, [
            {"passed": False, "skipped": True, "harness_error": True, "reason": "x"},
            {"passed": False, "skipped": True, "harness_error": True, "reason": "x"},
        ])
        out = V.run_block_equiv_gate("adder", str(pr / "r.v"), pr, seed=1)
        assert out["failed_closed"] is False
        assert out["skipped"] is True

    def test_gate_exception_fails_closed(self, pr, monkeypatch):
        self._enable(monkeypatch, fail_open=False)
        import orchestrator.langgraph.rtl_model_equiv as rme

        def _boom(*a, **k):
            raise RuntimeError("verilator segfault")
        monkeypatch.setattr(rme, "check_rtl_model_equivalence", _boom)
        out = V.run_block_equiv_gate("adder", str(pr / "r.v"), pr, seed=1)
        assert out["failed_closed"] is True


class TestVerifyRtl:
    def _patch_helpers(self, monkeypatch, *, lint_clean=True, sim_passed=True,
                       timed_out=False):
        import orchestrator.langgraph.pipeline_helpers as ph
        monkeypatch.setattr(ph, "lint_rtl", lambda rp, b, a: {
            "clean": lint_clean, "errors": "" if lint_clean else "%Error x",
            "log_path": "/lint.log",
        })
        monkeypatch.setattr(ph, "run_simulation", lambda spec, rp, tb, a: {
            "passed": sim_passed, "log": "sim log", "log_path": "/sim.log",
            "tests_passed": 5, "tests_total": 5, "tests_failed": 0,
            "sim_timed_out": timed_out,
        })

    def test_rtl_missing(self, pr):
        r = V.verify_rtl(pr, {"name": "ghost"})
        assert r.exit_code == 1
        assert r.details["stage"] == "rtl"

    def test_lint_fail(self, pr, monkeypatch):
        spec = _make_rtl(pr, "adder")
        self._patch_helpers(monkeypatch, lint_clean=False)
        r = V.verify_rtl(pr, spec)
        assert r.exit_code == 1
        assert r.details["stage"] == "lint"

    def test_lint_only_pass(self, pr, monkeypatch):
        spec = _make_rtl(pr, "adder")
        self._patch_helpers(monkeypatch)
        r = V.verify_rtl(pr, spec, lint_only=True)
        assert r.exit_code == 0
        assert r.details["stage"] == "lint"

    def test_sim_timeout_is_infra(self, pr, monkeypatch):
        spec = _make_rtl(pr, "adder")
        self._patch_helpers(monkeypatch, sim_passed=False, timed_out=True)
        r = V.verify_rtl(pr, spec, no_equiv=True)
        assert r.exit_code == 3

    def test_sim_pass_no_equiv(self, pr, monkeypatch):
        spec = _make_rtl(pr, "adder")
        self._patch_helpers(monkeypatch)
        r = V.verify_rtl(pr, spec, no_equiv=True)
        assert r.exit_code == 0

    def test_sim_pass_equiv_fail(self, pr, monkeypatch):
        spec = _make_rtl(pr, "adder")
        self._patch_helpers(monkeypatch)
        monkeypatch.setattr(V, "run_block_equiv_gate", lambda *a, **k: {
            "ran": True, "passed": False, "skipped": False, "failed_closed": False,
            "reason": "diverged", "checked_vectors": 0, "prev_error_text": "x",
        })
        r = V.verify_rtl(pr, spec)
        assert r.exit_code == 1
        assert r.details["stage"] == "equiv"

    def test_records_to_scoreboard(self, pr, monkeypatch):
        spec = _make_rtl(pr, "adder")
        self._patch_helpers(monkeypatch)
        sb = Scoreboard(pr)
        V.verify_rtl(pr, spec, no_equiv=True, record_source="gate", scoreboard=sb)
        row = sb.latest_dv(block="adder", scope="rtl")[0]
        assert row["source"] == "gate"
        assert row["passed"] == 1
        assert row["tests_total"] == 5


class TestVerifySynth:
    def _patch(self, monkeypatch, *, generic=None, cell=None, ff_budget=None,
               ceiling=100000):
        import orchestrator.langgraph.ppa_check as pc
        monkeypatch.setattr(pc, "probe_synth_generic", lambda rp, t, timeout_s=300: generic)
        monkeypatch.setattr(pc, "probe_synth_cellcount",
                            lambda rp, t, timeout_s=300, cwd=None: cell)
        monkeypatch.setattr(pc, "parse_ff_budget", lambda txt: ff_budget)
        monkeypatch.setattr(pc, "max_cell_ceiling", lambda: ceiling)

    def test_yosys_absent_skips(self, pr, monkeypatch):
        spec = _make_rtl(pr, "adder")
        self._patch(monkeypatch, generic=None)
        assert V.verify_synth(pr, spec).exit_code == 4

    def test_not_elaborated_fails(self, pr, monkeypatch):
        spec = _make_rtl(pr, "adder")
        self._patch(monkeypatch, generic={"elaborated": False, "reason": "loop"})
        assert V.verify_synth(pr, spec).exit_code == 1

    def test_over_ff_budget_fails(self, pr, monkeypatch):
        spec = _make_rtl(pr, "adder")
        self._patch(monkeypatch,
                    generic={"elaborated": True, "logic_ff": 500, "mem_bits": 0},
                    cell={"elaborated": True, "cell_count": 10},
                    ff_budget=100)
        r = V.verify_synth(pr, spec)
        assert r.exit_code == 1
        assert "flip-flops" in r.verdict

    def test_within_budget_pass(self, pr, monkeypatch):
        spec = _make_rtl(pr, "adder")
        self._patch(monkeypatch,
                    generic={"elaborated": True, "logic_ff": 50, "mem_bits": 0},
                    cell={"elaborated": True, "cell_count": 200},
                    ff_budget=100)
        assert V.verify_synth(pr, spec).exit_code == 0

    def test_records_ppa(self, pr, monkeypatch):
        spec = _make_rtl(pr, "adder")
        self._patch(monkeypatch,
                    generic={"elaborated": True, "logic_ff": 50, "mem_bits": 0},
                    cell={"elaborated": True, "cell_count": 200}, ff_budget=100)
        sb = Scoreboard(pr)
        V.verify_synth(pr, spec, scoreboard=sb)
        row = sb.latest_ppa("adder")
        assert row["ff"] == 50 and row["ppa_ok"] == 1


class TestVerifyChip:
    def test_no_integration_result_skips(self, pr):
        assert V.verify_chip(pr).exit_code == 4

    def test_runs_integration_sim(self, pr, monkeypatch):
        import json as _json
        (pr / ".coresmith" / "integration_result.json").write_text(_json.dumps({
            "design_name": "codec", "top_rtl_path": str(pr / "chip_top.v"),
            "block_rtl_paths": {}, "tb_path": str(pr / "tb.py"),
        }))
        (pr / "chip_top.v").write_text("module chip_top(); endmodule")
        (pr / "tb.py").write_text("# tb")
        import orchestrator.langgraph.integration_helpers as ih
        monkeypatch.setattr(ih, "run_integration_simulation",
                            lambda *a, **k: {"passed": True, "log": "ok",
                                             "log_path": "/i.log"})
        sb = Scoreboard(pr)
        r = V.verify_chip(pr, scoreboard=sb)
        assert r.exit_code == 0
        assert sb.latest_dv(block="codec", scope="chip")[0]["passed"] == 1
