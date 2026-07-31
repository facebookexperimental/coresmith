# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""A probe that ran no discriminating check has not PASSED anything.

Every block of the first hands-off run logged, in green::

    [BLOCK-MODEL] <block>: golden-feasibility OK (skipped)

-- an OK whose own parenthetical said the only check capable of returning the
other answer had not run. The stored artifact said the same thing:
``passed: true`` next to ``slice_reachability: {verdict: "skipped"}``.

What the probe CHECKS is unchanged here. What it CLAIMS is not.
"""
from __future__ import annotations

import json
from pathlib import Path

from orchestrator.architecture.model_integration import (
    _golden_feasibility_verdict,
    _persist_golden_feasibility,
    check_golden_feasibility,
)

#: Verbatim from exp-raster-auto-20260731 (.coresmith/blocks/zbuffer_sram/
#: golden_feasibility.json) -- the shape 8 of 8 blocks produced.
OBSERVED = {
    "block": "zbuffer_sram", "ran": True, "passed": True, "skipped": False,
    "reason": "",
    "checks": {
        "model_valid": True,
        "golden_resolvable": True,
        "slice_reachability": {
            "verdict": "skipped",
            "reason": "no confident slice (needs Phase-2 generalized mapping)",
            "calls": {},
        },
    },
}


class TestTheStoredVerdict:

    def test_the_observed_artifact_is_not_run_not_pass(self):
        r = _golden_feasibility_verdict(json.loads(json.dumps(OBSERVED)))
        assert r["verdict"] == "not_run"
        assert r["discriminating"] is False
        # actionable, in the [GATE-SIM] NOT RUN house style: it names the check
        # that did not conclude AND why
        assert "slice reachability" in r["not_run_reason"]
        assert "no confident slice" in r["not_run_reason"]
        # `passed` keeps its meaning -- the advisory gate's semantics are
        # untouched; only the claim got honest.
        assert r["passed"] is True

    def test_a_reachable_slice_is_a_real_pass(self):
        d = json.loads(json.dumps(OBSERVED))
        d["checks"]["slice_reachability"] = {
            "verdict": "reachable", "reason": "2 slice fn(s) produced output",
            "calls": {"f": {"n": 3, "produced": True}},
        }
        r = _golden_feasibility_verdict(d)
        assert r["verdict"] == "pass" and r["discriminating"] is True
        assert "not_run_reason" not in r

    def test_a_failure_is_still_a_failure(self):
        d = json.loads(json.dumps(OBSERVED))
        d["passed"] = False
        d["reason"] = "model invalid: import failed"
        assert _golden_feasibility_verdict(d)["verdict"] == "fail"

    def test_a_probe_that_never_ran_is_not_run(self):
        d = {"block": "b", "ran": False, "passed": True, "skipped": True,
             "reason": "block goldens disabled", "checks": {}}
        r = _golden_feasibility_verdict(d)
        assert r["verdict"] == "not_run" and r["discriminating"] is False

    def test_the_verdict_is_stamped_on_the_persisted_artifact(self, tmp_path):
        """Every return path in check_golden_feasibility goes through the
        persister, so stamping there is what makes the on-disk artifact
        honest for all of them."""
        _persist_golden_feasibility(str(tmp_path), "zbuffer_sram",
                                    json.loads(json.dumps(OBSERVED)))
        on_disk = json.loads(
            (tmp_path / ".coresmith" / "blocks" / "zbuffer_sram"
             / "golden_feasibility.json").read_text())
        assert on_disk["verdict"] == "not_run"
        assert on_disk["discriminating"] is False
        assert on_disk["not_run_reason"]


class TestTheProductionEntryPoint:

    def test_disabled_block_goldens_report_not_run(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_BLOCK_GOLDENS", "0")
        res = check_golden_feasibility(str(tmp_path), "b")
        assert res["ran"] is False and res["verdict"] == "not_run"

    def test_a_missing_model_is_a_failure_not_a_not_run(self, tmp_path,
                                                        monkeypatch):
        monkeypatch.setenv("CORESMITH_BLOCK_GOLDENS", "1")
        res = check_golden_feasibility(str(tmp_path), "b")
        assert res["ran"] is True and res["passed"] is False
        assert res["verdict"] == "fail"

    def test_a_valid_model_with_no_reachable_slice_is_not_run(self, tmp_path,
                                                              monkeypatch):
        """The observed case, driven through the real entry point: the model
        imports and exposes its Elaboratable, and nothing exercised its math."""
        monkeypatch.setenv("CORESMITH_BLOCK_GOLDENS", "1")
        from orchestrator.architecture import composition as _composition
        mdir = Path(tmp_path) / "arch" / _composition.BLOCK_MODELS_DIRNAME
        mdir.mkdir(parents=True)
        (mdir / "b.py").write_text(
            "from amaranth import Elaboratable, Module, Signal\n"
            "\n"
            "class b(Elaboratable):\n"
            "    def __init__(self, wb_clk_i, out):\n"
            "        self.wb_clk_i = wb_clk_i\n"
            "        self.out = out\n"
            "\n"
            "    def elaborate(self, platform):\n"
            "        m = Module()\n"
            "        m.d.sync += self.out.eq(Signal(4))\n"
            "        return m\n"
        )
        res = check_golden_feasibility(str(tmp_path), "b")
        assert res["ran"] is True and res["passed"] is True
        assert res["verdict"] == "not_run", res
        assert res["discriminating"] is False
        assert res["not_run_reason"]
