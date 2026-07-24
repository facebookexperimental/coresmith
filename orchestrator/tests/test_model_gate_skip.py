# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Audit F2: the model-integration gate must be tri-state (PASS / FAIL / SKIP).

A no-op (no models, no chip model, no stimulus, reference import failure) must
never surface as PASS -- the reference codec encoder CLI printed
"composed model == golden byte-exact" right after
"reference import failed: No module named 'reference_codec_vectors' -- no-op".
"""

from __future__ import annotations

import pytest

from orchestrator.architecture import model_integration as mi
from orchestrator.harness import verify as V


@pytest.fixture()
def goldens_on(monkeypatch):
    monkeypatch.setenv("CORESMITH_BLOCK_GOLDENS", "1")


def _mk_models(root, chip_model: bool = True):
    d = root / "arch" / "block_models"
    d.mkdir(parents=True)
    (d / "blk.py").write_text("def blk(x):\n    return x\n")
    if chip_model:
        (d / "_chip_model.py").write_text(
            "def simulate(stimulus):\n    return stimulus\n")
    return d


class TestSiblingImports:
    def test_module_dir_on_syspath_during_import(self, tmp_path):
        """A reference/stimulus file can import a sibling module by plain name
        (the 'No module named reference_codec_vectors' class)."""
        (tmp_path / "helper_sibling_mod.py").write_text("VALUE = 42\n")
        main = tmp_path / "main_golden.py"
        main.write_text(
            "import helper_sibling_mod\n"
            "def run(x):\n    return helper_sibling_mod.VALUE\n")
        mod = mi._import_module_from_path(main, "_t_sibling_import")
        assert mod.run(0) == 42


class TestGateSkipInfo:
    def test_flag_off_marks_skip(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CORESMITH_BLOCK_GOLDENS", "0")
        info: dict = {}
        viols = mi.run_model_integration_gate(str(tmp_path), result_info=info)
        assert viols == []
        assert info["skipped"] is True

    def test_no_models_marks_skip(self, tmp_path, goldens_on):
        info: dict = {}
        viols = mi.run_model_integration_gate(str(tmp_path), result_info=info)
        assert viols == []
        assert info["skipped"] is True
        assert "block models" in info["reason"]

    def test_no_chip_model_marks_skip(self, tmp_path, goldens_on):
        _mk_models(tmp_path, chip_model=False)
        info: dict = {}
        viols = mi.run_model_integration_gate(str(tmp_path), result_info=info)
        assert viols == []
        assert info["skipped"] is True
        assert "_chip_model" in info["reason"]

    def test_reference_import_failure_is_violation(
        self, tmp_path, goldens_on, monkeypatch
    ):
        """A reference that RESOLVES but cannot be IMPORTED is a broken oracle
        -> violation, never an empty no-op."""
        _mk_models(tmp_path)
        ref = tmp_path / "inputs"
        ref.mkdir()
        (ref / "broken_golden.py").write_text(
            "import module_that_does_not_exist_xyz\n")
        monkeypatch.setenv("CORESMITH_SOURCE_ROOT",
                           str(ref / "broken_golden.py"))
        info: dict = {}
        viols = mi.run_model_integration_gate(str(tmp_path), result_info=info)
        assert viols, "import failure must be a violation, not a no-op"
        assert viols[0]["criterion"] == "reference_import_failed"
        assert info["skipped"] is False


class TestVerifyChipModelSkip:
    def test_no_op_gate_is_skip_not_pass(self, tmp_path, goldens_on):
        """The CLI harness maps a no-op gate to SKIP (exit 4), never PASS."""
        res = V.verify_chip_model(tmp_path)
        assert res.passed is False
        assert res.skipped is True
        assert res.exit_code == 4
        assert "SKIP" in res.verdict
