# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Spec-only oracle re-baseline: authorized pre-RTL ERS/FRD triage edits become
the new baseline, but the golden reference + inputs/ stimulus can NEVER be
re-baselined -- editing those to make RTL 'match' stays ORACLE_TAMPER."""

from pathlib import Path

from orchestrator.state_store.trust import (
    write_oracle_manifest,
    check_oracle_manifest,
    rebaseline_oracle_specs,
)


def _mk(root: Path) -> None:
    (root / ".coresmith").mkdir()
    (root / "arch").mkdir()
    (root / "inputs").mkdir()
    (root / "inputs" / "golden.py").write_text("def f(): return 1\n")
    (root / "arch" / "ers_spec.md").write_text("ERS v1\n")
    (root / "arch" / "frd_spec.md").write_text("FRD v1\n")


def test_authorized_spec_edit_rebaselines_and_clears(tmp_path):
    _mk(tmp_path)
    write_oracle_manifest(tmp_path)
    assert check_oracle_manifest(tmp_path)["ok"]
    (tmp_path / "arch" / "ers_spec.md").write_text("ERS v2 (REWIND opcode)\n")
    # PR#12 finding #9: an authorized spec-ONLY edit now AUTO-re-baselines by
    # default (was ORACLE_TAMPER, which cascaded INFRASTRUCTURE_ERROR across
    # every re-processed block). ok stays True, the spec is reported, and a
    # second check is clean. The manual rebaseline_oracle_specs() call is still
    # available and idempotent.
    res = check_oracle_manifest(tmp_path)
    assert res["ok"]
    assert "arch/ers_spec.md" in res.get("spec_rebaselined", [])
    assert check_oracle_manifest(tmp_path)["ok"]
    assert rebaseline_oracle_specs(tmp_path) == []  # already re-baselined


def test_strict_env_keeps_authorized_spec_edit_a_fail(tmp_path, monkeypatch):
    # CORESMITH_STRICT_ORACLE_MANIFEST=1 restores the pre-#9 strict contract:
    # any drift (even a spec-only edit) is a hard fail.
    _mk(tmp_path)
    write_oracle_manifest(tmp_path)
    (tmp_path / "arch" / "ers_spec.md").write_text("ERS v2\n")
    monkeypatch.setenv("CORESMITH_STRICT_ORACLE_MANIFEST", "1")
    assert not check_oracle_manifest(tmp_path)["ok"]


def test_golden_tamper_never_rebaselined(tmp_path):
    _mk(tmp_path)
    write_oracle_manifest(tmp_path)
    (tmp_path / "inputs" / "golden.py").write_text("def f(): return 999\n")
    # a rebaseline call must ignore the golden entirely...
    assert rebaseline_oracle_specs(tmp_path) == []
    res = check_oracle_manifest(tmp_path)
    assert not res["ok"] and "inputs/golden.py" in res["changed"]


def test_spec_rebaseline_is_surgical(tmp_path):
    _mk(tmp_path)
    write_oracle_manifest(tmp_path)
    (tmp_path / "inputs" / "golden.py").write_text("def f(): return 999\n")
    (tmp_path / "arch" / "frd_spec.md").write_text("FRD v2\n")
    assert rebaseline_oracle_specs(tmp_path) == ["arch/frd_spec.md"]
    res = check_oracle_manifest(tmp_path)
    assert not res["ok"]
    assert "inputs/golden.py" in res["changed"]
    assert "arch/frd_spec.md" not in res["changed"]


def test_no_manifest_is_noop(tmp_path):
    (tmp_path / ".coresmith").mkdir()
    assert rebaseline_oracle_specs(tmp_path) == []
