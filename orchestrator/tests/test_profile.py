# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for A-Fix 1: profile-based flag defaults (orchestrator/profile.py).

Covers resolve/apply/reset semantics, the canonical truthy parsers, and the
9 gate-enable helpers under strict / legacy / explicit-env conditions.
"""

from __future__ import annotations

import os

import pytest

from orchestrator import profile

# The 9 rewired enable-helpers.
from orchestrator.architecture.composition import block_goldens_enabled
from orchestrator.architecture.fidelity import fidelity_gate_enabled
from orchestrator.langgraph.ppa_check import (
    ppa_gate_enabled,
    synth_cell_gate_enabled,
    logic_depth_gate_enabled,
)
from orchestrator.langgraph.pdk_characterize import stage_enabled as pdk_stage_enabled
from orchestrator.langgraph.latency_audit import audit_enabled as latency_audit_enabled
from orchestrator.langgraph.pipeline_helpers import _rtl_from_hw_golden_enabled
from orchestrator.langgraph.rtl_model_equiv import rtl_model_equiv_enabled


# (helper, env_flag, legacy_default, strict_seeds_on)
HELPERS = [
    (block_goldens_enabled, "CORESMITH_BLOCK_GOLDENS", False, True),
    (ppa_gate_enabled, "CORESMITH_PPA_GATE", False, True),
    (fidelity_gate_enabled, "CORESMITH_FIDELITY_GATE", False, True),
    (pdk_stage_enabled, "CORESMITH_PDK_CHAR", False, True),
    (latency_audit_enabled, "CORESMITH_LATENCY_AUDIT", False, True),
    (_rtl_from_hw_golden_enabled, "CORESMITH_RTL_FROM_HW_GOLDEN", False, True),
    (synth_cell_gate_enabled, "CORESMITH_SYNTH_CELL_GATE", True, True),
    (logic_depth_gate_enabled, "CORESMITH_LOGIC_DEPTH_GATE", True, True),
    (rtl_model_equiv_enabled, "CORESMITH_RTL_MODEL_EQUIV", True, True),
]

_ALL_FLAGS = [f for _, f, _, _ in HELPERS] + list(profile.STRICT_DEFAULTS)


def _clean_env(monkeypatch):
    for flag in set(_ALL_FLAGS):
        monkeypatch.delenv(flag, raising=False)
    profile.reset()


# ---------------------------------------------------------------------------
# resolve_profile
# ---------------------------------------------------------------------------
class TestResolveProfile:
    def test_default_is_strict(self, monkeypatch):
        monkeypatch.delenv("CORESMITH_PROFILE", raising=False)
        assert profile.resolve_profile() == "strict"

    def test_legacy_selected(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_PROFILE", "legacy")
        assert profile.resolve_profile() == "legacy"

    def test_strict_selected(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_PROFILE", "STRICT")
        assert profile.resolve_profile() == "strict"

    def test_unknown_falls_back_to_strict(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_PROFILE", "bogus")
        assert profile.resolve_profile() == "strict"


# ---------------------------------------------------------------------------
# apply / reset
# ---------------------------------------------------------------------------
class TestApply:
    def test_strict_seeds_all_defaults(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_PROFILE", "strict")
        _clean_env(monkeypatch)
        seeded = profile.apply()
        assert set(seeded) == set(profile.STRICT_DEFAULTS)
        for key, value in profile.STRICT_DEFAULTS.items():
            assert os.environ[key] == value

    def test_legacy_seeds_nothing(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_PROFILE", "legacy")
        _clean_env(monkeypatch)
        assert profile.apply() == []
        for key in profile.STRICT_DEFAULTS:
            assert key not in os.environ

    def test_explicit_env_wins_over_profile(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_PROFILE", "strict")
        _clean_env(monkeypatch)
        monkeypatch.setenv("CORESMITH_PPA_GATE", "0")  # explicit OFF
        seeded = profile.apply()
        assert "CORESMITH_PPA_GATE" not in seeded
        assert os.environ["CORESMITH_PPA_GATE"] == "0"  # not overwritten

    def test_apply_is_idempotent(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_PROFILE", "strict")
        _clean_env(monkeypatch)
        first = profile.apply()
        second = profile.apply()  # already applied -> no re-seed
        assert second == first

    def test_force_reapplies(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_PROFILE", "strict")
        _clean_env(monkeypatch)
        profile.apply()
        # Remove a seeded key, then force re-apply re-seeds it.
        monkeypatch.delenv("CORESMITH_PPA_GATE", raising=False)
        reseeded = profile.apply(force=True)
        assert "CORESMITH_PPA_GATE" in reseeded

    def test_reset_removes_seeded_keys(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_PROFILE", "strict")
        _clean_env(monkeypatch)
        profile.apply()
        assert os.environ.get("CORESMITH_PPA_GATE") == "1"
        profile.reset()
        assert "CORESMITH_PPA_GATE" not in os.environ

    def test_reset_does_not_remove_explicit_key(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_PROFILE", "strict")
        _clean_env(monkeypatch)
        monkeypatch.setenv("CORESMITH_BLOCK_GOLDENS", "1")  # explicit
        profile.apply()
        profile.reset()
        # Explicit var survives reset (it was never "seeded" by apply).
        assert os.environ.get("CORESMITH_BLOCK_GOLDENS") == "1"


# ---------------------------------------------------------------------------
# canonical parsers
# ---------------------------------------------------------------------------
class TestFlagParsers:
    @pytest.mark.parametrize("val,expected", [
        ("1", True), ("true", True), ("YES", True), ("on", True),
        ("0", False), ("false", False), ("no", False), ("off", False), ("", False),
    ])
    def test_flag_enabled_tokens(self, monkeypatch, val, expected):
        monkeypatch.setenv("CORESMITH_X", val)
        assert profile.flag_enabled("CORESMITH_X") is expected

    def test_flag_enabled_unset_uses_default(self, monkeypatch):
        monkeypatch.delenv("CORESMITH_X", raising=False)
        assert profile.flag_enabled("CORESMITH_X", default=False) is False
        assert profile.flag_enabled("CORESMITH_X", default=True) is True

    def test_flag_enabled_unrecognized_uses_default(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_X", "maybe")
        assert profile.flag_enabled("CORESMITH_X", default=True) is True
        assert profile.flag_enabled("CORESMITH_X", default=False) is False

    def test_flag_disabled_is_inverse(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_X", "1")
        assert profile.flag_disabled("CORESMITH_X") is False
        monkeypatch.setenv("CORESMITH_X", "0")
        assert profile.flag_disabled("CORESMITH_X") is True


# ---------------------------------------------------------------------------
# the 9 helpers x profile conditions
# ---------------------------------------------------------------------------
class TestHelpersUnderProfiles:
    @pytest.mark.parametrize("helper,flag,legacy_default,strict_on", HELPERS)
    def test_legacy_unset_is_code_default(self, monkeypatch, helper, flag, legacy_default, strict_on):
        monkeypatch.setenv("CORESMITH_PROFILE", "legacy")
        _clean_env(monkeypatch)
        assert helper() is legacy_default

    @pytest.mark.parametrize("helper,flag,legacy_default,strict_on", HELPERS)
    def test_strict_unset_matches_strict_posture(self, monkeypatch, helper, flag, legacy_default, strict_on):
        monkeypatch.setenv("CORESMITH_PROFILE", "strict")
        _clean_env(monkeypatch)
        assert helper() is strict_on

    @pytest.mark.parametrize("helper,flag,legacy_default,strict_on", HELPERS)
    def test_explicit_off_wins_under_strict(self, monkeypatch, helper, flag, legacy_default, strict_on):
        monkeypatch.setenv("CORESMITH_PROFILE", "strict")
        _clean_env(monkeypatch)
        monkeypatch.setenv(flag, "0")
        assert helper() is False

    @pytest.mark.parametrize("helper,flag,legacy_default,strict_on", HELPERS)
    def test_explicit_on_wins_under_legacy(self, monkeypatch, helper, flag, legacy_default, strict_on):
        monkeypatch.setenv("CORESMITH_PROFILE", "legacy")
        _clean_env(monkeypatch)
        monkeypatch.setenv(flag, "1")
        assert helper() is True
