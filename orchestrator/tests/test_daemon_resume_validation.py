# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the daemon `/run/resume` action validation (rung1 fix).

The daemon must reject a resume action the parked interrupt does not support
(returning HTTP 400 + the allowed list) instead of silently forwarding it into
the graph -- e.g. an ``approve`` sent to a DV-failure interrupt whose only stop
action is ``abort``. The pure decision helper ``_resume_action_error`` is tested
directly (the daemon has no HTTP test infra; this mirrors
``test_daemon_arch_warning.py`` importing a server helper)."""
from __future__ import annotations

from orchestrator.daemon.server import _resume_action_error, _resume_tick_or_park

# (block_name, supported_actions) tuples as the handler builds from intr.value.
_DV_FAIL = [("", ["retry", "fix_rtl", "fix_tb", "abort"])]
_SPEC = [("adder", ["approve", "revise"])]


class TestResumeActionError:
    def test_unsupported_action_rejected_with_allowed_list(self):
        err = _resume_action_error("approve", None, _DV_FAIL)
        assert err is not None
        block_name, bad_action, allowed = err
        assert bad_action == "approve"
        assert allowed == ["retry", "fix_rtl", "fix_tb", "abort"]

    def test_supported_action_accepted(self):
        assert _resume_action_error("abort", None, _DV_FAIL) is None
        assert _resume_action_error("approve", None, _SPEC) is None

    def test_empty_supported_actions_imposes_no_constraint(self):
        # Back-compat: an interrupt that declares no supported_actions accepts
        # anything (not every interrupt enumerates them).
        assert _resume_action_error("anything", None, [("", [])]) is None
        assert _resume_action_error("weird", None, []) is None

    def test_block_actions_effective_action_validated(self):
        # A per-block action for the interrupt's block is the effective action.
        interrupts = [("adder", ["retry", "skip", "abort"])]
        # Valid per-block action -> ok even if the default action is invalid.
        assert _resume_action_error(
            "approve", {"adder": "retry"}, interrupts) is None
        # Invalid per-block action -> rejected, reporting the effective action.
        err = _resume_action_error("retry", {"adder": "approve"}, interrupts)
        assert err is not None
        assert err[0] == "adder"
        assert err[1] == "approve"

    def test_default_action_applies_when_block_not_in_block_actions(self):
        interrupts = [("adder", ["retry", "abort"])]
        # block_actions targets a DIFFERENT block -> default action applies here.
        err = _resume_action_error("approve", {"other": "retry"}, interrupts)
        assert err is not None
        assert err[1] == "approve"

    def test_multi_interrupt_first_offender_reported(self):
        interrupts = [
            ("a", ["approve", "revise"]),        # accepts approve
            ("b", ["retry", "abort"]),           # rejects approve
        ]
        err = _resume_action_error("approve", None, interrupts)
        assert err is not None
        assert err[0] == "b"
        assert err[2] == ["retry", "abort"]


class TestResumeTickOrPark:
    """rung3-fixes-1 (defect 3): /run/resume gains a plain-tick path. When there
    is NO pending interrupt but the checkpoint still has next nodes, resume ticks
    the graph forward (cmd=None) instead of 409'ing -- matching the architecture
    endpoint's tick semantics. A PARKED run never ticks: it needs a real action.
    """

    def test_parked_run_requires_real_action(self):
        # A pending interrupt wins regardless of next nodes -> "resume" (the
        # supported_actions validation then applies). Parked never ticks.
        assert _resume_tick_or_park(True, True) == "resume"
        assert _resume_tick_or_park(True, False) == "resume"

    def test_not_parked_with_next_nodes_ticks(self):
        # Stranded/paused run: no interrupt, but the graph has next nodes ->
        # plain tick forward.
        assert _resume_tick_or_park(False, True) == "tick"

    def test_nothing_to_do_is_none(self):
        # No interrupt and no next nodes -> nothing to resume (HTTP 409).
        assert _resume_tick_or_park(False, False) == "none"
