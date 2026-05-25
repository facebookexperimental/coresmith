# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for `orchestrator.langchain.prompts.skills`.

Covers:
- Loader returns markdown body for known skills.
- Missing skill ids degrade to empty.
- `load_skills` concatenates and skips missing.
- Catalog includes the skills that downstream agents pull in
  (axi_stream, srdy_drdy, arithmetic_precision).
- Each shipped skill body contains the expected anchor keywords so a
  rename to a structurally-different document is caught.
"""
from __future__ import annotations

from orchestrator.langchain.prompts.skills import (
    load_skill,
    load_skills,
)


class TestLoadSkill:
    def test_missing_returns_empty(self):
        assert load_skill("definitely_not_a_skill_xyz") == ""

    def test_axi_stream_present(self):
        body = load_skill("axi_stream")
        assert body, "axi_stream.md must exist on disk"
        # Should reference the handshake protocol it's about.
        assert "axi" in body.lower()

    def test_srdy_drdy_present(self):
        body = load_skill("srdy_drdy")
        assert body, "srdy_drdy.md must exist on disk"
        assert "srdy" in body.lower() and "drdy" in body.lower()

    def test_arithmetic_precision_present(self):
        body = load_skill("arithmetic_precision")
        assert body, "arithmetic_precision.md must exist on disk"
        # Anchor keywords the agent prompts rely on so a rewrite that
        # drops the operative checklist is caught.
        for anchor in (
            "Bit-width derivation",
            "Q-format",
            "saturation",
            "wraparound",
            "Quick checklist",
        ):
            assert anchor.lower() in body.lower(), (
                f"arithmetic_precision skill is missing anchor: {anchor!r}"
            )


class TestLoadSkills:
    def test_concatenates_present_skills_with_separator(self):
        out = load_skills("axi_stream", "srdy_drdy", "arithmetic_precision")
        assert "---" in out  # default separator embeds the markdown rule
        # All three skill bodies must appear.
        assert "axi" in out.lower()
        assert "srdy" in out.lower()
        assert "Bit-width derivation".lower() in out.lower()

    def test_missing_is_silently_skipped(self):
        # The loader must not raise if an upstream agent references a
        # skill the repo doesn't ship yet.
        out = load_skills("axi_stream", "nope_not_real", "srdy_drdy")
        assert out, "expected the two real skills to come through"
        assert "axi" in out.lower() and "srdy" in out.lower()


class TestSkillsWiredIntoAgents:
    """The arithmetic-precision skill is useless if downstream agents
    don't include it in their system prompts. These tests pin the
    wiring so a refactor that drops the load can't go unnoticed."""

    def test_uarch_spec_generator_loads_arithmetic_precision(self):
        from orchestrator.langchain.agents import uarch_spec_generator
        assert "Bit-width derivation" in uarch_spec_generator.SYSTEM_PROMPT

    def test_interface_definition_loads_arithmetic_precision(self):
        from orchestrator.architecture.specialists import (
            interface_definition,
        )
        assert "Bit-width derivation" in interface_definition.SYSTEM_PROMPT

    def test_integration_lead_loads_arithmetic_precision(self):
        from orchestrator.langchain.agents import integration_lead
        assert "Bit-width derivation" in integration_lead.SYSTEM_PROMPT
