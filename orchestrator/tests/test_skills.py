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

    def test_axi_stream_accept_side_ack_and_drop_rule(self):
        # Rule 5 (rerun9): slave/accept-side tready asserted on a cycle an
        # internal drain/flush branch wins the if/elif -> upstream pops, slave
        # drops = ack-and-drop. Distinct from rule 4 (master/output-side skid).
        # Normalize whitespace + strip markdown emphasis so anchors are robust
        # to the file's hard line-wraps and **bold** markers.
        raw = load_skill("axi_stream").lower().replace("*", "")
        body = " ".join(raw.split())
        for anchor in (
            "ack-and-drop",
            "drain",
            "priority chain",
            "acknowledged and silently dropped",
            "a token is consumed iff it is latched",
        ):
            assert anchor in body, (
                f"axi_stream skill is missing accept-side handshake anchor: {anchor!r}"
            )

    def test_serialization_contract_cadence_rule(self):
        # rerun10: per-MB vs per-sub-block syntax cadence + the omitted
        # per-sub-block pred_mode field. Normalize whitespace/markdown so the
        # anchors survive the file's hard line-wraps.
        raw = load_skill("serialization_contract").lower().replace("*", "")
        body = " ".join(raw.split())
        assert body, "serialization_contract.md must exist on disk"
        for anchor in (
            "per-mb syntax vs per-subblock syntax",
            "nesting cadence",
            "pred_mode",
        ):
            assert anchor in body, (
                f"serialization_contract skill is missing cadence anchor: {anchor!r}"
            )

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

    def test_memory_macro_vs_flops_present(self):
        body = load_skill("memory_macro_vs_flops")
        assert body, "memory_macro_vs_flops.md must exist on disk"
        # Anchor keywords / triggers the uArch agent relies on so a
        # rewrite that drops the operative guidance is caught.
        for anchor in (
            "SRAM macro",
            "flip-flop",
            "sky130_sram_1kbyte_1rw1r_32x256_8",
            "sky130_sram_2kbyte_1rw1r_32x512_8",
            "sky130_sram_1kbyte_1rw1r_8x1024_8",
            "1rw1r",
            "max_onchip_sram_kb",
            "max_macro_count",
            "1-cycle",
            "Quick checklist",
        ):
            assert anchor.lower() in body.lower(), (
                f"memory_macro_vs_flops skill is missing anchor: {anchor!r}"
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

    def test_uarch_spec_generator_loads_memory_macro_vs_flops(self):
        # The SRAM-macro-vs-flops skill must reach the uArch agent so
        # every spec author chooses storage type explicitly.
        from orchestrator.langchain.agents import uarch_spec_generator
        prompt = uarch_spec_generator.SYSTEM_PROMPT
        assert "sky130_sram_1kbyte_1rw1r_32x256_8" in prompt
        assert "SRAM macro" in prompt

    def test_ers_doc_loads_memory_macro_vs_flops(self):
        # The ERS specialist binds memory implementation for the whole
        # pipeline; the SRAM-vs-flops skill must reach it so the ERS does
        # not blanket-mandate flip-flop scratchpads.
        from orchestrator.architecture.specialists import ers_doc
        assert ers_doc._SKILLS_TEXT
        assert "sky130_sram_1kbyte_1rw1r_32x256_8" in ers_doc._SKILLS_TEXT
        assert "SRAM macro" in ers_doc._SKILLS_TEXT

    def test_uarch_spec_generator_requires_storage_budget(self):
        # The uArch spec must emit a quantified flip-flop + SRAM budget so
        # the synthesis step has numbers to check actual results against
        # (PPA-friendliness: catch memory-as-flops before it ships).
        from orchestrator.langchain.agents import uarch_spec_generator
        prompt = uarch_spec_generator.SYSTEM_PROMPT
        assert "flip_flop_budget" in prompt
        assert "sram_budget" in prompt

    def test_synth_fixer_reviews_memory_as_flops_against_budget(self):
        # The synthesis fixer's checklist must flag memory-as-flops (a PPA
        # failure that synthesizes cleanly) and cross-check the synthesized
        # FF count against the uArch flip_flop_budget.
        from pathlib import Path

        import orchestrator.langgraph.pipeline_helpers as ph
        prompt = (
            Path(ph.__file__).resolve().parent.parent
            / "langchain" / "prompts" / "synth_fixer.md"
        ).read_text()
        assert "Memory-as-flops" in prompt
        assert "flip_flop_budget" in prompt

    def test_interface_definition_loads_arithmetic_precision(self):
        from orchestrator.architecture.specialists import (
            interface_definition,
        )
        assert "Bit-width derivation" in interface_definition.SYSTEM_PROMPT

    def test_integration_lead_loads_arithmetic_precision(self):
        from orchestrator.langchain.agents import integration_lead
        assert "Bit-width derivation" in integration_lead.SYSTEM_PROMPT

    def test_block_diagram_axi_stream_prd_carveout(self):
        # The generic "every datapath block must use AXI-Stream" rule must not
        # make the reviewer escalate when the PRD explicitly specifies a
        # dedicated-pin/combinational interface -- honor the PRD contract and
        # record a waiver instead of raising a blocking question.
        from orchestrator.architecture.specialists import block_diagram
        prompt = block_diagram.SYSTEM_PROMPT
        assert "PRD-contract carve-out" in prompt
        assert "do NOT escalate" in prompt
        assert "dedicated pins" in prompt
        assert "Record the waiver in the diagram notes" in prompt
