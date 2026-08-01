# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The contract is the naming authority; skills are selected, not dumped.

Three engine changes, tested through the production prompt constructors:

* **Task A** -- `port_naming` is inline in every authoring agent, and the RTL
  prompt carries the block's AUTHORITATIVE PORT NAMES table derived from the
  frozen contract by the SAME machinery the conformance gate uses.
* **Task B** -- reference skills are chosen per call from the block's evidence;
  what is not inlined is manifested with an absolute path to read.
* **Task C** -- a pre-simulation gate failure is diagnosed as `conformance`,
  with no waveform hunt.
"""
from __future__ import annotations

import json

from orchestrator.langchain.prompts.skills import (
    UARCH_SKILL_CANDIDATES,
    MissingSkillError,
    build_skill_section,
    load_skill,
    select_skills,
    skill_manifest,
    skill_path,
)


def _anchor(skill_id: str, n: int = 300) -> str:
    """A distinctive slice of a skill's BODY, to prove presence/absence."""
    body = load_skill(skill_id)
    assert body, f"{skill_id}.md must exist on disk"
    # Skip the title line; take a run of real prose.
    return body[len(body.splitlines()[0]):][:n].strip()


# ---------------------------------------------------------------------------
# Task B1 -- the deterministic classifier
# ---------------------------------------------------------------------------

class TestSelectSkillsEvidenceClasses:
    def test_memory_shaped_block_selects_memory_macro(self):
        sel = select_skills(
            {"name": "pixel_line_buffer",
             "description": "1024x32 line buffer for the scan stage"},
            None, None,
        )
        assert "memory_macro_vs_flops" in sel

    def test_memory_map_artifact_selects_memory_macro(self):
        sel = select_skills({"name": "b"}, None, {"memory_map": {"peripherals": []}})
        assert "memory_macro_vs_flops" in sel

    def test_axi_labels_in_contract_select_axi_stream(self):
        contracts = {"edges": [{
            "producer_block": "a", "consumer_block": "b",
            "producer_port": "out_stream", "consumer_port": "in_stream",
            "handshake_protocol": "axi_stream",
            "fields": [{"name": "tdata", "width": 32}],
        }]}
        sel = select_skills({"name": "a"}, contracts, None)
        assert "axi_stream" in sel

    def test_srdy_drdy_labels_select_srdy_skill(self):
        contracts = {"edges": [{
            "producer_block": "a", "consumer_block": "b",
            "producer_port": "coeff", "consumer_port": "coeff",
            "handshake_protocol": "srdy_drdy",
            "fields": [{"name": "data", "width": 8}],
        }]}
        assert "srdy_drdy" in select_skills({"name": "a"}, contracts, None)

    def test_serialization_evidence_selects_both_packing_skills(self):
        sel = select_skills(
            {"name": "stream_packer",
             "description": "packs symbols into the output bitstream"},
            None, None,
        )
        assert "serialization_contract" in sel
        assert "buffer_stride_contract" in sel, (
            "a packed buffer is read back by slot/stride -- the two travel "
            "together"
        )

    def test_pipeline_hint_selects_pipeline_contract(self):
        sel = select_skills(
            {"name": "b", "description": "four-stage pipelined datapath"},
            None, None)
        assert "pipeline_contract" in sel

    def test_throughput_budget_selects_throughput_contract(self):
        sel = select_skills({"name": "b", "perf": {"cycles_per_op": 4}}, None, None)
        assert "throughput_budget_contract" in sel

    def test_control_pulse_fields_select_control_pulse_skill(self):
        contracts = {"edges": [{
            "producer_block": "ctl", "consumer_block": "core",
            "producer_port": "job", "consumer_port": "job",
            "fields": [{"name": "start", "width": 1},
                       {"name": "done", "width": 1}],
        }]}
        assert "control_pulse_handshake" in select_skills(
            {"name": "ctl"}, contracts, None)

    def test_arithmetic_ops_in_model_select_arithmetic_precision(self):
        sel = select_skills(
            {"name": "b", "description": "widget",
             "model_source": "def f(a, b):\n    return (a * b) >> 8\n"},
            {"edges": []}, None,
        )
        assert "arithmetic_precision" in sel

    def test_plain_model_without_math_excludes_arithmetic_precision(self):
        sel = select_skills(
            {"name": "regfile", "description": "eight entry register file",
             "model_source": "def f(mem, addr):\n    return mem[addr]\n"},
            {"edges": [{"producer_block": "regfile", "producer_port": "rd",
                        "fields": [{"name": "data", "width": 8}]}]},
            None,
        )
        assert "arithmetic_precision" not in sel


class TestSelectSkillsConservative:
    def test_no_evidence_includes_everything(self):
        assert select_skills(None, None, None) == list(UARCH_SKILL_CANDIDATES)
        assert select_skills({}, {}, {}) == list(UARCH_SKILL_CANDIDATES)

    def test_missing_evidence_channel_votes_include(self):
        # Contracts are where the interface labels live. With none supplied we
        # cannot judge the handshake skills -- so they are INCLUDED.
        sel = select_skills({"name": "b", "description": "a widget"}, None, None)
        assert "axi_stream" in sel and "srdy_drdy" in sel
        # ...and with contracts present that say neither, they are excluded.
        contracts = {"edges": [{
            "producer_block": "b", "consumer_block": "c",
            "producer_port": "pins", "consumer_port": "pins",
            "handshake_protocol": "static",
            "fields": [{"name": "level", "width": 1}],
        }]}
        sel2 = select_skills({"name": "b", "description": "a widget"},
                             contracts, None)
        assert "axi_stream" not in sel2

    def test_no_model_source_votes_include_arithmetic(self):
        sel = select_skills(
            {"name": "b", "description": "a widget"}, {"edges": []}, None)
        assert "arithmetic_precision" in sel

    def test_selection_is_deterministic_and_ordered(self):
        spec = {"name": "pixel_line_buffer", "description": "1024x32 buffer"}
        first = select_skills(spec, None, None)
        assert first == select_skills(spec, None, None)
        assert first == [s for s in UARCH_SKILL_CANDIDATES if s in first]


class TestManifestAndAssembly:
    def test_manifest_line_carries_purpose_and_absolute_path(self):
        out = skill_manifest(["serialization_contract"])
        assert "`serialization_contract`" in out
        assert str(skill_path("serialization_contract")) in out
        assert skill_path("serialization_contract").is_absolute()
        assert "READ" in out

    def test_manifest_rejects_undescribed_skill(self):
        try:
            skill_manifest(["not_a_real_skill_xyz"])
        except MissingSkillError:
            return
        raise AssertionError("an undescribed skill must fail loudly")

    def test_missing_selected_skill_raises_loudly(self):
        try:
            build_skill_section(["not_a_real_skill_xyz"], candidates=())
        except MissingSkillError as exc:
            assert "not_a_real_skill_xyz" in str(exc)
            return
        raise AssertionError("a missing skill file must raise at first use")

    def test_port_naming_always_inline(self):
        section = build_skill_section([], candidates=UARCH_SKILL_CANDIDATES)
        assert "data_write_write_enable" in section
        # ...and every unselected skill is still reachable by path.
        for sid in UARCH_SKILL_CANDIDATES:
            assert str(skill_path(sid)) in section


# ---------------------------------------------------------------------------
# Task B1 -- the assembled uArch system prompt for a register-file block
# ---------------------------------------------------------------------------

_REGFILE_SPEC = {
    "name": "config_register_file",
    "description": "16-entry x 32-bit configuration register file with one "
                   "write port and one read port",
    "model_source": "def read(mem, addr):\n    return mem[addr]\n",
}

_REGFILE_CONTRACTS = {
    "edges": [
        {
            "edge_id": "e_w", "role": "consumer",
            "producer_block": "host_if", "consumer_block": "config_register_file",
            "producer_port": "data_write", "consumer_port": "data_write",
            "handshake_protocol": "static",
            "fields": [
                {"name": "addr", "width": 5},
                {"name": "wdata", "width": 32},
                {"name": "write_enable", "width": 1},
            ],
            "sideband_signals": [],
        },
        {
            "edge_id": "e_r", "role": "producer",
            "producer_block": "config_register_file", "consumer_block": "core",
            "producer_port": "data_read", "consumer_port": "data_read",
            "handshake_protocol": "static",
            "fields": [
                {"name": "req_addr", "width": 5},
                {"name": "rdata", "width": 32},
            ],
            "sideband_signals": [],
        },
    ]
}


class TestUarchSystemPromptForRegisterFile:
    def _prompt(self):
        from orchestrator.langchain.agents import uarch_spec_generator as u
        return u.build_system_prompt(
            block_spec=_REGFILE_SPEC,
            contracts=_REGFILE_CONTRACTS,
            block_diagram={"blocks": [{"name": "config_register_file"}]},
        )

    def test_serialization_body_is_not_inlined(self):
        prompt = self._prompt()
        assert _anchor("serialization_contract") not in prompt, (
            "a register file must not carry the 18 KB bitstream-serialization "
            "skill body"
        )

    def test_serialization_is_manifested_with_its_path(self):
        prompt = self._prompt()
        assert "`serialization_contract`" in prompt
        assert str(skill_path("serialization_contract")) in prompt

    def test_port_naming_body_is_inlined(self):
        prompt = self._prompt()
        assert "data_write_write_enable" in prompt
        assert _anchor("port_naming") in prompt

    def test_memory_skill_is_inlined_for_a_register_file(self):
        assert _anchor("memory_macro_vs_flops") in self._prompt()

    def test_smaller_than_the_unconditional_concatenation(self):
        from orchestrator.langchain.agents import uarch_spec_generator as u
        from orchestrator.langchain.prompts.skills import load_skills

        old_size = len(u.SYSTEM_PROMPT) + len(
            load_skills(*UARCH_SKILL_CANDIDATES))
        new_size = len(self._prompt())
        assert new_size < old_size, (old_size, new_size)


# ---------------------------------------------------------------------------
# Task A -- the contract port table in the production RTL prompt
# ---------------------------------------------------------------------------

_DOUBLED_TOKEN_CONTRACTS = {
    "contracts": [
        {
            "edge_id": "e1",
            "producer_block": "bus_frontend",
            "consumer_block": "register_map",
            "producer_port": "host_write",
            "consumer_port": "host_write",
            "handshake_protocol": "static",
            "fields": [
                {"name": "addr", "width": 24},
                {"name": "wdata", "width": 8},
                {"name": "write_enable", "width": 1},
            ],
            "sideband_signals": [{"name": "abort", "width": 1}],
        },
    ]
}


def _project(tmp_path):
    (tmp_path / ".coresmith").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".coresmith" / "interface_contracts.json").write_text(
        json.dumps(_DOUBLED_TOKEN_CONTRACTS), encoding="utf-8")
    return tmp_path


class TestContractPortTable:
    def test_rows_are_derived_from_the_conformance_machinery(self, tmp_path):
        from orchestrator.langgraph.contract_conformance import (
            check_block,
            contract_port_rows,
        )
        root = _project(tmp_path)
        rows = contract_port_rows(str(root), "register_map")
        names = {r["port"] for r in rows}
        assert "host_write_write_enable" in names
        assert "host_write_abort" in names, "sideband is part of the port set"

        # The gate must demand EXACTLY the names the table advertises.
        rtl = tmp_path / "register_map.v"
        rtl.write_text("module register_map(input wire clk);\nendmodule\n")
        missing = {port for _chan, port in
                   check_block(str(root), "register_map", rtl).missing}
        assert missing == names

    def test_doubled_token_row_is_flagged(self, tmp_path):
        from orchestrator.langgraph.contract_conformance import contract_port_rows
        root = _project(tmp_path)
        rows = {r["port"]: r for r in contract_port_rows(str(root), "register_map")}
        assert rows["host_write_write_enable"]["doubled_token"] is True
        assert rows["host_write_addr"]["doubled_token"] is False

    def test_rtl_prompt_carries_the_table_and_the_label(self, tmp_path):
        from orchestrator.langchain.agents.rtl_generator import build_user_message
        root = _project(tmp_path)
        prompt = build_user_message(
            block_name="register_map",
            description="register map + buffers",
            attempt=1,
            rtl_target="rtl/register_map.v",
            python_source_path="inputs/golden.py",
            project_root=str(root),
        )
        assert "AUTHORITATIVE PORT NAMES" in prompt
        assert "host_write_write_enable" in prompt
        assert "host_write_wdata" in prompt
        assert "DOUBLED TOKEN IS CORRECT" in prompt
        # The golden model must be demoted as a naming source.
        assert "take every port NAME from" in prompt

    def test_rtl_prompt_states_constraint_precedence(self, tmp_path):
        from orchestrator.langchain.agents.rtl_generator import build_user_message
        root = _project(tmp_path)
        prompt = build_user_message(
            block_name="register_map", rtl_target="rtl/x.v",
            project_root=str(root),
        )
        assert "PRECEDENCE" in prompt
        assert "constraints" in prompt.lower()

    def test_no_contract_edge_means_no_table(self, tmp_path):
        from orchestrator.langchain.agents.rtl_generator import build_user_message
        root = _project(tmp_path)
        prompt = build_user_message(
            block_name="unrelated_block", rtl_target="rtl/x.v",
            project_root=str(root),
        )
        assert "AUTHORITATIVE PORT NAMES" not in prompt

    def test_rtl_system_prompt_has_port_naming_skill(self):
        from orchestrator.langchain.agents import rtl_generator as rg
        assert "data_write_write_enable" in rg.SYSTEM_PROMPT

    def test_block_model_system_prompt_has_port_naming_skill(self):
        from orchestrator.langchain.agents import block_golden_generator as bgg
        prompt = bgg.build_system_prompt(
            block_spec=_REGFILE_SPEC, contracts=_REGFILE_CONTRACTS)
        assert "data_write_write_enable" in prompt
        # anti-cheat stays unconditional too
        assert _anchor("no_stimulus_keyed_memorization") in prompt


# ---------------------------------------------------------------------------
# Task C -- phase truthfulness
# ---------------------------------------------------------------------------

class TestDiagnosisPhaseLabelling:
    def test_conformance_prompt_has_no_waveform_hunt(self):
        from orchestrator.langchain.agents.debug_agent import (
            build_debug_user_message,
        )
        msg = build_debug_user_message("blk", "conformance")
        assert "Failed phase: conformance" in msg
        # No artifact PATH that a simulation would have produced is offered...
        for artifact in ("dump.vcd", "wavekit_audit.json", "sim_build/",
                         "tb/cocotb/"):
            assert artifact not in msg, artifact
        # ...and the agent is told, in as many words, not to go looking.
        assert "NO VCD" in msg and "NO WaveKit" in msg
        assert "do not look for them" in msg.lower()
        assert "contract_conformance.json" in msg
        assert "interface_contracts.json" in msg
        assert "no sim" in msg.lower()

    def test_sim_prompt_still_points_at_the_waveform(self):
        from orchestrator.langchain.agents.debug_agent import (
            build_debug_user_message,
        )
        msg = build_debug_user_message("blk", "sim")
        assert "Failed phase: sim" in msg
        assert "dump.vcd" in msg and "wavekit_audit.json" in msg

    def test_debug_system_prompt_carves_out_pre_sim_phases(self):
        from orchestrator.langchain.agents.debug_agent import (
            DEBUG_SYSTEM_PROMPT,
        )
        assert "Failed phase:" in DEBUG_SYSTEM_PROMPT
        assert "pre-simulation" in DEBUG_SYSTEM_PROMPT.lower()

    def test_conformance_gates_report_the_conformance_phase(self):
        import inspect

        from orchestrator.langgraph import pipeline_graph as pg
        src = inspect.getsource(pg.generate_testbench_node)
        assert '"phase": "conformance"' in src
        assert src.count('"phase": "conformance"') >= 2, (
            "both pre-TB contract gates (port widths + port names) must "
            "report the pre-sim phase"
        )

    def test_conformance_phase_routes_to_rtl_retry(self):
        from orchestrator.langgraph.pipeline_graph import _route_decision
        action = _route_decision(
            debug_result={"category": "INTERFACE_MISMATCH", "confidence": 0.9},
            attempt_history=[], attempt=1, max_attempts=5,
            phase="conformance",
        )
        assert action == "retry_rtl"


# ---------------------------------------------------------------------------
# Task B2 -- cache-friendly constraint fan-out
# ---------------------------------------------------------------------------

class TestConstraintSubagentPrefix:
    def _prompts(self):
        from orchestrator.architecture.constraints import build_subagent_prompt
        bundle = "## block_diagram.json\n" + ("x" * 5000)
        return [
            build_subagent_prompt({"id": f"c{i}", "description": f"rule {i}"},
                                  bundle)
            for i in range(8)
        ], bundle

    def test_shared_bundle_is_a_byte_identical_prefix(self):
        prompts, bundle = self._prompts()
        prefix_len = len("## Artifact bundle\n") + len(bundle)
        prefixes = {p[:prefix_len] for p in prompts}
        assert len(prefixes) == 1, "all subagents must share one exact prefix"
        assert prompts[0].startswith("## Artifact bundle\n")

    def test_charter_comes_last_and_still_names_the_constraint(self):
        prompts, bundle = self._prompts()
        for i, p in enumerate(prompts):
            assert p.index(f"`c{i}`") > p.index(bundle[:50])
            assert f"rule {i}" in p
            assert "JSON only" in p

    def test_fanout_warms_the_cache_with_one_call_first(self):
        import inspect

        from orchestrator.architecture import constraints as c
        src = inspect.getsource(c.check_constraints)
        assert "applicable[0]" in src, "one subagent must be issued first"
        assert "applicable[1:]" in src, "the rest fan out in parallel"
