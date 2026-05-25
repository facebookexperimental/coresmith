# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for `orchestrator.langchain.agents.contract_lookup`.

The helper bridges the Interface Definition arch stage's
`.coresmith/interface_contracts.json` into per-block generator prompts
(uArch spec gen + RTL gen). These tests cover the loader, the per-block
filter (with producer/consumer role tagging), and the prompt-fragment
formatter that gets dropped into agent user messages.
"""
from __future__ import annotations

import json

import pytest

from orchestrator.langchain.agents.contract_lookup import (
    filter_contracts_for_block,
    format_block_contracts_prompt,
    load_block_contracts,
    load_interface_contracts,
)


@pytest.fixture
def sample_contracts():
    return {
        "design_summary": "two-edge codec test design",
        "default_packing_convention": "msb_first_by_field_list",
        "default_endianness_rationale": "matches H.264 byte serialization",
        "contracts": [
            {
                "edge_id": "alpha__m_axis_pix__to__beta__s_axis_pix",
                "producer_block": "alpha",
                "consumer_block": "beta",
                "handshake_protocol": "axi_stream",
                "data_width_bits": 8,
                "fields": [
                    {"name": "pixel", "msb": 7, "lsb": 0, "width": 8,
                     "signed": False, "encoding": "binary"},
                ],
                "bootstrap_policy": {
                    "required": True,
                    "policy_type": "reset_seed",
                    "seed_value_hex": "0x0",
                    "rationale": "corner-MB boundary; H.264 intra has zero neighbors",
                },
            },
            {
                "edge_id": "beta__m_axis_mb__to__gamma__s_axis_mb",
                "producer_block": "beta",
                "consumer_block": "gamma",
                "handshake_protocol": "axi_stream",
                "data_width_bits": 16,
                "fields": [
                    {"name": "mb", "msb": 15, "lsb": 0, "width": 16,
                     "signed": False, "encoding": "binary"},
                ],
                "bootstrap_policy": {"required": False, "policy_type": "none"},
            },
        ],
    }


@pytest.fixture
def project_with_contracts(tmp_path, sample_contracts):
    coresmith_dir = tmp_path / ".coresmith"
    coresmith_dir.mkdir()
    (coresmith_dir / "interface_contracts.json").write_text(
        json.dumps(sample_contracts), encoding="utf-8"
    )
    return str(tmp_path)


class TestLoadInterfaceContracts:
    def test_missing_file_returns_empty(self, tmp_path):
        assert load_interface_contracts(str(tmp_path)) == {}

    def test_empty_project_root_returns_empty(self):
        assert load_interface_contracts("") == {}

    def test_malformed_json_returns_empty(self, tmp_path):
        coresmith = tmp_path / ".coresmith"
        coresmith.mkdir()
        (coresmith / "interface_contracts.json").write_text("not json{", encoding="utf-8")
        # Tolerant: returns empty rather than raising.
        assert load_interface_contracts(str(tmp_path)) == {}

    def test_well_formed_returns_full_dict(self, project_with_contracts, sample_contracts):
        loaded = load_interface_contracts(project_with_contracts)
        assert loaded == sample_contracts


class TestFilterContractsForBlock:
    def test_block_appears_in_no_edge(self, sample_contracts):
        out = filter_contracts_for_block(sample_contracts["contracts"], "nonexistent")
        assert out == []

    def test_producer_match(self, sample_contracts):
        out = filter_contracts_for_block(sample_contracts["contracts"], "alpha")
        assert len(out) == 1
        assert out[0]["edge_id"] == "alpha__m_axis_pix__to__beta__s_axis_pix"

    def test_consumer_match(self, sample_contracts):
        out = filter_contracts_for_block(sample_contracts["contracts"], "gamma")
        assert len(out) == 1
        assert out[0]["edge_id"] == "beta__m_axis_mb__to__gamma__s_axis_mb"

    def test_block_on_both_sides(self, sample_contracts):
        # `beta` is consumer of edge 1 and producer of edge 2 — both must
        # surface so the spec author knows about its full port surface.
        out = filter_contracts_for_block(sample_contracts["contracts"], "beta")
        assert len(out) == 2
        edge_ids = {e["edge_id"] for e in out}
        assert "alpha__m_axis_pix__to__beta__s_axis_pix" in edge_ids
        assert "beta__m_axis_mb__to__gamma__s_axis_mb" in edge_ids

    def test_empty_input(self):
        assert filter_contracts_for_block([], "alpha") == []
        assert filter_contracts_for_block(None, "alpha") == []


class TestLoadBlockContracts:
    def test_role_tagging(self, project_with_contracts):
        view = load_block_contracts(project_with_contracts, "beta")
        roles = {e["edge_id"]: e["role"] for e in view["edges"]}
        assert roles["alpha__m_axis_pix__to__beta__s_axis_pix"] == "consumer"
        assert roles["beta__m_axis_mb__to__gamma__s_axis_mb"] == "producer"

    def test_defaults_included(self, project_with_contracts):
        view = load_block_contracts(project_with_contracts, "alpha")
        assert view["defaults"]["default_packing_convention"] == "msb_first_by_field_list"
        assert view["defaults"]["design_summary"] == "two-edge codec test design"

    def test_block_with_no_edges(self, project_with_contracts):
        view = load_block_contracts(project_with_contracts, "unconnected_block")
        assert view["edges"] == []
        # Defaults are still returned; the prompt formatter will treat the
        # empty edges list as a no-op.
        assert view["defaults"]["default_packing_convention"]

    def test_missing_contracts_file(self, tmp_path):
        view = load_block_contracts(str(tmp_path), "anything")
        assert view == {"defaults": {}, "edges": []}


class TestFormatBlockContractsPrompt:
    def test_empty_edges_returns_empty_string(self):
        # When the block has no contracts, the caller can skip injection
        # without an `if`.
        out = format_block_contracts_prompt(
            "any_block", {"defaults": {}, "edges": []}
        )
        assert out == ""

    def test_contains_canonical_marker(self, project_with_contracts):
        view = load_block_contracts(project_with_contracts, "alpha")
        out = format_block_contracts_prompt("alpha", view)
        assert "CANONICAL INTERFACE CONTRACTS" in out
        assert "alpha__m_axis_pix__to__beta__s_axis_pix" in out
        # Authoritative-language marker for the agent to take seriously.
        assert "AUTHORITATIVE" in out

    def test_design_wide_convention_surfaces(self, project_with_contracts):
        view = load_block_contracts(project_with_contracts, "alpha")
        out = format_block_contracts_prompt("alpha", view)
        assert "msb_first_by_field_list" in out

    def test_bootstrap_policy_explicitly_called_out(self, project_with_contracts):
        # The autopilot v7 finding: bootstrap_policy was specified but the
        # RTL generator ignored it. The formatter must explicitly flag it
        # so the agent can't miss it inside the JSON blob.
        view = load_block_contracts(project_with_contracts, "alpha")
        out = format_block_contracts_prompt("alpha", view)
        assert "Bootstrap policy notice" in out
        assert "reset_seed" in out
        assert "corner-MB boundary" in out
        # The instruction to drive seed on cycle 1 must be present.
        assert "reset" in out.lower() and "seed" in out.lower()

    def test_no_bootstrap_notice_when_not_required(self, project_with_contracts):
        # `gamma` only consumes the second edge, which has
        # bootstrap.required=False; the notice should NOT appear.
        view = load_block_contracts(project_with_contracts, "gamma")
        out = format_block_contracts_prompt("gamma", view)
        assert "Bootstrap policy notice" not in out

    def test_role_visible_in_bootstrap_notice(self, project_with_contracts):
        view = load_block_contracts(project_with_contracts, "alpha")
        out = format_block_contracts_prompt("alpha", view)
        # `alpha` is the producer on the reset_seed edge.
        assert "role=producer" in out


@pytest.fixture
def flow_control_contracts():
    """Contracts featuring a feedback cycle with an elastic FIFO + a
    request/response edge — exercises the full set of flow_control
    semantics the v8 codec deadlock would have caught."""
    return {
        "design_summary": "feedback loop with elasticity",
        "default_packing_convention": "msb_first_by_field_list",
        "contracts": [
            {
                "edge_id": "sched__m_blk__to__resid__s_blk",
                "producer_block": "sched",
                "consumer_block": "resid",
                "handshake_protocol": "axi_stream",
                "data_width_bits": 8,
                "fields": [{"name": "blk", "msb": 7, "lsb": 0, "width": 8}],
                "flow_control_policy": {
                    "semantics": "elastic_fifo",
                    "min_buffer_depth_beats": 16,
                    "feedback_cycle": True,
                    "rationale": "absorbs prediction/history backpressure",
                },
            },
            {
                "edge_id": "resid__m_req__to__hist__s_req",
                "producer_block": "resid",
                "consumer_block": "hist",
                "handshake_protocol": "srdy_drdy",
                "data_width_bits": 16,
                "fields": [{"name": "addr", "msb": 15, "lsb": 0, "width": 16}],
                "flow_control_policy": {
                    "semantics": "request_response",
                    "feedback_cycle": True,
                    "rationale": "history lookups are demand-driven",
                },
            },
        ],
    }


@pytest.fixture
def project_with_flow_control(tmp_path, flow_control_contracts):
    coresmith_dir = tmp_path / ".coresmith"
    coresmith_dir.mkdir()
    (coresmith_dir / "interface_contracts.json").write_text(
        json.dumps(flow_control_contracts), encoding="utf-8"
    )
    return str(tmp_path)


class TestFormatFlowControl:
    def test_flow_control_notice_present_when_policy_exists(
        self, project_with_flow_control
    ):
        view = load_block_contracts(project_with_flow_control, "sched")
        out = format_block_contracts_prompt("sched", view)
        assert "Flow control policy notice" in out
        assert "elastic_fifo" in out
        assert "min_buffer_depth_beats=16" in out
        assert "feedback_cycle=true" in out

    def test_request_response_surfaces_for_consumer(
        self, project_with_flow_control
    ):
        view = load_block_contracts(project_with_flow_control, "hist")
        out = format_block_contracts_prompt("hist", view)
        assert "request_response" in out
        assert "role=consumer" in out

    def test_implementation_rules_per_semantics_emitted(
        self, project_with_flow_control
    ):
        view = load_block_contracts(project_with_flow_control, "resid")
        out = format_block_contracts_prompt("resid", view)
        # All five semantics need a one-line implementation rule that
        # the generator can map directly to RTL.
        for label in (
            "free_running",
            "skid",
            "elastic_fifo",
            "credit",
            "request_response",
        ):
            assert f"`{label}`" in out, f"missing rule for {label}"

    def test_no_flow_control_notice_when_absent(self, project_with_contracts):
        # The bootstrap fixture has NO flow_control_policy keys, so the
        # notice should not appear.
        view = load_block_contracts(project_with_contracts, "alpha")
        out = format_block_contracts_prompt("alpha", view)
        assert "Flow control policy notice" not in out
