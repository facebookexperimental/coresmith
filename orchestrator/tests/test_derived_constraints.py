# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic tests for generic derived architecture constraints.

These tests do not call an LLM. They verify the rule-based contract checker
that runs before the holistic LLM constraint review.
"""

from __future__ import annotations

import json


def _diagram_with_contract(contract: str) -> dict:
    return {
        "blocks": [
            {
                "name": "ingest",
                "tier": 2,
                "interfaces": {"pixels": {"width": 8}, "macroblocks": {"width": 768}},
                "semantic_contracts": [contract],
                "estimated_gates": 10000,
            },
            {
                "name": "encode",
                "tier": 3,
                "interfaces": {"macroblocks": {"width": 768}, "bytes": {"width": 8}},
                "semantic_contracts": ["Consumes macroblocks in the same raster order."],
                "estimated_gates": 20000,
            },
        ],
        "connections": [
            {"from": "ingest", "to": "encode", "interface": "macroblocks", "data_width": 768}
        ],
        "system_invariants": [contract],
    }


def test_derived_geometry_passes_for_consistent_contract(tmp_path):
    from orchestrator.architecture.constraints import _check_derived_constraints

    contract = (
        "For 640x360 frames split into 8x8 macroblocks, emit 80 macroblock "
        "columns and 45 macroblock rows in raster order. mb_x=0..79, "
        "mb_y=0..44. Exactly 3600 macroblocks per frame. mb_x uses 7 bits "
        "and mb_y uses 6 bits."
    )

    violations = _check_derived_constraints(
        block_diagram=_diagram_with_contract(contract),
        memory_map={},
        clock_tree={},
        register_spec={},
        requirements=contract,
        ers_spec={},
        project_root=str(tmp_path),
    )

    assert violations == []
    audit = json.loads((tmp_path / ".socmate" / "derived_constraints_audit.json").read_text())
    assert audit["facts"]["source_dim"][:2] == [640, 360]


def test_derived_geometry_ignores_physical_and_subblock_dimensions(tmp_path):
    from orchestrator.architecture.constraints import _check_derived_constraints

    contract = (
        "OpenFrame die dimensions are 3320 x 4988 um after margins. "
        "The transform datapath also has internal 4x4 blocks. "
        "For 640x360 video frames split into 8x8 macroblocks, emit 80 "
        "macroblock columns and 45 macroblock rows. mb_x=0..79, mb_y=0..44. "
        "Exactly 3600 macroblocks per frame."
    )

    violations = _check_derived_constraints(
        block_diagram=_diagram_with_contract(contract),
        memory_map={},
        clock_tree={},
        register_spec={},
        requirements=contract,
        ers_spec={},
        project_root=str(tmp_path),
    )

    assert violations == []
    audit = json.loads((tmp_path / ".socmate" / "derived_constraints_audit.json").read_text())
    assert audit["facts"]["source_dim"][:2] == [640, 360]


def test_derived_geometry_ignores_pixel_and_coefficient_indices(tmp_path):
    from orchestrator.architecture.constraints import _check_derived_constraints

    contract = (
        "For 640x360 frames split into 8x8 macroblocks, emit 80 macroblock "
        "columns and 45 macroblock rows. mb_x=0..79, mb_y=0..44. "
        "The pixel grid is 640 columns by 360 rows, pixel_x=0..639, "
        "pixel_y=0..359. Each 8x8 transform uses coeff_index=0..63 and "
        "quant_table_index=0..63."
    )

    violations = _check_derived_constraints(
        block_diagram=_diagram_with_contract(contract),
        memory_map={},
        clock_tree={},
        register_spec={},
        requirements=contract,
        ers_spec={},
        project_root=str(tmp_path),
    )

    assert violations == []


def test_derived_geometry_ignores_local_subblock_indices(tmp_path):
    from orchestrator.architecture.constraints import _check_derived_constraints

    contract = (
        "For 640x360 frames split into 8x8 macroblocks, emit 80 macroblock "
        "columns and 45 macroblock rows. mb_x=0..79, mb_y=0..44. "
        "Each 8x8 macroblock may contain four local 4x4 transform subblocks; "
        "local_subblock_idx=0..3 is a local selector, not a frame coordinate."
    )

    violations = _check_derived_constraints(
        block_diagram=_diagram_with_contract(contract),
        memory_map={},
        clock_tree={},
        register_spec={},
        requirements=contract,
        ers_spec={},
        project_root=str(tmp_path),
    )

    assert violations == []


def test_derived_geometry_catches_transposed_rows_and_columns(tmp_path):
    from orchestrator.architecture.constraints import _check_derived_constraints

    contract = (
        "For 640x360 frames split into 8x8 macroblocks, emit 45 macroblock "
        "columns and 80 macroblock rows. mb_x=0..44, mb_y=0..79. "
        "Exactly 3600 macroblocks per frame."
    )

    violations = _check_derived_constraints(
        block_diagram=_diagram_with_contract(contract),
        memory_map={},
        clock_tree={},
        register_spec={},
        requirements=contract,
        ers_spec={},
        project_root=str(tmp_path),
    )

    checks = {v["check"] for v in violations}
    assert "derived_geometry_columns_rows" in checks
    assert "derived_coordinate_range" in checks
    assert any("80 columns and 45 rows" in v["violation"] for v in violations)


def test_derived_geometry_catches_too_narrow_coordinate_width(tmp_path):
    from orchestrator.architecture.constraints import _check_derived_constraints

    contract = (
        "For 640x360 frames split into 8x8 macroblocks, emit 80 macroblock "
        "columns and 45 macroblock rows. mb_x=0..79, mb_y=0..44. "
        "Exactly 3600 macroblocks per frame. mb_x is 6 bits; mb_y is 6 bits."
    )

    violations = _check_derived_constraints(
        block_diagram=_diagram_with_contract(contract),
        memory_map={},
        clock_tree={},
        register_spec={},
        requirements=contract,
        ers_spec={},
        project_root=str(tmp_path),
    )

    width_violations = [v for v in violations if v["check"] == "derived_coordinate_width"]
    assert width_violations
    assert "requires at least 7 bits" in width_violations[0]["violation"]


def test_derived_geometry_catches_bad_total_count(tmp_path):
    from orchestrator.architecture.constraints import _check_derived_constraints

    contract = (
        "For a 96x64 matrix split into 16x8 tiles, emit 6 tile columns and "
        "8 tile rows. Exactly 47 tiles total."
    )

    violations = _check_derived_constraints(
        block_diagram=_diagram_with_contract(contract),
        memory_map={},
        clock_tree={},
        register_spec={},
        requirements=contract,
        ers_spec={},
        project_root=str(tmp_path),
    )

    assert any(v["check"] == "derived_transaction_count" for v in violations)


def test_derived_geometry_ignores_stale_repair_notes_and_accepts_commas(tmp_path):
    from orchestrator.architecture.constraints import _check_derived_constraints

    contract = (
        "For 640x360 video frames split into 4x4 blocks, emit 160 block "
        "columns and 90 block rows. Exactly 14,400 blocks per frame. "
        "Violation before/after: previous artifact had stale derived "
        "transaction count of 400 blocks per frame; repaired contract is "
        "14,400 blocks per frame."
    )

    violations = _check_derived_constraints(
        block_diagram=_diagram_with_contract(contract),
        memory_map={},
        clock_tree={},
        register_spec={},
        requirements=contract,
        ers_spec={},
        project_root=str(tmp_path),
    )

    assert violations == []


def test_derived_geometry_does_not_treat_grid_phrase_as_total(tmp_path):
    from orchestrator.architecture.constraints import _check_derived_constraints

    contract = (
        "For 640x360 video frames split into 4x4 blocks, emit 160 block "
        "columns and 90 block rows. Preserve deterministic raster block order: "
        "block_y-major, block_x-minor, 160 by 90 blocks per frame. "
        "The total is exactly 14400 blocks per frame."
    )

    violations = _check_derived_constraints(
        block_diagram=_diagram_with_contract(contract),
        memory_map={},
        clock_tree={},
        register_spec={},
        requirements=contract,
        ers_spec={},
        project_root=str(tmp_path),
    )

    assert violations == []


def test_performance_budget_catches_cycles_per_block_contradiction(tmp_path):
    from orchestrator.architecture.constraints import _check_performance_constraints

    contract = (
        "For 640x360 frames split into 8x8 macroblocks, emit 80 macroblock "
        "columns and 45 macroblock rows. Exactly 3600 macroblocks per frame. "
        "Target clock is 50 MHz and throughput is 30 fps. PERF-003 allows "
        "completion within 1024 + 4096 * macroblocks cycles after the last input pixel."
    )

    violations = _check_performance_constraints(
        block_diagram=_diagram_with_contract(contract),
        memory_map={},
        clock_tree={},
        register_spec={},
        requirements=contract,
        ers_spec={"prd": {"speed_and_feeds": {"target_clock_mhz": 50, "throughput_requirements": "30 fps"}}},
        project_root=str(tmp_path),
    )

    assert any(v["check"] == "performance_cycle_budget" for v in violations)
    assert "cycles/frame" in violations[0]["violation"]


def test_performance_budget_accepts_feasible_cycles_per_block(tmp_path):
    from orchestrator.architecture.constraints import _check_performance_constraints

    contract = (
        "For 640x360 frames split into 8x8 macroblocks, emit 80 macroblock "
        "columns and 45 macroblock rows. Exactly 3600 macroblocks per frame. "
        "Target clock is 50 MHz and throughput is 30 fps. The pipeline budget "
        "is 64 cycles per macroblock."
    )

    violations = _check_performance_constraints(
        block_diagram=_diagram_with_contract(contract),
        memory_map={},
        clock_tree={},
        register_spec={},
        requirements=contract,
        ers_spec={"prd": {"speed_and_feeds": {"target_clock_mhz": 50, "throughput_requirements": "30 fps"}}},
        project_root=str(tmp_path),
    )

    assert violations == []
