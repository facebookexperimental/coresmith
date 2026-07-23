# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Defect 2: area_budget parse robustness + structural-glue area floor.

The old ``area_budget_um2`` regex was "first digit run within 16 chars of the
token", so a wrapper spec like ``area_budget_um2` (see §2) ...`` parsed to
**2 um2** and the PPA gate held a correct ~882-um2 pad adapter to a 2-um2 budget
(hundreds-x false over-budget). Symmetrically a value just past the window
(``expected below `6000 um2```) matched nothing. The parser now anchors on a real
assignment/comparison signal or a trailing um2 unit, and a glue/wrapper/adapter
block's budget is floored so a pin-mux is never held to a sub-cell budget.
"""
from __future__ import annotations

from orchestrator.langgraph.ppa_check import (
    GLUE_AREA_FLOOR_UM2,
    floor_area_budget,
    is_structural_glue_block,
    parse_area_budget,
)


class TestParseAreaBudget:
    def test_canonical_assign_form(self):
        assert parse_area_budget("area_budget_um2 = 250000") == 250000.0

    def test_backtick_colon_form_with_commas(self):
        assert parse_area_budget("`die_area_budget_um2`: 1,200,000") == 1200000.0

    def test_below_keyword_past_16_char_window(self):
        # Real value sits well past the old 16-char window; the trailing `um2`
        # unit (and the `below` keyword) now anchor it.
        spec = "- `area_budget_um2`: wrapper glue is expected below `6000 um2`."
        assert parse_area_budget(spec) == 6000.0

    def test_stray_section_ref_is_not_grabbed(self):
        # The old parser returned 2 here (from the section ref); now the
        # unit-tagged real budget wins and the bare "2" is skipped.
        spec = "area_budget_um2 (see §2 for pin map); budget 500 um2 total"
        assert parse_area_budget(spec) == 500.0

    def test_pure_stray_digit_yields_none(self):
        # No assigned value and no unit -> cannot judge (None), NOT the stray 2.
        assert parse_area_budget("area_budget_um2 (details in note 2)") is None

    def test_tilde_tiny_value_still_parses(self):
        # The genuinely-wrong "~2 um2" DOES parse (2.0); the floor -- not the
        # parser -- is what neutralizes it (see TestGlueFloor).
        assert parse_area_budget("area_budget_um2 ~ 2 um2") == 2.0

    def test_no_declaration_returns_none(self):
        assert parse_area_budget("this spec has no area budget line") is None

    def test_target_of_keyword(self):
        assert parse_area_budget("area_budget_um2 target of 12000") == 12000.0


class TestGlueDetection:
    def test_wrapper_name(self):
        assert is_structural_glue_block("user_project_wrapper") is True

    def test_pad_adapter_name(self):
        assert is_structural_glue_block("ax25_pad_adapter") is True

    def test_pin_mux_name(self):
        assert is_structural_glue_block("gpio_pin_mux") is True

    def test_spec_self_describes_as_glue(self):
        assert is_structural_glue_block(
            "some_block",
            "This wrapper is structural glue only; no storage, arithmetic, FSM.",
        ) is True

    def test_real_datapath_block_is_not_glue(self):
        assert is_structural_glue_block(
            "dct_quant_zigzag_core",
            "8x8 DCT datapath with quantization and zigzag reorder.",
        ) is False


class TestGlueFloor:
    def test_tiny_wrapper_budget_is_floored(self):
        # The "2 um2" false-flag case: a wrapper's sub-cell budget is raised to
        # the floor so the area gate cannot false-flag a correct pad adapter.
        assert floor_area_budget(2.0, "user_project_wrapper") == GLUE_AREA_FLOOR_UM2

    def test_sane_wrapper_budget_passes_through(self):
        assert floor_area_budget(8000.0, "user_project_wrapper") == 8000.0

    def test_none_stays_none(self):
        # No declared budget -> gate can't judge; the floor does not invent one.
        assert floor_area_budget(None, "user_project_wrapper") is None

    def test_non_glue_block_not_floored(self):
        # A real block's tiny (probably-wrong) budget is NOT silently relaxed --
        # only glue is floored.
        assert floor_area_budget(2.0, "dct_quant_zigzag_core") == 2.0

    def test_floor_via_spec_signal(self):
        assert floor_area_budget(
            5.0, "iface_block",
            "purely structural glue routing; no datapath",
        ) == GLUE_AREA_FLOOR_UM2

    def test_end_to_end_tiny_wrapper(self):
        spec = "area_budget_um2 ~ 2 um2 (structural pin adapter)"
        parsed = parse_area_budget(spec)
        floored = floor_area_budget(parsed, "user_project_wrapper", spec)
        assert parsed == 2.0
        assert floored == GLUE_AREA_FLOOR_UM2
