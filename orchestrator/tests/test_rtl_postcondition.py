# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the RTL-generation postcondition.

Regression: the check required a literal ``module <block_name>``, which
rejected a correct, lint-clean block whose Verilog module name is fixed by an
external contract. A Caravel harness mandates ``module user_project_wrapper``;
the architecture names that block ``user_project_wrapper_io`` and encodes the
mandated name in ``rtl_target``. The block was failed with "the agent likely
wrote the wrong module name" and the flow stopped before testbench generation.
"""
from __future__ import annotations

from orchestrator.langgraph.pipeline_helpers import _assert_rtl_materialized

BODY = "\n".join(f"  wire w{i};" for i in range(60))  # push past the 200-byte floor


def _write(tmp_path, filename: str, module: str):
    p = tmp_path / filename
    p.write_text(f"// generated\nmodule {module} (\n  input clk\n);\n{BODY}\nendmodule\n")
    return p


class TestModuleNameMatching:
    def test_ordinary_block_matches_block_name(self, tmp_path):
        p = _write(tmp_path, "qspi_cdc_frontend.v", "qspi_cdc_frontend")
        assert _assert_rtl_materialized(p, "qspi_cdc_frontend") is None

    def test_externally_mandated_module_name_is_accepted(self, tmp_path):
        """The regression: block name != module name, and that is legitimate."""
        p = _write(tmp_path, "user_project_wrapper.v", "user_project_wrapper")
        assert _assert_rtl_materialized(p, "user_project_wrapper_io") is None

    def test_wrong_block_rtl_still_rejected(self, tmp_path):
        """The check must still catch a different block's RTL."""
        p = _write(tmp_path, "zbuffer_sram.v", "framebuffer_sram")
        err = _assert_rtl_materialized(p, "zbuffer_sram")
        assert err is not None and "does not contain" in err

    def test_error_names_both_acceptable_forms(self, tmp_path):
        p = _write(tmp_path, "user_project_wrapper.v", "something_else")
        err = _assert_rtl_materialized(p, "user_project_wrapper_io")
        assert err is not None
        assert "user_project_wrapper_io" in err and "user_project_wrapper" in err


class TestOtherPostconditions:
    def test_missing_file(self, tmp_path):
        err = _assert_rtl_materialized(tmp_path / "nope.v", "nope")
        assert err is not None and "did not write" in err

    def test_stub_too_small(self, tmp_path):
        p = tmp_path / "stub.v"
        p.write_text('`include "elsewhere.v"\n')
        err = _assert_rtl_materialized(p, "stub")
        assert err is not None and "bytes" in err

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.v"
        p.write_text("")
        err = _assert_rtl_materialized(p, "empty")
        assert err is not None


class TestRtlModuleNameResolver:
    """The resolver behind TOPLEVEL, --top-module and the postcondition."""

    def test_ordinary_block_returns_block_name(self, tmp_path):
        from orchestrator.langgraph.pipeline_helpers import rtl_module_name
        p = _write(tmp_path, "zbuffer_sram.v", "zbuffer_sram")
        assert rtl_module_name(p, "zbuffer_sram") == "zbuffer_sram"

    def test_locked_top_returns_file_stem(self, tmp_path):
        """The regression: TOPLEVEL must be the module Verilator can find."""
        from orchestrator.langgraph.pipeline_helpers import rtl_module_name
        p = _write(tmp_path, "user_project_wrapper.v", "user_project_wrapper")
        assert rtl_module_name(p, "user_project_wrapper_io") == "user_project_wrapper"

    def test_block_name_wins_when_both_declared(self, tmp_path):
        """Prefer the block name so normal blocks are unaffected."""
        from orchestrator.langgraph.pipeline_helpers import rtl_module_name
        p = tmp_path / "top.v"
        p.write_text(f"module blk (input clk);\n{BODY}\nendmodule\n"
                     f"module top (input clk);\n{BODY}\nendmodule\n")
        assert rtl_module_name(p, "blk") == "blk"

    def test_missing_file_falls_back_to_block_name(self, tmp_path):
        from orchestrator.langgraph.pipeline_helpers import rtl_module_name
        assert rtl_module_name(tmp_path / "nope.v", "blk") == "blk"

    def test_neither_declared_falls_back(self, tmp_path):
        """Postcondition is the gate for this case, not the resolver."""
        from orchestrator.langgraph.pipeline_helpers import rtl_module_name
        p = _write(tmp_path, "a.v", "totally_other")
        assert rtl_module_name(p, "blk") == "blk"
