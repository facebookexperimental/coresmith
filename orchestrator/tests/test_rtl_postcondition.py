# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The RTL-generation postcondition and the gate that validates the RTL.

``_assert_rtl_materialized`` checks PUBLICATION only: the bound RTL target is
a regular, nonempty file. It has no module-name regex and no byte floor (both
were heuristics that once failed a correct, lint-clean block whose module name
is fixed by an external contract: a Caravel harness mandates
``module user_project_wrapper`` while the architecture names the block
``user_project_wrapper_io``). Whether the file is valid RTL for the block --
the right top module, resolvable includes, synthesizable constructs -- is the
lint/elaboration gate's job (``lint_rtl`` with the bound target's
``--top-module``), and that is where those cases are covered here, with the
real Verilator.
"""
from __future__ import annotations

import shutil

import pytest

from orchestrator.langgraph import pipeline_helpers as ph
from orchestrator.langgraph.pipeline_helpers import _assert_rtl_materialized

BODY = "\n".join(f"  wire w{i};" for i in range(60))


def _write(tmp_path, filename: str, module: str):
    p = tmp_path / filename
    p.write_text(f"// generated\nmodule {module} (\n  input clk\n);\n{BODY}\nendmodule\n")
    return p


class TestPublicationPostcondition:
    """A regular, nonempty file is published; the name inside it is not the
    postcondition's business."""

    def test_ordinary_block_is_published(self, tmp_path):
        p = _write(tmp_path, "qspi_cdc_frontend.v", "qspi_cdc_frontend")
        assert _assert_rtl_materialized(p, "qspi_cdc_frontend") is None

    def test_externally_mandated_module_name_is_published(self, tmp_path):
        """The regression: block name != module name, and that is legitimate."""
        p = _write(tmp_path, "user_project_wrapper.v", "user_project_wrapper")
        assert _assert_rtl_materialized(p, "user_project_wrapper_io") is None

    def test_a_different_module_name_is_not_the_postconditions_call(self, tmp_path):
        p = _write(tmp_path, "zbuffer_sram.v", "framebuffer_sram")
        assert _assert_rtl_materialized(p, "zbuffer_sram") is None      # lint decides (below)

    def test_missing_file_names_the_target_and_the_block(self, tmp_path):
        err = _assert_rtl_materialized(tmp_path / "nope.v", "nope")
        assert err is not None and "RTL target is not a file" in err
        assert str(tmp_path / "nope.v") in err and "bind a file for nope" in err

    def test_a_directory_is_not_a_file(self, tmp_path):
        (tmp_path / "dir.v").mkdir()
        err = _assert_rtl_materialized(tmp_path / "dir.v", "dir")
        assert err is not None and "not a file" in err

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.v"
        p.write_text("")
        err = _assert_rtl_materialized(p, "empty")
        assert err is not None and "empty" in err and str(p) in err

    def test_a_one_line_stub_is_published_and_left_to_lint(self, tmp_path):
        """No byte floor: a nonempty stub is a file; its invalidity (an
        include that does not exist) is lint's finding, see below."""
        p = tmp_path / "stub.v"
        p.write_text('`include "elsewhere.v"\n')
        assert _assert_rtl_materialized(p, "stub") is None


_HAS_VERILATOR = shutil.which("verilator") is not None


@pytest.mark.skipif(not _HAS_VERILATOR, reason="Verilator not available")
class TestRtlValidityIsTheLintGate:
    """The real verification boundary for the cases the old heuristics
    guessed at: ``lint_rtl`` drives Verilator with the bound target's
    ``--top-module``, so the wrong block's RTL, a stub whose include is
    missing, and a mandated (non-block-name) top are decided by the tool."""

    @pytest.fixture
    def project(self, tmp_path, monkeypatch):
        from orchestrator.state_store.project_db import open_project
        monkeypatch.setattr(ph, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(ph, "_LOG_DIR", tmp_path / ".coresmith" / "step_logs")
        monkeypatch.setenv("CORESMITH_PROJECT_ROOT", str(tmp_path))
        monkeypatch.setenv("CORESMITH_SRAM_GATE", "0")
        open_project(tmp_path)
        return tmp_path

    @staticmethod
    def _bind(root, block: str, top: str, source: str):
        from orchestrator.harness.targets import bind
        return bind(root, block, {"top": top, "sources": [source]})

    def test_wrong_blocks_rtl_is_rejected_by_lint(self, project):
        p = _write(project, "rtl/zbuffer_sram.v".replace("rtl/", "") , "framebuffer_sram")
        self._bind(project, "zbuffer_sram", "zbuffer_sram", p.name)
        res = ph.lint_rtl(str(p), "zbuffer_sram")
        assert res["clean"] is False
        assert "zbuffer_sram" in res["errors"] and "%Error" in res["errors"]   # the expected top is named

    def test_externally_mandated_top_passes_lint(self, project):
        """The regression, at the gate that matters: block
        ``user_project_wrapper_io`` bound to top ``user_project_wrapper``."""
        p = _write(project, "user_project_wrapper.v", "user_project_wrapper")
        self._bind(project, "user_project_wrapper_io", "user_project_wrapper", p.name)
        res = ph.lint_rtl(str(p), "user_project_wrapper_io")
        assert res["clean"] is True, res

    def test_a_stub_with_a_missing_include_is_refused_by_the_lint_job(self, project):
        """The old byte floor guessed at this; the lint build job resolves the
        bound target's dependencies first and refuses the missing include by
        name (``CandidateError``), before Verilator runs."""
        from orchestrator.harness.top_module import CandidateError
        p = project / "stub.v"
        p.write_text('`include "elsewhere.v"\n')
        assert _assert_rtl_materialized(p, "stub") is None               # published ...
        self._bind(project, "stub", "stub", p.name)
        with pytest.raises(CandidateError, match="elsewhere.v"):          # ... refused by the build job
            ph.lint_rtl(str(p), "stub")
        p.write_text('`include "elsewhere.v"\nmodule stub(input clk); endmodule\n')
        (project / "elsewhere.v").write_text("// present\n")
        assert ph.lint_rtl(str(p), "stub")["clean"] is True               # resolvable: linted, clean

    def test_unbound_block_is_linted_as_written(self, project):
        """Without a bound target there is no --top-module: the file is
        linted as a unit, so a wrong module name is caught later (TOPLEVEL in
        simulation, the shell/integration elaboration), not by lint."""
        p = _write(project, "ordinary.v", "ordinary")
        assert ph.lint_rtl(str(p), "ordinary")["clean"] is True


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
