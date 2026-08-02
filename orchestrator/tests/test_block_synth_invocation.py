# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""How the per-block yosys synth is INVOKED (cwd + script preamble).

Block-level synthesis is not a toy check -- it is the synthesizability gate and
the input to the block PPA/area numbers -- but it was launched differently from
every other yosys site in the flow, and each difference showed up as a bogus
block-level failure or a bogus block-level timing number:

* it ran in whatever directory the daemon was started in, so a PROJECT-RELATIVE
  ``$readmemh`` / ``INIT_FILE`` path in the RTL (the same path DV resolves
  happily) was unreadable;
"""
from __future__ import annotations

import shutil

import pytest

from orchestrator.langgraph import pipeline_helpers as ph

_HAS_YOSYS = shutil.which("yosys") is not None


class _StopBeforeYosys(Exception):
    """Raised by the fake subprocess so the call is observed, not executed."""


def _project(tmp_path):
    """A project root with an `inputs/` artifact the RTL reads by relative path."""
    (tmp_path / "inputs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "inputs" / "table.memh").write_text(
        "\n".join(f"{i:02x}" for i in range(16)) + "\n")
    rtl = tmp_path / "rtl"
    rtl.mkdir(parents=True, exist_ok=True)
    f = rtl / "table_reader.v"
    f.write_text(
        "module table_reader(input clk, input [3:0] a, output reg [7:0] q);\n"
        "  reg [7:0] rom [0:15];\n"
        '  initial $readmemh("inputs/table.memh", rom);\n'
        "  always @(posedge clk) q <= rom[a];\n"
        "endmodule\n")
    return f


class TestSynthRunsAtTheProjectRoot:
    """A relative artifact path in the RTL must mean the same thing in synth as
    it does in simulation."""

    def test_yosys_is_launched_with_cwd_at_the_project_root(
            self, tmp_path, monkeypatch):
        monkeypatch.setattr(ph, "PROJECT_ROOT", tmp_path)
        monkeypatch.setenv("CORESMITH_SYNTH_GENERIC", "1")
        rtl = _project(tmp_path)
        seen = {}

        def _fake_run(cmd, **kw):
            seen["cwd"] = kw.get("cwd")
            raise _StopBeforeYosys()

        monkeypatch.setattr(ph.subprocess, "run", _fake_run)
        with pytest.raises(_StopBeforeYosys):
            ph.synthesize_block({"name": "table_reader"}, str(rtl))
        assert seen["cwd"] == str(tmp_path.resolve())

    @pytest.mark.skipif(not _HAS_YOSYS, reason="yosys not installed")
    def test_a_relative_readmemh_resolves(self, tmp_path, monkeypatch):
        """The real thing: yosys must find `inputs/table.memh` from the project
        root. Before the fix it looked in the daemon's cwd and the block failed
        synth with 'Can not open file' while DV read the identical path."""
        monkeypatch.setattr(ph, "PROJECT_ROOT", tmp_path)
        monkeypatch.setenv("CORESMITH_SYNTH_GENERIC", "1")
        rtl = _project(tmp_path)
        out = ph.synthesize_block({"name": "table_reader"}, str(rtl))
        assert out["success"], out.get("log", "")[-2000:]

    @pytest.mark.skipif(not _HAS_YOSYS, reason="yosys not installed")
    def test_a_missing_artifact_still_fails(self, tmp_path, monkeypatch):
        """The fix must not paper over a genuinely absent image -- rooting the
        run is what makes 'not found' mean 'not in the project'."""
        monkeypatch.setattr(ph, "PROJECT_ROOT", tmp_path)
        monkeypatch.setenv("CORESMITH_SYNTH_GENERIC", "1")
        rtl = _project(tmp_path)
        (tmp_path / "inputs" / "table.memh").unlink()
        out = ph.synthesize_block({"name": "table_reader"}, str(rtl))
        assert not out["success"]
