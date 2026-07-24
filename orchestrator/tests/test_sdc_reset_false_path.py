# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Finding 2 (pipeline-campaign-3): SDC reset false-path + dual-site unification.

Two SDC generation sites (``generate_sdc`` + ``synthesize_block``'s inline copy)
used to blanket-apply an input delay to ALL inputs INCLUDING the reset, so the
unbuffered pre-layout reset net masqueraded as the block WNS (a -13.58 ns
artifact) that masked real timing. The fix (a) unifies both sites onto ONE
generator (``_build_sdc_content``) and (b) exempts the reset with a guarded,
reset-name-agnostic ``set_false_path`` (env ``CORESMITH_SDC_RESET_FALSE_PATH``,
default ON).
"""
from __future__ import annotations

import asyncio
import inspect

from orchestrator.langgraph import pipeline_helpers as ph

RTL_WITH_RST = (
    "module blk(input clk, input rst_n, input [7:0] din, output [7:0] dout);\n"
    "endmodule\n"
)


class TestResetFalsePath:
    def test_reset_false_path_present_by_default(self, monkeypatch):
        monkeypatch.delenv("CORESMITH_SDC_RESET_FALSE_PATH", raising=False)
        sdc = ph._build_sdc_content(RTL_WITH_RST, 50.0)
        # guarded (no-op if the port is absent) false-path on the reset port
        assert "[llength [get_ports -quiet rst_n]] > 0" in sdc
        assert "set_false_path -from [get_ports rst_n]" in sdc
        # clock + blanket I/O delays still present
        assert "create_clock -name clk" in sdc
        assert "set_input_delay -clock clk" in sdc

    def test_env_off_restores_blanket_sdc(self, monkeypatch):
        monkeypatch.setenv("CORESMITH_SDC_RESET_FALSE_PATH", "0")
        sdc = ph._build_sdc_content(RTL_WITH_RST, 50.0)
        assert "set_false_path" not in sdc           # reset exemption gone
        assert "set_input_delay -clock clk" in sdc    # blanket delays remain

    def test_reset_name_agnostic(self, monkeypatch):
        monkeypatch.delenv("CORESMITH_SDC_RESET_FALSE_PATH", raising=False)
        rtl = "module blk(input clk, input aresetn, input x);\nendmodule\n"
        sdc = ph._build_sdc_content(rtl, 50.0)
        assert "set_false_path -from [get_ports aresetn]" in sdc

    def test_detector_ignores_data_ports_that_merely_contain_the_letters(self):
        # 'burst_len' / 'first_valid' contain r-s-t but are NOT resets -> the
        # detector must not false-path a real data input (which would MASK
        # real timing). With no true reset it falls back to the guarded rst_n.
        rtl = ("module b(input clk, input [3:0] burst_len, "
               "input first_valid);\nendmodule\n")
        assert ph._detect_reset_port(rtl) == "rst_n"

    def test_detector_picks_the_real_reset_over_a_burst_port(self):
        rtl = ("module b(input clk, input rst_n, input [3:0] burst_len);\n"
               "endmodule\n")
        assert ph._detect_reset_port(rtl) == "rst_n"


class TestBothSitesUnified:
    def test_generate_sdc_uses_shared_builder(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CORESMITH_SDC_RESET_FALSE_PATH", raising=False)
        out = tmp_path / "blk.sdc"
        asyncio.run(ph.generate_sdc("blk", RTL_WITH_RST, 50.0, str(out)))
        assert out.read_text() == ph._build_sdc_content(RTL_WITH_RST, 50.0)

    def test_synthesize_block_uses_shared_builder_not_inline_sdc(self):
        # Structural guard: the second SDC site must call the shared generator
        # and must NOT re-inline its own create_clock/set_input_delay SDC (that
        # is exactly the dual-site divergence this fix removed).
        src = inspect.getsource(ph.synthesize_block)
        assert "_build_sdc_content(" in src
        assert "set_input_delay -clock clk" not in src

    def test_both_sites_produce_identical_sdc(self, tmp_path, monkeypatch):
        # generate_sdc (async site) and the inline site (which now calls the
        # same _build_sdc_content) yield byte-identical SDC for the same inputs.
        monkeypatch.delenv("CORESMITH_SDC_RESET_FALSE_PATH", raising=False)
        out = tmp_path / "blk.sdc"
        asyncio.run(ph.generate_sdc("blk", RTL_WITH_RST, 50.0, str(out)))
        inline_sdc = ph._build_sdc_content(RTL_WITH_RST, 50.0)  # synth-site call
        assert out.read_text() == inline_sdc
        assert "set_false_path -from [get_ports rst_n]" in inline_sdc
