# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""RTL Acceptance DV (dv-hardening-16): native RTL-vs-golden at mission scale.

E2E uses a REAL verilator build of a tiny framed DUT (byte+offset echo with
tuser/tlast + one sideband) -- skipped when verilator is absent.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from orchestrator.langgraph.acceptance_dv import (
    classify_contract,
    discover_ports,
    generate_harness,
    map_stimulus,
    run_acceptance_dv,
)

FRAMED_HEADER = """
module toy_top (
    input  wire clk,
    input  wire rst_n,
    input  wire [7:0] s_axis_tdata,
    input  wire s_axis_tvalid,
    output wire s_axis_tready,
    input  wire s_axis_tuser,
    input  wire s_axis_tlast,
    output wire [7:0] m_axis_tdata,
    output wire m_axis_tvalid,
    input  wire m_axis_tready,
    output wire m_axis_tlast,
    input  wire [7:0] cfg_offset
);
endmodule
"""


class TestContractDiscovery:
    def test_framed_shape_classified(self):
        c = classify_contract(discover_ports(FRAMED_HEADER))
        assert c is not None
        assert c["clk"] == "clk" and c["rst"] == "rst_n"
        assert c["rst_active_low"] is True
        assert c["s_axis"]["prefix"] == "s_axis"
        assert c["m_axis"]["prefix"] == "m_axis"
        assert c["s_axis"]["has_tuser"] and c["s_axis"]["has_tlast"]
        assert c["sidebands"] == {"cfg_offset": 8}

    def test_unframed_shape_rejected(self):
        assert classify_contract(discover_ports(
            "module x (input wire clk, input wire rst, input wire [7:0] d);"
            "endmodule")) is None


class TestStimulusMapping:
    _contract = classify_contract(discover_ports(FRAMED_HEADER))

    def test_dict_with_substring_sideband(self):
        m = map_stimulus({"pixels": [1, 2, 3], "offset": 7}, self._contract)
        assert m["payload"] == [1, 2, 3]
        assert m["sidebands"] == {"cfg_offset": 7}
        assert m["unmapped"] == []

    def test_flat_list(self):
        m = map_stimulus([9, 8], self._contract)
        assert m["payload"] == [9, 8]
        assert m["sidebands"] == {}

    def test_unmappable(self):
        assert map_stimulus({"qp": 20}, self._contract) is None  # no payload


class TestHarnessGeneration:
    def test_generated_source_names_ports(self):
        c = classify_contract(discover_ports(FRAMED_HEADER))
        src = generate_harness(c, "toy_top", sorted(c["sidebands"]))
        for token in ("Vtoy_top", "s_axis_tvalid", "m_axis_tlast",
                      "cfg_offset = sb[0]", "rst_n = 0", "rst_n = 1"):
            assert token in src, token


# ---------------------------------------------------------------------------
# E2E: real verilator, tiny framed DUT
# ---------------------------------------------------------------------------

TOY_DUT = """
module toy_top (
    input  wire clk,
    input  wire rst_n,
    input  wire [7:0] s_axis_tdata,
    input  wire s_axis_tvalid,
    output wire s_axis_tready,
    input  wire s_axis_tuser,
    input  wire s_axis_tlast,
    output reg  [7:0] m_axis_tdata,
    output reg  m_axis_tvalid,
    input  wire m_axis_tready,
    output reg  m_axis_tlast,
    input  wire [7:0] cfg_offset
);
    assign s_axis_tready = !m_axis_tvalid || m_axis_tready;
    always @(posedge clk) begin
        if (!rst_n) begin
            m_axis_tvalid <= 1'b0;
            m_axis_tlast <= 1'b0;
        end else begin
            if (m_axis_tvalid && m_axis_tready) m_axis_tvalid <= 1'b0;
            if (s_axis_tvalid && s_axis_tready) begin
                m_axis_tdata <= s_axis_tdata + cfg_offset {BUG};
                m_axis_tvalid <= 1'b1;
                m_axis_tlast <= s_axis_tlast;
            end
        end
    end
endmodule
"""

REFERENCE = '''\
def run(stim):
    off = stim.get("offset", 0)
    return bytes((v + off) & 0xFF for v in stim["pixels"])
'''

ACCEPTANCE = '''\
cases = [
    ("small", {"pixels": [1, 2, 3, 4], "offset": 5}),
    ("mission", {"pixels": list(range(200)), "offset": 9}),
]
'''


# A DUT that presents each output for exactly ONE cycle and clears m_axis_tvalid
# UNCONDITIONALLY on the next edge (not gated on m_axis_tready). Correct when the
# consumer is always ready; loses beats the moment the consumer backpressures --
# the exact handshake-drop class the evaluation harness run-through caught in CoreSmith's
# RTL that its own no-backpressure harness passed.
DROP_DUT = """
module toy_top (
    input  wire clk,
    input  wire rst_n,
    input  wire [7:0] s_axis_tdata,
    input  wire s_axis_tvalid,
    output wire s_axis_tready,
    input  wire s_axis_tuser,
    input  wire s_axis_tlast,
    output reg  [7:0] m_axis_tdata,
    output reg  m_axis_tvalid,
    input  wire m_axis_tready,
    output reg  m_axis_tlast,
    input  wire [7:0] cfg_offset
);
    assign s_axis_tready = 1'b1;
    always @(posedge clk) begin
        if (!rst_n) begin
            m_axis_tvalid <= 1'b0;
            m_axis_tlast <= 1'b0;
        end else begin
            m_axis_tvalid <= 1'b0;   // BUG: clear regardless of acceptance
            if (s_axis_tvalid && s_axis_tready) begin
                m_axis_tdata <= s_axis_tdata + cfg_offset;
                m_axis_tvalid <= 1'b1;
                m_axis_tlast <= s_axis_tlast;
            end
        end
    end
endmodule
"""


# A 32-bit WORD-stream DUT (echo + offset). The old byte-granular harness drove
# only 8 bits of the 32-bit tdata and captured only the low byte -- it could not
# grade a word stream at all (CoreSmith's run-through limitation). dv-30
# width-aware driving packs 4 payload bytes/beat and captures 4 bytes/beat.
WORD_DUT = """
module toy_top (
    input  wire clk,
    input  wire rst_n,
    input  wire [31:0] s_axis_tdata,
    input  wire s_axis_tvalid,
    output wire s_axis_tready,
    input  wire s_axis_tuser,
    input  wire s_axis_tlast,
    output reg  [31:0] m_axis_tdata,
    output reg  m_axis_tvalid,
    input  wire m_axis_tready,
    output reg  m_axis_tlast,
    input  wire [7:0] cfg_offset
);
    assign s_axis_tready = !m_axis_tvalid || m_axis_tready;
    always @(posedge clk) begin
        if (!rst_n) begin
            m_axis_tvalid <= 1'b0;
            m_axis_tlast <= 1'b0;
        end else begin
            if (m_axis_tvalid && m_axis_tready) m_axis_tvalid <= 1'b0;
            if (s_axis_tvalid && s_axis_tready) begin
                m_axis_tdata <= s_axis_tdata + cfg_offset;
                m_axis_tvalid <= 1'b1;
                m_axis_tlast <= s_axis_tlast;
            end
        end
    end
endmodule
"""

WORD_REFERENCE = '''\
def run(stim):
    off = stim.get("offset", 0)
    px = stim["pixels"]   # little-endian u32 words, 4 bytes each
    out = bytearray()
    for i in range(0, len(px), 4):
        w = px[i] | (px[i+1] << 8) | (px[i+2] << 16) | (px[i+3] << 24)
        r = (w + off) & 0xFFFFFFFF
        out += bytes([r & 0xff, (r >> 8) & 0xff, (r >> 16) & 0xff, (r >> 24) & 0xff])
    return bytes(out)
'''

WORD_ACCEPTANCE = '''\
import random
_r = random.Random(3)
def _words(nw):
    b = []
    for _ in range(nw):
        w = _r.randrange(0, 1 << 32)
        b += [w & 0xff, (w >> 8) & 0xff, (w >> 16) & 0xff, (w >> 24) & 0xff]
    return b
cases = [
    ("small", {"pixels": _words(4), "offset": 7}),
    ("mission", {"pixels": _words(80), "offset": 100}),
]
'''


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _project(tmp_path: Path, bug: str = "") -> tuple[Path, Path]:
    root = tmp_path
    top = root / "rtl" / "toy_top.v"
    _write(top, TOY_DUT.replace("{BUG}", bug))
    _write(root / "inputs" / "toy_golden.py", REFERENCE)
    _write(root / "inputs" / "acceptance_stimulus.py", ACCEPTANCE)
    return root, top


def _env(monkeypatch):
    for var in ("CORESMITH_REFERENCE_ENTRY", "CORESMITH_ACCEPTANCE_STIMULUS",
                "CORESMITH_FIDELITY_GATE", "CORESMITH_FIDELITY_METRIC",
                "CORESMITH_ACCEPTANCE_DV"):
        monkeypatch.delenv(var, raising=False)


needs_verilator = pytest.mark.skipif(
    not shutil.which("verilator"), reason="verilator not on PATH")


@needs_verilator
class TestAcceptanceDVEndToEnd:
    def test_correct_dut_passes(self, tmp_path, monkeypatch):
        _env(monkeypatch)
        root, top = _project(tmp_path)
        res = run_acceptance_dv(str(root), str(top))
        assert not res["skipped"], res["reason"]
        assert res["passed"], res
        assert len(res["cases"]) == 2
        assert all(r["ok"] for r in res["cases"])
        assert (root / ".coresmith" / "acceptance_dv.json").exists()

    def test_wrong_dut_diverges_with_offset(self, tmp_path, monkeypatch):
        _env(monkeypatch)
        root, top = _project(tmp_path, bug="+ 8'd1")
        res = run_acceptance_dv(str(root), str(top))
        assert not res["skipped"], res["reason"]
        assert not res["passed"]
        v = res["violations"][0]
        assert v["criterion"] == "acceptance_dv_divergence"
        assert v["first_divergence_offset"] == 0

    def test_no_artifact_honest_skip(self, tmp_path, monkeypatch):
        _env(monkeypatch)
        root, top = _project(tmp_path)
        (root / "inputs" / "acceptance_stimulus.py").unlink()
        res = run_acceptance_dv(str(root), str(top))
        assert res["skipped"]
        assert "acceptance stimulus" in res["reason"]

    def test_kill_switch(self, tmp_path, monkeypatch):
        _env(monkeypatch)
        monkeypatch.setenv("CORESMITH_ACCEPTANCE_DV", "0")
        root, top = _project(tmp_path)
        res = run_acceptance_dv(str(root), str(top))
        assert res["skipped"]

    def test_backpressure_catches_dropped_beat(self, tmp_path, monkeypatch):
        # dv-hardening-27: the drop-on-transfer-edge DUT must FAIL when the
        # harness randomizes output backpressure (the default).
        _env(monkeypatch)
        root = tmp_path
        top = root / "rtl" / "toy_top.v"
        _write(top, DROP_DUT)
        _write(root / "inputs" / "toy_golden.py", REFERENCE)
        _write(root / "inputs" / "acceptance_stimulus.py", ACCEPTANCE)
        res = run_acceptance_dv(str(root), str(top))
        assert not res["skipped"], res["reason"]
        assert not res["passed"], "backpressure must expose the dropped beat"

    def test_no_backpressure_misses_dropped_beat(self, tmp_path, monkeypatch):
        # The SAME buggy DUT PASSES with backpressure disabled -- documenting
        # exactly the blind spot that shipped the CoreSmith handshake bug.
        _env(monkeypatch)
        monkeypatch.setenv("CORESMITH_ACCEPTANCE_DV_BACKPRESSURE", "0")
        root = tmp_path
        top = root / "rtl" / "toy_top.v"
        _write(top, DROP_DUT)
        _write(root / "inputs" / "toy_golden.py", REFERENCE)
        _write(root / "inputs" / "acceptance_stimulus.py", ACCEPTANCE)
        res = run_acceptance_dv(str(root), str(top))
        assert not res["skipped"], res["reason"]
        assert res["passed"], "no-backpressure harness misses the drop (blind spot)"

    def test_word_stream_driven_and_captured(self, tmp_path, monkeypatch):
        # dv-hardening-30: a 32-bit WORD stream (the run-through's matmul shape)
        # must be driven/captured at full width -- 4 payload bytes per beat --
        # and match the oracle byte-exact, WITH backpressure on.
        _env(monkeypatch)
        root = tmp_path
        top = root / "rtl" / "toy_top.v"
        _write(top, WORD_DUT)
        _write(root / "inputs" / "toy_golden.py", WORD_REFERENCE)
        _write(root / "inputs" / "acceptance_stimulus.py", WORD_ACCEPTANCE)
        res = run_acceptance_dv(str(root), str(top))
        assert not res["skipped"], res["reason"]
        assert res["passed"], res

    def test_word_stream_wrong_offset_diverges(self, tmp_path, monkeypatch):
        # A word DUT with the wrong offset must FAIL -- proving the width-aware
        # capture actually compares all 4 bytes, not just the low byte.
        _env(monkeypatch)
        root = tmp_path
        top = root / "rtl" / "toy_top.v"
        _write(top, WORD_DUT.replace("s_axis_tdata + cfg_offset",
                                     "s_axis_tdata + cfg_offset + 32'h100"))
        _write(root / "inputs" / "toy_golden.py", WORD_REFERENCE)
        _write(root / "inputs" / "acceptance_stimulus.py", WORD_ACCEPTANCE)
        res = run_acceptance_dv(str(root), str(top))
        assert not res["skipped"], res["reason"]
        assert not res["passed"], "wrong upper-byte offset must be caught"


class TestTopModuleScoping:
    """dv-hardening-23 (armD defect #9): helper modules in the chip-top file
    (rst_sync_2ff etc.) must not leak their ports into the top contract's
    sideband map -- the harness would drive nonexistent top pins."""

    def test_helper_module_ports_not_in_sidebands(self):
        from orchestrator.langgraph.acceptance_dv import (
            classify_contract,
            discover_ports,
        )

        rtl = """
module chip_top (
    input  wire clk,
    input  wire rst_n,
    input  wire [7:0] s_axis_tdata,
    input  wire s_axis_tvalid,
    output wire s_axis_tready,
    input  wire s_axis_tlast,
    input  wire [5:0] cfg_qp,
    output wire [7:0] m_axis_tdata,
    output wire m_axis_tvalid,
    input  wire m_axis_tready,
    output wire m_axis_tlast
);
endmodule

module rst_sync_2ff (
    input  wire clk,
    input  wire rst_n_in,
    output wire rst_n_out
);
endmodule
"""
        import re
        m = re.search(r"\bmodule\s+([A-Za-z_][A-Za-z0-9_]*)", rtl)
        end = rtl.find("endmodule", m.start())
        span = rtl[m.start():end]
        contract = classify_contract(discover_ports(span))
        assert contract is not None
        assert "rst_n_in" not in contract["sidebands"]
        assert set(contract["sidebands"]) == {"cfg_qp"}


class TestShapeDerivedSidebands:
    """dv-hardening-24 (armD defect #10): width/height sidebands must be
    derived from the payload's 2D shape when the stimulus doesn't name them --
    unmapped sidebands drove 0 and the chip waited forever on a 0x0 frame."""

    def test_width_height_from_2d_shape(self):
        import numpy as np

        from orchestrator.langgraph.acceptance_dv import map_stimulus

        contract = {
            "sidebands": {"cfg_frame_width": 8, "cfg_frame_height": 8,
                          "cfg_qp": 6},
        }
        stim = {"frames": np.zeros((1, 144, 176), dtype=np.uint8), "qp": 28}
        m = map_stimulus(stim, contract)
        assert m is not None
        assert m["sidebands"]["cfg_frame_width"] == 176
        assert m["sidebands"]["cfg_frame_height"] == 144
        assert m["sidebands"]["cfg_qp"] == 28
        assert m["unmapped"] == []

    def test_explicit_values_not_overridden(self):
        from orchestrator.langgraph.acceptance_dv import map_stimulus

        contract = {"sidebands": {"cfg_frame_width": 8}}
        stim = {"frames": [[1, 2], [3, 4]], "frame_width": 99}
        m = map_stimulus(stim, contract)
        assert m is not None
        assert m["sidebands"]["cfg_frame_width"] == 99


class TestMultiInputStreamHonestSkip:
    """dv-hardening-25 (aes128 OOD): a chip_top with two input stream groups
    (s_axis + s_axis_key) can't be driven by the single-input-stream harness;
    classify_contract must return None (-> honest skip) rather than mapping the
    2nd stream to sidebands and false-failing (aes128 live: 3/3 rtl_bytes=0 on
    a byte-exact RTL)."""

    _AES_TOP = """
module aes_top (
    input  wire clk,
    input  wire rst_n,
    input  wire [7:0] s_axis_tdata,
    input  wire s_axis_tvalid,
    output wire s_axis_tready,
    input  wire s_axis_tlast,
    input  wire [7:0] s_axis_key_tdata,
    input  wire s_axis_key_tvalid,
    output wire s_axis_key_tready,
    input  wire s_axis_key_tlast,
    output wire [7:0] m_axis_tdata,
    output wire m_axis_tvalid,
    input  wire m_axis_tready,
    output wire m_axis_tlast
);
endmodule
"""

    _SINGLE_TOP = """
module enc_top (
    input  wire clk,
    input  wire rst_n,
    input  wire [7:0] s_axis_tdata,
    input  wire s_axis_tvalid,
    output wire s_axis_tready,
    input  wire s_axis_tlast,
    input  wire [5:0] cfg_qp,
    output wire [7:0] m_axis_tdata,
    output wire m_axis_tvalid,
    input  wire m_axis_tready,
    output wire m_axis_tlast
);
endmodule
"""

    def test_two_input_streams_skips(self):
        from orchestrator.langgraph.acceptance_dv import (
            classify_contract,
            discover_ports,
        )
        assert classify_contract(discover_ports(self._AES_TOP)) is None

    def test_single_input_stream_still_classifies(self):
        from orchestrator.langgraph.acceptance_dv import (
            classify_contract,
            discover_ports,
        )
        c = classify_contract(discover_ports(self._SINGLE_TOP))
        assert c is not None
        assert c["s_axis"]["prefix"] == "s_axis"
        assert "cfg_qp" in c["sidebands"]
