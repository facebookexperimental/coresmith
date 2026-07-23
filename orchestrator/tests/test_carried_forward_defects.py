# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Section 3b: an ADVISORY composition-gate bypass must not SILENTLY swallow a
quantified mismatch -- it records a carried-forward defect (naming the SPECIFIC
unmodeled bus role) surfaced in the final report + validation-DV context.

Covers: the multi-role bus classifier (DUT-mastered second bus), the
carried-forward-defect ledger record/read + de-dup, and final-report surfacing.
"""
from __future__ import annotations

import json

from orchestrator.langgraph.bfm_lib import (
    describe_unmodeled_roles,
    detect_bus_roles,
    detect_dut_mastered_buses,
)


# ---- multi-role classifier -------------------------------------------------

_TOP_WITH_ROM = """
module user_project_wrapper (
  input  wire wb_clk_i, input wire wb_rst_i,
  input  wire [37:0] io_in, output wire [37:0] io_out, output wire [37:0] io_oeb,
  output wire rom_csn, output wire rom_sck,
  output wire rom_io0, input wire rom_io1
);
endmodule
"""

_TOP_NO_ROM = """
module user_project_wrapper (
  input  wire wb_clk_i, input wire wb_rst_i,
  input  wire [37:0] io_in, output wire [37:0] io_out, output wire [37:0] io_oeb
);
endmodule
"""


def test_detect_dut_mastered_bus():
    assert detect_dut_mastered_buses(_TOP_WITH_ROM) == ["rom"]
    assert detect_dut_mastered_buses(_TOP_NO_ROM) == []


def test_describe_unmodeled_roles_names_the_specific_bus(tmp_path):
    desc = describe_unmodeled_roles(str(tmp_path), _TOP_WITH_ROM)
    assert "rom_*" in desc
    assert "DUT-mastered second bus" in desc
    # NOT a generic single-role label
    assert desc != "QSPI-slave"


def test_detect_bus_roles_unmodeled_list(tmp_path):
    roles = detect_bus_roles(str(tmp_path), _TOP_WITH_ROM)
    assert roles["dut_master_buses"] == ["rom"]
    assert "dut_mastered_rom_bus" in roles["unmodeled"]


# ---- carried-forward-defect ledger -----------------------------------------

def test_record_and_read_carried_forward_defect(tmp_path):
    from orchestrator.langgraph.pipeline_graph import (
        record_carried_forward_defect, read_carried_forward_defects,
    )
    d = {"gate": "model_integration", "kind": "composition_mismatch",
         "unmodeled": "DUT-mastered second bus (rom_*) not modeled",
         "first_divergence_block": "acc", "violation_count": 3}
    record_carried_forward_defect(str(tmp_path), d)
    got = read_carried_forward_defects(str(tmp_path))
    assert len(got) == 1 and got[0]["unmodeled"].startswith("DUT-mastered")
    # de-dup: same key does not append twice
    record_carried_forward_defect(str(tmp_path), dict(d))
    assert len(read_carried_forward_defects(str(tmp_path))) == 1


def test_final_report_surfaces_carried_forward_defects(tmp_path):
    from orchestrator.langgraph import final_report as fr
    (tmp_path / ".coresmith").mkdir(parents=True)
    (tmp_path / ".coresmith" / "carried_forward_defects.json").write_text(
        json.dumps([{"gate": "model_integration", "kind": "composition_mismatch",
                     "violation_count": 2, "first_divergence_block": "acc",
                     "unmodeled": "DUT-mastered second bus (rom_*) not modeled"}]))
    report = fr.build_final_report({}, str(tmp_path))
    assert report["signoff"]["carried_forward_defect_count"] == 1
    assert report["carried_forward_defects"][0]["gate"] == "model_integration"
    md = fr.render_markdown(report)
    assert "Carried-forward defects" in md
    assert "rom_*" in md


# ---- Section 5g: structural failure-signature fingerprint -------------------

def test_failure_signature_stable_across_volatile_tokens():
    from orchestrator.langgraph.pipeline_graph import _failure_signature
    a = ("ERROR at /run/abc123/rtl/foo.v:412: net 0xDEADBEEF multi-driven; "
         "123 cells")
    b = ("ERROR at /run/xyz999/rtl/foo.v:87: net 0x00FF12AB multi-driven; "
         "4096 cells")
    # same STRUCTURAL error, different line/addr/path/counts -> same signature
    assert _failure_signature(a) == _failure_signature(b)


def test_failure_signature_differs_on_structural_change():
    from orchestrator.langgraph.pipeline_graph import _failure_signature
    a = "UNSYNTHESIZABLE COMBINATIONAL CLOUD in foo"
    b = "WIDE FLAT PACKED STORAGE WITH DYNAMIC PART-SELECT in foo"
    assert _failure_signature(a) != _failure_signature(b)
