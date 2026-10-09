# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Physical constants must be driven by deployment-owned tie cells."""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.langgraph.backend_helpers import (
    generate_flat_synthesis_script,
    verify_physical_constant_mapping,
)
from orchestrator.pdk.base import Deployment
from orchestrator.pdk.pdk_config import CellConfig, PDKConfig

HILOMAP = (
    "hilomap -hicell sky130_fd_sc_hd__conb_1 HI "
    "-locell sky130_fd_sc_hd__conb_1 LO"
)


def _script(path: Path, *, mapping: str = HILOMAP):
    path.write_text(
        "abc -liberty cells.lib\n"
        f"{mapping}\n"
        "clean\n"
        "write_verilog out.v\n",
        encoding="utf-8",
    )


def test_literal_dff_sink_fails_even_when_script_has_hilomap(tmp_path):
    script = tmp_path / "synth_top.ys"
    netlist = tmp_path / "top_netlist.v"
    _script(script)
    netlist.write_text(
        "module top(input clk, output q);\n"
        "  sky130_fd_sc_hd__dfxtp_1 ff (.CLK(clk), .D(1'h0), .Q(q));\n"
        "endmodule\n",
        encoding="utf-8",
    )

    result = verify_physical_constant_mapping(
        script, netlist, HILOMAP, ("sky130_fd_sc_hd__conb_1",)
    )

    assert result["ok"] is False
    assert result["literal_pin_constants"] == [{"pin": "D", "value": "1'h0"}]
    assert "literal constant cell-pin" in result["reason"]


@pytest.mark.parametrize("literal", ["1'h1", "1'd1"])
def test_high_literal_sink_fails_even_with_valid_script_order(tmp_path, literal):
    script = tmp_path / "synth_top.ys"
    netlist = tmp_path / "top_netlist.v"
    _script(script)
    netlist.write_text(
        "module top(input clk, output q);\n"
        f"  sky130_fd_sc_hd__dfxtp_1 ff (.CLK(clk), .D({literal}), .Q(q));\n"
        "endmodule\n",
        encoding="utf-8",
    )

    result = verify_physical_constant_mapping(
        script, netlist, HILOMAP, ("sky130_fd_sc_hd__conb_1",)
    )

    assert result["ok"] is False
    assert result["literal_pin_constants"] == [{"pin": "D", "value": literal}]


def test_tie_driven_dff_sink_passes_and_counts_tie_instance(tmp_path):
    script = tmp_path / "synth_top.ys"
    netlist = tmp_path / "top_netlist.v"
    _script(script)
    netlist.write_text(
        "module top(input clk, output q);\n"
        "  wire zero_;\n"
        "  sky130_fd_sc_hd__conb_1 tie0 (.LO(zero_));\n"
        "  sky130_fd_sc_hd__dfxtp_1 ff (.CLK(clk), .D(zero_), .Q(q));\n"
        "endmodule\n",
        encoding="utf-8",
    )

    result = verify_physical_constant_mapping(
        script, netlist, HILOMAP, ("sky130_fd_sc_hd__conb_1",)
    )

    assert result["ok"] is True
    assert result["tie_instance_count"] == 1
    assert result["literal_pin_constants"] == []


def test_hilomap_must_follow_abc_and_precede_write(tmp_path):
    script = tmp_path / "synth_top.ys"
    netlist = tmp_path / "top_netlist.v"
    _script(script, mapping="")
    netlist.write_text("module top(); endmodule\n", encoding="utf-8")

    result = verify_physical_constant_mapping(
        script, netlist, HILOMAP, ("sky130_fd_sc_hd__conb_1",)
    )

    assert result["ok"] is False
    assert "after final abc and before write_verilog" in result["reason"]


def test_flat_synth_script_maps_constants_without_macro_instances(tmp_path):
    top = tmp_path / "top.v"
    top.write_text("module top(input clk, output q); assign q = 1'b0; endmodule\n")

    script_path = generate_flat_synthesis_script(
        "top", str(top), {}, output_dir=str(tmp_path / "out"), top_module="top"
    )
    script = Path(script_path).read_text(encoding="utf-8")

    assert HILOMAP in script
    assert script.index("abc -liberty") < script.index(HILOMAP)
    assert script.index(HILOMAP) < script.index("\nclean\n")


class _BareDeployment(Deployment):
    name = "bare"

    def __init__(self):
        self._pdk = PDKConfig(
            name="bare", process_nm=1, std_cell_library="cells",
            site_name="site", supply_voltage=1.0, default_corner="tt",
            cells=CellConfig(),
        )

    @property
    def pdk(self):
        return self._pdk

    def tools(self):
        return {}


def test_byo_pdk_without_tie_mapping_reports_unsupported_capability():
    assert _BareDeployment().yosys_hilomap_command() == ""


def test_flat_synth_fails_closed_when_byo_pdk_has_no_tie_mapping(
        tmp_path, monkeypatch):
    top = tmp_path / "top.v"
    top.write_text("module top(output q); assign q = 1'b0; endmodule\n")
    monkeypatch.setattr(
        "orchestrator.pdk.registry.get_deployment", lambda: _BareDeployment()
    )

    with pytest.raises(ValueError, match="does not declare physical constant cells"):
        generate_flat_synthesis_script(
            "top", str(top), {}, output_dir=str(tmp_path / "out"),
            top_module="top",
        )


def test_incomplete_byo_pdk_tie_mapping_is_configuration_error():
    dep = _BareDeployment()
    dep.pdk.cells.tie_low_cell = "tie0"

    try:
        dep.yosys_hilomap_command()
    except ValueError as exc:
        assert "incomplete constant-cell mapping" in str(exc)
        assert "cells.tie_high_cell" in str(exc)
    else:
        raise AssertionError("partial tie mapping must fail closed")
