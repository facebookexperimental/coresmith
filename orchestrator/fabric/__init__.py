# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The SoC fabric primitive (B1).

An SoC's bus -- N initiators onto M targets with address decode, AXI4 /
AXI-Lite / APB bridging and an error slave -- is generated from a small
declarative ``FabricSpec``, never hand-written by the RTL agent. The
generator renders a SystemVerilog wrapper over the vendored pulp-platform
``axi`` IP (``langgraph/rtl_lib/fabric/sv``, SHL-0.51), elaborates it with
yosys-slang and writes plain Verilog-2001 (``cs_fabric_<name>.v``) plus a
cocotb testbench (cocotbext-axi) that exercises decode, decode errors,
bursts, per-ID ordering, backpressure and fairness.

Public API:
  * ``FabricSpec`` / ``FabricMaster`` / ``FabricSlave`` (spec.py)
  * ``CHANNELS`` -- canonical AMBA signal lists per family (amba.py)
  * ``render_wrapper_sv(spec)``, ``elaborate(spec, out_dir)``,
    ``generate_fabric(spec, out_dir)`` (generate.py)
  * ``render_testbench(spec)`` (tb_template.py)
"""
from __future__ import annotations

from .amba import CHANNELS, port_name
from .generate import (
    FabricArtifacts,
    elaborate,
    generate_fabric,
    render_wrapper_sv,
    slang_available,
)
from .spec import FabricMaster, FabricSlave, FabricSpec
from .tb_template import render_testbench

__all__ = [
    "FabricSpec", "FabricMaster", "FabricSlave", "CHANNELS", "port_name",
    "render_wrapper_sv", "elaborate", "generate_fabric", "render_testbench",
    "FabricArtifacts", "slang_available",
]
