# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Interface VIP: one generated verification component per contract edge (A2).

The block testbenches used to hand-model their neighbours from prose, so the
two sides of an edge were tested against two different readings of it and
first met at chip integration. Now the Interface Definition stage's frozen
contract (fields, handshake family, structured ``timing``) is rendered --
deterministically, no LLM -- into a cocotb module per edge:

    .coresmith/vip/<edge_id>.py       Driver / Monitor / Scoreboard / assertions()
    .coresmith/vip/<edge_id>_sva.sv   the same timing rules as a bindable SVA module

Both the producer's and the consumer's testbench import the SAME module
(``lint_tb_imports`` enforces it), so a neighbour can only ever be modelled
by the contract. The Python ``assertions(dut)`` coroutine is the oracle
(portable across simulators); the SVA bind runs under ``verilator --assert``
for the subset it supports.

Public API:
  * ``EdgeContract.from_contract(edge)`` / ``EdgeContract.side(...)``
  * ``render_vip_module(edge)`` -> Python source
  * ``render_sva_bind(edge, dut_module, prefix)`` -> SystemVerilog source
  * ``write_vip(project_root, edge)`` -> {py, sva, fingerprint}
  * ``write_all_vips(project_root, contracts)`` -> index dict
  * ``vip_index(project_root)`` / ``vips_for_block(project_root, block)``
  * ``lint_tb_imports(tb_text, required)`` -> list[str] problems
"""
from __future__ import annotations

from .codegen import (
    render_sva_bind,
    render_vip_module,
    vip_index,
    vips_for_block,
    write_all_vips,
    write_vip,
)
from .contract_model import EdgeContract, SideSpec
from .lint import lint_tb_imports

__all__ = [
    "EdgeContract", "SideSpec", "render_vip_module", "render_sva_bind", "write_vip",
    "write_all_vips", "vip_index", "vips_for_block", "lint_tb_imports",
]
