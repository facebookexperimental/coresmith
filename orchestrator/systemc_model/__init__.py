# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""SystemC TLM-2.0 loosely-timed model of the SoC (B2).

The uArch phase delivers, besides every block's microarchitecture spec, a
per-block ``sc_module`` written to the conventions in ``conventions.py``
(one socket / fifo / signal per contract channel, named after the port). The
assembler turns the block diagram + contracts into ``soc_model.cpp`` that
instantiates every block model and binds each edge, plus a Makefile; the
result runs firmware / stimulus at native speed and is the integration
golden for RTL-vs-model comparison.

Public API:
  * ``channel_binding(edge)`` -- the C++ types/names both ends must use
  * ``render_block_skeleton(block, edges)`` -- the header the agent fills
  * ``render_soc_model(blocks, edges, ...)`` -- soc_model.cpp source
  * ``write_build(project_root, blocks)`` -- Makefile + sources
  * ``build(model_dir)`` / ``smoke(model_dir)`` -- compile and run
  * ``detect()`` -- toolchain availability
"""
from __future__ import annotations

from .assemble import render_soc_model, write_build
from .conventions import channel_binding, render_block_skeleton
from .toolchain import build, detect, smoke

__all__ = ["channel_binding", "render_block_skeleton", "render_soc_model", "write_build",
           "build", "smoke", "detect"]
