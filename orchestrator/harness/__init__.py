# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Engine-owned verification harness, exposed as ``coresmith verify ...``.

The harness re-runs the SAME deterministic DV/synth/PPA functions the gate
uses, as direct in-process calls (not daemon HTTP -- verify must work when the
daemon is parked or dead). Agents call it to iterate cheaply (advisory,
``source='agent'``); the engine re-runs the identical function at gate-accept
(authoritative, ``source='gate'``) so parity holds by construction.

IMPORTANT: this package's ``cli`` module MUST import without importing
``orchestrator.langgraph`` (``pipeline_helpers.PROJECT_ROOT`` freezes at import
time). All heavy imports are therefore deferred into function bodies.
"""
