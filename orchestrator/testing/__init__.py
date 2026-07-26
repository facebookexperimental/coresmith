# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""orchestrator.testing -- test-only fault injection / property harness.

IMPORTANT: production code must NEVER import this package. ``coresmith_llm``
reaches it only lazily (via ``importlib`` when ``CORESMITH_LLM_PROVIDER`` is a
testing provider), so a normal run has zero coupling to the test harness.

Contents:
  - ``faults``          -- FaultClass taxonomy, FaultSpec, FaultSchedule.
  - ``fault_provider``  -- FaultBackend: a ClaudeLLM backend that injects faults.
  - ``success_scripts`` -- CannedDesignScript: writes the artifacts a non-faulted
                           call is expected to produce (so the baseline run passes).
  - ``eda_stubs``       -- stub_eda(): standardize the lint/sim/synth/equiv patches.
  - ``properties``      -- drive a block subgraph to terminal + P1-P4 invariants.

The canned artifacts here validate PLUMBING (routing, guards, retry bounds),
NOT RTL quality; real EDA belongs in the T1/nightly tiers.
"""
