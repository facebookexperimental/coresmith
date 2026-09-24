# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Accepted diagram edits must reach both saved state and implementation."""
import copy
import json

import pytest

from orchestrator.architecture.state import load_state
from orchestrator.langgraph.architecture_graph import finalize_node
from orchestrator.state_store.project_db import open_project
from orchestrator.utils import atomic_write


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["database", "view", "fresh_graph"])
async def test_finalize_preserves_accepted_diagram(tmp_path, source):
    old = {"blocks": [{"name": "example", "description": "old behavior"}],
           "connections": [], "system_invariants": ["3024 bits = 189 bytes"]}
    corrected = copy.deepcopy(old)
    corrected["blocks"][0]["description"] = "accepted behavior"
    corrected["system_invariants"] = ["3024 coded bits = 1512 input bits = 189 input bytes"]
    db = open_project(tmp_path)
    if source == "database":
        db.import_block_diagram(corrected)
    elif source == "view":
        db.import_block_diagram(old)
        # The existing registry adopts diagram edits when exporting its views.
        atomic_write(tmp_path / ".coresmith/block_diagram.json", json.dumps(corrected))

    state = {"project_root": str(tmp_path), "round": 1, "requirements": "example",
             "target_clock_mhz": 64.0,
             "block_diagram": corrected if source == "fresh_graph" else old}
    update = await finalize_node(state)
    saved = load_state(str(tmp_path))
    handed_off = {**state, **update}["block_diagram"]

    assert saved.block_diagram["system_invariants"] == corrected["system_invariants"]
    assert handed_off["system_invariants"] == corrected["system_invariants"]
    assert handed_off["blocks"][0]["description"] == "accepted behavior"
    assert db.block_specs()[0]["description"] == "accepted behavior"
