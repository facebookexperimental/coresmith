# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Discover generated VIPs available to a testbench. Recommendations only."""
from __future__ import annotations

import re


def lint_tb_imports(tb_text: str, required: list[dict]) -> list[str]:
    """Problems with a block testbench against the VIPs its edges require.

    ``required`` rows: ``{"edge_id", "module", "role"}`` (from
    ``vips_for_block``). A VIP is "used" when the TB imports its module
    (``from vip.<module> import ...`` / ``import vip.<module>``) or calls
    ``vip_load("<module>")``.
    """
    problems: list[str] = []
    text = tb_text or ""
    for row in required:
        mod = row["module"]
        pat = (rf"(from\s+vip\.{re.escape(mod)}\s+import|import\s+vip\.{re.escape(mod)}\b|"
               rf"vip_load\(\s*['\"]{re.escape(mod)}['\"])")
        if not re.search(pat, text):
            problems.append(
                f"edge {row['edge_id']} ({row['role']} side) has a generated VIP "
                f"(.coresmith/vip/{mod}.py) but the testbench does not import it: "
                f"add `from vip.{mod} import Driver, Monitor, Scoreboard, assertions` "
                "to reuse the supplied channel driver and checks.")
    return problems
