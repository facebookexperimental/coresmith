# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Canned success artifacts for the fault provider (Package C).

A NON-faulted LLM call must WRITE the file(s) the caller expects on disk -- else
the baseline (fault-free) run itself fails with "agent did not write ...". The
agent CLI would normally do this via its file-write tools; the in-process fault
backend has no tools, so :class:`CannedDesignScript` scrapes the target path(s)
out of the prompt and materializes minimal-but-valid artifacts.

These artifacts validate PLUMBING only -- they are not meaningful RTL/TB.
"""

from __future__ import annotations

import re
from pathlib import Path

# Absolute or relative paths mentioned in a prompt. Matches tokens that look
# like file paths ending in a known extension.
_PATH_RE = re.compile(r"[A-Za-z0-9_./-]+\.(?:v|sv|py|md)\b")


def _rtl_module(name: str) -> str:
    """A >=200 byte, syntactically-plausible Verilog module named ``name``.

    Must satisfy pipeline_helpers._assert_rtl_materialized: exists, >=200 bytes,
    contains ``module <name>``.
    """
    return (
        f"// Auto-generated canned RTL for block '{name}' (test harness).\n"
        f"// This is a plumbing stub, not meaningful hardware.\n"
        f"module {name} (\n"
        f"    input  wire        clk,\n"
        f"    input  wire        rst_n,\n"
        f"    input  wire [7:0]  data_in,\n"
        f"    input  wire        valid_in,\n"
        f"    output reg  [7:0]  data_out,\n"
        f"    output reg         valid_out\n"
        f");\n"
        f"    always @(posedge clk or negedge rst_n) begin\n"
        f"        if (!rst_n) begin\n"
        f"            data_out  <= 8'd0;\n"
        f"            valid_out <= 1'b0;\n"
        f"        end else begin\n"
        f"            data_out  <= data_in;\n"
        f"            valid_out <= valid_in;\n"
        f"        end\n"
        f"    end\n"
        f"endmodule\n"
    )


def _cocotb_tb(name: str) -> str:
    return (
        f"# Auto-generated canned cocotb TB for '{name}' (test harness).\n"
        f"import cocotb\n"
        f"from cocotb.triggers import Timer\n\n\n"
        f"@cocotb.test()\n"
        f"async def test_{name}(dut):\n"
        f"    await Timer(1, units='ns')\n"
        f"    assert True\n"
    )


def _uarch_spec(name: str) -> str:
    return (
        f"# uArch Spec: {name}\n\n"
        f"## 1. Block Overview\n\n"
        f"Auto-generated canned uArch spec for '{name}' (test harness).\n\n"
        f"- flip_flop_budget: 64\n"
        f"- area_budget_um2: 5000\n"
        f"- target_clock_mhz: 50\n"
    )


class CannedDesignScript:
    """Materialize the artifacts a non-faulted call is expected to write."""

    def write_artifacts(self, *, prompt: str, system: str, project_root: str) -> list[str]:
        """Scrape target paths from ``prompt``/``system`` and write canned
        content for each. Returns the list of absolute paths written.
        """
        root = Path(project_root)
        written: list[str] = []
        seen: set[str] = set()
        for text in (prompt or "", system or ""):
            for m in _PATH_RE.finditer(text):
                rel = m.group(0)
                if rel in seen:
                    continue
                seen.add(rel)
                path = (root / rel) if not rel.startswith("/") else Path(rel)
                # Only write into the run dir -- never touch paths outside it.
                try:
                    path.resolve().relative_to(root.resolve())
                except (ValueError, OSError):
                    continue
                ext = path.suffix.lstrip(".")
                stem = path.stem
                if ext in ("v", "sv") and (
                    "rtl" in path.parts or "/rtl/" in str(path)
                ):
                    content = _rtl_module(stem)
                elif ext == "py" and stem.startswith("test_"):
                    content = _cocotb_tb(stem[len("test_"):] or stem)
                elif ext == "md":
                    content = _uarch_spec(stem)
                else:
                    continue
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(content, encoding="utf-8")
                    written.append(str(path))
                except OSError:
                    continue
        return written

    def response_text(self, written: list[str]) -> str:
        """The canned assistant response for a successful call."""
        if written:
            return (
                "Done. Wrote:\n" + "\n".join(f"- {p}" for p in written) +
                "\nThe module compiles and matches the reference."
            )
        return "Done. Analysis complete; no files required for this step."
